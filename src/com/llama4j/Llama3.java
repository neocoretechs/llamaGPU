///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 21+
//PREVIEW
//COMPILE_OPTIONS --add-modules=jdk.incubator.vector
//RUNTIME_OPTIONS --add-modules=jdk.incubator.vector -Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0
//MAIN com.llama4j.Llama3

// Practical Llama 3 (and 3.1) inference in a single Java file
// Author: Alfonso² Peterssen
// Based on Andrej Karpathy's llama2.c and minbpe projects
//
// Supports llama.cpp's GGUF format, restricted to Q4_0 and Q8_0 quantized models
// Multi-threaded matrix vector multiplication routines implemented using Java's Vector API
// Simple CLI with --chat and --instruct mode
//
// To run just:
// jbang Llama3.java --help
//
// Remember: Llama models use GPT2 vocabulary while non-Llama models use Llama vocabulary!
// Enjoy!
package com.llama4j;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import java.io.PrintStream;
import java.io.PrintWriter;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.IntConsumer;
import java.util.function.IntFunction;
import java.util.function.LongConsumer;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.LongStream;
import java.util.stream.Stream;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import com.neocoretechs.relatrix.client.asynch.AsynchRelatrixClientTransaction;
import com.neocoretechs.relatrix.Result;
import com.neocoretechs.rocksack.TransactionId;

import com.neocoretechs.rocksack.Alias;

import com.llama4j.ffi.NativeLoader;

public class Llama3 {
	private static final Log log = LogFactory.getLog(Llama3.class);
    // Batch-size used in prompt evaluation.
    private static int BATCH_SIZE = Integer.getInteger("llama.BatchSize", 16);
    public final static boolean DEBUG = false;
    public final static boolean DISPLAY_METADATA = true;
	public static boolean SDOT_CPU = true;
    public static AsynchRelatrixClientTransaction dbClient = null;
    public static TransactionId xid = null;
    public static Alias tensorAlias = null;
    // metadata dump
	public static BufferedWriter outputStream = null;
	public static PrintWriter output = null;
	public static FileWriter fileWriter = null;
	// Arena
	public static Arena autoArena = Arena.ofAuto();
	public static Arena sharedArena = Arena.ofShared();
	//
	public static MethodHandle getFloatQ8;
	public static MethodHandle getFloat;
	public static MethodHandle sdotSliceDeviceHandle;
    public static MethodHandle cudaGetMemInfo;
    public static MethodHandle sdotSimple;
    public static MethodHandle launchRmsnorm;
    public static MethodHandle launchQK;
	public static MethodHandle launchSoftmaxInplace;
    public static MethodHandle launchAV;
	public static MethodHandle launchMatmul;
	public static MethodHandle launchRope;
	public static MethodHandle copyHostToDeviceMH;
	public static MethodHandle copyDeviceToHostMH;
	public static MethodHandle freeDevicePtr;
	public static MethodHandle allocDevicePtr;
	public static MethodHandle cudaInit;
	
	static {
		NativeLoader.load();
	}
	

    static void runInteractive(Llama model, Options options) {
        Llama.State state = null;
        List<Integer> conversationTokens = new ArrayList<>();
        ChatFormatInterface chatFormat;
        if(DISPLAY_METADATA) {
        	Llama3.output.println("Begin Special tokens:");
        	Llama3.output.println(model.tokenizer().getSpecialTokens());
        	Llama3.output.println("End Special tokens.\r\n");
        }
        // Chat format seems solely based on individual model, so we extract a name in model loader from Metada general.name
        if(ModelLoader.name.equals("mistral")) {
        	chatFormat = new MistralChatFormat(model.tokenizer());
        } else {
        	if(ModelLoader.name.equals("llama")) {
        		chatFormat = new ChatFormat(model.tokenizer());
        	} else {
        		if(ModelLoader.name.equals("qwen")) {
        			BATCH_SIZE = 1;
        			chatFormat = new ChatMLFormat(model.tokenizer());
        		} else {
        			if(ModelLoader.name.equals("magistral")) {
        				chatFormat = new MistralChatFormat(model.tokenizer());
        			} else
        				throw new IllegalArgumentException("expected metadata general.name containing mistral, magistral, llama, or qwen but found "+ModelLoader.name);
        		}
        	}
        }
        conversationTokens.add(chatFormat.getBeginOfText());
        if (options.systemPrompt() != null) {
            conversationTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.SYSTEM, options.systemPrompt())));
        }
        if(options.localNode() != null) {
        	try {
        		dbClient = new AsynchRelatrixClientTransaction(options.localNode(), options.remoteNode(), options.remotePort());
        		xid = dbClient.getTransactionId();
        		tensorAlias = new Alias("Tensors");
        		try {
        			if(dbClient.getAlias(tensorAlias).get() == null)
        				dbClient.setRelativeAlias(tensorAlias);
        		} catch(ExecutionException | InterruptedException ie) {}
        		if(DEBUG)
        			System.out.println("Relatrix transaction Id:"+xid);
        	} catch(IOException ioe) {
        		ioe.printStackTrace();
        	}
        }
        int startPosition = 0;
        Scanner in = new Scanner(System.in);
        loop: while (true) {
        	boolean storeDb = true;
            System.out.print("> ");
            System.out.flush();
            String userText = in.nextLine();
            switch (userText) {
                case "/quit":
                case "/exit": break loop;
                case "/context": {
                    System.out.printf("%d out of %d context tokens used (%d tokens remaining)%n",
                            conversationTokens.size(),
                            model.configuration().contextLength,
                            model.configuration().contextLength - conversationTokens.size());
                    continue;
                }
            }
            if(userText.startsWith("www.") || userText.startsWith("http://") || userText.startsWith("https://")) {
            	String[] urlc = userText.split(" ");
            	Element result = parseLinks(urlc);
            	// replace userText
            	if(result == null)
            		continue;
            	userText = result.text();
            	System.out.println(userText);
            } else {
            	if(userText.startsWith("/recalltime")) {
            		storeDb = false;
            		String[] query = userText.split(" ");
            		String s = parseTime(query);
            		if(s == null)
            			continue;
            		userText = s;
                  	System.out.println(userText);
            	} else {
                  	if(userText.startsWith("/recallwords")) {
                  		storeDb = false;
                		String[] query = userText.split(" ");
                		String s = parseKeywords(query);
                		if(s == null)
                			continue;
                		userText = s;
                      	System.out.println(userText);
                  	}
            	}
            }
            if (state == null) {
                state = model.createNewState(BATCH_SIZE, chatFormat.getBeginOfText());
            }
            conversationTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, userText)));
            conversationTokens.addAll(chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
            Set<Integer> stopTokens = chatFormat.getStopTokens();
            List<Integer> responseTokens = null;
            int token = 0;
     //       try (Timer timer = Timer.log("Forward inference..")) {
            	if(ModelLoader.name.equals("qwen")) {
            		responseTokens = null;//Llama.generateTokensQwen(model, state, startPosition, conversationTokens.subList(startPosition, conversationTokens.size()), stopTokens, options.maxTokens(), sampler, options.echo(), token -> {
            			if (options.stream()) {
            				int tokenType = model.tokenizer().getTokenType(token);
            				if (tokenType == 1 || tokenType == 6) {
            					System.out.print(model.tokenizer().decode(List.of(token)));
            				}
            			}
            		//});
            	} else {
            		responseTokens = null;//Llama.generateTokens(model, state, startPosition, conversationTokens.subList(startPosition, conversationTokens.size()), stopTokens, options.maxTokens(), sampler, options.echo(), token -> {
            			if (options.stream()) {
            				if (!model.tokenizer().isSpecialToken(token)) {
            					System.out.print(model.tokenizer().decode(List.of(token)));
            				}
            			}
            		//});
            	}
            //}

            // Include stop token in the prompt history, but not in the response displayed to the user.
            conversationTokens.addAll(responseTokens);
            startPosition = conversationTokens.size();
            Integer stopToken = null;
            if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
                stopToken = responseTokens.getLast();
                responseTokens.removeLast();
            }
            if (!options.stream()) {
                String responseText = model.tokenizer().decode(responseTokens);
                System.out.println(responseText);
                if(dbClient != null && storeDb)
                	dbClient.store(xid, System.currentTimeMillis(), userText, responseText);//.thenAccept(result-> {
                		//System.out.println("Response from storage:"+result);
                	//});
            } else {
                if(dbClient != null && storeDb)
                	dbClient.store(xid, System.currentTimeMillis(), userText, model.tokenizer().decode(responseTokens));//.thenAccept(result-> {
                		//System.out.println("Response from storage:"+result);
                	//});
            }
            if (stopToken == null) {
                System.err.println("Ran out of context length...");
                break;
            }
        }
        if(dbClient != null) {
        	try {
        		dbClient.commit(xid).get();
        		dbClient.endTransaction(xid).get();
        		dbClient.close();
        	} catch(InterruptedException | ExecutionException ie) {}
        }
        if(Llama3.DISPLAY_METADATA) {
        	try {
        		Llama3.outputStream.flush();
        		Llama3.output.close();
        	} catch (final IOException e) {
        		System.err.println("Could not flush metadata file "+e);
        	} finally {
        		try {
        			if (Llama3.outputStream != null) {
        				Llama3.outputStream.close();
        			}
        			if (Llama3.output != null) {
        				Llama3.output.close();
        			}
        		} catch (final IOException e) {
        			System.err.println("Failed to close file: "+e);
        		}
        	}
        }
    }

    static void runInstructOnce(Llama model, Options options) {
        ChatFormatInterface chatFormat;
        // Chat format seems solely based on individual model, so we extract a name in model loader from Metada general.name
        if(ModelLoader.name.equals("mistral")) {
        	chatFormat = new MistralChatFormat(model.tokenizer());
        } else {
        	if(ModelLoader.name.equals("llama")) {
        		chatFormat = new ChatFormat(model.tokenizer());
        	} else {
        		if(ModelLoader.name.equals("qwen")) {
        			chatFormat = new ChatMLFormat(model.tokenizer());
        		} else {
        			throw new IllegalArgumentException("expected metadata general.name containing mistral, llama, or qwen but found "+ModelLoader.name);
        		}
        	}
        }
        Llama.State state = model.createNewState(BATCH_SIZE, chatFormat.getBeginOfText());
        List<Integer> promptTokens = new ArrayList<>();
        promptTokens.add(chatFormat.getBeginOfText());
        if (options.systemPrompt() != null) {
            promptTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.SYSTEM, options.systemPrompt())));
        }
        promptTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, options.prompt())));
        promptTokens.addAll(chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));

        Set<Integer> stopTokens = chatFormat.getStopTokens();
        List<Integer> responseTokens = Llama.generateTokens(model, state, 0, promptTokens, stopTokens, options.maxTokens(), options.echo(), token -> {
            if (options.stream()) {
                if (!model.tokenizer().isSpecialToken(token)) {
                    System.out.print(model.tokenizer().decode(List.of(token)));
                }
            }
        });
        if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
            responseTokens.removeLast();
        }
        if (!options.stream()) {
            String responseText = model.tokenizer().decode(responseTokens);
            System.out.println(responseText);
        }
    }
    /**
     * Parse the command line for url and xpath directive
     * @param urlc array of cmdl args, link at 0
     * @return The Element that matches directive
     */
    private static Element parseLinks(String[] urlc) {
    	//try {	
    		Document doc = null;
    		if(urlc == null || urlc.length < 2)
    			return null;
    		try {
    	 		doc = Jsoup.connect(urlc[0])
        			.userAgent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        			.get();
    		} catch(IOException ioe) {
    			ioe.printStackTrace();
    			return null;
    		}
    		Element result = null;
    		Elements results = null;
    		//for(int i = 1; i < urlc.length; i++) {
    		//	results = doc.select(urlc[i]);
    		//}
    		results = doc.selectXpath(urlc[1]);
    		if(results == null)
    			return null;
    		result = results.first();
    		if(result == null)
    			return null;
    		if(result.is("a"))
    			return parseLinks(new String[] {result.attr("href"),"//a"});
    		return result;
    		//System.out.printf("toString:%s text:%s wholeText:%s%n", result.toString(),result.text(),result.wholeText());
    		//System.out.printf("result is a:%b result is a[href]:%b%n",result.is("a"),result.is("a[href]"));
    	//} catch(MalformedURLException e) {
    	//	e.printStackTrace();
    	//}
    	//return null;
    }
    
    /**
     * element 0 is command <br> /recalltime 
     * arg day time to end day time
     * @param query the command line with command times
     * @return String of Result instances from db that contain 2 elements of question/answer string in time range
     */
    private static String parseTime(String[] query) {
    	CompletableFuture<Stream> s;
		String tq,tqe;
		LocalDateTime localDateTime;
		long millis,millise;
    	if(query == null)
    		return null;
    	if(query.length == 5) {
    		// day time to end day time
    		tq = String.format("%s %s", query[1], query[2]);
    		tqe = String.format("%s %s", query[3], query[4]);
    		localDateTime = LocalDateTime.parse(tq, DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss") );
    		millis = localDateTime.atZone(ZoneId.systemDefault()).toInstant().toEpochMilli();
    		localDateTime = LocalDateTime.parse(tqe, DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss") );
    		millise = localDateTime.atZone(ZoneId.systemDefault()).toInstant().toEpochMilli();
    		s = dbClient.findSubStream(xid,'*','?','?',millis,millise,String.class,String.class);
    		StringBuilder sb = new StringBuilder();
    		try {
    			s.get().forEach(e->{
    				sb.append(((Result)e).get(0));
    				sb.append(((Result)e).get(1));
    			});
    		} catch(InterruptedException | ExecutionException ie) {}
    		return sb.toString();
    	}
    	return null;
    }
    /**
     * Element 0 is command /recallwords
     * @param query the command line with command keywords
     * @return the string of question/answer containing keywords
     */
    private static String parseKeywords(String[] query) {
      	if(query == null || query.length < 2)
    		return null;
     	StringBuilder sb = new StringBuilder();
      	CompletableFuture<Stream> s = dbClient.findStream(xid, '*', '?', '?');
      	try {
      		s.get().forEach(e->{
      			String s1 = (String)((Result)e).get(0);
      			for(int i = 1; i < query.length; i++) {
      				if(s1.contains(query[i])) {
      					sb.append(s1);
      					break;
      				}
      			}
      			s1 = (String)((Result)e).get(1);
      			for(int i = 1; i < query.length; i++) {
      				if(s1.contains(query[i])) {
      					sb.append(s1);
      					break;
      				}
      			}
      		});
      	} catch(InterruptedException | ExecutionException ie) {}
      	return sb.toString();
    }
    
    record Options(Path modelPath, String prompt, String systemPrompt, boolean interactive,
                   float temperature, float topp, long seed, int maxTokens, boolean stream, boolean echo,
                   String localNode, String remoteNode, int remotePort) {

        static final int DEFAULT_MAX_TOKENS = 512;

        Options {
            require(modelPath != null, "Missing argument: --model <path> is required");
            require(interactive || prompt != null, "Missing argument: --prompt is required in --instruct mode e.g. --prompt \"Why is the sky blue?\"");
            require(0 <= temperature, "Invalid argument: --temperature must be non-negative");
            require(0 <= topp && topp <= 1, "Invalid argument: --top-p must be within [0, 1]");
        }

        static void require(boolean condition, String messageFormat, Object... args) {
            if (!condition) {
                System.out.println("ERROR " + messageFormat.formatted(args));
                System.out.println();
                printUsage(System.out);
                System.exit(-1);
            }
        }

        static void printUsage(PrintStream out) {
            out.println("Usage:  jbang Llama3.java [options]");
            out.println();
            out.println("Options:");
            out.println("  --model, -m <path>            required, path to .gguf file");
            out.println("  --interactive, --chat, -i     run in chat mode");
            out.println("  --instruct                    run in instruct (once) mode, default mode");
            out.println("  --prompt, -p <string>         input prompt");
            out.println("  --system-prompt, -sp <string> (optional) system prompt");
            out.println("  --temperature, -temp <float>  temperature in [0,inf], default 0.1");
            out.println("  --top-p <float>               p value in top-p (nucleus) sampling in [0,1] default 0.95");
            out.println("  --seed <long>                 random seed, default System.nanoTime()");
            out.println("  --max-tokens, -n <int>        number of steps to run for < 0 = limited by context length, default " + DEFAULT_MAX_TOKENS);
            out.println("  --stream <boolean>            print tokens during generation; may cause encoding artifacts for non ASCII text, default true");
            out.println("  --echo <boolean>              print ALL tokens to stderr, if true, recommended to set --stream=false, default false");
            out.println("  --localNode <string>          local database client node");
            out.println("  --remoteNode <string>         remote database client node");
            out.println("  --remotePort <int>            remote database port");
            out.println();
            out.println("Examples:");
            out.println("  jbang Llama3.java --model llama3.2-1b-q4_0.gguf --prompt \"Tell me a joke\"");
            out.println("  jbang Llama3.java --model llama3.2-1b-q4_0.gguf --system-prompt \"Reply concisely, in French\" --prompt \"Who was Marie Curie?\"");
            out.println("  jbang Llama3.java --model llama3.2-1b-q4_0.gguf --system-prompt \"Answer concisely\" --chat");
            out.println("  jbang Llama3.java --model llama3.2-1b-q4_0.gguf --chat");
            out.println("  jbang Llama3.java --model llama3.2-1b-q4_0.gguf --prompt \"Print 5 emojis\" --stream=false");
        }

        static Options parseOptions(String[] args) {
            String prompt = null;
            String systemPrompt = null;
            float temperature = 0.1f;
            float topp = 0.95f;
            Path modelPath = null;
            long seed = System.nanoTime();
            // Keep max context length small for low-memory devices.
            int maxTokens = DEFAULT_MAX_TOKENS;
            boolean interactive = false;
            boolean stream = true;
            boolean echo = false;
            String localNode = null;
            String remoteNode = null;
            int remotePort = 0;

            for (int i = 0; i < args.length; i++) {
                String optionName = args[i];
                require(optionName.startsWith("-"), "Invalid option %s", optionName);
                switch (optionName) {
                    case "--interactive", "--chat", "-i" -> interactive = true;
                    case "--instruct" -> interactive = false;
                    case "--help", "-h" -> {
                        printUsage(System.out);
                        System.exit(0);
                    }
                    default -> {
                        String nextArg;
                        if (optionName.contains("=")) {
                            String[] parts = optionName.split("=", 2);
                            optionName = parts[0];
                            nextArg = parts[1];
                        } else {
                            require(i + 1 < args.length, "Missing argument for option %s", optionName);
                            nextArg = args[i + 1];
                            i += 1; // skip arg
                        }
                        switch (optionName) {
                            case "--prompt", "-p" -> prompt = nextArg;
                            case "--system-prompt", "-sp" -> systemPrompt = nextArg;
                            case "--temperature", "--temp" -> temperature = Float.parseFloat(nextArg);
                            case "--top-p" -> topp = Float.parseFloat(nextArg);
                            case "--model", "-m" -> modelPath = Paths.get(nextArg);
                            case "--seed", "-s" -> seed = Long.parseLong(nextArg);
                            case "--max-tokens", "-n" -> maxTokens = Integer.parseInt(nextArg);
                            case "--stream" -> stream = Boolean.parseBoolean(nextArg);
                            case "--echo" -> echo = Boolean.parseBoolean(nextArg);
                            case "--localNode" -> localNode = nextArg;
                            case "--remoteNode" -> remoteNode = nextArg;
                            case "--remotePort" -> remotePort = Integer.parseInt(nextArg);
                            default -> require(false, "Unknown option: %s", optionName);
                        }
                    }
                }
            }
            return new Options(modelPath, prompt, systemPrompt, interactive, temperature, topp, seed, maxTokens, stream, echo, localNode, remoteNode, remotePort);
        }
    }

    public static void main(String[] args) throws IOException {
        NativeLoader.loadMethods();
        Options options = Options.parseOptions(args);
        if(Llama3.DISPLAY_METADATA) {
        	try {
        		Llama3.fileWriter = new FileWriter(options.modelPath.toString()+".metadata", false);
        		Llama3.outputStream = new BufferedWriter(fileWriter);
        		Llama3.output = new PrintWriter(outputStream);
        	} catch (final IOException e) {
        		System.err.println("Could not open file " + options.modelPath.toString()+".metadata\r\n"+e);
        	}
        }
    
        if (options.interactive()) {
           
        } else {
            
        }
    }
}


interface Timer extends AutoCloseable {
    @Override
    void close(); // no Exception
    static Timer log(String label) {
        return log(label, TimeUnit.MILLISECONDS);
    }
    static Timer log(String label, TimeUnit timeUnit) {
        return new Timer() {
            final long startNanos = System.nanoTime();
            @Override
            public void close() {
                long elapsedNanos = System.nanoTime() - startNanos;
                System.err.println(label + ": " + timeUnit.convert(elapsedNanos, TimeUnit.NANOSECONDS) + " " + timeUnit.toChronoUnit().name().toLowerCase());
            }
        };
    }
}
/**
 * Load model, get GGUF metadata, load vocabulary, create tokenizer, create config, if loadWeights - load tensors, load weights
 * create Llama with config, tokenizer, weights
 */
final class ModelLoader {
    static final String TOKENIZER_GPT2_MODEL = "gpt2"; // Llama3 uses gpt2!
    static final String TOKENIZER_LLAMA_MODEL = "llama"; // non Llama uses llama!
    public static String model = "gpt2"; // default for Llama models!
    public static String name = null; // Name is based solely on name of model, they all seem to have their own ChatFormat not based on model
    private static final String LLAMA_3_PATTERN = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    private static Vocabulary loadVocabulary(Map<String, Object> metadata) {
        model = (String) metadata.get("tokenizer.ggml.model");
        name = (String) metadata.get("general.name");
        if(name.toLowerCase().contains("llama")) // Meta Llama etc. etc.
        	name = "llama";
        else
        	if(name.toLowerCase().contains("mistral")) //models--mistralai etc. etc.
        		name="mistral";
        	else
        		if(name.toLowerCase().contains("qwen"))
        			name="qwen";
        		else
        			if(name.toLowerCase().contains("magistral"))
        				name="magistral";
        String[] tokens = (String[]) metadata.get("tokenizer.ggml.tokens");
        if(TOKENIZER_LLAMA_MODEL.equals(model)) {
        	float[] scores = (float[]) metadata.get("tokenizer.ggml.scores");
        	return new Vocabulary(tokens, scores);
        } else {
        	if(TOKENIZER_GPT2_MODEL.equals(model)) {
        		return new Vocabulary(tokens, null);
        	} else {
        		throw new IllegalArgumentException("expected " + TOKENIZER_GPT2_MODEL + " or "+ TOKENIZER_LLAMA_MODEL+ " but found " + model);
        	}
        }
    }

    

    public static Llama loadModel(FileChannel fileChannel, int contextLength, boolean loadWeights) throws IOException {
        try (var ignored = Timer.log("Load model")) {
            Map<String, Object> metadata = null;//gguf.getMetadata();
            if(Llama3.DISPLAY_METADATA) {
            	Llama3.output.println("Begin GGUF Metadata:");
            	metadata.forEach((k, v) -> {
            		String valueStr;
            		if (v != null && v.getClass().isArray()) {
            			Class<?> componentType = v.getClass().getComponentType();
            			if (componentType == int.class) {
            				valueStr = Arrays.toString((int[]) v);
            			} else if (componentType == byte.class) {
            				valueStr = Arrays.toString((byte[]) v);
            			} else if (componentType == double.class) {
            				valueStr = Arrays.toString((double[]) v);
            			} else if (componentType == boolean.class) {
            				valueStr = Arrays.toString((boolean[]) v);
            			} else if (componentType == char.class) {
            				valueStr = Arrays.toString((char[]) v);
            			} else if (componentType == long.class) {
            				valueStr = Arrays.toString((long[]) v);
            			} else if (componentType == float.class) {
            				valueStr = Arrays.toString((float[]) v);
            			} else if (componentType == short.class) {
            				valueStr = Arrays.toString((short[]) v);
            			} else {
            				valueStr = Arrays.toString((Object[]) v); // for Object arrays
            			}
            		} else {
            			valueStr = String.valueOf(v);
            		}
            		Llama3.output.println(k + "=" + valueStr);
            	});
            	Llama3.output.println("End GGUF Metadata.\r\n");
            }
            Vocabulary vocabulary = loadVocabulary(metadata);
            TokenizerInterface tokenizer;
            Llama.Configuration config;
        
            String arch = (String) metadata.get("general.architecture");
            if(ModelLoader.name.equals("mistral")) {
           		tokenizer = createLlamaTokenizer(metadata, vocabulary);
        		config = createConfig(arch, metadata, vocabulary, contextLength);
        		if (loadWeights) {
        			// loadTensors corresponds to getTensorEntries in old version
        			
        		}
            } else {
            	if(ModelLoader.name.equals("llama")) {
                   	tokenizer = createGPT2Tokenizer(metadata, vocabulary);
                	config = createConfig(arch, metadata, vocabulary, contextLength);
                    if (loadWeights) {
                    	// loadTensors corresponds to getTensorEntries in old version
             
                    }
            	} else {
            		if(ModelLoader.name.equals("qwen")) {
                      	tokenizer = createQwen2Tokenizer(metadata, vocabulary);
                    	config = createConfig(arch, metadata, vocabulary, contextLength);
                        if (loadWeights) {
                   
                        }
            		} else {
            			if(ModelLoader.name.equals("magistral")) {
                          	tokenizer = createMagistralTokenizer(metadata, vocabulary);
                        	config = createConfig(arch, metadata, vocabulary, contextLength);
                            if (loadWeights) {
                      
                            }
            			} else
            				throw new IllegalArgumentException("expected metadata general.name containing mistral, magistral, llama, or qwen but found "+ModelLoader.name);
            		}
            	}
            }
            return new Llama(config, tokenizer);
        }
    }
    
    static Llama.Configuration createConfig(String arch, Map<String, Object> metadata, Vocabulary vocabulary, int contextLength) {
        Llama.Configuration config = new Llama.Configuration(
                (int) metadata.get(arch+".embedding_length"),
                (int) metadata.get(arch+".feed_forward_length"),
                (int) metadata.get(arch+".block_count"),
                (int) metadata.get(arch+".attention.head_count"),

                metadata.containsKey(arch+".attention.head_count_kv")
                        ? (int) metadata.get(arch+".attention.head_count_kv")
                        : (int) metadata.get(arch+".attention.head_count"),

                vocabulary.size(),
                (int) metadata.get(arch+".context_length"),
                (float) metadata.getOrDefault(arch+".attention.layer_norm_rms_epsilon", 1e-5f),
                (float) metadata.getOrDefault(arch+".rope.freq_base", 10000f)
        ).withContextLength(contextLength);
        return config;
    }
    

   

    private final static String QWEN2_PATTERN = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    private static Tokenizer createQwen2Tokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        int[] tokenTypes = (int[]) metadata.get("tokenizer.ggml.token_type");
        String[] mergeLines = (String[]) metadata.get("tokenizer.ggml.merges");
        List<Pair<Integer, Integer>> merges = Arrays.stream(mergeLines)
                .map(line -> line.split(" "))
                .map(parts ->
                        new Pair<>(
                                vocabulary.getIndex(parts[0]).orElseThrow(),
                                vocabulary.getIndex(parts[1]).orElseThrow())
                ).toList();

        int allTokens = vocabulary.size();
        int baseTokens = vocabulary.getIndex("<|endoftext|>").orElseThrow(); // assume all tokens after the base ones are special.
        int reservedSpecialTokens = allTokens - baseTokens;
        List<String> specialTokensList = Arrays.stream(vocabulary.tokens(), baseTokens, allTokens).toList();

        assert specialTokensList.stream().allMatch(token -> vocabulary.getIndex(token).isPresent());

        Map<String, Integer> specialTokens =
                IntStream.range(0, specialTokensList.size())
                        .boxed()
                        .collect(Collectors.toMap(
                                i -> specialTokensList.get(i),
                                i -> baseTokens + i)
                        );

        return new Tokenizer(vocabulary, merges, QWEN2_PATTERN, specialTokens, tokenTypes);
    }

    private static Tokenizer createGPT2Tokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        String[] mergeLines = (String[]) metadata.get("tokenizer.ggml.merges");
        List<Pair<Integer, Integer>> merges = Arrays.stream(mergeLines)
                .map(line -> line.split(" "))
                .map(parts ->
                        new Pair<>(
                                vocabulary.getIndex(parts[0]).orElseThrow(),
                                vocabulary.getIndex(parts[1]).orElseThrow())
                ).toList();

        int allTokens = vocabulary.size();
        int baseTokens = 128000; // assume all tokens after the base ones are special.
        int reservedSpecialTokens = allTokens - baseTokens;
        List<String> specialTokensList = Arrays.stream(vocabulary.tokens(), baseTokens, allTokens).toList();

        assert specialTokensList.stream().allMatch(token -> vocabulary.getIndex(token).isPresent());

        Map<String, Integer> specialTokens =
                IntStream.range(0, specialTokensList.size())
                        .boxed()
                        .collect(Collectors.toMap(
                                i -> specialTokensList.get(i),
                                i -> baseTokens + i)
                        );

        return new Tokenizer(vocabulary, merges, LLAMA_3_PATTERN, specialTokens);
    }
    
    private static MistralTokenizer createLlamaTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        int[] tokenTypes = (int[]) metadata.get("tokenizer.ggml.token_type");
        List<Integer> specialTokensList = IntStream.range(0, vocabulary.size()).filter(t -> tokenTypes[t] != 1 && tokenTypes[t] != 6).boxed().toList();
        Map<String, Integer> specialTokens =
                IntStream.range(0, specialTokensList.size())
                        .boxed()
                        .collect(Collectors.toMap(
                                t -> vocabulary.get(t),
                                t -> t)
                        );
        return new MistralTokenizer(vocabulary, null, specialTokens, tokenTypes);
    }
    
    private static Tokenizer createMagistralTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        int[] tokenTypes = (int[]) metadata.get("tokenizer.ggml.token_type");
        List<Integer> specialTokensList = IntStream.range(0, vocabulary.size()).filter(t -> tokenTypes[t] != 1 && tokenTypes[t] != 6).boxed().toList();
        Map<String, Integer> specialTokens =
                IntStream.range(0, specialTokensList.size())
                        .boxed()
                        .collect(Collectors.toMap(
                                t -> vocabulary.get(t),
                                t -> t)
                        );
        String[] mergeLines = (String[]) metadata.get("tokenizer.ggml.merges");
        List<Pair<Integer, Integer>> merges = Arrays.stream(mergeLines)
                .map(line -> line.split(" "))
                .map(parts ->
                        new Pair<>(
                                vocabulary.getIndex(parts[0]).orElseThrow(),
                                vocabulary.getIndex(parts[1]).orElseThrow())
                ).toList();

        return new Tokenizer(vocabulary, merges, LLAMA_3_PATTERN, specialTokens, tokenTypes);
    }
    

}

record Llama(Configuration configuration, TokenizerInterface tokenizer) {
    private static boolean DEBUG = false;

	public State createNewState(int batchsize, int beginOfText) {
        State state = null;//new State(configuration(), batchsize);
        state.latestToken = beginOfText; // was tokenizer.getSpecialTokens().get("<|begin_of_text|>");, now we get from ChatFormat.beginOfText() which does the same
        return state;
    }

    public static final class Configuration {
        public final int dim; // transformer dimension
        public final int hiddenDim; // for ffn layers
        public final int numberOfLayers; // number of layers
        public final int numberOfHeads; // number of query heads
        public final int numberOfKeyValueHeads; // number of key/value heads (can be < query heads because of multiquery)
        public final int vocabularySize; // vocabulary size, usually 256 (byte-level)
        public final int contextLength; // max sequence length
        public final float rmsNormEps;
        public final float ropeTheta;
        public final int headSize;

        Configuration withContextLength(int newContextLength) {
            if (newContextLength < 0) {
                return this; // no change
            }
            return new Configuration(this.dim, this.hiddenDim, this.numberOfLayers, this.numberOfHeads, this.numberOfKeyValueHeads, this.vocabularySize, newContextLength, this.rmsNormEps, this.ropeTheta);
        }

        public Configuration(int dim, int hiddenDim, int numberOfLayers, int numberOfHeads, int numberOfKeyValueHeads, int vocabularySize, int contextLength, float rmsNormEps, float ropeTheta) {
            this.dim = dim;
            this.hiddenDim = hiddenDim;
            this.numberOfLayers = numberOfLayers;
            this.numberOfHeads = numberOfHeads;
            this.numberOfKeyValueHeads = numberOfKeyValueHeads;
            this.vocabularySize = vocabularySize;
            this.contextLength = contextLength;
            this.rmsNormEps = rmsNormEps;
            this.ropeTheta = ropeTheta;
            this.headSize = dim / numberOfHeads;
        }
    }

    public static final class State {

        // current wave of activations
        public final int batchsize;
        public final FloatTensor[] x; // activation at current time stamp (dim,)
        public final FloatTensor[] xb; // same, but inside a residual branch (dim,)
        public final FloatTensor[] xb2; // an additional buffer just for convenience (dim,)
        public final FloatTensor[] hb; // buffer for hidden dimension in the ffn (hidden_dim,)
        public final FloatTensor[] hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
        public final FloatTensor[] q; // query (dim,)
        public final FloatTensor[] k; // key (dim,)
        public final FloatTensor[] v; // value (dim,)
        public final FloatTensor[] att; // buffer for scores/attention values (n_heads, seq_len)
        public final FloatTensor logits; // output logits

        // kv cache
        public final FloatTensor[] keyCache;   // (n_layer, seq_len, kv_dim)
        public final FloatTensor[] valueCache; // (n_layer, seq_len, kv_dim)
        
        /** last index in previous block */
        int idxPrevBlock;

        public int latestToken;

        State(Configuration config, int batchsize) {
            this.batchsize = batchsize;
            this.x = allocate(batchsize, config.dim);
            this.xb = allocate(batchsize, config.dim);
            this.xb2 = allocate(batchsize, config.dim);
            this.hb = allocate(batchsize, config.hiddenDim);
            this.hb2 = allocate(batchsize, config.hiddenDim);
            this.q = allocate(batchsize, config.dim);
            this.k = allocate(batchsize, config.dim);
            this.v = allocate(batchsize, config.dim);
            this.att = allocate(batchsize, config.numberOfHeads, config.contextLength);
            idxPrevBlock = -1;

            this.logits = ArrayFloatTensor.allocate(config.vocabularySize);
            int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;
            this.keyCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength, kvDim)).limit(config.numberOfLayers).toArray(FloatTensor[]::new);
            this.valueCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength, kvDim)).limit(config.numberOfLayers).toArray(FloatTensor[]::new);
        }
    }

    static FloatTensor[] allocate(int numTokens, int... dims) {
        return IntStream.range(0, numTokens)
                .mapToObj(i -> ArrayFloatTensor.allocate(dims))
                .toArray(FloatTensor[]::new);
    }
    /**
     * FloatBuffer weights for Vector processing
     * @param out output tensor
     * @param x target
     * @param weight Weights loaded to floatbuffer
     * @param size size of tensor
     * @param rmsNormEps epsilon omega point
     */
    static void rmsnorm(FloatTensor out, FloatTensor x, FloatBuffer weight, int size, float rmsNormEps) {
        // calculate sum of squares
        float ss = x.reduce(0, size, 0f, (acc, xi) -> acc + xi * xi);
        ss /= size;
        ss += rmsNormEps;
        ss = (float) (1.0 / Math.sqrt(ss));
        // normalize and scale
        final float finalss = ss; // for the lambda
        out.mapWithIndexInPlace(0, size, (value, index) -> weight.get(index) * (finalss * x.getFloat(index)));
    }
    /**
     * FloatTensor weights for GPU processing
     * @param out output tensor
     * @param x target
     * @param weight Weights loaded to floattensor
     * @param size size of tensor
     * @param rmsNormEps epsilon omega point
     */
    static void rmsnorm(FloatTensor out, FloatTensor x, FloatTensor weight, int size, float rmsNormEps) {
    
    	// calculate sum of squares
    	float ss = x.reduce(0, size, 0f, (acc, xi) -> acc + xi * xi);
    	ss /= size;
    	ss += rmsNormEps;
    	ss = (float) (1.0 / Math.sqrt(ss));
    	// normalize and scale
    	final float finalss = ss; // for the lambda
    	out.mapWithIndexInPlace(0, size, (value, index) -> weight.getFloat(index) * (finalss * x.getFloat(index)));
    }
   
    /**
     * LLM generation entry point, ingest prompt tokens and generates new tokens.
     *
     * <p>
     * All prompt tokens are ingested first, then inference starts, until a stop token is found.
     * The returned tokens only include generated/inferred tokens.
     *
     * @param model            model to run inference (including weights, configuration, tokenizer ...)
     * @param state            state of the model e.g. key/value caches ... this is mutated by this call
     * @param startPosition    start prompt ingestion + inference at this position in the context e.g. useful if state was kept across calls (chained generation). 0 implies run with no previous context.
     * @param promptTokens     prompt tokens to ingest, all the prompt tokens will be ingested, given there's enough capacity left in the context
     * @param stopTokens       set of tokens that abort generation during inference, stop tokens do not affect prompt ingestion
     * @param maxTokens        maximum number of tokens (can go up to {@link Configuration#contextLength context length}
     *                         if this value is negative or greater than {@link Configuration#contextLength context length}
     * @param sampler          {@link Sampler strategy} used to select tokens
     * @param echo             debugging flag, prints ALL, prompt and inferred tokens, to {@link System#err stderr}
     * @param onTokenGenerated callback, if non-null, it's called every time a token is inferred e.g. it's not called when ingesting prompt tokens
     * @return list of generated/inferred tokens, including the stop token, if any e.g. does not include any token from the prompt
     */
    public static List<Integer> generateTokens(Llama model, State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, boolean echo,
                                               IntConsumer onTokenGenerated) {
        long startNanos = System.nanoTime();
        long startGen = 0;
        if (maxTokens < 0 || model.configuration().contextLength < maxTokens) {
            maxTokens = model.configuration().contextLength;
        }
        List<Integer> generatedTokens = new ArrayList<>(maxTokens);
        int token = state.latestToken; // BOS?
        int nextToken;
        int promptIndex = 0;
        for (int position = startPosition; position < maxTokens; ++position) {
            if (promptIndex < promptTokens.size()) {
                final int nTokens = Math.min(maxTokens - position, Math.min(promptTokens.size() - promptIndex, state.batchsize));
                final int[] tokens = new int[nTokens];
                for (int i = 0; i < nTokens; i++) {
                    tokens[i] = promptTokens.get(promptIndex + i);
                    if (echo) {
                        // log prompt token (different color?)
                        System.err.print(Tokenizer.replaceControlCharacters(model.tokenizer().decode(List.of(tokens[i]))));
                    }
                }
                if (echo) {
                    System.out.format("position=%d, promptIdx=%d, promptSize=%d, tokens=%s%n", position, promptIndex, promptTokens.size(), Arrays.toString(tokens));
                }
                // Only compute logits on the very last batch.
                boolean computeLogits = promptIndex + nTokens >= promptTokens.size();
               
                position += nTokens - 1; // -1 -> incremented later in the for loop
                promptIndex += nTokens;
                if (promptIndex < promptTokens.size()) {
                    continue;
                }
                startGen = System.nanoTime();
            } else {
        
            }
            nextToken = 0;//sampler.sampleToken(state.logits);
            if (echo) {
                // log inferred token
                System.err.print(Tokenizer.replaceControlCharacters(model.tokenizer().decode(List.of(nextToken))));
            }
            generatedTokens.add(nextToken);
            if (onTokenGenerated != null) {
                onTokenGenerated.accept(nextToken);
            }
            if (stopTokens.contains(nextToken)) {
                break;
            }
            state.latestToken = token = nextToken;
        }
        long elapsedNanos = System.nanoTime() - startNanos;
        long promptNanos = startGen - startNanos;
        long genNanos = elapsedNanos - startGen + startNanos;
        System.err.printf("%ncontext: %d/%d prompt: %.2f tokens/s (%d) generation: %.2f tokens/s (%d)%n",
                startPosition + promptIndex + generatedTokens.size(), model.configuration().contextLength,
                promptTokens.size() / (promptNanos / 1_000_000_000.0), promptTokens.size(),
                generatedTokens.size() / (genNanos / 1_000_000_000.0), generatedTokens.size());

        return generatedTokens;
    }

    /**
     * Qwen specific calls forwardQwen.
     * @param model
     * @param state
     * @param startPosition
     * @param promptTokens
     * @param stopTokens
     * @param maxTokens
     * @param sampler
     * @param echo
     * @param onTokenGenerated
     * @return
     */
    public static List<Integer> generateTokensQwen(Llama model, State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, boolean echo,
    		IntConsumer onTokenGenerated) {
        long startNanos = System.nanoTime();
        if (maxTokens < 0 || model.configuration().contextLength < maxTokens) {
            maxTokens = model.configuration().contextLength;
        }
        List<Integer> generatedTokens = new ArrayList<>(maxTokens);
        int token = state.latestToken; // BOS?
        int nextToken;
        int promptIndex = 0;
        for (int position = startPosition; position < maxTokens; ++position) {
            //forwardQwen(model, state, token, position);
            if (promptIndex < promptTokens.size()) {
                // Force-pick token from prompt.
                nextToken = promptTokens.get(promptIndex++);
                if (echo) {
                    // log prompt token (different color?)
                    System.err.print(Tokenizer.replaceControlCharacters(model.tokenizer().decode(List.of(nextToken))));
                }
            } else {
                nextToken = 0;// sampler.sampleToken(state.logits);
                if (echo) {
                    // log inferred token
                    System.err.print(Tokenizer.replaceControlCharacters(model.tokenizer().decode(List.of(nextToken))));
                }
                generatedTokens.add(nextToken);
                if (onTokenGenerated != null) {
                    onTokenGenerated.accept(nextToken);
                }
                if (stopTokens.contains(nextToken)) {
                    break;
                }
            }
            state.latestToken = token = nextToken;
        }

        long elapsedNanos = System.nanoTime() - startNanos;
        int totalTokens = promptIndex + generatedTokens.size();
        System.err.printf("%n%.2f tokens/s (%d)%n", totalTokens / (elapsedNanos / 1_000_000_000.0), totalTokens);

        return generatedTokens;
    }
    
}

interface TokenizerInterface {
	 public Map<String, Integer> getSpecialTokens();
	 public boolean isSpecialToken(int tokenIndex);
	 public String decode(List<Integer> tokens);
	 public List<Integer> encodeAsList(String text);
	 public int getTokenType(int tokenIndex);
	 public Collection<? extends Integer> encode(String text);
}
/**
 * Byte Pair Encoding tokenizer.
 * <p>
 * Based on <a href="https://github.com/karpathy/minbpe">minbpe</a>, algorithmically follows along the
 * <a href="https://github.com/openai/gpt-2/blob/master/src/encoder.py">GPT 2 tokenizer</a>
 */
class Tokenizer implements TokenizerInterface {
    private final Pattern compiledPattern;
    private final Vocabulary vocabulary;
    private final Map<Pair<Integer, Integer>, Integer> merges;
    private final Map<String, Integer> specialTokens;
    private int[] tokenTypes; // qwen2

    public String regexPattern() {
        if (compiledPattern == null) {
            return null;
        }
        return compiledPattern.pattern();
    }
    @Override
    public Map<String, Integer> getSpecialTokens() {
        return specialTokens;
    }
    @Override
    public boolean isSpecialToken(int tokenIndex) {
        return specialTokens.containsValue(tokenIndex);
    }
    @Override
    public int getTokenType(int tokenIndex) {
        return tokenTypes[tokenIndex];
    }
    
    public Tokenizer(Vocabulary vocabulary, List<Pair<Integer, Integer>> merges, String regexPattern, Map<String, Integer> specialTokens) {
        this.vocabulary = vocabulary;
        this.compiledPattern = regexPattern != null ? Pattern.compile(regexPattern) : null;
        this.specialTokens = new HashMap<>(specialTokens);
        this.merges = new HashMap<>();
        for (Pair<Integer, Integer> pair : merges) {
            int firstIndex = pair.first();
            int secondIndex = pair.second();
            int mergeIndex = vocabulary.getIndex(vocabulary.get(firstIndex) + vocabulary.get(secondIndex)).orElseThrow();
            this.merges.put(pair, mergeIndex);
        }
    }

    public Tokenizer(Vocabulary vocabulary, List<Pair<Integer, Integer>> merges, String regexPattern, Map<String, Integer> specialTokens, int[] tokenTypes) {
    	this(vocabulary, merges, regexPattern, specialTokens);
    	this.tokenTypes = tokenTypes;
    }
    
    private int[] encodeImpl(Collection<? extends Integer> intc) {
    	return intc.stream().mapToInt(i -> i).toArray();
    }

    /**
     * Unlike {@link #encodeOrdinary(String)}, this function handles special tokens.
     * allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
     * if none_raise, then an error is raised if any special token is encountered in text
     * this is the default tiktoken behavior right now as well
     * any other behavior is either annoying, or a major footgun.
     */
    List<Integer> encode(String text, Set<String> allowedSpecial) {
        // decode the user desire w.r.t. handling of special tokens
        Set<String> special = allowedSpecial;
        assert getSpecialTokens().keySet().containsAll(special);
        if (special.isEmpty()) {
            // shortcut: if no special tokens, just use the ordinary encoding
            return encodeOrdinary(text);
        }

        // otherwise, we have to be careful with potential special tokens in text
        // we handle special tokens by splitting the text
        // based on the occurrence of any exact match with any of the special tokens
        // we can use re.split for this. note that surrounding the pattern with ()
        // makes it into a capturing group, so the special tokens will be included
        String specialPattern = special
                .stream()
                .map(Pattern::quote)
                .collect(Collectors.joining("|", "(", ")"));

        String[] specialChunks = text.split(specialPattern);
        // now all the special characters are separated from the rest of the text
        // all chunks of text are encoded separately, then results are joined
        List<Integer> ids = new ArrayList<>();
        for (String part : specialChunks) {
            if (special.contains(part)) {
                // this is a special token, encode it separately as a special case
                ids.add(getSpecialTokens().get(part));
            } else {
                // this is an ordinary sequence, encode it normally
                ids.addAll(encodeOrdinary(part));
            }
        }
        return ids;
    }

    private static List<String> findAll(Pattern pattern, String text) {
        List<String> allMatches = new ArrayList<>();
        Matcher matcher = pattern.matcher(text);
        while (matcher.find()) {
            allMatches.add(matcher.group());
        }
        return allMatches;
    }

    /**
     * Encoding that ignores any special tokens.
     */
    public List<Integer> encodeOrdinary(String text) {
        // split text into chunks of text by categories defined in regex pattern
        List<String> textChunks = findAll(compiledPattern, text);
        // all chunks of text are encoded separately, then results are joined
        List<Integer> ids = new ArrayList<>();
        for (String chunk : textChunks) {
            List<Integer> chunkIds = encodeChunk(chunk);
            ids.addAll(chunkIds);
        }
        return ids;
    }

    private Map<Pair<Integer, Integer>, Integer> getStats(List<Integer> ids) {
        Map<Pair<Integer, Integer>, Integer> map = new HashMap<>();
        for (int i = 0; i + 1 < ids.size(); i++) {
            Pair<Integer, Integer> key = new Pair<>(ids.get(i), ids.get(i + 1));
            map.put(key, map.getOrDefault(key, 0) + 1);
        }
        return map;
    }

    private List<Integer> encodeChunk(String chunk) {
        // return the token ids
        // let's begin. first, convert all bytes to integers in range 0..255
        List<Integer> ids = new ArrayList<>();
        for (int b : chunk.toCharArray()) {
            int tokenIndex = this.vocabulary.getIndex(String.valueOf((char) b)).orElseThrow();
            ids.add(tokenIndex);
        }

        while (ids.size() >= 2) {
            // find the pair with the lowest merge index
            Map<Pair<Integer, Integer>, Integer> stats = getStats(ids);
            Pair<Integer, Integer> pair = stats.keySet().stream().min(Comparator.comparingInt(key -> this.merges.getOrDefault(key, Integer.MAX_VALUE))).orElseThrow();
            // subtle: if there are no more merges available, the key will
            // result in an inf for every single pair, and the min will be
            // just the first pair in the list, arbitrarily
            // we can detect this terminating case by a membership check
            if (!this.merges.containsKey(pair)) {
                break; // nothing else can be merged anymore
            }
            // otherwise let's merge the best pair (lowest merge index)
            int idx = this.merges.get(pair);
            ids = merge(ids, pair, idx);
        }
        return ids;
    }

    private static List<Integer> merge(List<Integer> ids, Pair<Integer, Integer> pair, int idx) {
        List<Integer> newids = new ArrayList<>();
        int i = 0;
        while (i < ids.size()) {
            // if not at the very last position AND the pair matches, replace it
            if (ids.get(i).equals(pair.first()) && i < ids.size() - 1 && ids.get(i + 1).equals(pair.second())) {
                newids.add(idx);
                i += 2;
            } else {
                newids.add(ids.get(i));
                i += 1;
            }
        }
        return newids;
    }

    public String decodeImpl(List<Integer> tokens) {
        StringBuilder sb = new StringBuilder();
        for (int token : tokens) {
            String tokenString = vocabulary.get(token);
            sb.append(tokenString);
        }
        return sb.toString();
    }

    /**
     * Returns list of utf-8 byte and a corresponding list of unicode strings.
     * The reversible bpe codes work on unicode strings.
     * This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
     * When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
     * This is a significant percentage of your normal, say, 32K bpe vocab.
     * To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
     * And avoids mapping to whitespace/control characters the bpe code barfs on.
     */
    private static Map<Integer, Integer> bytesToUnicode() {
        List<Integer> bs = new ArrayList<>();
        IntStream.rangeClosed('!', '~').forEach(bs::add);
        IntStream.rangeClosed('¡', '¬').forEach(bs::add);
        IntStream.rangeClosed('®', 'ÿ').forEach(bs::add);

        List<Integer> cs = new ArrayList<>(bs);
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (!bs.contains(b)) {
                bs.add(b);
                cs.add(256 + n);
                n += 1;
            }
        }

        // return dict(zip(bs, cs))
        return IntStream.range(0, bs.size())
                .boxed()
                .collect(Collectors.toMap(bs::get, cs::get));
    }

    static final Map<Integer, Integer> BYTE_ENCODER = bytesToUnicode();
    static final Map<Integer, Integer> BYTE_DECODER = BYTE_ENCODER.entrySet()
            .stream()
            .collect(Collectors.toMap(Map.Entry::getValue, Map.Entry::getKey));

    public Collection<? extends Integer> encode(String text) {
        StringBuilder sb = new StringBuilder();
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        for (byte b : bytes) {
        	sb.appendCodePoint(BYTE_ENCODER.get(Byte.toUnsignedInt(b)));
        }
        return encode(sb.toString(), Set.of());
    }

    public static String replaceControlCharacters(int[] codePoints) {
        // we don't want to print control characters
        // which distort the output (e.g. \n or much worse)
        // https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
        // http://www.unicode.org/reports/tr44/#GC_Values_Table\
        StringBuilder chars = new StringBuilder();
        for (int cp : codePoints) {
            if (Character.getType(cp) == Character.CONTROL && cp != '\n') {
                chars.append("\\u").append(HexFormat.of().toHexDigits(cp, 4)); // escape
            } else {
                chars.appendCodePoint(cp); // this character is ok
            }
        }
        return chars.toString();
    }

    public static String replaceControlCharacters(String str) {
        return replaceControlCharacters(str.codePoints().toArray());
    }
    @Override
    public List<Integer> encodeAsList(String text) {
        return Arrays.stream(encodeImpl(encode(text))).boxed().toList();
    }
    @Override
    public String decode(List<Integer> tokens) {
        String decoded = decodeImpl(tokens);
        int[] decodedBytesAsInts = decoded.codePoints().map(BYTE_DECODER::get).toArray();
        byte[] rawBytes = new byte[decodedBytesAsInts.length];
        for (int i = 0; i < decoded.length(); i++) {
            rawBytes[i] = (byte) decodedBytesAsInts[i];
        }
        return new String(rawBytes, StandardCharsets.UTF_8);
    }
}

/**
 * Wherein Llama models metadata.get("tokenizer.ggml.model") = gpt2
 * and Mistral uses metadata.get("tokenizer.ggml.model") = llama.
 */
class MistralTokenizer implements TokenizerInterface {
    private final Pattern compiledPattern;
    private final Vocabulary vocabulary;
    private final Map<String, Integer> specialTokens;
    private final int[] tokenType;
    private final int byte0;

    public String regexPattern() {
        if (compiledPattern == null) {
            return null;
        }
        return compiledPattern.pattern();
    }
    @Override
    public Map<String, Integer> getSpecialTokens() {
        return specialTokens;
    }
    @Override
    public boolean isSpecialToken(int tokenIndex) {
        return getTokenType(tokenIndex) != 1;
    }
    @Override
    public int getTokenType(int tokenIndex) {
        return tokenType[tokenIndex];
    }

    public MistralTokenizer(Vocabulary vocabulary, String regexPattern, Map<String, Integer> specialTokens, int[] tokenType) {
        this.vocabulary = vocabulary;
        this.compiledPattern = regexPattern != null ? Pattern.compile(regexPattern) : null;
        this.specialTokens = new HashMap<>(specialTokens);
        this.tokenType = tokenType;
        this.byte0 = vocabulary.getIndex("<0x00>").orElseThrow();
    }

    public List<Integer> encode(String text) {
        return encodeImpl(text.replace(' ', '▁'));
    }

    private List<Integer> encodeImpl(String text) {

        List<Integer> tokens = new ArrayList<>();

        // first encode every individual codepoint in the input string
        for (int i = 0, cpi; i < text.length(); i += Character.charCount(cpi)) {
            cpi = text.codePointAt(i);

            String singleCodepoint = Character.toString(cpi);
            int id = vocabulary.getIndex(singleCodepoint).orElse(-1);

            if (id != -1) {
                // we found this codepoint in vocab, add it as a token
                tokens.add(id);
            } else {
                // byte_fallback encoding: just encode each byte as a token
                // +byte0 here to skip all the control and special tokens e.g. <unk>, <s>, </s>
                // so the individual bytes only start at token <0x00>
                for (byte b : singleCodepoint.getBytes(StandardCharsets.UTF_8)) {
                    tokens.add(Byte.toUnsignedInt(b) + byte0);
                }
            }
        }


        // merge the best consecutive pair each iteration, according the scores in vocab_scores
        while (true) {
            float best_score = -1e10f;
            int best_id = -1;
            int best_idx = -1;

            for (int i = 0; i < tokens.size() - 1; ++i) {
                // check if we can merge the pair (tokens[i], tokens[i+1])
                String str_buffer = vocabulary.get(tokens.get(i)) + vocabulary.get(tokens.get(i + 1));
                int id = vocabulary.getIndex(str_buffer).orElse(-1);
                if (id != -1 && vocabulary.getScore(id) > best_score) {
                    // this merge pair exists in vocab! record its score and position
                    best_score = vocabulary.getScore(id);
                    best_id = id;
                    best_idx = i;
                }
            }

            if (best_idx == -1) {
                break; // we couldn't find any more pairs to merge, so we're done
            }

            // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
            tokens.set(best_idx, best_id);
            tokens.remove(best_idx + 1);
        }

        return tokens;
    }
    @Override
    public String decode(List<Integer> tokens) {
        StringBuilder sb = new StringBuilder();
        for (int token : tokens) {
            String tokenString = vocabulary.get(token);
            if (isSpecialToken(token)) {
                // some tokens designate raw bytes e.g. '<0x10>'
                String prefix = "<0x";
                String suffix = ">";
                if (tokenString.length() == 6 && tokenString.startsWith(prefix) && tokenString.endsWith(suffix)) {
                    String code = tokenString.substring(prefix.length(), tokenString.length() - suffix.length());
                    int cp = Integer.parseInt(code, 16);
                    tokenString = Character.toString(cp);
                }
            } else {
                tokenString = tokenString.replace('▁', ' ');

            }
            sb.append(tokenString);
        }
        return sb.toString();
    }

    public static String replaceControlCharacters(int[] codePoints) {
        // we don't want to print control characters
        // which distort the output (e.g. \n or much worse)
        // https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
        // http://www.unicode.org/reports/tr44/#GC_Values_Table\
        StringBuilder chars = new StringBuilder();
        for (int cp : codePoints) {
            if (Character.getType(cp) == Character.CONTROL && cp != '\n') {
                chars.append("\\u").append(HexFormat.of().toHexDigits(cp, 4)); // escape
            } else {
                chars.appendCodePoint(cp); // this character is ok
            }
        }
        return chars.toString();
    }

    public static String replaceControlCharacters(String str) {
        return replaceControlCharacters(str.codePoints().toArray());
    }

    public List<Integer> encodeAsList(String text) {
        return encode(text);
    }
}

final class Parallel {
    public static void parallelFor(int startInclusive, int endExclusive, IntConsumer action) {
        if (startInclusive == 0 && endExclusive == 1) {
            action.accept(0);
            return;
        }
        IntStream.range(startInclusive, endExclusive).parallel().forEach(action);
    }

    public static void parallelForLong(long startInclusive, long endExclusive, LongConsumer action) {
        if (startInclusive == 0 && endExclusive == 1) {
            action.accept(0);
            return;
        }
        LongStream.range(startInclusive, endExclusive).parallel().forEach(action);
    }
}

record Pair<First, Second>(First first, Second second) {
}


final class RoPE {
	/**
	 * For GPT2 vocab
	 * @param contextLength
	 * @param headSize
	 * @param theta
	 * @param ropeScaling
	 * @param scaleFactor
	 * @param loFreqFactor
	 * @param hiFreqFactor
	 * @param oldContextLength
	 * @return
	 */
    public static Pair<float[], float[]> precomputeFreqsCis(int contextLength, int headSize, double theta,
                                                            boolean ropeScaling, float scaleFactor, float loFreqFactor, float hiFreqFactor, float oldContextLength) {
        assert headSize % 2 == 0;
        float[] cr = new float[contextLength * (headSize / 2)];
        float[] ci = new float[contextLength * (headSize / 2)];
        int n = 0;
        for (int pos = 0; pos < contextLength; ++pos) {
            for (int i = 0; i < headSize; i += 2) {
                float freq = (float) (1.0 / Math.pow(theta, i / (double) headSize));
                if (ropeScaling) {
                    // Llama 3.1 scaling
                    float loFreqWavelen = oldContextLength / loFreqFactor;
                    float hiFreqWavelen = oldContextLength / hiFreqFactor;
                    float wavelen = (float) (2.0 * Math.PI / freq);
                    if (wavelen < hiFreqWavelen) {
                        freq = freq;
                    } else if (wavelen > loFreqWavelen) {
                        freq = freq / scaleFactor;
                    } else {
                        float smooth = (oldContextLength / wavelen - loFreqFactor) / (hiFreqFactor - loFreqFactor);
                        freq = (1.0f - smooth) * freq / scaleFactor + smooth * freq;
                    }
                }
                float val = pos * freq;
                cr[n] = (float) Math.cos(val);
                ci[n] = (float) Math.sin(val);
                n++;
            }
        }
        assert contextLength * (headSize / 2) == n;
        return new Pair<>(cr, ci);
    }
    /**
     * for Llama vocab
     * @param contextLength
     * @param headSize
     * @param theta
     * @return
     */
    public static Pair<float[], float[]> precomputeFreqsCis(int contextLength, int headSize, double theta) {
        assert headSize % 2 == 0;
        float[] cr = new float[contextLength * (headSize / 2)];
        float[] ci = new float[contextLength * (headSize / 2)];
        int n = 0;
        for (int pos = 0; pos < contextLength; ++pos) {
            for (int i = 0; i < headSize; i += 2) {
                float freq = (float) (1.0 / Math.pow(theta, i / (double) headSize));
                float val = pos * freq;
                cr[n] = (float) Math.cos(val);
                ci[n] = (float) Math.sin(val);
                n++;
            }
        }
        assert contextLength * (headSize / 2) == n;
        return new Pair<>(cr, ci);
    }

}

record Vocabulary(String[] tokens, float[] scores, Map<String, Integer> tokenToIndex) {
    public Vocabulary(String[] vocabulary, float[] scores) {
        this(vocabulary, scores,
                IntStream.range(0, vocabulary.length)
                        .boxed()
                        .collect(Collectors.toMap(i -> vocabulary[i], i -> i))
        );
    }

    public String get(int tokenIndex) {
        return tokens[tokenIndex];
    }

    public OptionalInt getIndex(String token) {
        Integer value = tokenToIndex.get(token);
        return value != null ? OptionalInt.of(value) : OptionalInt.empty();
    }

    public int size() {
        return tokens.length;
    }
    /**
     * Added from Mistral Vocabulary - Groff
     * @param tokenIndex
     * @return
     */
    public float getScore(int tokenIndex) {
        return scores[tokenIndex];
    }
    
    public boolean scoresNull() {
    	return scores == null;
    }

}

interface ChatFormatInterface {
	 public TokenizerInterface getTokenizer();
	 public Set<Integer> getStopTokens();
	 public List<Integer> encodeHeader(ChatFormat.Message message);
	 public List<Integer> encodeMessage(ChatFormat.Message message);
	 public List<Integer> encodeDialogPrompt(boolean appendAssistantTurn, List<ChatFormat.Message> dialog);
	 public int getBeginOfText();
}
/**
 * Utility tailored for Llama 3 instruct prompt format.
 */
class ChatFormat implements ChatFormatInterface {

    final Tokenizer tokenizer;
    final int beginOfText;
    final int endHeader;
    final int startHeader;
    final int endOfTurn;
    final int endOfText;
    final int endOfMessage;
    final Set<Integer> stopTokens;

    public ChatFormat(TokenizerInterface tokenizer) {
        this.tokenizer = (Tokenizer)tokenizer;
        Map<String, Integer> specialTokens = this.tokenizer.getSpecialTokens();
        this.beginOfText = specialTokens.get("<|begin_of_text|>");
        this.startHeader = specialTokens.get("<|start_header_id|>");
        this.endHeader = specialTokens.get("<|end_header_id|>");
        this.endOfTurn = specialTokens.get("<|eot_id|>");
        this.endOfText = specialTokens.get("<|end_of_text|>");
        this.endOfMessage = specialTokens.getOrDefault("<|eom_id|>", -1); // only in 3.1
        this.stopTokens = Set.of(endOfText, endOfTurn);
    }
    @Override
    public TokenizerInterface getTokenizer() {
        return tokenizer;
    }
    @Override
    public Set<Integer> getStopTokens() {
        return stopTokens;
    }
    @Override
    public int getBeginOfText() {
    	return beginOfText;
    }
    @Override
    public List<Integer> encodeHeader(ChatFormat.Message message) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(startHeader);
        tokens.addAll(this.tokenizer.encodeAsList(message.role().name()));
        tokens.add(endHeader);
        tokens.addAll(this.tokenizer.encodeAsList("\n"));
        return tokens;
    }
    @Override
    public List<Integer> encodeMessage(ChatFormat.Message message) {
        List<Integer> tokens = this.encodeHeader(message);
        tokens.addAll(this.tokenizer.encodeAsList(message.content().strip()));
        tokens.add(endOfTurn);
        return tokens;
    }
    @Override
    public List<Integer> encodeDialogPrompt(boolean appendAssistantTurn, List<ChatFormat.Message> dialog) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(beginOfText);
        for (ChatFormat.Message message : dialog) {
            tokens.addAll(this.encodeMessage(message));
        }
        if (appendAssistantTurn) {
            // Add the start of an assistant message for the model to complete.
            tokens.addAll(this.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
        }
        return tokens;
    }

    public record Message(ChatFormat.Role role, String content) {
    }

    public record Role(String name) {
        public static ChatFormat.Role SYSTEM = new ChatFormat.Role("system");
        public static ChatFormat.Role USER = new ChatFormat.Role("user");
        public static ChatFormat.Role ASSISTANT = new ChatFormat.Role("assistant");

        @Override
        public String toString() {
            return name;
        }
    }
}

/**
* Utility tailored for Mistral v0.3 instruct prompt format.
*/
final class MistralChatFormat implements ChatFormatInterface {

   protected final TokenizerInterface tokenizer;
   protected final int unknownToken;
   protected final int beginOfText;
   protected final int endOfText;
   protected final int beginOfInstruction;
   protected final int endOfInstruction;
   protected final int toolCalls;
   protected final int beginOfAvailableTools;
   protected final int endOfAvailableTools;
   protected final int beginOfToolResults;
   protected final int endOfToolResults;
   protected final int prefix;
   protected final int middle;
   protected final int suffix;

   public MistralChatFormat(TokenizerInterface tokenizer) {
       this.tokenizer = tokenizer;
       Map<String, Integer> specialTokens = this.tokenizer.getSpecialTokens();
       this.unknownToken = specialTokens.get("<unk>");
       this.beginOfText = specialTokens.get("<s>");
       this.endOfText = specialTokens.get("</s>");
       this.beginOfInstruction = specialTokens.get("[INST]");
       this.endOfInstruction = specialTokens.get("[/INST]");
       this.toolCalls = specialTokens.get("[TOOL_CALLS]");
       this.beginOfAvailableTools = specialTokens.get("[AVAILABLE_TOOLS]");
       this.endOfAvailableTools = specialTokens.get("[/AVAILABLE_TOOLS]");
       this.beginOfToolResults = specialTokens.get("[TOOL_RESULTS]");
       this.endOfToolResults = specialTokens.get("[/TOOL_RESULTS]");
       // Only Codestral supports FIM tokens.
       this.prefix = specialTokens.getOrDefault("[PREFIX]", unknownToken);
       this.suffix = specialTokens.getOrDefault("[SUFFIX]", unknownToken);
       this.middle = specialTokens.getOrDefault("[MIDDLE]", unknownToken);
   }
   @Override
   public TokenizerInterface getTokenizer() {
       return tokenizer;
   }
   @Override
   public Set<Integer> getStopTokens() {
       return Set.of(endOfText);
   }
   @Override
   public int getBeginOfText() {
   	return beginOfText;
   }
 
   public List<Integer> encodeMessage(String userMessage, boolean addHeader, boolean addFooter) {
       List<Integer> tokens = new ArrayList<>();
       if (addHeader) {
           tokens.add(this.beginOfInstruction);
       }
       if (userMessage != null) {
           tokens.addAll(this.tokenizer.encodeAsList(userMessage.strip()));
       }
       if (addFooter) {
           tokens.add(endOfInstruction);
       }
       return tokens;
   }

   public List<Integer> encodeFillInTheMiddle(String prefix, String suffix) {
       List<Integer> tokens = new ArrayList<>();
       tokens.add(this.suffix);
       tokens.addAll(tokenizer.encode(suffix));
       tokens.add(this.prefix);
       tokens.addAll(tokenizer.encode(prefix));
       return tokens;
   }
   @Override
   public List<Integer> encodeHeader(ChatFormat.Message message) {
       List<Integer> tokens = new ArrayList<>();
       tokens.add(this.beginOfInstruction);
       tokens.addAll(this.tokenizer.encodeAsList(message.role().name()));
       tokens.add(endOfInstruction);
       return tokens;
   }
   @Override
   public List<Integer> encodeMessage(ChatFormat.Message message) {
	   List<Integer> tokens = new ArrayList<>();
	   tokens.add(this.beginOfInstruction);
       tokens.addAll(this.tokenizer.encodeAsList(message.content().strip()));
       tokens.add(endOfInstruction);
       return tokens;
   }
   @Override
   public List<Integer> encodeDialogPrompt(boolean appendAssistantTurn, List<ChatFormat.Message> dialog) {
       List<Integer> tokens = new ArrayList<>();
       tokens.add(beginOfText);
       for (ChatFormat.Message message : dialog) {
           tokens.addAll(this.encodeMessage(message));
       }
       //if (appendAssistantTurn) {
       //    // Add the start of an assistant message for the model to complete.
       //    tokens.addAll(this.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
       //}
       tokens.add(endOfText);
       return tokens;
   }
}

/**
 * Utility tailored for the Chat Markup Language (ChatML) Qwen prompt format.
 */
class ChatMLFormat implements ChatFormatInterface {

    protected final TokenizerInterface tokenizer;
    protected final int imStart;
    protected final int endOfText;
    protected final int imEnd;

    public ChatMLFormat(TokenizerInterface tokenizer) {
        this.tokenizer = tokenizer;
        Map<String, Integer> specialTokens = this.tokenizer.getSpecialTokens();
        this.imStart = specialTokens.get("<|im_start|>");
        this.imEnd = specialTokens.get("<|im_end|>");
        this.endOfText = specialTokens.get("<|endoftext|>");
    }

    public TokenizerInterface getTokenizer() {
        return tokenizer;
    }

    public Set<Integer> getStopTokens() {
        return Set.of(imEnd, endOfText);
    }
    
    @Override
    public int getBeginOfText() {
    	return imStart;
    }
    
    public List<Integer> encodeHeader(ChatFormat.Message message) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(imStart);
        tokens.addAll(this.tokenizer.encodeAsList(message.role().name()));
        tokens.addAll(this.tokenizer.encodeAsList("\n"));
        return tokens;
    }

    public List<Integer> encodeMessage(ChatFormat.Message message) {
        List<Integer> tokens = this.encodeHeader(message);
        tokens.addAll(this.tokenizer.encodeAsList(message.content().strip()));
        tokens.add(imEnd);
        return tokens;
    }

    public List<Integer> encodeDialogPrompt(boolean appendAssistantTurn, List<ChatFormat.Message> dialog) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(imStart);
        for (ChatFormat.Message message : dialog) {
            tokens.addAll(this.encodeMessage(message));
        }
        if (appendAssistantTurn) {
            // Add the start of an assistant message for the model to complete.
            tokens.addAll(this.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
        }
        return tokens;
    }

}

/**
 * Support for AOT preloading of GGUF metadata with GraalVM's Native Image.
 *
 * <p>
 * To preload a model at build time, pass {@code -Dllama.PreloadGGUF=/path/to/model.gguf}
 * to the native-image builder command. At runtime, the preloaded model will be used
 * iff the specified and preloaded file names (base name) match.
 */
final class AOT {
 
   
    private static void preLoadGGUF(String modelPath) {
        if (modelPath == null || modelPath.isEmpty()) {
           
        }
        try {
            Path path = Path.of(modelPath);
            if (!Files.exists(path) || !Files.isRegularFile(path)) {
                throw new IllegalArgumentException("Cannot pre-load model: " + path);
            }
      
            try (FileChannel fileChannel = FileChannel.open(path, StandardOpenOption.READ)) {
               
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

   
}

