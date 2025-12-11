package com.llama4j;

import java.io.IOException;
import java.io.PrintStream;
import java.io.PrintWriter;

import java.lang.foreign.Arena;
import java.lang.invoke.MethodHandle;

import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;

import java.util.function.IntConsumer;
import java.util.function.LongConsumer;
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

import com.llama4j.ffi.NativeLoader;
/**
 * Foreign Function Interface to Llama.cpp model runner to take full advantage to GPU enabled platforms.
 * Most of the internal machinery of inference is abstracted away behind the Llama.cpp native runtime,
 * despite a lot of utility function native handles being exposed for convenience purposes and for future research.
 * We trade prompt strings and tokenized responses to and fro.<p>
 * Remember: Llama models use GPT2 vocabulary while non-Llama models use Llama vocabulary!
 * @author Jonathan Groff Copyright (C) NeoCoreTechs 2025
 */
public class Llama3 {
	private static final Log log = LogFactory.getLog(Llama3.class);
    public final static boolean DEBUG = false;
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
	public static MethodHandle copyFromNativeMH;
	public static MethodHandle freeDevicePtr;
	public static MethodHandle allocDevicePtr;
	public static MethodHandle cudaInit;
	public static MethodHandle copyDeviceToHostMH;
	public static MethodHandle runModelMH;
	public static MethodHandle runModelTokenizeMH;
	public static MethodHandle loadModelMH;
	public static MethodHandle stringToTokenMH;
	public static MethodHandle tokenToStringMH;
	
	static Options options = null;
	
	static {
		NativeLoader.load();
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
	
    public static void main(String[] args) throws IOException {
        NativeLoader.loadMethods();
        options = Options.parseOptions(args);
	    NativeLoader.loadMethods();
		StringTensor s = new StringTensor(options.modelPath().toString());
		try(Timer _ = Timer.log("load model")) {
			DeviceManager.loadModel(s, options.getMaxTokens());
		}

        if (options.interactive()) {
            ChatFormat chatFormat = new ChatFormat();
            List<ChatFormat.Message> dialog = new ArrayList<ChatFormat.Message>();
            Scanner in = new Scanner(System.in);
            loop: while (true) {
            	//boolean storeDb = true;
                System.out.print("> ");
                System.out.flush();
                String userText = in.nextLine();
                switch (userText) {
                    case "/quit":
                    case "/exit": break loop;
                }
                ChatFormat.Message responseMessage = new ChatFormat.Message(ChatFormat.Role.USER, userText);
                dialog.add(responseMessage);
                //List<Integer> dialogTokens = chatFormat.encodeDialogPrompt(true, dialog);
                //IntTensor it = new IntTensor(dialogTokens);
                //StringTensor p = new StringTensor(new byte[dialogTokens.size()+2]);
                //DeviceManager.tokenToString(it, dialogTokens.size(), p);
                StringTensor p = chatFormat.extractDialogPrompt(true, dialog);
        		System.out.println("prompt:"+p);
        		int tokNum = 0;
        		IntTensor retTokens = IntTensor.allocate(options.getMaxTokens());
        		try(Timer _ = Timer.log("run model interactive")) {
        			tokNum = DeviceManager.runModelTokenize(p, options.temperature(), options.minp(), options.topp(), retTokens);
        			System.out.println("Returned Tokens="+tokNum);
        		}
        		if(tokNum == -1) {
        			log.error("Context length exceeded, exiting");
        			break;
        		}
        		StringTensor toks = new StringTensor(new byte[options.getMaxTokens()]);
        		int strLen = DeviceManager.tokenToString(retTokens, tokNum, toks);
        		System.out.println("returned prompt len="+strLen);
        		System.out.println(toks.toString().substring(0,strLen));
                responseMessage = new ChatFormat.Message(ChatFormat.Role.ASSISTANT, toks.toString().substring(0,strLen));
                dialog.add(responseMessage);
            }
            in.close();
        } else {
        	StringTensor p = new StringTensor(options.prompt());
    		System.out.println("prompt:"+p);
    		IntTensor it = IntTensor.allocate(2048);
    		try(Timer _ = Timer.log("run model")) {
    			int tokNum = DeviceManager.runModel(p, options.temperature(), options.minp(), options.topp(), it);
    			System.out.println("Tokens="+tokNum);
    		}
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
