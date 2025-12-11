package com.llama4j;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Utility tailored for Llama 3 instruct prompt format.
 */
class ChatFormat {
	final int beginOfText;
	final int endHeader;
	final int startHeader;
	final int endOfTurn;
	final int endOfText;
	int endOfMessage;
	final Set<Integer> stopTokens;
	final String startHeaderStr = "<|start_header_id|>";
	final String endHeaderStr = "<|end_header_id|>";
	
	public ChatFormat() {
		StringTensor buf = new StringTensor("<|begin_of_text|> <|end_of_text|>");
		IntTensor it = IntTensor.allocate(8);
		int siz = DeviceManager.stringToToken(buf, it);
		//for(int i = 0; i < siz; i++)
		//	System.out.println(i+".) "+it.getInt(i));
		this.beginOfText = it.getInt(1);
		this.endOfText = it.getInt(3);
		//System.out.println("BOT="+this.beginOfText+" EOT="+this.endOfText);
		it = IntTensor.allocate(8);
		buf = new StringTensor("<|start_header_id|> <|end_header_id|>");
		DeviceManager.stringToToken(buf, it);
		//for(int i = 0; i < siz; i++)
		//	System.out.println(i+".) "+it.getInt(i));
		this.startHeader = it.getInt(1);
		this.endHeader = it.getInt(3);
		//System.out.println("startHeader="+this.startHeader+" endHeader="+this.endHeader);
		buf = new StringTensor("<|eot_id|>");
		it = IntTensor.allocate(8);
		DeviceManager.stringToToken(buf, it);
		this.endOfTurn = it.getInt(1);
		//System.out.println("EOTurn="+this.endOfTurn);
		buf = new StringTensor("<|eom_id|>");
		it = IntTensor.allocate(8);
		DeviceManager.stringToToken(buf, it);
		//this.endOfMessage = it.getInt(1); // only in 3.1
		if(this.endOfMessage == 0)
			this.endOfMessage = -1;
		//System.out.println("EOMessage="+this.endOfMessage);
		this.stopTokens = Set.of(endOfText, endOfTurn);	
	}

	
	public Set<Integer> getStopTokens() {
		return stopTokens;
	}
	public int getBeginOfText() {
		return beginOfText;
	}
	public List<Integer> encodeHeader(ChatFormat.Message message) {
		List<Integer> tokens = new ArrayList<>();
		tokens.add(startHeader);
		tokens.addAll(this.encodeAsList(message.role().name()));
		tokens.add(endHeader);
		tokens.addAll(this.encodeAsList("\n"));
		return tokens;
	}
	public String encodeHeaderString(ChatFormat.Message message) {
		StringBuilder tokens = new StringBuilder();
		tokens.append(startHeaderStr);
		tokens.append(message.role().name());
		tokens.append(endHeaderStr);
		tokens.append("\n");
		return tokens.toString();
	}
	public List<Integer> encodeMessage(ChatFormat.Message message) {
		List<Integer> tokens = this.encodeHeader(message);
		tokens.addAll(this.encodeAsList(message.content().strip()));
		tokens.add(endOfTurn);
		return tokens;
	}
	public List<Integer> encodeMessage(ChatFormat.Message message, List<Integer> tokenList) {
		List<Integer> tokens = this.encodeHeader(message);
		tokens.addAll(tokenList);
		tokens.add(endOfTurn);
		return tokens;
	}
	/**
	 * Encode beginOfText, then follow with list of supplied messages
	 * @param appendAssistantTurn true to add a blank ASSISTANT header at the end of the list of prompts
	 * @param dialog List of messages to tokenize
	 * @return the tokenized list of appended messages
	 */
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
	
	public StringTensor extractDialogPrompt(boolean appendAssistantTurn, List<Message> dialog) {
		StringBuilder sb = new StringBuilder();
		for (ChatFormat.Message message : dialog) {
			sb.append(startHeaderStr);
			sb.append(message.role().getRole());
			sb.append(endHeaderStr);
			sb.append(message.content());
		}
		if (appendAssistantTurn) {
			// Add the start of an assistant message for the model to complete.
			sb.append(startHeaderStr);
			sb.append(ChatFormat.Role.ASSISTANT.getRole());
			sb.append(endHeaderStr);
		}
		return new StringTensor(sb.toString());
	}
	
	public String stripFormatting(String input) {
		return input.replaceAll("<\\|.*?\\|>", "")
				.replaceAll("\\*+", "")
				.replaceAll("(?m)^USER:|AI:", "")
				.trim();
	}

	public record Message(ChatFormat.Role role, String content) {
	}

	public enum Role {
		SYSTEM("SYSTEM"),
		USER("USER"),
		ASSISTANT("ASSISTANT");
		private final String role;
		Role(String role) {
			this.role = role;
		}
		public String getRole() {
			return role;
		}
		@Override
		public String toString() {
			return role;
		}
	}
	  private int[] encodeImpl(Collection<? extends Integer> intc) {
	    	return intc.stream().mapToInt(i -> i).toArray();
	    }

	    private static List<String> findAll(Pattern pattern, String text) {
	        List<String> allMatches = new ArrayList<>();
	        Matcher matcher = pattern.matcher(text);
	        while (matcher.find()) {
	            allMatches.add(matcher.group());
	        }
	        return allMatches;
	    }

	    private Map<Pair<Integer, Integer>, Integer> getStats(List<Integer> ids) {
	        Map<Pair<Integer, Integer>, Integer> map = new HashMap<>();
	        for (int i = 0; i + 1 < ids.size(); i++) {
	            Pair<Integer, Integer> key = new Pair<>(ids.get(i), ids.get(i + 1));
	            map.put(key, map.getOrDefault(key, 0) + 1);
	        }
	        return map;
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
	    	StringTensor st = new StringTensor(text);
			IntTensor it = IntTensor.allocate(2048);
			int toks = DeviceManager.stringToToken(st, it);
			IntTensor trimToks = new IntTensor(it, toks);
	        return Arrays.stream(trimToks.toArray()).boxed().toList();
	    }
	    
	    public Collection<? extends Integer> encodeAsCollection(String text) {
	    	return encodeAsList(text);
	    }
	 
}

