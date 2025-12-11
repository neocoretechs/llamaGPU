package com.llama4j;

import java.util.Collection;
import java.util.List;
import java.util.Set;

/**
 * Encapsulates ChatFormatInterface tokenizer and manages raw and formatted token lists
 */
final class PromptFrame {
	private ChatFormat.Message message;
	private final ChatFormat chatFormat;
	private Collection<? extends Integer> rawTokens;
	private List<Integer> formattedTokens;

	public PromptFrame(ChatFormat format) {
		this.chatFormat = format;
	}
	public void setMessage(ChatFormat.Message message) {
		this.message = message;
		this.rawTokens = chatFormat.encodeAsCollection(chatFormat.stripFormatting(message.content()));
		this.formattedTokens = chatFormat.encodeMessage(message); // Includes headers + role
	}
	public List<Integer> getFormattedTokens() {
		return formattedTokens;
	}
	public int getBeginOfTextToken() {
		return chatFormat.getBeginOfText();
	}
	public Set<Integer> getStopTokens() {
		return chatFormat.getStopTokens();
	}
	public ChatFormat.Message getMessage() {
		return message;
	}
	public Collection<? extends Integer> getRawTokens() {
		return rawTokens;
	}
}

