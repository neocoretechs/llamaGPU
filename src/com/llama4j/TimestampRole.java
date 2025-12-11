package com.llama4j;

import java.io.Serializable;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.Objects;

/**
 * Serves as morphism map for relating LSH index to token vector
 */
class TimestampRole implements Serializable, Comparable {
	private static final long serialVersionUID = 1L;
	private Long timestamp;
	private ChatFormat.Role role;
	public TimestampRole() {}
	public TimestampRole(Long timestamp, ChatFormat.Role role) {
		this.timestamp = timestamp;
		this.role = role;
	}
	public Long getTimestamp() {
		return timestamp;
	}
	public void setTimestamp(Long timestamp) {
		this.timestamp = timestamp;
	}
	public ChatFormat.Role getRole() {
		return role;
	}
	public void setRole(ChatFormat.Role role) {
		this.role = role;
	}
	@Override
	public int hashCode() {
		return Objects.hash(role, timestamp);
	}
	@Override
	public boolean equals(Object obj) {
		if (this == obj) {
			return true;
		}
		if (!(obj instanceof TimestampRole)) {
			return false;
		}
		TimestampRole other = (TimestampRole) obj;
		return role == other.role && Objects.equals(timestamp, other.timestamp);
	}
	@Override
	public String toString() {
		return LocalDateTime.ofInstant(Instant.ofEpochMilli(timestamp), ZoneId.systemDefault()).toString()+" "+role.getRole();
	}
	@Override
	public int compareTo(Object o) {
		int res;
		res = timestamp.compareTo(((TimestampRole)o).timestamp);
		if(res != 0)
			return res;
		return role.compareTo(((TimestampRole)o).role);
	}	
}
