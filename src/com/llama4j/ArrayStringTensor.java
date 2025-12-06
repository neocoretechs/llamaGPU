package com.llama4j;

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.lang.foreign.AddressLayout;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;

public final class ArrayStringTensor implements Externalizable, Comparable {
	private MemorySegment memorySegment;
	private MemorySegment[] buffers;
	private int length;

	public ArrayStringTensor() {
	}

	public ArrayStringTensor(String[] strings) {
		this.memorySegment = getArena().allocate(strings.length * ValueLayout.ADDRESS.byteSize());  
		this.length = strings.length;
		this.buffers = new MemorySegment[length];
		for (int i = 0; i < length; i++) {
			byte[] utf8 = strings[i].getBytes(StandardCharsets.UTF_8);
			buffers[i] = getArena().allocate(utf8.length + 1);
			buffers[i].copyFrom(MemorySegment.ofArray(utf8));
			buffers[i].set(ValueLayout.JAVA_BYTE, utf8.length, (byte)0);
			memorySegment.setAtIndex(AddressLayout.ADDRESS, i, buffers[i]);
		}
	}
	public ArrayStringTensor(List<String> strings) {
		this.memorySegment = getArena().allocate(strings.size() * ValueLayout.ADDRESS.byteSize());  
		this.length = strings.size();
		this.buffers = new MemorySegment[length];
		for (int i = 0; i < length; i++) {
			byte[] utf8 = strings.get(i).getBytes(StandardCharsets.UTF_8);
			buffers[i] = getArena().allocate(utf8.length + 1);
			buffers[i].copyFrom(MemorySegment.ofArray(utf8));
			buffers[i].set(ValueLayout.JAVA_BYTE, utf8.length, (byte)0);
			memorySegment.setAtIndex(AddressLayout.ADDRESS, i, buffers[i]);
		}
	}

	public MemorySegment asCharStarStar() {
		return memorySegment;
	}
	/**
	 * Length of array of string
	 * @return
	 */
	 public int length() {
		return length;
	}
	public Arena getArena() {
		return Llama3.sharedArena;
	}

	public StringTensor getStringTensor(int index) {
		return new StringTensor(buffers[index].asByteBuffer().array());
	}
	public static byte[] getUTF8(String s) {
		return s.getBytes(StandardCharsets.UTF_8);
	}
	public void allocate(int index, String s) {
		allocate(index, getUTF8(s));
	}
	public void allocate(int index, byte[] utf8) {
		buffers[index] = getArena().allocate(utf8.length + 1);
		buffers[index].copyFrom(MemorySegment.ofArray(utf8));
		buffers[index].set(ValueLayout.JAVA_BYTE, utf8.length, (byte)0);
		memorySegment.setAtIndex(AddressLayout.ADDRESS, index, buffers[index]);
	}
	public void copy(int index, byte[] utf8Bytes) {
		allocate(index, utf8Bytes);

	}
	public void copy(int index, String s) {
		copy(index, getUTF8(s));
	}
	public String toString() {
		StringBuilder sb = new StringBuilder();
		for(int i = 0; i < length; i++) {
			int len = 0;
			byte[] utfBytes = buffers[i].toArray(ValueLayout.JAVA_BYTE);
			while(utfBytes[len] != 0) len++;
			sb.append(new String(utfBytes, 0, len, StandardCharsets.UTF_8));
		}
		return sb.toString();
	}
	public int size() {
		return (int) memorySegment.byteSize();
	}
	public boolean isImmutable() {
		return false;
	}   

	public static int numberOfElements(int... dimensions) {
		assert Arrays.stream(dimensions).allMatch(i -> i > 0);
		return Arrays.stream(dimensions).reduce(Math::multiplyExact).orElseThrow();
	}

	public MemorySegment getSegment() {
		return memorySegment;
	}

	byte getByte(int indexThis, int offset) {
		return buffers[indexThis].get(ValueLayout.JAVA_BYTE, offset);
	}
	void setByte(int indexThis, int index, byte payLoad) {
		buffers[indexThis].set(ValueLayout.JAVA_BYTE, index, payLoad);
	}
	void copyTo(int indexThis, int thisOffset, ArrayStringTensor that, int indexThat, int thatOffset, int size) {
		that.mapWithIndexInPlace(indexThat, thatOffset, size, (value, index) -> buffers[indexThis].getAtIndex(ValueLayout.JAVA_BYTE, index - thatOffset + thisOffset));
	}

	@FunctionalInterface
	interface MapFunction {
		byte apply(byte value);
	}

	@FunctionalInterface
	interface MapWithIndexFunction {
		byte apply(byte value, int indexThis, int index);
	}

	ArrayStringTensor mapInPlace(int indexThis, int thisOffset, int size, MapFunction mapFunction) {
		int endIndex = thisOffset + size;
		for (int i = thisOffset; i < endIndex; ++i) {
			setByte(indexThis,i, mapFunction.apply(getByte(indexThis,i)));
		}
		return this;
	}

	ArrayStringTensor mapInPlace(int indexThis, MapFunction mapFunction) {
		return mapInPlace(indexThis, 0, size(), mapFunction);
	}
	ArrayStringTensor mapWithIndexInPlace(int indexThis, int thisOffset, int size, StringTensor.MapWithIndexFunction mapWithIndexFunction) {
		int endOffset = thisOffset + size;
		for (int i = thisOffset; i < endOffset; ++i) {
			//System.out.println("setFloat:"+i+" of size:"+size);
			setByte(indexThis,i, mapWithIndexFunction.apply(getByte(indexThis,i), i));
		}
		return this;
	}

	/**
	 * @param thisOffset offset into this tensor
	 * @param size number of elements to fill
	 * @param value value to fill with
	 * @return this filled tensor
	 */
	ArrayStringTensor fillInPlace(int indexThis, int thisOffset, int size, byte value) {
		return mapInPlace(indexThis, thisOffset, size, unused -> value);
	}

	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		out.writeInt(size());
		out.write(memorySegment.asByteBuffer().array());
		out.writeInt(length);
		for(int i = 0; i < length; i++) {
			out.writeInt((int) buffers[i].byteSize());
			out.write(buffers[i].asByteBuffer().array());
		}
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
		int vsize = in.readInt();
		// allocate off-heap space for headSize floats
		memorySegment = getArena().allocate(ValueLayout.ADDRESS, vsize);
		length = in.readInt();
		for(int i = 0; i < length; i++) {
			vsize = in.readInt();
			byte[] utf8 = new byte[vsize];
			in.read(utf8);
			buffers[i] = getArena().allocate(utf8.length + 1);
			buffers[i].copyFrom(MemorySegment.ofArray(utf8));
			buffers[i].set(ValueLayout.JAVA_BYTE, utf8.length, (byte)0);
			memorySegment.setAtIndex(ValueLayout.ADDRESS, i, buffers[i]);
		}
	}

	@Override
	public int compareTo(Object o) {
		return toString().compareTo(((ArrayStringTensor)o).toString());
	}

}
