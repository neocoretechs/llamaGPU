package com.llama4j;

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

import com.llama4j.ffi.NativeLoader;

public final class StringTensor implements Externalizable, Comparable {
	public static boolean DEBUG = false;
	MemorySegment memorySegment;

	private long devicePtr; // 0 if not uploaded
	private boolean uploaded = false;
	private volatile DeviceMemoryReclaim deviceReclaim;

	static short readShort(MemorySegment memorySegment, long offset) {
		return memorySegment.get(ValueLayout.JAVA_SHORT, offset);
	}  
	static int readInt(MemorySegment memorySegment, long offset) {
		return memorySegment.get(ValueLayout.JAVA_INT, offset);
	}
	static float readFloat(MemorySegment memorySegment, long offset) {
		return memorySegment.get(ValueLayout.JAVA_FLOAT, offset);
	}  
	static byte readByte(MemorySegment memorySegment, long offset) {
		return memorySegment.get(ValueLayout.JAVA_BYTE, offset);
	}
	static Character readUTF16(MemorySegment memorySegment, long offset) {
		return memorySegment.get(ValueLayout.JAVA_CHAR, offset);
	}

	public StringTensor() {}

	public StringTensor(String s) {
		copy(s);
	}
	public StringTensor(byte[] b) {
		copy(b);
	}

	public Arena getArena() {
		return Llama3.sharedArena;
	}

	public static byte[] getUTF8(String s) {
		return s.getBytes(StandardCharsets.UTF_8);
	}
	public void allocate(String s) {
		allocate(getUTF8(s));
	}
	public void allocate(byte[] utf8Bytes) {
		memorySegment = getArena().allocate(utf8Bytes.length + 1);
	}
	public void copy(byte[] utf8Bytes) {
		allocate(utf8Bytes);
		memorySegment.copyFrom(MemorySegment.ofArray(utf8Bytes));
		memorySegment.set(ValueLayout.JAVA_BYTE, utf8Bytes.length, (byte)0); // null terminator
	}
	public void copy(String s) {
		copy(getUTF8(s));
	}
	public String toString() {
		int len = 0;
		byte[] utfBytes = memorySegment.toArray(ValueLayout.JAVA_BYTE);
		while(utfBytes[len] != 0) len++;
		return new String(utfBytes, 0, len, StandardCharsets.UTF_8);
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

	byte getByte(int index) {
		return memorySegment.get(ValueLayout.JAVA_BYTE, index);
	}
	void setByte(int index, byte payLoad) {
		memorySegment.set(ValueLayout.JAVA_BYTE, index, payLoad);
	}
	void copyTo(int thisOffset, StringTensor that, int thatOffset, int size) {
		that.mapWithIndexInPlace(thatOffset, size, (value, index) -> this.getSegment().getAtIndex(ValueLayout.JAVA_BYTE, index - thatOffset + thisOffset));
	}

	@FunctionalInterface
	interface MapFunction {
		byte apply(byte value);
	}

	@FunctionalInterface
	interface MapWithIndexFunction {
		byte apply(byte value, int index);
	}

	StringTensor mapInPlace(int thisOffset, int size, MapFunction mapFunction) {
		int endIndex = thisOffset + size;
		for (int i = thisOffset; i < endIndex; ++i) {
			setByte(i, mapFunction.apply(getByte(i)));
		}
		return this;
	}

	StringTensor mapInPlace(MapFunction mapFunction) {
		return mapInPlace(0, size(), mapFunction);
	}
	StringTensor mapWithIndexInPlace(int thisOffset, int size, StringTensor.MapWithIndexFunction mapWithIndexFunction) {
		int endOffset = thisOffset + size;
		for (int i = thisOffset; i < endOffset; ++i) {
			//System.out.println("setFloat:"+i+" of size:"+size);
			setByte(i, mapWithIndexFunction.apply(getByte(i), i));
		}
		return this;
	}

	/**
	 * @param thisOffset offset into this tensor
	 * @param size number of elements to fill
	 * @param value value to fill with
	 * @return this filled tensor
	 */
	StringTensor fillInPlace(int thisOffset, int size, byte value) {
		return mapInPlace(thisOffset, size, unused -> value);
	}
	
	public long strlen() throws Throwable {
		//allocate(s); assume called on ctor
        // Link and call the native function (e.g., strlen)
        Linker linker = Linker.nativeLinker();
        SymbolLookup stdLib = linker.defaultLookup();
        MemorySegment strlenAddr = stdLib.find("strlen").orElseThrow();
        FunctionDescriptor strlenSig = FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS);
        MethodHandle strlen = linker.downcallHandle(strlenAddr, strlenSig);
        // Invoke the native function
        return (long) strlen.invokeExact(getSegment());
	}
	
	public void copyFromNative() {
		long bytes = size();
		MemorySegment hostSeg = getSegment();
		long addr = hostSeg.address(); // strong field keeps reachability
		try {
			Llama3.copyFromNativeMH.invokeExact(addr, bytes);
		} catch (Throwable e) {
			throw new RuntimeException("CopyFromNative transfer failed , "+ this.getSegment(), e);
		}
	}
	
	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		out.writeInt(size());
		out.write(memorySegment.asByteBuffer().array());
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
		int vsize = in.readInt();
		// allocate off-heap space for headSize floats
		memorySegment = getArena().allocate(ValueLayout.JAVA_CHAR, vsize);
	}

	@Override
	public int compareTo(Object o) {
		return toString().compareTo(((StringTensor)o).toString());
	}
	
	public static void main(String[] args) throws Throwable {
		Llama3 llama = new Llama3();
	    NativeLoader.loadMethods();
		StringTensor s = new StringTensor(args[0]);
		DeviceManager.runModel(s);
		//System.out.println(s+" = "+s.strlen());
		//StringTensor y = new StringTensor(new byte[15]);
		//y.copyFromNative();
		//System.out.println(y);
	}

}
