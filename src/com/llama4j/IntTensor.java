package com.llama4j;

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Arrays;

import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorSpecies;
/**
 * IntTensor maintaining array of ints backed by a MemorySegment.
 */
final class IntTensor implements Externalizable, Comparable {
	public static boolean DEBUG = false;
	MemorySegment memorySegment;
	private boolean modified = false;

	public IntTensor() {
	}

	IntTensor(int[] values) {
		memorySegment = getArena().allocate(ValueLayout.JAVA_INT, values.length);
		MemorySegment.copy(values, 0, memorySegment, ValueLayout.JAVA_INT, 0, values.length);
	}

	public static IntTensor allocate(int... dims) {
		int numberOfElements = FloatTensor.numberOfElements(dims);
		return new IntTensor(new int[numberOfElements]);
	}

	public int size() {
		return (int) (memorySegment.byteSize() / Integer.BYTES);
	}

	public int getInt(int index) {
		return memorySegment.getAtIndex(ValueLayout.JAVA_INT, index);
	}

	public void setInt(int index, int value) {
		setModified();
		memorySegment.setAtIndex(ValueLayout.JAVA_INT, index, value);
	}

	public void setModified() {
		modified = true;	
	}

	int getFormatType() {
		return 6;
	}
	protected long totalBytes() { 
		return size() * (long) Integer.BYTES; 
	}
	
	public boolean isImmutable() {
		return false;
	}   
	
	public IntTensor fillInPlace(int thisOffset, int size, int value) {
		setModified();
		for(int index = thisOffset; index < thisOffset+size; index++)
			memorySegment.setAtIndex(ValueLayout.JAVA_INT, index, value);
		return this;
	}

	public Arena getArena() {
		return Llama3.sharedArena;
	}

	public MemorySegment getSegment() {
		return memorySegment;
	}
	
	public IntVector getIntVector(VectorSpecies<Integer> species, int index) {
		if (!FloatTensor.USE_VECTOR_API) {
			throw new UnsupportedOperationException();
		}
		return IntVector.fromMemorySegment(species, memorySegment, (long) index * Integer.BYTES, ByteOrder.nativeOrder());
	}
	
	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		out.writeInt(size());
		for(int i = 0; i < size(); i++)
			out.write(getInt(i));
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
		int vsize = in.readInt();
		memorySegment = getArena().allocate(ValueLayout.JAVA_INT, vsize);
		for(int i = 0; i < vsize; i++)
			setInt(i, in.readInt());
	}

	@Override
	public int compareTo(Object o) {
		return Arrays.compare(memorySegment.toArray(ValueLayout.JAVA_INT),((IntTensor)o).getSegment().toArray(ValueLayout.JAVA_INT));
	}

	@Override
	public String toString() {
		return getSegment().toString();//Arrays.toString(getSegment().toArray(ValueLayout.JAVA_INT));
	}

}