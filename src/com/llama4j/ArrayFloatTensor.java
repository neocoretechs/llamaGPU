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

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
/**
 * ArrayFloatTensor maintaining array of floats backed by a MemorySegment.
 * Controlled by FloatTensor.USE_CUDA flag.
 */
final class ArrayFloatTensor extends FloatTensor implements Externalizable, Comparable {
	public static boolean DEBUG = false;
	MemorySegment memorySegment;

	public ArrayFloatTensor() {
	}

	ArrayFloatTensor(float[] values) {
		// allocate off-heap space for headSize floats
		memorySegment = getArena().allocate(ValueLayout.JAVA_FLOAT, values.length);
		// bulk copy from heap arrays if you have them
		MemorySegment.copy(values, 0, memorySegment, ValueLayout.JAVA_FLOAT, 0, values.length);
	}

	public static FloatTensor allocate(int... dims) {
		int numberOfElements = FloatTensor.numberOfElements(dims);
		return new ArrayFloatTensor(new float[numberOfElements]);
	}

	@Override
	public int size() {
		return (int) (memorySegment.byteSize() / Float.BYTES);
	}

	@Override
	public float getFloat(int index) {
		return memorySegment.getAtIndex(ValueLayout.JAVA_FLOAT, index);
	}

	@Override
	public void setFloat(int index, float value) {
		setModified();
		memorySegment.setAtIndex(ValueLayout.JAVA_FLOAT, index, value);
	}

	@Override
	int getFormatType() {
		return 5;
	}
	protected long totalBytes() { 
		return size() * (long) Float.BYTES; 
	}
	@Override
	public boolean isImmutable() {
		return false;
	}   
	@Override
	public FloatTensor fillInPlace(int thisOffset, int size, float value) {
		setModified();
		for(int index = thisOffset; index < thisOffset+size; index++)
			memorySegment.setAtIndex(ValueLayout.JAVA_FLOAT, index, value);
		return this;
	}

	@Override
	public FloatVector getFloatVector(VectorSpecies<Float> species, int index) {
		if (!USE_VECTOR_API) {
			throw new UnsupportedOperationException();
		}
		return FloatVector.fromMemorySegment(species, memorySegment, (long) index * Float.BYTES, ByteOrder.nativeOrder());
	}
	@Override
	public Arena getArena() {
		return Llama3.sharedArena;
	}
	@Override
	public MemorySegment getSegment() {
		return memorySegment;
	}

	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		out.writeInt(size());
		for(int i = 0; i < size(); i++)
			out.writeFloat(getFloat(i));
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
		int vsize = in.readInt();
		// allocate off-heap space for headSize floats
		memorySegment = getArena().allocate(ValueLayout.JAVA_FLOAT, vsize);
		for(int i = 0; i < vsize; i++)
			setFloat(i, in.readFloat());
	}

	@Override
	public int compareTo(Object o) {
		return Arrays.compare(memorySegment.toArray(ValueLayout.JAVA_FLOAT),((ArrayFloatTensor)o).getSegment().toArray(ValueLayout.JAVA_FLOAT));
	}


	@Override
	public String toString() {
		return getSegment().toString();//Arrays.toString(getSegment().toArray(ValueLayout.JAVA_FLOAT));
	}


}