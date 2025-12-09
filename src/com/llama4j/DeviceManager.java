package com.llama4j;

import java.lang.foreign.MemorySegment;

public final class DeviceManager {
	private static boolean DEBUG = false;

    static void loadModel(StringTensor model, int contextSize) {
		MemorySegment hostSeg = model.getSegment();
		long addr = hostSeg.address();
		try {
			Llama3.loadModelMH.invokeExact(addr, contextSize);
		} catch (Throwable e) {
			throw new RuntimeException(e);
		}
}
    static int runModel(StringTensor prompt, float temp, IntTensor returnTokens) {
    		MemorySegment hostSeg = prompt.getSegment();
    		long addr = hostSeg.address();
    		MemorySegment tokSegment = returnTokens.getSegment();
    		long addr2 = tokSegment.address();
    		try {
    			return (int) Llama3.runModelMH.invokeExact(addr, temp, addr2);
    		} catch (Throwable e) {
    			throw new RuntimeException(e);
    		}
    }
  
}
