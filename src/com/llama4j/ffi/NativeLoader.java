package com.llama4j.ffi;

import java.io.File;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.atomic.AtomicReference;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.llama4j.Llama3;

public final class NativeLoader {
	public static boolean DEBUG = true;
	private static final Log log = LogFactory.getLog(NativeLoader.class);
    private static volatile boolean loaded = false;
    private NativeLoader() {}
    private enum LibraryState {
		NOT_LOADED,
		LOADING,
		LOADED
	}
	private static final AtomicReference<LibraryState> libraryLoaded = new AtomicReference<>(LibraryState.NOT_LOADED);

	static {
			NativeLoader.loadLibrary(new File(System.getProperty("java.library.path")).list());
	}

	public static void load() {
			NativeLoader.loadLibrary(new File(System.getProperty("java.library.path")).list());
	}
	
	/**
	 * Tries to load the necessary library files from the given list of
	 * directories.
	 *
	 * @param paths a list of strings where each describes a directory of a library.
	 */
	public static void loadLibrary(final String[] paths) {
		if (libraryLoaded.get() == LibraryState.LOADED) {
			return;
		}
		if(libraryLoaded.compareAndSet(LibraryState.NOT_LOADED,LibraryState.LOADING)) {
			synchronized (NativeLoader.class) {
				//.out.println("Loading from paths list of length:"+paths.size());
				for (final String path : paths) {
					//if(DEBUG) log.info(path);
					if(path.endsWith(".so") || path.endsWith(".dll")) {
						String fname = new File(path).getName();
						fname = fname.substring(0,fname.indexOf("."));
						if(DEBUG)
							log.info("Trying load for:"+fname);
						System.loadLibrary(fname);
					}
				}
			}
			libraryLoaded.set(LibraryState.LOADED);
		}
		while (libraryLoaded.get() == LibraryState.LOADING) {
			try {
				log.info("Waiting for load, retry..");
				Thread.sleep(10);
			} catch(final InterruptedException e) {}
		}
	}

	public static void loadMethods() {
		Linker linker = Linker.nativeLinker();
		//if(DEBUG) log.info("linker:"+linker);
		SymbolLookup lookup = SymbolLookup.loaderLookup();
		//if(DEBUG) log.info("Loader:"+lookup);
		//
		//float getFloat(const uint64_t q, int index, int blockSize, int typeSize, int headerBytes) {
		//    const float* d_q = reinterpret_cast<const float*>(q);
		Llama3.getFloat = linker.downcallHandle(
				lookup.find("getFloat").get(),
				FunctionDescriptor.of(
						ValueLayout.JAVA_FLOAT, // return float
						ValueLayout.JAVA_LONG,  // device A
						ValueLayout.JAVA_INT,   // index
						ValueLayout.JAVA_INT,	// format
						ValueLayout.JAVA_INT,	// blocksize
						ValueLayout.JAVA_INT,	// typesize
						ValueLayout.JAVA_INT	// headerBytes
						));
		if(DEBUG)
		log.info("getFloat:"+Llama3.getFloat);
		//
		Llama3.sdotSliceDeviceHandle = linker.downcallHandle(
				lookup.find("sdotSliceDevice").get(),
				FunctionDescriptor.of(
						//float sdotSliceDevice(const uint8_t* qA, int indexA, int formatA, int blockSizeA, int typeSizeA, int headerBytesA,
						//	    const uint8_t* qB, int indexB, int formatB, int blockSizeB, int typeSizeB, int headerBytesB,
						//	    int N)
						ValueLayout.JAVA_FLOAT,   // return float
						ValueLayout.JAVA_LONG,    // device A
						ValueLayout.JAVA_INT ,    // offset A
						ValueLayout.JAVA_INT,     // format A (1-5 = Q8,Q4,F16,BF16,F32)
						ValueLayout.JAVA_INT,     // blockSize A for format
						ValueLayout.JAVA_INT,     // typeSize A for format
						ValueLayout.JAVA_INT,     // headerBytes A for format
						ValueLayout.JAVA_LONG,    // device B
						ValueLayout.JAVA_INT,     // offset B
						ValueLayout.JAVA_INT,     // format B
						ValueLayout.JAVA_INT,     // blocksize B
						ValueLayout.JAVA_INT,     // typeSize B
						ValueLayout.JAVA_INT,     // headerBytes B
						ValueLayout.JAVA_INT	  // Number of elements in tensor
						));
		if(DEBUG) log.info("sdotSliceDevice:"+Llama3.sdotSliceDeviceHandle);
		//if(DEBUG) log.info("cublasHandle:"+Llama3.cublasGetHandle);
		//Llama3.cublasFreeHandle = linker.downcallHandle(
		//		lookup.find("cublasHandleDestroy").get(),
		//		FunctionDescriptor.ofVoid(
		//				// return void
		//				ValueLayout.JAVA_LONG  // pass long handle
		//				));
		//if(DEBUG) log.info("cublasHandleDestroy:"+Llama3.cublasFreeHandle);
		Llama3.cudaInit = linker.downcallHandle(
				lookup.find("cudaInit").get(),
				FunctionDescriptor.ofVoid());
		Llama3.cudaGetMemInfo = linker.downcallHandle(
				lookup.find("cudaGetMemInfo").get(),
				FunctionDescriptor.ofVoid(
						ValueLayout.ADDRESS,    // size_t* free, writes to memorysegments
						ValueLayout.ADDRESS     // size_t* total
						));
		if(DEBUG) log.info("cudaGetMemInfo:"+Llama3.cudaGetMemInfo);
		//launch_rmsnorm_fp32_rowmajor(uint8_t* qA, int indexA, int formatA, int blockSizeA, int typeSizeA, int headerBytesA,
	    //uint8_t* qB, int indexB, int formatB, int blockSizeB, int typeSizeB, int headerBytesB,
	    //float* out, int size, float eps) {
		Llama3.launchRmsnorm = linker.downcallHandle(
				lookup.find("launch_rmsnorm_fp32_rowmajor").get(),
				FunctionDescriptor.ofVoid(
						ValueLayout.JAVA_LONG, // deviceptr x
						ValueLayout.JAVA_INT, // offset into x
						ValueLayout.JAVA_INT, // format x
						ValueLayout.JAVA_INT, // blocksize x
						ValueLayout.JAVA_INT, // typesiez x
						ValueLayout.JAVA_INT, // headerbytes s
						ValueLayout.JAVA_LONG, // deviceptr weights
						ValueLayout.JAVA_INT, // offset into weights
						ValueLayout.JAVA_INT, // format weights
						ValueLayout.JAVA_INT, // blocksize weights
						ValueLayout.JAVA_INT, // typesize weights
						ValueLayout.JAVA_INT, // headerbytes weights
						ValueLayout.JAVA_LONG, // deviceptr out
						ValueLayout.JAVA_INT,   // size
						ValueLayout.JAVA_FLOAT  // eps
						)
				);
		if(DEBUG) log.info("launch_rmsnorm_fp32_rowmajor:"+Llama3.launchRmsnorm);
		Llama3.launchSoftmaxInplace = linker.downcallHandle(
				lookup.find("launch_row_softmax_inplace_fp32").get(),
				FunctionDescriptor.ofVoid(
						ValueLayout.JAVA_LONG, // S device address
						ValueLayout.JAVA_INT, // offset
						ValueLayout.JAVA_INT  // size
						)
				);
		if(DEBUG) log.info("launch_row_softmax_inplace_fp32:"+Llama3.launchSoftmaxInplace);
		//void launch_weighted_sum(uint8_t* Att, uint8_t* xb, uint8_t* vCache, int h, int headSize, 
		// int attOffset, int xbOffset, int vcOffset, int kvDim, int kvMul, int position, int token, int size) 
		Llama3.launchAV = linker.downcallHandle(
			    lookup.find("launch_weighted_sum").get(),
			    FunctionDescriptor.ofVoid(
			        ValueLayout.JAVA_LONG, // Att
			        ValueLayout.JAVA_LONG, // xb
			        ValueLayout.JAVA_LONG, // vCache
			        //int h, int headSize, int attOffset, int xbOffset,  
			        ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
			        ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, //int kvDim, int kvMul
			        ValueLayout.JAVA_INT, ValueLayout.JAVA_INT //int position, int token
			    )
			);
		if(DEBUG) log.info("launch_weighted_sum:"+Llama3.launchAV);
		// void launchMatmul(const uint8_t* qA, int indexA, int formatA, int blockSizeA, int typeSizeA, int headerBytesA,
		//	    const uint8_t* qB, int indexB, int formatB, int blockSizeB, int typeSizeB, int headerBytesB,
		//	    uint8_t* out, int dim0, int dim1) {
		Llama3.launchMatmul = linker.downcallHandle(
			    lookup.find("launch_Matmul").get(),
			    FunctionDescriptor.ofVoid(
			        ValueLayout.JAVA_LONG, // qA
			        //int indexA, formatA, blockSizeA, typeSizeA, headerBytesA
			        ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
			        ValueLayout.JAVA_LONG, // qB
			        //int indexA, formatA, blockSizeA, typeSizeA, headerBytesA
			        ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
			        ValueLayout.JAVA_LONG, // out
			        ValueLayout.JAVA_INT, // dim0
			        ValueLayout.JAVA_INT // dim1
			    )
			);
		if(DEBUG) log.info("launch_Matmul:"+Llama3.launchMatmul);
		//float launch_cpu_scalar_Dot(const uint8_t* d_q, int indexA, int formatA, int blockSizeA, int typeSizeA, int headerBytesA,
		//	    const uint8_t* d_k, int indexB, int formatB, int blockSizeB, int typeSizeB, int headerBytesB, int size)
		Llama3.sdotSimple = linker.downcallHandle(
				lookup.find("launch_cpu_scalar_Dot").get(),
				FunctionDescriptor.of(
						ValueLayout.JAVA_FLOAT,   // return float
						ValueLayout.JAVA_LONG,    // device A
						ValueLayout.JAVA_INT ,    // offset A
						ValueLayout.JAVA_INT,     // format A (1-5 = Q8,Q4,F16,BF16,F32)
						ValueLayout.JAVA_INT,     // blockSize A for format
						ValueLayout.JAVA_INT,     // typeSize A for format
						ValueLayout.JAVA_INT,     // headerBytes A for format
						ValueLayout.JAVA_LONG,    // device B
						ValueLayout.JAVA_INT,     // offset B
						ValueLayout.JAVA_INT,     // format B
						ValueLayout.JAVA_INT,     // blocksize B
						ValueLayout.JAVA_INT,     // typeSize B
						ValueLayout.JAVA_INT,     // headerBytes B
						ValueLayout.JAVA_INT	  // Number of elements in tensor
						));
		if(DEBUG) log.info("launch_cpu_scalar_Dot:"+Llama3.sdotSimple);
		//void launch_qkscores(uint8_t* q, int qOffset, int formatA, int blockSizeA, int typeSizeA, int headerBlockA,
	    //uint8_t* keyCache, int keyCacheOffset, int formatB, int blockSizeB, int typeSizeB, int headerBlockB, 
	    //uint8_t* Att, int attOffset, 
	    //int position, int token, int h, int headSize, int kvDim, int kvMul )
		Llama3.launchQK = linker.downcallHandle(
				lookup.find("launch_qkscores").get(),
				FunctionDescriptor.ofVoid(
						ValueLayout.JAVA_LONG,    // q
						ValueLayout.JAVA_INT ,    // offset q
						ValueLayout.JAVA_INT,     // format q (1-5 = Q8,Q4,F16,BF16,F32)
						ValueLayout.JAVA_INT,     // blockSize q for format
						ValueLayout.JAVA_INT,     // typeSize q for format
						ValueLayout.JAVA_INT,     // headerBytes q for format
						ValueLayout.JAVA_LONG,    // keyCache (keyCacheOffset computed in device kernel)
						ValueLayout.JAVA_INT,     // format 
						ValueLayout.JAVA_INT,     // blocksize 
						ValueLayout.JAVA_INT,     // typeSize 
						ValueLayout.JAVA_INT,     // headerBytes keyCache
						ValueLayout.JAVA_LONG,    // Att
						ValueLayout.JAVA_INT,     // attOffset 
						ValueLayout.JAVA_INT,	  // position
						ValueLayout.JAVA_INT,     // token 
						ValueLayout.JAVA_INT,     // h
						ValueLayout.JAVA_INT,     // headSize
						ValueLayout.JAVA_INT,     // numHeads
						ValueLayout.JAVA_INT,     // contextLength
						ValueLayout.JAVA_INT,     // kvDim
						ValueLayout.JAVA_INT	  // kvMul
						));
		if(DEBUG) log.info("launch_qkscores:"+Llama3.launchQK);
	    //void launch_rope(const uint8_t* d_real, int indexA, int formatA, int blockSizeA, int typeSizeA, int headerBytesA,
	    //const uint8_t* d_imag, int indexB, int formatB, int blockSizeB, int typeSizeB, int headerBytesB,
	    //uint8_t* d_q, uint8_t* d_k, // state.q , state.k
	    //int nTokens, int dim, int position, int headSize, int kvDim)
		Llama3.launchRope = linker.downcallHandle(
				lookup.find("launch_rope").get(),
				FunctionDescriptor.ofVoid(
						ValueLayout.JAVA_LONG,    // d_real weight
						ValueLayout.JAVA_INT,    // index d_real
						ValueLayout.JAVA_INT,     // format q (1-5 = Q8,Q4,F16,BF16,F32)
						ValueLayout.JAVA_INT,     // blockSize q for format
						ValueLayout.JAVA_INT,     // typeSize q for format
						ValueLayout.JAVA_INT,     // headerBytes q for format
						ValueLayout.JAVA_LONG,    // d_imag weight
						ValueLayout.JAVA_INT,     // index d_imag
						ValueLayout.JAVA_INT,     // format 
						ValueLayout.JAVA_INT,     // blocksize 
						ValueLayout.JAVA_INT,     // typeSize 
						ValueLayout.JAVA_INT,     // headerBytes
						ValueLayout.JAVA_LONG,    // d_q DeviceTensor state.q
						ValueLayout.JAVA_LONG,    // d_k DeviceTensor state.k
						ValueLayout.JAVA_INT,     // nTokens
						ValueLayout.JAVA_INT,	  // dim
						ValueLayout.JAVA_INT,     // position
						ValueLayout.JAVA_INT,     // headSize
						ValueLayout.JAVA_INT     // kvDim
						));
		if(DEBUG) log.info("launch_rope:"+Llama3.launchRope);
	    Llama3.allocDevicePtr = linker.downcallHandle(
		        lookup.find("allocDevicePtr").get(),
		        FunctionDescriptor.of(ValueLayout.JAVA_LONG, // uint64_t device ptr
		        					ValueLayout.JAVA_LONG) // size 
		    );
		if(DEBUG) log.info("allocDevicePtr:"+Llama3.allocDevicePtr);
	    Llama3.freeDevicePtr = linker.downcallHandle(
		        lookup.find("freeDevicePtr").get(),
		        FunctionDescriptor.ofVoid(ValueLayout.JAVA_LONG) // uint64_t device ptr
		    );
		if(DEBUG) log.info("freeDevicePtr:"+Llama3.freeDevicePtr);
		 // copyHostToDevice
	    Llama3.copyHostToDeviceMH = linker.downcallHandle(
	        lookup.find("copyHostToDevice").get(),
	        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS,   // uint8_t* tensor
	                              ValueLayout.JAVA_LONG, // uint64_t device ptr
	                              ValueLayout.JAVA_LONG) // int bytes
	    );
		if(DEBUG) log.info("copyHostToDevice:"+Llama3.copyHostToDeviceMH);
	    // copyDeviceToHost
	    // copyDeviceToHost
	    Llama3.copyDeviceToHostMH = linker.downcallHandle(
	        lookup.find("copyDeviceToHost").get(),
	        FunctionDescriptor.ofVoid(ValueLayout.JAVA_LONG, // uint64_t device pt
	                                  ValueLayout.JAVA_LONG,   // uint8_t* tensor
	                                  ValueLayout.JAVA_LONG) // size_t bytes
	    );
		if(DEBUG) log.info("copyDeviceToHost:"+Llama3.copyDeviceToHostMH);

	    Llama3.copyFromNativeMH = linker.downcallHandle(
	        lookup.find("copyFromNative").get(),
	        FunctionDescriptor.ofVoid(ValueLayout.JAVA_LONG,   // uint8_t* tensor, or uint8_t** arraytensor
	                                  ValueLayout.JAVA_LONG) // size_t bytes
	    );
		if(DEBUG) log.info("copyFromNative:"+Llama3.copyFromNativeMH);
		Llama3.loadModelMH = linker.downcallHandle(
			        lookup.find("load_model").get(),
			        FunctionDescriptor.ofVoid(ValueLayout.JAVA_LONG, // uint8_t* tensor model path
			        							ValueLayout.JAVA_INT)  // context size
			);
		if(DEBUG) log.info("load_model:"+Llama3.loadModelMH);
	    Llama3.runModelMH = linker.downcallHandle(
		        lookup.find("run_model").get(),
		        FunctionDescriptor.of(ValueLayout.JAVA_INT,
		        						ValueLayout.JAVA_LONG, // prompt StringTensor
		        						ValueLayout.JAVA_FLOAT, // temp
		        						ValueLayout.JAVA_FLOAT, // mip_p
		        						ValueLayout.JAVA_FLOAT, // top_p
		        						ValueLayout.JAVA_LONG // IntTensor return tokens
		        						) // StringTensor return dialog uint8_t* tensor, or uint8_t** ArrayTensor
		    );
		if(DEBUG) log.info("run_model:"+Llama3.runModelMH);
	    Llama3.runModelTokenizeMH = linker.downcallHandle(
		        lookup.find("run_model_tokenize").get(),
		        FunctionDescriptor.of(ValueLayout.JAVA_INT,
		        						ValueLayout.JAVA_LONG, // prompt StringTensor
		        						ValueLayout.JAVA_FLOAT, // temp
		        						ValueLayout.JAVA_FLOAT, // mip_p
		        						ValueLayout.JAVA_FLOAT, // top_p
		        						ValueLayout.JAVA_LONG // IntTensor return tokens
		        						) // StringTensor return dialog uint8_t* tensor, or uint8_t** ArrayTensor
		    );
		if(DEBUG) log.info("run_model_tokenize:"+Llama3.runModelTokenizeMH);
		Llama3.stringToTokenMH = linker.downcallHandle(
			    lookup.find("string_to_token").get(),
			    FunctionDescriptor.of(ValueLayout.JAVA_INT,
			        					ValueLayout.JAVA_LONG, // prompt StringTensor
			        					ValueLayout.JAVA_LONG // IntTensor return tokens
			        					) // StringTensor return dialog uint8_t* tensor, or uint8_t** ArrayTensor
			    );
		if(DEBUG) log.info("string_to_token:"+Llama3.stringToTokenMH);
		Llama3.tokenToStringMH = linker.downcallHandle(
				lookup.find("token_to_string").get(),
				FunctionDescriptor.of(ValueLayout.JAVA_INT,
				        					ValueLayout.JAVA_LONG, // IntTensor of tokens
				        					ValueLayout.JAVA_INT, // size
				        					ValueLayout.JAVA_LONG // StringTensor return string
				        					)
				);
		if(DEBUG) log.info("token_to_string:"+Llama3.tokenToStringMH);
	}
}
