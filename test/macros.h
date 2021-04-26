#ifndef MACROS
#define MACROS

//макросы для обработки ошибок 

#define CUDA_CHECK_RETURN(value) {\
	cudaError_t _m_cudaStat = value;\
	if (_m_cudaStat != cudaSuccess) {\
		fprintf(stderr, "Error \"%s\" at line %d in file %s\n",\
			cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);\
		exit(1);\
	}\
}

#define CUBLAS_CHECK_RETURN(value) {\
	cublasStatus_t stat = value;\
	if (stat != CUBLAS_STATUS_SUCCESS) {\
		fprintf(stderr, "Error at line %d in file %s\n",\
			__LINE__, __FILE__);\
		exit(1);\
	}\
} 

#define CUFFT_CHECK_RETURN(value) {\
	cufftResult stat = value;\
	if (stat != CUFFT_SUCCESS) {\
		fprintf(stderr, "Error at line %d in file %s\n",\
			__LINE__, __FILE__);\
		exit(1);\
	}\
}

#endif
