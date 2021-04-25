#include <malloc.h>
#include <string.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <cctype>
#include <list>
#include <stdlib.h>
#include <ctime>
#include <fstream> //для открытия файлов
#include <iostream> //для вывода информации на экран
using namespace std;

#include <cufft.h>
#include "cublas_v2.h"

#define BATCH 1
#define SZ (1<<25)
#define ALPHA 3.0f

#define CUDA_CHECK_RETURN(value) {\
	cudaError_t _m_cudaStat = value;\
	if (_m_cudaStat != cudaSuccess) {\
		fprintf(stderr, "Error \"%s\" at line %d in file %s\n",\
			cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);\
		exit(1);\
	}\
} //макрос для обработки ошибок 

#define CUDA_CHECK_RETURN_CUBLAS(value) {\
	cublasStatus_t stat = value;\
	if (stat != CUBLAS_STATUS_SUCCESS) {\
		fprintf(stderr, "Error at line %d in file %s\n",\
			__LINE__, __FILE__);\
		exit(1);\
	}\
} //макрос для обработки ошибок 

int main(void) {
	string line, buffer;
	cufftHandle plan;
	cufftComplex *cpu_data, *gpu_data;
	vector<string> numline;
	vector<float> DATA; 
	vector<float> freq; 
	vector<float> power;
	ifstream in;
	cufftComplex *data_h;
	int j = 0;
	
	for (int i = 1938; i <= 1991; i++) {
		string path = string("data/w") + to_string(i) + string(".dat");
		in.open(path);
		if (!in.is_open()) {
			fprintf(stderr, "can't open a file data/w%d.dat\n", i);
			return -1;
		}
		
		while(getline(in, line)) {
			buffer = "";
			line += " ";
			for (int k = 0; k < line.size(); k++) {
				if (line[k] != ' ') {
					buffer += line[k];
				}
				else {
					if (buffer != "")
						numline.push_back(buffer);
						buffer = "";
				}
			}
			
			if (numline.size() != 0) {
				if (numline[2] == "999") {
					numline[2] = to_string(DATA.back());
				}
				DATA.push_back(stoi(numline[2]));
				numline.clear();
			}
			j++;
		} //end of while(getline(in, line))

		in.close();
	} //end of for(int i = 1938; i <= 1991; i++)
	
	int N = DATA.size();
	
	cudaMalloc((void**)&gpu_data, sizeof(cufftComplex) * N * BATCH);
	data_h = (cufftComplex*) calloc(N, sizeof(cufftComplex));
	cpu_data = new cufftComplex[N * BATCH];
	for (int i = 0; i < N * BATCH; i++) {
		cpu_data[i].x = DATA[i];
		cpu_data[i].y = 0.0f;
	}
	cudaMemcpy(gpu_data, cpu_data, sizeof(cufftComplex) * N * BATCH, cudaMemcpyHostToDevice);
	
	if (cufftPlan1d(&plan, N * BATCH, CUFFT_C2C, BATCH) != CUFFT_SUCCESS) {
		cerr << "error in cufftPlan1d()\n";
		return -1;
	}
	if (cufftExecC2C(plan, gpu_data, gpu_data, CUFFT_FORWARD) != CUFFT_SUCCESS) {
		cerr << "error in cufftPlan1d()\n";
		return -1;
	}
	if (cudaDeviceSynchronize() != CUFFT_SUCCESS) {
		cerr << "error in cufftPlan1d()\n";
		return -1;
	}
	cudaMemcpy(data_h, gpu_data, N * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
	
	power.resize(N / 2 + 1);
	for (int i = 1; i <= N / 2; i++) {
		power[i] = sqrt(data_h[i].x * data_h[i].x + data_h[i].y * data_h[i].y);
	}
	
	float max_freq = 0.5;
	freq.resize(N / 2 + 1);
	for (int i = 1; i <= N / 2; i++) {
		freq[i] = 1 / (float(i) / float(N/2) * max_freq);
	}
	
	int maxind = 1;
	for (int i = 1 ; i <= N / 2; i++) {
		if (power[i] > power[maxind])
			maxind = i;
	}
	
	printf("calculated periodicity = %f years\n", freq[maxind] / 365);
	
	cufftDestroy(plan);
	cudaFree(gpu_data);
	free(data_h);
	free(cpu_data);
	
	return 0;
}
