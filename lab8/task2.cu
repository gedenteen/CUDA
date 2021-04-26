#include <malloc.h>
#include <string.h>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <iostream>
using namespace std;
#include <cufft.h>

#define BATCH 1 //размер данных для обработки

#define CUDA_CHECK_RETURN(value) {\
	cudaError_t _m_cudaStat = value;\
	if (_m_cudaStat != cudaSuccess) {\
		fprintf(stderr, "Error \"%s\" at line %d in file %s\n",\
			cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);\
		exit(1);\
	}\
} //макрос для обработки ошибок 

#define CUFFT_CHECK_RETURN(value) {\
	cufftResult stat = value;\
	if (stat != CUFFT_SUCCESS) {\
		fprintf(stderr, "Error at line %d in file %s\n",\
			__LINE__, __FILE__);\
		exit(1);\
	}\
} //макрос для обработки ошибок 

int main(void) {
	string line, buffer;
	cufftHandle plan; //дескриптор плана конфигурации (нужен для оптимизации под выбранную аппаратуру)
	cufftComplex *hos_data, *dev_data; //массивы для комплексных чисел
	vector<string> file_string; //одна строка в файле = массив из 4 чисел в виде string 
	vector<float> Wolf_nums; //числа Вольфа (весь диск)
	vector<float> freq; //массив частот после преобразования Фурье (на каждый день одно число)
	vector<float> power; //массив значений ^ 2 из преобразования Фурье
	ifstream in;
	
	for (int i = 1938; i <= 1991; i++) { //цикл по годам
		//открыть файл i-го года
		string path = string("data/w") + to_string(i) + string(".dat"); 
		in.open(path);
		if (!in.is_open()) {
			fprintf(stderr, "can't open a file data/w%d.dat\n", i);
			return -1;
		}
		
		while(getline(in, line)) { //считать строку из файла
			buffer = ""; 
			line += " "; //добавить пробел для обработки последнего числа
			for (int k = 0; k < line.size(); k++) {
				if (line[k] != ' ') {
					buffer += line[k]; //записать число посимвольно в buffer 
				}
				else { //если пробел, то есть получено число в виде строки
					if (buffer != "")
						file_string.push_back(buffer);
						buffer = "";
				}
			}
			
			if (file_string.size() != 0) {
				if (file_string[2] == "999") { //если число Вульфа неизвестно,
					file_string[2] = to_string(Wolf_nums.back()); //то взять предыдущее значение
				}
				Wolf_nums.push_back(stoi(file_string[2])); //преобразовать строку в число, добавить в массив чисел
				file_string.clear(); //очистить строку
			}
		} //end of while(getline(in, line))

		in.close();
	} //end of for(int i = 1938; i <= 1991; i++)
	
	int N = Wolf_nums.size();
	
	cudaMalloc((void**)&dev_data, sizeof(cufftComplex) * N * BATCH);
	hos_data = new cufftComplex[N * BATCH];
	for (int i = 0; i < N * BATCH; i++) {
		hos_data[i].x = Wolf_nums[i]; //действительная часть
		hos_data[i].y = 0.0f; //мнимая часть
	}
	cudaMemcpy(dev_data, hos_data, sizeof(cufftComplex) * N * BATCH, cudaMemcpyHostToDevice);
	
	//создание плана, преобразование Фурье происходит из комплексных чисел в комплексные:
	CUFFT_CHECK_RETURN(cufftPlan1d(&plan, N * BATCH, CUFFT_C2C, BATCH));
	//совершить быстрое преобразование Фурье (FFT):
	CUFFT_CHECK_RETURN(cufftExecC2C(plan, dev_data, dev_data, CUFFT_FORWARD));
	//синхронизация:
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	//копируем обратно на хост то, что получено после FFT:
	CUDA_CHECK_RETURN(cudaMemcpy(hos_data, dev_data, N * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
	
	power.resize(N / 2 + 1);
	for (int i = 1; i <= N / 2; i++) {
		//преобразовать значения, т.к. комплексные числа тяжело интерпретировать:
		power[i] = sqrt(hos_data[i].x * hos_data[i].x + hos_data[i].y * hos_data[i].y);
	}
	
	float max_freq = 0.5; //максимальная частота
	freq.resize(N / 2 + 1);
	for (int i = 1; i <= N / 2; i++) {
		//получаем равномерно распределенную сетку частот:
		freq[i] = 1 / (float(i) / float(N/2) * max_freq); 
	}
	
	int maxind = 1; //найти максимальное значение 
	for (int i = 1 ; i <= N / 2; i++) {
		if (power[i] > power[maxind])
			maxind = i;
	}
	
	//freq[maxind] - это количество дней при максимальной частоте?
	printf("calculated periodicity = %f years\n", freq[maxind] / 365);
	
	cufftDestroy(plan);
	cudaFree(dev_data);
	free(hos_data);
	return 0;
}
