gpu: tsp_gpu

tsp_gpu: tsp_2opt.o tsp_preprocessing.o
	g++ -g --std=c++17 -o tsp_gpu -L/opt/cuda/lib64 -lcuda -lcudart src/tsp_gpu.cpp tsp_2opt.o tsp_preprocessing.o -fopenmp -lpthread -O3 -Wextra -Wall

tsp: tsp_2opt.o tsp_preprocessing.o
	g++ -g --std=c++17 -o tsp -L/opt/cuda/lib64 -lcuda -lcudart src/tsp.cpp tsp_2opt.o tsp_preprocessing.o -fopenmp -lpthread -O3 -Wextra -Wall

tsp_2opt.o:
	nvcc -std=c++17 -c -arch=sm_61 src/tsp_2opt.cu --use_fast_math

tsp_preprocessing.o:
	nvcc -std=c++17 -c -arch=sm_61 src/tsp_preprocessing.cu --use_fast_math

clean: 
	rm -f *.o tsp tsp_gpu