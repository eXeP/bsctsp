gpu: tsp_gpu

solver: cuda_2opt.o tsp_preprocessing.o preprocessing
	g++ -g --std=c++17 -o solver -L/opt/cuda/lib64 -lcuda -lcudart src/solver.cpp cuda_2opt.o tsp_preprocessing.o -fopenmp -lpthread -O3 -Wextra -Wall

tsp_gpu: cuda_2opt.o cuda_preprocessing.o
	g++ -g --std=c++17 -o tsp_gpu -L/opt/cuda/lib64 -lcuda -lcudart src/tsp_gpu.cpp cuda_2opt.o cuda_preprocessing.o -fopenmp -lpthread -O3 -Wextra -Wall

preprocessing: cuda_2opt.o cuda_preprocessing.o
	g++ -g --std=c++17 -o tsp -L/opt/cuda/lib64 -lcuda -lcudart src/preprocessing.cpp cuda_2opt.o cuda_preprocessing.o -fopenmp -lpthread -O3 -Wextra -Wall

cuda_2opt.o:
	nvcc -std=c++17 -c -arch=sm_61 src/cuda_2opt.cu --use_fast_math

cuda_preprocessing.o:
	nvcc -std=c++17 -c -arch=sm_61 src/cuda_preprocessing.cu --use_fast_math

clean: 
	rm -f *.o tsp tsp_gpu