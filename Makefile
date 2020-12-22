
solver: cuda_2opt.o cuda_preprocessing.o 
	g++ --std=c++17 -o solver -L/opt/cuda/lib64 -lcuda -lcudart src/solver.cpp src/2opt.cpp src/preprocessing.cpp src/util.cpp cuda_2opt.o cuda_preprocessing.o  -fopenmp -lpthread -O3 -Wextra -Wall

cuda_2opt.o:
	nvcc -std=c++17 -c -arch=sm_61 src/cuda_2opt.cu --use_fast_math -O3

cuda_preprocessing.o:
	nvcc -std=c++17 -c -arch=sm_61 src/cuda_preprocessing.cu --use_fast_math -O3

clean: 
	rm -f *.o tsp tsp_gpu solver