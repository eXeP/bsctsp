gpu: tsp_gpu

tsp_gpu: tsp_2opt.o
	g++ -g --std=c++17 -o tsp_gpu -L/opt/cuda/lib64 -lcuda -lcudart tsp_gpu.cpp tsp_2opt.o -fopenmp -lpthread -O3 -Wextra -Wall

tsp: tsp_2opt.o
	g++ -g --std=c++17 -o tsp -L/opt/cuda/lib64 -lcuda -lcudart tsp.cpp -fopenmp -lpthread -O3 -Wextra -Wall

tsp_2opt.o:
	nvcc -std=c++11 -g -c -arch=sm_61 tsp_2opt.cu 

clean: rm -f *.o tsp