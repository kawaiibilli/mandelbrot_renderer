CC = g++
NVCC = nvcc

cpu_render: cpu_main.cpp
	$(CXX) -g cpu_main.cpp -lmpfr -fopenmp -o cpu_render `pkg-config --cflags --libs opencv`
gpu_render: kernel.o gpu_main.o
	$(CC) kernel.o gpu_main.o  -o gpu_render -L/usr/local/cuda/lib64 -lcudart `pkg-config --cflags --libs opencv`
gpu_main.o: gpu_main.cpp
	$(CC) -c -I. gpu_main.cpp  -o gpu_main.o -L/usr/local/cuda/lib64 -lcudart `pkg-config --cflags --libs opencv`
kernel.o: kernel.cu kernel.h
	$(NVCC) -c -I. -I/usr/local/cuda/include kernel.cu -o kernel.o
gpu_run: gpu_render
	./gpu_render
cpu_run: cpu_render
	./cpu_render
clean:
	rm -f gpu_render cpu_render kernel.o gpu_main.o
