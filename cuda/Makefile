cuda_conv: main.o funcs.o cuda_convolute.o
	nvcc -o cuda_conv main.o funcs.o cuda_convolute.o -O2

main.o: main.c 
	gcc -c main.c -O2

funcs.o: funcs.c
	gcc -c funcs.c -O2

cuda_convolute.o: cuda_convolute.cu
	nvcc -c cuda_convolute.cu

clean:
	\rm -f *.o cuda_conv
