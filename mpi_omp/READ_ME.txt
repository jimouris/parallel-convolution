compile:
mpicc -openmp -o mpi_omp_conv mpi_omp_conv.c 

run:
mpirun -np 4 ./mpi_omp_conv waterfall_grey_1920_2520.raw 1920 2520 50 grey
