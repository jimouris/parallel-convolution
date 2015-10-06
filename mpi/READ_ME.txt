compile:
mpicc -o mpi_conv mpi_conv.c

run:
mpirun -np 4 ./mpi_conv waterfall_grey_1920_2520.raw 1920 2520 50 grey
