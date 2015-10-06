#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include "cuda_convolute.h"
#include "funcs.h"

int main(int argc, char** argv) {
	int fd, width, height, loops;
	char *image;
	color_t imageType;
	
	Usage(argc, argv, &image, &width, &height, &loops, &imageType);	

	/* Host vectors */
	uint8_t *src = NULL;
	/* Count time */ 
	uint64_t c = micro_time(); 

	/* Read bytes from picture */
	if ((fd = open(image, O_RDONLY)) < 0) {
		fprintf(stderr, "cannot open %s\n", argv[1]);
		return EXIT_FAILURE;
	}
	size_t bytes = (imageType == GREY) ? height * width : height * width*3;	
	src = (uint8_t *) calloc(bytes, sizeof(uint8_t));
	read_all(fd, src, bytes);
	close(fd);

	gpuConvolute(src, width, height, loops, imageType);

	/* Create new picture - Write bytes */
	int fd_out;
	char *outImage = (char*) malloc((strlen(image) + 9) * sizeof(char));
	strcpy(outImage, "blur_");
	strcat(outImage, image);
	if ((fd_out = open(outImage, O_CREAT | O_WRONLY, 0644)) == -1) {
		fprintf(stderr, "cannot open-create %s\n", outImage);
		return EXIT_FAILURE;
	}
	write_all(fd_out, src, bytes);
	close(fd_out);
	free(outImage);

	/* compute time */
	c = micro_time() - c;
	double million = 1000 * 1000;
	fprintf(stdout, "Execution time: %.3f sec\n", c / million);

    /* De-allocate space */
    free(src);
	return EXIT_SUCCESS;
}
