#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <assert.h>
#include "funcs.h"

void Usage(int argc, char **argv, char **image, int *width, int *height, int *loops, color_t *imageType) {
	if (argc == 6 && !strcmp(argv[5], "grey")) {
		*image = (char *)malloc((strlen(argv[1])+1) * sizeof(char));
		strcpy(*image, argv[1]);	
		*width = atoi(argv[2]);
		*height = atoi(argv[3]);
		*loops = atoi(argv[4]);
		*imageType = GREY;
	} else if (argc == 6 && !strcmp(argv[5], "rgb")) {
		*image = (char *)malloc((strlen(argv[1])+1) * sizeof(char));
		strcpy(*image, argv[1]);	
		*width = atoi(argv[2]);
		*height = atoi(argv[3]);
		*loops = atoi(argv[4]);
		*imageType = RGB;
	} else {
		fprintf(stderr, "Error Input!\n%s image_name width height loops [rgb/grey].\n", argv[0]);
		exit(EXIT_FAILURE);
	}
}

int write_all(int fd , uint8_t* buff , int size) {
	int n, sent;
	for (sent = 0 ; sent < size ; sent += n)
		if ((n = write(fd, buff + sent, size - sent)) == -1)
			return -1;
	return sent;
}

int read_all(int fd , uint8_t* buff , int size) {
	int n, sent;
	for (sent = 0 ; sent < size ; sent += n)
		if ((n = read(fd, buff + sent, size - sent)) == -1)
			return -1;
	return sent;
}

uint64_t micro_time(void) {
	struct timeval tv;
	assert(gettimeofday(&tv, NULL) == 0);
	return tv.tv_sec * 1000 * 1000 + tv.tv_usec;
}

