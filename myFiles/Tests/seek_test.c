// hitting problem where the fread function in eccodes does not return the same thing as the 
// function in my personal eccodes
// ftello(f) is an error that is hit first. returns 1 when it should return 4

/*
PERSONAL LIB
- offset = -3
- n = 3
- buf or tmp = "GRIBRIB"

ECCODES
- stdio_read gets hit 4 times before the read_grib function is hit
- offset = 0
- n = 3
- buf or tmp = "GRIB"

*/

// seems like the solution is to just call stdio_read a few more times

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
int main(){
    FILE* in = NULL;
    const char* filename = "/media/kaleb/extraSpace/wrf/2020/20200101/hrrr.20200101.00.00.grib2";

    in = fopen(filename, "rb");
    if (!in) {
        fprintf(stderr, "Err: filenoopen %s\n", filename);
        return 1;
    }
    unsigned char* buf;
    buf = (unsigned char*)malloc(32768);
    buf[0] = 'G';
    buf[1] = 'R';
    buf[2] = 'I';
    buf[3] = 'B';
    size_t len = 3;
    int err = 0;
    off_t offset = ftello(in) - 4;
    printf("The val of the offset is %d\n", offset);
    
    size_t n = fread(&buf[4], 1, len, in);
    printf("The val of n: %d\n", n);
    printf("The val of buf: %s\n", buf);
    
    // the second pass
    len = 1;
    n = fread(&buf[7], 1, len, in);
    printf("\nAfter the second pass:\n");
    printf("The value of n: %d\n", n);
    printf("The value of buf: %s\n", buf);
    
    long edition = buf[7];
    printf("The edition is: %d\n");

    free(buf);
    fclose(in);
}