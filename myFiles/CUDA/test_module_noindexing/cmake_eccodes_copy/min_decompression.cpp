#include "src/eccodes.h"
#include <stdio.h>
#include <stdlib.h>


int main() {
    int err              = 0;
    size_t i             = 0;
    FILE* in             = NULL;
    const char* filename = "/media/kaleb/extraSpace/wrf/2020/20200101/hrrr.20200101.00.00.grib2";
    long numberOfPoints  = 0;
    const double missing = 1.0e36;
    double *lats, *lons, *values; /* arrays */

    in = fopen(filename, "rb");
    if (!in) {
        fprintf(stderr, "Error: unable to open input file %s\n", filename);
        return 1;
    }
    printf("Hello world\n");

    grib_handle* h = grib_handle_new_from_file(0, in, &err);
    if (h==NULL) {
        printf("Err: unable to create handle\n");
        return 1;
    }
    printf("Grib handle has been created\n");
    grib_handle_delete(h);
    return 0;
}