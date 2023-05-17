/*
This file will demonstrate the minimum calls to ECCODES necessary to decompress a layer
of a grib2 file. 


*/

#include <stdio.h>
#include <stdlib.h>
// #include "eccodes.h"
// it. is. time.
// #include "src/eccodes.h"
#include "src/grib_api_internal.h"

// all the .h files from eccodes
/*
#include "./src/grib_box_class.h"
#include "./src/eccodes_prototypes.h"
#include "./src/grib_optimize_decimal_factor.h"
#include "./src/grib_nearest_factory.h"
#include "./src/grib_accessor_factory.h"
#include "./src/grib_emoslib.h"
#include "./src/md5.h"
#include "./src/grib_box_class.h"
*/

int main(){

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

    /* create new handle from a message in a file */
    grib_handle* h = grib_handle_new_from_file(0, in, &err);
    if (h == NULL) {
        fprintf(stderr, "Error: unable to create handle from file %s\n", filename);
        return 1;
    }

    // CODES_CHECK(codes_get_long(h, "numberOfPoints", &numberOfPoints), 0);
    // CODES_CHECK(codes_set_double(h, "missingValue", missing), 0);

    codes_get_long(h, "numberOfPoints", &numberOfPoints);
    // codes_set_double(h, "missingValue", missing);

    lats = (double*)malloc(numberOfPoints * sizeof(double));
    if (!lats) {
        fprintf(stderr, "Error: unable to allocate %ld bytes\n", (long)(numberOfPoints * sizeof(double)));
        return 1;
    }
    lons = (double*)malloc(numberOfPoints * sizeof(double));
    if (!lons) {
        fprintf(stderr, "Error: unable to allocate %ld bytes\n", (long)(numberOfPoints * sizeof(double)));
        free(lats);
        return 1;
    }
    values = (double*)malloc(numberOfPoints * sizeof(double));
    if (!values) {
        fprintf(stderr, "Error: unable to allocate %ld bytes\n", (long)(numberOfPoints * sizeof(double)));
        free(lats);
        free(lons);
        return 1;
    }

    grib_get_data(h, lats, lons, values);

    free(lats);
    free(lons);
    free(values);
    grib_handle_delete(h);
    return 0;
}