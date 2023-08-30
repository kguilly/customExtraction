#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <eccodes.h>

int main() {
    const char* wrf_file_path = "/home/kaleb/wrf_data/";
    FILE* f;
    try {
        f = fopen(wrf_file_path, "rb");
        if (!f) throw(wrf_file_path);
    } catch (const char* file_path) {
        printf("Error: could not open wrf file %s\n", file_path);
        exit(0);
    }

    int err = 0; long num_points = 0;
    const double missing = 1.0e36;
    double *grib_lats, *grib_lons, *grib_values;
    codes_handle *h = NULL;

    while ((h = codes_handle_new_from_file(0, f, PRODUCT_GRIB, &err))!= NULL) {
        CODES_CHECK(codes_get_long(h, "numberOfPoints", &num_points), 0);
        CODES_CHECK(codes_set_double(h, "missingValue", missing), 0);
        
        grib_lats = (double*)malloc(num_points * sizeof(double));
        if(!grib_lats){
            fprintf(stderr, "Error: unable to allocate %ld bytes\n", (long)(num_points * sizeof(double)));
            exit(0);
        }
        grib_lons = (double*)malloc(num_points * sizeof(double));
        if (!grib_lons){
            fprintf(stderr, "Error: unable to allocate %ld bytes\n", (long)(num_points * sizeof(double)));
            std::free(grib_lats);
            exit(0);
        }
        grib_values = (double*)malloc(num_points * sizeof(double));
        if(!grib_values){
            fprintf(stderr, "Error: unable to allocate %ld bytes\n", (long)(num_points * sizeof(double)));
            std::free(grib_lats);
            std::free(grib_lons);
            exit(0);
        }
        CODES_CHECK(codes_grib_get_data(h, grib_lats, grib_lons, grib_values), 0);
        
        std::free(grib_values);
        std::free(grib_lats);
        std::free(grib_lons);

        codes_handle_delete(h);
    }
}