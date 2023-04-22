/*
This file will test if its possible to send a file pointer to CUDA and then extract the GRIB 2 file using 
an object generated by ECCODES that will indicate where to find the values 

// pseudo code

int main() {
    pass array of shortnames and levels { 
        char * passed_params[numPassed][2] = {("t2", "2"),
                                                ("u", "1000"),
                                                ("v", "1000")}
        OR 

        char *** passed_params;
        passed_params = malloc(numPassed * sizeof(char**));
        for (int i=0; i<numPassed; i++) {
            passed_params[i] = malloc(2 * sizeof(char*));
        }
        passed_params[0][0] = "t2";
        passed_params[0][1] = "2";
        char* my_value = passed_params[0][0];

    }

    for i in range numPassedParams:
        codes_index_select_string(index, "shortName", passed_params[i][0]);
        codes_index_select_long(index, "level", passed_params[i][1]);
        codes_index_select_long(index, "step", 0);

        while ((h=codes_handle_new_from_index(index, &ret)) != NULL) {
            - print info about the layer and decompress
        }
}
*/



#include <iostream>
#include <stdlib.h>
#include "eccodes.h"
#include <cuda_runtime.h>
#include <vector>
#include <string>

// personal functions
#include "decompress_grib_test.h"
#include "shared_test_objs.h"

#define MAX_NUM_STRINGS 100
#define MAX_STRING_LENGTH 100
#define numParams 3


const char * grib_file_path = "/media/kaleb/extraSpace/wrf/";
const char * output_file_path = "/home/kaleb/Desktop/cuda_4-3/";
std::vector<std::string> vctrDate = {"2020", "01", "01"};
codes_index* gr_idx_obj;
int numStations;

const char* passed_params[numParams][2] = {{"2t", "2"},
                                     {"u", "1000"},
                                     {"v", "1000"}};

void st_closest_pts(station_t*);
void nonThreaded_decompression(station_t*);


int main() {

    station_t st_1;
    station_t st_2;
    st_1.lat = 30.2241;
    st_1.lon = 272.0198;

    st_2.lat = 29.7958;
    st_2.lon = 270.8229;

    station_t * stationArr = new station_t[2];
    numStations = 2;
    stationArr[0] = st_1;
    stationArr[1] = st_2;

    // build the grib_index object
    st_closest_pts(stationArr);
    
    std::cout << "\n" << std::endl;
    
    nonThreaded_decompression(stationArr);

    codes_index_delete(gr_idx_obj);
    delete [] stationArr;

    return 0;
}

void st_closest_pts (station_t* stationArr) {
    codes_handle* h    = NULL;
    gr_idx_obj = NULL;
    long *steps, *levels, *numbers; /* arrays */
    char** shortName = NULL;
    int i, j, k, l;
    int ret = 0, count = 0, err = 0, missing = 0;
    long num_points = 0;
    FILE* f;

    // if (argc != 2) usage(argv[0]);
    //infile = argv[1];

    printf("finding the nearest points for the stations...\n");
    std::string full_path = grib_file_path + vctrDate.at(0) + "/" + vctrDate.at(0) + \
                            vctrDate.at(1) + vctrDate.at(2) + "/" + "hrrr."  + vctrDate.at(0) + \
                            vctrDate.at(1) + vctrDate.at(2) + ".00.00.grib2";
    try {
        f = fopen(full_path.c_str(), "rb");
        if (!f) throw(full_path);
    } catch (std::string full_path) {
        std::cout << "could not open file" << std::endl;
        return;
    }

    double* grib_lats, *grib_lons, *gr_vals;
    codes_handle * idx_h = codes_handle_new_from_file(0, f, PRODUCT_GRIB, &err);

    if (!idx_h || err != 0) {
        std::cout << "could not make handle" << std::endl;
        return;
    }
    CODES_CHECK(codes_get_long(idx_h, "numberOfPoints", &num_points), 0);
    CODES_CHECK(codes_set_double(idx_h, "missingValue", missing), 0);

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
    gr_vals = (double*)malloc(num_points * sizeof(double));
    if(!gr_vals){
        fprintf(stderr, "Error: unable to allocate %ld bytes\n", (long)(num_points * sizeof(double)));
        std::free(grib_lats);
        std::free(grib_lons);
        exit(0);
    }
    CODES_CHECK(codes_grib_get_data(idx_h, grib_lats, grib_lons, gr_vals), 0);

    stationArr = extract_indexes(stationArr, grib_lats, grib_lons, numStations, num_points);

    std::free(grib_lats);
    std::free(grib_lons);
    std::free(gr_vals);
    fclose(f);

    printf("The nearest points for the stations are: \n");
    for (int i=0; i<numStations; i++) {
        int closestPt = stationArr[i].closestPoint;
        printf("Station %d's closest point: %d\n", i, closestPt);

    }
}


void nonThreaded_decompression(station_t* stationArr) {
    int ret = 0;
    size_t stepSize = 0, levelSize = 0, shortNameSize = 0;
    char oshortName[200];
    long ostep, olevel, onumber;
    grib_handle* h;
    /*create a new index obj*/
    gr_idx_obj = codes_index_new(0, "shortName,level,step", &ret);
    /*vars for decompressing the data*/
    long numberOfPoints = 0;
    const double missing = 1.0e36;
    double *lats, *lons, *values;

    std::string full_path = grib_file_path + vctrDate.at(0) + "/" + vctrDate.at(0) + \
                            vctrDate.at(1) + vctrDate.at(2) + "/" + "hrrr."  + vctrDate.at(0) + \
                            vctrDate.at(1) + vctrDate.at(2) + ".00.00.grib2";

    /* index the particular file */
    ret = codes_index_add_file(gr_idx_obj, full_path.c_str());
    if (ret) {
        fprintf(stderr, "Error: %s\n", codes_get_error_message(ret));
        exit(ret);
    }

    /* get the number of distinct values of "step" in the index */
    CODES_CHECK(codes_index_get_size(gr_idx_obj, "step", &stepSize), 0);
    CODES_CHECK(codes_index_get_size(gr_idx_obj, "level", &levelSize), 0);
    CODES_CHECK(codes_index_get_size(gr_idx_obj, "level", &levelSize), 0);

    // bool get_size_flag = true;
    
    for (int i=0; i<numParams; i++) {
        codes_index_select_string(gr_idx_obj, "shortName", passed_params[i][0]);
        codes_index_select_long(gr_idx_obj, "level", std::stol(passed_params[i][1]));
        codes_index_select_long(gr_idx_obj, "step", 0.0);

        while ((h=codes_handle_new_from_index(gr_idx_obj, &ret)) != NULL) {
            // print info about the layer and decompress
            if (ret) {
                fprintf(stderr, "Error: %s\n", codes_get_error_message(ret));
                exit(ret);
            }
            // if (get_size_flag) {
            //     // get the size of the index and handle object
            //     get_size_flag = false;
            //     CODES_CHECK(codes_get_size())
            // }
            size_t lenshortName = 200;
            codes_get_string(h, "shortName", oshortName, &lenshortName);
            codes_get_long(h, "level", &olevel);
            codes_get_long(h, "step", &ostep);

            printf("Shortname: %s, level: %ld.\n", oshortName, olevel);

            CODES_CHECK(codes_get_long(h, "numberOfPoints", &numberOfPoints), 0);
            CODES_CHECK(codes_set_double(h, "missingValue", missing), 0);

            lats = (double*)malloc(numberOfPoints * sizeof(double));
            if (!lats) {
                fprintf(stderr, "Error: unable to allocate %ld bytes\n", (long)(numberOfPoints * sizeof(double)));
                return;
            }
            lons = (double*)malloc(numberOfPoints * sizeof(double));
            if (!lons) {
                fprintf(stderr, "Error: unable to allocate %ld bytes\n", (long)(numberOfPoints * sizeof(double)));
                free(lats);
                return;
            }
            values = (double*)malloc(numberOfPoints * sizeof(double));
            if (!values) {
                fprintf(stderr, "Error: unable to allocate %ld bytes\n", (long)(numberOfPoints * sizeof(double)));
                free(lats);
                free(lons);
                return;
            }

            CODES_CHECK(codes_grib_get_data(h, lats, lons, values), 0);
            for (int i=0; i<numStations; i++) {
                station_t curr_st = stationArr[i];
                int close_pt = curr_st.closestPoint;
                double selected_val = values[close_pt];

                printf("Station %d's %s value: %0.3f\n", i, oshortName, selected_val);
            }
            std::cout << "\n" << std::endl;
            free(lats);
            free(lons);
            free(values);
            codes_handle_delete(h);
        }
    }


}




