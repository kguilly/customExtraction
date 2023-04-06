/*
This file will test if its possible to send a file pointer to CUDA and then extract the GRIB 2 file using 
an object generated by ECCODES that will indicate where to find the values 


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
codes_index* index = NULL;
char shortnames[MAX_NUM_STRINGS][MAX_STRING_LENGTH]= {"2t", "u", "v"};
long passed_levels[numParams] = {2, 1000, 1000};
int param_indexes[numParams][numParams] = {{-1, -1, -1},
                                           {-1, -1, -1},
                                           {-1, -1, -1}}

station_t * stationArr;
int numStations;

void build_index_obj();


int main() {

    station_t st_1;
    station_t st_2;
    st_1.lat = 30.2241;
    st_1.lon = 272.0198;

    st_2.lat = 29.7958;
    st_2.lon = 270.8229;

    station_t* stationArr = new station_t[2];
    numStations = 2;
    stationArr[0] = st_1;
    stationArr[1] = st_2;

    std::cout << "BEFORE" << std::endl;
    for (int i=0; i<numStations; i++) {
        station_t st = stationArr[i];
        std::cout << "CL PT for st " << i << ": " << st.closestPoint << std::endl;
    }

    // stationArr = extract_indexes(stationArr);
    std::cout << "\nAFTER" << std::endl;
    for (int i=0; i<numStations; i++) {
        station_t st = stationArr[i];
        std::cout << "CL PT for st " << i << ": " << st.closestPoint << std::endl;
    }

    // build the grib_index object
    build_index_obj();


    codes_index_delete(index);
    delete [] stationArr;

    return 0;
}

void build_index_obj () {
    codes_handle* h    = NULL;
    const char* infile       = "/media/kaleb/extraSpace/wrf/2020/20200101/hrrr.20200101.00.00.grib2";
    long *steps, *levels, *numbers; /* arrays */
    char** shortName = NULL;
    int i, j, k, l;
    size_t stepSize, levelSize, shortNameSize, numberSize;
    long ostep, olevel, onumber;
    char oshortName[200];
    size_t lenshortName = 200;
    int ret = 0, count = 0;

    // if (argc != 2) usage(argv[0]);
    //infile = argv[1];

    printf("indexing...\n");

    /* create an index given set of keys*/
    index = codes_index_new(0, "shortName,level,step", &ret);
    if (ret) {
        fprintf(stderr, "Error: %s\n", codes_get_error_message(ret));
        exit(ret);
    }

    /* indexes a file */
    ret = codes_index_add_file(index, infile);
    if (ret) {
        fprintf(stderr, "Error: %s\n", codes_get_error_message(ret));
        exit(ret);
    }
    printf("end indexing...\n");

    /* get the number of distinct values of "step" in the index */
    CODES_CHECK(codes_index_get_size(index, "step", &stepSize), 0);
    steps = (long*)malloc(sizeof(long) * stepSize);
    if (!steps) exit(1);

    /* get the list of distinct steps from the index */
    /* the list is in ascending order */
    CODES_CHECK(codes_index_get_long(index, "step", steps, &stepSize), 0);
    printf("stepSize=%ld\n", (long)stepSize);
    for (i = 0; i < stepSize; i++)
        printf("%ld ", steps[i]);
    printf("\n");

    /*same as for "step"*/
    CODES_CHECK(codes_index_get_size(index, "level", &levelSize), 0);
    levels = (long*)malloc(sizeof(long) * levelSize);
    if (!levels) exit(1);

    /*same as for "step"*/
    CODES_CHECK(codes_index_get_long(index, "level", levels, &levelSize), 0);
    printf("levelSize=%ld\n", (long)levelSize);
    for (i = 0; i < levelSize; i++)
        printf("%ld ", levels[i]);
    printf("\n");

    CODES_CHECK(codes_index_get_size(index, "shortName", &shortNameSize), 0);
    shortName = (char**)malloc(sizeof(char*) * shortNameSize);
    if (!shortName) exit(1);
    /*same as for "step"*/
    CODES_CHECK(codes_index_get_string(index, "shortName", shortName, &shortNameSize), 0);
    printf("shortNameSize=%ld\n", (long)shortNameSize);
    for (i = 0; i < shortNameSize; i++)
        printf("%s ", shortName[i]);
    printf("\n");

    count = 0;
    /* nested loops on the keys values of the index */
    /* different order of the nested loops doesn't affect performance*/
    for (i = 0; i < shortNameSize; i++) {
        /* select the GRIB with shortName=shortName[i] */
        codes_index_select_string(index, "shortName", shortName[i]);

        for (l = 0; l < levelSize; l++) {
            /* select the GRIB with level=levels[l] */
            codes_index_select_long(index, "level", levels[l]);

            // for (j = 0; j < numberSize; j++) {
            //     /* select the GRIB with number=numbers[j] */
            //     codes_index_select_long(index, "number", numbers[j]);

            for (k = 0; k < stepSize; k++) {
                /* select the GRIB with step=steps[k] */
                codes_index_select_long(index, "step", steps[k]);

                /* create a new codes_handle from the index with the constraints
                    imposed by the select statements. It is a loop because
                    in the index there could be more than one GRIB with those
                    constraints */
                while ((h = codes_handle_new_from_index(index, &ret)) != NULL) {
                    count++;
                    if (ret) {
                        fprintf(stderr, "Error: %s\n", codes_get_error_message(ret));
                        exit(ret);
                    }
                    lenshortName = 200;
                    codes_get_string(h, "shortName", oshortName, &lenshortName);
                    codes_get_long(h, "level", &olevel);
                    // codes_get_long(h, "number", &onumber);
                    codes_get_long(h, "step", &ostep);
                    // printf("shortName=%s ", oshortName);
                    // printf("level=%ld ", olevel);
                    // // printf("number=%ld ", onumber);
                    // printf("step=%ld \n", ostep);

                    // for each element of the array
                        // if the shortnames match and the levels match
                            // store it in the paramIdxes array 
                            // print it out
                    for (int j=0; j<numParams; j++) {
                        long currLevel = passed_levels[j];
                        char currShortname[MAX_STRING_LENGTH] = shortName[j];
                        int result = strcmp(oshortName, currShortname);
                        if (result == 0 && currLevel == olevel) {
                            printf("\nShortname: %s was found at shortnameidx: %d\n", currShortname, i);
                            printf("and at levelIdx: %d\n", l);
                        }
                    }


                    codes_handle_delete(h);
                }
                if (ret && ret != GRIB_END_OF_INDEX) {
                    fprintf(stderr, "Error: %s\n", codes_get_error_message(ret));
                    exit(ret);
                }
            }
            
        }
    }
    printf("  %d messages selected\n", count);
}


