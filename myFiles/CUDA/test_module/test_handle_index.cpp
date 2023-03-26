/*
This file was written in an attempt to get the "codes_index" type to work,
however, I could not. The examples provided did not work

*/


#include <stdio.h>
#include <stdlib.h>
#include "eccodes.h"
#include <time.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <vector> 
#include <bits/stdc++.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <map>
#include <time.h>
#include <cassert>
#include <cmath>
#include "semaphore.h"
#define MAX_VAL_LEN 1024

int main(){

    codes_handle * h = NULL; // use to unpack each layer of the file
    const double missing = 1.0e36;        // placeholder for when the value cannot be found in the grib file
    int msg_count = 0;  // KEY: will match with the layer of the passed parameters
    int err = 0;
    size_t vlen = MAX_VAL_LEN;
    char value_1[MAX_VAL_LEN];
    bool flag = true; 
    int numberOfPoints=0;
    double *grib_values;
    std::string name_space = "parameter";
    const char* file_path = "/media/kaleb/extraSpace/wrf/2020/20200101/hrrr.20200101.00.00.grib2";
    const char* index_path = "/media/kaleb/extraSpace/wrf/2020/20200101/index_20200101.00.00.grib";
    codes_index * index;
    
    // int layerNum = 2;

    // // create idx
    // codes_index* index = codes_index_new_from_file(NULL, file_path, "parameterNumber");
    // codes_index* index = codes_index_new()

    // // build the index
    // int err = 0;
    // err = codes_index_build(index, file_path, "parameterNumber");
    // if (err != CODES_SUCCESS){
    //     std::cout << "Error building the index" << std::endl;
    //     return 1;
    // }

    // // write the index out to the file
    // err = codes_index_write(index, index_path, "parameterNumber");
    // if (err != CODES_SUCCESS){
    //     std::cout << "Error writing out the index" << std::endl;
    //     return 1;
    // }

    // FILE* f;
    
    // f = fopen(file_path, "rb");
    // if (!f) std::cout << "NO OPEN GRIBBBB" << std::endl;

    // unsigned long key_iterator_filter_flags = CODES_KEYS_ITERATOR_ALL_KEYS |
    //                                           CODES_KEYS_ITERATOR_SKIP_DUPLICATES;

    // // Get the handle to the second message in the index
    // codes_handle* handle = codes_handle_new_from_index(index, &err);

    // // Set the input file for the handle
    // codes_handle* h = codes_grib_handle_new_from_file(NULL, f, &err);

    // CODES_CHECK(err, 0);

    // // Copy data from h to handle
    // const char* name = "handle";
    // codes_copy_namespace(handle, name, h);
    

    // // print out the header information to see if we did this right
    // codes_keys_iterator* kiter = codes_keys_iterator_new(handle, key_iterator_filter_flags, name_space.c_str());
    // if (!kiter) {
    //     fprintf(stderr, "Error: unable to create keys iterator while reading params\n");
    //     exit(1);
    // }
    // while(codes_keys_iterator_next(kiter)){
    //     const char*name = codes_keys_iterator_get_name(kiter);
    //     vlen = MAX_VAL_LEN;
    //     memset(value_1, 0, vlen);
    //     CODES_CHECK(codes_get_string(handle, name, value_1, &vlen), name);
    //     std::cout << "NAME: " << name << ", Val: " << value_1 << std::endl;
    // }



        
    // codes_handle_delete(h);
    // codes_handle_delete(handle);
    // codes_index_delete(index);

    // Attempt number 2
    unsigned long key_iterator_filter_flags = CODES_KEYS_ITERATOR_ALL_KEYS;

    const char * paramName = "temperature";
    long paramId;
    codes_handle * handle = NULL;
    FILE* fp = std::fopen(file_path, "rb");
    if (!fp) {
        std::cout << "no file" << std::endl;
        return 1;
    }
    index = codes_index_new(NULL, paramName, &err);
    if (!index) {
        std::cout << "no index" << std::endl;
        std::fclose(fp);
        return 1;
    }
    codes_index_add_file(index, file_path);

    // paramId = codes_grib_util_get_param_id(paramName);
    // if (paramId < 0) {
    //     sdt::cout << "no find param" << std::endl;
    //     std::fclose(fp);
    //     return 1;
    // }

    handle = codes_handle_new_from_index(index, &err);
    if (!handle) {
        std::cout << "no handle \n";
        std::fclose(fp);
        return 1;
    }
    codes_keys_iterator* kiter = codes_keys_iterator_new(handle, key_iterator_filter_flags, "parameter");
    if (!kiter) {
        fprintf(stderr, "Error: unable to create keys iterator while reading params\n");
        exit(1);
    }
    std::string strUnits, strName, strValue, strHeader, strnametosendout;
    while(codes_keys_iterator_next(kiter)){
        const char*name = codes_keys_iterator_get_name(kiter);
        vlen = MAX_VAL_LEN;
        memset(value_1, 0, vlen);
        CODES_CHECK(codes_get_string(h, name, value_1, &vlen), name);
        strName = name, strValue = value_1;
        std::cout << "Name: " << name << ", Value: " << value_1 << std::endl;
    }

    codes_index_delete(index);
    codes_handle_delete(handle);
    std::fclose(fp);

    

}