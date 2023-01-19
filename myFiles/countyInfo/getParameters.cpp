// #include <filesystem>
// namespace fs = std::filesystem;
#include <stdio.h>
#include <stdlib.h>
#include "eccodes.h"
// #include <time.h>
// #include <iostream>
#include <fstream>
#include <cstring>
#include <vector> 
#include <bits/stdc++.h>
// #include <sys/types.h>
// #include <sys/stat.h>
// #include <map>
// #include <time.h>
// #include <cassert>
// #include <cmath>
// #include "semaphore.h"
using namespace std;
#define MAX_VAL_LEN 1024

int main(){

    // string fulldate = vctrDay.at(3), year = vctrDay.at(0);
    // string strFirstHour = (arrHourRange.at(0) < 10) ? "0"+ to_string(arrHourRange.at(0)) : to_string(arrHourRange.at(0)); 
    string fullpathtofile = "/home/kaleb/Desktop/weekInputData/2021/20210601/hrrr.20210601.00.00.grib2";
    string strOutput = "";
    strOutput.append("layer,name,units\n");

    //init params
    unsigned long key_iterator_filter_flags = CODES_KEYS_ITERATOR_ALL_KEYS | 
                                                CODES_KEYS_ITERATOR_SKIP_DUPLICATES;
    const char* name_space = "parameter";
    size_t vlen = MAX_VAL_LEN;
    char value[MAX_VAL_LEN];
    int err = 0, layerNum =0;
    FILE* gribFile;
    codes_handle* handle = NULL;
    gribFile = fopen(fullpathtofile.c_str(), "rb");
    if(!gribFile){
        fprintf(stderr, "Error: unable to open file while reading parameters \n %s \n", fullpathtofile);
        exit(1);
    }
    
    // now open the file for reading, read the contents into a csv 
    while((handle = codes_handle_new_from_file(0, gribFile, PRODUCT_GRIB, &err))!=NULL){
        codes_keys_iterator * kiter = NULL;
        layerNum++;

        kiter = codes_keys_iterator_new(handle, key_iterator_filter_flags, name_space);
        if(!kiter){
            fprintf(stderr, "Error: Unable to create keys iterator while reading parameters\n");
            exit(1);
        }
        string strUnits;
        while (codes_keys_iterator_next(kiter)){
            const char*name = codes_keys_iterator_get_name(kiter);
            vlen = MAX_VAL_LEN;
            memset(value, 0, vlen);
            CODES_CHECK(codes_get_string(handle, name, value, &vlen), name);
            
            // append that booshaka to the output string
            // start reading when the name = units
            // if the units = unknown, break
            string strname = name, strvalue = value;
            if(strname.find("units")!= string::npos){
                if(strvalue.find("unknown") != string::npos) break;
                strUnits = value;
                strOutput.append(to_string(layerNum)+",");
            }
            if(strname.find("name") != string::npos){
                if(strvalue.find("unknown") != string::npos) break;
                strOutput.append(strvalue+","+strUnits+"\n");
            }
        }
        codes_keys_iterator_delete(kiter);
        codes_handle_delete(handle);
    }

    fclose(gribFile);

    FILE* paramFile = fopen("parameterInfo.csv", "w");
    fwrite(strOutput.c_str(), strOutput.size(), 1, paramFile);
    printf("%s", strOutput.c_str());
    fclose(paramFile);
    return 0;
}