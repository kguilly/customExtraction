/* test to read the 1 hour prediction subset precipitation data */
// want to double check that the parameter we read is precipitation
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <vector>
#include <iostream>
#include <eccodes.h>
#include <dirent.h>
#include <sys/stat.h>
#include <filesystem>

#define MAX_VAL_LEN 1024

using namespace std;

string precip_file_path = "/media/kaleb/extraSpace/precip_data/";
vector<string> strCurrentDay = {"2020", "01", "01", "20200101"};
string hour = "00";
int arrHourRange[5] = {0, 1, 2, 3, 4};
int intHourRange = 5;

bool dirExists(string str){ return true;}

int main() {
    
    string year = strCurrentDay.at(0);
    string month = strCurrentDay.at(1);
    string day = strCurrentDay.at(2);
    string ymd = strCurrentDay.at(3);

    if (dirExists(precip_file_path) == false) {
        fprintf(stderr, "Error: could not find wrf precipitation file directory %s", precip_file_path.c_str());
            exit(1);
    }

    string fullfilepath = precip_file_path + "hrrr/" + ymd + "/";
    string searchString = "hrrr.t" + hour;
    string fileName;

    // search through the directory to find the corresponding file
    bool filenotfound = true;
    for (const auto& entry : std::filesystem::directory_iterator(fullfilepath)) {
        if (entry.is_regular_file()) {
            fileName = entry.path().filename().string();
            if (fileName.find(searchString) != string::npos) {
                    filenotfound = false;
                    break;
                }
        }
    }

    if (filenotfound) {
        fprintf(stderr, "Error: precip file not found in directory %s", fullfilepath.c_str());
        exit(1);
    }

    fullfilepath += fileName;

    // now open the file and use eccodes to read the data from it
    FILE* f;
    try {
        f = fopen(fullfilepath.c_str(), "rb");
        if (!f) throw(fullfilepath);
    } catch (string file) {
        printf("Error: could not open file %s\n", fullfilepath.c_str());
    }
    
    codes_handle * h = NULL; // use to unpack each layer of the file
    const double missing = 1.0e36;        // placeholder for when the value cannot be found in the grib file
    int msg_count = 0;  // KEY: will match with the layer of the passed parameters
    int err = 0;
    size_t vlen = MAX_VAL_LEN;
    char value_1[MAX_VAL_LEN];
    bool flag = true; 
    // numberOfPoints=0;
    double *grib_values, *grib_lats, *grib_lons;
    std::string name_space = "parameter";
    unsigned long key_iterator_filter_flags = CODES_KEYS_ITERATOR_ALL_KEYS |
                                              CODES_KEYS_ITERATOR_SKIP_DUPLICATES;
    
    while((h = codes_handle_new_from_file(0, f, PRODUCT_GRIB, &err)) != NULL){
        msg_count++;


        // add the information to the header
        codes_keys_iterator* kiter = codes_keys_iterator_new(h, key_iterator_filter_flags, name_space.c_str());
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
            strName = name;
            strValue = value_1;
            if(strName.find("name")!=std::string::npos){
                // strValue.erase(remove(strValue.begin(), strValue.end(), ','), strValue.end());
                //strHeader.append(to_std::string(msg_count)+"_"+strValue);
                strnametosendout = strValue;
            }
            else if(strName.find("units")!=std::string::npos){
                //strHeader.append("("+strValue+")");
                strUnits = "(" + strValue + ")";
            }
        }
        strHeader = strnametosendout + " " + strUnits;

        // 
        if (strHeader.find("Total Precipitation") != string::npos) {

            long num_points = 0;
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

            Station *station;
            for (int i=0; i<numStations; i++) {
                station = &stationArr[i];
                float avglat = (station->latll + station->latur) / 2;
                float avglon = (station->lonll + station->lonur) / 2;
                int closestIdx = 0;
                for (int j=0; j<num_points;j++){
                    double distance = pow(grib_lats[j]- avglat, 2) + pow(grib_lons[j]-avglon, 2);
                    double closestDistance = pow(grib_lats[closestIdx] - avglat, 2) + pow(grib_lons[closestIdx] - avglon,2);
                    if(distance < closestDistance) closestIdx = j;
                }
                station->closestPoint = closestIdx;
            }
            free(grib_values);
            free(grib_lats);
            free(grib_lons);
        }
        codes_handle_delete(h);
    }
    
    
    return 0;
}

