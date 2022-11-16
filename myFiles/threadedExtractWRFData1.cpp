/*
This implementation comes off the back of extractWRFData3.cpp. Will thread
each hour into its own threading function. First attempt will use pthreads,
as PTHREADs is a C library, matching ECCodes as a C library, as well as there
being an example included in the eccodes library 


Compile:
g++ -Wall -threadedExtractWRFData1.cpp -leccodes -lpthread

*/

#include <stdio.h>
#include <stdlib.h>
#include "eccodes.h"
#include <time.h>
#include <iostream>
#include <cstring>
#include <vector> 
#include <sys/types.h>
#include <sys/stat.h>
#include <map>
#include <time.h>
#include <pthread.h>
#include <cassert>
#define MAX_VAL_LEN 1024
using namespace std;

/* Timing variables */
struct timespec startExtract; 
struct timespec endExtract;
struct timespec startMatching;
struct timespec endMatching;
struct timespec startTotal;
struct timespec endTotal;
double extractTime;
double matchTime;
double totalTime;

vector<int> beginDay = {2019, 1, 2}; // arrays for the begin days and end days. End Day is NOT inclusive. 
                                    // when passing a single day, pass the day after beginDay for endDay
                                    // FORMAT: {yyyy, mm, dd}
vector<int> endDay = {2019, 1, 3};

string filePath = "/home/kalebg/Desktop/School/Y4S1/REU/extraction/UtilityTools/extractTools/data/";  // path to "data" folder. File expects structure to be: 
                                        // .../data/<year>/<yyyyMMdd>/hrrr.<yyyyMMdd>.<hh>.00.grib2
                                        // for every hour of every day included. be sure to include '/' at end

// Structure for holding the selected station data. Default will be the 5 included in the acadmeic
// paper: "Regional Weather Forecasting via Neural Networks with Near-Surface Observational and Atmospheric Numerical Data."
struct Station{
    string name;
    float lat;
    float lon;
    double *values; // holds the values of the parameters. Index of this array will 
                    // correspond to index if the Parameter array. This will be a single hour's data
    map<string, double*> dataMap; // this will hold the output data. structured as {"yyyymmddHH" : [param1, param2, ...]}}

    int closestPoint; // index in the grib file of the point closest to the station's lats and lons
    
    // NOT IMPLEMENTED YET
    // the next values will store the index of the parameters around the station that are returned by the GRIB file
    int topL[2]; // index of the lat and lons of the point up and to the left. 
    int topR[2]; 
    int botL[2];
    int botR[2];
};

struct Parameter{
    int layer; // order that the parameter is extracted in decompression. parameter number in ACM paper
               // layer is NOT reliable. Some parameters are repeated in decompression. Parameters with a height attached to them
               // are not shown in the "name" from the key iterator
    int paramId; // given from codes_keys_iterator object
    string shortName; //given
    string units; //given
    string name; // given
};

// Struct for passing one to many arguments into individual threading functions 
// Currently not in use, only need a single argument for the moment
struct threadArgs{
    FILE*f;
    string fileName;
    string pathName;
};

Station *stationArr; 
Parameter *objparamArr; // array of the WRF parameter names. Default is the 30 in 
                          // the academic paper "Regional Weather Forecasting via Neural
                          // Networks with Near-Surface Observational and Atmospheric Numerical 
                          // Data." Names are changed to fit the exact names
bool blnParamArr[149]; // this will be used to quickly index whether a parameter needs to be 
                        // extracted or not. Putting 149 spaces for 148 parameters because the
                        // layers of the parameters start at 1 rather than 0

int numStations, numParams;

// 24 hours in a day. Use this to append to the file name and get each hour for each file
string hours[] = {"00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11",
                       "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"};

// function to handle arguments passed. Will either build the Station array and/or paramter array
// based off of the arguments passed or will build them with default values
void handleInput(int, char**);

/* function to convert the lats and lons from the passed representation to the 
    way they are represented in the grib file. ONLY WORKS for coordinates in USA*/
void convertLatLons();

// function to check the begin day vs the end day to see if a valid range has been passed
bool checkDateRange(vector<int>, vector<int>);

// function to get the next day after the day passed
vector<int> getNextDay(vector<int>);

/* function to format the date as a vector of strings
   returns vector<string> date = {yyyy, mm, dd, yyyymmdd}*/
vector<string> formatDay(vector<int>);

/* function to check if a given directory exists*/
bool dirExists(string filePath);

/* function to read the data from a passed grib file */
void *readData(void*);

/* shorthand function for calculating the index of the closest point in the grib file to the lat and lon of a given station*/
static bool indexofClosestPoint(double*, double*,float,float, int, int);

/* function to map the data in the station's values array to the station's map */
void mapData(string, string);

int main(int argc, char*argv[]){
    clock_gettime(CLOCK_MONOTONIC, &startTotal);

    handleInput(argc, argv);
    convertLatLons();

    // validate the dates passed
    bool boolDateRange = checkDateRange(beginDay, endDay);
    if (boolDateRange == false){
        fprintf(stderr, "Invalid Date Range\n");
        exit(1);
    }
    vector<int> intcurrentDay = beginDay;

    // for each day
    while(checkDateRange(intcurrentDay, endDay)){
        vector<string> strCurrentDay = formatDay(intcurrentDay);

        // concatenate the file path
        string filePath1 = filePath + strCurrentDay.at(0) + "/" + strCurrentDay.at(3) + "/";
        // check if the file path exists
        if(dirExists(filePath1) == false){
            fprintf(stderr, "Error: could not find directory %s", filePath1.c_str());
            exit(1);
        }

        // Thread the hours 
        // allocate the threads and do some error checking
        pthread_t *threads = (pthread_t*)malloc(24 * sizeof(pthread_t)); // will be freed at the end of this iteration
        if(!threads){
            fprintf(stderr, "Error: unable to allocate %ld bytes for threads.\n", (long)(24*sizeof(pthread_t)));
            exit(0);
        }
        FILE* f[24]; // use to open the file for each hour

        // allocate the structs and do some error checking
        

        int threaderr; // keep track if the threading runs into an error
        for(int i=0;i<1;i++){ // for each hour, thread the file and filename
            f[i] = NULL;
            string hour = hours[i];
            string fileName = "hrrr."+strCurrentDay.at(3)+"."+hour+".00.grib2";
            string filePath2 = filePath1 + fileName;
            
            // place the arguments to pass to the thread function in the hour's struct 
            threadArgs arg;
            arg.f = f[i];
            arg.fileName = fileName;
            arg.pathName = filePath2;
            threadArgs *threadArg = (threadArgs*)malloc(sizeof(threadArgs));
            if(!threadArg){
                fprintf(stderr, "error: unable to allocate %ld bytes for thread structs.\n", (long)(24*sizeof(threadArgs)));
                exit(0);
            }
            *threadArg = arg;

            threaderr = pthread_create(&threads[i], NULL, &readData, threadArg);
            if(threaderr){
                assert(0);
                return 1;
            }

        }
        for(int i=0;i<24;i++){
            pthread_join(threads[i], NULL);
        }

        for (string hour:hours){
            mapData(strCurrentDay.at(3), hour);
        }        
        free(threads);
        intcurrentDay = getNextDay(intcurrentDay);
    }


    // print out all the elements in all the station's data maps
    for (int i=0; i<numStations;i++){
        Station station = stationArr[i];
        cout << "\n\nSTATION: " << station.name << endl;
        for(auto itr = station.dataMap.begin(); itr != station.dataMap.end(); ++itr){
            cout << itr->first << '\t';
            for (int i = 0; i<numParams; i++){
                 cout << *itr->second << " ";
                 itr->second ++;

            }
            cout << endl;
        }
    }

    for(int i =0; i<numStations;i++){
        // delete each station's value array
        delete [] stationArr[i].values;
    }
    delete [] objparamArr;
    delete [] stationArr;

    clock_gettime(CLOCK_MONOTONIC, &endTotal);
    totalTime = (endTotal.tv_sec - startTotal.tv_sec) * 1000.0;
    totalTime+= (endTotal.tv_nsec - startTotal.tv_nsec) / 1000000.0;
    printf("\n\nRuntime in ms:: %f\n", totalTime);
    printf("Extract Time in ms:: %f\n", extractTime);
    printf("Time to find index: %f\n\n", matchTime);

    return 0;
 }

 void handleInput(int argc, char* argv[]){
    
    // NOT FINISHED. DO NOT PASS ARGS
    if(argc > 1){
        // check to see if the correct arguments have been passed to fill the parameter and/or 
        // station arrays
    }
    else{
        // build the default staion array and parameter array
        numParams = 30;
        numStations = 5;
        
        
        objparamArr = new Parameter[numParams];
        Parameter sv, sws, gph, temp, mslp, gph1k, sp, stemp, gm, temp2m, 
                  ptemp2m, sphum2m, dewtemp2m, rhum2m, wspd10m, sfv, sshnf, slhnf, iblsli, lcc,
                  sdswrf, sdlwrd, suswrf, sulwrf, svbdsf, svddsf, srh3km, vucs, izp, htfp;
        sv.layer = 4, sv.name = "Visibility", sv.paramId = 3020, sv.shortName = "vis", sv.units = "m";
        sws.layer = 8, sws.name = "Wind speed (gust)", sws.paramId = 260065, sws.shortName = "gust", sws.units = "m s**-1";
        gph.layer = 18, gph.name = "Geopotential height", gph.paramId = 156, gph.shortName = "gh", gph.units = "gpm";
        temp.layer = 28, temp.name = "Temperature", temp.paramId = 130, temp.shortName = "t", temp.units = "K";
        mslp.layer = 39, mslp.name = "MSLP (MAPS System Reduction)", mslp.paramId = 260323, mslp.shortName = "mslma", mslp.units = "Pa";
        gph1k.layer = 40, gph1k.name = "Geopotential height", gph1k.paramId = 156, gph1k.shortName = "gpm", gph1k.units = "gpm";
        sp.layer = 57, sp.name = "Surface pressure", sp.paramId = 134, sp.shortName = "sp", sp.units = "Pa";
        stemp.layer = 59, stemp.name = "Temperature", stemp.paramId = 130, stemp.shortName = "t", stemp.units = "K";
        gm.layer = 61, gm.name = "Moisture availibility", gm.paramId = 260187, gm.shortName = "mstav", gm.units = "%";
        temp2m.layer = 66, temp2m.name = "2 metre temperature", temp2m.paramId = 167, temp2m.shortName = "2t", temp2m.units = "K"; 
        ptemp2m.layer = 67, ptemp2m.name = "Potential temperature", ptemp2m.paramId = 3, ptemp2m.shortName = "pt", ptemp2m.units = "K"; 
        sphum2m.layer = 68, sphum2m.name = "2 metre specific humidity", sphum2m.paramId = 174096, sphum2m.shortName = "2sh", sphum2m.units = "kg kg**-1";
        dewtemp2m.layer = 69, dewtemp2m.name = "2 metre dewpoint temperature", dewtemp2m.paramId = 168, dewtemp2m.shortName = "2d", dewtemp2m.units = "K";
        rhum2m.layer = 70, rhum2m.name = "2 metre relative humidity", rhum2m.paramId = 260242, rhum2m.shortName = "2r", rhum2m.units = "%";
        wspd10m.layer = 73, wspd10m.name = "10 metre wind speed", wspd10m.paramId = 207, wspd10m.shortName = "10si", wspd10m.units = "m s**-1";
        sfv.layer = 89, sfv.name = "Frictional velocity", sfv.paramId = 260073, sfv.shortName = "fricv", sfv.units = "m s**-1";
        sshnf.layer = 90, sshnf.name = "Sensible heat net flux", sshnf.paramId = 260003, sshnf.shortName = "shtfl", sshnf.units = "W m**-2";
        slhnf.layer = 91, slhnf.name = "Latent heat net flux", slhnf.paramId = 260002, slhnf.shortName = "lhtfl", slhnf.units = "W m**-2";
        iblsli.layer = 94, iblsli.name = "Surface lifted index", iblsli.paramId = 260127, iblsli.shortName = "lftx", iblsli.units = "K";
        lcc.layer = 98, lcc.name = "Low cloud cover", lcc.paramId = 3073, lcc.shortName = "lcc", lcc.units = "%";
        sdswrf.layer = 108, sdswrf.name = "Downward short-wave radiation flux", sdswrf.paramId = 260087, sdswrf.shortName = "dswrf", sdswrf.units = "W m**-2";
        sdlwrd.layer = 109, sdlwrd.name = "Downward long-wave radiation flux", sdlwrd.paramId = 260097, sdlwrd.shortName = "dlwrf", sdlwrd.units = "W m**-2";
        suswrf.layer = 110, suswrf.name = "Upward short-wave radiation flux", suswrf.paramId = 260088, suswrf.shortName = "uswrf", suswrf.units = "W m**-2";
        sulwrf.layer = 111, sulwrf.name = "Upward long-wave radiation flux", sulwrf.paramId = 260098, sulwrf.shortName = "ulwrf", sulwrf.units = "W m**-2";
        svbdsf.layer = 112, svbdsf.name = "Visile Beam Downward Solar Flux", svbdsf.paramId = 260346, svbdsf.shortName = "vbdsf", svbdsf.units = "W m**-2";
        svddsf.layer = 113, svddsf.name = "Visible Diffuse Downward Solar Flux", svddsf.paramId = 260347, svddsf.shortName = "vddsf", svddsf.units = "W m**-2";
        srh3km.layer = 115, srh3km.name = "Storm relative helicity", srh3km.paramId = 260125, srh3km.shortName = "hlcy", srh3km.units = "m**2 s**-2";
        vucs.layer = 121, vucs.name = "Vertical u-component shear", vucs.paramId = 3045, vucs.shortName = "vucsh", vucs.units = "s**-1";
        izp.layer = 125, izp.name = "Pressure", izp.paramId = 54, izp.shortName = "pres", izp.units = "Pa";
        htfp.layer = 128, htfp.name = "Pressure", htfp.paramId = 54, htfp.shortName = "pres", htfp.units = "Pa";

        *(objparamArr) = sv; *(objparamArr+1) = sws; *(objparamArr+2) = gph; *(objparamArr+3) = temp; *(objparamArr+4) = mslp;
        *(objparamArr+5) = gph1k; *(objparamArr+6) = sp; *(objparamArr+7) = stemp; *(objparamArr+8) = gm; *(objparamArr+9) = temp2m;
        *(objparamArr+10) = ptemp2m; *(objparamArr+11) = sphum2m; *(objparamArr+12) = dewtemp2m; *(objparamArr+13) = rhum2m; *(objparamArr+14) = wspd10m;
        *(objparamArr+15) = sfv; *(objparamArr+16) = sshnf; *(objparamArr+17) = slhnf; *(objparamArr+18) = iblsli; *(objparamArr+19) = lcc;
        *(objparamArr+20) = sdswrf; *(objparamArr+21) =sdlwrd; *(objparamArr+22) = suswrf; *(objparamArr+23) = sulwrf; *(objparamArr+24) = svbdsf;
        *(objparamArr+25) = svddsf; *(objparamArr+26) = srh3km; *(objparamArr+27) = vucs; *(objparamArr+28) = izp; *(objparamArr+29) = htfp;

        int layer =0;
        for (int i = 0; i<numParams; i++){
            layer = objparamArr[i].layer;
            blnParamArr[layer] = true;
        }        

        stationArr = new Station[numStations];

        Station bmtn, ccla, farm, huey, lxgn;
        bmtn.name = "BMTN";bmtn.lat = 36.91973;bmtn.lon = -82.90619;
        *(stationArr+0) = bmtn;
        ccla.name = "CCLA";ccla.lat = 37.67934; ccla.lon = -85.97877;
        *(stationArr+1) = ccla;
        farm.name = "FARM"; farm.lat = 36.93; farm.lon = -86.47;
        *(stationArr+2) = farm;
        huey.name = "HUEY"; huey.lat = 38.96701; huey.lon = -84.72165;
        *(stationArr+3) = huey;
        lxgn.name = "LXGN"; lxgn.lat = 37.97496; lxgn.lon = -84.53354;
        *(stationArr+4) = lxgn;

        // initialize the pointer array for each station to be of the length of the number of params
        for (int i=0; i<numStations;i++){
            stationArr[i].values = new double[numParams];
        }
    }
}

void convertLatLons(){
    for(int i=0; i<numStations;i++){
        Station station = stationArr[i];
        station.lon = (station.lon +360); // circle, going clockwise vs counterclockwise
    }
}

bool checkDateRange(vector<int> beginDate, vector<int> endDate){
	// check that the array is of the size expected // WORKS
	if((beginDate.size() != 3) || (endDate.size() != 3)) return false;

	// check that endDay is further than endDay
	if(beginDate.at(0)>endDate.at(0)) return false;
	else if(beginDate.at(0) == endDate.at(0)){
		if(beginDate.at(1) > endDate.at(1)) return false;
		else if(beginDate.at(1) == endDate.at(1)){
			if(beginDate.at(2) >= endDate.at(2)) return false;
		}
		
	}

	// check that they have actually passed a valid date

	return true;
}

// based on: https://www.studymite.com/cpp/examples/program-to-print-the-next-days-date-month-year-cpp/
vector<int> getNextDay(vector<int> beginDate){
	
	int day, month, year;
	day = beginDate.at(2);
	month = beginDate.at(1);
	year = beginDate.at(0);

	if (day > 0 && day < 28) day+=1; //checking for day from 0-27, can just increment
	else if(day == 28){
		if(month == 2){ // if it is february, need special care
			if ((year % 400 == 0) || (year % 100 != 0 || year % 4 == 0)){ // is a leap year
				day == 29;
			}
			else{
				day = 1;
				month = 3;
			}
		}
		else // its not feb
		{
			day += 1; 
		}
	}
	else if(day == 29) // last day check for feb on a leap year
	{
		if(month == 2){
			day = 1;
			month = 3;
		}
	}
	else if(day == 30) // last day check for april, june, sept, nov
	{
		if(month == 1 || month == 3 || month == 5 || month == 7 || month == 8 || month == 10 || month == 12)
		{
			day += 1;
		}
		else
		{
			day = 1;
			month +=1;
		}
	}
	else if(day == 31) // last day of the month
	{
		day = 1;
		if(month == 12) // last day of the year
		{
			year += 1;
			month = 1;
		}
		else month+=1;
	}
	vector<int> nextDay = {year, month, day};
	return nextDay;
}


vector<string> formatDay(vector<int> date){
	string strYear, strMonth, strDay;
	int intYear = date.at(0);
	int intMonth = date.at(1);
	int intDay = date.at(2);

	if(intMonth < 10){
		strMonth = "0" + to_string(intMonth);
	}
	else strMonth = to_string(intMonth);
	if(intDay < 10) strDay = "0" + to_string(intDay);
	else strDay = to_string(intDay);
	strYear = to_string(intYear);

	vector<string> formattedDate = {strYear , strMonth, strDay , (strYear+strMonth+strDay)};
	return formattedDate;
}


bool dirExists(string filePath){
 	struct stat info;

	if(stat(filePath.c_str(), &info) != 0){
		return false;
	}
	else if(info.st_mode & S_IFDIR){
		return true;
	}
	else return false;
}


void *readData(void *args){
    struct threadArgs threadArg = *(struct threadArgs*)args;

    //TESTING
    //cout << "entered threading function " << endl;

    FILE*f = (threadArg).f;
    string filePath2 = (threadArg).pathName;
    string fileName = (threadArg).fileName;


    cout << "Opening file: " << filePath2 << endl;
    //try to open the file
    try{
        f = fopen(filePath2.c_str(), "rb");
        if(!f) throw(filePath2);
    }
    catch(string file){
        printf("Error: could not open filename %s in directory %s", fileName.c_str(), file.c_str());
        return nullptr;
    }
    unsigned long key_iterator_filter_flags = CODES_KEYS_ITERATOR_ALL_KEYS |
                                              CODES_KEYS_ITERATOR_SKIP_DUPLICATES;
    // codes_grib_multi_support_on(NULL);

    codes_handle * h = NULL; // use to unpack each layer of the file

    const double missing = 1.0e36;        // placeholder for when the value cannot be found in the grib file
    int msg_count = 0;  // KEY: will match with the layer of the passed parameters
    int err = 0;

    char value[MAX_VAL_LEN];
    size_t vlen = MAX_VAL_LEN;

    bool flag = true; 

    long numberOfPoints=0;

    double *lats, *lons, *values; // lats, lons, and values returned from extracted grib file

    while((h=codes_handle_new_from_file(0, f, PRODUCT_GRIB, &err))!=NULL) // loop through every layer of the file
    {
        assert(h);
        msg_count++; // will be in layer 1 on the first run
        if (blnParamArr[msg_count] == true){

            clock_gettime(CLOCK_MONOTONIC, &startExtract);
            // extract the data
            CODES_CHECK(codes_get_long(h, "numberOfPoints", &numberOfPoints), 0);
            CODES_CHECK(codes_set_double(h, "missingValue", missing), 0);
            lats = (double*)malloc(numberOfPoints * sizeof(double));
            if(!lats){
                fprintf(stderr, "Error: unable to allocate %ld bytes\n", (long)(numberOfPoints * sizeof(double)));
                exit(0);
            }
            lons = (double*)malloc(numberOfPoints * sizeof(double));
            if (!lons){
                fprintf(stderr, "Error: unable to allocate %ld bytes\n", (long)(numberOfPoints * sizeof(double)));
                free(lats);
                exit(0);
            }
            values = (double*)malloc(numberOfPoints * sizeof(double));
            if(!values){
                fprintf(stderr, "Error: unable to allocate %ld bytes\n", (long)(numberOfPoints * sizeof(double)));
                free(lats);
                free(lons);
                exit(0);
            }
            CODES_CHECK(codes_grib_get_data(h, lats,lons,values), 0);
            // all the lats,lons, and values from this layer of the grib file are now stored, now match them
            // to their respective stations

            clock_gettime(CLOCK_MONOTONIC, &endExtract);
            double thisTime= (endExtract.tv_sec - startExtract.tv_sec) * 1000.0;
            thisTime+= (endExtract.tv_nsec - startExtract.tv_nsec) / 1000000.0;
            extractTime += thisTime;

            // if it is the first time, extract the index
            if (flag == true){
                clock_gettime(CLOCK_MONOTONIC, &startMatching);

                
                ////////////////////////
                // NEEDS OPTIMIZATION //
                ////////////////////////
              /*
                // // loop through each station and find the indexes of the 4 point nearest to the station
                // int closestPoint_1 = 0;
                // for (int i = 0; i<numStations; i++){
                //     Station *station = &stationArr[i];
                //     closestPoint_1=0;
                //     for (int j = 0; j<numberOfPoints; j++){
                //         if (values[j] != missing){
                    
                //             // find the point on the grib file that is closest to the latitude and longitude of the station
                //             if((pow((lats[j] - station->lat), 2) + pow((lons[j]-station->lon), 2)) <= (pow((lats[closestPoint_1] - station->lat), 2) + pow((lons[closestPoint_1] - station->lon), 2))){
                //                 // this is the closest point
                //                 closestPoint_1 = j;
                //             }

                //         }
                //     }
                //     // set the station's closest point
                //     station->closestPoint = closestPoint_1;
                // }        
              */  
                
                for(int i=0; i<numStations; i++){
                    Station *station = &stationArr[i];
                    // using a front to back algorithm
                    int front = 0, back = numberOfPoints-1, closestPoint = 0;
                    while (front <= back){
                        if(indexofClosestPoint(lats, lons, station->lat, station->lon, front, closestPoint)) closestPoint = front;
                        if(indexofClosestPoint(lats, lons, station->lat, station->lon, back, closestPoint)) closestPoint = back;
                        front++;back--;
                    }
                    station->closestPoint = closestPoint;
                }
                flag = false;
                clock_gettime(CLOCK_MONOTONIC, &endMatching);
                double timeeeeee = (endMatching.tv_sec - startMatching.tv_sec) * 1000.0;
                timeeeeee+=(endMatching.tv_nsec - startMatching.tv_nsec) / 1000000.0;
                matchTime+= timeeeeee;
            }

            // we've got the index of the closest lats and lons, now we just have to map them to each station's values arr
            // figuring out which index of each station's parameter array to put the values at 
            for (int i=0; i<numStations; i++){
                Station *station = &stationArr[i];
                int index = 0;
                for (int j =0; j<numParams;j++){
                    if(objparamArr[j].layer == msg_count){
                        index = j;
                        break;
                    }
                }
                *(station->values+index) = values[station->closestPoint]; 

            }

            free(lats);
            free(lons);
            free(values);
        }
        codes_handle_delete(h);

    }
    fclose(f);
    free(args);
    pthread_exit(NULL);

}

static bool indexofClosestPoint(double* lats, double* lons, float stationLat, float stationLon, int i, int prevClosestidx){
    bool isCloser = (pow((lats[i] - stationLat), 2) + pow((lons[i]-stationLon), 2)) <= (pow((lats[prevClosestidx] - stationLat), 2) + pow((lons[prevClosestidx] - stationLon), 2));
    return isCloser;
}

void mapData(string date, string hour){
    string hourlyDate = date + hour;
    for (int i = 0; i<numStations; i++){
        Station *station = &stationArr[i];
        station->dataMap.insert({hourlyDate, station->values});
    }

}
