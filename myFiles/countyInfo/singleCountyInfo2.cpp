/*
This implementation is built on top of threadedExtractWRFData.cpp. I will be scraping 
the files included in the "countyInfo" directory in order to extract the WRF data for a 
given date range across all counties in the United States. 


Compile:
g++ -Wall -threadedExtractWRFData1.cpp -leccodes -lpthread

*/

#include <stdio.h>
#include <stdlib.h>
#include <filesystem>
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
using namespace std;

/* Timing variables */
struct timespec startTotal;
struct timespec endTotal;
double totalTime;

vector<int> beginDay = {2020, 1, 1}; // arrays for the begin days and end days. END DAY IS NOT INCLUSIVE.  
                                     // when passing a single day, pass the day after beginDay for endDay
                                     // FORMAT: {yyyy, mm, dd}
vector<int> endDay = {2020, 1, 8};   // NOT INCLUSIVE

vector<int> arrHourRange = {0,23}; // array for the range of hours one would like to extract from
                                   // FORMAT: {hh, hh} where the first hour is the lower hour, second is the higher
                                   // accepts hours from 0 to 23 (IS INCLUSIVE)

int intHourRange; 

string filePath = "/home/kalebg/Desktop/weekInputData/";  // path to "data" folder. File expects structure to be: 
                                        // .../data/<year>/<yyyyMMdd>/hrrr.<yyyyMMdd>.<hh>.00.grib2
                                        // for every hour of every day included. be sure to include '/' at end

string writePath = "/home/kalebg/Desktop/WRFDataThreaded/"; // path to write the extracted data to,
                                                    // point at a WRFData folder
string repositoryPath = "/home/kalebg/Documents/GitHub/customExtraction/";//PATH OF THE CURRENT REPOSITORY
                                                                          // important when passing args                                                    

// Structure for holding the selected station data. Default will be the 5 included in the acadmeic
// paper: "Regional Weather Forecasting via Neural Networks with Near-Surface Observational and Atmospheric Numerical Data."
struct Station{
    string name = "00";
    string state;
    string county;
    string fipsCode;
    float lat;
    float lon;
    double **values; // holds the values of the parameters. Index of this array will 
                    // correspond to index if the Parameter array. This will be a single hour's data
    map<string, vector<double>> dataMap; // this will hold the output data. structured as {"yyyymmddHH" : [param1, param2, ...]}}
    // hold the average output data for a full day, {"yyyymmdd": [averagedData]}
    map<string, vector<double>> dailydatamap;
    // hold the average output data for a full week, {"yyyymmdd - yyyymmdd" : [averagedData]}
    map<string, vector<string>> weeklydatamap;
    int *closestPoint; // index in the grib file of the point closest to the station's lats and lons
    
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
    int threadIndex;
    string hour;
    vector<string> strCurrentDay;
};
struct writeThreadArgs{ //this is clumsily implemented, when making changes, keep in mind
                        // that this structure also needs to be changed in the writeData func
    Station* station;
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
string *hours;

/*Initialization of semaphores to be used for protection*/
sem_t hProtection; // protect when creating the handle object for each layer
sem_t *mapProtection; // protect when writing to the maps
sem_t pathCreationSem; // protect when writeData makes new file paths
sem_t *valuesProtection; // protect when writing values to the values array
sem_t writeProtection; // only one thread should be able to write to a file at a time

// function to handle arguments passed. Will either build the Station array and/or paramter array
// based off of the arguments passed or will build them with default values. Will also construct
// the hour array based on passed beginning and end hours
void handleInput(int, char**);

// Function to build the default station array (all counties in continential US)
// through reading from the files in the countyInfo file
void defaultStations(); void readCountycsv(); void matchState();
// similar to default station, but for the parameter arrays
void defaultParams();

void buildHours();// builds the hour arrays given the hour range specified

void semaphoreInit(); // initialize all semaphores used

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


/* function to map the data in the station's values array to the station's map */
void mapData(string, string, int);

/*function to take the weekly average of the data and place into a new map*/
void mapWeeklyData();

/*Function to find the standard deviation of the values for each week,
takes an array of the values for the week and their average, outputs a stdDev*/
static double standardDev(vector<double>, double&);
/*Function to create the paths and files for the maps to be written to*/
void createPath();

/* function to write the data in the staion maps to a .csv file */
void writeData(void*);

void garbageCollection();

int main(int argc, char*argv[]){


    clock_gettime(CLOCK_MONOTONIC, &startTotal);

    handleInput(argc, argv);

    semaphoreInit();
    
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
        pthread_t *threads = (pthread_t*)malloc(intHourRange * sizeof(pthread_t)); // will be freed at the end of this iteration
        if(!threads){
            fprintf(stderr, "Error: unable to allocate %ld bytes for threads.\n", (long)(intHourRange*sizeof(pthread_t)));
            exit(0);
        }
        
         
        FILE* f[intHourRange]; // use to open the file for each hour
        threadArgs *arrThreadArgs = new threadArgs[intHourRange];
        

        int threaderr; // keep track if the threading runs into an error
        for(int i=0;i<intHourRange;i++){ // for each hour, thread the file and filename
            f[i] = NULL;
            string hour = hours[i];
            string fileName = "hrrr."+strCurrentDay.at(3)+"."+hour+".00.grib2";
            string filePath2 = filePath1 + fileName;
            
            arrThreadArgs[i].f = f[i];
            arrThreadArgs[i].fileName = fileName;
            arrThreadArgs[i].pathName = filePath2;
            arrThreadArgs[i].threadIndex = i;
            arrThreadArgs[i].hour = hour;
            arrThreadArgs[i].strCurrentDay = strCurrentDay;            
            threaderr = pthread_create(&threads[i], NULL, &readData, &arrThreadArgs[i]);
            if(threaderr){
                assert(0);
                return 1;
            }

        }
        for(int i=0;i<intHourRange;i++){
            pthread_join(threads[i], NULL);
        }
        std::free(threads);
        intcurrentDay = getNextDay(intcurrentDay);
        delete [] arrThreadArgs;
    }

    mapWeeklyData();
    // the data maps are finished being built, now its time to write the maps to csv files
    // since each station has their own map, we can thread the stations without having to 
    // use semaphores. 
    createPath();
    
    writeThreadArgs *arg = new writeThreadArgs[numStations];
    int threaderr;
    for(int i=0; i<numStations;i++){
        arg[i].station = &stationArr[i];
        writeData(&arg[i]);
    }
    
    delete [] arg;



    garbageCollection();
    clock_gettime(CLOCK_MONOTONIC, &endTotal);
    totalTime = (endTotal.tv_sec - startTotal.tv_sec) * 1000.0;
    totalTime+= (endTotal.tv_nsec - startTotal.tv_nsec) / 1000000.0;
    printf("\n\nRuntime in ms:: %f\n", totalTime);


    return 0;
 }

void handleInput(int argc, char* argv[]){
    
    // NOT FINISHED. DO NOT PASS ARGS
    if(argc > 1){
        // check to see if the correct arguments have been passed to fill the parameter and/or 
        // station arrays
        int fipsindex=0;
        bool fipsflag = false;
        for(int i=0; i<argc;i++){
            if(strcmp(argv[i], "--fips") == 0){
                fipsflag = true;
                fipsindex = i;
                break;
            }
        }
        vector<string> fipscodes(0);
        if(fipsflag){
            for(int i=fipsflag+1;i<argc;i++){
                string arg = argv[i];
                if (arg.length() == 5){
                    fipscodes.insert(fipscodes.begin(), argv[i]);
                }
                else{
                    cout << "Invalid length of fips argument passed" << endl;
                    exit(0);
                }
            }

            
            // buildHours();
            // defaultParams();
            // readCountycsv();
            // matchState();
        }else{
            cout << "Incorrect arguments passed" << endl;
            exit(0);
        }

        // now call the cmd line to run the python file
        string command; int status;
        for(int i=0;i<fipscodes.size();i++){
            string command = "cd " + repositoryPath + "/myFiles/countyInfo/sentinel-hub ; python3 geo_gridedit_1-5-23.py --fips " +
                            fipscodes.at(i);
            status = system(command.c_str());
            if(status==-1) std::cerr << "Python call error: " <<strerror(errno) << endl;

        }
        buildHours();
        defaultParams();
        // defaultStations();
        readCountycsv();
        matchState();
        
    }
    else{
        buildHours();
        defaultParams();
        // defaultStations();
        readCountycsv();
        matchState();
        // potential improvement: for each month, run the file and make new parameter array based off
        //                        of what the file returns
        // // print out to make sure I did it right
        // for(int i=0;i<numStations;i++){
        //     cout << stationArr[i].state << " " << stationArr[i].county << " " << stationArr[i].fipsCode;
        //     cout << " " << stationArr[i].lat << endl;
        // }
    }
}

void buildHours(){
    // build the hour array
    // make sure correct values have been passed to the hour array 
    try{
        intHourRange = arrHourRange.at(1) - arrHourRange.at(0)+1;
        if(intHourRange < 1) throw(intHourRange);
    }catch(exception e){
        fprintf(stderr, "Error, problems with hour range.");
        exit(0);
    }

    hours = (string*)malloc(intHourRange*sizeof(string));
    if(!hours){
        fprintf(stderr, "Error, unable to allocate hours");
        exit(0);
    }
    
    int endHour = arrHourRange.at(1);
    int beginHour = arrHourRange.at(0);
    int index=0;
    for(int hour = beginHour; hour<=endHour;hour++){
        if(hour < 10){ // put a 0 in front then insert in the hours arr
            string strHour = "0"+to_string(hour);
            hours[index++] = strHour;
        }
        else{ // just convert to a string and insert into the hours arr
            string strHour = to_string(hour);
            hours[index++] = strHour;
        }
    }
}

void defaultStations(){
    struct coordinates{
        float lat; float lon;
    };
    map<string, string> stateMap;
    map<string, string> countyMap;
    map<string, coordinates> coordinatesMap;

    string strcountyFipsCodes;
    ifstream filecountyFipsCodes;
    filecountyFipsCodes.open("./countyFipsCodes.txt");
    if(!filecountyFipsCodes){
        cerr << "Error: the FIPS file could not be opened.\n";
        exit(1);
    }
    bool readstates = false;
    bool readcounties = false;

    string strCountyInfo;
    string strStateInfo;
    while(getline(filecountyFipsCodes, strcountyFipsCodes)){
        // the line to start reading the county info
        if(strcmp(strcountyFipsCodes.c_str(), " ------------    --------------")==0){
            readcounties = true;
            continue; // continue to the next line
        }
        // line to stop reading states
        if(strcmp(strcountyFipsCodes.c_str(), " county-level      place")==0){
            readstates = false;
        }
        // line to start reading states
        if(strcmp(strcountyFipsCodes.c_str(), "   -----------   -------")==0){
            readstates = true;
            continue;
        }

        // load all states and their fips info into an obj
        if(readstates){
            // store fips code as the key, store statename as value
            if(strcountyFipsCodes.length() < 3){
                // do nothing, this is not a necessary line
                continue;
            }else{
                string delimiter = ",";
                string strfips = strcountyFipsCodes.substr(0,strcountyFipsCodes.find(delimiter));
                string strState = strcountyFipsCodes.erase(0, strcountyFipsCodes.find(delimiter)+delimiter.length());

                stateMap.insert({strfips, strState});
            }
        }

        // load county information into the map
        if(readcounties){
            string delimiter = ",";
            string strFips = strcountyFipsCodes.substr(0,strcountyFipsCodes.find(delimiter));
            string strCountyName = strcountyFipsCodes.erase(0, strcountyFipsCodes.find(delimiter)+delimiter.length());

            countyMap.insert({strFips, strCountyName}); 
        }
    }
    filecountyFipsCodes.close();

    // load the counties and their coordinates into the respective maps
    string strLine;
    ifstream fileCoordinates;
    fileCoordinates.open("./countyFipsandCoordinates.csv");
    if(!fileCoordinates){
        cerr << "Error: the COORDINATES file could not be opened.\n";
        exit(1);
    }
    vector<string> row;
    bool firstLine = true; // skip the header
    while(getline(fileCoordinates, strLine)){
        row.clear();
        if(strLine.length() > 100) continue; // link to gitHub of csv
        if(firstLine){
            firstLine = false;
            continue;
        }
        stringstream s(strLine);
        string currLine;
        while(getline(s, currLine, ',')){
            row.push_back(currLine);
        }
        if(row.size() > 2){
            // row[0] = fips, row[1] = county, row[2] = longitude, row[3] = latitude
            coordinates c; c.lat = stof(row.at(3)), c.lon = stof(row.at(2));
            coordinatesMap.insert({row.at(0), c});
        }
    }
    fileCoordinates.close();

    // match the counties to their respective states by matching fips codes
    // throw out unneeded counties (02, 15, 72)
    int tmpNumStations = 3233; // max value, will change after knowing full size
    Station tmpStationArr[tmpNumStations];
    int arridx = 0;
    map<string, string>::iterator stateItr;
    map<string, coordinates>::iterator coordinatesItr;
    for(auto itr = countyMap.begin(); itr!=countyMap.end(); ++itr){
        string countyFips = itr->first;
        string stateFips = countyFips.substr(0,2);
        stateItr = stateMap.find(stateFips);
        coordinatesItr = coordinatesMap.find(countyFips);
        coordinates c = coordinatesItr->second;

        if(stateItr==stateMap.end()){
            cout << "when building default stationArr, state was not found in map" << endl;
            exit(0);
        }else if(stateItr->first == "02" || stateItr->first == "15" || stateItr->first == "72"){
            // alaska, hawaii, and puerto rico, do not include
            continue;
        }
        Station st;
        st.fipsCode = countyFips;
        st.name = itr->second;
        st.county = itr->second;
        st.state = stateItr->second;
        st.lat = c.lat;
        st.lon = c.lon;
        tmpStationArr[arridx] = st;
        arridx++;
    }
    // numStations will be arridx+1 since we're starting at 0
    numStations = arridx;

    // map the temp array to the global array
    stationArr = new Station[numStations];
    int itr=0;
    for(int i=0;i<numStations;i++){
        *(stationArr+i) = tmpStationArr[i];
    }

    // initialize the pointer array for each station to be of the length of the number of params
    // for each station, allocate some memory for each values array
    // **values = size of hour range
    // *values[i] = size of numparams
    for (int i=0; i<numStations;i++){
        //ALSO: initialize closestPoint array
        stationArr[i].closestPoint = new int[intHourRange];
        stationArr[i].values = new double*[intHourRange];
        for(int j=0; j<intHourRange; j++){
            stationArr[i].values[j] = new double[numParams];
        }
    }

}

void readCountycsv(){
    
    map<string, Station> tmpStationMap;
    string strLine;
    ifstream filewrfdata;
    filewrfdata.open("./WRFoutput/wrfOutput.csv");
    if(!filewrfdata){
        cerr << "Error: the WRFOUTPUT file could not be opened.\n";
        exit(1);
    }
    vector<string> row;
    bool firstLine = true;
    int arridx = 0;
    while(getline(filewrfdata, strLine)){
        row.clear();
        if(firstLine){
            firstLine = false;
            continue;
        }
        stringstream s(strLine);
        string currLine;
        while(getline(s, currLine, ',')){
            row.push_back(currLine);
        }
        // row[0] = indx, row[1] = fips, row[2] = statefips, row[3] = county, 4 = lat, 5 = lon
        Station st; st.name = row.at(0); st.fipsCode = row.at(1); st.county = "\""+row.at(3)+"\"";
        st.lat = stof(row.at(4)); st.lon = stof(row.at(5));
        tmpStationMap.insert({row.at(0), st});
        arridx++;
    }
    filewrfdata.close();
    numStations = arridx;
    stationArr = new Station[numStations];
    int i = 0;
    for(auto itr = tmpStationMap.begin(); itr!=tmpStationMap.end(); ++itr){
        *(stationArr+i) = itr->second;
        i++;
    }

    // initialize the pointer array for each station to be of the length of the number of params
    // for each station, allocate some memory for each values array
    // **values = size of hour range
    // *values[i] = size of numparams
    for (int i=0; i<numStations;i++){
        //ALSO: initialize closestPoint array
        stationArr[i].closestPoint = new int[intHourRange];
        stationArr[i].values = new double*[intHourRange];
        for(int j=0; j<intHourRange; j++){
            stationArr[i].values[j] = new double[numParams];
        }
    }
}

void matchState(){
    // all the points will have the same state, so search through the file
    // that has the state fips and put it as the state for each of the 
    // stations
    map<string, string> stateMap;
    string strcountyfipscodes;
    ifstream filecountyfipscodes;
    filecountyfipscodes.open("./countyFipsCodes.txt");
    if(!filecountyfipscodes){
        cerr << "Error: the FIPS file could not be opened.\n";
        exit(1);
    }
    bool readstates = false;
    string strStateInfo;
    while(getline(filecountyfipscodes, strcountyfipscodes)){
        if(strcmp(strcountyfipscodes.c_str(), " county-level      place")==0){
            readstates = false;
            break;
        }
        // line to start reading states
        if(strcmp(strcountyfipscodes.c_str(), "   -----------   -------")==0){
            readstates = true;
            continue;
        }
        // load all states and their fips info into an obj
        if(readstates){
            // store fips code as the key, store statename as value
            if(strcountyfipscodes.length() < 3){
                // do nothing, this is not a necessary line
                continue;
            }else{
                string delimiter = ",";
                string strfips = strcountyfipscodes.substr(0,strcountyfipscodes.find(delimiter));
                string strState = strcountyfipscodes.erase(0, strcountyfipscodes.find(delimiter)+delimiter.length());

                stateMap.insert({strfips, strState});
            }
        }
    }
    filecountyfipscodes.close();

    // loop through the station array and match the fips code to the respective state
    map<string,string>::iterator stateItr;
    for(int i=0;i<numStations;i++){
        string fips = stationArr[i].fipsCode;
        string stateFips = fips.substr(0,2);
        stateItr = stateMap.find(stateFips);

        // insert the state into the objs
        stationArr[i].state = stateItr->second;
    }

}

void defaultParams(){
    map<int, Parameter> paramMap;
    string paramline;
    ifstream paramFile;
    paramFile.open("./parameterInfo.csv");
    if(!paramFile){
        cerr << "Error: the PARAMETER file could not be opened.\n";
        exit(1);
    }
    vector<string> paramRow;
    bool firstline = true;
    int countParams = 0;
    while(getline(paramFile, paramline)){
        paramRow.clear();
        if(firstline){ // do not include the header
            firstline = false;
            continue;
        }
        stringstream s(paramline);
        string currLine;
        while(getline(s, currLine, ',')){
            paramRow.push_back(currLine);
        }
        if(paramRow.size() > 2){
            Parameter p; p.layer = stoi(paramRow.at(0)); p.name = paramRow.at(1);
            p.units = paramRow.at(2);
            countParams++;
            paramMap.insert({stoi(paramRow.at(0)), p});
        }
    }
    numParams = countParams;
    objparamArr = new Parameter[numParams];
    int count =0;
    for(auto itr = paramMap.begin(); itr!=paramMap.end(); ++itr){
        *(objparamArr + count) = itr->second;
        count++;
    }

    // build the boolean param array
    int layer =0;
    for (int i = 0; i<numParams; i++){
        layer = objparamArr[i].layer;
        blnParamArr[layer] = true;
    }    
}

void semaphoreInit(){
    mapProtection = (sem_t*)malloc(sizeof(sem_t)*numStations);
    for(int i=0; i< numStations; i++){
        if(sem_init(&mapProtection[i], 0, 1)==-1){
            perror("sem_init");
            exit(EXIT_FAILURE);
        }
    }
    if(sem_init(&pathCreationSem, 0, 1)==-1){
        perror("sem_init");
        exit(EXIT_FAILURE);
    }
    if(sem_init(&hProtection, 0, 1) == -1){
        perror("sem_init");
        exit(EXIT_FAILURE);
    }
    valuesProtection = (sem_t*)malloc(sizeof(sem_t)*numStations);
    for(int i =0;i<numStations;i++){
        if(sem_init(&valuesProtection[i], 0, 1)==-1){
            perror("sem_init");
            exit(EXIT_FAILURE);
        }
    }
    if(sem_init(&writeProtection, 0, 1) == -1){
        perror("sem_init");
        exit(EXIT_FAILURE);
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
    // struct threadArgs threadArg = *(struct threadArgs*)args;
    struct threadArgs *threadArg = (struct threadArgs*)args;
    FILE*f =(*threadArg).f;
    string filePath2 = (*threadArg).pathName;
    string fileName = (*threadArg).fileName;
    int threadIndex = (*threadArg).threadIndex;
    string hour = (*threadArg).hour;
    vector<string> strCurrentDay = (*threadArg).strCurrentDay;

    printf("\nOpening File: %s", filePath2.c_str());

    //try to open the file
    try{
        f = fopen(filePath2.c_str(), "rb");
        if(!f) throw(filePath2);
    }
    catch(string file){
        printf("\nError: could not open filename %s in directory %s", fileName.c_str(), file.c_str());
        exit(0);
    }
    // unsigned long key_iterator_filter_flags = CODES_KEYS_ITERATOR_ALL_KEYS |
    //                                           CODES_KEYS_ITERATOR_SKIP_DUPLICATES;
    codes_grib_multi_support_on(NULL);

    codes_handle * h = NULL; // use to unpack each layer of the file

    const double missing = 1.0e36;        // placeholder for when the value cannot be found in the grib file
    int msg_count = 0;  // KEY: will match with the layer of the passed parameters
    int err = 0;

    // char value[MAX_VAL_LEN];
    // size_t vlen = MAX_VAL_LEN;

    bool flag = true; 

    long numberOfPoints=0;

    double *lats, *lons, *values; // lats, lons, and values returned from extracted grib file

    sem_wait(&hProtection);
    while((h=codes_handle_new_from_file(0, f, PRODUCT_GRIB, &err))!=NULL) // loop through every layer of the file
    {
        sem_post(&hProtection);
        assert(h);
        msg_count++; // will be in layer 1 on the first run
        if (blnParamArr[msg_count] == true){

        
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
                std::free(lats);
                exit(0);
            }
            values = (double*)malloc(numberOfPoints * sizeof(double));
            if(!values){
                fprintf(stderr, "Error: unable to allocate %ld bytes\n", (long)(numberOfPoints * sizeof(double)));
                std::free(lats);
                std::free(lons);
                exit(0);
            }
            CODES_CHECK(codes_grib_get_data(h, lats,lons,values), 0);
            // all the lats,lons, and values from this layer of the grib file are now stored, now match them
            // to their respective stations


            // if it is the first time, extract the index
            if (flag == true){
                
                for(int i =0; i<numStations; i++){
                    Station *station = &stationArr[i];
                    int closestPoint = 0;
                    for (int j=0; j<numberOfPoints;j++){
                        double distance = pow(lats[j]- station->lat, 2) + pow(lons[j]-station->lon, 2);
                        double closestDistance = pow(lats[closestPoint] - station->lat, 2) + pow(lons[closestPoint] - station->lon,2);
                        if(distance < closestDistance) closestPoint = j;
                    }
                    station->closestPoint[threadIndex] = closestPoint;
                }
                flag = false;
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
                double dblCurrentValue = values[station->closestPoint[threadIndex]];
                sem_wait(&valuesProtection[i]);
                station->values[threadIndex][index] = dblCurrentValue;
                sem_post(&valuesProtection[i]);
            }

            std::free(lats);
            std::free(lons);
            std::free(values);
            
        }
        codes_handle_delete(h);

    }
    sem_post(&hProtection);
    fclose(f);
    // call the mapData function to map the hour's parameter's to each station's map
    mapData(strCurrentDay.at(3), hour, threadIndex);
    pthread_exit(0);

}

void mapData(string date, string hour, int threadIndex){
    string hourlyDate = date + hour;
    for (int i = 0; i<numStations; i++){
        Station *station = &stationArr[i];
        // we can't directly insert the pointer to the data into the dataMap, so we need to insert a copy of the array
        std::vector<double> valuesCpy(numParams);
        
        //semaphore to protect reading the values over into the copy
        sem_wait(&valuesProtection[i]);
        double *valuesCpy_1 = station->values[threadIndex];
        for(int j=0; j<numParams; j++){
            valuesCpy[j] = *(valuesCpy_1+j);
        }
        sem_post(&valuesProtection[i]);

        sem_wait(&mapProtection[i]);
        station->dataMap.insert({hourlyDate, valuesCpy});
        sem_post(&mapProtection[i]);
    }
}

void mapWeeklyData(){



    for(int i=0;i<numStations;i++){

        // loop through the days to find the weekly average
        Station*station = &stationArr[i];
        map<string, vector<double>> datamap = station->dataMap;
        map<string, vector<double>> tmpMap; // stores the elements before the averages are found
        map<string, vector<double>> *dailyDataMap = &station->dailydatamap;
        map<string, vector<string>> *weeklyDataMap = &station->weeklydatamap;
        map<string, vector<string>> weeklyAverages; // stores the averages
        int expectedelements =0;
        
        // before finding the weekly averages, find the daily averages
        for(auto itr = datamap.begin(); itr!=datamap.end(); ++itr){
            expectedelements++;
            string day = itr->first.substr(0,8);
            int hour = stoi(itr->first.substr(8,2));

            if(hour >= arrHourRange.at(1)){
                tmpMap.insert({to_string(hour), itr->second});
                
                vector<double> vctrAverges(numParams);
                for(auto tmpItr=tmpMap.begin(); tmpItr!=tmpMap.end();++tmpItr){
                    for(int j=0; j<numParams;j++){
                        vctrAverges[j]+=tmpItr->second[j];
                    }
                }
                //found the summation of all params, now find the averages
                for(int j=0;j<numParams;j++){
                    vctrAverges[j] = vctrAverges[j] / tmpMap.size();
                }
                dailyDataMap->insert({day, vctrAverges});

                tmpMap.clear();
            }else{
                tmpMap.insert({to_string(hour), itr->second});
            }
        }
        
        //  paramLayer, all layer's elements, used to find stdDev
        map<int, vector<double>> elements; 
        for(int j=0; j<numParams; j++){
            vector<double> vctrElements(0); // make the vector of the size number of expected elements
            elements.insert({j, vctrElements});
        }
        
        string lastDay = datamap.rbegin()->first.substr(0,8);
        int lastHour = stoi(datamap.rbegin()->first.substr(8,2));
        int firstHour = arrHourRange.at(0);
        string firstdayofweek;//,previousDay;
        int dayItr = 0; // when this hits 6, you're on the final day of the week
        for(auto itr = datamap.begin(); itr!=datamap.end();++itr){
            
            string day = itr->first.substr(0,8);
            int hour = stoi(itr->first.substr(8,2));
            if(dayItr == 0){
                firstdayofweek = day;
            }

            if((dayItr==6 && hour == lastHour)|| (day == lastDay && hour == lastHour)){
                tmpMap.insert({day, itr->second});
                // loop through all values in itr->second and append them to their elements map
                for(auto j=0; j<itr->second.size(); j++){
                    auto nodehandler = elements.find(j);
                    if(nodehandler != elements.end()){
                        // insert the element into the element map's vector
                        vector<double> vctrElem = nodehandler->second;
                        vctrElem.insert(vctrElem.begin(),itr->second.at(j));
                        nodehandler->second = vctrElem; // insert the vector back into the map
                    }else{
                        std::cerr << "\nError: unable to find key in map." << endl;
                    }
                }
                // loop through the elements map, find stdDev and average
                vector<string> vctrAverages(0);
                for(auto elemItr = elements.rbegin(); elemItr!=elements.rend(); ++elemItr){
                    // double*ptrmean;
                    vector<double> thoseElementsIneed = elemItr->second;
                    double mean = 0.0;
                    double &refmean = mean;
                    double stdDev = standardDev(thoseElementsIneed, refmean);
                    // double mean = *ptrmean;
                    string strval = std::to_string(mean) + "+-" + std::to_string(stdDev);
                    vctrAverages.insert(vctrAverages.begin(), strval);
                }
                // place the averages in the map
                string weekRange = firstdayofweek + "-" + day;
                weeklyAverages.insert({weekRange,vctrAverages});

                // wipe the tmpMap
                tmpMap.clear();
                // reset the dayitr
                dayItr = 0;
            }
            else{
                // append the values and increment the dayitr
                tmpMap.insert({day, itr->second});
                for(auto j=0; j<itr->second.size(); j++){
                    auto nodehandler = elements.find(j);
                    if(nodehandler != elements.end()){
                        // insert the element into the element map's vector
                        vector<double> vctrElem = nodehandler->second;
                        vctrElem.insert(vctrElem.begin(),itr->second.at(j));
                        // insert the vector back into the map
                        nodehandler->second = vctrElem;
                    }else{
                        std::cerr << "\nError: unable to find key in map." << endl;
                    }
                }
                if(hour == lastHour){
                    dayItr++;
                }
            }
            
            
        }
        *weeklyDataMap = weeklyAverages;
    }
}
static double standardDev(vector<double> elements, double&refmean){
    double stdDev = 0, sum=0, mean=0;
    
    for(int i=0;i<elements.size();i++){
        sum+=elements.at(i);
    }
    refmean = sum / elements.size();
    for(int i=0; i<elements.size(); i++){
        stdDev += pow(elements.at(i) - refmean, 2);
    }
    stdDev = sqrt(stdDev / elements.size());
    if(isnan(refmean)){
        std::cerr << "\nError: NaN value in standard deviation calculation" << endl;
    }
    // refmean = &mean;
    return stdDev;
}

void createPath(){
    string strcmd; int status;
    if(!dirExists(writePath)){
            strcmd = "mkdir -p "+ writePath;
            status = system(strcmd.c_str());
            if(status==-1) std::cerr << "writepatherrorrror: " << strerror(errno) << endl;
            
    }
    
    // for each of the years passed, create a year folder
    // if the begin date and end date have different years, create multiple folders
    // create the begin year folder
    strcmd = "cd " + writePath + "; mkdir "+to_string(beginDay.at(0));
    status = system(strcmd.c_str());
    if(status==-1) std::cerr << "writepatherrorrror: " << strerror(errno) << endl;
    
    if(beginDay.at(0) != endDay.at(0)){
        int years = endDay.at(0) - beginDay.at(0);
        int currYear = beginDay.at(0)+1;
        do{
            // create a folder before the current year reaches the
            // end year 
            strcmd = "cd " + writePath + "; mkdir "+to_string(currYear);
            status = system(strcmd.c_str());
            if(status==-1) std::cerr << "writepatherrorrror: " << strerror(errno) << endl;
            currYear++;

        }while(currYear!= endDay.at(0));
    }
    // for each station, if this is the first time seeing its state or county,
    // make a directory and make one for the particular station as well
    string currPath;
    for (const auto &entry : std::filesystem::directory_iterator(writePath)){
        currPath = entry.path();

        vector<string> vctrCounties(0);
        vector<string> vctrStates(0);
        for(int i=0;i<numStations;i++){
            Station*station = &stationArr[i];

            if(std::find(vctrStates.begin(), vctrStates.end(), station->state)==vctrStates.end()){
                vctrStates.insert(vctrStates.begin(), station->state);
                strcmd = "cd " + currPath + "; mkdir "+station->state;
                status = system(strcmd.c_str());
                if(status==-1) std::cerr << "writepatherrorrror: " << strerror(errno) << endl;

                if(std::find(vctrCounties.begin(), vctrCounties.end(), station->county)==vctrCounties.end()){
                    vctrCounties.insert(vctrCounties.begin(), station->county);
                    strcmd = "cd " + currPath + "/" + station->state + "; mkdir "+station->county;
                    status = system(strcmd.c_str());
                    if(status==-1) std::cerr << "writepatherrorrror: " << strerror(errno) << endl;
                }
            }

            // after the directory for the state and county are made, then go into it and make the directory
            // for the individual station
            string stationPath = currPath + "/" + station->state + "/" + station->county;
            strcmd = "cd " + stationPath + "; mkdir " + station->name;
            status = system(strcmd.c_str());
            if(status==-1) std::cerr << "writepatherrorrror: " << strerror(errno) << endl;
        }
    }

}

void writeData(void*arg){

    struct writeThreadArgs writeThreadArgs = *(struct writeThreadArgs*)arg;

    Station*station = (writeThreadArgs).station;

    // the entire file will be written to in one iteration. We will append the strings
    // of all days together until the next day is reached, at which point we will write out
    string output, filePath, fileName, year, county, state, name, strcmd;
    int status;
    // loop through each key of the file, every time you run into a new day, 
    // make a new file
    for(auto itr = station->weeklydatamap.begin(); itr!=station->weeklydatamap.end();++itr){
        string weekrange = itr->first;
        year = weekrange.substr(0,4);
        county = station->county;
        state = station->state;
        name = station->name;
        
        filePath = writePath + year + "/" + state + "/" + county + "/" + name + "/";
        fileName = weekrange + ".csv";
        // strcmd = "cd " + filePath + "; touch " +fileName;
        // status = system(strcmd.c_str());
        // if(status==-1) std::cerr << "\nFile Creation Error: " << strerror(errno) << endl;

        strcmd = "cd " + filePath + "; echo \"Year, Month, Day, State, County, Fips Code,";
        // append the name of each parameter to the headings of the files
        for(int j=0;j<numParams;j++) strcmd += objparamArr[j].name + " ( " + objparamArr[j].units + "),";
        
        strcmd += "\" > "+ fileName;
        status = system(strcmd.c_str());
        if(status==-1) std::cerr << "\n Write to file error: " << strerror(errno) << endl;
        
    
        // append information to the output string
        // send each day of the week's data to the output file
        string firstdayoftheweek = itr->first.substr(0,8);
        string lastdayoftheweek = itr->first.substr(9,8);
        string fulldaymapday,daymapyear,daymapmonth,daymapday;
        output = "";
        for(auto dayitr = station->dailydatamap.begin(); dayitr != station->dailydatamap.end(); ++dayitr){
            fulldaymapday = dayitr->first;

            if(fulldaymapday <= lastdayoftheweek && fulldaymapday >= firstdayoftheweek){
                daymapyear = dayitr->first.substr(0,4);
                daymapmonth = dayitr->first.substr(4,2);
                daymapday = dayitr->first.substr(6,2);
                output.append(daymapyear +","+ daymapmonth +","+ daymapday + "," + station->state + "," + station->county + "," + station->fipsCode + ",");

                // loop through the vector associated with the map and append to output
                for (auto j=0;j<dayitr->second.size();j++){
                    output.append(to_string(dayitr->second.at(j))+",");
                }
                // output.append("\n"
                // send the output string as a line to the file
                strcmd = "cd " + filePath + "; echo " + "\"" + output + "\" >> " + fileName;
                status = system(strcmd.c_str());
                if(status == -1) std::cerr << "\nDaily File Write Error: " << strerror(errno) << endl;
                output = "";
                
            }

        }
        output = "Weekly Averages, , , , ,"+station->fipsCode+",";
        for(auto j=0; j<itr->second.size();j++){
            output.append((itr->second.at(j))+",");
        }
        strcmd = "cd " + filePath + "; echo " + "\"" + output + "\" >> " + fileName;
        status = system(strcmd.c_str());
        if(status == -1) std::cerr << "\nDaily File Write Error: " << strerror(errno) << endl;
        output = "";
       
    }

}

void garbageCollection(){
    for(int i =0; i<numStations;i++){
        // delete each station's closest point array
        delete [] stationArr[i].closestPoint;
        // delete each station's value array
        for(int j=0; j<intHourRange;j++){
            delete [] stationArr[i].values[j];
        }
        delete [] stationArr[i].values;
    }
    delete [] objparamArr;
    delete [] stationArr;
    std::free(hours);
    if(sem_destroy(&hProtection)==-1){
        perror("sem_destroy");
        exit(EXIT_FAILURE);
    }
    if(sem_destroy(&pathCreationSem)==-1){
        perror("sem_destroy");
        exit(EXIT_FAILURE);
    }
    if(sem_destroy(&writeProtection)==-1){
        perror("sem_destroy");
        exit(EXIT_FAILURE);
    }
    free(mapProtection);
    for (int i =0; i<numStations; i++){
        sem_destroy(&mapProtection[i]);
    }

}