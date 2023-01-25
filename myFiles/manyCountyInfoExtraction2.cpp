/*
NOTE:

TODO: for every call to a file, make sure there is either an absolute path or the path
is configured by the passed "repository path"

Before running, please configure beginday, endday, arrhourrange, filePath, writepath, 
repositoryPath

If g++ version is outdated compile with flag -std=c++17


This implementation is based on the first iteration of manyCountyInfo, except I will call
to python to analyze the data rather than doing it in C


Compile:
g++ -Wall -threadedExtractWRFData1.cpp -leccodes -lpthread

*/

// #ifndef __has_include
//   static_assert(false, "__has_include not supported");
// #else
// #  if __cplusplus >= 201703L && __has_include(<filesystem>)
// #    include <filesystem>
//      namespace fs = std::filesystem;
// #  elif __has_include(<experimental/filesystem>)
// #    include <experimental/filesystem>
//      namespace fs = std::experimental::filesystem;
// #  elif __has_include(<boost/filesystem.hpp>)
// #    include <boost/filesystem.hpp>
//      namespace fs = boost::filesystem;
// #  endif
// #endif
#include <filesystem>
// namespace fs = std::filesystem;
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

using namespace std;
/* Timing variables */
struct timespec startTotal;
struct timespec endTotal;
double totalTime;

vector<int> beginDay = {2021, 6, 1}; // arrays for the begin days and end days. END DAY IS NOT INCLUSIVE.  
                                     // when passing a single day, pass the day after beginDay for endDay
                                     // FORMAT: {yyyy, mm, dd}
vector<int> endDay = {2021, 6, 2};   // NOT INCLUSIVEe

vector<int> arrHourRange = {0,2}; // array for the range of hours one would like to extract from
                                   // FORMAT: {hh, hh} where the first hour is the lower hour, second is the higher
                                   // accepts hours from 0 to 23 (IS INCLUSIVE)

int intHourRange; 

string filePath = "/home/kaleb/Desktop/weekInputData/";  // path to "data" folder. File expects structure to be: 
                                        // .../data/<year>/<yyyyMMdd>/hrrr.<yyyyMMdd>.<hh>.00.grib2
                                        // for every hour of every day included. be sure to include '/' at end

string writePath = "/home/kaleb/Desktop/WRFDataThreaded/"; // path to write the extracted data to,
                                                    // point at a WRFData folder
string repositoryPath = "/home/kaleb/Documents/GitHub/customExtraction/";//PATH OF THE CURRENT REPOSITORY
                                                                          // important when passing args                                                    

// Structure for holding the selected station data. Default will be the 5 included in the acadmeic
// paper: "Regional Weather Forecasting via Neural Networks with Near-Surface Observational and Atmospheric Numerical Data."
struct Station{
    string name = "00";
    string state;
    string stateAbbrev;
    string county;
    string fipsCode;
    float lat;
    float lon;
    double **values; // holds the values of the parameters. Index of this array will 
                    // correspond to index if the Parameter array. This will be a single hour's data
    int* closestPoint;

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
bool blnParamArr[200]; // TEMPORARY FIX
                        // this will be used to quickly index whether a parameter needs to be 
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
void getStateAbbreviations();
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

/* function to write the parameter information for a given day out to csv*/
void paramInfotoCsv(vector<string>);

/* function to read the data from a passed grib file */
void *readData(void*);


/* function to map the data in the station's values array to the station's map */
void mapData(string, string, int);

/*function to take the weekly average of the data and place into a new map*/
void mapMonthlyData();

/*Function to find the standard deviation of the values for each week,
takes an array of the values for the week and their average, outputs a stdDev*/
static double standardDev(vector<double>, double&);
/*Function to create the paths and files for the maps to be written to*/
void createPath();

/*functions to help create paths by finding length of string and splitting strnig on delimeter*/
int len(string); vector<string> splitonDelim(string, char);

/* function to write the data in the staion maps to a .csv file */
void writeData(void*);

void garbageCollection();

int main(int argc, char*argv[]){
    for (auto sum : blnParamArr){
        cout<< sum << endl;
    }
    exit();

    clock_gettime(CLOCK_MONOTONIC, &startTotal);

    // function calls to initialize everything
    handleInput(argc , argv);
    semaphoreInit();
    convertLatLons();

    // validate the dates passed
    bool boolDateRange = checkDateRange(beginDay, endDay);
    if (boolDateRange == false){
        fprintf(stderr, "Invalid Date Range\n");
        exit(1);
    }
    
    // before writing to anything, make sure all necessary directories exist
    createPath();
    vector<int> intcurrentDay = beginDay;
    // for each day
    while(checkDateRange(intcurrentDay, endDay)){
        vector<string> strCurrentDay = formatDay(intcurrentDay);

        // if the day is not the first or 14th day, skip it
        // string onlytheday = strCurrentDay.at(2);
        // if(!(onlytheday == "01" || onlytheday == "14")){
        //     intcurrentDay = getNextDay(intcurrentDay);
        //     continue;
        // }

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

        //TODO: After each day, write the values out to their respective file
        // CREATE THE PATH, then write to the file
        // or maybe create each path needed before reading

        intcurrentDay = getNextDay(intcurrentDay);

        delete [] arrThreadArgs;
    }
    delete [] blnParamArr;


    garbageCollection();
    clock_gettime(CLOCK_MONOTONIC, &endTotal);
    totalTime = (endTotal.tv_sec - startTotal.tv_sec) * 1000.0;
    totalTime+= (endTotal.tv_nsec - startTotal.tv_nsec) / 1000000.0;
    printf("\n\nRuntime in ms:: %f\n", totalTime);


    return 0;
 }

void handleInput(int argc, char* argv[]){
    // TODO: add a --param arg that allows the user to select one or multiple specific integer parameters
    // these parameters will be set in the bln param array // NEED to tell python about this
    vector<string> vctrstrBeginDay = formatDay(beginDay);
    // check to see if the correct arguments have been passed to fill the parameter and/or 
    // station arrays
    int fipsindex=0, begin_date_index=0, end_date_index=0, begin_hour_index=0, end_hour_index=0, param_index=0;
    bool fipsflag = false, begin_date_flag = false, end_date_flag = false, begin_hour_flag = false, end_hour_flag = false, param_flag=false;
    for(int i=0; i<argc;i++){
        if(strcmp(argv[i], "--fips") == 0){
            fipsflag = true;
            fipsindex = i;      
        }else if(strcmp(argv[i], "--help")==0 || strcmp(argv[i], "-h")==0){
            cout<<"####################### Help Mode ###########################\n";
            cout<<"This is a script to decompress grib files for a selected\n";
            cout<<"county region. The file read to know the counties is the \n";
            cout<<"wrfOutput.csv file in the directory:\n";
            cout<<"customExtraction/myFiles/countyInfo/WRFOutput/. \n\n";
            cout<<"Currently, the possible parameters to pass are as follows:\n";
            cout<<"--fips...............FIPs code of the county you would like\n";
            cout<<"                     to extract information from.";
            cout<<"--begin_date.........First WRF day to read from, format\n";
            cout<<"                     as YYYYmmdd";
            cout<<"--end_date...........Last WRF day to read from, format\n";
            cout<<"                     as YYYYmmdd";
            cout<<"--begin_hour.........First hour's worth of WRF data to\n";
            cout<<"                     read from for each day passed.";
            cout<<"--end_hour...........Last hour to read from";
            cout<<"--param..............Integer number of the layer(s) of \n";
            cout<<"                     Grib files to read from\n\n";
            cout<<"NOTE: arg which can have multiple inputs (fips, param)\n";
            cout<<"are read from either until the end of the arg list or until\n";
            cout<<"the next arg is reached. Flags cannot be stacked, e.g. only\n";
            cout<<"pass args with the double hyphen.\n";
            cout<<"Questions / Comments: guillotkaleb01@gmail.com";

            exit(0);
        }
        else if(strcmp(argv[i], "--begin_date")==0){
            begin_date_flag = true;
            begin_date_index = i;
        }
        else if(strcmp(argv[i], "--end_date")==0){
            end_date_flag = true;
            end_date_index = i;
        }else if(strcmp(argv[i], "--begin_hour")==0){
            begin_hour_flag = true;
            begin_hour_index = i;
        }else if(strcmp(argv[i], "--end_hour")==0){
            end_hour_flag = true;
            end_hour_index = i;
        }else if(strcmp(argv[i], "--param")==0){
            param_flag = true;
            param_index = i;
        }
    }
    if(param_flag){
        fill_n(blnParamArr, 200, false);
        for(int i=param_index+1;i<argc;i++){
            string arg = argv[i];
            if(arg.substr(0,1)=="-"){
                break;
            }
            try{
                int selectedParam = stoi(arg);
                blnParamArr[selectedParam] = true;
            }catch(exception e){
                cout << "Error in passing parameters" << endl;
                exit(0);
            }
        }
    }else{
        fill_n(blnParamArr, 200, true);
    }
    if(begin_hour_flag){
        string begin_hour = argv[begin_hour_index+1];
        if(begin_hour.length() > 2) {
            cout << "Error in the begin hour" << endl;
            exit(0);
        }
        int intbegin_hour = stoi(begin_hour);
        if(intbegin_hour > 23 || intbegin_hour < 0){
            cout << "Error in the size of the begin hour" << endl;
            exit(0);
        }
        arrHourRange[0] = intbegin_hour;
    }
    if(end_hour_flag){
        string strend_hour = argv[end_hour_index + 1];
        if(strend_hour.length()  > 2){
            cout << "Error in the end hour" << endl;
            exit(0);
        }
        int intendhour = stoi(strend_hour);
        if(intendhour > 23 || intendhour < 0 || intendhour < arrHourRange[0]){
            cout << "Error in the end hour size checking" << endl;
            exit(0);
        }
        arrHourRange[1] = intendhour;
    }
    if((begin_date_flag && !end_date_flag) | (end_date_flag && !begin_date_flag)){
        cout << "Must pass both begin and end date" << endl;
        exit(0);
    }
    if(begin_date_flag){
        // meaning we've passed both begin and end date flags
        // store them 
        string begin_date = argv[begin_date_index+1];
        string end_date = argv[end_date_index+1];

        //check their length
        if (end_date.length() != 8 || begin_date.length() != 8){
            cout << "Error: Begin / End date arguments have not been passed correctly" << endl;
            exit(0);
        }

        // pass as the actual beginning and ending date
        beginDay.at(0) = stoi(begin_date.substr(0,4));
        beginDay.at(1) = stoi(begin_date.substr(4,2));
        beginDay.at(2) = stoi(begin_date.substr(6,2));
        
        endDay.at(0) = stoi(end_date.substr(0,4));
        endDay.at(1) = stoi(end_date.substr(4,2));
        endDay.at(2) = stoi(end_date.substr(6,2));


    }
    vector<string> fipscodes(0);
    if(fipsflag){
        for(int i=fipsindex+1;i<argc;i++){
            string arg = argv[i];
            if (arg.substr(0,1) == "-"){
                break;
            }
            if (arg.length() == 5){
                fipscodes.insert(fipscodes.begin(), argv[i]);
            }
            else{
                cout << "Invalid length of fips argument passed" << endl;
                exit(0);
            }
        }
        
    }
    // if all the flags are false and argc > 1, then something went wront
    if(!param_flag && !begin_date_flag && !end_date_flag && !fipsflag && argc > 1 && !end_hour_flag && !begin_hour_flag){
        cout << "Incorrect Arguments Passed" << endl;
        exit(0);
    }
    // now call the cmd line to run the python file
    string command; int status;
    command = "cd " + repositoryPath + "myFiles/countyInfo/sentinel-hub ; python geo_gridedit_1-13-23.py --fips ";
    for(int i=0;i<fipscodes.size();i++){
        command += fipscodes.at(i) + " ";
    }
    status = system(command.c_str());
    if(status==-1) std::cerr << "Python call error: " <<strerror(errno) << endl;

    
    buildHours();
    defaultParams();
    readCountycsv();
    matchState();
    getStateAbbreviations();
        
}
void getStateAbbreviations(){
    // read the us-state-ansi-fips.csv file into a map, 
    // KEY: fips, VALUE: abbrev
    map<string, string> stateabbrevmap;
    ifstream abbrevs;
    string abbrev_path = repositoryPath + "myFiles/countyInfo/us-state-ansi-fips.csv";
    abbrevs.open(abbrev_path);
    if(!abbrevs){
        cerr << "Error opening the abbreviations file." << endl;
        exit(1);
    }
    vector<string> row; 
    string strLine;
    bool firstline = true;
    while(getline(abbrevs, strLine)){
        row.clear();
        if(firstline){
            firstline = false;
            continue;
        }
        stringstream s(strLine);
        string currline;
        while(getline(s, currline, ',')){
            row.push_back(currline);
        }
        if(row.size() > 2){
            string fips = row.at(1); string abbrev = row.at(2);
            fips.erase(remove_if(fips.begin(), fips.end(), ::isspace));
            abbrev.erase(remove_if(abbrev.begin(), abbrev.end(), ::isspace));
            stateabbrevmap.insert({fips, abbrev});
        }
    }

    ///////////////////////////////////////////////////////////
    // based off of the map, give each station their abbrev
    map<string, string>::iterator itr;
    for(int i=0; i<numStations; i++){
        Station *station = &stationArr[i];
        string statefips = station->fipsCode.substr(0,2);
        itr = stateabbrevmap.find(statefips);
        if(itr!= stateabbrevmap.end()) station->stateAbbrev = itr->second;
        else{
            cout << "Error in finding state abbrevs" <<endl;
            exit(1);
        }
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

    hours = new string[intHourRange];
       
    int endHour = arrHourRange.at(1);
    int beginHour = arrHourRange.at(0);
    int index=0;
    for(int hour = beginHour; hour<=endHour;hour++){
        if(hour < 10){ // put a 0 in front then insert in the hours arr
            string strHour = "0"+to_string(hour);
            hours[index] = strHour;
            index++;
        }
        else{ // just convert to a string and insert into the hours arr
            string strHour = to_string(hour);
            hours[index] = strHour;
            index++;
        }
    }
}

void readCountycsv(){
    
    map<string, Station> tmpStationMap;
    string strLine;
    ifstream filewrfdata;
    string path_to_data_file = repositoryPath + "myFiles/WRFoutput/wrfOutput.csv";
    filewrfdata.open(path_to_data_file);
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
        string stationMapidx = st.fipsCode + "_" + st.name;
        tmpStationMap.insert({stationMapidx, st});
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
    string filecountyfipscodes_path = repositoryPath + "myFiles/countyInfo/countyFipsCodes.txt";
    filecountyfipscodes.open(filecountyfipscodes_path);
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
        if(stateItr!=stateMap.end()) stationArr[i].state = stateItr->second;
        else{
            cout << "Error in the state matching." << endl; exit(0);
        }
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
			if(year % 400 != 0){
				day = 1;
				month = 3;
			}
            else if(year % 100 == 0 && year % 4 != 0){
                day = 1;
                month = 3;
            }
			else if ((year % 400 == 0) || ((year % 100 != 0) && (year % 4 == 0))){ // is a leap year
				day == 29;
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
        else{
            day +=1;
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
		// cannot access the directory, e.g. it may not exist
        return false;
        // printf("Error checking directory: %s", filePath.c_str());
        // exit(0);
	}
	else if(info.st_mode & S_IFDIR){
		// the directory already exists
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

void createPath(){

    // separate the write path by slashes, pass each hypen to the dirExists function,
    // if it doesnt exist, keep appending toa new fielpath
    vector<string> splittedWritePath = splitonDelim(writePath, '/');
    string writePath_1;
    for(int i=0; i<splittedWritePath.size(); i++){
        writePath_1.append("/" + splittedWritePath[i]);
        if(!dirExists(writePath_1)){
            //does not exist, need to create the path
            if(mkdir(writePath_1.c_str(), 0777)==-1){
                printf("Unable to create directory to write files: %s\n", writePath.c_str());
                exit(0);
            }
        }
    }


    // for each of the stations passed, make a directory for their fips
    // and then in each fips folder, make a directory for each year passed
    vector<int> allyearspassed = {}
    I stopped here
    for(int i=0; i<numStations; i++){
        Station * station = &stationArr[i];
        string fips = station->fipsCode;
        string writePath_1 = writePath + fips;
        if(!dirExists(writePath_1)){
            strcmd = "mkdir -p " + writePath_1;
            status = system(strcmd.c_str());
            if(status==-1)std::cerr << "writePathError: " << strerror(errno) << endl;
        }

    }

    // for each of the years passed, create a folder
    // for each folder in write path, pass each year passed
    string currpath; 
    for (const auto &entry : std::filesystem::directory_iterator(writePath)){
        currpath = entry.path();
        strcmd = "cd " + currpath + "; mkdir " + to_string(beginDay.at(0));
        status = system(strcmd.c_str());
        if(status==-1) cerr << "WritePathERROR: " << strerror(errno) << endl;
        
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
    }
    

}

void writeData(void*arg){

    struct writeThreadArgs writeThreadArgs = *(struct writeThreadArgs*)arg;

    Station*station = (writeThreadArgs).station;

    // the entire file will be written to in one iteration. We will append the strings
    // of all days together until the next day is reached, at which point we will write out
    string output, filePath_out, fileName, year, county, state, name, fips, strcmd, month;
    int status;
    ifstream outputFile;
    // loop through each key of the file, every time you run into a new day, 
    // make a new file
    for(auto itr = station->weeklydatamap.begin(); itr!=station->weeklydatamap.end();++itr){
        string weekrange = itr->first;
        year = weekrange.substr(0,4);
        month = weekrange.substr(4,2);
        county = station->county;
        state = station->state;
        name = station->name;
        fips = station->fipsCode;

        filePath_out = writePath + fips + "/" + year + "/";
        fileName = "HRRR_"+station->fipsCode.substr(0,2)+"_"+station->stateAbbrev+"_"+year+month+ ".csv";
        outputFile.open(filePath_out+fileName);
        if(!outputFile){
            // the file does not exists, need to write out the header 
            //strcmd = "cd " + filePath_out + "; echo \"CountyIndexNum,Day/Month,Year, Month, Day, State, County, FIPS Code,";
            strcmd = "cd " + filePath_out + "; echo \"Year,Month,Day,Day/Month,State,County,FIPS Code,GridIndex,Lat,Lon(-180-180),";
            // append the name of each parameter to the headings of the files
            for(int j=0;j<numParams;j++){
                // if the param name has temperature in the name, then 
                    // include a heading for min and max temperature
                //else 
                    // just write normally
                strcmd += objparamArr[j].name + " ( " + objparamArr[j].units + "),";
                if(objparamArr[j+1].name.find("Temperature") != string::npos){ //the param name has "temperature" in the name
                    strcmd += "Min Temperature (" + objparamArr[j].units + "), Max Temperature (" + objparamArr[j].units + "),";
                }
            }    
            
            strcmd += "\" > "+ fileName;
            status = system(strcmd.c_str());
            if(status==-1) std::cerr << "\n Write to file error: " << strerror(errno) << endl;
        }
        outputFile.close();
        // strcmd = "cd " + filePath_out + "; touch " +fileName;
        // status = system(strcmd.c_str());
        // if(status==-1) std::cerr << "\nFile Creation Error: " << strerror(errno) << endl;

        
    
        // append information to the output string
        // send each day of the week's data to the output file
        string firstdayoftheweek = itr->first.substr(0,8);
        string lastdayoftheweek = itr->first.substr(9,8);
        string full_daymapday,daymapyear,daymapmonth,daymapday;
        output = "";
        for(auto dayitr = station->dailydatamap.begin(); dayitr != station->dailydatamap.end(); ++dayitr){
            full_daymapday = dayitr->first;

            if(full_daymapday <= lastdayoftheweek && full_daymapday >= firstdayoftheweek){
                daymapyear = dayitr->first.substr(0,4);
                daymapmonth = dayitr->first.substr(4,2);
                daymapday = dayitr->first.substr(6,2);
                // output.append(name +",Daily,"+daymapyear +","+ daymapmonth +","+ daymapday + "," + station->state + "," + station->county + "," + fips + ",");
                output.append(daymapyear+","+daymapmonth+","+daymapday+",Daily,"+station->state+","+station->county+","+fips+","+name+"," + to_string(station->lat) + "," + to_string(station->lon)+ ",");
                // loop through the vector associated with the map and append to output
                for (auto j=0;j<dayitr->second.size();j++){
                    output.append(to_string(dayitr->second.at(j))+",");
                    
                    if(objparamArr[j+1].name.find("Temperature") != string::npos){ //the param name has "temperature" in the name
                        double minvalue = station->dailyMinParams.find(full_daymapday)->second.at(j);
                        double maxvalue = station->dailyMaxParams.find(full_daymapday)->second.at(j);
                        output.append(to_string(minvalue)+","+to_string(maxvalue)+",");
                    }
                }
                // output.append("\n"
                // send the output string as a line to the file
                strcmd = "cd " + filePath_out + "; echo " + "\"" + output + "\" >> " + fileName;
                status = system(strcmd.c_str());
                if(status == -1) std::cerr << "\nDaily File Write Error: " << strerror(errno) << endl;
                output = "";
                
            }

        }
        // output = name + ",Monthly," + year+ "," + month + ", , , ,"+fips+",";
        // for(auto j=0; j<itr->second.size();j++){
        //     output.append((itr->second.at(j))+",");
        // }
        // strcmd = "cd " + filePath_out + "; echo " + "\"" + output + "\" >> " + fileName;
        // status = system(strcmd.c_str());
        // if(status == -1) std::cerr << "\nDaily File Write Error: " << strerror(errno) << endl;
        output = "";
       
    }

}

int len(string str){
    int length = 0;
    for(int i=0; str[i]!= '\0';i++){
        length++;
    }
    return length;
}

vector<string> splitonDelim(string str, char sep){
    vector<string> returnedStr;
    int curridx = 0, i=0;
    int startidx=0, endidx = 0;
    while(i<=len(str)){
        if(str[i]==sep || i==len(str)){
            endidx=i;
            string subStr = "";
            subStr.append(str, startidx, endidx-startidx);
            returnedStr.push_back(subStr);
            curridx++;
            startidx = endidx+1;
        }
        i++;
    }
    return returnedStr;
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
    delete [] stationArr;
    delete [] hours;
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