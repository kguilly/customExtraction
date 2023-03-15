/*
NOTE:


Before running, please configure beginday, endday, arrhourrange, filePath, writepath, 
repositoryPath

This file is the first attempt at using CUDA to speed up GRIB2 file extraction. This file
will not be fully optimized, even in its final form. 

Compile:
nvcc cuda_extraction1.cu -g -leccodes -rdc=true -lcudadevrt

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

using namespace std;
/* Timing variables */
struct timespec startTotal;
struct timespec endTotal;
double totalTime;

vector<int> beginDay = {2023, 1, 31}; // arrays for the begin days and end days. END DAY IS NOT INCLUSIVE.
                                     // when passing a single day, pass the day after beginDay for endDay
                                     // FORMAT: {yyyy, mm, dd}
vector<int> endDay = {2023, 2, 3};   // NOT INCLUSIVEe

vector<int> arrHourRange = {0,23}; // array for the range of hours one would like to extract from
                                   // FORMAT: {hh, hh} where the first hour is the lower hour, second is the higher
                                   // accepts hours from 0 to 23 (IS INCLUSIVE)

int intHourRange; 

string filePath = "/media/kaleb/extraSpace/wrf/";  // path to "data" folder. File expects structure to be:
                                        // .../data/<year>/<yyyyMMdd>/hrrr.<yyyyMMdd>.<hh>.00.grib2
                                        // for every hour of every day included. be sure to include '/' at end

string writePath = "/home/kaleb/Desktop/WRFextract_2-3/"; // path to write the extracted data to,
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
    float latll; // each grid index will include the lat and lons of
    float lonll; // the lower left and upper right corners
    float latur;
    float lonur;
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
    bool *header_flag; // decides whether the header need to be written
    bool thread_header_flag; // decides whether this particular thread will make the header
};

Station *stationArr;
bool *blnParamArr;
                        // this will be used to quickly index whether a parameter needs to be
                        // extracted or not. Putting 149 spaces for 148 parameters because the
                        // layers of the parameters start at 1 rather than 0
vector<string> vctrHeader;
int numStations, numParams;

// 24 hours in a day. Use this to append to the file name and get each hour for each file
string *hours;

/*Initialization of semaphores to be used for protection*/
sem_t hProtection; // protect when creating the handle object for each layer
sem_t *mapProtection; // protect when writing to the maps
sem_t pathCreationSem; // protect when writeData makes new file paths
sem_t *valuesProtection; // protect when writing values to the values array
sem_t writeProtection; // only one thread should be able to write to a file at a time
sem_t headerProtection; // only one thread can read the header flag at a time

// function to handle arguments passed. Will either build the Station array and/or paramter array
// based off of the arguments passed or will build them with default values. Will also construct
// the hour array based on passed beginning and end hours
void handleInput(int, char**);

// Function to build the default station array (all counties in continential US)
// through reading from the files in the countyInfo file
void defaultStations(); void readCountycsv(); void matchState();
void getStateAbbreviations();
// similar to default station, but for the parameter arrays
void defaultParams(bool);

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

/*Function to find the standard deviation of the values for each week,
takes an array of the values for the week and their average, outputs a stdDev*/
static double standardDev(vector<double>, double&);

/*Function to create the paths and files for the maps to be written to*/
void createPath();

/*functions to help create paths by finding length of string and splitting strnig on delimeter*/
int len(string); vector<string> splitonDelim(string, char);

/*function to write the data after each day is extracted*/
void writeHourlyData(bool, vector<string>);

void garbageCollection();

int main(int argc, char*argv[]){
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
    bool header_flag = true; // want to make a header for the files that we write out
    bool*ptrHeader_flag = &header_flag; // want all threads to point to the same one
    bool header_write_flag = false; // has the header been written out to the file yet?
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
            arrThreadArgs[i].header_flag = ptrHeader_flag;
            arrThreadArgs[i].thread_header_flag = false;
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

        writeHourlyData(header_write_flag, strCurrentDay);
        header_write_flag = true;

        if(formatDay(getNextDay(intcurrentDay)).at(1)!=strCurrentDay.at(1)){
            *ptrHeader_flag = true;
            header_write_flag = false;
        }
        intcurrentDay = getNextDay(intcurrentDay);

        delete [] arrThreadArgs;
    }
    delete [] blnParamArr;

//    string strcmd; int status;
//    strcmd = "cd " + repositoryPath + "myFiles/ ; python processWRF_cpp.py --repo_path ";
//    strcmd += repositoryPath+" --wrf_path " + writePath;
//    // status = system(strcmd.c_str());
//    // if(status==-1)std::cerr << "Call to python formatting data error: " << strerror(errno) << endl;
//
//    strcmd = "cd " + repositoryPath + "myFiles/pythonPygrib/ ; python gribMessages.py --repo_path ";
//    strcmd += repositoryPath + " --wrf_path " + writePath + " --grib2_path " + filePath;
//    // status = system(strcmd.c_str());
//    // if(status==-1)std::cerr << "Call to python grib messages error: " << strerror(errno) << endl;


    garbageCollection();
    clock_gettime(CLOCK_MONOTONIC, &endTotal);
    totalTime = (endTotal.tv_sec - startTotal.tv_sec) * 1000.0;
    totalTime+= (endTotal.tv_nsec - startTotal.tv_nsec) / 1000000.0;
    int h=0, m=0;
    double sec;
    sec = totalTime / 1000;
    if(sec/60 > 1){
        m = int (sec) / 60;
        sec = sec - (m*60);
    }
    if(m/60 > 1){
        h = int (m) / 60;
        m = m - (h*60);
    }
    printf("\n\nRuntime:\nHours: %s, Minutes: %s, Seconds %f\n", to_string(h).c_str(), to_string(m).c_str(), sec);


    return 0;
 }

void handleInput(int argc, char* argv[]){
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
            cout<<"                     to extract information from.\n";
            cout<<"--begin_date.........First WRF day to read from, format\n";
            cout<<"                     as YYYYmmdd\n";
            cout<<"--end_date...........Last WRF day to read from, format\n";
            cout<<"                     as YYYYmmdd\n";
            cout<<"--begin_hour.........First hour's worth of WRF data to\n";
            cout<<"                     read from for each day passed.\n";
            cout<<"--end_hour...........Last hour to read from\n";
            cout<<"--param..............Integer number of the layer(s) of \n";
            cout<<"                     Grib files to read from\n\n";
            cout<<"NOTE: arg which can have multiple inputs (fips, param)\n";
            cout<<"are read from either until the end of the arg list or until\n";
            cout<<"the next arg is reached. Flags cannot be stacked, e.g. only\n";
            cout<<"pass args with the double hyphen.\n";
            cout<<"Questions / Comments: guillotkaleb01@gmail.com\n\n";

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
        blnParamArr = new bool[200];
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
    if(fipsflag){
        string command; int status;
        command = "cd " + repositoryPath + "myFiles/countyInfo/sentinel-hub ; python geo_gridedit_1-30-23.py --fips ";
        for(int i=0;i<fipscodes.size();i++){
            command += fipscodes.at(i) + " ";
        }
        command += " --write_path " + repositoryPath + "myFiles/countyInfo/";
        status = system(command.c_str());
        if(status==-1) std::cerr << "Python call error: " <<strerror(errno) << endl;
    }


    buildHours();
    defaultParams(param_flag);
    readCountycsv();
    matchState();
    getStateAbbreviations();

}

void defaultParams(bool param_flag){
    string paramline;
    ifstream paramFile;
    string paramfiledir = repositoryPath + "myFiles/countyInfo/parameterInfo.csv";
    paramFile.open(paramfiledir);
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
            countParams++;
        }
    }
    numParams = countParams;


    // build the boolean param array
    if(!param_flag){
        blnParamArr = new bool[numParams+1];
        for (int i = 0; i<numParams; i++){
            blnParamArr[i+1] = true;
        }
    }
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
    string path_to_data_file = repositoryPath + "myFiles/countyInfo/WRFoutput/wrfOutput.csv";
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
        st.latur = stof(row.at(4)); st.lonur = stof(row.at(5));
        st.latll = stof(row.at(6)); st.lonll = stof(row.at(7));
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
    if(sem_init(&headerProtection, 0, 1)==-1){
        perror("sem_init");
        exit(EXIT_FAILURE);
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
        Station *station = &stationArr[i];
        station->lonll = (station->lonll +360); // circle, going clockwise vs counterclockwise
        station->lonur = (station->lonur+360);
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
    bool *ptrHeader_flag = (*threadArg).header_flag;
    bool thread_header_flag = (*threadArg).thread_header_flag;


    printf("\nOpening File: %s", filePath2.c_str());

    //try to open the file
    try{
        f = fopen(filePath2.c_str(), "rb");
        if(!f) throw(filePath2);
    }
    catch(string file){
        printf("\nError: could not open filename %s in directory %s", fileName.c_str(), file.c_str());
        pthread_exit(0);
        return(void*) nullptr;
    }
    // check to see if the header needs to be formulated
    sem_wait(&headerProtection);
    if(*ptrHeader_flag == true){
        *ptrHeader_flag = false;
        thread_header_flag = true; // this is the chosen thread to get the keys (header)
        // from the grib file
        vctrHeader.clear();
        vctrHeader.push_back("Year"); vctrHeader.push_back("Month");vctrHeader.push_back("Day");vctrHeader.push_back("Hour");
        vctrHeader.push_back("State");vctrHeader.push_back("County");
        vctrHeader.push_back("Grid Index"); vctrHeader.push_back("FIPS Code");vctrHeader.push_back("lat(llcrnr)");
        vctrHeader.push_back("lon(llcrnr)"); vctrHeader.push_back("lat(urcrnr)");vctrHeader.push_back("lon(urcrnr)");
    }
    sem_post(&headerProtection);


    // unsigned long key_iterator_filter_flags = CODES_KEYS_ITERATOR_ALL_KEYS |
    //                                           CODES_KEYS_ITERATOR_SKIP_DUPLICATES;
    codes_grib_multi_support_on(NULL);

    codes_handle * h = NULL; // use to unpack each layer of the file

    const double missing = 1.0e36;        // placeholder for when the value cannot be found in the grib file
    int msg_count = 0;  // KEY: will match with the layer of the passed parameters
    int err = 0;
    size_t vlen = MAX_VAL_LEN;
    char value_1[MAX_VAL_LEN];

    // char value[MAX_VAL_LEN];
    // size_t vlen = MAX_VAL_LEN;

    bool flag = true;

    long numberOfPoints=0;

    double *lats, *lons, *values; // lats, lons, and values returned from extracted grib file
    int curParamIdx =0; // the index which we will place the value into each station's
                        // values array
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

            if(thread_header_flag){
                string name_space = "parameter";
                unsigned long key_iterator_filter_flags = CODES_KEYS_ITERATOR_ALL_KEYS |
                                                            CODES_KEYS_ITERATOR_SKIP_DUPLICATES;
                codes_keys_iterator *kiter = codes_keys_iterator_new(h, key_iterator_filter_flags, name_space.c_str());
                if(!kiter){
                    fprintf(stderr, "Error: unable to create keys iterator while reading params\n");
                    exit(1);
                }
                string strUnits, strName, strValue, strHeader, strnametosendout;
                while(codes_keys_iterator_next(kiter)){
                    const char*name = codes_keys_iterator_get_name(kiter);
                    vlen = MAX_VAL_LEN;
                    memset(value_1, 0, vlen);
                    CODES_CHECK(codes_get_string(h, name, value_1, &vlen), name);
                    strName = name, strValue = value_1;
                    if(strName.find("name")!=string::npos){
                        strValue.erase(remove(strValue.begin(), strValue.end(), ','), strValue.end());
                        //strHeader.append(to_string(msg_count)+"_"+strValue);
                        strnametosendout = strValue;
                    }
                    else if(strName.find("units")!=string::npos){
                        //strHeader.append("("+strValue+")");
                        strUnits = "(" + strValue + ")";
                    }
                }
                strHeader = strnametosendout + " " + strUnits;
                vctrHeader.push_back(strHeader);
            }
            // if it is the first time, extract the index
            if (flag){
                Station *station;
                for(int i =0; i<numStations; i++){
                    station = &stationArr[i];
                    float avglat = (station->latll + station->latur) / 2;
                    float avglon = (station->lonll + station->lonur) / 2;
                    int closestPoint = 0;
                    for (int j=0; j<numberOfPoints;j++){
                        double distance = pow(lats[j]- avglat, 2) + pow(lons[j]-avglon, 2);
                        double closestDistance = pow(lats[closestPoint] - avglat, 2) + pow(lons[closestPoint] - avglon,2);
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
                double dblCurrentValue = values[station->closestPoint[threadIndex]];
                sem_wait(&valuesProtection[i]);
                station->values[threadIndex][curParamIdx] = dblCurrentValue;
                sem_post(&valuesProtection[i]);
            }
            curParamIdx++;
            std::free(lats);
            std::free(lons);
            std::free(values);

        }
        codes_handle_delete(h);

    }
    sem_post(&hProtection);
    fclose(f);
    // call the mapData function to map the hour's parameter's to each station's map
    pthread_exit(0);

}

void createPath(){

    // separate the write path by slashes, pass each hypen to the dirExists function,
    // if it doesnt exist, keep appending toa new fielpath
    writePath += "Hourly/"; // the data we'll be extracting will be separate from the daily/monthly format
    vector<string> splittedWritePath = splitonDelim(writePath, '/');
    string writePath_1;
    for(int i=0; i<splittedWritePath.size(); i++){
        writePath_1.append("/" + splittedWritePath[i]);
        if(!dirExists(writePath_1)){
            //does not exist, need to create the path
            if(mkdir(writePath_1.c_str(), 0777)==-1){
                printf("Unable to create directory to write files: %s\n", writePath_1.c_str());
                exit(0);
            }
        }
    }


    // for each of the stations passed, make a directory for their fips
    // and then in each fips folder, make a directory for each year passed
    vector<int> allyearspassed;
    int yearRange = endDay[0] - beginDay[0];
    for(int i=0;i<=yearRange;i++){
        // append all years needed to the vector, including the first day
        allyearspassed.push_back(beginDay[0]+i);
    }
    Station station;
    for(int i=0; i<numStations;i++){
        station=stationArr[i];
        string fips = station.fipsCode;
        string writePath_2 = writePath+fips+"/";
        if(!dirExists(writePath_2)){
            if(mkdir(writePath_2.c_str(), 0777)==-1){
                printf("Unable to create fips directory to write files: %s\n", writePath_2.c_str());
                exit(0);
            }
        }
        // now the fips directory for the station has been created, now make the
        // year directories
        for(int j=0;j<allyearspassed.size();j++){
            string writePath_3 = writePath_2 + to_string(allyearspassed[j]) + "/";
            if(!dirExists(writePath_3)){
                if(mkdir(writePath_3.c_str(),0777)==-1){
                    printf("Unable to create year directories to write to files %s\n", writePath_3.c_str());
                }
            }
        }
    }
}

void writeHourlyData(bool header_write_flag, vector<string>formattedDay){


    if(!header_write_flag){
        // grab the station fips code, if the current station fips cannot be found
        // in the vector, write out the header and append the current fips code
        // to the vector
        string fips = ""; Station station;
        vector<string> fipsCodes;
        for (int i=0; i<numStations; i++){
            station = stationArr[i];
            fips = station.fipsCode;
            if(find(fipsCodes.begin(), fipsCodes.end(), fips)==fipsCodes.end()){
                // the header for this file has not been written to yet,
                // find the directory of the file, clear it, and write the header
                // to it
                fipsCodes.push_back(fips);
                string year = formattedDay.at(0); string month = formattedDay.at(1);
                string fipsDir = writePath+fips+"/";
                string yearDir = fipsDir+year+"/";
                string fileName = "HRRR_"+fips+"_"+station.stateAbbrev+"_"+year+month+".csv";
                ofstream outputFile(yearDir+fileName);
                for(int j=0; j<vctrHeader.size();j++){
                    outputFile << vctrHeader.at(j) << ",";
                }
                outputFile << "\n";
                outputFile.close();


            }
        }
    }

    string year = formattedDay.at(0); string month = formattedDay.at(1);
    string fipsDir;
    Station station;
    for(int i=0;i<numStations;i++){
        station = stationArr[i];
        string fips = station.fipsCode;
        fipsDir = writePath+fips+"/";

        string yearDir = fipsDir+year+"/";
        string fileName = "HRRR_"+fips+"_"+station.stateAbbrev+"_"+year+month+".csv";
        string strOutput;
        // loop through the station's values array
        int currHour = arrHourRange.at(0);
        for(int j=0; j<intHourRange; j++){
            // append the init info to strOutput:
            // Year, Month, Day, Daily/Monthly, State, County, Grid Index, FIPS code, lat, lon
            strOutput.append(year+","+month+","+formattedDay.at(2)+","+to_string(currHour)+","+station.state+",");
            strOutput.append(station.county+","+station.name+","+station.fipsCode+","+to_string(station.latll)+","+to_string(station.lonll)+",");
            strOutput.append(to_string(station.latur)+","+to_string(station.lonur)+",");
            for(int k=0; k< numParams; k++){
                // write out station.values[j][k]
                // outputFile << station.values[j][k] << ",";
                // append the value and the comma to the output string
                if(k>=vctrHeader.size()-12){
                    break;
                }
                strOutput.append(to_string(station.values[j][k])+",");
            }
            // strOutput.append("\n");
            // append strOutput to the file and clear the string
            string strCommand; int status;
            strCommand = "cd " + yearDir + "; echo \"" + strOutput + "\" >> "+fileName;
            status = system(strCommand.c_str());
            if(status==-1) std::cerr << "\nWrite to file error in writeHourlyData: " << strerror(errno) << endl;
            strOutput = "";
            currHour++;
        }



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
    if(sem_destroy(&headerProtection)==-1){
        perror("sem_destroy");
        exit(EXIT_FAILURE);
    }
    free(mapProtection);
    for (int i =0; i<numStations; i++){
        sem_destroy(&mapProtection[i]);
    }

}