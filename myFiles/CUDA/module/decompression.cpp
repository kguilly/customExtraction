/*
TODO: Optimize filling in each station's data, mirror the way its done in python



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

// Personal Headers
#include "shared_objs.h"
#include "decompress_funcs.h"
#include "decompress_cuda.h"
#define MAX_VAL_LEN 1024

/* Timing variables */
struct timespec startTotal;
struct timespec endTotal;
double totalTime;

std::vector<int> beginDay = {2020, 1, 1}; // arrays for the begin days and end days. END DAY IS NOT INCLUSIVE.
                                     // when passing a single day, pass the day after beginDay for endDay
                                     // FORMAT: {yyyy, mm, dd}
std::vector<int> endDay = {2020, 1, 2};   // NOT INCLUSIVEe

std::vector<int> arrHourRange = {0,23}; // array for the range of hours one would like to extract from
                                   // FORMAT: {hh, hh} where the first hour is the lower hour, second is the higher
                                   // accepts hours from 0 to 23 (IS INCLUSIVE)

int intHourRange; 

std::string filePath = "/media/kaleb/extraSpace/wrf/";  // path to "data" folder. File expects structure to be:
                                        // .../data/<year>/<yyyyMMdd>/hrrr.<yyyyMMdd>.<hh>.00.grib2
                                        // for every hour of every day included. be sure to include '/' at end

std::string writePath = "/home/kaleb/Desktop/cuda_test_3-22/"; // path to write the extracted data to,
                                                    // point at a WRFData folder
std::string repositoryPath = "/home/kaleb/Documents/GitHub/customExtraction/";//PATH OF THE CURRENT REPOSITORY
                                                                          // important when passing args                                                    

station_t *stationArr; 
                        // this will be used to quickly index whether a parameter needs to be 
                        // extracted or not. Putting 149 spaces for 148 parameters because the
                        // layers of the parameters start at 1 rather than 0
std::vector<std::string> vctrHeader;
int numStations, numParams;
bool* blnParamArr;

// 24 hours in a day. Use this to append to the file name and get each hour for each file
std::string *hours;

sem_t *valuesProtection; // protect when writing values to the values array
sem_t *barrier; // protect when passing shared values to and from the GPU

/*functions to help create paths by finding length of std::string and splitting strnig on delimeter*/
int len(std::string); std::vector<std::string> splitonDelim(std::string, char);

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
    std::vector<int> intCurrentDay = beginDay;
    std::string prevMonth = "";

    // print out data from each of the stations:
    for (int i=0; i<numStations; i++){
        station_t station = stationArr[i];
        std::cout << station.stateAbbrev << ", IDX: " << station.grid_idx << std::endl;
        std::cout << station.latll << "," << station.lonll << ", closest pt = " << station.closestPoint << std::endl;
        std::cout << std::endl;

    }
    
    double* grib_lats = (double*)malloc(sizeof(double));
    double* grib_lons = (double*)malloc(sizeof(double));
    long numberOfPoints;
    bool first_hour_flag, last_hour_flag, new_month_flag;

    while(checkDateRange(intCurrentDay, endDay)){
        
        std::vector<std::string> strCurrentDay = formatDay(intCurrentDay);
        std::string currMonth = strCurrentDay.at(1);
        if (currMonth != prevMonth) {
            prevMonth = currMonth;
            std::free(grib_lats);
            std::free(grib_lons);
            new_month_flag = true;
            std::cout << "Getting Station Indexes for Date " << strCurrentDay.at(3) << std::endl;
            get_nearest_indexes(strCurrentDay, grib_lats, grib_lons, numberOfPoints);
        }

        pthread_t *threads = (pthread_t*)malloc(intHourRange * sizeof(pthread_t)); // will be freed at the end of this iteration
        if(!threads){
            fprintf(stderr, "Error: unable to allocate %ld bytes for threads.\n", (long)(intHourRange*sizeof(pthread_t)));
            exit(0);
        }

        FILE* f[intHourRange]; // use to open the file for each hour
        threadArgs_t *arrThreadArgs = new threadArgs[intHourRange];

        for (int i=0; i<intHourRange; i++){
            if (i==0) first_hour_flag = true;
            else first_hour_flag = false;
            if (i==intHourRange-1) last_hour_flag = true;
            else last_hour_flag = false;

            // place args into the 
            f[i] = NULL;
            string hour = hours[i];
            string fileName = "hrrr."+strCurrentDay.at(3)+"."+hour+".00.grib2";
            string filePath2 = filePath1 + fileName;
            arrThreadArgs[i].f = f[i];
            arrThreadArgs[i].pathName = filePath2;
            arrThreadArgs[i].threadIndex = i;
            arrThreadArgs[i].hour = hour;
            arrThreadArgs[i].strCurrentDay = strCurrentDay; 
            arrThreadArgs[i].first_hour_flag = first_hour_flag;
            arrThreadArgs[i].last_hour_flag = last_hour_flag;
            arrThreadArgs[i].strCurrentDay = strCurrentDay.at(3);
            arrThreadArgs[i].values_protection = valuesProtection;
            arrThreadArgs[i].barrier = &barrier;
            arrThreadArgs[i].blnParamArr = blnParamArr;

            threaderr = pthread_create(&threads[i], NULL, &read_grib_data, &arrThreadArgs[i]);
            if(threaderr){
                assert(0);
                return 1;
            }
            ////////////////////
            if (new_month_flag) new_month_flag = false;

        }

        for (int i=0; i<intHourRange; i++) {
            pthread_join(threads[i], NULL);
        }

        std::free(threads);
        delete [] arrThreadArgs;

        intCurrentDay = getNextDay(intCurrentDay);
    }

    // print out data from each of the stations:
    std::cout << "\n\nAfter returning from the kernel:\n";
    for (int i=0; i<numStations; i++){
        station_t station = stationArr[i];

        std::cout << station.stateAbbrev << ", IDX: " << station.grid_idx << std::endl;
        std::cout << station.latll << "," << station.lonll << ", closest pt = " << station.closestPoint << std::endl;
        std::cout << std::endl;

    }
    garbageCollection();
    return 0;
}

void handleInput(int argc, char* argv[]){
    std::vector<std::string> vctrstrBeginDay = formatDay(beginDay);
    // check to see if the correct arguments have been passed to fill the parameter and/or 
    // station arrays
    int fipsindex=0, begin_date_index=0, end_date_index=0, begin_hour_index=0, end_hour_index=0, param_index=0;
    bool fipsflag = false, begin_date_flag = false, end_date_flag = false, begin_hour_flag = false, end_hour_flag = false, param_flag=false;
    for(int i=0; i<argc;i++){
        if(strcmp(argv[i], "--fips") == 0){
            fipsflag = true;
            fipsindex = i;      
        }else if(strcmp(argv[i], "--help")==0 || strcmp(argv[i], "-h")==0){
            std::cout<<"####################### Help Mode ###########################\n";
            std::cout<<"This is a script to decompress grib files for a selected\n";
            std::cout<<"county region. The file read to know the counties is the \n";
            std::cout<<"wrfOutput.csv file in the directory:\n";
            std::cout<<"customExtraction/myFiles/countyInfo/WRFOutput/. \n\n";
            std::cout<<"Currently, the possible parameters to pass are as follows:\n";
            std::cout<<"--fips...............FIPs code of the county you would like\n";
            std::cout<<"                     to extract information from.\n";
            std::cout<<"--begin_date.........First WRF day to read from, format\n";
            std::cout<<"                     as YYYYmmdd\n";
            std::cout<<"--end_date...........Last WRF day to read from, format\n";
            std::cout<<"                     as YYYYmmdd\n";
            std::cout<<"--begin_hour.........First hour's worth of WRF data to\n";
            std::cout<<"                     read from for each day passed.\n";
            std::cout<<"--end_hour...........Last hour to read from\n";
            std::cout<<"--param..............Integer number of the layer(s) of \n";
            std::cout<<"                     Grib files to read from\n\n";
            std::cout<<"NOTE: arg which can have multiple inputs (fips, param)\n";
            std::cout<<"are read from either until the end of the arg list or until\n";
            std::cout<<"the next arg is reached. Flags cannot be stacked, e.g. only\n";
            std::cout<<"pass args with the double hyphen.\n";
            std::cout<<"Questions / Comments: guillotkaleb01@gmail.com\n\n";

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
            std::string arg = argv[i];
            if(arg.substr(0,1)=="-"){
                break;
            }
            try{
                int selectedParam = stoi(arg);
                blnParamArr[selectedParam] = true;
            }catch(std::exception e){
                std::cout << "Error in passing parameters" << std::endl;
                exit(0);
            }
        }
    }
    if(begin_hour_flag){
        std::string begin_hour = argv[begin_hour_index+1];
        if(begin_hour.length() > 2) {
            std::cout << "Error in the begin hour" << std::endl;
            exit(0);
        }
        int intbegin_hour = stoi(begin_hour);
        if(intbegin_hour > 23 || intbegin_hour < 0){
            std::cout << "Error in the size of the begin hour" << std::endl;
            exit(0);
        }
        arrHourRange[0] = intbegin_hour;
    }
    if(end_hour_flag){
        std::string strend_hour = argv[end_hour_index + 1];
        if(strend_hour.length()  > 2){
            std::cout << "Error in the end hour" << std::endl;
            exit(0);
        }
        int intendhour = stoi(strend_hour);
        if(intendhour > 23 || intendhour < 0 || intendhour < arrHourRange[0]){
            std::cout << "Error in the end hour size checking" << std::endl;
            exit(0);
        }
        arrHourRange[1] = intendhour;
    }
    if((begin_date_flag && !end_date_flag) | (end_date_flag && !begin_date_flag)){
        std::cout << "Must pass both begin and end date" << std::endl;
        exit(0);
    }
    if(begin_date_flag){
        // meaning we've passed both begin and end date flags
        // store them 
        std::string begin_date = argv[begin_date_index+1];
        std::string end_date = argv[end_date_index+1];

        //check their length
        if (end_date.length() != 8 || begin_date.length() != 8){
            std::cout << "Error: Begin / End date arguments have not been passed correctly" << std::endl;
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
    std::vector<std::string> fipscodes(0);
    if(fipsflag){
        for(int i=fipsindex+1;i<argc;i++){
            std::string arg = argv[i];
            if (arg.substr(0,1) == "-"){
                break;
            }
            if (arg.length() == 5){
                fipscodes.insert(fipscodes.begin(), argv[i]);
            }
            else{
                std::cout << "Invalid length of fips argument passed" << std::endl;
                exit(0);
            }
        }
        
    }
    // if all the flags are false and argc > 1, then something went wront
    if(!param_flag && !begin_date_flag && !end_date_flag && !fipsflag && argc > 1 && !end_hour_flag && !begin_hour_flag){
        std::cout << "Incorrect Arguments Passed" << std::endl;
        exit(0);
    }
    // now call the cmd line to run the python file
    if(fipsflag){
        std::string command; int status;
        command = "cd " + repositoryPath + "myFiles/pythonPygrib ; python geo_grid_recent.py --fips ";
        for(int i=0;i<fipscodes.size();i++){
            command += fipscodes.at(i) + " ";
        }
        command += " --write_path " + repositoryPath + "myFiles/CUDA/";
        status = system(command.c_str());
        if(status==-1) std::cerr << "Python call error: " <<strerror(errno) << std::endl;
    }

    
    buildHours();
    defaultParams(param_flag);
    readCountycsv();
    matchState();
    getStateAbbreviations();
        
}

void buildHours(){
    // build the hour array
    // make sure correct values have been passed to the hour array 
    try{
        intHourRange = arrHourRange.at(1) - arrHourRange.at(0)+1;
        if(intHourRange < 1) throw(intHourRange);
    }catch(std::exception e){
        fprintf(stderr, "Error, problems with hour range.");
        exit(0);
    }

    hours = new std::string[intHourRange];
       
    int endHour = arrHourRange.at(1);
    int beginHour = arrHourRange.at(0);
    int index=0;
    for(int hour = beginHour; hour<=endHour;hour++){
        if(hour < 10){ // put a 0 in front then insert in the hours arr
            std::string strHour = "0"+std::to_string(hour);
            hours[index] = strHour;
            index++;
        }
        else{ // just convert to a std::string and insert into the hours arr
            std::string strHour = std::to_string(hour);
            hours[index] = strHour;
            index++;
        }
    }
}

void defaultParams(bool param_flag){
    numParams = 200;
    // build the boolean param array
    if(!param_flag){
        blnParamArr = new bool[numParams+1];
        for (int i = 0; i<numParams; i++){
            blnParamArr[i+1] = true;
        }    
    }
}

void readCountycsv(){
    std::string strLine;
    std::ifstream filewrfdata;
    std::string path_to_data_file = repositoryPath + "myFiles/CUDA/WRFoutput/wrfOutput.csv";
    filewrfdata.open(path_to_data_file);
    if(!filewrfdata){
        std::cerr << "Error: the WRFOUTPUT file could not be opened.\n";
        exit(1);
    }
    bool firstLine = true;
    int arridx = 0;
    while(getline(filewrfdata, strLine)){
        if(firstLine){
            firstLine = false;
            continue;
        }
        arridx++;
    }
    filewrfdata.close();
    numStations = arridx;
    stationArr = new station_t[numStations];
    
    filewrfdata.open(path_to_data_file);
    if(!filewrfdata){
        std::cerr << "Error: the WRFOUTPUT file could not be opened.\n";
        exit(1);
    }
    int i=0;
    std::vector<std::string> row;
    firstLine = true;
    while(getline(filewrfdata, strLine)){
        row.clear();
        if(firstLine){
            firstLine = false;
            continue;
        }
        std::stringstream s(strLine);
        std::string currLine;
        while(getline(s, currLine, ',')){
            row.push_back(currLine);
        }
        station_t station;
        station.grid_idx = row.at(0).c_str(); station.fipsCode = row.at(1).c_str();
        station.latur = stof(row.at(4)); station.lonur = stof(row.at(5));
        station.latll = stof(row.at(6)); station.lonll = stof(row.at(7));
        // station.county = row.at(3).c_str();
        station.values = new double*[intHourRange];
        for(int j=0; j<intHourRange; j++){
            station.values[j] = new double[numParams];
        }
        stationArr[i] = station;
        i++;
    }
}

void matchState(){
    // all the points will have the same state, so search through the file
    // that has the state fips and put it as the state for each of the 
    // stations
    std::map<std::string, std::string> stateMap;
    std::string strcountyfipscodes;
    std::ifstream filecountyfipscodes;
    std::string filecountyfipscodes_path = repositoryPath + "myFiles/countyInfo/countyFipsCodes.txt";
    filecountyfipscodes.open(filecountyfipscodes_path);
    if(!filecountyfipscodes){
        std::cerr << "Error: the FIPS file could not be opened.\n";
        exit(1);
    }
    bool readstates = false;
    std::string strStateInfo;
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
                std::string delimiter = ",";
                std::string strfips = strcountyfipscodes.substr(0,strcountyfipscodes.find(delimiter));
                std::string strState = strcountyfipscodes.erase(0, strcountyfipscodes.find(delimiter)+delimiter.length());

                stateMap.insert({strfips, strState});
            }
        }
    }

    filecountyfipscodes.close();
    // loop through the station array and match the fips code to the respective state
    std::map<std::string,std::string>::iterator stateItr;
    for(int i=0;i<numStations;i++){
        std::string state_fips = std::string(stationArr[i].fipsCode).substr(0,2);
        stateItr = stateMap.find(state_fips);

        // insert the state into the objs
        if(stateItr!=stateMap.end()) {
            stationArr[i].stateAbbrev = stateItr->second.c_str();
        }
        else{
            std::cout << "Error in the state matching." << std::endl; exit(0);
        }
    }   

}

void getStateAbbreviations(){
    // read the us-state-ansi-fips.csv file into a map, 
    // KEY: fips, VALUE: abbrev
    std::map<std::string, std::string> stateabbrevmap;
    std::ifstream abbrevs;
    std::string abbrev_path = repositoryPath + "myFiles/countyInfo/us-state-ansi-fips2.csv";
    abbrevs.open(abbrev_path);
    if(!abbrevs){
        std::cerr << "Error opening the abbreviations file." << std::endl;
        exit(1);
    }
    std::vector<std::string> row; 
    std::string strLine;
    bool firstline = true;
    while(getline(abbrevs, strLine)){
        row.clear();
        if(firstline){
            firstline = false;
            continue;
        }
        std::stringstream s(strLine);
        std::string currline;
        while(getline(s, currline, ',')){
            row.push_back(currline);
        }
        if(row.size() > 2){
            std::string fips = row.at(1); std::string abbrev = row.at(2);
            stateabbrevmap.insert({fips, abbrev});
        }
    }

    ///////////////////////////////////////////////////////////
    // based off of the map, give each station their abbrev
    std::map<std::string, std::string>::iterator itr;
    for(int i=0; i<numStations; i++){
        station_t *station = &stationArr[i];
        std::string statefips = std::string(station->fipsCode).substr(0,2);
        station->state_fips = statefips.c_str();
        itr = stateabbrevmap.find(statefips);
        if(itr!= stateabbrevmap.end()) station->stateAbbrev = itr->second.c_str();
        else{
            std::cout << "Error in finding state abbrevs" <<std::endl;
            exit(1);
        }
    }
}

void semaphoreInit(){

    if (sem_init(&barrier, 0, 0) == -1) {
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
    
}

void convertLatLons(){
    for(int i=0; i<numStations;i++){
        station_t * station = &stationArr[i];
        station->lonll = (station->lonll +360); // circle, going clockwise vs counterclockwise
        station->lonur = (station->lonur+360);
    }
}

void createPath(){

    // separate the write path by slashes, pass each hypen to the dirExists function,
    // if it doesnt exist, keep appending toa new fielpath
    writePath += "Hourly/"; // the data we'll be extracting will be separate from the daily/monthly format
    std::vector<std::string> splittedWritePath = splitonDelim(writePath, '/');
    std::string writePath_1;
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
    std::vector<int> allyearspassed;
    int yearRange = endDay[0] - beginDay[0];
    for(int i=0;i<=yearRange;i++){
        // append all years needed to the vector, including the first day
        allyearspassed.push_back(beginDay[0]+i);
    }
    station_t station;
    for(int i=0; i<numStations;i++){
        station=stationArr[i];
        std::string fips = station.fipsCode;
        std::string writePath_2 = writePath+fips+"/";
        if(!dirExists(writePath_2)){
            if(mkdir(writePath_2.c_str(), 0777)==-1){
                printf("Unable to create fips directory to write files: %s\n", writePath_2.c_str());
                exit(0);
            }
        }
        // now the fips directory for the station has been created, now make the 
        // year directories
        for(int j=0;j<allyearspassed.size();j++){
            std::string writePath_3 = writePath_2 + std::to_string(allyearspassed[j]) + "/";
            if(!dirExists(writePath_3)){
                if(mkdir(writePath_3.c_str(),0777)==-1){
                    printf("Unable to create year directories to write to files %s\n", writePath_3.c_str());
                }
            }
        }
    }
}

std::vector<std::string> formatDay(std::vector<int> date){
	std::string strYear, strMonth, strDay;
	int intYear = date.at(0);
	int intMonth = date.at(1);
	int intDay = date.at(2);

	if(intMonth < 10){
		strMonth = "0" + std::to_string(intMonth);
	}
	else strMonth = std::to_string(intMonth);
	if(intDay < 10) strDay = "0" + std::to_string(intDay);
	else strDay = std::to_string(intDay);
	strYear = std::to_string(intYear);

	std::vector<std::string> formattedDate = {strYear , strMonth, strDay , (strYear+strMonth+strDay)};
	return formattedDate;
}

std::vector<std::string> splitonDelim(std::string str, char sep){
    std::vector<std::string> returnedStr;
    int curridx = 0, i=0;
    int startidx=0, endidx = 0;
    while(i<=len(str)){
        if(str[i]==sep || i==len(str)){
            endidx=i;
            std::string subStr = "";
            subStr.append(str, startidx, endidx-startidx);
            returnedStr.push_back(subStr);
            curridx++;
            startidx = endidx+1;
        }
        i++;
    }
    return returnedStr;
}

int len(std::string str){
    int length = 0;
    for(int i=0; str[i]!= '\0';i++){
        length++;
    }
    return length;
}

bool checkDateRange(std::vector<int> beginDate, std::vector<int> endDate){
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
	// TODO: check that they have actually passed a valid date

	return true;
}

std::vector<int> getNextDay(std::vector<int> beginDate){
	
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
	std::vector<int> nextDay = {year, month, day};
	return nextDay;
}

void get_nearest_indexes(std::vector<std::string> strCurrentDay, double* grib_lats, double* grib_lons, long& numberOfPoints){
    /*
    There are several goals of this function
        - Open a grib file from the day passed
        - Grab the header to write out to the csv files
        - Find the nearest indexes for the corresponding stations
    */

    // concatenate the file path
    std::string filePath1 = filePath + strCurrentDay.at(0) + "/" + strCurrentDay.at(3) + "/";
    // check if the file path exists
    if(dirExists(filePath1) == false){
        fprintf(stderr, "Error: could not find directory %s", filePath1.c_str());
        exit(1);
    }

    // in the directory, find a suitable grib file to open and read the index
    std::string year = strCurrentDay.at(0);
    std::string month = strCurrentDay.at(1);
    std::string day = strCurrentDay.at(2);
    std::string hour, grib_file_path;
    FILE* f;
    for (int i=0; i<intHourRange; i++){
        hour = hours[i];
        grib_file_path = filePath1 + "/hrrr." + year + \
                         month + day + "." + hour + ".00.grib2"; 

        // now try to open the file. If it cannot, then go to the next iteration
        try{
            f = fopen(grib_file_path.c_str(), "rb");
            if (!f) throw(grib_file_path);
            else break;
        }
        catch (std::string file) {
            std::cout << "Could not open file " << file << std::endl;
            continue;
        }
    }

    // Once the file has been found and opened, it's time to read from it and
    // - get the header
    // - get the nearest indexes
    // - get the total parameter count ? maybe
    /////////////////
    // init 
    // codes_grib_multi_support_on(NULL);
    codes_handle * h = NULL; // use to unpack each layer of the file
    const double missing = 1.0e36;        // placeholder for when the value cannot be found in the grib file
    int msg_count = 0;  // KEY: will match with the layer of the passed parameters
    int err = 0;
    size_t vlen = MAX_VAL_LEN;
    char value_1[MAX_VAL_LEN];
    bool flag = true; 
    numberOfPoints=0;
    double *grib_values;
    std::string name_space = "parameter";
    unsigned long key_iterator_filter_flags = CODES_KEYS_ITERATOR_ALL_KEYS |
                                              CODES_KEYS_ITERATOR_SKIP_DUPLICATES;
    
    while((h = codes_handle_new_from_file(0, f, PRODUCT_GRIB, &err)) != NULL){
        msg_count++;

        if (blnParamArr[msg_count] == false) {
            codes_handle_delete(h);
            continue;
        } 

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
            strName = name, strValue = value_1;
            if(strName.find("name")!=std::string::npos){
                strValue.erase(remove(strValue.begin(), strValue.end(), ','), strValue.end());
                //strHeader.append(to_std::string(msg_count)+"_"+strValue);
                strnametosendout = strValue;
            }
            else if(strName.find("units")!=std::string::npos){
                //strHeader.append("("+strValue+")");
                strUnits = "(" + strValue + ")";
            }
        }
        strHeader = strnametosendout + " " + strUnits;
        vctrHeader.push_back(strHeader);
        
        // if this flag is set, then grab the nearest indexes
        if (flag) {
            flag = false;
            CODES_CHECK(codes_get_long(h, "numberOfPoints", &numberOfPoints), 0);
            CODES_CHECK(codes_set_double(h, "missingValue", missing), 0);
            grib_lats = (double*)malloc(numberOfPoints * sizeof(double));
            if(!grib_lats){
                fprintf(stderr, "Error: unable to allocate %ld bytes\n", (long)(numberOfPoints * sizeof(double)));
                exit(0);
            }
            grib_lons = (double*)malloc(numberOfPoints * sizeof(double));
            if (!grib_lons){
                fprintf(stderr, "Error: unable to allocate %ld bytes\n", (long)(numberOfPoints * sizeof(double)));
                std::free(grib_lats);
                exit(0);
            }
            grib_values = (double*)malloc(numberOfPoints * sizeof(double));
            if(!grib_values){
                fprintf(stderr, "Error: unable to allocate %ld bytes\n", (long)(numberOfPoints * sizeof(double)));
                std::free(grib_lats);
                std::free(grib_lons);
                exit(0);
            }
            CODES_CHECK(codes_grib_get_data(h, grib_lats, grib_lons, grib_values), 0);

            index_extraction(stationArr, grib_lats, grib_lons, numStations, numberOfPoints);
            
            std::free(grib_values);
        }
        codes_handle_delete(h);
    }
}

bool dirExists(std::string filePath){
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

void garbageCollection(){
    for(int i =0; i<numStations;i++){
        // delete each station's closest point array
        // delete each station's value array
        for(int j=0; j<intHourRange;j++){
            delete [] stationArr[i].values[j];
        }
        delete [] stationArr[i].values;
    }
    delete [] stationArr;
    delete [] hours;
    delete [] blnParamArr;
    
    for (int i =0; i<numStations; i++){
        sem_destroy(&valuesProtection[i]);
    }
    free(valuesProtection);
    if (sem_destroy(&barrier) == -1) {
        perror("sem_destroy");
        exit(EXIT_FAILURE);
    }
}








