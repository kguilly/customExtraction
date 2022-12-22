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
#include <sys/types.h>
#include <sys/stat.h>
#include <map>
#include <time.h>
#include <cassert>
#include "semaphore.h"
#define MAX_VAL_LEN 1024
using namespace std;

/* Timing variables */
struct timespec startTotal;
struct timespec endTotal;
double totalTime;

vector<int> beginDay = {2019, 3, 1}; // arrays for the begin days and end days. END DAY IS NOT INCLUSIVE.  
                                    // when passing a single day, pass the day after beginDay for endDay
                                    // FORMAT: {yyyy, mm, dd}
vector<int> endDay = {2019, 3, 2};

vector<int> arrHourRange = {12,19}; // array for the range of hours one would like to extract from
                                 // FORMAT: {hh, hh} where the first hour is the lower hour, second is the higher
                                 // accepts hours from 0 to 23 (IS INCLUSIVE)

int intHourRange; 

string filePath = "/home/kaleb/Desktop/SchoolFiles/Y4S1/REU/gribData/";  // path to "data" folder. File expects structure to be: 
                                        // .../data/<year>/<yyyyMMdd>/hrrr.<yyyyMMdd>.<hh>.00.grib2
                                        // for every hour of every day included. be sure to include '/' at end

string writePath = "/home/kaleb/Desktop/WRFDataThreaded/"; // path to write the extracted data to,
                                                    // point at a WRFData folder

// Structure for holding the selected station data. Default will be the 5 included in the acadmeic
// paper: "Regional Weather Forecasting via Neural Networks with Near-Surface Observational and Atmospheric Numerical Data."
struct Station{
    string name;
    string state;
    string county;
    int fipsCode;
    float lat;
    float lon;
    double **values; // holds the values of the parameters. Index of this array will 
                    // correspond to index if the Parameter array. This will be a single hour's data
    map<string, vector<double>> dataMap; // this will hold the output data. structured as {"yyyymmddHH" : [param1, param2, ...]}}

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

// function to handle arguments passed. Will either build the Station array and/or paramter array
// based off of the arguments passed or will build them with default values. Will also construct
// the hour array based on passed beginning and end hours
void handleInput(int, char**);

void semaphoreInit();

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

/* function to write the data in the staion maps to a .csv file */
void* writeData(void*);

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

    // the data maps are finished being built, now its time to write the maps to csv files
    // since each station has their own map, we can thread the stations without having to 
    // use semaphores. 
    pthread_t *writeThreads = (pthread_t*)malloc(numStations * sizeof(pthread_t)); // will be freed at the end of this iteration
    if(!writeThreads){
        fprintf(stderr, "Error: unable to allocate %ld bytes for threads.\n", (long)(numStations*sizeof(pthread_t)));
        exit(0);
    }
    writeThreadArgs *arg = new writeThreadArgs[numStations];
    int threaderr;
    for(int i=0; i<numStations;i++){
        arg[i].station = &stationArr[i];
        threaderr= pthread_create(&writeThreads[i], NULL, &writeData, &arg[i]);
        if(threaderr){
            assert(0);
            return 1;
        }
    }
    for(int i=0;i<numStations;i++){
        pthread_join(writeThreads[i], NULL);
    }
    std::free(writeThreads);
    delete [] arg;




    // print out all the elements in all the station's data maps
    for (int i=0; i<numStations;i++){
        Station station = stationArr[i];
        cout << "\n\nSTATION: " << station.name << endl;
        for(auto itr = station.dataMap.begin(); itr != station.dataMap.end(); ++itr){
            cout << itr->first << '\t';
            for (auto i = 0; i<itr->second.size(); i++){
                 cout << itr->second.at(i) << " ";
            }
            cout << endl;
        }
    }


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
    }
    else{
        // build the default staion array and parameter array
        // potential improvement: for each month, run the file and make new parameter array based off
        //                        of what the file returns
        numParams = 133;
        numStations = 6;
        

        
        int itr = 0;
        objparamArr = new Parameter[numParams];
        for (auto param : tempParamarr){
            *(objparamArr + itr) = param;
            itr++;
        }

        int layer =0;
        for (int i = 0; i<numParams; i++){
            layer = objparamArr[i].layer;
            blnParamArr[layer] = true;
        }        

        stationArr = new Station[numStations];

        Station bmtn, ccla, farm, huey, lxgn, lafy;
        bmtn.name = "BMTN";bmtn.lat = 36.91973;bmtn.lon = -82.90619; bmtn.state = "Kentucky";bmtn.county = "Harlan"; bmtn.fipsCode = 21095;
        *(stationArr+0) = bmtn;
        ccla.name = "CCLA";ccla.lat = 37.67934; ccla.lon = -85.97877; ccla.state = "Kentucky";ccla.county = "Hardin"; ccla.fipsCode = 21157;
        *(stationArr+1) = ccla;
        farm.name = "FARM"; farm.lat = 36.93; farm.lon = -86.47; farm.state = "Kentucky"; farm.county = "Warren"; farm.fipsCode = 21227;
        *(stationArr+2) = farm;
        huey.name = "HUEY"; huey.lat = 38.96701; huey.lon = -84.72165; huey.state = "Kentucky"; huey.county = "Boone"; huey.fipsCode = 21015;
        *(stationArr+3) = huey;
        lxgn.name = "LXGN"; lxgn.lat = 37.97496; lxgn.lon = -84.53354; lxgn.state = "Kentucky"; lxgn.county = "Fayette"; lxgn.fipsCode = 21067;
        *(stationArr+4) = lxgn;
        lafy.name = "Lafayette", lafy.lat = 30.216667; lafy.lon = -92.033333; lafy.state = "Louisiana"; lafy.county = "Lafayette"; lafy.fipsCode = 22055;
        *(stationArr+5) = lafy;



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
        printf("Error: could not open filename %s in directory %s", fileName.c_str(), file.c_str());
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

void* writeData(void*arg){

    struct writeThreadArgs writeThreadArgs = *(struct writeThreadArgs*)arg;

    Station*station = (writeThreadArgs).station;

    // the entire file will be written to in one iteration. We will append the strings
    // of all days together until the next day is reached, at which point we will write out
    string output;
    // string prevDay = station->dataMap.begin()->first;
    // prevDay = prevDay.substr(6,2);
    // loop through each key of the file, every time you run into a new day, 
    // make a new file
    for(auto itr = station->dataMap.begin(); itr!=station->dataMap.end();++itr){
        string hourlyDate = itr->first;
        string hour = hourlyDate.substr(8,2);
        string day =  hourlyDate.substr(6,2);
        string month =  hourlyDate.substr(4,2);
        string year =  hourlyDate.substr(0,4);

        // if the path does not exist, make the path
        if(!dirExists(writePath)){
            sem_wait(&pathCreationSem);
            if(mkdir(writePath.c_str(), 0777) == -1){
                sem_post(&pathCreationSem);
                cerr << "Error: " << strerror(errno) << endl;
            }
            sem_post(&pathCreationSem);
        }

        // make the directory structure:    yyyy/yyyymmdd/station.yyyymmdd.csv
        //                              ex. 2019/20190101/BMTN.20190101.csv
        // for each year, make a folder
        string yearWritePath = writePath+year+"/";
        if(!dirExists(yearWritePath)){
            sem_wait(&pathCreationSem);
            if(mkdir(yearWritePath.c_str(),0777)==-1){
                sem_post(&pathCreationSem);
                cerr << "Error: " << strerror(errno) << endl;
            }
            sem_post(&pathCreationSem);
        }
        // for each day, make a folder
        string dayWritePath = yearWritePath + year+month+day + "/";
        if(!dirExists(dayWritePath)){
            sem_wait(&pathCreationSem);
            if(mkdir(dayWritePath.c_str(), 0777)==-1){
                sem_post(&pathCreationSem);
                cerr << "Error: " << strerror(errno) << endl;
            }
            sem_post(&pathCreationSem);
        }

        // check to see if the file for this day and station exists
        string fullFilePath = dayWritePath + station->name + "." +year+month+day+".csv";
        if(!std::filesystem::exists(fullFilePath)){

            // give the new file the appropriate headers
            ofstream outfile;
            outfile.open(fullFilePath, std::ios_base::app);
            outfile << "Year, Month, Day, Hour, State, County, Fips Code, ";
            
            // append the name of each parameter to the headings of the files
            for(int j=0;j<numParams;j++){
                outfile << " " << objparamArr[j].name << " (" << objparamArr[j].units << ")" <<",";
            }
            outfile << "\n";
            outfile.close();
        }
    
        // append information to the output string
        output.append(year +","+ month +","+ day + "," + hour + "," + station->state + "," + station->county + "," + std::to_string(station->fipsCode) + ",");
        for(auto j=0; j<itr->second.size();j++){
            output.append(std::to_string(itr->second.at(j))+",");
        }
        output.append("\n");

        string finalHour = hours[intHourRange-1];
        // if the current hour that we're on is equal to the last hour in the hour range, send the output to the day's file
        if(strcmp(hour.c_str(), finalHour.c_str())==0){
            ofstream csvFile;
            csvFile.open(fullFilePath, std::ios_base::app);
            csvFile << output << "\n";
            output = "";
            csvFile.close();
        }
    }
    pthread_exit(0);
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
    free(mapProtection);
    for (int i =0; i<numStations; i++){
        sem_destroy(&mapProtection[i]);
    }

}