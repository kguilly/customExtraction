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
#include <string>
#include <stdlib.h>

using namespace std;
struct station{
    std::string name;
    string state;
    string county;
    string fips;
    float lat;
    float lon;
};
// <FIPS, state> 
map<string, string> stateMap;
// <FIPS, county>
map<string, string> countyMap;

struct coordinates{
    long latitude;
    long longitude;
};
// <FIPS, (lat, longitude)>
map<string, coordinates> coordinatesMap;
station *stationArr;

int main(){
    std::string strcountyFipsCodes;
    std::ifstream filecountyFipsCodes;
    filecountyFipsCodes.open("./countyInfo/countyFipsCodes.txt");
    if(!filecountyFipsCodes){
        std::cerr << "Error: the FIPS file could not be opened.\n";
        exit(1);
    }  
    bool readstates = false;
    bool readcounties = false;

    std::string strCountyInfo;
    std::string strStateInfo;
    while(getline(filecountyFipsCodes, strcountyFipsCodes)){
        
        // where to start reading county info from
        if(strcmp(strcountyFipsCodes.c_str(), " ------------    --------------")==0){
            readcounties = true;
            continue;
        }
        // this is the line where to stop reading states from
        if(strcmp(strcountyFipsCodes.c_str(), " county-level      place")==0){
            readstates = false;
        }
        // this is the location where to start reading states from
        if(strcmp(strcountyFipsCodes.c_str(), "   -----------   -------")== 0){
            readstates = true;
            continue;
        }

        // load all the states and their fips information into an obj
        if(readstates){
            // read strcountyfipscodes string, while it equals a number,
            // store into the map as the key (fips code), while it equals
            // a string, store into the map as the value (statename)
            if(strcountyFipsCodes.length() < 3){
                // do nothing, this is not a necessary line
                strcountyFipsCodes.clear();
            }else{
                string delimiter = ",";
                string strfips = strcountyFipsCodes.substr(0,strcountyFipsCodes.find(delimiter));
                string strState = strcountyFipsCodes.erase(0, strcountyFipsCodes.find(delimiter)+delimiter.length());

                stateMap.insert({strfips, strState});
                
            }

        }

        // load all the county information into an objarray
        if(readcounties){
            string delimiter = ",";
            string strFips = strcountyFipsCodes.substr(0,strcountyFipsCodes.find(delimiter));
            string strCountyName = strcountyFipsCodes.erase(0, strcountyFipsCodes.find(delimiter)+delimiter.length());

            countyMap.insert({strFips, strCountyName});   
        }
    }

    filecountyFipsCodes.close();

    string strLine;
    ifstream fileCoordinates;
    fileCoordinates.open("./countyInfo/countyFipsandCoordinates.csv");
    if(!fileCoordinates){
        cerr<<"Error: the coordinates file could not be opened\n";
        exit(1);
    }
    vector<string> row;
    while(getline(fileCoordinates, strLine)){
        row.clear();
        if(strLine.length() > 100) continue; // I put the link to github in one line

        stringstream s(strLine);
        string currLine;
        while(getline(s, currLine, ',')){
            row.push_back(currLine);
        }
        // row[0] = fips, row[1] = county, row[2] = longitude, row[3] = latitude
        coordinates c; c.latitude = stol(row.at(3)), c.longitude = stol(row.at(2));
        coordinatesMap.insert({row.at(0), c});
        
    }
    // match the counties to their respecitve states by matching fips codes
    // throw out counties from alaska and hawaii (02 and 15)
    
    // now read from file countyfipsandcoordinates.csv and map the 
    int arridx = 0;
    map<string, string>::iterator stateItr;
    map<string, coordinates>::iterator coordinatesItr;
    for(auto itr = countyMap.begin(); itr!=countyMap.end(); ++itr){
        string countyFips = itr->first;
        string stateFips = countyFips.substr(0,2);
        stateItr = stateMap.find(stateFips);
        coordinatesItr = coordinatesMap.find(countyFips);
        coordinates c = coordinatesItr->second;

        if(stateItr == stateMap.end()){
            cout << "state not found in map" << endl;
            exit(0);
        }else if(stateItr->first == "02" || stateItr->first == "15"){ // this is alaska or hawaii, do not include
            continue;
        }
        station *newstation = (station*)malloc(1*sizeof(station));
        newstation->fips = stoi(countyFips);
        newstation->name = itr->second;
        newstation->county = itr->second;
        newstation->state = stateItr->second;
        newstation->lat = c.latitude;
        newstation->lon = c.longitude;
        *(stationArr+arridx) = *newstation;
        arridx++;
        free(newstation);
    
    }

    // print out the station array
    for(int i=0;i<sizeof(stationArr)/sizeof(station);i++){
        station s = stationArr[i];
        cout << s.fips << " " << s.state << " " <<s.county << " " << s.lat << " " << s.lon << "\n";
    }

}
