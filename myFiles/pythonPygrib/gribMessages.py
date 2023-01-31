'''
This script will function alike the scripts to download each county's data,
excpet I will pass the argument for fips codes that will call sentinel hub
to run the data, then read the outputted csv 

TO START; I should only read the csv

TODO: implement ability to pass multiple years and multiple months and multiple states

###########################33
debug with pdb command line arg:
python3 -m pdb extractWRF_passFips.py --fips <fips codes> 
'''
import time
import csv
import pygrib
import numpy as np
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
import sys
import geo_grid as gg
import os

class extraction():
    def __init__(self):
        self.info = 'extract wrf information from a grib2 file for a given location'
        self.csv_file_path = "/home/kaleb/Desktop/cppWRFExtract_1-26/"
        self.repository_path = "/home/kaleb/Documents/GitHub/customExtraction/" # POINT TO CURRENT REPO: should end in "/customExtraction/"
        self.grib2_file_path = "/media/kaleb/extraSpace/wrf/"
        self.parameter = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104,  105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146]

    def main(self):
        self.readArgs()
        csvFiles = self.findFiles()
        if len(csvFiles) < 1:
            print("Error, the file list is less than one.")
            exit(0)
        
        for file in csvFiles:
            if(file.find("daily_monthly")!=-1 or file.find(".~lock.")!=-1):
                continue
            print(file)

            df = pd.DataFrame(columns=["Layer", "Name", "Units", "Level", "TypeOfLevel"])
            year = file[len(file)-10:len(file)-6]
            month = file[len(file)-6:len(file)-4]
            write_path_full = self.csv_file_path+"GribMessages/GribMessages_"+year+"_"+month+".csv"
            if((Path(write_path_full).is_file())):
                continue
            
            grib2_year_path = self.grib2_file_path + year + '/'
            day_folder = ""
            for gribfolders in sorted(os.listdir(grib2_year_path)):
                if(gribfolders.rfind(year+month)!=-1):
                    day_folder = grib2_year_path+gribfolders+'/'
                    break
            
            hour_file = sorted(os.listdir(day_folder))        
            grib_file = day_folder + hour_file[0]
            # print(grib_file)
            self.read_data(grib_file, df=df)
            filepathsep = self.csv_file_path.split('/')
            newfilepath = ''
            for i in range(len(filepathsep)-1):
                if filepathsep[i].rfind("Hourly") != -1:
                    break
                newfilepath += filepathsep[i] + '/'
            newfilepath += "GribMessages/" 
            Path(newfilepath).mkdir(parents=True, exist_ok=True)
            try:
                write_path_1 = newfilepath + "GribMessages_"+year+"_"+month+".csv"
                df.to_csv(path_or_buf=write_path_1,index=False,header=True)
            except:
                print("could not write df to file")
                continue
            
    def readArgs(self):
        if(len(sys.argv)>1):
            for i in range(1, len(sys.argv)):
                if sys.argv[i]=="--repo_path":
                    self.repository_path = sys.argv[i+1]
                elif sys.argv[i]=="--wrf_path":
                    self.csv_file_path = sys.argv[i+1]
                elif sys.argv[i]=="--grib2_path":
                    self.grib2_file_path = sys.argv[i+1]

    def findFiles(self):
        fipsFolders = sorted(os.listdir(self.csv_file_path))
        # print(fipsFolders)
        csvFiles = []
        
        for ff in fipsFolders:
            fipsPath = self.csv_file_path + ff + '/'
            # print(fipsPath)
            if not os.path.isdir(fipsPath):
                continue
            
            fipsPath_1 = sorted(os.listdir(fipsPath))
            for year in fipsPath_1:
                yearPath = fipsPath + year + '/'
                if not os.path.isdir(yearPath):
                    continue
                
                yearPath_1 = sorted(os.listdir(yearPath))
                for file in yearPath_1:
                    # print(file)
                    fullfilePath = yearPath + file
                    # print(fullfilePath)
                    found = fullfilePath.rfind('.csv')
                    if found != -1:
                        csvFiles.append(fullfilePath)
        return csvFiles

    def read_data(self, data_path, df=pd.DataFrame()):
        grib = pygrib.open(data_path)
        idx=0
        for p in self.parameter:
            tmpmsgs = grib[p]
            lt, ln = tmpmsgs.latlons()
            # print("Layer: %s Name: %s  Units: %s  Level: %s (%s)" % (p, tmpmsgs.name, tmpmsgs.units, tmpmsgs.level, tmpmsgs.typeOfLevel))
            df.at[idx, 'Layer'] = p
            df.at[idx, 'Name'] = tmpmsgs.name
            df.at[idx, 'Units'] = tmpmsgs.units
            df.at[idx, 'Level'] = tmpmsgs.level
            df.at[idx, 'TypeOfLevel'] = tmpmsgs.typeOfLevel
            
            
            idx+=1
   

start_time = time.time()
e = extraction()
e.main()
time_sec = time.time() - start_time
h=0
m=0
if(time_sec // 60 > 1):
    m = time_sec // 60
    time_sec = time_sec - (m*60)
if(m//60 > 1):
    h = m // 60
    m = m - (h*60)
# print("\n\nRuntime:\nHours: %d, Minutes: %d, Seconds: %s\n"%(h, m, time_sec))


