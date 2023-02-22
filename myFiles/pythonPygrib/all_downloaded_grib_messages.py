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
        self.csv_file_path = "/home/kaleb/Desktop/GribMessages/"
        self.repository_path = "/home/kaleb/Documents/GitHub/customExtraction/" # POINT TO CURRENT REPO: should end in "/customExtraction/"
        self.grib2_file_path = "/mnt/wrf/"
        self.parameter = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104,  105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164,165,167,168,169]
        self.start_year = "2018"

    def main(self):
        # self.readArgs()
        # csvFiles = self.findFiles()
        # if len(csvFiles) < 1:
            # print("Error, the file list is less than one.")
            # exit(0)
        
        for year_folder in sorted(os.listdir(self.grib2_file_path)):
                # som
                day_dir = self.grib2_file_path + year_folder + '/'
                currMonth = ""
                for day_folder in sorted(os.listdir(day_dir)):
                    
                    file_dir = day_dir+day_folder + '/'
                    # grab the messages from the first file then go to the next month
                    df = pd.DataFrame(columns=["Layer", "Name", "Units", "Level", "TypeOfLevel"])
                    year = day_folder[0:4]
                    # print("year: %s, start_year: %s"%(year, self.start_year))
                    month = day_folder[4:6]
                    if month == currMonth:
                        continue
                    if int(year) <= int(self.start_year):
                        continue
                    currMonth = month
                    write_path_full = self.csv_file_path + "/GribMessages_"+year+"_"+month+"_.csv"
                    # print(sorted(os.listdir(file_dir)))
                    print(file_dir)
                    
                    try:
                        hour_file = sorted(os.listdir(file_dir))[0]
                    except:
                        continue
                    grib_file_path = file_dir + hour_file
                    self.read_data(grib_file_path, df=df)

                    Path(self.csv_file_path).mkdir(parents=True, exist_ok=True)
                    try:
                        df.to_csv(path_or_buf=write_path_full,index=False,header=True)
                    except:
                        print("could not write df to file")
            
            
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
            try:
                tmpmsgs = grib[p]
            except:
                print("only %s messages in %s" % (p-1, data_path))
                break

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


