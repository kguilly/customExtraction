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

class extraction():
    def __init__(self):
        self.info = 'extract wrf information from a grib2 file for a given location'
        self.wrf_data_path = "/media/kaleb/extraSpace/wrf/"
        self.write_path = "/home/kaleb/Desktop/pythonWRFOutput_1-26/"
        self.repository_path = "/home/kaleb/Documents/GitHub/customExtraction/" # POINT TO CURRENT REPO: should end in "/customExtraction/"

        self.passedFips = ["22107", "22007"]
        self.station_fips = []
        # how to write tuples to an array? 
        self.st_latlons = []
        # need to configure unique names
        self.st_names = []
        self.st_states = []
        self.st_stateabbrevs = []
        self.st_indexes = []
        self.parameter = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104,  105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146]
        self.parameter_layerNum = []
        self.parameter_names = []
        self.parameter_units = []
        self.parameter_levels = []
        self.parameter_typeofLevels = []
        
        self.start_date = date(2021, 4, 28)
        self.end_date = date(2021,5,2)

    def daterange(self, start_date, end_date):
        for n in range(int((end_date- start_date).days)):
            yield start_date + timedelta(n)
            
    def hourrange(self):
        for n in range(24): 
            yield "%02d"%n


    def readArgs(self):
        if len(sys.argv) > 1:
            fips_flag = False
            fips_index = 0
            for i in range(1, len(sys.argv)):
                if sys.argv[i] == "--fips":
                    fips_flag = True
                    fips_index = i
                
            if fips_flag:
                # if there is a fips flag then we want to put each of the fips codes passed
                # into an array and then call the sentinel func
                self.passedFips = []
                for i in range(fips_index+1, len(sys.argv)):
                    arg = sys.argv[i]
                    if len(arg) == 5:
                        # this is the correct length, add it to the array
                        self.passedFips.append(arg)
                    else:
                        # this is the incorrect length, exit
                        print("Error, incorrect length of fips argument passed, please try again.")
                        exit(0)

        # we have built the fips arg array, now call the sentinel script to make the wrfOutput
        gg.county(fips=self.passedFips)
        # scrape wrfOutput for the values we need
        wrfOutputdirectory = self.repository_path + "myFiles/pythonPygrib/WRFoutput/wrfOutput.csv"
        countycsv = pd.read_csv(wrfOutputdirectory)
        countycsv = countycsv.reset_index()
        for index, row in countycsv.iterrows():
            self.station_fips.append(str(row['FIPS']))
            self.st_names.append(row['county']+str(row['countyGridIndex']))
            self.st_indexes.append(row['countyGridIndex'])
            # make a tuple of double from the lats and lons
            lat = float(row['lat'])
            lon = float(row['lon'])
            latlons = [lat,lon]
            latlons = tuple(latlons)
            self.st_latlons.append(latlons)

    def getstates(self):
        file_path = self.repository_path + "myFiles/countyInfo/us-state-ansi-fips2.csv"
        df = pd.read_csv(file_path, dtype=str)
        for fips in self.station_fips:
            state_fips = fips[0:2]
            for index, row in df.iterrows():
                if row['st'] == state_fips:
                    self.st_stateabbrevs.append(row['stusps'])
                    self.st_states.append(row['stname'])
    
    def read_loop(self):
        self.readArgs()
        self.getstates()
        headerFlag = True
        # file_name = "HRRR_22_LA_"+self.start_date.strftime("%Y%m")+".csv"
        for single_date in self.daterange(self.start_date, self.end_date):
            for single_hour in self.hourrange():
                data_path = self.wrf_data_path+single_date.strftime("%Y")+"/"+single_date.strftime("%Y%m%d")+"/"+"hrrr."+single_date.strftime("%Y%m%d")+"."+single_hour+".00.grib2"
                write_path = self.write_path+'/'+self.station_fips[0]+'/'+single_date.strftime("%Y")+"/"# +single_date.strftime("%Y%m")+"/"
                if(headerFlag):
                    data = self.read_data(data_path, write_path, headerFlag=True)
                    headerFlag = False
                else:
                    data = self.read_data(data_path, write_path, headerFlag=False)

                itr=0
                for fips in self.passedFips:
                    write_path1 = self.write_path+fips+'/'+single_date.strftime("%Y")+'/'
                    file_name_1 = "HRRR_"+fips[0:2]+self.st_stateabbrevs[itr]+single_date.strftime("%Y%m")+".csv"
                    self.write_data(data,write_path1, single_date, single_hour, file_name_1, fips=fips)
                    itr+=1
        itr=0
        for fips in self.passedFips:
            write_path2 = self.write_path+fips+'/'+single_date.strftime("%Y")+'/'
            file_name_2 = "HRRR_"+fips[0:2]+self.st_stateabbrevs[itr]+single_date.strftime("%Y%m")+".csv"
            self.write_header(write_path2, file_name_2)
            itr+=1


    def read_data(self, data_path, write_path, headerFlag):
        grib = pygrib.open(data_path)
        print(data_path)
        data_dic = {}
        p_lt_dic = {}
        p_ln_dic = {}
        flag = 1
        for l in self.st_names:
            data_dic[l] = []
            p_lt_dic[l] = []
            p_ln_dic[l] = []

        for p in self.parameter:
            tmpmsgs = grib[p]
            lt, ln = tmpmsgs.latlons()
            # print("Layer: %s Name: %s  Units: %s  Level: %s (%s)" % (p, tmpmsgs.name, tmpmsgs.units, tmpmsgs.level, tmpmsgs.typeOfLevel))
            if headerFlag:
                self.parameter_layerNum.append(p)
                self.parameter_names.append(tmpmsgs.name)
                self.parameter_units.append(tmpmsgs.units)
                self.parameter_levels.append(tmpmsgs.level)
                self.parameter_typeofLevels.append(tmpmsgs.typeOfLevel)

            data = tmpmsgs.values

            for(l_lt, l_ln), l_info in zip(self.st_latlons, self.st_names): ### should fix
                if flag == 1:
                    l_lt_m = np.full_like(lt, l_lt) # make an array of shape (grib lats) and fill with station lats
                    l_ln_m = np.full_like(ln, l_ln)
                    dis_mat = (lt-l_lt_m)**2 + (ln - l_ln_m)**2 # find the closest point with linear alg
                    p_lt, p_ln = np.unravel_index(dis_mat.argmin(), dis_mat.shape)
                    p_lt_dic[l_info].append(p_lt)
                    p_ln_dic[l_info].append(p_ln)

                value = data[p_lt_dic[l_info], p_ln_dic[l_info]]
                data_dic[l_info].append(value[0])
            flag = 0
        
        return data_dic

    def write_data(self, data, w_data_path, single_date, single_hour, file_name, fips):
        # use itr to find the correct station's data
        itr = 0
        # find the separate 
        
        for k,v in data.items():
            # if the station name at this index does not match the passed fips,
            # increment the itr and continue
            if fips != self.station_fips[itr]:
                itr+=1
                continue

            w_path = w_data_path 
            Path(w_path).mkdir(parents=True, exist_ok=True)

            
            w_path_file = w_path + file_name
            with open(w_path_file, 'a', newline='') as file_out:
                writer = csv.writer(file_out, delimiter=',')
                write_list = []
                
                write_list.append(single_date.strftime("%Y"))
                write_list.append(single_date.strftime("%m"))
                write_list.append(single_date.strftime("%d"))
                write_list.append(single_hour)
                write_list.append(self.st_states[itr])
                write_list.append(self.st_names[itr])
                write_list.append(self.st_indexes[itr])
                write_list.append(self.station_fips[itr])
                write_list.append(self.st_latlons[itr][0])
                write_list.append(self.st_latlons[itr][1])
                write_list=write_list+v
                writer.writerow(write_list)

                itr+=1

    def write_header(self, w_data_path, file_name):
        file_path = w_data_path + file_name
        df = pd.read_csv(file_path, header=None)
        print(df)
        header = ['Year', 'Month', 'Day', 'Hour', 'State', "County", "GridIndex", "FIPS Code", "Lat", "Lon(-180 to 180)"]
        for i in range(len(self.parameter_names)):
            name = self.parameter_names[i]
            unit = self.parameter_units[i]
            header.append(name + '('+unit+')')
        df.to_csv(file_path, header=header)

start_time = time.time()
e = extraction()
e.read_loop()
time_sec = time.time() - start_time
h=0
m=0
if(time_sec // 60 > 1):
    m = time_sec // 60
    time_sec = time_sec - (m*60)
if(m//60 > 1):
    h = m // 60
    m = m - (h*60)
print("\n\nRuntime:\nHours: %d, Minutes: %d, Seconds: %s\n"%(h, m, time_sec))


