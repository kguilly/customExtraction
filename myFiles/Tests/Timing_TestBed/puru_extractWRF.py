'''
The copy of this file was made for the purposes of timing the file for a select
number of stations for a select number of hours
'''

import csv
import pygrib
import numpy as np
from datetime import date, timedelta
from pathlib import Path
import time
import pandas as pd

class extractor():
    def __init__(self):
        self.version = '1.0'
        self.info = 'extract location point data from WRF dataset'

        self.rw = "r"
        self.repository_path = "/home/kaleb/Documents/GitHub/customExtraction/"
        self.wrfOutput_path = self.repository_path + "myFiles/Tests/Timing_TestBed/WRFoutput/wrfOutput.csv"
        self.data_path = "/media/kaleb/extraSpace/wrf/"  # data source (grib file location)
        self.write_data_path = "/home/kaleb/Desktop/puru_output/" # path to store extracted data
        self.loc_info = ["BMTN", "CCLA", "FARM", "HUEY", "LXGN"]
        self.loc = [(36.91973, -82.90619), (37.67934, -85.97877), (36.93, -86.47), (38.96701, -84.72165), (37.97496, -84.53354)]
        #self.parameter =[16, 17, 102, 105, 6, 5, 92, 71, 72, 9, 10, 55, 56, 57]
        #self.parameter_info = ['u_500hpa', 'v_500hpa', 'cloud_base_pressure', 'cloud_top_pressure', '3000_h', '1000_h', 'ground_mis', 'u_10m', 'v_10m', 'u_250hpa', 'v_250hpa', 'u_80m', 'v_80m', 'surface_pressure']
        self.parameter =[9, 36, 37, 71, 75, 123]
        # self.parameter_info = [	"ATT1", "ATT2", "ATT3", "ATT4", "ATT5", "ATT6", "ATT7", "ATT8", "ATT9", "ATT10", "ATT11", "ATT12", "ATT13", "ATT14", "ATT15", "ATT16", "ATT17", "ATT18", "ATT19", "ATT20", "ATT21", "ATT22", "ATT23", "ATT24", "ATT25", "ATT26", "ATT27", "ATT28", "ATT29", "ATT30","ATT31", "ATT32", "ATT33", "ATT34", "ATT35", "ATT36", "ATT37", "ATT38", "ATT39", "ATT40", "ATT41", "ATT42", "ATT43", "ATT44", "ATT45","ATT46", "ATT47", "ATT48", "ATT49", "ATT50", "ATT51", "ATT52", "ATT53", "ATT54", "ATT55", "ATT56", "ATT57", "ATT58", "ATT59", "ATT60", "ATT61", "ATT62", "ATT63", "ATT64", "ATT65", "ATT66", "ATT67", "ATT68", "ATT69", "ATT70", "ATT71", "ATT72", "ATT73", "ATT74", "ATT75", "ATT76", "ATT77", "ATT78", "ATT79", "ATT80", "ATT81", "ATT82", "ATT83", "ATT84", "ATT85", "ATT86", "ATT87", "ATT88", "ATT89", "ATT90", "ATT91", "ATT92", "ATT93", "ATT94", "ATT95", "ATT96", "ATT97", "ATT98", "ATT99", "ATT100", "ATT101", "ATT102", "ATT103", "ATT104", "ATT105",	"ATT106", "ATT107", "ATT108", "ATT109", "ATT110", "ATT111",	"ATT112", "ATT113", "ATT114", "ATT115", "ATT116","ATT117", "ATT118", "ATT119", "ATT120", "ATT121", "ATT122", "ATT123","ATT124", "ATT125", "ATT126", "ATT127", "ATT128", "ATT129", "ATT130","ATT131", "ATT132"]
        #self.data_path = "N:/weather/WRF/"
        #self.write_data_path = "E:/extract_weather_parameter/Data/WRF/"
        
        self.start_date = date(2022,1,1)  # start date
        self.end_date = date(2022, 1,2)   # end date

    def daterange(self, start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)

    def hourrage(self):
        for n in range(24):
            yield "%02d"%n

    
    def read_loop(self):
        every_county_df = pd.read_csv(self.wrfOutput_path)
        self.loc_info = every_county_df['FIPS']
        self.loc = []
        for index, row in every_county_df.iterrows():
            self.loc.append((row['lat (urcrnr)'], row['lon (urcrnr)']))

        for single_date in self.daterange(self.start_date, self.end_date):
            for single_hour in self.hourrage():
                data_path = self.data_path+single_date.strftime("%Y")+"/"+single_date.strftime("%Y%m%d")+"/"+"hrrr."+single_date.strftime("%Y%m%d")+"."+single_hour+".00.grib2"
                write_path = self.write_data_path+single_date.strftime("%Y")+"/"+single_date.strftime("%Y%m%d")+"/"
                data = self.read_data(data_path, write_path)
                self.write_data(data, write_path, single_date,single_hour)
    '''
    def read_data(self, data_path, write_path):
        grib = pygrib.open(data_path)

        data_dic = {}
        for l in self.loc_info:
            data_dic[l] = []

        for p in self.parameter:
            print(p)
            tmpmsgs = grib[p]
            lt, ln = tmpmsgs.latlons() # lt - latitude, ln - longitude
            data = tmpmsgs.values      

            for (l_lt, l_ln), l_info in zip(self.loc, self.loc_info):
                
                l_lt_m = np.full_like(lt, l_lt)
                l_ln_m = np.full_like(ln, l_ln)
                dis_mat = (lt-l_lt_m)**2+(ln-l_ln_m)**2
                p_lt, p_ln = np.unravel_index(dis_mat.argmin(), dis_mat.shape)
                value = data[p_lt, p_ln]

                data_dic[l_info].append(value)

        return data_dic   
    '''
    def read_data(self, data_path, write_path):
        grib = pygrib.open(data_path)
        print(data_path)
        data_dic = {}
        p_lt_dic = {}
        p_ln_dic = {}
        flag = 1
        for l in self.loc_info:
            data_dic[l] = []
            p_lt_dic[l] = []
            p_ln_dic[l] = []

        for p in self.parameter:
            #print(p)
            tmpmsgs = grib[p]
            lt, ln = tmpmsgs.latlons() # lt - latitude, ln - longitude
            data = tmpmsgs.values      

            for (l_lt, l_ln), l_info in zip(self.loc, self.loc_info):
                #if it is first time, extract index
                if(flag == 1):
                    l_lt_m = np.full_like(lt, l_lt) # make an array of shape (grib lats) and fill with (station lats)
                    l_ln_m = np.full_like(ln, l_ln)
                    dis_mat = (lt-l_lt_m)**2+(ln-l_ln_m)**2 # find the closest point
                    p_lt, p_ln = np.unravel_index(dis_mat.argmin(), dis_mat.shape)
                    p_lt_dic[l_info].append(p_lt)
                    p_ln_dic[l_info].append(p_ln)                
                
                value = data[p_lt_dic[l_info], p_ln_dic[l_info]]
                data_dic[l_info].append(value[0])
                
            flag = 0

        return data_dic  

    def write_data(self, data, w_data_path,single_date,single_hour):

        for k, v in data.items():
            w_path = w_data_path
            Path(w_path).mkdir(parents=True, exist_ok=True)
    
            w_path_file = w_data_path+str(k)+"."+single_date.strftime("%Y%m%d")+".csv"
            with open(w_path_file, 'a', newline='') as file_out:
                writer = csv.writer(file_out, delimiter=',')
                write_list = []
                #write_list.append(single_date.strftime("%Y/%m/%d")+":"+single_hour)
                write_list.append(single_date.strftime("%Y"))
                #write_list.append(",")
                write_list.append(single_date.strftime("%m"))
                #write_list.append(",")
                write_list.append(single_date.strftime("%d"))
                #write_list.append(",")
                write_list.append(single_hour)
                #write_list = write_list+ ","+ single_date.strftime("%m")+","+single_date.strftime("%d")+","+single_hour
                write_list = write_list + v
                writer.writerow(write_list)

e = extractor()
start_time = time.time() # start time
e.read_loop()
end_time = time.time()#end time
total_time_s = end_time - start_time
total_time_ms = total_time_s * 1000
print("Total time in Seconds: ", total_time_s, "\nTotal time in ms: ", total_time_ms)
#print(e.parameter_info)
