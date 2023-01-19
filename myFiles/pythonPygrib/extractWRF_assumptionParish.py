import csv
import pygrib
import numpy as np
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

class extraction():
    def __init__(self):
        self.info = 'extract wrf information from a grib2 file for a given location'
        self.wrf_data_path = "/home/kaleb/Desktop/weekInputData/"
        self.write_path = "/home/kaleb/Desktop/pythonWRFOutput/"

        self.arrfips = ["22007", "22007", "22007", "22007", "22007", "22007", "22007", "22007", "22007", "22007", "22007"]
        self.st_latlons = [(29.79139875,-91.19708816666667),(29.90141925,-91.19708816666667),(30.01143975,-91.19708816666667),(29.68137825,-91.07248849999999),(29.79139875,-91.07248849999999),(29.90141925,-91.07248849999999),(30.01143975,-91.07248849999999),(29.68137825,-90.94788883333334),(29.79139875,-90.94788883333334),(29.90141925,-90.94788883333334),(30.01143975,-90.94788883333334)]
        self.st_names = ["Assumption_Parish0","Assumption_Parish1","Assumption_Parish2","Assumption_Parish3","Assumption_Parish4","Assumption_Parish5","Assumption_Parish6","Assumption_Parish7","Assumption_Parish8","Assumption_Parish9","Assumption_Parish10"]
        self.st_indexes = ['0','1','2','3','4','5','6','7','8','9','10']
        self.parameter = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104,  105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146]
        self.parameter_layerNum = []
        self.parameter_names = []
        self.parameter_units = []
        self.parameter_levels = []
        self.parameter_typeofLevels = []
        
        self.start_date = date(2021, 6, 1)
        self.end_date = date(2021,7,1)

    def daterange(self, start_date, end_date):
        for n in range(int((end_date- start_date).days)):
            yield start_date + timedelta(n)
            
    def hourrange(self):
        for n in range(24): 
            yield "%02d"%n

    def read_loop(self):

        file_name = "HRRR_22_LA_"+self.start_date.strftime("%Y%m")+".csv"
        for single_date in self.daterange(self.start_date, self.end_date):
            for single_hour in self.hourrange():
                data_path = self.wrf_data_path+single_date.strftime("%Y")+"/"+single_date.strftime("%Y%m%d")+"/"+"hrrr."+single_date.strftime("%Y%m%d")+"."+single_hour+".00.grib2"
                write_path = self.write_path+'/'+self.arrfips[0]+'/'+single_date.strftime("%Y")+"/"# +single_date.strftime("%Y%m")+"/"
                if(single_date.strftime("%d") == "01" and single_hour == "00"):
                    data = self.read_data(data_path, write_path, headerFlag=True)
                else:
                    data = self.read_data(data_path, write_path, headerFlag=False)
                self.write_data(data,write_path, single_date, single_hour, file_name)
        write_path = self.write_path+self.arrfips[0]+'/'+self.start_date.strftime("%Y")+'/'
        self.write_header(write_path, file_name)


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

    def write_data(self, data, w_data_path, single_date, single_hour, file_name):
        itr = 0
        
        for k,v in data.items():
            w_path = w_data_path 
            Path(w_path).mkdir(parents=True, exist_ok=True)
            # w_path_file = w_data_path+self.arrfips[0]+'/'+self.start_date.strftime("%Y")+'/'+"HRRR_22_LA_"+self.start_date.strftime("%Y%m")+".csv"
            # w_path_nofile = w_data_path+self.arrfips[0]+'/'+self.start_date.strftime("%Y")+'/'
            # Path(w_path_nofile).mkdir(parents=True, exist_ok=True)
            w_path_file = w_path + file_name
            with open(w_path_file, 'a', newline='') as file_out:
                writer = csv.writer(file_out, delimiter=',')
                write_list = []
                
                write_list.append(single_date.strftime("%Y"))
                write_list.append(single_date.strftime("%m"))
                write_list.append(single_date.strftime("%d"))
                write_list.append(single_hour)
                write_list.append('LA')
                write_list.append(self.st_names[itr])
                write_list.append(self.st_indexes[itr])
                write_list.append(self.arrfips[itr])
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

e = extraction()
e.read_loop()


