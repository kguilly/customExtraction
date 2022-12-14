import csv
import os
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path        

# point at "/home/canadmin/prefer/WRF/" 
data_path = "WRF_Data/"

#move to 2017 and 2018           
year_folders = sorted(os.listdir(data_path))

print(year_folders)
#'''

for f_alabma  in  year_folders:
    print(f_alabma)
    newPath = data_path + f_alabma+'/'
    month_folders = sorted(os.listdir(newPath))

    for station_loc  in  month_folders:
        stationPath = newPath + station_loc +'/'
        station = sorted(os.listdir(stationPath))
        print("")
        print(station_loc)

        for location in station:
            file_name = stationPath+location
            print(file_name)

            df = pd.read_csv(file_name, header=None)
            #print(df)
            df.to_csv(file_name, header = ["Year","Month","Date","Hour", "ATT1", "ATT2", "ATT3", "ATT4", "ATT5", "ATT6", "ATT7", "ATT8", "ATT9", "ATT10", "ATT11", "ATT12", "ATT13", "ATT14", "ATT15", "ATT16", "ATT17", "ATT18", "ATT19", "ATT20", "ATT21", "ATT22", "ATT23", "ATT24", "ATT25", "ATT26", "ATT27", "ATT28", "ATT29", "ATT30","ATT31", "ATT32", "ATT33", "ATT34", "ATT35", "ATT36", "ATT37", "ATT38", "ATT39", "ATT40", "ATT41", "ATT42", "ATT43", "ATT44", "ATT45","ATT46", "ATT47", "ATT48", "ATT49", "ATT50", "ATT51", "ATT52", "ATT53", "ATT54", "ATT55", "ATT56", "ATT57", "ATT58", "ATT59", "ATT60", "ATT61", "ATT62", "ATT63", "ATT64", "ATT65", "ATT66", "ATT67", "ATT68", "ATT69", "ATT70", "ATT71", "ATT72", "ATT73", "ATT74", "ATT75", "ATT76", "ATT77", "ATT78", "ATT79", "ATT80", "ATT81", "ATT82", "ATT83", "ATT84", "ATT85", "ATT86", "ATT87", "ATT88", "ATT89", "ATT90", "ATT91", "ATT92", "ATT93", "ATT94", "ATT95", "ATT96", "ATT97", "ATT98", "ATT99", "ATT100", "ATT101", "ATT102", "ATT103", "ATT104", "ATT105",     "ATT106", "ATT107", "ATT108", "ATT109", "ATT110", "ATT111",     "ATT112", "ATT113", "ATT114", "ATT115", "ATT116","ATT117", "ATT118", "ATT119", "ATT120", "ATT121", "ATT122", "ATT123","ATT124", "ATT125", "ATT126", "ATT127", "ATT128", "ATT129", "ATT130","ATT131", "ATT132"],index = False)
