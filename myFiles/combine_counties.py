import csv
import os
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path
import sys
import time

import pygrib
from herbie import Herbie
"""
    Concat all fips files from the same state into one single monthly file
    pseudo code:
        df_states = pd.df
        for fips folder in daily/monthly dir:
            fips_dir = ...
            for monthly file in fips_dir:
                df_states[<statefips>_yyyy_mm].append(fullfiledir)
        for col in df_states:
            df_out = pd.df
            for len(col):
                df_out = pd.merge(df_out, col.df)
            df_out.to_csv()
"""
data_path = '/home/kaleb/Desktop/2021data/'
filepathsep = data_path.split('/')
# hourly_file_path = ''
# for i in range(len(filepathsep) - 1):
#     if filepathsep[i].rfind("Hourly") != -1:
#         break
#     hourly_file_path += filepathsep[i] + '/'
# hourly_file_path += "Daily_Monthly/"
hourly_file_path = data_path
df_states = {}
for fips_folder in sorted(os.listdir(hourly_file_path)):
    fips_dir = hourly_file_path + fips_folder + '/'
    for monthly_file in sorted(os.listdir(fips_dir)):
        state_fips = monthly_file[5:7]
        state_abbrev = monthly_file[11:13]
        year = monthly_file[14:18]
        month = monthly_file[18:20]
        col_name = state_fips + '_' + year + '_' + month
        full_path = fips_dir + monthly_file
        if col_name not in df_states:
            df_states[col_name] = []
        df_states[col_name].append(full_path)

# the dictionary is now built, now we need to merge them into stately files
for col in df_states:
    df_out = pd.DataFrame()
    for file in df_states[col]:
        df = pd.read_csv(file, na_filter=False, na_values='N/A')
        if df_out.empty:
            df_out = df
        else:
            df_out = df_out.merge(df, how='outer')

    file_path_out = ''
    file_path_out_sep = data_path.split('/')
    for i in range(len(file_path_out_sep) - 1):
        if file_path_out_sep[i].rfind("Hourly") != -1:
            break
        file_path_out += file_path_out_sep[i] + '/'

    in_file_path_sep = df_states[col][0].split('/')
    needed_info_sep = in_file_path_sep[-1].split('_')
    fips = needed_info_sep[1]
    yyyymmcsv = needed_info_sep[3]
    state_fips = fips[0:2]
    year = yyyymmcsv[0:4]
    month = yyyymmcsv[4:6]
    state_abbrev = needed_info_sep[2]
    file_path_out += 'data/' + year + '/' + state_abbrev + '/'
    file_path_with_file = file_path_out + 'HRRR_' + state_fips + '_' + \
                          state_abbrev + '_' + year + '-' + month + '.csv'
    Path(file_path_out).mkdir(parents=True, exist_ok=True)
    # replace all null values with 'N/A'
    # df_out.replace(np.nan, 'N/A')
    df_out.to_csv(file_path_with_file)
