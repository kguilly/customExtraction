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


class formatWRF():
    def __init__(self):
        self.wrf_data_path = "/home/kaleb/Desktop/2017data/Hourly/"
        self.repository_path = "/home/kaleb/Documents/GitHub/customExtraction/"

    def main(self):
        self.readargs()
        csvFiles = self.findFiles()
        if len(csvFiles) < 1:
            print("Error, file list is less than one. List: ", csvFiles)
            exit(0)
        for file in csvFiles:
            # start_time = time.time()
            # if the file is open elsewhere, or if the file is already a processed file, skip
            if (file.find("daily_monthly") != -1 or file.find(".~lock.") != -1):
                print("Skipping file: %s" % file)
                continue
            print("Opening File: %s" % file)
            df = self.removeUnkn(file=file)
            # print(df)
            # since the F00 file does not include accumulated temp,
            # we consult the F01 file in order to do so
            df = self.grab_precip(df=df)

            df = self.fixCountyNames(df=df)

            df = self.dailyAvgMinMax(df=df)

            # based off of the daily averages, calculate the VPD
            df = self.windSpeed_vpd(df=df)

            # before getting the monthly average, need to convert the lons back
            df = self.convert_lons(df=df)

            # get the monthly average
            df = self.monthlyAvgs(df=df)

            df = self.more_renaming(df=df)
            # write out to new csv
            self.dftocsv(df=df, fullfilepath=file)

            # time_sec = time.time() - start_time
            # h = 0
            # m = 0
            # if (time_sec // 60 > 1):
            #     m = time_sec // 60
            #     time_sec = time_sec - (m * 60)
            # if (m // 60 > 1):
            #     h = m // 60
            #     m = m - (h * 60)
            # print("\n\nSingle File Extraction time: %s\nHours: %d, Minutes: %d, Seconds: %s\n" % (file,h, m, time_sec))

        self.concat_states()


    def readargs(self):
        if (len(sys.argv) > 1):
            # repo_path_flag=False
            # repo_path_index =0
            # wrf_data_path_flag = False
            # wrf_data_path_index = 0
            for i in range(1, len(sys.argv)):
                if sys.argv[i] == "--repo_path":
                    self.repository_path = sys.argv[i + 1]
                elif sys.argv[i] == "--wrf_path":
                    self.wrf_data_path = sys.argv[i + 1]

    def findFiles(self):
        input_file_path = self.wrf_data_path
        try:
            fips_folders = sorted(os.listdir(input_file_path))
        except FileNotFoundError:
            print("The FIPS folders were not found")
            exit()

        # print(fips_folders)
        csvFiles = []

        for ff in fips_folders:
            fipsPath = input_file_path + ff + '/'
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

    def removeUnkn(self, file):
        df = pd.read_csv(file, header=0)
        # print(df)
        try:
            df.rename(columns={"2 metre temperature (K)": "Temperature(K)",
                               "2 metre relative humidity (%)": "Relative Humidity (%)"}, inplace=True)
            df.rename(columns={"Wind speed (gust) (m s**-1)": "Wind Gust (m s**-1)"}, inplace=True)
            df.drop(columns='Unnamed: 0', inplace=True)
        except:
            print("")

        for col in df:
            # for each column, if the string 'unknown' is found, remove it
            if col.rfind('unknown') != -1:
                df.drop(columns=col, inplace=True)
            elif col.rfind('Unnamed:') != -1:
                df.drop(columns=col, inplace=True)
            elif col.rfind("Precipitation") != -1:
                df.drop(columns=col, inplace=True)

        df['Grid Index'] = df['Grid Index'].astype(int)
        df['Day'] = df['Day'].astype(int)

        # print(df)
        return df

    def grab_precip(self, df: pd.DataFrame()) -> pd.DataFrame():
        """
        :param df: - input df of extracted hourly WRF data
        :return: df - with the appended "total precipitation" column

        Pseudo Code:
        for each day:
            store day
            for each hour:
                store hour
                search file path:
                if path DNE:
                    make herbie obj and download file
                open file as grib obj
                match latlons to closest point of df's latlons
                append value to precip_df
        merge precip_df with df
        return df
        """
        herb_dir = self.wrf_data_path + "herbie_data/"
        precip_values = []
        first_itr_flag = True
        for i in range(len(df)):
            year = df['Year'].iloc[i]
            month = df['Month'].iloc[i]
            day = df['Day'].iloc[i]
            hour = df['Hour'].iloc[i]

            if first_itr_flag:
                first_itr_flag = False
                if int(hour) == 0:
                    precip_values.append(0)
                    continue
            prev_hour = 0
            if hour <= 10:
                if hour == 0:
                    prev_hour = df['Hour'].iloc[len(df)-1]
                else:
                    prev_hour = '0' + str(int(hour) - 1)
            else:
                prev_hour = str(int(hour-1))

            # calculate the middle lat and lon of the current grid point
            lat_st_mid = (df['lat(llcrnr)'].iloc[i] + df['lat(urcrnr)'].iloc[i]) / 2
            lon_st_mid = (df['lon(llcrnr)'].iloc[i] + df['lon(urcrnr)'].iloc[i]) / 2

            # convert the lons from (0, 360) to (-180, 180)
            lon_st_mid -= 360

            if month < 10:
                month = '0' + str(month)
            else:
                month = str(month)
            if day < 10:
                day = '0' + str(day)
            else:
                day = str(day)

            # try to find the herbie file, if it has not been created yet, set
            # the herb flag and download it
            herb_flag = False
            herb_dir_1 = herb_dir + "hrrr/" + str(year) + str(month) + str(day) + '/'

            if not os.path.exists(herb_dir_1):
                herb_flag = True
            else:
                file_found = False
                for file in sorted(os.listdir(herb_dir_1)):
                    if file.rfind('hrrr.t'+str(prev_hour)+'z.') != -1:
                        file_found = True
                if not file_found:
                    herb_flag = True

            if herb_flag:
                str_date = str(year) + str(month) + str(day) + " " + str(prev_hour) + ":00"
                print("Herb obj not found for %s" % str_date)
                precip_values.append(0)
                continue

            # open the file with pygrib and index the closest point to the st
            grib_file_name = ""
            for file in sorted(os.listdir(herb_dir_1)):
                if file.rfind('hrrr.t'+str(prev_hour)+'z.') != -1:
                    grib_file_name = file
                    break
            try:
                grib = pygrib.open(herb_dir_1 + grib_file_name)
                # print("Opening file: %s" % herb_dir_1 + grib_file_name)
            except:
                print("Error when opening grib file: %s" % herb_dir_1 + grib_file_name)
                exit()
            try:
                g = grib[1]
            except:
                grib.close()
                print("No messages in file %s" % herb_dir_1 + grib_file_name)
                precip_values.append(0)
                continue
            grib.close()
            lats, lons = g.latlons()
            values = g.values
            lat_matrix = np.full_like(lats, lat_st_mid)
            lon_matrix = np.full_like(lons, lon_st_mid)
            distance_matrix = (lats - lat_matrix) ** 2 + (lons - lon_matrix) ** 2
            p_lat, p_lon = np.unravel_index(distance_matrix.argmin(), distance_matrix.shape)
            val = values[p_lat, p_lon]
            precip_values.append(val)

        df['Total Precipitation (kg m**-2)'] = precip_values


        return df

    def fixCountyNames(self, df: pd.DataFrame()):
        # find the column for the county name
        # for each element, capitalize, and then if they have the term
        # 'PARISH' or 'COUNTY' then remove it
        idx = 0
        for row in df['County']:
            row = row.upper()
            if (row.rfind('COUNTY') != -1):
                rowarr = row.split("COUNTY")
                row = ""
                for elem in rowarr:
                    row += elem
                # print(row)

            if (row.rfind('PARISH') != -1):
                rowarr = row.split("PARISH")
                row = ""
                for elem in rowarr:
                    row += elem
                # print(row)
            df.at[idx, 'County'] = row
            idx += 1

        return df

    def windSpeed_vpd(self, df=pd.DataFrame()):
        """
        turn the columns for U comp and V comp of wind
        into numpy ndarrays, then feed into the get_wind_speed
        functions. delete the U comp and V comp columns, make a new
        column "Wind Speed(m s**-1)" and put the return value as the
        values of the column

        turn the columns for temperature and humidity into numpy
        ndarrays, then feed into get_vpd function, make a new
        column "Vapor Pressure Deficit (kPa)" and put the return
        value as the values of the column
        """
        u_comp_wind = df['U component of wind (m s**-1)'].to_numpy().astype(dtype=float)
        v_comp_wind = df['V component of wind (m s**-1)'].to_numpy().astype(dtype=float)
        # df.drop(columns=['U component of wind (m s**-1)', 'V component of wind (m s**-1)'], inplace=True)

        temp = np.asarray(df['Temperature(K)'].values)
        relh = np.asarray(df['Relative Humidity (%)'].values)

        wind_speed = self.get_wind_speed(u_comp_wind, v_comp_wind)
        df['Wind Speed (m s**-1)'] = wind_speed

        vpd = self.get_VPD(temp, relh)
        df['Vapor Pressure Deficit (kPa)'] = vpd

        return df

    def get_wind_speed(self, U: np.ndarray, V: np.ndarray) -> np.ndarray:
        '''
        Calculate the average wind speed
        :param U: array_like, U component of wind in 1000 hPa (i.e., layer 32 in HRRR)
        :param V: array_like, V component of wind in 1000 hPa (i.e., layer 32 in HRRR)
        :return:
        '''
        return np.sqrt(np.power(U, 2) + np.power(V, 2))

    def get_VPD(self, T: np.ndarray, RH: np.ndarray) -> np.ndarray:
        '''
        Calculate the Vapor Pressure Deficit (VPD), unit: kPa
        :param T: array_like, temperate (unit: K) at 2 metre (layer 54 in HRRR)
        :param RH: array_like, relative humidity at 2 metre (i.e., layer 58 in HRRR)
        :return:
        '''

        # covert the temperature in Kelvin to the temperature in Celsius
        T = T - 273.15
        E = 7.5 * T / (237.3 + T)
        VP_sat = 610.7 * np.power(10, E) / 1000
        VP_air = VP_sat * RH / 100
        VPD = VP_sat - VP_air

        return VPD

    def dailyAvgMinMax(self, df=pd.DataFrame()):
        # find the daily averages as well as the daily mins and maxes, return the df
        # for each day, make a separate dataframe, then concatenate them,
        # print(df)
        df.sort_values(['Day'], inplace=True)
        dftoreturn = pd.DataFrame(columns=list(df.columns))
        # dftoreturn.drop(columns='Hour', inplace=True)
        dftoreturn.rename(columns={'Hour': 'Daily/Monthly'}, inplace=True)
        # print(df)
        for col in dftoreturn:
            if col.rfind('Temperature') != -1:
                # need to insert new columns at index of temperature
                index = dftoreturn.columns.get_loc(col)
                dftoreturn.insert(index + 1, "Max Temperature (K)", value=None, allow_duplicates=True)
                dftoreturn.insert(index + 1, "Min Temperature (K)", value=None, allow_duplicates=True)
        # print(list(dftoreturn.columns))

        ##########################################
        # look for a more efficient way to do this
        ##########################################
        grouped_gridindex = df.groupby(df['Grid Index'])
        # for each station df
        for gridgroupdf in grouped_gridindex:
            # print(gridgroupdf[1])
            gridindexDf = pd.DataFrame(gridgroupdf[1])
            # print(gridindexDf)
            grouped_day = gridindexDf.groupby(gridindexDf.Day)

            # for each day in each station
            for group in grouped_day:
                try:
                    current_day = group[0]
                    station_dayDf = pd.DataFrame(group[1])
                except:
                    print("Did not grab grouped_day.get_group(%s)" % (group))
                    continue
                avgedRow = []
                # print(station_dayDf)

                for col in station_dayDf:
                    if col.rfind('Hour') != -1:
                        avgedRow.append('Daily')
                    elif col.rfind('Year') != -1 or col.rfind('Month') != -1 or col.rfind('Day') != -1 or col.rfind(
                            'State') != -1 or col.rfind('County') != -1 or col.rfind('Grid Index') != -1 or col.rfind(
                            'FIPS Code') != -1 or col.rfind('lat') != -1 or col.rfind('lon(') != -1:
                        avgedRow.append(station_dayDf[col].iloc[0])
                        # avgedRow.concat()
                        # pd.concat([avgedRow, station_dayDf[col].iloc[0]], ignore_index=True)
                    elif col.rfind('Precipitation') != -1 or col.rfind('radiation') != -1:
                        avgedRow.append(station_dayDf[col].sum())
                    else:
                        # get the average of the row and append it to the avg list
                        try:
                            avgedRow.append(station_dayDf[col].mean())
                        except:
                            avgedRow.append('NaN')
                        # if the column is temperature, find the min, max, then append
                        if col.rfind("Temperature") != -1:
                            avgedRow.append(station_dayDf[col].min())
                            avgedRow.append(station_dayDf[col].max())

                # now that all the columns have been collected, we need to append the row to the dftoreturn
                dftoreturn.loc[len(dftoreturn)] = avgedRow
        # print(dftoreturn)
        return dftoreturn

    def monthlyAvgs(self, df=pd.DataFrame()):

        monthlyavgs = []
        # for each column in df (not Year, Month, Day, State, County, GridIndex, FIPS Code, Lat, Lon(-180 to 180)), find avg and append to bottom
        # for Year, Month, State, County, write the first Index, for day write 'Monthly Average'

        # writeavgflag = False
        for col in df:
            # if col.rfind('Maximum/Composite') != -1:
            #     writeavgflag = True

            # if not writeavgflag:
            #     # do sum
            if col.rfind('Daily/Monthly') != -1:
                monthlyavgs.append("Monthly")
            elif col.rfind('Year') != -1 or col.rfind('Month') != -1 or col.rfind('State') != -1 or col.rfind(
                    'County') != -1 or col.rfind('FIPS') != -1:
                monthlyavgs.append(df[col].iloc[0])
            elif (col.rfind('lat') != -1 or col.rfind('lon') != -1) and col.rfind('elative') == -1:
                monthlyavgs.append('N/A')
            elif col.rfind('Grid Index') != -1 or col.rfind('Day') != -1:
                monthlyavgs.append('N/A')
            elif col.rfind('recipitation') != -1 or col.rfind('radiation') != -1:
                df_new = df.sort_values(by=['Day', 'Grid Index'])
                last_day = df_new['Day'][len(df_new)-1]
                first_day = df_new['Day'][0]
                total_grid_indexes = df_new['Grid Index'][len(df_new)-1] + 1
                sum = df[col].sum()
                val = sum / ((last_day - first_day + 1) * total_grid_indexes)
                monthlyavgs.append(val)
            else:
                # find the average of the column and append to the monthlyavg arr
                try:
                    monthlyavgs.append(df[col].mean())
                    # monthlyavgs.append(df[col].values.mean())
                except:
                    monthlyavgs.append('NaN')

        # df = df.append(pd.Series(monthlyavgs, index=df.columns[:len(monthlyavgs)]), ignore_index=True)
        # df = pd.concat([df, pd.Series(monthlyavgs)], ignore_index=True, axis=0, join='outer')
        df.loc[len(df.index)] = monthlyavgs
        # print(df)
        return df

    def more_renaming(self, df=pd.DataFrame()):
        df['Month'] = df['Month'].astype(int)

        df.rename(columns={"Total Precipitation (kg m**-2)": "Precipitation (kg m**-2)"}, inplace=True)
        df.rename(columns={"Temperature(K)": "Avg Temperature (K)"}, inplace=True)
        df.rename(columns={"Downward short-wave radiation flux (W m**-2)":
                           "Downward Shortwave Radiation Flux (W m**-2)"}, inplace=True)
        df.rename(columns={"lat(llcrnr)": "Lat (llcrnr)", "lon(llcrnr)": "Lon (llcrnr)"}, inplace=True)
        df.rename(columns={"lat(urcrnr)": "Lat (urcrnr)", "lon(urcrnr)": "Lon (urcrnr)"}, inplace=True)
        df.rename(columns={"U component of wind (m s**-1)": "U Component of Wind (m s**-1)",
                           "V component of wind (m s**-1)": "V Component of Wind (m s**-1)"}, inplace=True)

        for col in df:
            if col.rfind('elative Humidi') != -1:
                df[col] = df[col].round(decimals=1)
                continue
            try:
                df[col] = df[col].round(decimals=3)
            except:
                continue

        # Changing the order of the column to fit fudong's needed format:
        df = df[['Year', 'Month', 'Day', 'Daily/Monthly', 'State', 'County', 'FIPS Code','Grid Index',
                 'Lat (llcrnr)', 'Lon (llcrnr)', 'Lat (urcrnr)', 'Lon (urcrnr)',
                 'Avg Temperature (K)', 'Max Temperature (K)', 'Min Temperature (K)', 'Precipitation (kg m**-2)',
                 'Relative Humidity (%)', 'Wind Gust (m s**-1)', 'Wind Speed (m s**-1)',
                 'U Component of Wind (m s**-1)', 'V Component of Wind (m s**-1)',
                 'Downward Shortwave Radiation Flux (W m**-2)', 'Vapor Pressure Deficit (kPa)']]
        return df

    def convert_lons(self, df=pd.DataFrame()):
        for col in df:
            if col.rfind('lon(') != -1:
                # if the column has the name 'lon' in it, then we want to convert the value from the range
                # of 0 to 360 to -180 to 180
                df[col] = df[col].astype(float)
                vals = df[col].values
                newvals = []
                for val in vals:
                    newvals.append(round(val - 360, 6))

                df[col].replace(to_replace=df[col].values, value=newvals, inplace=True)
        return df

    def dftocsv(self, df=pd.DataFrame(), fullfilepath=''):
        filepathsep = fullfilepath.split('/')
        newfilepath = ''
        fipscode = ''
        for i in range(len(filepathsep) - 1):
            if filepathsep[i].rfind("Hourly") != -1:
                fipscode = filepathsep[i + 1]
                break
            newfilepath += filepathsep[i] + '/'
        newfilepath += "Daily_Monthly/" + fipscode + '/'
        Path(newfilepath).mkdir(parents=True, exist_ok=True)
        write_path = newfilepath + filepathsep[len(filepathsep) - 1]
        df.to_csv(write_path, index=False)
        return

    def concat_states(self):
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
        filepathsep = self.wrf_data_path.split('/')
        hourly_file_path = ''
        for i in range(len(filepathsep) - 1):
            if filepathsep[i].rfind("Hourly") != -1:
                break
            hourly_file_path += filepathsep[i] + '/'
        hourly_file_path += "Daily_Monthly/"

        df_states = {}
        for fips_folder in sorted(os.listdir(hourly_file_path)):
            fips_dir = hourly_file_path + fips_folder + '/'
            for monthly_file in sorted(os.listdir(fips_dir)):
                state_fips = monthly_file[5:7]
                state_abbrev = monthly_file[11:13]
                year = monthly_file[14:18]
                month = monthly_file[18:20]
                col_name = state_fips + '_' + year + '_' + month
                full_path = fips_dir+monthly_file
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
            file_path_out_sep = self.wrf_data_path.split('/')
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


formatWRF().main()
