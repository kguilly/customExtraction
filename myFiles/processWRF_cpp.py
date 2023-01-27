import csv
import os
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path
import sys

class formatWRF():
    def __init__(self):
        self.wrf_data_path = ""
        self.repository_path = ""

    def main(self):
        self.readargs()
        csvFiles = self.findFiles()
        if len(csvFiles) < 1:
            print("Error, file list is less than one. List: ", csvFiles)
            exit(0)
        for file in csvFiles:
            print(file)
            df = self.removeUnkn(file=file)
            # print(df)
            df = self.dailyAvgMinMax(df=df)
            # get the monthly average
            df = self.monthlyAvgs(df=df)
            # write out to new csv
            self.dftocsv(df=df, fullfilepath=file)

    def readargs(self):
        if(len(sys.argv)>1):
            # repo_path_flag=False
            # repo_path_index =0
            # wrf_data_path_flag = False
            # wrf_data_path_index = 0
            for i in range(1, len(sys.argv)):
                if sys.argv[i]=="--repo_path":
                    self.repository_path = sys.argv[i+1]
                elif sys.argv[i]=="--wrf_path":
                    self.wrf_data_path = sys.argv[i+1]

    def findFiles(self):
        fipsFolders = sorted(os.listdir(self.wrf_data_path))
        # print(fipsFolders)
        csvFiles = []
        
        for ff in fipsFolders:
            fipsPath = self.wrf_data_path + ff + '/'
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
            df.drop(columns='Unnamed: 0', inplace=True)
        except:
            print("")
        for col in df:
            # for each column, if the string 'unknown' is found, remove it 
            if col.rfind('unknown') != -1:
                df.drop(columns=col, inplace=True)
        
        # print(df)
        return df

    def dailyAvgMinMax(self, df=pd.DataFrame()):
        # find the daily averages as well as the daily mins and maxes, return the df
        # for each day, make a separate dataframe, then concatenate them, 
        # print(df)
        df.sort_values(['Day'], inplace=True)
        dftoreturn = pd.DataFrame(columns=list(df.columns))
        # dftoreturn.drop(columns='Hour', inplace=True)
        dftoreturn.rename(columns= {'Hour': 'Daily/Monthly'}, inplace=True)

        for col in dftoreturn:
            if col.rfind('Temperature') != -1:
                # need to insert new columns at index of temperature 
                index = dftoreturn.columns.get_loc(col)
                dftoreturn.insert(index+1, "Max Temperature (K)", value=None,allow_duplicates=True)
                dftoreturn.insert(index+1, "Min Temperatrue (K)", value=None,allow_duplicates=True)
        # print(list(dftoreturn.columns))

       
        ##########################################
        # look for a more efficient way to do this
        ##########################################
        grouped_gridindex = df.groupby(df.GridIndex)
        # for each station
        for i in range(len(grouped_gridindex.size())):
            gridindexDf = grouped_gridindex.get_group(i)
            grouped_day = gridindexDf.groupby(gridindexDf.Day)

            # for each day in each station
            for j in range(len(grouped_day.size())):
                try:
                    station_dayDf = grouped_day.get_group(j)
                    # print(station_dayDf)
                except:
                    continue
                avgedRow = []

                for col in station_dayDf:
                    if col.rfind('Hour') != -1:
                        avgedRow.append('Daily')
                    elif col.rfind('Year') != -1 or col.rfind('Month') != -1 or col.rfind('Day') != -1 or col.rfind('State') != -1 or col.rfind('County') != -1 or col.rfind('GridIndex') != -1 or col.rfind('FIPS Code') != -1 or col.rfind('Lat') != -1 or col.rfind('Lon(-180') != -1:
                        avgedRow.append(station_dayDf[col].iloc[0])
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
        writeavgflag = False
        for col in df:
            if col.rfind('Maximum/Composite') != -1:
                writeavgflag = True
            
            if not writeavgflag:
                # do sum
                if col.rfind('Daily/Monthly') != -1:
                    monthlyavgs.append("Monthly")
                elif col.rfind('Year') != -1 or col.rfind('Month') != -1 or col.rfind('State') != -1 or col.rfind('County') != -1 or col.rfind('FIPS') != -1:
                    monthlyavgs.append(df[col].iloc[0])
                else:
                    monthlyavgs.append('N/A')
            else:
                # find the average of the column and append to the monthlyavg arr
                try:
                    monthlyavgs.append(df[col].mean())
                except:
                    monthlyavgs.append('NaN')
        # df.loc[len(df)] = monthlyavgs
        # df = df.append(pd.Series(monthlyavgs, index=df.columns[:len(monthlyavgs)]))
        df = df.append(pd.Series(monthlyavgs, index=df.columns[:len(monthlyavgs)]), ignore_index=True)
        print(df)
        return df

    def dftocsv(self, df=pd.DataFrame(), fullfilepath=''):
        filepathsep = fullfilepath.split('/')
        newfilepath = ''

        for i in range(len(filepathsep)-1):
            newfilepath += filepathsep[i] + '/'
        
        filename = filepathsep[len(filepathsep)-1]
        newfilename = filename[0:len(filename)-4] + 'daily_monthly.csv'
        newfilepath += newfilename
        df.to_csv(newfilepath)
        return


f = formatWRF()
f.main()
