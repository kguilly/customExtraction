import os
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description="enter the write path")
parser.add_argument('--path' , default='/home/kalebg/Desktop/WRFDataThreaded/')
args = parser.parse_args()
directory = args.path

fipsfolders = sorted(os.listdir(directory))

for fips in fipsfolders:
    
    newpath = directory + fips + '/'
    yearfolders = sorted(os.listdir(newpath))

    for years in yearfolders:
        # print(years)
        anothernewpath = newpath+ years + '/'
        files = sorted(os.listdir(anothernewpath))
        for file in files:
            # print(file)
            fullpath = anothernewpath + file
            df = pd.read_csv(fullpath)
            print(df)
            findavgflag = False
            introflag = True
            # write the 
            for column in df.columns:
                # make sure that
                
                if findavgflag: 
                    avg = df[column].mean(axis=1)

                else: # write the introductory information to the file
                    if introflag:
                        introflag = False
                        print(column)
                        year = df[column].values[0]
                        month = df[column].values[1]
                        monthly = "Monthly"
                        state = df[column].values[3]
                        County = df[column].values[4]
                        fipscode = df[column].values[5]
                        grididx = " "
                        df.concat([year, month, monthly, state, County, fipscode, grididx], axis=0)
                        print(df)
                    if column == 'GridIndex':
                        findavgflag = True
# for fipsfolders in newdir:
#     fipspath = directory + fipsfolders + '/'
#     fipspaths = sorted(os.listdir(fipspath))

#     for yearfolders in fipspaths:
#         yearpath = fipspaths + yearfolders + '/'
#         years = sorted(os.listdir(yearpath))
#         print(years)

#         for files in years:
#             filepath = yearpath + files
#             df = pd.read_csv(files)
