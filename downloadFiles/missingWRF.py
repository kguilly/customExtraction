'''
This file will search a given directory and detect missing WRF files, days, and / or years
'''

import os
import datetime
from datetime import date, timedelta

class file_finda():

    def __init__(self):
        self.path = "/home/kalebg/Documents/GitHub/customExtraction/data/"

    def finda(self):

        year = sorted(os.listdir(self.path))
        if len(year) < 1:
            exit

        print("The first year found is: " + year[0])
        for i in range (0, len(year)):
            currYear = year[i]
            print(currYear)
            dayDirs = sorted(os.listdir(self.path+year[i]))
            if len(dayDirs) == 365:
                print("- No Days Missing")
            else:
                print(" - DAYS MISSING FROM " + currYear)

            currday = year[i]+"0101"
            for days in dayDirs:
                if days != currday:
                    print("-- " + currday + " missing")

                    while currday != days:
                        # keep getting the next day until the day is found
                        currday = date(int(currday[0:4]), int(currday[4:6]), int(currday[6:]))
                        currday = currday + datetime.timedelta(days=1)
                        currday = currday.strftime('%Y%m%d')
                        if(currday == days):
                            break
                        else:
                            print("-- " + currday + " missing")
                
                wrfFiles = sorted(os.listdir(self.path+year[i]+'/'+days))
                if len(wrfFiles) < 1:
                    print("--- No files in " + currday)
                    break 
                intHour = 0
                strHour = "00"
                hourFile = "hrrr."+currday+"." + strHour + ".00.grib2"
                missingHours = []
                for file in wrfFiles:
                    if file != hourFile:
                        while file!=hourFile and intHour <=23:
                            missingHours.append(intHour)
                            # print("--- Hour: " + str(intHour) + " missing")
                            intHour+=1
                            if intHour < 10:
                                strHour = "0" + str(intHour) 
                            else:
                                strHour = str(intHour)
                            hourFile = "hrrr."+currday+"."+strHour+".00.grib2"
                    else:
                        intHour+=1
                        if intHour < 10:
                            strHour = "0" + str(intHour) 
                        else:
                            strHour = str(intHour)
                        hourFile = "hrrr."+currday+"."+strHour+".00.grib2"

                if len(missingHours) > 1:
                    print("--- Hours missing from ", currday, ": ", missingHours)
                # get the next day before the next iteration of the loop
                currday = date(int(currday[0:4]), int(currday[4:6]), int(currday[6:]))
                currday = currday + datetime.timedelta(days=1)
                currday = currday.strftime('%Y%m%d')

                    
            if i>0 and int(year[i])-int(year[i-1])!= 1:
                # there is a year or more missing
                x = int(year[i])-int(year[i-1]) # there are x days missing
                for j in range (1,x):
                    missingYear = int(year[i-1])+j
                    print(missingYear, " missing!")


if __name__=="__main__":
    f = file_finda()
    f.finda()
