'''
this file will run the spider.py file for a full year's worth of files,
running the file for each day of the year until the final day, at which point
the file will send an email of either a success of a failure. 
'''
import os
import datetime
from datetime import date, timedelta
import subprocess
import smtplib,ssl
import time


class spider_runna():

    def __init__(self):
        self.email = "guillotkaleb01@gmail.com"
        self.year = "2021"
        self.spider_file_path = "/home/kalebg/Documents/GitHub/customExtraction/downloadFiles/"
        self.spider_output_path = "/home/kalebg/Documents/GitHub/customExtraction/data/"

    def email_me(self):
        # send me an email to tell me that you're done
        print("email me!")
        port = 465
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
            server.login("mimaemailalerts@gmail.com", "bttkxogqnqslpnfn")
            message = "SUBJECT: THE YEAR HAS FINISHED DOWNLOADING"
            server.sendmail("mimaemailalerts@gmail.com", self.email, message)

    def get_year(self):
        strfirstday = self.year+"0101"
        currday = str(strfirstday)
        
        while 1:
            # run call the command line to run the file for the current
            # day, and the next day as the end day
            
            currday = date(int(currday[0:4]), int(currday[4:6]), int(currday[6:]))
            
            nextday = currday + datetime.timedelta(days=1)
            nextday = nextday.strftime('%Y%m%d')
            currday = currday.strftime('%Y%m%d')

            # only call the command line if currday's day == 1, 15, or if nextday's month != currday's 
            currday_day = int(currday[6:])
            currday_month = int(currday[4:6])
            nextday_month = int(nextday[4:6])
            if not (currday_day == 1 or currday_day == 14):
                currday = nextday
                continue

            print(currday)
            cmd = ["python", self.spider_file_path+"spider_updated.py", "--begin_date", currday, "--end_date", nextday]
            strcmd = "python "+self.spider_file_path+"spider_updated.py --begin_date "+currday+" --end_date "+nextday+" &>> yearlyOutput.log"
            
            os.system(strcmd)
            # search the directory, whenever the 23rd hour file is present (without .tmp behind it)
            # run the next day
            directory = self.spider_output_path + self.year + '/' + currday +'/'
            flag = True
            while flag:
                for root, dirs, files in os.walk(directory):
                    files.sort()
                    files.reverse()
                    if len(files) > 10:
                        if files[0].endswith(".23.00.grib2"):
                            flag = False
                        else:
                            time.sleep(10)
                    else:
                        time.sleep(60)
                       
            # once the loop breaks we can kill the previous command
            os.system("^C")
            os.system("clear")

            # if the day is the final day of the year,
            if currday == self.year+"1231":
                self.email_me()
                break

            currday = nextday


if __name__ =="__main__":
    s=spider_runna()
    s.get_year()