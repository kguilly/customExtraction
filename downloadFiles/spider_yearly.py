'''
this file will run the spider.py file for a full year's worth of files,
running the file for each day of the year until the final day, at which point
the file will send an email of either a success of a failure. 
'''
import os
import datetime
from datetime import date, timedelta


class spider_runna():

    def __init__(self):
        self.year = "2016"
        self.spider_file_path = "./"

    def email_me():
        # send me an email to tell me that you're done
        print("email me!")

    def get_year(self):
        strfirstday = self.year+"0101"
        currday = str(strfirstday)
        os.system("cd " +self.spider_file_path)
        
        while 1:
            # run call the command line to run the file for the current
            # day, and the next day as the end day
            
            currday = date(int(currday[0:4]), int(currday[4:6]), int(currday[6:]))
            # currday = datetime.strptime(currday,'%Y%m%d')
            nextday = currday + datetime.timedelta(days=1)
            nextday = nextday.strftime('%Y%m%d')
            print(nextday)
            # os.system("python spider_updated.py --begin_date "+ str(currday)+ " --end_date " + str(nextday))
            pid = os.fork()
            if pid: 
                os.system("python spider_updated.py --begin_date "+ str(currday)+ " --end_date " + str(nextday))
                os.wait()

            # wait for the file to finish running before moving onto next
            # iteration
            

            # if the day is the final day of the year,
            if currday == self.year+"1231":
                email_me()
                break

            currday = nextday


if __name__ =="__main__":
    s=spider_runna()
    s.get_year()