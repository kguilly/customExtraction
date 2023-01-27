import requests
from bs4 import BeautifulSoup
import re
import os
import datetime
import urllib
import logging
from logging.handlers import TimedRotatingFileHandler
import threading, time
from multiprocessing import Process, Manager
from threading import RLock
from datetime import date, timedelta

import argparse


class spider():
    
    def __init__(self):
        self.version = 1.0
        self.info = 'grab weather data'
        self.data_path = 'data/'
        
        self.url = "http://home.chpc.utah.edu/~u0553130/Brian_Blaylock/cgi-bin/hrrr_download.cgi"
        self.manager = Manager()
        self.store_data = self.manager.dict()
        self.cache = {}
        self.lock = RLock()

        self.worker = 3

        self.WAIT_TIME_SECONDS = 1*60*30    # 30 min
        self.logger = None

    def log(self):
        self.logger = None
        now = datetime.datetime.now()
        self.logger = logging.getLogger('weather_spider_')
        self.logger.setLevel(logging.INFO)
        self.path_check("Log/")
        fh = logging.FileHandler('Log/weather_spider'+str(now.year)+str(now.month)+str(now.day))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def path_check(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
                

    def grab_html(self):
        headers = {'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36'}
        url = self.url
        x = requests.get(url, headers = headers)

        # print(x.text)
        soup = BeautifulSoup(x.text,"html.parser",from_encoding="utf-8")
        return soup
    

    def data_check(self, soup, check_date=None):
        folder_path = ''
        if check_date:
            folder_path = self.data_path+str(check_date)+'/'
            if not os.path.exists(folder_path):    # create folder path
                os.makedirs(folder_path)

            # new version only get data for hour 00
            # for hour in range(0,24):
            #     if hour%6 == 0:
            #         # for pre_hour in range(0,37):
            #         for pre_hour in range(0,19):
            #             data_name = (check_date, str(hour).zfill(2), str(pre_hour).zfill(2))
            #             if not data_name in self.store_data.keys():
            #                 self.store_data[data_name] = 0
            #     else:
            #         for pre_hour in range(0,19):
            #             data_name = (check_date, str(hour).zfill(2), str(pre_hour).zfill(2))
            #             if not data_name in self.store_data.keys():
            #                 self.store_data[data_name] = 0
            
            for hour in range(0,24):
                data_name = (check_date, str(hour).zfill(2), str(0).zfill(2))
                if not data_name in self.store_data.keys():
                    self.store_data[data_name] = 0

        
        else:
            hour_blocks = soup.findAll('div',attrs={'class':'mybtn-group'})
            for div in hour_blocks:
                links = div.findAll('a')
                for a in links:
                    a_name = re.sub("[^0-9]", " ", a['href'])
                    a_name = a_name.split()
                    year_month = a_name[1]
                    hour = a_name[2]
                    pre_hour = a_name[3]
                    # folder_path = self.data_path+str(year_month)+'/'
                    
                    data_name = (year_month, hour, pre_hour)
                    folder_path = self.data_path+str(year_month)+'/'

                    if hour == "00" and pre_hour == "00":
                        if not os.path.exists(folder_path):
                            os.makedirs(folder_path)

                    if not data_name in self.store_data.keys():
                        self.store_data[data_name] = 0


        saved_files = os.listdir(folder_path)
        for saved_file in saved_files:
            
            file_name_split = saved_file.split('.')
            
            if not file_name_split[-1] == 'grib2':
                continue
            
            l_data_name = (file_name_split[-4],file_name_split[-3],file_name_split[-2])
            
            if l_data_name in self.store_data.keys():
                self.store_data[l_data_name] = 1
        self.logger.info("data check")

    def thread_check(self):
        for k in self.store_data.keys():
            if self.store_data[k] == 1 or self.store_data[k] == 2:
                continue
            self.store_data[k] = 2
            return k
        return True
            

    
    def download_data(self):
        
        self.lock.acquire()
        k = self.thread_check()
        self.lock.release()

        if k == True:
            return
        
            
        download_url = "https://pando-rgw01.chpc.utah.edu/hrrr/sfc/"+k[0]+"/hrrr.t"+k[1]+"z.wrfsfcf"+k[2]+".grib2"
        data_path_tmp = self.data_path+k[0]+"/hrrr."+k[0]+"."+k[1]+"."+k[2]+".grib2.tmp"
        data_path = self.data_path+k[0]+"/hrrr."+k[0]+"."+k[1]+"."+k[2]+".grib2"
        print(data_path)

        self.logger.info('start downloading: '+data_path)
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            with open(data_path_tmp, 'wb') as f: 
                for chunk in r.iter_content(chunk_size=8192): 
                    if chunk: 
                        f.write(chunk)
        self.lock.acquire()
        os.rename(data_path_tmp, data_path)
        self.store_data[k] = 1
        self.lock.release()
        self.logger.info('finish downloading: '+data_path)




    def run(self, check_date = None):
        
        soup = None
        grab_flag = True
        
        while True:
            
            try:
                if grab_flag:
                    grab_flag = False
                if check_date:
                    soup = None
                else:
                    soup = self.grab_html()
                
                self.data_check(soup, check_date = check_date)
                

                threads_list = []
                for i in range(self.worker):
                    self.logger.info('worker: '+str(i))
                    t = threading.Thread(target=self.download_data, daemon=True)
                    threads_list.append(t)
                    t.start()
                

    
                while 1:
                    alive = False
                    for s in threads_list:
                        alive = alive or s.is_alive()
                    if not alive:
                        break

                threads_list = []
                
                sleep_flag = True
                for k,v in self.store_data.items():
                    if v == 0:
                        sleep_flag = False
                        break

                if sleep_flag:
                    break
                    # ticker = threading.Event()
                    # while not ticker.wait(self.WAIT_TIME_SECONDS):
                    #     self.logger.info('start checking')
                    #     grab_flag = True
                    #     break
            except:
                self.logger.info('sleep one minute')
                grab_flag = True
                time.sleep(20)


            

    def request_range(self,begin_date = None, end_date=None):
        self.log()
        if not begin_date:
            return
        if not end_date:
            return
        if end_date == "now":
            edate = datetime.date.today()
            # edate = d.strftime('%Y-%m-%d')
        else:
            edate = date(int(end_date[0:4]), int(end_date[4:6]), int(end_date[6:]))
        bdate = date(int(begin_date[0:4]), int(begin_date[4:6]), int(begin_date[6:]))


        delta = edate - bdate       # as timedelta
        for i in range(delta.days):    # do not grab current day
            day = bdate + timedelta(days=i)
            cdate = day.strftime('%Y-%m-%d')
            cdate_split = cdate.split('-')
            cdate_new = cdate_split[0]+cdate_split[1]+cdate_split[2]
            data_path_temp = self.data_path
            self.data_path += cdate_split[0]+'/'
            self.run(check_date=cdate_new)
            self.data_path = data_path_temp







if __name__ == "__main__":
    # add arguments 
    parser = argparse.ArgumentParser(description='enter a date range')
    parser.add_argument('--begin_date', default=None, type=str, help='the beginning date')
    parser.add_argument('--end_date', default=None, type=str, help='the end date')
    args = parser.parse_args()
    bd = args.begin_date
    ed=args.end_date

    s = spider()
    s.request_range(begin_date=bd, end_date=ed)    # for example: "20191231" or None