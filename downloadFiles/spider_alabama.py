import requests
from bs4 import BeautifulSoup
import re
import os
import json
import datetime
import urllib
import logging
from logging.handlers import TimedRotatingFileHandler
import threading, time
from multiprocessing import Process, Manager
from threading import RLock
from pprint import pprint
from datetime import date, timedelta



class spider():
    
    def __init__(self):
        self.version = 1.0
        self.info = 'grab alabama weather data'
        self.data_path = 'alabma/'
        self.url = "http://chiliweb.southalabama.edu/archived_download.php"
        self.manager = Manager()
        self.store_data = self.manager.dict()
        self.cache = {}
        self.lock = RLock()

        self.worker = 6

        self.WAIT_TIME_SECONDS = 1*60*30    # 30 min
        self.logger = None

        self.loc = ["agricola","andalusia","ashford_n","atmore","bayminette","castleberry","disl","dixie","elberta","fairhope","florala","foley","gasque","geneva","grandbay","jay","kinston","leakesville","loxley","mobiledr","mobileusa","saraland","mobileusaw","mtvernon","pascagoula","poarch","robertsdale","walnuthill"]

    def log(self):
        now = datetime.datetime.now()

        self.logger = logging.getLogger('alabama_spider_')
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler('Log/alabama_spider'+str(now.year)+str(now.month)+str(now.day))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)


    def request_range(self,begdate = None, enddate=None):
        if not begdate:
            return
        if not enddate:
            return
        if enddate == "now":
            edate = datetime.date.today()
            # edate = d.strftime('%Y-%m-%d')
        else:
            edate = date(int(enddate[0:4]), int(enddate[4:6]), int(enddate[6:]))
        bdate = date(int(begdate[0:4]), int(begdate[4:6]), int(begdate[6:]))


        delta = edate - bdate       # as timedelta
        for i in range(delta.days):    # do not grab current day
            day = bdate + timedelta(days=i)
            cdate = day.strftime('%Y-%m-%d')
            self.request_data(begdate=cdate)
        


    def request_data(self, begdate = "2019-11-01", enddate=None):
        '''default: no enddate, store each day seprately'''
        begdate_split = begdate.split('-')
        begyear = begdate_split[0]
        begdate = begdate_split[0]+begdate_split[1]+begdate_split[2]
        folder_path = self.data_path+begyear+'/'+begdate+'/'
        if not os.path.exists(folder_path):    # create folder path
            os.makedirs(folder_path)
        
        saved_files = os.listdir(folder_path)

        begdt = begdate[0:4]+'-'+begdate[4:6]+'-'+begdate[6:]
        enddt = ''

        if not enddate:
            enddt = begdt
        else:
            enddt = enddate

        for loc in self.loc:
            loc_file = "alabama_"+loc+'_'+begdate[0:4]+'_'+begdate[4:6]+'_'+begdate[6:]+".csv"
            data_path_tmp = folder_path+loc_file+".tmp"                 
            data_path = folder_path+loc_file
            
            if loc_file in saved_files:
                if (os.path.getsize(data_path) >> 10) > 20:
                    continue
                os.remove(data_path)

            data = "begdt="+begdt+"&enddt="+enddt+"&station="+loc+"&fmt=csv&select_deselect=1&var%5B%5D=RecId&var%5B%5D=TableCode&var%5B%5D=Year&var%5B%5D=Month&var%5B%5D=DayOfMon&var%5B%5D=DayOfYear&var%5B%5D=Hour&var%5B%5D=Minute&var%5B%5D=StationID&var%5B%5D=Lat&var%5B%5D=Lon&var%5B%5D=Elev&var%5B%5D=Sign&var%5B%5D=Door&var%5B%5D=Batt&var%5B%5D=ObsInSumm_Tot&var%5B%5D=Precip_TB3_Tot&var%5B%5D=Precip_TX_Tot&var%5B%5D=Precip_TB3_Today&var%5B%5D=Precip_TX_Today&var%5B%5D=PTemp&var%5B%5D=IRTS_Trgt&var%5B%5D=IRTS_Body&var%5B%5D=SoilSfcT&var%5B%5D=SoilT_5cm&var%5B%5D=SoilT_10cm&var%5B%5D=SoilT_20cm&var%5B%5D=SoilT_50cm&var%5B%5D=SoilT_100cm&var%5B%5D=AirT_1pt5m&var%5B%5D=AirT_2m&var%5B%5D=AirT_9pt5m&var%5B%5D=AirT_10m&var%5B%5D=RH_2m&var%5B%5D=RH_10m&var%5B%5D=Pressure_1&var%5B%5D=Pressure_2&var%5B%5D=TotalRadn&var%5B%5D=QuantRadn&var%5B%5D=WndDir_2m&var%5B%5D=WndDir_10m&var%5B%5D=WndSpd_2m&var%5B%5D=WndSpd_10m&var%5B%5D=WndSpd_Vert&var%5B%5D=Vitel_100cm_a&var%5B%5D=Vitel_100cm_b&var%5B%5D=Vitel_100cm_c&var%5B%5D=Vitel_100cm_d&var%5B%5D=WndSpd_2m_Max&var%5B%5D=WndSpd_10m_Max&var%5B%5D=WndSpd_Vert_Max&var%5B%5D=WndSpd_Vert_Min&var%5B%5D=WndSpd_2m_Std&var%5B%5D=WndSpd_10m_Std&var%5B%5D=SoilType&var%5B%5D=eR&var%5B%5D=eI&var%5B%5D=Temp_C&var%5B%5D=eR_tc&var%5B%5D=eI_tc&var%5B%5D=wfv&var%5B%5D=NaCI&var%5B%5D=SoilCond&var%5B%5D=SoilCond_tc&var%5B%5D=SoilWaCond_tc&var%5B%5D=WndSpd_2m_WVc_1&var%5B%5D=WndSpd_2m_WVc_2&var%5B%5D=WndSpd_2m_WVc_3&var%5B%5D=WndSpd_2m_WVc_4&var%5B%5D=WndSpd_10m_WVc_1&var%5B%5D=WndSpd_10m_WVc_2&var%5B%5D=WndSpd_10m_WVc_3&var%5B%5D=WndSpd_10m_WVc_4&var%5B%5D=WndSpd_Vert_Tot"
        
       
            headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:69.0) Gecko/20100101 Firefox/69.0',
                        'Host': 'chiliweb.southalabama.edu',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.5',
                        'Accept-Encoding': 'gzip, deflate',
                        'Content-Type': 'application/x-www-form-urlencoded',
                        'Connection': 'keep-alive',
                        'Referer': 'http://chiliweb.southalabama.edu/archived_data.php',
                        'Upgrade-Insecure-Requests': '1'}
        
            # req = requests.post(self.url,headers=headers, data=data)

            with requests.post(self.url, headers=headers, data=data, stream=True) as r:
                r.raise_for_status()
                with open(data_path_tmp, 'wb') as f: 
                    for chunk in r.iter_content(chunk_size=8192): 
                        if chunk: 
                            f.write(chunk)
                
            os.rename(data_path_tmp, data_path)

            self.logger.info('finish downloading: '+data_path)
            time.sleep(1)
        


 # data="begdt=2019-11-01&enddt=2019-11-02&station=agricola&fmt=csv&select_deselect=1&var%5B%5D=RecId&var%5B%5D=TableCode&var%5B%5D=Year&var%5B%5D=Month&var%5B%5D=DayOfMon&var%5B%5D=DayOfYear&var%5B%5D=Hour&var%5B%5D=Minute&var%5B%5D=StationID&var%5B%5D=Lat&var%5B%5D=Lon&var%5B%5D=Elev&var%5B%5D=Sign&var%5B%5D=Door&var%5B%5D=Batt&var%5B%5D=ObsInSumm_Tot&var%5B%5D=Precip_TB3_Tot&var%5B%5D=Precip_TX_Tot&var%5B%5D=Precip_TB3_Today&var%5B%5D=Precip_TX_Today&var%5B%5D=PTemp&var%5B%5D=IRTS_Trgt&var%5B%5D=IRTS_Body&var%5B%5D=SoilSfcT&var%5B%5D=SoilT_5cm&var%5B%5D=SoilT_10cm&var%5B%5D=SoilT_20cm&var%5B%5D=SoilT_50cm&var%5B%5D=SoilT_100cm&var%5B%5D=AirT_1pt5m&var%5B%5D=AirT_2m&var%5B%5D=AirT_9pt5m&var%5B%5D=AirT_10m&var%5B%5D=RH_2m&var%5B%5D=RH_10m&var%5B%5D=Pressure_1&var%5B%5D=Pressure_2&var%5B%5D=TotalRadn&var%5B%5D=QuantRadn&var%5B%5D=WndDir_2m&var%5B%5D=WndDir_10m&var%5B%5D=WndSpd_2m&var%5B%5D=WndSpd_10m&var%5B%5D=WndSpd_Vert&var%5B%5D=Vitel_100cm_a&var%5B%5D=Vitel_100cm_b&var%5B%5D=Vitel_100cm_c&var%5B%5D=Vitel_100cm_d&var%5B%5D=WndSpd_2m_Max&var%5B%5D=WndSpd_10m_Max&var%5B%5D=WndSpd_Vert_Max&var%5B%5D=WndSpd_Vert_Min&var%5B%5D=WndSpd_2m_Std&var%5B%5D=WndSpd_10m_Std&var%5B%5D=SoilType&var%5B%5D=eR&var%5B%5D=eI&var%5B%5D=Temp_C&var%5B%5D=eR_tc&var%5B%5D=eI_tc&var%5B%5D=wfv&var%5B%5D=NaCI&var%5B%5D=SoilCond&var%5B%5D=SoilCond_tc&var%5B%5D=SoilWaCond_tc&var%5B%5D=WndSpd_2m_WVc_1&var%5B%5D=WndSpd_2m_WVc_2&var%5B%5D=WndSpd_2m_WVc_3&var%5B%5D=WndSpd_2m_WVc_4&var%5B%5D=WndSpd_10m_WVc_1&var%5B%5D=WndSpd_10m_WVc_2&var%5B%5D=WndSpd_10m_WVc_3&var%5B%5D=WndSpd_10m_WVc_4&var%5B%5D=WndSpd_Vert_Tot"
  
            









if __name__ == "__main__":
    s = spider()
    s.log()
    s.request_range(begdate="20160101", enddate="2017101")
    # s.run(check_date="20191111")    # for example: "20190101"