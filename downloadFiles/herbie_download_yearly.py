# this file will download the F01 file for precipitation for a given year
import urllib.error

from herbie import Herbie
from datetime import datetime, timedelta
import os
import pathlib
year = "2021"
herb_dir = "/home/kaleb/Desktop/herbie_data_" + year + '/'
start_day = "0101"
start_time = "00:00"
dt = year + start_day + ' ' + start_time

dtobj = datetime.strptime(dt, "%Y%m%d %H:%M")
while 1:
    try:
        herb = Herbie(dtobj, model='hrrr',
                      product='nat', save_dir=herb_dir,
                      fxx=1, verbose=True,
                      overwrite=False).download(":APCP:surface:")
    except ValueError:
        print("no herb obj for " + str(dtobj))
    except urllib.error.URLError:
        print("url error")
    next_hour = dtobj + timedelta(hours=1)
    print(next_hour)
    if next_hour.year == int(year) + 1:
        break

    dtobj = next_hour


