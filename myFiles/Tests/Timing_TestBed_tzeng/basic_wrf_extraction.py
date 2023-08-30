import pygrib
import sys
import os
from memory_profiler import profile
from datetime import date, timedelta, datetime
import time

# @profile
def main():
    wrf_file_path = "/media/kaleb/extraSpace/wrf/"

    ymd = "20220101"
    hour = "00"

    if len(sys.argv) < 1:
        print("pass args")
        exit()

    for i in range(1, len(sys.argv)):
        
        if sys.argv[i] == "--ymd":
            ymd = sys.argv[i+1]

        if sys.argv[i] == "--hour":
            hour = str(sys.argv[i+1]).zfill(2)

    year = ymd[0:4]
    month = ymd[4:6]
    day = ymd[6:8]

    full_wrf_file_path = wrf_file_path + year + '/' + ymd + '/hrrr.' + ymd + \
                        '.' + hour + '.00.grib2'


    print("Extracting file %s" % full_wrf_file_path)

    grib = pygrib.open(full_wrf_file_path)
    if not grib:
        print("Could not open grib file")
        exit()

    for g in grib:
        lats,lons = g.latlons()
        values = g.values


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print("\n\nTime: %s" % (end_time - start_time))
