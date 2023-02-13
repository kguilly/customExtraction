import os

from herbie import Herbie
import urllib.error
from herbie import Herbie
from datetime import datetime, timedelta
from pathlib import Path

class HerbDownload():
    def __init__(self):
        self.info = "Download hourly HRRR WRF data for a select date range"
        self.data_path = "/home/kaleb/data/" # the path to download the WRF data
        self.begin_date = "20160101" # format: yyyymmdd
        self.end_date = "20170101" # not inclusive of last day
        self.start_time = "00:00"

    def main(self):
        dt = self.begin_date + " " + self.start_time
        dtobj = datetime.strptime(dt, "%Y%m%d %H:%M")
        while 1:
            print(dtobj.date())
            try:
                herb = Herbie(dtobj, model='hrrr',
                              product='sfc', save_dir=self.data_path, verbose=True,
                              priority=['pando', 'pando2', 'aws', 'nomads',
                                        'ecmwf', 'aws-old', 'google', 'azure'],
                              fxx=0,
                              overwrite=False)
            except:
                print("Could not find grib object for date: %s" % dtobj.date())
                next_hour = dtobj + timedelta(hours=1)
                if next_hour.strftime("%Y%m%d") == self.end_date:
                    break
                dtobj = next_hour
                continue

            herb.download()
            original_file_name = self.data_path + "hrrr/" + str(dtobj.strftime("%Y%m%d"))\
                                + '/' + 'hrrr.t' + str(dtobj.strftime("%H")) + 'z.wrfsfcf00.grib2'
            changed_file_path = self.data_path + "wrf/" + str(dtobj.year) + '/'\
                                + str(dtobj.strftime("%Y%m%d")) + '/'
            Path(changed_file_path).mkdir(parents=True, exist_ok=True)
            changed_file_name = changed_file_path + 'hrrr.'\
                                + str(dtobj.strftime("%Y%m%d"))\
                                + '.' + str(dtobj.strftime("%H")) + '.00.grib2'
            os.rename(original_file_name, changed_file_name)

            next_hour = dtobj + timedelta(hours=1)
            if next_hour.strftime("%Y%m%d") == self.end_date:
                break
            dtobj = next_hour


if __name__ == "__main__":
    s = HerbDownload()
    s.main()
    # HerbDownload.main()