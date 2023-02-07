from herbie import Herbie
from datetime import datetime
import numpy as np

date = "20230201"
save_dir = '/home/kaleb/data/'

pred_hours = 1

date_hour = date
sum = 0
for i in range(0,24):
    if i < 10:
        date_hour = date + " 0" + str(i) + ":00"
    else:
        date_hour = date + " " + str(i) + ':00'

    herb_obj = Herbie(date_hour, model="hrrr",
                      product="nat", save_dir=save_dir,
                      fxx=pred_hours, verbose=True,
                      overwrite=True).xarray(":APCP:", remove_grib=True).herbie.nearest_points(points=[(-91.0198, 30.2241)])
    print("hr: %s, val: %s" % (i, herb_obj.tp.values))
    sum += herb_obj.tp.values

print("total: %s" % sum)