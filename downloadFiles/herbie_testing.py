import sys
from herbie import Herbie
from datetime import datetime, timedelta
import pygrib
import numpy as np
import pandas as pd

parameters = np.linspace(1, 150, num=150, dtype=int)
date = "20230201"
dt = date + " 10:00"
save_dir = '/home/kaleb/data/'
dtobj = datetime.strptime(dt, "%Y%m%d %H:%M")
pred_hours = 6

H = Herbie(
    dt,
    model="hrrr",
    product="nat",
    fxx=pred_hours,
    save_dir=save_dir,
    verbose=True,
    overwrite=True
)
H.download(":APCP:")

grib_path = save_dir + "hrrr/" + date + \
            '/subset_d7b2ef06__hrrr.t00z.wrfnatf06.grib2'
grib = pygrib.open(grib_path)

for p in parameters:
    try:
        gribmsgs = grib[int(p)]
    except OSError:
        print("Not that many messages")
        break
    print("Layer: %s Name: %s  Units: %s  Level: %s (%s)" % (
    p, gribmsgs.name, gribmsgs.units, gribmsgs.level, gribmsgs.typeOfLevel))
    data = gribmsgs.values
    lt, ln = gribmsgs.latlons()

    for val in data:
        for subval in val:
            if subval > 0:
                print(subval)

    # for i in range(len(data)):
    #     if data[i] > 0:
    #         print("Val:%s, lat: %s, lon: %s" % (data[i], lt[i], ln[i]))

