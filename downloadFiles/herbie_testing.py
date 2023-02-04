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
pred_hours = 5

dt2 = date + " 00:00"
H2 = Herbie(
    dt2,
    model="hrrr",
    product="nat",
    save_dir=save_dir,
    fxx=pred_hours,
    verbose=True,
    overwrite=False
)
H2.tell_me_everything()
# H2.download(":APCP:surface:")
balls = H2.xarray(":APCP:")
points = balls.herbie.nearest_points(points=[(-91.0198, 30.2241)])
print(points.tp)
exit()

grib_path = '/home/kaleb/data/hrrr/20230201/subset_d7b5ef06__hrrr.t00z.wrfnatf23.grib2'
grib = pygrib.open(grib_path)

for p in parameters:
    try:
        grib_msgs = grib[int(p)]
    except OSError:
        print("Not that many messages")
        break
    print("Layer: %s Name: %s  Units: %s  Level: %s (%s)" % (
    p, grib_msgs.name, grib_msgs.units, grib_msgs.level, grib_msgs.typeOfLevel))

    lat_st_mid = 30.2241
    lon_st_mid = -92.0198
    data = grib_msgs.values
    lt, ln = grib_msgs.latlons()
    st_lt_m = np.full_like(lt, lat_st_mid)
    st_ln_m = np.full_like(ln, lon_st_mid)
    dis_mat = (lt - st_lt_m) ** 2 + (ln - st_ln_m) ** 2
    p_lt, p_ln = np.unravel_index(dis_mat.argmin(), dis_mat.shape)
    p_lt_dict = {}
    p_ln_dict = {}
    index = str(30)
    p_lt_dict[index] = []
    p_ln_dict[index] = []
    p_lt_dict[index].append(p_lt)
    p_ln_dict[index].append(p_ln)
    value = data[p_lt_dict[index], p_ln_dict[index]]
    print(p_lt_dict[index], " ", p_ln_dict[index])
    print()
    print(value)
    # for i in range(len(data)):
    #     if data[i] > 0:
    #         print("Val:%s, lat: %s, lon: %s" % (data[i], lt[i], ln[i]))

