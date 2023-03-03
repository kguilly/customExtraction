import sys
from herbie import Herbie
from datetime import datetime, timedelta
import pygrib
import numpy as np
import pandas as pd

grib_path = '/Users/kkjesus/data/hrrr/20230201/subset_85e4fa12__hrrr.t10z.wrfnatf01.grib2'
grib = pygrib.open(grib_path)
for g in grib:
    print(g)
exit()
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

