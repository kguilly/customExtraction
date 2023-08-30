import pygrib

data_path = "/home/kaleb/wrf/"
parameter_list = [9, 36, 37, 38]
grib = pygrib.open(data_path)

for p in parameter_list:
    grib_messages = grib[p]
    lat, lon = grib_messages.latlons()
    grib_values = grib_messages.values