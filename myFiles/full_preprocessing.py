import pandas as pd
from herbie import Herbie
import numpy as np
from datetime import date, timedelta, datetime
import sys
import os
import geo_grid_recent as gg


class PreprocessWRF():
    def __init__(self):
        self.write_path = "/home/kaleb/Desktop/full_preprocessing_output/"

        self.begin_hour = 0
        self.end_hour = 23
        self.begin_date = "20200101"  # format as "yyyymmdd"
        self.end_date = "20200102"
        self.begin_hour = "00:00"
        self.end_hour = "23:00"
        self.county_df = pd.DataFrame()
        self.passedFips = []

    def main(self):
        self.handle_args()
        every_county_df = pd.read_csv("./WRFoutput/wrfOutput.csv")
        param_dict_arr = self.separate_by_state(df=every_county_df)
        self.read_data(df=every_county_df, dict=param_dict_arr)

    def handle_args(self):
        if len(sys.argv) > 1:
            fips_flag = False
            fips_index = 0

            for i in range(1, len(sys.argv)):

                if sys.argv[i] == "--begin_date":
                    self.begin_date = sys.argv[i + 1]

                elif sys.argv[i] == "--end_date":
                    self.end_date = sys.argv[i + 1]

                elif sys.argv[i] == "--fips":
                    fips_flag = True
                    fips_index = i + 1

            if fips_flag:
                self.passedFips = []
                for i in range(fips_index + 1, len(sys.argv)):
                    arg = sys.argv[i]
                    if len(arg) == 5:
                        # the correct length for a fips code
                        self.passedFips.append(arg)
                    else:
                        print("Error, the incorrect length of fips argument passed, please try again")
                        exit(0)
            gg.county(fips=self.passedFips)

    def separate_by_state(self, df=pd.DataFrame()):
        """

        :param df: a dataframe of fips codes and information obtained by running
                geo_grid
        :return: a dictionary, separated by state. Each state has its own dictionary of
                counties, each county has its own dictionary of parameter information
        """
        state_dict = {}
        for i in range(len(df)):
            if str(df['stateFips']) not in state_dict:
                state_dict[str(df['stateFips'])] = {}
                if str(df['FIPS']) not in state_dict[str(df['stateFips'])]:
                    state_dict[str(df['stateFips'])][str(df['FIPS'])] = {}
                    if str(df['countyGridIndex']) not in state_dict[str(df['stateFips'])][str(df['FIPS'])]:
                        state_dict[str(df['stateFips'])][str(df['FIPS'])][str(df['countyGridIndex'])] = {}

        return state_dict

    def read_data(self, df=pd.DataFrame, dict={}):
        """

        :param df: a df acting as the object, holding all of the
                information for all of the counties and their grid indexes
        :param dict: A dict that will hold a given hour's parameter information.
                dict[state][county][countyIndex] = One hour's parameters
        :return: Nun
        """

        # make dictonary of grid indexes and their avg lats and lons to put to find
        # closest points in grib files
        lon_lats, grid_names = self.make_lat_lon_name_arr(df)
        # for each hour in each day passed, try to read all the parameters that fudong needs
        # if the file cannot be found, then print a statement notifying the user and append 0s continue
        dt = self.begin_date + " " + self.begin_hour
        dtobj = datetime.strptime(dt, "%Y%m%d %H:%M")
        while 1:
            print(dtobj)
            # check to see if we're on the last hour or the last day
            next_hour = dtobj + timedelta(hours=1)
            if next_hour.strftime("%Y%m%d") == self.end_date:
                break
            elif dtobj.strftime("%H:%M") == self.end_hour:
                next_hour = datetime.strptime(dtobj.strftime("%Y%m%d") + " " + self.begin_hour, "%Y%m%d %H:%M")

            try:
                herb = Herbie(dtobj, model='hrrr', product='sfc',
                              save_dir=self.write_path, verbose=True,
                              priority=['pando', 'pando2', 'aws', 'nomads',
                                        'ecmwf', 'aws-old', 'google', 'azure'],
                              fxx=0, overwrite=False)
                precip_herb = Herbie(dtobj, model='hrrr', product='sfc',
                                     save_dir=self.write_path, verbose=True,
                                     priority=['pando', 'pando2', 'aws', 'nomads',
                                               'ecmwf', 'aws-old', 'google', 'azure'],
                                     fxx=1, overwrite=False)
            except:
                print("Could not find grib object for date: %s" % dtobj.date())
                dtobj = next_hour
                continue

            temp_rh_obj = herb.xarray(":(?:TMP|RH):2 m").herbie.nearest_points(points=lon_lats, names=grid_names)
            precip_herb_obj = herb.xarray(":APCP:surface:").herbie.nearest_points(points=lon_lats, names=grid_names)
            u_v_wind_obj = herb.xarray(":(U|V)GRD:1000 mb:").herbie.nearest_points(points=lon_lats, names=grid_names)
            dswrf_obj = herb.xarray(":DSWRF:surface:anl").herbie.nearest_points(points=lon_lats, names=grid_names)

            temp_vals = temp_rh_obj.t2m.values
            precip_vals = precip_herb_obj.tp.values
            dswrf_vals = dswrf_obj.dswrf.values
            rh_vals = temp_rh_obj.r2.values
            u_wind_vals = u_v_wind_obj.u.values
            v_wind_vals = u_v_wind_obj.v.values

            dtobj = next_hour

    def make_lat_lon_name_arr(self, df=pd.DataFrame()):
        """

        :param df: the dataframe of the station data
        :return: a dictionary, key=grid_county_name , val = tuple of avglon, avglat
        """
        names, lon_lats = []
        for i in range(len(df)):
            grid_county_name = str(df['FIPS'][i]) + '_' + str(df['countyGridIndex'][i])
            avg_lon = (float(df['lon (urcrnr)']) + float(df['lon (llcrnr)'])) / 2
            avg_lat = (float(df['lat (urcrnr)']) + float(df['lat (llcrnr)'])) / 2

            names.append(grid_county_name)
            lon_lats.append((avg_lon, avg_lat))


        return names, lon_lats

if __name__ == "__main__":
    p = PreprocessWRF()
    p.main()
