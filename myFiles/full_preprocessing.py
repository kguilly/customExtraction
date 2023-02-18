import pandas as pd
from herbie import Herbie
import numpy as np
from datetime import date, timedelta, datetime
import sys
import os
import geo_grid_recent as gg
from pathlib import Path


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
        state_abbrev_df = self.get_state_abbrevs(df=every_county_df)
        every_county_df['county'] = every_county_df['county'].apply(self.fix_county_names)
        self.read_data(df=every_county_df, dict=param_dict_arr, state_abbrev_df=state_abbrev_df)

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
                for i in range(fips_index, len(sys.argv)):
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
            if str(df['stateFips'][i]) not in state_dict:
                state_dict[str(df['stateFips'][i])] = {}
            if str(df['FIPS'][i]) not in state_dict[str(df['stateFips'][i])]:
                state_dict[str(df['stateFips'][i])][str(df['FIPS'][i])] = {}
            if str(df['countyGridIndex'][i]) not in state_dict[str(df['stateFips'][i])][str(df['FIPS'][i])]:
                state_dict[str(df['stateFips'][i])][str(df['FIPS'][i])][str(df['countyGridIndex'][i])] = []

        return state_dict

    def get_state_abbrevs(self, df=pd.DataFrame()):
        """
        Make a map of the states and their abbreviations
        :param df: data frame with the state info
        :return: df:: cols = ['stname', 'st', 'stusps'] where stname = full state name,
                     st = state fips and stusps = state abbreviation
        """
        state_file_path = "./countyInfo/us-state-ansi-fips2.csv"
        state_abbrev_df = pd.read_csv(state_file_path)
        return state_abbrev_df

    def fix_county_names(self, val):
        # find the column for the county name
        # for each element, capitalize, and then if they have the term
        # 'PARISH' or 'COUNTY' then remove it
        val = val.upper()
        if val.rfind('COUNTY') != -1:
            arr = val.split("COUNTY")
            val = ''
            for elem in arr:
                val += elem
        elif val.rfind('PARISH') != -1:
            arr = val.split('PARISH')
            val = ''
            for elem in arr:
                val += elem
        val.strip()
        return val

    def read_data(self, df=pd.DataFrame, dict={}, state_abbrev_df=pd.DataFrame):
        """

        :param df: a df acting as the object, holding all of the
                information for all of the counties and their grid indexes
        :param dict: A dict that will hold a given hour's parameter information.
                dict[state][county][countyIndex] = One hour's parameters
        :return: Nun
        """

        # make dictonary of grid indexes and their avg lats and lons to put to find
        # closest points in grib files
        grid_names, lon_lats = self.make_lat_lon_name_arr(df=df)

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
                              save_dir=self.write_path, verbose=False,
                              priority=['pando', 'pando2', 'aws', 'nomads',
                                        'ecmwf', 'aws-old', 'google', 'azure'],
                              fxx=0, overwrite=False)
            except:
                print("Could not find grib object for date: %s" % dtobj.date())
                dtobj = next_hour
                continue
            try:
                precip_herb = Herbie(dtobj, model='hrrr', product='sfc',
                                     save_dir=self.write_path, verbose=False,
                                     priority=['pando', 'pando2', 'aws', 'nomads',
                                               'ecmwf', 'aws-old', 'google', 'azure'],
                                     fxx=1, overwrite=False)
            except:
                print("Could not find precipitation object for date: %s" % dtobj.date())
                dtobj = next_hour
                continue

            temp_rh_obj = herb.xarray(":(?:TMP|RH):2 m").herbie.nearest_points(points=lon_lats, names=grid_names)
            precip_herb_obj = precip_herb.xarray(":APCP:surface:").herbie.nearest_points(points=lon_lats, names=grid_names)
            u_v_wind_obj = herb.xarray(":(U|V)GRD:1000 mb:").herbie.nearest_points(points=lon_lats, names=grid_names)
            dswrf_obj = herb.xarray(":DSWRF:surface:anl").herbie.nearest_points(points=lon_lats, names=grid_names)
            gust_obj = herb.xarray(":GUST:").herbie.nearest_points(points=lon_lats, names=grid_names)

            # each index of these values arrays will correspond to the index in the grid_names array
            temp_vals = temp_rh_obj.t2m.values
            precip_vals = precip_herb_obj.tp.values
            dswrf_vals = dswrf_obj.dswrf.values
            rh_vals = temp_rh_obj.r2.values
            u_wind_vals = u_v_wind_obj.u.values
            v_wind_vals = u_v_wind_obj.v.values
            gust_vals = gust_obj.gust.values

            # match the parameter to the index in the dict, then write out to file
            grid_name_idx = 0
            for stateFips in dict:
                for countyFips in dict[stateFips]:
                    for countyIndex in dict[stateFips][countyFips]:
                        if not grid_names[grid_name_idx] == countyFips + "_" + str(countyIndex):
                            # find the correct grid_name_idx
                            grid_name_idx = -1
                            for i in range(len(grid_names)):
                                if grid_names[i] == countyFips + "_" + str(countyIndex):
                                    grid_name_idx = i
                                    break
                            if grid_name_idx == -1:
                                print("Error when finding the matching county in grid_names.")
                                exit()

                        # append the values to the index of the dictionary
                        #  Year, Month, Day, Daily/Monthly, State, County, FIPS Code, Grid Index,
                        # Lat (llcrnr), Lon (llcrnr), Lat (urcrnr), Lon (urcrnr), Avg Temperature (K),
                        # Precipitation (kg m**-2), Relative Humidity (%), Wind Gust (m s**-1),
                        # U Component of Wind (m s**-1), V Component of Wind (m s**-1), Downward Shortwave
                        # Radiation Flux (W m**-2)
                        state_abbrev = state_abbrev_df['stusps'].where(state_abbrev_df['st']==int(stateFips)).dropna().values[0]
                        county_name = df['county'].where(df['FIPS'] == int(countyFips)).dropna().values[0]
                        df_idx = df.index[(df['FIPS'] == int(countyFips)) & (df['countyGridIndex'] == int(countyIndex))].tolist()[0]
                        dict[stateFips][countyFips][countyIndex].append([str(dtobj.year), str(dtobj.month),
                                                                        str(dtobj.day), 'Daily', state_abbrev, county_name,
                                                                        str(countyFips), str(countyIndex), df['lat (llcrnr)'][df_idx],
                                                                        df['lon (llcrnr)'][df_idx], df['lat (urcrnr)'][df_idx],
                                                                        df['lon (urcrnr)'][df_idx], temp_vals[grid_name_idx],
                                                                        precip_vals[grid_name_idx], rh_vals[grid_name_idx],
                                                                        gust_vals[grid_name_idx], u_wind_vals[grid_name_idx],
                                                                        v_wind_vals[grid_name_idx], dswrf_vals[grid_name_idx]])
                        # append this to the end of the appropriate file
                        self.write_dict_row(row=dict[stateFips][countyFips][countyIndex][0])
                        # clear the array
                        dict[stateFips][countyFips][countyIndex] = []
                        grid_name_idx += 1
            dtobj = next_hour

    def make_lat_lon_name_arr(self, df=pd.DataFrame()):
        """

        :param df: the dataframe of the station data
        :return: a dictionary, key=grid_county_name , val = tuple of avglon, avglat
        """
        names = []
        lon_lats = []
        for i in range(len(df)):
            grid_county_name = str(df['FIPS'][i]) + '_' + str(df['countyGridIndex'][i])
            avg_lon = (float(df['lon (urcrnr)'][i]) + float(df['lon (llcrnr)'][i])) / 2
            avg_lat = (float(df['lat (urcrnr)'][i]) + float(df['lat (llcrnr)'][i])) / 2

            names.append(grid_county_name)
            lon_lats.append((avg_lon, avg_lat))


        return names, lon_lats

    def write_dict_row(self, row=[]):
        """

        :param row: a single row from a dictionary, containing all the necessary information
                    to find the file
        :return: nadda
        """
        # find the right file that we needa write to
        # write path + daily_data + year + state_abbrev + countyFips + monthly file
        # <HRRR_<state_fips>_<state_abbrev>_<year>-<month>>.csv
        year = row[0]
        state_abbrev = row[4]
        month = row[1]
        county_fips = row[6]
        state_fips = county_fips[0:2]
        output_directory = self.write_path + "daily_data/" + year + '/' + state_abbrev + '/' + \
                           county_fips + '/'
        # if it does not already exist, make it
        Path(output_directory).mkdir(parents=True, exist_ok=True)
        output_file_path = output_directory + "HRRR_" + state_fips + '_' + state_abbrev + '_' + year\
                           + '-' + month + '.csv'
        # if the file does not already exist, write the appropriate header out to it
        if not os.path.exists(output_file_path):
            # make the file
            f = open(output_file_path, mode='wt')
            f.write('Year,Month,Day,Daily/Monthly,State,County,FIPS Code,Grid Index,' +
                    'Lat (llcrnr),Lon (llcrnr),Lat (urcrnr),Lon (urcrnr),Avg Temperature (K),' +
                    'Precipitation (kg m**-2),Relative Humidity (%),Wind Gust (m s**-1),' +
                    'U Component of Wind (m s**-1),V Component of Wind (m s**-1),' +
                    'Downward Shortwave Radiation Flux (W m**-2)\n')
            f.close()
        # now we are free to write out the appropriate information to it
        f = open(output_file_path, mode='a')
        for r in row:
            f.write(str(r) + ',')
        f.write('\n')
        f.close()

if __name__ == "__main__":
    p = PreprocessWRF()
    p.main()
