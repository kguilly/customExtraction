import pandas as pd
from herbie import Herbie
import numpy as np
from datetime import date, timedelta, datetime
import time
import sys
import os
import geo_grid_recent as gg
from pathlib import Path
import threading
import multiprocessing
import logging
import warnings
import queue

'''
ADD self.max_threads that will explicitly state the maximum amount of threads
that are allowed for a given machine - too many procs running on a slower machine 
will break it

Thread the monthly extraction - do this with threading instead of multiprocessing
follow self.max_threads
'''


class PreprocessWRF:
    def __init__(self):
        self.write_path = "/home/kaleb/Desktop/3-1_testing/"

        self.begin_date = "20200101"  # format as "yyyymmdd"
        self.end_date = "20200102"
        self.begin_hour = "00:00"
        self.end_hour = "24:00"
        self.county_df = pd.DataFrame()
        self.passedFips = []
        self.timeout_time = 801

        self.lock = multiprocessing.Lock()
        self.herb_lock = multiprocessing.Lock()
        self.precip_lock = multiprocessing.Lock()


    def main(self):
        start_time = time.time()
        # configure logging
        Path(self.write_path).mkdir(parents=True, exist_ok=True)
        logfilename = self.write_path + 'full_preproc_' + self.begin_date + '-' + self.end_date + '.log'
        logging.basicConfig(filemode='w', filename=logfilename, format='%(levelname)s - %(message)s')
        #
        self.handle_args()
        every_county_df = pd.read_csv("./WRFoutput/wrfOutput.csv")
        param_dict_arr = self.separate_by_state(df=every_county_df)
        state_abbrev_df = self.get_state_abbrevs(df=every_county_df)
        every_county_df['county'] = every_county_df['county'].apply(self.fix_county_names)
        self.read_data(df=every_county_df, st_dict=param_dict_arr, state_abbrev_df=state_abbrev_df)

        # the data is read into hourly, stately files, now read them into daily / monthly files
        csvFiles = self.reopen_files()
        for file in csvFiles:
            print("Get Monthly Avg for %s" % file)
            df = pd.read_csv(file, index_col=False, na_filter=False, na_values='N/A')
            df = self.wind_speed_vpd(df=df)
            df = self.dailyAvgMinMax(df=df)
            df = self.monthlyAvgs(df=df)
            self.final_sendoff(df=df, fullfilepath=file)
        finish = time.time()
        print("\n\n------------- %s seconds ---------------" % (finish - start_time))

    def handle_args(self):
        if len(sys.argv) > 1:
            fips_flag = False
            fips_index = 0
            help_flag = False

            for i in range(1, len(sys.argv)):

                if sys.argv[i] == "--begin_date":
                    self.begin_date = sys.argv[i + 1]

                elif sys.argv[i] == "--end_date":
                    self.end_date = sys.argv[i + 1]

                elif sys.argv[i] == "--fips":
                    fips_flag = True
                    fips_index = i + 1

                elif sys.argv[i] == '-h' or sys.argv[i] == '--help':
                    help_flag = True
                    break

            if help_flag:
                print("This is a file to extract select paramters for a select date range from counties"
                      "or states that are passed as arguments. The argument list is as follows:"
                      "-h --help           Display this screen"
                      "--fips              pass one to a number of fips codes, separated by spaces"
                      "--fips state=...    pass state abbreviations, will extract every county in the given"
                      "                    states. separate states by commas. Only put space b/w states"
                      "                    if the argument is encapsulated by quotes. "
                      "                    Ex. --fips \"state=LA, MS, IL\"")

            if fips_flag:
                self.passedFips = []
                for i in range(fips_index, len(sys.argv)):
                    arg = sys.argv[i]
                    if arg.upper().rfind("STATE") != -1:
                        sep = arg.split('=')
                        states = sep[1].split(',')
                        for j in range(len(states)):
                            states[j] = states[j].strip()
                            states[j] = states[j].upper()
                            # print(states[j])
                        all_county_df = pd.read_csv("./countyInfo/countyFipsandCoordinates.csv", dtype=str)
                        state_abbrev_df = pd.read_csv("./countyInfo/us-state-ansi-fips2.csv", dtype=str)
                        county_fips = []
                        for state in states:
                            this_state_fips = \
                                state_abbrev_df['st'].where(state_abbrev_df['stusps'] == state).dropna().values[0]
                            county_fips_from_state = all_county_df[
                                all_county_df['fips_code'].str.startswith(this_state_fips)]
                            county_fips_from_state = county_fips_from_state['fips_code'].dropna().values
                            county_fips.append(county_fips_from_state)
                        for arr in county_fips:
                            for val in arr:
                                self.passedFips.append(val)

                    else:
                        if len(arg) == 5:
                            # the correct length for a fips code
                            self.passedFips.append(arg)
                        else:
                            print("Error, the incorrect length of fips argument passed, please try again")
                            exit(0)
                gg.county(fips=self.passedFips)
            else:
                # we're either just gonna read the WRFoutput file or read the self.passedFips arg
                if len(self.passedFips) > 1:
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
        state_abbrev_df = pd.read_csv(state_file_path, dtype=str)
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

    def read_data(self, df=pd.DataFrame(), st_dict={}, state_abbrev_df=pd.DataFrame()):
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
        begin_day_dt = datetime.strptime(self.begin_date, "%Y%m%d")
        end_day_dt = datetime.strptime(self.end_date, "%Y%m%d")
        begin_hour_dt = datetime.strptime(self.begin_hour, "%H:%M")
        end_hour_dt = datetime.strptime(self.end_hour, "%H:%M")

        hour_range = (end_hour_dt - begin_hour_dt).seconds // (60 * 60)
        date_range = (end_day_dt - begin_day_dt).days

        # define the max amount of time for a process to run in seconds
        TIMEOUT = self.timeout_time
        for i in range(0, date_range):
            threads = []
            dtobj = begin_day_dt + timedelta(days=i)
            print(dtobj.strftime("%Y%m%d"))
            proc_start_time = time.time()
            for j in range(0, hour_range):
                dtobj = begin_day_dt + timedelta(days=i, hours=j)
                dict = st_dict
                # t = threading.Thread(target=self.threaded_read, args=(dtobj, dict, lon_lats, grid_names,
                #                                                      state_abbrev_df, df))
                t = multiprocessing.Process(target=self.threaded_read, args=(dtobj, dict, lon_lats, grid_names,
                                                                             state_abbrev_df, df))
                t.start()
                threads.append(t)

            completed_threads = []
            while time.time() - proc_start_time <= TIMEOUT:
                for t in threads:
                    if not t.is_alive():
                        threads.remove(t)
                        completed_threads.append(t)
                        break
                    time.sleep(3)
                if len(threads) < 1:
                    break

            for t in threads:
                completed_threads.append(t)
                logger = logging.getLogger()
                logger.warning("TIMEOUT: killing process %s for date %s" % (t.name, dtobj.strftime("%Y%m%d")))
                t.terminate()

            for t in completed_threads:
                t.join()

            print("------------------ %s seconds --------------" % (time.time() - proc_start_time))

    def threaded_read(self, dtobj: datetime, dict={}, lon_lats={}, grid_names={}, state_abbrev_df=[], df=[]):
        gust_vals, dswrf_vals, v_wind_vals, u_wind_vals, precip_vals, rh_vals, temp_vals = self.grab_herbie_arrays(
            dtobj, lon_lats, grid_names)
        # match the parameter to the index in the dict, then write out to file
        print("Grabbed Herb Arrs for " + dtobj.strftime("%Y%m%d %H:%M"))
        for stateFips in dict:
            for countyFips in dict[stateFips]:
                for countyIndex in dict[stateFips][countyFips]:
                    grid_name_idx = -1
                    break_flag = False
                    for state in grid_names:
                        if break_flag:
                            break
                        for i in range(len(grid_names[state])):
                            if grid_names[state][i] == countyFips + '_' + str(countyIndex):
                                grid_name_idx = i
                                break_flag = True
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
                    state_abbrev = \
                        state_abbrev_df['stusps'].where(state_abbrev_df['st'] == stateFips).dropna().values[0]
                    state_name = \
                        state_abbrev_df['stname'].where(state_abbrev_df['st'] == stateFips).dropna().values[0]
                    county_name = df['county'].where(df['FIPS'] == int(countyFips)).dropna().values[0]
                    df_idx = df.index[
                        (df['FIPS'] == int(countyFips)) & (df['countyGridIndex'] == int(countyIndex))].tolist()[
                        0]
                    dict[stateFips][countyFips][countyIndex].append(
                        [str(dtobj.strftime("%Y")), str(dtobj.strftime("%m")),
                         str(dtobj.strftime("%d")), str(dtobj.strftime("%H")), 'Daily', state_name.upper(), county_name,
                         str(countyFips), str(countyIndex), df['lat (llcrnr)'][df_idx],
                         df['lon (llcrnr)'][df_idx], df['lat (urcrnr)'][df_idx],
                         df['lon (urcrnr)'][df_idx], temp_vals[grid_name_idx],
                         precip_vals[grid_name_idx], rh_vals[grid_name_idx],
                         gust_vals[grid_name_idx], u_wind_vals[grid_name_idx],
                         v_wind_vals[grid_name_idx], dswrf_vals[grid_name_idx]])
                    # append this to the end of the appropriate file
                    with self.lock:
                        self.write_dict_row(row=dict[stateFips][countyFips][countyIndex][0], state_abbrev=state_abbrev)
                    # clear the array
                    dict[stateFips][countyFips][countyIndex] = []

    def grab_herbie_arrays(self, dtobj, lon_lats=[], grid_names=[]):
        logger = logging.getLogger()
        with self.herb_lock:
            try:
                herb = Herbie(dtobj, model='hrrr', product='sfc',
                              save_dir=self.write_path, verbose=False,
                              priority=['pando', 'pando2', 'aws', 'nomads',
                                        'google', 'azure'],  # , 'ecmwf', 'aws-old'],
                              fxx=0, overwrite=False)
            except:
                # print("Could not find grib object for date: %s" % dtobj.strftime("%Y-%m-%d %H:%M"))
                logger.warning("Herb obj not found for date " + dtobj.strftime("%Y%m%d %H:%M"))
                herb = None

        with self.precip_lock:
            try:
                precip_herb = Herbie(dtobj, model='hrrr', product='sfc',
                                     save_dir=self.write_path, verbose=False,
                                     priority=['pando', 'pando2', 'aws', 'nomads',
                                               'google', 'azure'],  # , 'ecmwf', 'aws-old'],
                                     fxx=1, overwrite=False)
            except:
                # print("Could not find precipitation object for date: %s" % dtobj.strftime("%Y-%m-%d %H:%M"))
                logger.warning("Precipitation obj not found for date " + dtobj.strftime("%Y%m%d %H:%M"))
                precip_herb = None

        # index out each of the necessary points with their search string and place them into arrays to be sorted
        # some may fail so they have been put into try catch blocks

        temp_vals = np.empty((1,))
        rh_vals = np.empty((1,))
        precip_vals = np.empty((1,))
        u_wind_vals = np.empty((1,))
        v_wind_vals = np.empty((1,))
        dswrf_vals = np.empty((1,))
        gust_vals = np.empty((1,))

        temp_vals = np.delete(temp_vals, 0)
        rh_vals = np.delete(rh_vals, 0)
        precip_vals = np.delete(precip_vals, 0)
        u_wind_vals = np.delete(u_wind_vals, 0)
        v_wind_vals = np.delete(v_wind_vals, 0)
        dswrf_vals = np.delete(dswrf_vals, 0)
        gust_vals = np.delete(gust_vals, 0)

        try:
            temp_rh_obj = herb.xarray(":(?:TMP|RH):2 m").herbie
        except:
            temp_rh_obj = None
            if herb:
                logger.warning("Temp / RH vals not found for date %s" % (dtobj.strftime("%Y%m%d %H:%M")))
        try:
            precip_obj = precip_herb.xarray(":APCP:surface:", remove_grib=True).herbie
        except:
            precip_obj = None
            if precip_herb:
                logger.warning("Precip vals not found for date %s" % (dtobj.strftime("%Y%m%d %H:%M")))
        try:
            u_v_wind_obj = herb.xarray(":(U|V)GRD:1000 mb:").herbie
        except:
            u_v_wind_obj = None
            if herb:
                logger.warning("Wind dir vals not found for date %s " % (dtobj.strftime("%Y%m%d %H:%M")))
        try:
            dswrf_obj = herb.xarray(":DSWRF:surface:anl").herbie
        except:
            dswrf_obj = None
            if herb:
                logger.warning("DSWRF vals not found for date %s" % (dtobj.strftime("%Y%m%d %H:%M")))
        try:
            gust_obj = herb.xarray(":GUST:", remove_grib=True).herbie
        except:
            gust_obj = None
            if herb:
                logger.warning("Wind Gust vals not found for date %s" % (dtobj.strftime("%Y%m%d %H:%M")))

        for state in lon_lats:
            try:
                t2m_rh_herb = temp_rh_obj.nearest_points(points=lon_lats[state],
                                                                names=grid_names[state])
                temp_vals = np.concatenate((temp_vals, t2m_rh_herb.t2m.values))
                rh_vals = np.concatenate((rh_vals, t2m_rh_herb.r2.values))
            except:
                temp_vals = np.concatenate((temp_vals, np.full((len(grid_names[state]),), -12345.678)))
                rh_vals = np.concatenate((rh_vals, np.full((len(grid_names[state]),), -12345.678)))

            try:
                precip_herb_obj = precip_obj.nearest_points(points=lon_lats[state],
                                                                   names=grid_names[state])
                precip_vals = np.concatenate((precip_vals, precip_herb_obj.tp.values))
            except:
                precip_vals = np.concatenate((precip_vals, np.zeros((len(grid_names[state]),))))

            try:
                u_v_wind_herb_obj = u_v_wind_obj.nearest_points(points=lon_lats[state],
                                                                       names=grid_names[state])
                u_wind_vals = np.concatenate((u_wind_vals, u_v_wind_herb_obj.u.values))
                v_wind_vals = np.concatenate((v_wind_vals, u_v_wind_herb_obj.v.values))
            except:
                u_wind_vals = np.concatenate((u_wind_vals, np.full((len(grid_names[state]),), -12345.678)))
                v_wind_vals = np.concatenate((v_wind_vals, np.full((len(grid_names[state]),), -12345.678)))

            try:
                dswrf_herb = dswrf_obj.nearest_points(points=lon_lats[state],
                                                             names=grid_names[state])
                dswrf_vals = np.concatenate((dswrf_vals, dswrf_herb.dswrf.values))
            except:
                dswrf_vals = np.concatenate((dswrf_vals, np.zeros((len(grid_names[state]),))))

            try:
                gust_herb = gust_obj.nearest_points(points=lon_lats[state],
                                                           names=grid_names[state])
                gust_vals = np.concatenate((gust_vals, gust_herb.gust.values))
            except:
                gust_vals = np.concatenate((gust_vals, np.zeros((len(grid_names[state]),))))

        return gust_vals, dswrf_vals, v_wind_vals, u_wind_vals, precip_vals, rh_vals, temp_vals

    def make_lat_lon_name_arr(self, df=pd.DataFrame()):
        """

        :param df: the dataframe of the station data
        :return: a dictionary, key=grid_county_name , val = tuple of avglon, avglat
        """
        names = {}
        lon_lats = {}
        for i in range(len(df)):
            state_fips = str(df['stateFips'][i])
            county_fips = str(df['FIPS'][i])
            grid_county_name = str(df['FIPS'][i]) + '_' + str(df['countyGridIndex'][i])
            avg_lon = (float(df['lon (urcrnr)'][i]) + float(df['lon (llcrnr)'][i])) / 2
            avg_lat = (float(df['lat (urcrnr)'][i]) + float(df['lat (llcrnr)'][i])) / 2

            if state_fips not in names:
                names[state_fips] = []
                lon_lats[state_fips] = []

            names[state_fips].append(grid_county_name)
            lon_lats[state_fips].append((avg_lon, avg_lat))

        return names, lon_lats

    def write_dict_row(self, row=[], state_abbrev=''):
        """

        :param row: a single row from a dictionary, containing all the necessary information
                    to find the file
        :return: nadda
        """
        # find the right file that we needa write to
        # write path + daily_data + year + state_abbrev + countyFips + monthly file
        # <HRRR_<state_fips>_<state_abbrev>_<year>-<month>>.csv
        year = row[0]
        state_name = row[5]
        month = row[1]
        county_fips = row[7]
        state_fips = county_fips[0:2]
        output_directory = self.write_path + "daily_data/" + year + '/' + state_abbrev + '/'
        # if it does not already exist, make it
        Path(output_directory).mkdir(parents=True, exist_ok=True)
        output_file_path = output_directory + "HRRR_" + state_fips + '_' + state_abbrev + '_' + year \
                           + '-' + month + '.csv'
        # if the file does not already exist, write the appropriate header out to it
        if not os.path.exists(output_file_path):
            # make the file
            f = open(output_file_path, mode='wt')
            f.write('Year,Month,Day,Hour,Daily/Monthly,State,County,FIPS Code,Grid Index,' +
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

    def reopen_files(self):
        csvFiles = []
        directory = self.write_path + "daily_data/"
        for year_folders in sorted(os.listdir(directory)):
            year_path = directory + year_folders + '/'
            if not os.path.isdir(year_path):
                continue
            for state in sorted(os.listdir(year_path)):
                state_path = year_path + state + '/'
                for file in sorted(os.listdir(state_path)):
                    full_path = state_path + file
                    csvFiles.append(full_path)

        return csvFiles

    def wind_speed_vpd(self, df=pd.DataFrame()):
        """
               turn the columns for U comp and V comp of wind
               into numpy ndarrays, then feed into the get_wind_speed
               functions. delete the U comp and V comp columns, make a new
               column "Wind Speed(m s**-1)" and put the return value as the
               values of the column

               turn the columns for temperature and humidity into numpy
               ndarrays, then feed into get_vpd function, make a new
               column "Vapor Pressure Deficit (kPa)" and put the return
               value as the values of the column
        """
        u_comp_wind = df['U Component of Wind (m s**-1)']
        u_comp_wind.replace(-12345.678, np.nan, inplace=True)
        u_comp_wind.ffill(inplace=True)
        u_comp_wind.bfill(inplace=True)
        u_comp_wind = u_comp_wind.to_numpy().astype(dtype=float)
        df['U Component of Wind (m s**-1)'] = u_comp_wind

        v_comp_wind = df['V Component of Wind (m s**-1)']
        v_comp_wind.replace(-12345.678, np.nan, inplace=True)
        v_comp_wind.ffill(inplace=True)
        v_comp_wind.bfill(inplace=True)
        v_comp_wind = v_comp_wind.to_numpy().astype(dtype=float)
        df['V Component of Wind (m s**-1)'] = v_comp_wind

        temp = df['Avg Temperature (K)']
        temp.replace(-12345.678, np.nan, inplace=True)
        temp.ffill(inplace=True)
        temp.bfill(inplace=True)
        temp = temp.to_numpy().astype(dtype=float)
        df['Avg Temperature (K)'] = temp

        relh = df['Relative Humidity (%)']
        relh.replace(-12345.678, np.nan, inplace=True)
        relh.ffill(inplace=True)
        relh.bfill(inplace=True)
        relh = relh.to_numpy().astype(dtype=float)
        df['Relative Humidity (%)'] = relh

        wind_speed = self.get_wind_speed(u_comp_wind, v_comp_wind)
        df['Wind Speed (m s**-1)'] = wind_speed

        vpd = self.get_VPD(temp, relh)
        df['Vapor Pressure Deficit (kPa)'] = vpd

        return df

    def get_wind_speed(self, U: np.ndarray, V: np.ndarray) -> np.ndarray:
        '''
        Calculate the average wind speed
        :param U: array_like, U component of wind in 1000 hPa (i.e., layer 32 in HRRR)
        :param V: array_like, V component of wind in 1000 hPa (i.e., layer 32 in HRRR)
        :return:
        '''
        return np.sqrt(np.power(U, 2) + np.power(V, 2))

    def get_VPD(self, T: np.ndarray, RH: np.ndarray) -> np.ndarray:
        '''
        Calculate the Vapor Pressure Deficit (VPD), unit: kPa
        :param T: array_like, temperate (unit: K) at 2 metre (layer 54 in HRRR)
        :param RH: array_like, relative humidity at 2 metre (i.e., layer 58 in HRRR)
        :return:
        '''

        # covert the temperature in Kelvin to the temperature in Celsius
        T = T - 273.15
        E = 7.5 * T / (237.3 + T)
        VP_sat = 610.7 * np.power(10, E) / 1000
        VP_air = VP_sat * RH / 100
        VPD = VP_sat - VP_air

        return VPD

    def dailyAvgMinMax(self, df=pd.DataFrame()):
        df.drop(columns='Hour', inplace=True)
        df.sort_values(['Day'], inplace=True)
        dftoreturn = pd.DataFrame(columns=list(df.columns))
        # dftoreturn.drop(columns='Hour', inplace=True)
        # print(df)
        for col in dftoreturn:
            if col == 'Avg Temperature (K)':
                # need to insert new columns at index of temperature
                index = dftoreturn.columns.get_loc(col)
                dftoreturn.insert(index + 1, "Max Temperature (K)", value=None, allow_duplicates=True)
                dftoreturn.insert(index + 1, "Min Temperature (K)", value=None, allow_duplicates=True)
        # print(list(dftoreturn.columns))
        ##########################################
        # look for a more efficient way to do this
        ##########################################
        grouped_counties = df.groupby(df['County'])
        for county in grouped_counties:
            countydf = pd.DataFrame(county[1])
            grouped_gridindex = countydf.groupby(df['Grid Index'])

            # for each station df
            for gridgroupdf in grouped_gridindex:
                # print(gridgroupdf[1])
                gridindexDf = pd.DataFrame(gridgroupdf[1])
                # print(gridindexDf)
                grouped_day = gridindexDf.groupby(gridindexDf.Day)

                # for each day in each station
                for group in grouped_day:
                    try:
                        current_day = group[0]
                        station_dayDf = pd.DataFrame(group[1])
                    except:
                        print("Did not grab grouped_day.get_group(%s)" % (group))
                        continue
                    avgedRow = []
                    # print(station_dayDf)

                    for col in station_dayDf:
                        if col.rfind('Year') != -1 or col.rfind('Month') != -1 or col.rfind(
                                'Day') != -1 or col.rfind(
                            'State') != -1 or col.rfind('County') != -1 or col.rfind(
                            'Grid Index') != -1 or col.rfind(
                            'FIPS Code') != -1 or col.rfind('Lat (') != -1 or col.rfind('Lon (') != -1:
                            avgedRow.append(station_dayDf[col].iloc[0])
                            # avgedRow.concat()
                            # pd.concat([avgedRow, station_dayDf[col].iloc[0]], ignore_index=True)
                        elif col.rfind('Precipitation') != -1 or col.rfind('radiation') != -1:
                            avgedRow.append(station_dayDf[col].sum())
                        else:
                            # get the average of the row and append it to the avg list
                            try:
                                avgedRow.append(station_dayDf[col].mean())
                            except:
                                avgedRow.append('NaN')
                            # if the column is temperature, find the min, max, then append
                            if col == 'Avg Temperature (K)':
                                avgedRow.append(station_dayDf[col].min())
                                avgedRow.append(station_dayDf[col].max())

                    # now that all the columns have been collected, we need to append the row to the dftoreturn
                    dftoreturn.loc[len(dftoreturn)] = avgedRow
        return dftoreturn

    def monthlyAvgs(self, df=pd.DataFrame()):
        monthlyavgs = []
        # for each column in df (not Year, Month, Day, State, County, GridIndex, FIPS Code, Lat, Lon(-180 to 180)), find avg and append to bottom
        # for Year, Month, State, County, write the first Index, for day write 'Monthly Average'

        # writeavgflag = False
        for col in df:
            # if col.rfind('Maximum/Composite') != -1:
            #     writeavgflag = True

            # if not writeavgflag:
            #     # do sum
            if col.rfind('Daily/Monthly') != -1:
                monthlyavgs.append("Monthly")
            elif col.rfind('Year') != -1 or col.rfind('Month') != -1 or col.rfind('State') != -1:
                monthlyavgs.append(df[col].iloc[0])
            elif (col.rfind('Lat (') != -1 or col.rfind('Lon (') != -1) and col.rfind('elative') == -1:
                monthlyavgs.append('N/A')
            elif col.rfind('Grid Index') != -1 or col.rfind('Day') != -1 or col.rfind('County') != -1 or \
                    col.rfind('FIPS') != -1:
                monthlyavgs.append('N/A')
            elif col.rfind('recipitation') != -1 or col.rfind('radiation') != -1:
                df_new = df.sort_values(by=['Day', 'Grid Index'])
                last_day = df_new['Day'][len(df_new) - 1]
                first_day = df_new['Day'][0]
                total_grid_indexes = df_new['Grid Index'][len(df_new) - 1] + 1
                sum = df[col].sum()
                val = sum / ((int(last_day) - int(first_day) + 1) * total_grid_indexes)
                monthlyavgs.append(val)
            else:
                # find the average of the column and append to the monthlyavg arr
                try:
                    monthlyavgs.append(df[col].mean())
                    # monthlyavgs.append(df[col].values.mean())
                except:
                    monthlyavgs.append('NaN')

        # df = df.append(pd.Series(monthlyavgs, index=df.columns[:len(monthlyavgs)]), ignore_index=True)
        # df = pd.concat([df, pd.Series(monthlyavgs)], ignore_index=True, axis=0, join='outer')
        df.loc[len(df.index)] = monthlyavgs
        # print(df)
        for col in df:
            if col.rfind('elative Hum') != -1:
                df[col] = df[col].round(decimals=1)
                continue
            try:
                df[col] = df[col].round(decimals=3)
            except:
                continue
        df = df[['Year', 'Month', 'Day', 'Daily/Monthly', 'State', 'County', 'FIPS Code', 'Grid Index',
                 'Lat (llcrnr)', 'Lon (llcrnr)', 'Lat (urcrnr)', 'Lon (urcrnr)',
                 'Avg Temperature (K)', 'Max Temperature (K)', 'Min Temperature (K)', 'Precipitation (kg m**-2)',
                 'Relative Humidity (%)', 'Wind Gust (m s**-1)', 'Wind Speed (m s**-1)',
                 'U Component of Wind (m s**-1)', 'V Component of Wind (m s**-1)',
                 'Downward Shortwave Radiation Flux (W m**-2)', 'Vapor Pressure Deficit (kPa)']]

        return df

    def final_sendoff(self, df=pd.DataFrame(), fullfilepath=''):
        fullpathsep = fullfilepath.split('/')
        newfilepath = ''
        state_abbrev = ''
        year = ''
        for i in range(len(fullpathsep)):
            if fullpathsep[i].rfind('daily_data') != -1:
                state_abbrev = fullpathsep[i + 2]
                year = fullpathsep[i + 1]
                break
            newfilepath += fullpathsep[i] + '/'
        newfilepath += "monthly_data/" + year + '/' + state_abbrev + '/'
        Path(newfilepath).mkdir(parents=True, exist_ok=True)
        write_path = newfilepath + fullpathsep[len(fullpathsep) - 1]
        df.to_csv(write_path, index=False)
        return


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*More than one.*")
    warnings.filterwarnings("ignore", message=".*This pattern is interpreted as a regular expression,.*")
    p = PreprocessWRF()
    p.main()
