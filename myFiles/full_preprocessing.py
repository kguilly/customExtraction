import pandas as pd
from herbie import Herbie
import numpy as np
from datetime import date, timedelta
import sys
import os
import geo_grid_recent as gg



class PreprocessWRF():
    def __init__(self):
        self.write_path = "/home/kaleb/Desktop/full_preprocessing_output/"

        self.begin_hour = 0
        self.end_hour = 23
        self.begin_date = "20200101" #format as "yyyymmdd"
        self.end_date = "20200102"
        self.county_df = pd.DataFrame()
        self.passedFips = []

    def main(self):
        self.handle_args()
        every_county_df = pd.read_csv("./WRFoutput/wrfOutput.csv")
        param_dict_arr = self.separate_by_state(df=every_county_df)

    def handle_args(self):
        if len(sys.argv) > 1:
            fips_flag = False
            fips_index = 0

            for i in range(1, len(sys.argv)):

                if sys.argv[i] == "--begin_date":
                    self.begin_date = sys.argv[i+1]

                elif sys.argv[i] == "--end_date":
                    self.end_date = sys.argv[i+1]

                elif sys.argv[i] == "--fips":
                    fips_flag = True
                    fips_index = i+1

            if fips_flag:
                self.passedFips = []
                for i in range(fips_index+1, len(sys.argv)):
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

if __name__ == "__main__":
    p = PreprocessWRF()
    p.main()