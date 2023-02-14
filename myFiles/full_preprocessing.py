from herbie import Herbie
import numpy as np
from datetime import date, timedelta


class PreprocessWRF():
    def __init__(self):
        self.write_path = "/home/kaleb/Desktop/full_preprocessing_output/"
        self.repository_path = "/home/kaleb/Documents/GitHub/customExtraction/"

        self.begin_hour = 0
        self.end_hour = 23
        self.begin_date = date(2020, 1, 1)
        self.end_date = date(2020, 2, 1)

    def main(self):
        county_fips_df = self.handle_args()

    def handle_args(self):
        
        
if __name__ == "__main__":
    p = PreprocessWRF()
    p.main()