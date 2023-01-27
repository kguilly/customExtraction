import shapely
from shapely.geometry import Polygon
from shapely.prepared import prep
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import json
import argparse
import os
import csv


def grid_bounds(geom, delta=0.08):
    """
        Return geometrical grids with equal size
    :param geom:
    :param delta: The edge length of each grid
        delta = 0.01 results in a 1km x 1km grid (approximately)
        delta = 0.03 results in a 3km x 3km grid (approximately)
        delta = 0.05 results in a 5km x 5km grid (approximately)
        delta = 0.08 results in a 10km x 10km grid (approximately)

        Use https://www.nhc.noaa.gov/gccalc.shtml to approximate the size of grid
    :return:
    """

    minx, miny, maxx, maxy = geom.bounds
    nx = int((maxx - minx) / delta)
    ny = int((maxy - miny) / delta)
    gx, gy = np.linspace(minx, maxx, nx), np.linspace(miny, maxy, ny)
    grid = []
    for i in range(len(gx) - 1):
        for j in range(len(gy) - 1):
            poly_ij = Polygon([[gx[i], gy[j]], [gx[i], gy[j + 1]], [gx[i + 1], gy[j + 1]], [gx[i + 1], gy[j]]])
            grid.append(poly_ij)
    return grid


def partition(geom, delta):
    """
        Partition a geometry map of a county to multiple grids
    :param geom:
    :param delta:
    :return:
    """

    prepared_geom = prep(geom)
    grid = list(filter(prepared_geom.intersects, grid_bounds(geom, delta)))
    return grid


def demo():
    geom = Polygon([[0, 0], [0, 2], [1.5, 1], [0.5, -0.5], [0, 0]])
    grid = partition(geom, 0.1)

    fig, ax = plt.subplots(figsize=(15, 15))
    gpd.GeoSeries(grid).boundary.plot(ax=ax)
    gpd.GeoSeries([geom]).boundary.plot(ax=ax, color="red")
    plt.show()


def county(fips=None):

    dict = {} # hold the fips code info
    #####################3
    # parse the arguments that I passed]
    #####################3
    for fipscode in fips:

        if(type(fips) == str): # if only one argument was passed
            passedcountyfips = fips[2:6]
            passedstateFips = fips[0:2]
            fipscode = fips    
        else: # if more than one argument was passed
            passedcountyfips = fipscode[2:6]
            passedstateFips = fipscode[0:2]

        geoData = gpd.read_file(
            'https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/US-counties.geojson')

        # Make sure the "id" column is an integer
        geoData.id = geoData.id.astype(str).astype(int)

        # Remove Alaska, Hawaii and Puerto Rico.
        stateToRemove = ['02', '15', '72']
        geoData = geoData[~geoData.STATE.isin(stateToRemove)]

        county = geoData[(geoData.STATE == passedstateFips) & (geoData.COUNTY == passedcountyfips)]

        geom = county.geometry.iloc[0]
        print(county)


        grid = partition(geom, 0.08)

        geometry = gpd.GeoSeries(grid)
        geometry_json = geometry.to_json()
        geometry_json = json.loads(geometry_json)
        # print(geometry_json)

        # fig, ax = plt.subplots(figsize=(15, 15))
        # gpd.GeoSeries(grid).boundary.plot(ax=ax)
        # gpd.GeoSeries([geom]).boundary.plot(ax=ax, color="red")
        # plt.show()

        ##################################################
        # Make a dictionary:
        #      - county grid index : [countyGridIndex, fips, state, county, lat, lon]
        # need to find the center of each lat and lon
        ##################################################
        
        header = ['countyGridIndex', 'FIPS', 'stateFips', 'county', 'lat', 'lon']
        
        features = geometry_json['features'] 
        for i in features:
            grididx = i['id']
            bbox = i['bbox']
            avglat = (bbox[1] + bbox[3]) / 2
            avglon = (bbox[0] + bbox[2]) / 2
            stateFips = str(county.STATE.iloc[0])
            thisfips = stateFips + str(county.COUNTY.iloc[0])
            countyName = str(county.NAME.iloc[0]) + " " + county.LSAD.iloc[0]
            dict[str(grididx) + "_" +fipscode] = [str(grididx), thisfips, stateFips, countyName, avglat, avglon]

        if type(fips) == str:
            break
    # print(dict)
        

    # output the necessary information to a .csv file 
    directory = "./WRFoutput/"
    if not (os.path.isdir(directory)):
        os.mkdir(directory)
    csvDir = directory + "wrfOutput.csv"
    with open(csvDir, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for line in dict:
            writer.writerow(dict[str(line)])



if __name__ == '__main__':
    # demo()
    parser = argparse.ArgumentParser(description='enter the fips code')
    parser.add_argument("--fips", default='22055', nargs="+", help='The 5 digit fips code')
    args = parser.parse_args()
    fips=args.fips
    county(fips=fips)
