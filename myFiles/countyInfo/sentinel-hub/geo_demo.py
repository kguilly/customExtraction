import geopandas as gpd
import json
import h3
from shapely.geometry import Polygon
import numpy as np


def geometry_demo():
    # Load the json file with county coordinates
    geoData = gpd.read_file(
        'https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/US-counties.geojson')

    # Make sure the "id" column is an integer
    geoData.id = geoData.id.astype(str).astype(int)

    # Remove Alaska, Hawaii and Puerto Rico.
    stateToRemove = ['02', '15', '72']
    geoData = geoData[~geoData.STATE.isin(stateToRemove)]

    county = geoData[(geoData.STATE == '01') & (geoData.COUNTY == '001')]

    # latitude and longitude
    centroid = county.centroid
    lat, lon = centroid.x[0], centroid.y[0]

    # convert to json
    geo_json = county.to_json()
    geo_json = json.loads(geo_json)["features"][0]['geometry']

    print(geo_json)


def bound():
    # Load the json file with county coordinates
    geoData = gpd.read_file(
        'https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/US-counties.geojson')

    # Make sure the "id" column is an integer
    geoData.id = geoData.id.astype(str).astype(int)

    # Remove Alaska, Hawaii and Puerto Rico.
    stateToRemove = ['02', '15', '72']
    geoData = geoData[~geoData.STATE.isin(stateToRemove)]

    county = geoData[(geoData.STATE == '01') & (geoData.COUNTY == '001')]

    xmin, ymin, xmax, ymax = county.geometry[0].bounds

    length = xmax - xmin
    wide = ymax - ymin

    cols = list(np.arange(xmin, xmax + wide, wide))
    rows = list(np.arange(ymin, ymax + length, length))

    polygons = []
    for x in cols[:-1]:
        for y in rows[:-1]:
            polygons.append(Polygon([(x, y), (x + wide, y), (x + wide, y + length), (x, y + length)]))

    grid = gpd.GeoDataFrame({'geometry': polygons})
    print(grid)


if __name__ == '__main__':
    geometry_demo()
    # bound()