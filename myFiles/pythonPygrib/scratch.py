import shapely
from shapely.geometry import Polygon
from shapely.prepared import prep
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
state_id = "22"
county_id = "007"

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

# Load geometry information for US counties
geoData = gpd.read_file('/home/kaleb/Documents/GitHub/customExtraction/myFiles/countyInfo/sentinel-hub/input/US-counties.geojson')

# Make sure the "id" column is an integer
geoData.id = geoData.id.astype(str).astype(int)

# Remove Alaska, Hawaii and Puerto Rico.
stateToRemove = ['02', '15', '72']
geoData = geoData[~geoData.STATE.isin(stateToRemove)]

county_data = geoData[(geoData.STATE == state_id) & (geoData.COUNTY == county_id)]

# partition a county into 10km * 10km grid
geom = county_data.geometry.iloc[0]
grid = partition(geom, 0.08)

# get boundary info of grids
boundary_array = []
for i in range(len(grid)):
    lons, lats = grid[i].exterior.coords.xy
    lats = lats.tolist()
    lons = lons.tolist()
    array = np.array([lats[0], lons[0], lats[2], lons[2]])
    boundary_array.append(array)

print (boundary_array)
