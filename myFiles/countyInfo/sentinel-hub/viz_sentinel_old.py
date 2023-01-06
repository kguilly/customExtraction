import argparse

import h3
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt

import geopandas as gpd
import json
import geo_utils

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--client_id', type=str, default="7b4a1dab-5812-4a59-8dcb-288c6ff85f4d",
                    help='client id for sentinel hub')
parser.add_argument('--client_secret', type=str, default="p%DXOx8I(JPf@-&KK*}M_1}n:zbSv0X7BeTrEyJ/",
                    help='client secret for sentinel hub')

args = parser.parse_args()


def get_token():
    """
        Request token from sentinel hub
        :return: token
    """

    # set up credentials
    client = BackendApplicationClient(client_id=args.client_id)
    oauth = OAuth2Session(client=client)

    # get an authentication token
    token = oauth.fetch_token(token_url='https://services.sentinel-hub.com/oauth/token',
                              client_id=args.client_id, client_secret=args.client_secret)
    return oauth, token


def viz_moisture():
    oauth, token = get_token()

    # moisture index evalscript
    evalscript = """
    //VERSION=3
    let index = (B8A - B11)/(B8A + B11);

    let val = colorBlend(index, [-0.8, -0.24, -0.032, 0.032, 0.24, 0.8], [[0.5,0,0], [1,0,0], [1,1,0], [0,1,1], [0,0,1], [0,0,0.5]]);
    val.push(dataMask);
    return val;

    """

    json_request = {
        "input": {
            "bounds": {
                "geometry": build_geometry()
                # "geometry": build_geometry_h3()
            },
            "data": [
                {
                    "dataFilter": {
                        "timeRange": {
                            "from": "2022-11-19T00:00:00Z",
                            "to": "2022-12-19T23:59:59Z"
                        }
                    },
                    "type": "sentinel-2-l2a"
                }
            ]
        },
        "output": {
            # "width": 512,
            # "height": 657.648,
            "responses": [
                {
                    "identifier": "default",
                    "format": {
                        "type": "image/png"
                    }
                }
            ],
            "delivery": {
                "s3": {}
            }
        },
        "evalscript": evalscript
    }

    # Set the request url and headers
    url_request = 'https://services.sentinel-hub.com/api/v1/process'
    headers_request = {
        "Authorization": "Bearer %s" % token['access_token']
    }

    print(json_request)

    # Send the request
    response = oauth.request(
        "POST", url_request, headers=headers_request, json=json_request
    )

    # read the image as numpy array
    image_arr = np.array(Image.open(io.BytesIO(response.content)))

    # plot the image for visualization
    plt.figure(figsize=(25, 25))
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(image_arr)
    plt.show()


def build_geometry(state='01', county='001'):
    # load the json file with
    geoData = gpd.read_file('./input/US-counties.geojson')

    # Load the json file with county coordinates from remote server
    # geoData = gpd.read_file(
    #     'https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/US-counties.geojson')

    # Make sure the "id" column is an integer
    geoData.id = geoData.id.astype(str).astype(int)

    # Remove Alaska, Hawaii and Puerto Rico.
    stateToRemove = ['02', '15', '72']
    geoData = geoData[~geoData.STATE.isin(stateToRemove)]

    # Filter a specific county by state ANSI and county ANSI
    county = geoData[(geoData.STATE == state) & (geoData.COUNTY == county)]

    # Get geometry info
    geometry = json.loads(county.to_json())["features"][0]['geometry']

    return geometry


def build_geometry_h3(state='01', county='001'):
    # load the json file with
    geoData = gpd.read_file('./input/US-counties.geojson')

    # Load the json file with county coordinates from remote server
    # geoData = gpd.read_file(
    #     'https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/US-counties.geojson')

    # Make sure the "id" column is an integer
    geoData.id = geoData.id.astype(str).astype(int)

    # Remove Alaska, Hawaii and Puerto Rico.
    stateToRemove = ['02', '15', '72']
    geoData = geoData[~geoData.STATE.isin(stateToRemove)]

    # Filter a specific county by state ANSI and county ANSI
    county = geoData[(geoData.STATE == state) & (geoData.COUNTY == county)]

    # latitude and longitude
    centroid = county.centroid
    lat, lon = centroid.x[0], centroid.y[0]

    # Get geometry info
    # geometry = json.loads(county.to_json())["features"][0]['geometry']

    # index the point in the H3 hexagon of given index resolution
    h = h3.geo_to_h3(lat=lat, lng=lon, resolution=6)
    h_geom = {"type": "Polygon",
              "coordinates": [h3.h3_to_geo_boundary(h=h, geo_json=True)]
              }

    return h_geom


def build_geometry_grids(state='01', county='001'):
    geoData = gpd.read_file(
        'https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/US-counties.geojson')

    # Make sure the "id" column is an integer
    geoData.id = geoData.id.astype(str).astype(int)

    # Remove Alaska, Hawaii and Puerto Rico.
    stateToRemove = ['02', '15', '72']
    geoData = geoData[~geoData.STATE.isin(stateToRemove)]

    county = geoData[(geoData.STATE == state) & (geoData.COUNTY == county)]

    geom = county.geometry[0]

    grid = geo_utils.partition(geom, 0.08)

    geometry = gpd.GeoSeries(grid)
    geometry_json = geometry.to_json()
    geometry_json = json.loads(geometry_json)

    geometry_girds = geometry_json["features"]
    return geometry_girds


if __name__ == '__main__':
    viz_moisture()
