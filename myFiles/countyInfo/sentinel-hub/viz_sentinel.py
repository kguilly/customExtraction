import argparse
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt

import geopandas as gpd
import json

from datetime import date, timedelta

import os

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


# function to acquire moisture imagery of county
# state should be a two digit string
# county should be a three digit string
# start should be a string in the format "YYYY-MM-DD"
# end should be a string in the format "YYYY-MM-DD"
def viz_moisture(state, county, start, end, save_path):
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
                "geometry": build_geometry(state, county)
            },
            "data": [
                {
                    "dataFilter": {
                        "timeRange": {
                            "from": str(start + "T00:00:00Z"),
                            "to": str(end + "T23:59:59Z")
                        }
                    },
                    "type": "sentinel-2-l1c"
                }
            ]
        },
        "output": {
            "width": 1000,
            "height": 1000,
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

    # Send the request
    response = oauth.request(
        "POST", url_request, headers=headers_request, json=json_request
    )

    # read the image as numpy array
    image_arr = np.array(Image.open(io.BytesIO(response.content)))

    # plot the image for visualization
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(image_arr)
    plt.savefig(save_path)
    plt.close()


# function to acquire agriculture imagery of county
# state should be a two digit string
# county should be a three digit string
# start should be a string in the format "YYYY-MM-DD"
# end should be a string in the format "YYYY-MM-DD"
def viz_agriculture(state, county, start, end, save_path):
    oauth, token = get_token()

    # agriculture evalscript
    evalscript = """
    //VERSION=3
    let minVal = 0.0;
    let maxVal = 0.4;

    let viz = new HighlightCompressVisualizer(minVal, maxVal);

    function setup() {
      return {
        input: ["B02", "B08", "B11", "dataMask"],
        output: { bands: 4 }
      };
    }

    function evaluatePixel(samples) {
        let val = [samples.B11, samples.B08, samples.B02, samples.dataMask];
        return viz.processList(val);
    }

    """

    json_request = {
        "input": {
            "bounds": {
                "geometry": build_geometry(state, county)
            },
            "data": [
                {
                    "dataFilter": {
                        "timeRange": {
                            "from": str(start + "T00:00:00Z"),
                            "to": str(end + "T23:59:59Z")
                        }
                    },
                    "type": "sentinel-2-l1c"
                }
            ]
        },
        "output": {
            "width": 1000,
            "height": 1000,
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

    # Send the request
    response = oauth.request(
        "POST", url_request, headers=headers_request, json=json_request
    )

    # read the image as numpy array
    image_arr = np.array(Image.open(io.BytesIO(response.content)))

    # plot the image for visualization
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(image_arr)
    plt.savefig(save_path)
    plt.close()


# function to acquire vegetation index imagery of county
# state should be a two digit string
# county should be a three digit string
# start should be a string in the format "YYYY-MM-DD"
# end should be a string in the format "YYYY-MM-DD"
def viz_vegetation(state, county, start, end, save_path):
    oauth, token = get_token()

    # vegetation index evalscript
    evalscript = """
    //VERSION=3

    let viz = ColorMapVisualizer.createDefaultColorMap();

    function evaluatePixel(samples) {
        let val = index(samples.B08, samples.B04);
        val = viz.process(val);
        val.push(samples.dataMask);
        return val;
    }

    function setup() {
      return {
        input: [{
          bands: [
            "B04",
            "B08",
            "dataMask"
          ]
        }],
        output: {
          bands: 4
        }
      }
    }

    """

    json_request = {
        "input": {
            "bounds": {
                "geometry": build_geometry(state, county)
            },
            "data": [
                {
                    "dataFilter": {
                        "timeRange": {
                            "from": str(start + "T00:00:00Z"),
                            "to": str(end + "T23:59:59Z")
                        }
                    },
                    "type": "sentinel-2-l1c"
                }
            ]
        },
        "output": {
            "width": 1000,
            "height": 1000,
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

    # Send the request
    response = oauth.request(
        "POST", url_request, headers=headers_request, json=json_request
    )

    # read the image as numpy array
    image_arr = np.array(Image.open(io.BytesIO(response.content)))

    # plot the image for visualization
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(image_arr)
    plt.savefig(save_path)
    plt.close()


def build_geometry(state, county):
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


# returns the center point of the county
def center(state='01', county='001'):
    return build_geometry(state, county).centroid


# function to get a date range
def get_daterange(start_date, end_date, step):
    while start_date <= end_date:
        yield start_date
        start_date += step


# get the moisture, vegetation, and agriculture imagery
# creates a folder of the FIPS code then stores imagery there per day
# for a certain county on a series of days
# state should be a two digit string
# county should be a three digit string
# start should be a string in the format "YYYY-MM-DD"
# end should be a string in the format "YYYY-MM-DD"
def get_imagery(state, county, start, end):
    # get target path
    cwd = os.getcwd()
    target_dir = cwd + '/sat_imagery'

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    target_dir = target_dir + '/' + state + county

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    # iterate through dates to get imagery for each day
    for d in get_daterange(date(int(start[0:4]), int(start[5:7]), int(start[8:10])),
                           date(int(end[0:4]), int(end[5:7]), int(end[8:10])), timedelta(days=1)):
        viz_moisture(state, county, str(d), str(d), target_dir + '/' + str(d) + '_moisture.png')
        viz_vegetation(state, county, str(d), str(d), target_dir + '/' + str(d) + '_vegetation.png')
        viz_agriculture(state, county, str(d), str(d), target_dir + '/' + str(d) + '_agriculture.png')


if __name__ == '__main__':
    get_imagery('05', '085', "2022-12-04", "2022-12-04")
