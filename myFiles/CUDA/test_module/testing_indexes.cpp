#include "shared_objs.h"
#include "iostream"
#include "decompress_cuda.h"

int main(){
    station_t st_1;
    station_t st_2;

    station_t* stationArr = new station_t[2];
    

    int numstations = 2;

    double * lats = new double[4]{32.3, 33.4, 35.6, 36.7};
    double * lons = new double[4]{44.1, 45.2, 46.3, 47.4};

    int numberOfPoints = 4;

    st_1.latll = 32.1;
    st_1.latur = 32.2;
    st_1.lonll = 44.0;
    st_1.lonur = 44.2;

    st_2.latll = 36.1;
    st_2.latur = 37.2;
    st_2.lonll = 46.0;
    st_2.lonur = 47.2;

    stationArr[0] = st_1;
    stationArr[1] = st_2;

    index_extraction(stationArr, lats, lons, numstations, numberOfPoints);

    for (int i=0; i<numstations; i++){
        station_t *station = &stationArr[i];
        std::cout << "Station " << i << ":" << station->closestPoint << std::endl;
    }

    delete [] stationArr;
    delete [] lats;
    delete [] lons;
}