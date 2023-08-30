
extern "C" {
#include "cuda_eccodes.h"
#include "cuda_eccodes_prototypes.h"
#include "cuda_grib_api.h"
#include "cuda_grib_api_internal.h"
}

int main(){
    int ret = 0;
    grib_index* index = codes_index_new(0, "shortName,level,step", &ret);
    return 0;
}