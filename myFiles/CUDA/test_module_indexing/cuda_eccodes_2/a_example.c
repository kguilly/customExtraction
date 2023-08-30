
int main(){
    int ret = 0;
    grib_index* index = grib_index_new(0, "shortName,level,step", &ret);
    return 0;
}