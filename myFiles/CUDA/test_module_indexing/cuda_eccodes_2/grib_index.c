#include "structs.h"
#include "funcs.h"


static const char* mars_keys =
    "mars.date,mars.time,mars.expver,mars.stream,mars.class,mars.type,"
    "mars.step,mars.param,mars.levtype,mars.levelist,mars.number,mars.iteration,"
    "mars.domain,mars.fcmonth,mars.fcperiod,mars.hdate,mars.method,"
    "mars.model,mars.origin,mars.quantile,mars.range,mars.refdate,mars.direction,mars.frequency";
    
grib_index* grib_index_new(grib_context* c, const char* key, int* err)
{
    grib_index* index;
    grib_index_key* keys = NULL;
    char* q;
    int type;
    char* p;

    if (!strcmp(key, "mars"))
        return grib_index_new(c, mars_keys, err);

    p = grib_context_strdup(c, key);
    q = p;

    *err = 0;
    if (!c)
        c = grib_context_get_default();

    index = (grib_index*)grib_context_malloc_clear(c, sizeof(grib_index));
    if (!index) {
        grib_context_log(c, GRIB_LOG_ERROR, "unable to create index");
        *err = GRIB_OUT_OF_MEMORY;
        return NULL;
    }
    index->context = c;
    index->product_kind = PRODUCT_GRIB;
    index->unpack_bufr = 0;

    while ((key = get_key(&p, &type)) != NULL) {
        keys = grib_index_new_key(c, keys, key, type, err);
        if (*err)
            return NULL;
    }
    index->keys   = keys;
    index->fields = (grib_field_tree*)grib_context_malloc_clear(c,
                                                                sizeof(grib_field_tree));
    if (!index->fields) {
        *err = GRIB_OUT_OF_MEMORY;
        return NULL;
    }

    grib_context_free(c, q);
    return index;
}