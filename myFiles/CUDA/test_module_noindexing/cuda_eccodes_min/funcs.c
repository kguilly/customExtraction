#include "header.h"


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>

void codes_assertion_failed(const char* message, const char* file, int line)
{
    /* Default behaviour is to abort
     * unless user has supplied his own assertion routine */
    if (assertion == NULL) {
        grib_context* c = grib_context_get_default();
        fprintf(stderr, "ecCodes assertion failed: `%s' in %s:%d\n", message, file, line);
        if (!c->no_abort) {
            abort();
        }
    }
    else {
        char buffer[10240];
        sprintf(buffer, "ecCodes assertion failed: `%s' in %s:%d", message, file, line);
        assertion(buffer);
    }
}

void codes_check(const char* call, const char* file, int line, int e, const char* msg)
{
    grib_check(call, file, line, e, msg);
}

grib_accessor* grib_find_accessor(const grib_handle* h, const char* name)
{
    grib_accessor* aret = NULL;
    Assert(h);
    if (h->product_kind == PRODUCT_GRIB) {
        aret = _grib_find_accessor(h, name); /* ECC-144: Performance */
    }
    else {
        char attribute_name[512] = {0,};
        grib_accessor* a = NULL;

        char* accessor_name = grib_split_name_attribute(h->context, name, attribute_name);

        a = _grib_find_accessor(h, accessor_name);

        if (*attribute_name == 0) {
            aret = a;
        }
        else if (a) {
            aret = grib_accessor_get_attribute(a, attribute_name);
            grib_context_free(h->context, accessor_name);
        }
    }
    return aret;
}

grib_accessors_list* grib_find_accessors_list(const grib_handle* ch, const char* name)
{
    char* str                  = NULL;
    grib_accessors_list* al    = NULL;
    codes_condition* condition = NULL;
    grib_accessor* a           = NULL;
    grib_handle* h             = (grib_handle*)ch;

    if (name[0] == '/') {
        condition = (codes_condition*)grib_context_malloc_clear(h->context, sizeof(codes_condition));
        str       = get_condition(name, condition);
        if (str) {
            al = search_by_condition(h, str, condition);
            grib_context_free(h->context, str);
            if (condition->left)
                grib_context_free(h->context, condition->left);
            if (condition->rightString)
                grib_context_free(h->context, condition->rightString);
        }
        grib_context_free(h->context, condition);
    }
    else if (name[0] == '#') {
        a = grib_find_accessor(h, name);
        if (a) {
            char* str2;
            int r;
            al   = (grib_accessors_list*)grib_context_malloc_clear(h->context, sizeof(grib_accessors_list));
            str2 = get_rank(h->context, name, &r);
            grib_accessors_list_push(al, a, r);
            grib_context_free(h->context, str2);
        }
    }
    else {
        a = grib_find_accessor(h, name);
        if (a) {
            al = (grib_accessors_list*)grib_context_malloc_clear(h->context, sizeof(grib_accessors_list));
            grib_find_same_and_push(al, a);
        }
    }

    return al;
}

grib_context* grib_context_get_default()
{
    GRIB_MUTEX_INIT_ONCE(&once, &init);
    GRIB_MUTEX_LOCK(&mutex_c);

    if (!default_grib_context.inited) {
        const char* write_on_fail                       = NULL;
        const char* large_constant_fields               = NULL;
        const char* no_abort                            = NULL;
        const char* debug                               = NULL;
        const char* gribex                              = NULL;
        const char* ieee_packing                        = NULL;
        const char* io_buffer_size                      = NULL;
        const char* log_stream                          = NULL;
        const char* no_big_group_split                  = NULL;
        const char* no_spd                              = NULL;
        const char* keep_matrix                         = NULL;
        const char* bufrdc_mode                         = NULL;
        const char* bufr_set_to_missing_if_out_of_range = NULL;
        const char* bufr_multi_element_constant_arrays  = NULL;
        const char* grib_data_quality_checks            = NULL;
        const char* file_pool_max_opened_files          = NULL;

#ifdef ENABLE_FLOATING_POINT_EXCEPTIONS
        feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);
#endif

        write_on_fail                       = codes_getenv("ECCODES_GRIB_WRITE_ON_FAIL");
        bufrdc_mode                         = getenv("ECCODES_BUFRDC_MODE_ON");
        bufr_set_to_missing_if_out_of_range = getenv("ECCODES_BUFR_SET_TO_MISSING_IF_OUT_OF_RANGE");
        bufr_multi_element_constant_arrays  = getenv("ECCODES_BUFR_MULTI_ELEMENT_CONSTANT_ARRAYS");
        grib_data_quality_checks            = getenv("ECCODES_GRIB_DATA_QUALITY_CHECKS");
        large_constant_fields               = codes_getenv("ECCODES_GRIB_LARGE_CONSTANT_FIELDS");
        no_abort                            = codes_getenv("ECCODES_NO_ABORT");
        debug                               = codes_getenv("ECCODES_DEBUG");
        gribex                              = codes_getenv("ECCODES_GRIBEX_MODE_ON");
        ieee_packing                        = codes_getenv("ECCODES_GRIB_IEEE_PACKING");
        io_buffer_size                      = codes_getenv("ECCODES_IO_BUFFER_SIZE");
        log_stream                          = codes_getenv("ECCODES_LOG_STREAM");
        no_big_group_split                  = codes_getenv("ECCODES_GRIB_NO_BIG_GROUP_SPLIT");
        no_spd                              = codes_getenv("ECCODES_GRIB_NO_SPD");
        keep_matrix                         = codes_getenv("ECCODES_GRIB_KEEP_MATRIX");
        file_pool_max_opened_files          = getenv("ECCODES_FILE_POOL_MAX_OPENED_FILES");

        /* On UNIX, when we read from a file we get exactly what is in the file on disk.
         * But on Windows a file can be opened in binary or text mode. In binary mode the system behaves exactly as in UNIX.
         */
#ifdef ECCODES_ON_WINDOWS
        _set_fmode(_O_BINARY);
#endif

        default_grib_context.inited = 1;
        default_grib_context.io_buffer_size = io_buffer_size ? atoi(io_buffer_size) : 0;
        default_grib_context.no_big_group_split = no_big_group_split ? atoi(no_big_group_split) : 0;
        default_grib_context.no_spd = no_spd ? atoi(no_spd) : 0;
        default_grib_context.keep_matrix = keep_matrix ? atoi(keep_matrix) : 1;
        default_grib_context.write_on_fail = write_on_fail ? atoi(write_on_fail) : 0;
        default_grib_context.no_abort = no_abort ? atoi(no_abort) : 0;
        default_grib_context.debug = debug ? atoi(debug) : 0;
        default_grib_context.gribex_mode_on = gribex ? atoi(gribex) : 0;
        default_grib_context.large_constant_fields = large_constant_fields ? atoi(large_constant_fields) : 0;
        default_grib_context.ieee_packing = ieee_packing ? atoi(ieee_packing) : 0;
        default_grib_context.grib_samples_path = codes_getenv("ECCODES_SAMPLES_PATH");
        default_grib_context.log_stream = stderr;
        if (!log_stream) {
            default_grib_context.log_stream = stderr;
        }
        else if (!strcmp(log_stream, "stderr")) {
            default_grib_context.log_stream = stderr;
        }
        else if (!strcmp(log_stream, "stdout")) {
            default_grib_context.log_stream = stdout;
        }

#ifdef ECCODES_SAMPLES_PATH
        if (!default_grib_context.grib_samples_path)
            default_grib_context.grib_samples_path = ECCODES_SAMPLES_PATH;
#endif

        default_grib_context.grib_definition_files_path = codes_getenv("ECCODES_DEFINITION_PATH");
#ifdef ECCODES_DEFINITION_PATH
        if (!default_grib_context.grib_definition_files_path) {
            default_grib_context.grib_definition_files_path = strdup(ECCODES_DEFINITION_PATH);
        }
        else {
            default_grib_context.grib_definition_files_path = strdup(default_grib_context.grib_definition_files_path);
        }
#endif

        /* GRIB-779: Special case for ECMWF testing. Not for external use! */
        /* Append the new path to our existing path */
        {
            const char* test_defs = codes_getenv("_ECCODES_ECMWF_TEST_DEFINITION_PATH");
            const char* test_samp = codes_getenv("_ECCODES_ECMWF_TEST_SAMPLES_PATH");
            if (test_defs) {
                char buffer[ECC_PATH_MAXLEN]= {0,};
                if (default_grib_context.grib_definition_files_path) {
                    strcpy(buffer, default_grib_context.grib_definition_files_path);
                    strcat(buffer, ":");
                }
                strcat(buffer, test_defs);
                free(default_grib_context.grib_definition_files_path);
                default_grib_context.grib_definition_files_path = strdup(buffer);
            }
            if (test_samp) {
                char buffer[ECC_PATH_MAXLEN]= {0,};
                if (default_grib_context.grib_samples_path) {
                    strcpy(buffer, default_grib_context.grib_samples_path);
                    strcat(buffer, ":");
                }
                strcat(buffer, test_samp);
                default_grib_context.grib_samples_path = strdup(buffer);
            }
        }

        /* Definitions path extra: Added at the head of (i.e. before) existing path */
        {
            const char* defs_extra = getenv("ECCODES_EXTRA_DEFINITION_PATH");
            if (defs_extra) {
                char buffer[ECC_PATH_MAXLEN]= {0,};
                ecc_snprintf(buffer, ECC_PATH_MAXLEN, "%s%c%s", defs_extra, ECC_PATH_DELIMITER_CHAR, default_grib_context.grib_definition_files_path);
                free(default_grib_context.grib_definition_files_path);
                default_grib_context.grib_definition_files_path = strdup(buffer);
            }
        }
#ifdef ECCODES_DEFINITION_PATH
        {
            /* ECC-1088 */
            if (strstr(default_grib_context.grib_definition_files_path, ECCODES_DEFINITION_PATH) == NULL) {
                char buffer[ECC_PATH_MAXLEN]= {0,};
                ecc_snprintf(buffer, ECC_PATH_MAXLEN, "%s%c%s", default_grib_context.grib_definition_files_path,
                             ECC_PATH_DELIMITER_CHAR, ECCODES_DEFINITION_PATH);
                free(default_grib_context.grib_definition_files_path);
                default_grib_context.grib_definition_files_path = strdup(buffer);
            }
        }
#endif

        /* Samples path extra: Added at the head of (i.e. before) existing path */
        {
            const char* samples_extra = getenv("ECCODES_EXTRA_SAMPLES_PATH");
            if (samples_extra) {
                char buffer[ECC_PATH_MAXLEN];
                ecc_snprintf(buffer, ECC_PATH_MAXLEN, "%s%c%s", samples_extra, ECC_PATH_DELIMITER_CHAR, default_grib_context.grib_samples_path);
                default_grib_context.grib_samples_path = strdup(buffer);
            }
        }
#ifdef ECCODES_SAMPLES_PATH
        {
            if (strstr(default_grib_context.grib_samples_path, ECCODES_SAMPLES_PATH) == NULL) {
                char buffer[ECC_PATH_MAXLEN];
                ecc_snprintf(buffer, ECC_PATH_MAXLEN, "%s%c%s", default_grib_context.grib_samples_path,
                             ECC_PATH_DELIMITER_CHAR, ECCODES_SAMPLES_PATH);
                default_grib_context.grib_samples_path = strdup(buffer);
            }
        }
#endif

        grib_context_log(&default_grib_context, GRIB_LOG_DEBUG, "Definitions path: %s",
                         default_grib_context.grib_definition_files_path);
        grib_context_log(&default_grib_context, GRIB_LOG_DEBUG, "Samples path:     %s",
                         default_grib_context.grib_samples_path);

        default_grib_context.keys_count = 0;
        default_grib_context.keys       = grib_hash_keys_new(&(default_grib_context), &(default_grib_context.keys_count));

        default_grib_context.concepts_index = grib_itrie_new(&(default_grib_context), &(default_grib_context.concepts_count));
        default_grib_context.hash_array_index = grib_itrie_new(&(default_grib_context), &(default_grib_context.hash_array_count));
        default_grib_context.def_files = grib_trie_new(&(default_grib_context));
        default_grib_context.lists = grib_trie_new(&(default_grib_context));
        default_grib_context.classes = grib_trie_new(&(default_grib_context));
        default_grib_context.bufrdc_mode = bufrdc_mode ? atoi(bufrdc_mode) : 0;
        default_grib_context.bufr_set_to_missing_if_out_of_range = bufr_set_to_missing_if_out_of_range ? atoi(bufr_set_to_missing_if_out_of_range) : 0;
        default_grib_context.bufr_multi_element_constant_arrays = bufr_multi_element_constant_arrays ? atoi(bufr_multi_element_constant_arrays) : 0;
        default_grib_context.grib_data_quality_checks = grib_data_quality_checks ? atoi(grib_data_quality_checks) : 0;
        default_grib_context.file_pool_max_opened_files = file_pool_max_opened_files ? atoi(file_pool_max_opened_files) : DEFAULT_FILE_POOL_MAX_OPENED_FILES;
    }

    GRIB_MUTEX_UNLOCK(&mutex_c);
    return &default_grib_context;
}

grib_handle* grib_handle_new_from_file(grib_context* c, FILE* f, int* error)
{
    return grib_new_from_file(c, f, 0, error);
}

grib_handle* grib_new_from_file(grib_context* c, FILE* f, int headers_only, int* error)
{
    grib_handle* h = 0;
    if (!f) {
        *error = GRIB_IO_PROBLEM;
        return NULL;
    }

    if (c == NULL)
        c = grib_context_get_default();

    if (c->multi_support_on)
        h = grib_handle_new_from_file_multi(c, f, error);
    else
        h = grib_handle_new_from_file_no_multi(c, f, headers_only, error);

    if (h && h->offset == 0)
        grib_context_set_handle_file_count(c, 1);

    if (h) {
        h->product_kind = PRODUCT_GRIB;
    }

    if (!c->no_fail_on_wrong_length && *error == GRIB_WRONG_LENGTH) {
        grib_handle_delete(h);
        h = NULL;
    }

    return h;
}

static grib_handle* grib_handle_new_from_file_no_multi(grib_context* c, FILE* f, int headers_only, int* error)
{
    void* data              = NULL;
    size_t olen             = 0;
    grib_handle* gl         = NULL;
    off_t gts_header_offset = 0;
    off_t offset = 0, end_msg_offset = 0;
    char *gts_header = 0, *save_gts_header = 0;
    int gtslen = 0;

    if (c == NULL)
        c = grib_context_get_default();

    gts_header_offset = grib_context_tell(c, f);
    data              = wmo_read_grib_from_file_malloc(f, headers_only, &olen, &offset, error);
    end_msg_offset    = grib_context_tell(c, f);

    if (*error != GRIB_SUCCESS) {
        if (data)
            grib_context_free(c, data);

        if (*error == GRIB_END_OF_FILE)
            *error = GRIB_SUCCESS;
        return NULL;
    }

    if (c->gts_header_on) {
        int g = 0;
        grib_context_seek(c, gts_header_offset, SEEK_SET, f);
        gtslen          = offset - gts_header_offset;
        gts_header      = (char*)grib_context_malloc(c, sizeof(unsigned char) * gtslen);
        save_gts_header = gts_header;
        grib_context_read(c, gts_header, gtslen, f);
        g = gtslen;
        while (gts_header != NULL && g != 0 && *gts_header != '\03') {
            /*printf("--------%d %X \n",gtslen,*gts_header);*/
            gts_header++;
            g--;
        }
        if (g > 8) {
            gts_header++;
            gtslen = g - 1;
        }
        else
            gts_header = save_gts_header;
        grib_context_seek(c, end_msg_offset, SEEK_SET, f);
    }

    if (headers_only) {
        gl = grib_handle_new_from_partial_message(c, data, olen);
    }
    else {
        gl = grib_handle_new_from_message(c, data, olen);
    }

    if (!gl) {
        *error = GRIB_DECODING_ERROR;
        grib_context_log(c, GRIB_LOG_ERROR, "grib_handle_new_from_file_no_multi: cannot create handle\n");
        grib_context_free(c, data);
        return NULL;
    }

    gl->offset           = offset;
    gl->buffer->property = GRIB_MY_BUFFER;

    grib_context_increment_handle_file_count(c);
    grib_context_increment_handle_total_count(c);

    if (c->gts_header_on && gtslen >= 8) {
        gl->gts_header = (char*)grib_context_malloc(c, sizeof(unsigned char) * gtslen);
        DebugAssert(gts_header);
        if (gts_header) memcpy(gl->gts_header, gts_header, gtslen);
        gl->gts_header_len = gtslen;
        grib_context_free(c, save_gts_header);
    }
    else {
        gl->gts_header = NULL;
    }

    return gl;
}

int codes_get_long(const grib_handle* h, const char* key, long* value)
{
    return grib_get_long(h, key, value);
}

void grib_context_set_handle_file_count(grib_context* c, int new_count)
{
    if (!c)
        c = grib_context_get_default();
    GRIB_MUTEX_INIT_ONCE(&once, &init);
    GRIB_MUTEX_LOCK(&mutex_c);
    c->handle_file_count = new_count;
    GRIB_MUTEX_UNLOCK(&mutex_c);
}

void grib_check(const char* call, const char* file, int line, int e, const char* msg)
{
    grib_context* c=grib_context_get_default();
    if (e) {
        if (file) {
            fprintf(stderr,"%s at line %d: %s failed: %s",
                file,line, call,grib_get_error_message(e));
            if (msg) fprintf(stderr," (%s)",msg);
            printf("\n");
        } else {
            grib_context_log(c,GRIB_LOG_ERROR,"%s",grib_get_error_message(e));
        }
        exit(e);
    }
}

int grib_get_data(const grib_handle* h, double* lats, double* lons, double* values)
{
    int err             = 0;
    grib_iterator* iter = NULL;
    double *lat, *lon, *val;

    iter = grib_iterator_new(h, 0, &err);
    if (!iter || err != GRIB_SUCCESS)
        return err;

    lat = lats;
    lon = lons;
    val = values;
    while (grib_iterator_next(iter, lat++, lon++, val++)) {}

    grib_iterator_delete(iter);

    return err;
}

int grib_get_long(const grib_handle* h, const char* name, long* val)
{
    size_t length           = 1;
    grib_accessor* a        = NULL;
    grib_accessors_list* al = NULL;
    int ret                 = 0;

    if (name[0] == '/') {
        al = grib_find_accessors_list(h, name);
        if (!al)
            return GRIB_NOT_FOUND;
        ret = grib_unpack_long(al->accessor, val, &length);
        grib_context_free(h->context, al);
    }
    else {
        a = grib_find_accessor(h, name);
        if (!a)
            return GRIB_NOT_FOUND;
        ret = grib_unpack_long(a, val, &length);
    }
    return ret;
}

int grib_iterator_next(grib_iterator* i, double* lat, double* lon, double* value)
{
    grib_iterator_class* c = i->cclass;
    while (c) {
        grib_iterator_class* s = c->super ? *(c->super) : NULL;
        if (c->next)
            return c->next(i, lat, lon, value);
        c = s;
    }
    Assert(0);
    return 0;
}

int grib_unpack_long(grib_accessor* a, long* v, size_t* len)
{
    grib_accessor_class* c = a->cclass;
    /*grib_context_log(a->context, GRIB_LOG_DEBUG, "(%s)%s is unpacking (long)",(a->parent->owner)?(a->parent->owner->name):"root", a->name ); */
    while (c) {
        if (c->unpack_long) {
            return c->unpack_long(a, v, len);
        }
        c = c->super ? *(c->super) : NULL;
    }
    DebugAssert(0);
    return 0;
}

static void default_buffer_free(const grib_context* c, void* p)
{
    free(p);
}

static void* default_buffer_malloc(const grib_context* c, size_t size)
{
    void* ret;
    ret = malloc(size);
    if (!ret) {
        grib_context_log(c, GRIB_LOG_FATAL, "default_buffer_malloc: error allocating %lu bytes", (unsigned long)size);
        Assert(0);
    }
    return ret;
}

static void* default_buffer_realloc(const grib_context* c, void* p, size_t size)
{
    void* ret;
    ret = realloc(p, size);
    if (!ret) {
        grib_context_log(c, GRIB_LOG_FATAL, "default_buffer_realloc: error allocating %lu bytes", (unsigned long)size);
        Assert(0);
    }
    return ret;
}

static void default_log(const grib_context* c, int level, const char* mess)
{
    if (!c)
        c = grib_context_get_default();
    if (level == GRIB_LOG_ERROR) {
        fprintf(c->log_stream, "ECCODES ERROR   :  %s\n", mess);
        /*Assert(1==0);*/
    }
    if (level == GRIB_LOG_FATAL)
        fprintf(c->log_stream, "ECCODES ERROR   :  %s\n", mess);
    if (level == GRIB_LOG_DEBUG && c->debug > 0)
        fprintf(c->log_stream, "ECCODES DEBUG   :  %s\n", mess);
    if (level == GRIB_LOG_WARNING)
        fprintf(c->log_stream, "ECCODES WARNING :  %s\n", mess);
    if (level == GRIB_LOG_INFO)
        fprintf(c->log_stream, "ECCODES INFO    :  %s\n", mess);

    if (level == GRIB_LOG_FATAL) {
        Assert(0);
    }

    if (getenv("ECCODES_FAIL_IF_LOG_MESSAGE")) {
        long n = atol(getenv("ECCODES_FAIL_IF_LOG_MESSAGE"));
        if (n >= 1 && level == GRIB_LOG_ERROR)
            Assert(0);
        if (n >= 2 && level == GRIB_LOG_WARNING)
            Assert(0);
    }
}

static void default_long_lasting_free(const grib_context* c, void* p)
{
    free(p);
}

static void* default_long_lasting_malloc(const grib_context* c, size_t size)
{
    void* ret;
    ret = malloc(size);
    if (!ret) {
        grib_context_log(c, GRIB_LOG_FATAL, "default_long_lasting_malloc: error allocating %lu bytes", (unsigned long)size);
        Assert(0);
    }
    return ret;
}

static int default_feof(const grib_context* c, void* stream)
{
    return feof((FILE*)stream);
}

static void default_free(const grib_context* c, void* p)
{
    free(p);
}

static void* default_malloc(const grib_context* c, size_t size)
{
    void* ret;
    ret = malloc(size);
    if (!ret) {
        grib_context_log(c, GRIB_LOG_FATAL, "default_malloc: error allocating %lu bytes", (unsigned long)size);
        Assert(0);
    }
    return ret;
}

static void default_print(const grib_context* c, void* descriptor, const char* mess)
{
    fprintf((FILE*)descriptor, "%s", mess);
}

static size_t default_read(const grib_context* c, void* ptr, size_t size, void* stream)
{
    return fread(ptr, 1, size, (FILE*)stream);
}

static void* default_realloc(const grib_context* c, void* p, size_t size)
{
    void* ret;
    ret = realloc(p, size);
    if (!ret) {
        grib_context_log(c, GRIB_LOG_FATAL, "default_realloc: error allocating %lu bytes", (unsigned long)size);
        Assert(0);
    }
    return ret;
}

static off_t default_seek(const grib_context* c, off_t offset, int whence, void* stream)
{
    return fseeko((FILE*)stream, offset, whence);
}

static off_t default_tell(const grib_context* c, void* stream)
{
    return ftello((FILE*)stream);
}

static size_t default_write(const grib_context* c, const void* ptr, size_t size, void* stream)
{
    return fwrite(ptr, 1, size, (FILE*)stream);
}



