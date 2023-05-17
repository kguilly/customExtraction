#include "header.h"
#include <float.h>
// #include "accessor_class/grib_accessor_class.h"
// #include "iterator_class/grib_iterator_factory.h"


// #include <stdio.h>
// #include <stdlib.h>
// #include <unistd.h>
// #include <sys/types.h>


int codes_access(const char* name, int mode)
{
    /* F_OK tests for the existence of the file  */
    if (mode != F_OK) {
        return access(name, mode);
    }

    if (codes_memfs_exists(name)) { /* Check memory */
        return 0;
    }

    return access(name, mode);
}

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

static grib_action* grib_parse_stream(grib_context* gc, const char* filename)
{
    GRIB_MUTEX_INIT_ONCE(&once, &init);
    GRIB_MUTEX_LOCK(&mutex_stream);

    grib_action* grib_parser_all_actions = 0;
    // grib_parser_all_actions = 0;

    if (parse(gc, filename) == 0) {
        if (grib_parser_all_actions) {
            GRIB_MUTEX_UNLOCK(&mutex_stream)
            return grib_parser_all_actions;
        }
        else {
            grib_action* ret = grib_action_create_noop(gc, filename);
            GRIB_MUTEX_UNLOCK(&mutex_stream)
            return ret;
        }
    }
    else {
        GRIB_MUTEX_UNLOCK(&mutex_stream);
        return NULL;
    }
}

grib_action_file* grib_find_action_file(const char* fname, grib_action_file_list* afl)
{
    grib_action_file* act = afl->first;
    while (act) {
        if (grib_inline_strcmp(act->filename, fname) == 0)
            return act;
        act = act->next;
    }
    return 0;
}

grib_accessor* _grib_accessor_get_attribute(grib_accessor* a, const char* name, int* index)
{
    int i = 0;
    while (i < MAX_ACCESSOR_ATTRIBUTES && a->attributes[i]) {
        if (!grib_inline_strcmp(a->attributes[i]->name, name)) {
            *index = i;
            return a->attributes[i];
        }
        i++;
    }
    return NULL;
}

grib_accessor* grib_accessor_get_attribute(grib_accessor* a, const char* name)
{
    int index                  = 0;
    const char* p              = 0;
    char* basename             = NULL;
    const char* attribute_name = NULL;
    grib_accessor* acc         = NULL;
    p                          = name;
    while (*(p + 1) != '\0' && (*p != '-' || *(p + 1) != '>'))
        p++;
    if (*(p + 1) == '\0') {
        return _grib_accessor_get_attribute(a, name, &index);
    }
    else {
        size_t size    = p - name;
        attribute_name = p + 2;
        basename       = (char*)grib_context_malloc_clear(a->context, size + 1);
        basename       = (char*)memcpy(basename, name, size);
        acc            = _grib_accessor_get_attribute(a, basename, &index);
        grib_context_free(a->context, basename);
        if (acc)
            return grib_accessor_get_attribute(acc, attribute_name);
        else
            return NULL;
    }
}

grib_accessor* grib_accessor_get_attribute_by_index(grib_accessor* a, int index)
{
    if (index < MAX_ACCESSOR_ATTRIBUTES)
        return a->attributes[index];

    return NULL;
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

static grib_accessor* _grib_find_accessor(const grib_handle* ch, const char* name)
{
    grib_handle* h   = (grib_handle*)ch;
    grib_accessor* a = NULL;
    char* p          = NULL;
    DebugAssert(name);

    p = strchr((char*)name, '.');
    if (p) {
        int i = 0, len = 0;
        char name_space[MAX_NAMESPACE_LEN];
        char* basename = p + 1;
        p--;
        len = p - name + 1;

        for (i = 0; i < len; i++)
            name_space[i] = *(name + i);

        name_space[len] = '\0';

        a = search_and_cache(h, basename, name_space);
    }
    else {
        a = search_and_cache(h, name, NULL);
    }

    if (a == NULL && h->main)
        a = grib_find_accessor(h->main, name);

    return a;
}

static grib_accessor* search(grib_section* s, const char* name, const char* name_space)
{
    grib_accessor* match = NULL;

    grib_accessor* a = s ? s->block->first : NULL;
    grib_accessor* b = NULL;

    if (!a || !s)
        return NULL;

    while (a) {
        grib_section* sub = a->sub_section;

        if (matching(a, name, name_space))
            match = a;

        if ((b = search(sub, name, name_space)) != NULL)
            match = b;

        a = a->next;
    }

    return match;
}

static grib_accessor* search_and_cache(grib_handle* h, const char* name, const char* the_namespace)
{
    grib_accessor* a = NULL;

    if (name[0] == '#') {
        int rank       = -1;
        char* basename = get_rank(h->context, name, &rank);
        a              = search_by_rank(h, basename, rank, the_namespace);
        grib_context_free(h->context, basename);
    }
    else {
        a = _search_and_cache(h, name, the_namespace);
    }

    return a;
}

static grib_accessor* _search_and_cache(grib_handle* h, const char* name, const char* the_namespace)
{
    if (h->use_trie) {
        grib_accessor* a = NULL;
        int id           = -1;

        if (h->trie_invalid && h->kid == NULL) {
            int i = 0;
            for (i = 0; i < ACCESSORS_ARRAY_SIZE; i++)
                h->accessors[i] = NULL;

            if (h->root)
                rebuild_hash_keys(h, h->root);

            h->trie_invalid = 0;
            id              = grib_hash_keys_get_id(h->context->keys, name);
        }
        else {
            id = grib_hash_keys_get_id(h->context->keys, name);

            if ((a = h->accessors[id]) != NULL &&
                (the_namespace == NULL || matching(a, name, the_namespace)))
                return a;
        }

        a                = search(h->root, name, the_namespace);
        h->accessors[id] = a;

        return a;
    }
    else {
        return search(h->root, name, the_namespace);
    }
}

static grib_accessor* _search_by_rank(grib_accessor* a, const char* name, int rank)
{
    grib_trie_with_rank* t = accessor_bufr_data_array_get_dataAccessorsTrie(a);
    grib_accessor* ret     = (grib_accessor*)grib_trie_with_rank_get(t, name, rank);
    return ret;
}

static grib_accessor* search_by_rank(grib_handle* h, const char* name, int rank, const char* the_namespace)
{
    grib_accessor* data = search_and_cache(h, "dataAccessors", the_namespace);
    if (data) {
        return _search_by_rank(data, name, rank);
    }
    else {
        int rank2;
        char* str          = get_rank(h->context, name, &rank2);
        grib_accessor* ret = _search_and_cache(h, str, the_namespace);
        grib_context_free(h->context, str);
        return ret;
    }
}

grib_action* grib_parse_file(grib_context* gc, const char* filename)
{
    grib_action_file* af;

    GRIB_MUTEX_INIT_ONCE(&once, &init);
    GRIB_MUTEX_LOCK(&mutex_file);

    af = 0;

    gc = gc ? gc : grib_context_get_default();
    grib_context* grib_parser_context = 0;
    grib_parser_context = gc;

    if (!gc->grib_reader)
        gc->grib_reader = (grib_action_file_list*)grib_context_malloc_clear_persistent(gc, sizeof(grib_action_file_list));
    else {
        af = grib_find_action_file(filename, gc->grib_reader);
    }

    if (!af) {
        grib_action* a;
        grib_context_log(gc, GRIB_LOG_DEBUG, "Loading %s", filename);

        a = grib_parse_stream(gc, filename);

        if (error) {
            if (a)
                grib_action_delete(gc, a);
            GRIB_MUTEX_UNLOCK(&mutex_file);
            return NULL;
        }

        af = (grib_action_file*)grib_context_malloc_clear_persistent(gc, sizeof(grib_action_file));

        af->root = a;

        af->filename = grib_context_strdup_persistent(gc, filename);
        grib_push_action_file(af, gc->grib_reader); /* Add af to grib_reader action file list */
    }
    else
        grib_context_log(gc, GRIB_LOG_DEBUG, "Using cached version of %s", filename);

    GRIB_MUTEX_UNLOCK(&mutex_file);
    return af->root;
}

grib_accessors_list* accessor_bufr_data_array_get_dataAccessors(grib_accessor* a)
{
    grib_accessor_bufr_data_array* self = (grib_accessor_bufr_data_array*)a;
    return self->dataAccessors;
}

grib_accessors_list* grib_accessors_list_last(grib_accessors_list* al)
{
    /*grib_accessors_list* last=al;*/
    /*grib_accessors_list* next=al->next;*/

    /*
    while(next) {
      last=next;
      next=last->next;
    }
     */
    return al->last;
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

static grib_accessors_list* search_by_condition(grib_handle* h, const char* name, codes_condition* condition)
{
    grib_accessors_list* al;
    grib_accessors_list* result = NULL;
    grib_accessor* data         = search_and_cache(h, "dataAccessors", 0);
    if (data && condition->left) {
        al = accessor_bufr_data_array_get_dataAccessors(data);
        if (!al)
            return NULL;
        result = (grib_accessors_list*)grib_context_malloc_clear(al->accessor->context, sizeof(grib_accessors_list));
        search_accessors_list_by_condition(al, name, condition, result);
        if (!result->accessor) {
            grib_accessors_list_delete(h->context, result);
            result = NULL;
        }
    }

    return result;
}

grib_buffer* grib_new_buffer(const grib_context* c, const unsigned char* data, size_t buflen)
{
    grib_buffer* b = (grib_buffer*)grib_context_malloc_clear(c, sizeof(grib_buffer));

    if (b == NULL) {
        grib_context_log(c, GRIB_LOG_ERROR, "grib_new_buffer: cannot allocate buffer");
        return NULL;
    }

    b->property     = GRIB_USER_BUFFER;
    b->length       = buflen;
    b->ulength      = buflen;
    b->ulength_bits = buflen * 8;
    b->data         = (unsigned char*)data;

    return b;
}

static grib_context default_grib_context = {
    0,               /* inited                     */
    0,               /* debug                      */
    0,               /* write_on_fail              */
    0,               /* no_abort                   */
    0,               /* io_buffer_size             */
    0,               /* no_big_group_split         */
    0,               /* no_spd                     */
    0,               /* keep_matrix                */
    0,               /* grib_definition_files_path */
    0,               /* grib_samples_path          */
    0,               /* grib_concept_path          */
    0,               /* grib_reader                */
    0,               /* user data                  */
    GRIB_REAL_MODE8, /* real mode for fortran      */

#if MANAGE_MEM
    &grib_transient_free,    /* free_mem                   */
    &grib_transient_malloc,  /* alloc_mem                  */
    &grib_transient_realloc, /* realloc_mem                */

    &grib_permanent_free,   /* free_persistant_mem        */
    &grib_permanent_malloc, /* alloc_persistant_mem       */

    &grib_buffer_free,    /* buffer_free_mem            */
    &grib_buffer_malloc,  /* buffer_alloc_mem           */
    &grib_buffer_realloc, /* buffer_realloc_mem         */

#else

    &default_free,    /* free_mem                  */
    &default_malloc,  /* alloc_mem                 */
    &default_realloc, /* realloc_mem               */

    &default_long_lasting_free,   /* free_persistant_mem       */
    &default_long_lasting_malloc, /* alloc_persistant_mem      */

    &default_buffer_free,    /* free_buffer_mem           */
    &default_buffer_malloc,  /* alloc_buffer_mem          */
    &default_buffer_realloc, /* realloc_buffer_mem        */
#endif

    &default_read,  /* file read procedure        */
    &default_write, /* file write procedure       */
    &default_tell,  /* lfile tell procedure       */
    &default_seek,  /* lfile seek procedure       */
    &default_feof,  /* file feof procedure        */

    &default_log,   /* output_log                 */
    &default_print, /* print                      */
    0,              /* codetable                  */
    0,              /* smart_table                */
    0,              /* outfilename                */
    0,              /* multi_support_on           */
    0,              /* multi_support              */
    0,              /* grib_definition_files_dir  */
    0,              /* handle_file_count          */
    0,              /* handle_total_count         */
    0,              /* message_file_offset        */
    0,              /* no_fail_on_wrong_length    */
    0,              /* gts_header_on              */
    0,              /* gribex_mode_on             */
    0,              /* large_constant_fields      */
    0,              /* keys                       */
    0,              /* keys_count                 */
    0,              /* concepts_index             */
    0,              /* concepts_count             */
    {0,}, /* concepts                   */
    0, /* hash_array_index           */
    0, /* hash_array_count           */
    {0,},                                 /* hash_array                 */
    0,                                 /* def_files                  */
    0,                                 /* blocklist                  */
    0,                                 /* ieee_packing               */
    0,                                 /* bufrdc_mode                */
    0,                                 /* bufr_set_to_missing_if_out_of_range */
    0,                                 /* bufr_multi_element_constant_arrays */
    0,                                 /* grib_data_quality_checks   */
    0,                                 /* log_stream                 */
    0,                                 /* classes                    */
    0,                                 /* lists                      */
    0,                                 /* expanded_descriptors       */
    DEFAULT_FILE_POOL_MAX_OPENED_FILES /* file_pool_max_opened_files */
#if GRIB_PTHREADS
    ,
    PTHREAD_MUTEX_INITIALIZER /* mutex                      */
#endif
};
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

static grib_handle* grib_handle_create(grib_handle* gl, grib_context* c, const void* data, size_t buflen)
{
    grib_action* next = NULL;
    int err           = 0;

    if (gl == NULL)
        return NULL;

    gl->use_trie     = 1;
    gl->trie_invalid = 0;
    gl->buffer       = grib_new_buffer(gl->context, (const unsigned char*)data, buflen);

    if (gl->buffer == NULL) {
        grib_handle_delete(gl);
        return NULL;
    }

    gl->root = grib_create_root_section(gl->context, gl);

    if (!gl->root) {
        grib_context_log(c, GRIB_LOG_ERROR, "grib_handle_create: cannot create root section");
        grib_handle_delete(gl);
        return NULL;
    }

    if (!gl->context->grib_reader || !gl->context->grib_reader->first) {
        grib_context_log(c, GRIB_LOG_ERROR, "grib_handle_create: cannot create handle, no definitions found");
        grib_handle_delete(gl);
        return NULL;
    }

    gl->buffer->property = GRIB_USER_BUFFER;

    next = gl->context->grib_reader->first->root;
    while (next) {
        if (grib_create_accessor(gl->root, next, NULL) != GRIB_SUCCESS)
            break;
        next = next->next;
    }

    err = grib_section_adjust_sizes(gl->root, 0, 0);
    if (err) {
        grib_handle_delete(gl);
        return NULL;
    }

    grib_section_post_init(gl->root);

    return gl;
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

grib_handle* grib_handle_of_accessor(const grib_accessor* a)
{
    if (a->parent == NULL) {
        return a->h;
    }
    else {
        return a->parent->h;
    }
}

grib_handle* grib_handle_new_from_message(grib_context* c, const void* data, size_t buflen)
{
    grib_handle* gl          = NULL;
    grib_handle* h           = NULL;
    ProductKind product_kind = PRODUCT_ANY;
    if (c == NULL)
        c = grib_context_get_default();
    gl               = grib_new_handle(c);
    gl->product_kind = PRODUCT_GRIB; /* See ECC-480 */
    h                = grib_handle_create(gl, c, data, buflen);

    /* See ECC-448 */
    if (determine_product_kind(h, &product_kind) == GRIB_SUCCESS) {
        h->product_kind = product_kind;
    }

    if (h->product_kind == PRODUCT_GRIB) {
        if (!grib_is_defined(h, "7777")) {
            grib_context_log(c, GRIB_LOG_ERROR, "grib_handle_new_from_message: No final 7777 in message!");
            /* TODO: Return NULL. An incomplete message is no use to anyone.
             * But first check the MARS Client and other applications
             */
        }
    }
    return h;
}

grib_handle* grib_handle_new_from_partial_message(grib_context* c, const void* data, size_t buflen)
{
    grib_handle* gl = NULL;
    if (c == NULL)
        c = grib_context_get_default();
    grib_context_set_handle_file_count(c, 0);
    grib_context_set_handle_total_count(c, 0);
    gl          = grib_new_handle(c);
    gl->partial = 1;
    return grib_handle_create(gl, c, data, buflen);
}

static grib_handle* grib_handle_new_from_file_multi(grib_context* c, FILE* f, int* error)
{
    void *data = NULL, *old_data = NULL;
    size_t olen = 0, len = 0;
    grib_handle* gl         = NULL;
    long edition            = 0;
    size_t seclen           = 0;
    unsigned char* secbegin = 0;
    int secnum = 0, seccount = 0;
    int err = 0, i = 0;
    grib_multi_support* gm  = NULL;
    off_t gts_header_offset = 0;
    off_t end_msg_offset = 0, offset = 0;
    char *gts_header = 0, *save_gts_header = 0;
    int gtslen = 0;

    if (c == NULL)
        c = grib_context_get_default();

    gm = grib_get_multi_support(c, f);

    if (!gm->message) {
        gts_header_offset = grib_context_tell(c, f);
        data              = wmo_read_grib_from_file_malloc(f, 0, &olen, &offset, error);
        end_msg_offset    = grib_context_tell(c, f);

        gm->message_length = olen;
        gm->message        = (unsigned char*)data;
        gm->offset         = offset;
        if (*error != GRIB_SUCCESS || !data) {
            if (data)
                grib_context_free(c, data);

            if (*error == GRIB_END_OF_FILE)
                *error = GRIB_SUCCESS;
            gm->message_length = 0;
            gm->message        = NULL;
            return NULL;
        }
        if (c->gts_header_on) {
            int g = 0;
            grib_context_seek(c, gts_header_offset, SEEK_SET, f);
            gtslen          = offset - gts_header_offset;
            gts_header      = (char*)grib_context_malloc_clear(c, sizeof(unsigned char) * gtslen);
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
    }
    else
        data = gm->message;

    edition = grib_decode_unsigned_byte_long((const unsigned char*)data, 7, 1);

    if (edition == 2) {
        olen = gm->message_length;
        if (gm->section_number == 0) {
            gm->sections[0] = (unsigned char*)data;
        }
        secbegin = gm->sections[gm->section_number];
        seclen   = gm->sections_length[gm->section_number];
        secnum   = gm->section_number;
        seccount = 0;
        while (grib2_get_next_section((unsigned char*)data, olen, &secbegin, &seclen, &secnum, &err)) {
            seccount++;
            /*printf("   - %d - section %d length=%d\n",(int)seccount,(int)secnum,(int)seclen);*/

            gm->sections[secnum]        = secbegin;
            gm->sections_length[secnum] = seclen;

            if (secnum == 6) {
                /* Special case for inherited bitmaps */
                if (grib_decode_unsigned_byte_long(secbegin, 5, 1) == 254) {
                    if (!gm->bitmap_section) {
                        grib_context_log(c, GRIB_LOG_ERROR, "grib_handle_new_from_file_multi: cannot create handle, missing bitmap\n");
                        grib_context_free(c, data);
                        return NULL;
                    }
                    gm->sections[secnum]        = gm->bitmap_section;
                    gm->sections_length[secnum] = gm->bitmap_section_length;
                }
                else {
                    if (gm->bitmap_section) {
                        grib_context_free(c, gm->bitmap_section);
                        gm->bitmap_section = NULL;
                    }
                    gm->bitmap_section        = (unsigned char*)grib_context_malloc(c, seclen);
                    gm->bitmap_section        = (unsigned char*)memcpy(gm->bitmap_section, secbegin, seclen);
                    gm->bitmap_section_length = seclen;
                }
            }

            if (secnum == 7) {
                old_data = data;
                len      = olen;
                grib2_build_message(c, gm->sections, gm->sections_length, &data, &len);

                if (grib2_has_next_section((unsigned char*)old_data, olen, secbegin, seclen, &err)) {
                    gm->message        = (unsigned char*)old_data;
                    gm->section_number = secnum;
                    olen               = len;
                }
                else {
                    if (gm->message)
                        grib_context_free(c, gm->message);
                    gm->message = NULL;
                    for (i = 0; i < 8; i++)
                        gm->sections[i] = NULL;
                    gm->section_number = 0;
                    gm->message_length = 0;
                    olen               = len;
                }
                break;
            }
        }
    }
    else if (edition == 3) {
        /* GRIB3: Multi-field mode not yet supported */
        printf("WARNING: %s\n", "grib_handle_new_from_file_multi: GRIB3 multi-field mode not yet implemented! Reverting to single-field mode");
        gm->message_length = 0;
        gm->message        = NULL;
    }
    else {
        gm->message_length = 0;
        gm->message        = NULL;
    }

    gl = grib_handle_new_from_message(c, data, olen);
    if (!gl) {
        *error = GRIB_DECODING_ERROR;
        grib_context_log(c, GRIB_LOG_ERROR, "grib_handle_new_from_file_multi: cannot create handle \n");
        grib_context_free(c, data);
        return NULL;
    }

    gl->offset           = gm->offset;
    gl->buffer->property = GRIB_MY_BUFFER;
    grib_context_increment_handle_file_count(c);
    grib_context_increment_handle_total_count(c);

    if (c->gts_header_on && gtslen >= 8) {
        gl->gts_header = (char*)grib_context_malloc_clear(c, sizeof(unsigned char) * gtslen);
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
    // TODO: 
    // end_msg offset returns 503402, when it should return 0
    // p end_msg_offset = 0
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

grib_handle* grib_new_handle(grib_context* c)
{
    grib_handle* g = NULL;
    if (c == NULL)
        c = grib_context_get_default();
    g = (grib_handle*)grib_context_malloc_clear(c, sizeof(grib_handle));

    if (g == NULL) {
        grib_context_log(c, GRIB_LOG_ERROR, "grib_new_handle: cannot allocate handle");
    }
    else {
        g->context      = c;
        g->product_kind = PRODUCT_ANY; /* Default. Will later be set to a specific product */
    }

    grib_context_log(c, GRIB_LOG_DEBUG, "grib_new_handle: allocated handle %p", (void*)g);

    return g;
}

const struct grib_keys_hash* grib_keys_hash_get (register const char *str, register size_t len)
{
  if (len <= MAX_WORD_LENGTH && len >= MIN_WORD_LENGTH)
    {
      register unsigned int key = hash_keys (str, len);

      if (key <= MAX_HASH_VALUE)
        if (len == lengthtable[key])
          {
            register const char *s = wordlist[key].name;

            if (*str == *s && !memcmp (str + 1, s + 1, len - 1))
              return &wordlist[key];
          }
    }
  return 0;
}

grib_iterator_class* grib_iterator_class_gaussian;
grib_iterator_class* grib_iterator_class_gen;
grib_iterator_class* grib_iterator_class_lambert_conformal;
grib_iterator_class* grib_iterator_class_latlon;
grib_iterator_class* grib_iterator_class_regular;
grib_iterator_class* grib_iterator_class_gaussian_reduced;
grib_iterator_class* grib_iterator_class_lambert_azimuthal_equal_area;
grib_iterator_class* grib_iterator_class_latlon_reduced;
grib_iterator_class* grib_iterator_class_polar_stereographic;
grib_iterator_class* grib_iterator_class_mercator;
grib_iterator_class* grib_iterator_class_space_view;
static const struct iterator_table_entry iterator_table[] = {
{ "gaussian", &grib_iterator_class_gaussian, },
{ "gaussian_reduced", &grib_iterator_class_gaussian_reduced, },
{ "gen", &grib_iterator_class_gen, },
{ "lambert_azimuthal_equal_area", &grib_iterator_class_lambert_azimuthal_equal_area, },
{ "lambert_conformal", &grib_iterator_class_lambert_conformal, },
{ "latlon", &grib_iterator_class_latlon, },
{ "latlon_reduced", &grib_iterator_class_latlon_reduced, },
{ "mercator", &grib_iterator_class_mercator, },
{ "polar_stereographic", &grib_iterator_class_polar_stereographic, },
{ "regular", &grib_iterator_class_regular, },
{ "space_view", &grib_iterator_class_space_view, },
};

void grib_get_reduced_row(long pl, double lon_first, double lon_last, long* npoints, long* ilon_first, long* ilon_last)
{
    long long Ni_globe = pl;
    Fraction_type west;
    Fraction_type east;
    long long the_count;
    double the_lon1, the_lon2;

    while (lon_last < lon_first)
        lon_last += 360;
    west = fraction_construct_from_double(lon_first);
    east = fraction_construct_from_double(lon_last);

    gaussian_reduced_row(
        Ni_globe, /*plj*/
        west,     /*lon_first*/
        east,     /*lon_last*/
        &the_count,
        &the_lon1,
        &the_lon2);
    *npoints    = (long)the_count;
    *ilon_first = (the_lon1 * pl) / 360.0;
    *ilon_last  = (the_lon2 * pl) / 360.0;
}
void grib_get_reduced_row_legacy(long pl, double lon_first, double lon_last, long* npoints, long* ilon_first, long* ilon_last)
{
    double range = 0, dlon_first = 0, dlon_last = 0;
    long irange;
    range = lon_last - lon_first;
    if (range < 0) {
        range += 360;
        lon_first -= 360;
    }

    /* computing integer number of points and coordinates without using floating point resolution*/
    *npoints    = (range * pl) / 360.0 + 1;
    *ilon_first = (lon_first * pl) / 360.0;
    *ilon_last  = (lon_last * pl) / 360.0;

    irange = *ilon_last - *ilon_first + 1;

#if EFDEBUG
    printf("  pl=%ld npoints=%ld range=%.10e ilon_first=%ld ilon_last=%ld irange=%ld\n",
           pl, *npoints, range, *ilon_first, *ilon_last, irange);
#endif

    if (irange != *npoints) {
#if EFDEBUG
        printf("       ---> (irange=%ld) != (npoints=%ld) ", irange, *npoints);
#endif
        if (irange > *npoints) {
            /* checking if the first point is out of range*/
            dlon_first = ((*ilon_first) * 360.0) / pl;
            if (dlon_first < lon_first) {
                (*ilon_first)++;
                irange--;
#if EFDEBUG
                printf(" dlon_first=%.10e < lon_first=%.10e\n", dlon_first, lon_first);
#endif
            }

            /* checking if the last point is out of range*/
            dlon_last = ((*ilon_last) * 360.0) / pl;
            if (dlon_last > lon_last) {
                (*ilon_last)--;
                irange--;
#if EFDEBUG
                printf(" dlon_last=%.10e < lon_last=%.10e\n", dlon_last, lon_last);
#endif
            }
        }
        else {
            int ok = 0;
            /* checking if the point before the first is in the range*/
            dlon_first = ((*ilon_first - 1) * 360.0) / pl;
            if (dlon_first > lon_first) {
                (*ilon_first)--;
                irange++;
                ok = 1;
#if EFDEBUG
                printf(" dlon_first1=%.10e > lon_first=%.10e\n", dlon_first, lon_first);
#endif
            }

            /* checking if the point after the last is in the range*/
            dlon_last = ((*ilon_last + 1) * 360.0) / pl;
            if (dlon_last < lon_last) {
                (*ilon_last)++;
                irange++;
                ok = 1;
#if EFDEBUG
                printf(" dlon_last1=%.10e > lon_last=%.10e\n", dlon_last, lon_first);
#endif
            }

            /* if neither of the two are triggered then npoints is too large */
            if (!ok) {
                (*npoints)--;
#if EFDEBUG
                printf(" (*npoints)--=%ld\n", *npoints);
#endif
            }
        }

        /*Assert(*npoints==irange);*/
#if EFDEBUG
        printf("--  pl=%ld npoints=%ld range=%.10e ilon_first=%ld ilon_last=%ld irange=%ld\n",
               pl, *npoints, range, *ilon_first, *ilon_last, irange);
#endif
    }
    else {
        /* checking if the first point is out of range*/
        dlon_first = ((*ilon_first) * 360.0) / pl;
        if (dlon_first < lon_first) {
            (*ilon_first)++;
            (*ilon_last)++;
#if EFDEBUG
            printf("       ---> dlon_first=%.10e < lon_first=%.10e\n", dlon_first, lon_first);
            printf("--  pl=%ld npoints=%ld range=%.10e ilon_first=%ld ilon_last=%ld irange=%ld\n",
                   pl, *npoints, range, *ilon_first, *ilon_last, irange);
#endif
        }
    }

    if (*ilon_first < 0)
        *ilon_first += pl;

    return;
}

grib_iterator* grib_iterator_factory(grib_handle* h, grib_arguments* args, unsigned long flags, int* ret)
{
    int i;
    const char* type = (char*)grib_arguments_get_name(h, args, 0);

    /*
        NOTE: 
        - "table" includes a complicated process to implement. When stepping through the 
           library, the type that was identified was "lambert conformal" 
           I plan to only keep the code associated to lambert conformal. 

    */
    for (i = 0; i < NUMBER(iterator_table); i++)
        if (strcmp(type, iterator_table[i].type) == 0) {
            grib_iterator_class* c = *(iterator_table[i].cclass);
            grib_iterator* it      = (grib_iterator*)grib_context_malloc_clear(h->context, c->size);
            it->cclass             = c;
            it->flags              = flags;
            *ret                   = GRIB_SUCCESS;
            *ret                   = grib_iterator_init(it, h, args);
            if (*ret == GRIB_SUCCESS)
                return it;
            grib_context_log(h->context, GRIB_LOG_ERROR, "Geoiterator factory: Error instantiating iterator %s (%s)",
                             iterator_table[i].type, grib_get_error_message(*ret));
            grib_iterator_delete(it);
            return NULL;
        }

    grib_context_log(h->context, GRIB_LOG_ERROR, "Geoiterator factory: Unknown type: %s for iterator", type);

    return NULL;
}

grib_iterator* grib_iterator_new(const grib_handle* ch, unsigned long flags, int* error)
{
    grib_handle* h              = (grib_handle*)ch;
    grib_accessor* a            = NULL;
    grib_accessor_iterator* ita = NULL;
    grib_iterator* iter         = NULL;
    *error                      = GRIB_NOT_IMPLEMENTED;
    a                           = grib_find_accessor(h, "ITERATOR");
    ita                         = (grib_accessor_iterator*)a;

    if (!a)
        return NULL;

    iter = grib_iterator_factory(h, ita->args, flags, error);

    if (iter)
        *error = GRIB_SUCCESS;

    return iter;
}

static grib_multi_support* grib_get_multi_support(grib_context* c, FILE* f)
{
    int i                    = 0;
    grib_multi_support* gm   = c->multi_support;
    grib_multi_support* prev = NULL;

    while (gm) {
        if (gm->file == f)
            return gm;
        prev = gm;
        gm   = gm->next;
    }

    if (!gm) {
        gm = grib_multi_support_new(c);
        if (!c->multi_support) {
            c->multi_support = gm;
        }
        else {
            if (prev)
                prev->next = gm;
        }
    }

    gm->next = 0;
    if (gm->message)
        grib_context_free(c, gm->message);
    gm->message            = NULL;
    gm->section_number     = 0;
    gm->sections_length[0] = 16;
    for (i = 1; i < 8; i++)
        gm->sections_length[i] = 0;
    gm->sections_length[8] = 4;
    gm->file               = f;

    return gm;
}

static grib_multi_support* grib_multi_support_new(grib_context* c)
{
    int i = 0;
    grib_multi_support* gm =
        (grib_multi_support*)grib_context_malloc_clear(c, sizeof(grib_multi_support));
    gm->file                  = NULL;
    gm->message               = NULL;
    gm->message_length        = 0;
    gm->bitmap_section        = NULL;
    gm->bitmap_section_length = 0;
    gm->section_number        = 0;
    gm->next                  = 0;
    gm->sections_length[0]    = 16;
    for (i = 1; i < 8; i++)
        gm->sections_length[i] = 0;
    gm->sections_length[8] = 4;

    return gm;
}

grib_section* grib_create_root_section(const grib_context* context, grib_handle* h)
{
    char* fpath     = 0;
    grib_section* s = (grib_section*)grib_context_malloc_clear(context, sizeof(grib_section));

    GRIB_MUTEX_INIT_ONCE(&once, &init);
    GRIB_MUTEX_LOCK(&mutex1);
    if (h->context->grib_reader == NULL) {
        if ((fpath = grib_context_full_defs_path(h->context, "boot.def")) == NULL) {
            grib_context_log(h->context, GRIB_LOG_FATAL,
                             "Unable to find boot.def. Context path=%s\n"
                             "\nPossible causes:\n"
                             "- The software is not correctly installed\n"
                             "- The environment variable ECCODES_DEFINITION_PATH is defined but incorrect\n",
                             context->grib_definition_files_path);
        }
        grib_parse_file(h->context, fpath);
    }
    GRIB_MUTEX_UNLOCK(&mutex1);

    s->h        = h;
    s->aclength = NULL;
    s->owner    = NULL;
    s->block    = (grib_block_of_accessors*)
        grib_context_malloc_clear(context, sizeof(grib_block_of_accessors));
    grib_context_log(context, GRIB_LOG_DEBUG, "Creating root section");
    return s;
}

grib_itrie* grib_hash_keys_new(grib_context* c, int* count)
{
    grib_itrie* t = (grib_itrie*)grib_context_malloc_clear(c, sizeof(grib_itrie));
    t->context    = c;
    t->id         = -1;
    t->count      = count;
    return t;
}

grib_itrie* grib_itrie_new(grib_context* c, int* count)
{
    grib_itrie* t = (grib_itrie*)grib_context_malloc_clear(c, sizeof(grib_itrie));
    t->context    = c;
    t->id         = -1;
    t->count      = count;
    return t;
}

grib_trie* grib_trie_new(grib_context* c)
{
#ifdef RECYCLE_TRIE
    grib_trie* t = grib_context_malloc_clear_persistent(c, sizeof(grib_trie));
#else
    grib_trie* t = (grib_trie*)grib_context_malloc_clear(c, sizeof(grib_trie));
#endif
    t->context = c;
    t->first   = TRIE_SIZE;
    t->last    = -1;
    return t;
}

grib_trie_with_rank* accessor_bufr_data_array_get_dataAccessorsTrie(grib_accessor* a)
{
    grib_accessor_bufr_data_array* self = (grib_accessor_bufr_data_array*)a;
    return self->dataAccessorsTrie;
}

char* codes_getenv(const char* name)
{
    /* Look for the new ecCodes environment variable names */
    /* if not found, then look for old grib_api ones for backward compatibility */
    char* result = getenv(name);
    if (result == NULL) {
        const char* old_name = name;

        /* Test the most commonly used variables first */
        if (STR_EQ(name, "ECCODES_SAMPLES_PATH"))
            old_name = "GRIB_SAMPLES_PATH";
        else if (STR_EQ(name, "ECCODES_DEFINITION_PATH"))
            old_name = "GRIB_DEFINITION_PATH";
        else if (STR_EQ(name, "ECCODES_DEBUG"))
            old_name = "GRIB_API_DEBUG";

        else if (STR_EQ(name, "ECCODES_FAIL_IF_LOG_MESSAGE"))
            old_name = "GRIB_API_FAIL_IF_LOG_MESSAGE";
        else if (STR_EQ(name, "ECCODES_GRIB_WRITE_ON_FAIL"))
            old_name = "GRIB_API_WRITE_ON_FAIL";
        else if (STR_EQ(name, "ECCODES_GRIB_LARGE_CONSTANT_FIELDS"))
            old_name = "GRIB_API_LARGE_CONSTANT_FIELDS";
        else if (STR_EQ(name, "ECCODES_NO_ABORT"))
            old_name = "GRIB_API_NO_ABORT";
        else if (STR_EQ(name, "ECCODES_GRIBEX_MODE_ON"))
            old_name = "GRIB_GRIBEX_MODE_ON";
        else if (STR_EQ(name, "ECCODES_GRIB_IEEE_PACKING"))
            old_name = "GRIB_IEEE_PACKING";
        else if (STR_EQ(name, "ECCODES_IO_BUFFER_SIZE"))
            old_name = "GRIB_API_IO_BUFFER_SIZE";
        else if (STR_EQ(name, "ECCODES_LOG_STREAM"))
            old_name = "GRIB_API_LOG_STREAM";
        else if (STR_EQ(name, "ECCODES_GRIB_NO_BIG_GROUP_SPLIT"))
            old_name = "GRIB_API_NO_BIG_GROUP_SPLIT";
        else if (STR_EQ(name, "ECCODES_GRIB_NO_SPD"))
            old_name = "GRIB_API_NO_SPD";
        else if (STR_EQ(name, "ECCODES_GRIB_KEEP_MATRIX"))
            old_name = "GRIB_API_KEEP_MATRIX";
        else if (STR_EQ(name, "_ECCODES_ECMWF_TEST_DEFINITION_PATH"))
            old_name = "_GRIB_API_ECMWF_TEST_DEFINITION_PATH";
        else if (STR_EQ(name, "_ECCODES_ECMWF_TEST_SAMPLES_PATH"))
            old_name = "_GRIB_API_ECMWF_TEST_SAMPLES_PATH";
        else if (STR_EQ(name, "ECCODES_GRIB_JPEG"))
            old_name = "GRIB_JPEG";
        else if (STR_EQ(name, "ECCODES_GRIB_DUMP_JPG_FILE"))
            old_name = "GRIB_DUMP_JPG_FILE";
        else if (STR_EQ(name, "ECCODES_PRINT_MISSING"))
            old_name = "GRIB_PRINT_MISSING";

        result = getenv(old_name);
    }
    return result;
}

int codes_get_long(const grib_handle* h, const char* key, long* value)
{
    return grib_get_long(h, key, value);
}

int codes_memfs_exists(const char* path) {
    size_t dummy;
    return find(path, &dummy) != NULL;
}

char* codes_resolve_path(grib_context* c, const char* path)
{
    char* result = NULL;
    #if defined(ECCODES_HAVE_REALPATH)
    char resolved[ECC_PATH_MAXLEN + 1];
    if (!realpath(path, resolved)) {
        result = grib_context_strdup(c, path); /* Failed to resolve. Use original path */
    }
    else {
        result = grib_context_strdup(c, resolved);
    }
    #else
    result = grib_context_strdup(c, path);
    #endif

    return result;
}

static int condition_true(grib_accessor* a, codes_condition* condition)
{
    int ret = 0, err = 0;
    long lval   = 0;
    double dval = 0;

    /* The condition has to be of the form:
     *   key=value
     * and the value has to be a single scalar (integer or double).
     * If the key is an array of different values, then the condition is false.
     * But if the key is a constant array and the value matches then it's true.
     */

    switch (condition->rightType) {
        case GRIB_TYPE_LONG:
            err = get_single_long_val(a, &lval);
            if (err)
                ret = 0;
            else
                ret = lval == condition->rightLong ? 1 : 0;
            break;
        case GRIB_TYPE_DOUBLE:
            err = get_single_double_val(a, &dval);
            if (err)
                ret = 0;
            else
                ret = dval == condition->rightDouble ? 1 : 0;
            break;
        default:
            ret = 0;
            break;
    }
    return ret;
}

static int determine_product_kind(grib_handle* h, ProductKind* prod_kind)
{
    int err    = 0;
    size_t len = 0;
    err        = grib_get_length(h, "identifier", &len);
    if (!err) {
        char id_str[64] = {0,};
        err = grib_get_string(h, "identifier", id_str, &len);
        if (grib_inline_strcmp(id_str, "GRIB") == 0)
            *prod_kind = PRODUCT_GRIB;
        else if (grib_inline_strcmp(id_str, "BUFR") == 0)
            *prod_kind = PRODUCT_BUFR;
        else if (grib_inline_strcmp(id_str, "METAR") == 0)
            *prod_kind = PRODUCT_METAR;
        else if (grib_inline_strcmp(id_str, "GTS") == 0)
            *prod_kind = PRODUCT_GTS;
        else if (grib_inline_strcmp(id_str, "TAF") == 0)
            *prod_kind = PRODUCT_TAF;
        else
            *prod_kind = PRODUCT_ANY;
    }
    return err;
}

static const unsigned char* find(const char* path, size_t* length) {
    size_t i;

    // TODO: could not trace back the definition of the entries array
    // after debugging through the lib, the parent function returns 
    // null anyway, so we can probably safely take this out
    // for(i = 0; i < entries_count; i++) {
    //     if(strcmp(path, entries[i].path) == 0) {
    //         /*printf("Found in MEMFS %s\\n", path);*/
    //         *length = entries[i].length;
    //         return entries[i].content;
    //     }
    // }

    return NULL;
}

static char* get_condition(const char* name, codes_condition* condition)
{
    char* equal        = (char*)name;
    char* endCondition = NULL;
    char* str          = NULL;
    char* end          = NULL;
    long lval;
    grib_context* c = grib_context_get_default();

    condition->rightType = GRIB_TYPE_UNDEFINED;

    Assert(name[0] == '/');

    while (*equal != 0 && *equal != '=')
        equal++;
    if (*equal == 0)
        return NULL;

    endCondition = equal;
    while (*endCondition != 0 && *endCondition != '/')
        endCondition++;
    if (*endCondition == 0)
        return NULL;

    str = (char*)grib_context_malloc_clear(c, strlen(name));
    memcpy(str, equal + 1, endCondition - equal - 1);

    end  = NULL;
    lval = strtol(str, &end, 10);
    if (*end != 0) { /* strtol failed. Not an integer */
        double dval;
        dval = strtod(str, &end);
        if (*end == 0) { /* strtod passed. So a double */
            condition->rightType   = GRIB_TYPE_DOUBLE;
            condition->rightDouble = dval;
        }
    }
    else {
        condition->rightType = GRIB_TYPE_LONG;
        condition->rightLong = lval;
    }

    if (condition->rightType != GRIB_TYPE_UNDEFINED) {
        strcpy(str, endCondition + 1);
        condition->left = (char*)grib_context_malloc_clear(c, equal - name);
        memcpy(condition->left, name + 1, equal - name - 1);
    }
    else {
        grib_context_free(c, str);
        str = NULL;
    }
    return str;
}

static char* get_rank(grib_context* c, const char* name, int* rank)
{
    char* p   = (char*)name;
    char* end = p;
    char* ret = NULL;

    *rank = -1;

    if (*p == '#') {
        *rank = strtol(++p, &end, 10);
        if (*end != '#') {
            *rank = -1;
        }
        else {
            DebugAssert(c);
            end++;
            ret = grib_context_strdup(c, end);
        }
    }
    return ret;
}

static int get_single_double_val(grib_accessor* a, double* result)
{
    grib_context* c = a->context;
    int err         = 0;
    size_t size     = 1;
    if (c->bufr_multi_element_constant_arrays) {
        long count = 0;
        grib_value_count(a, &count);
        if (count > 1) {
            size_t i        = 0;
            double val0     = 0;
            int is_constant = 1;
            double* values  = (double*)grib_context_malloc_clear(c, sizeof(double) * count);
            size            = count;
            err             = grib_unpack_double(a, values, &size);
            val0            = values[0];
            for (i = 0; i < size; i++) {
                if (val0 != values[i]) {
                    is_constant = 0;
                    break;
                }
            }
            if (is_constant) {
                *result = val0;
                grib_context_free(c, values);
            }
            else {
                return GRIB_ARRAY_TOO_SMALL;
            }
        }
        else {
            err = grib_unpack_double(a, result, &size);
        }
    }
    else {
        err = grib_unpack_double(a, result, &size);
    }
    return err;
}

static int get_single_long_val(grib_accessor* a, long* result)
{
    grib_context* c = a->context;
    int err         = 0;
    size_t size     = 1;
    if (c->bufr_multi_element_constant_arrays) {
        long count = 0;
        grib_value_count(a, &count);
        if (count > 1) {
            size_t i        = 0;
            long val0       = 0;
            int is_constant = 1;
            long* values    = (long*)grib_context_malloc_clear(c, sizeof(long) * count);
            size            = count;
            err             = grib_unpack_long(a, values, &size);
            val0            = values[0];
            for (i = 0; i < size; i++) {
                if (val0 != values[i]) {
                    is_constant = 0;
                    break;
                }
            }
            if (is_constant) {
                *result = val0;
                grib_context_free(c, values);
            }
            else {
                return GRIB_ARRAY_TOO_SMALL;
            }
        }
        else {
            err = grib_unpack_long(a, result, &size);
        }
    }
    else {
        err = grib_unpack_long(a, result, &size);
    }
    return err;
}

static void grib2_build_message(grib_context* context, unsigned char* sections[], size_t sections_len[], void** data, size_t* len)
{
    int i              = 0;
    const char* theEnd = "7777";
    unsigned char* p   = 0;
    size_t msglen      = 0;
    long bitp          = 64;
    if (!sections[0]) {
        *data = NULL;
        return;
    }

    for (i = 0; i < 8; i++)
        msglen += sections_len[i];
    msglen += 4;
    if (*len < msglen)
        msglen = *len;

    *data = (unsigned char*)grib_context_malloc(context, msglen);
    p     = (unsigned char*)*data;

    for (i = 0; i < 8; i++) {
        if (sections[i]) {
            memcpy(p, sections[i], sections_len[i]);
            p += sections_len[i];
        }
    }

    memcpy(p, theEnd, 4);

    grib_encode_unsigned_long((unsigned char*)*data, msglen, &bitp, 64);

    *len = msglen;
}

static int grib2_get_next_section(unsigned char* msgbegin, size_t msglen, unsigned char** secbegin, size_t* seclen, int* secnum, int* err)
{
    if (!grib2_has_next_section(msgbegin, msglen, *secbegin, *seclen, err))
        return 0;

    *secbegin += *seclen;
    *seclen = grib_decode_unsigned_byte_long(*secbegin, 0, 4);
    *secnum = grib_decode_unsigned_byte_long(*secbegin, 4, 1);

    if (*secnum < 1 || *secnum > 7) {
        *err = GRIB_INVALID_SECTION_NUMBER;
        return 0;
    }
    return 1;
}

static int grib2_has_next_section(unsigned char* msgbegin, size_t msglen, unsigned char* secbegin, size_t seclen, int* err)
{
    long next_seclen;
    *err = 0;

    next_seclen = (msgbegin + msglen) - (secbegin + seclen);

    if (next_seclen < 5) {
        if ((next_seclen > 3) && !strncmp((char*)secbegin, "7777", 4))
            *err = GRIB_SUCCESS;
        else
            *err = GRIB_7777_NOT_FOUND;
        return 0;
    }

    /*secbegin += seclen;*/

    return 1;
}

void grib_accessors_list_delete(grib_context* c, grib_accessors_list* al)
{
    grib_accessors_list* tmp;

    while (al) {
        tmp = al->next;
        /*grib_accessor_delete(c, al->accessor);*/
        grib_context_free(c, al);
        al = tmp;
    }
}

void grib_accessors_list_push(grib_accessors_list* al, grib_accessor* a, int rank)
{
    grib_accessors_list* last;
    grib_context* c = a->context;

    last = grib_accessors_list_last(al);
    if (last && last->accessor) {
        last->next           = (grib_accessors_list*)grib_context_malloc_clear(c, sizeof(grib_accessors_list));
        last->next->accessor = a;
        last->next->prev     = last;
        last->next->rank     = rank;
        al->last             = last->next;
    }
    else {
        al->accessor = a;
        al->rank     = rank;
        al->last     = al;
    }
}

int grib_accessors_list_unpack_double(grib_accessors_list* al, double* val, size_t* buffer_len)
{
    int err             = GRIB_SUCCESS;
    size_t unpacked_len = 0;
    size_t len          = 0;

    while (al && err == GRIB_SUCCESS) {
        len = *buffer_len - unpacked_len;
        err = grib_unpack_double(al->accessor, val + unpacked_len, &len);
        unpacked_len += len;
        al = al->next;
    }

    *buffer_len = unpacked_len;
    return err;
}

int grib_accessors_list_value_count(grib_accessors_list* al, size_t* count)
{
    long lcount = 0;
    *count      = 0;
    while (al) {
        grib_value_count(al->accessor, &lcount);
        *count += lcount;
        al = al->next;
    }
    return 0;
}

void grib_action_delete(grib_context* context, grib_action* a)
{
    grib_action_class* c = a->cclass;
    init(c);
    while (c) {
        if (c->destroy_gac)
            c->destroy_gac(context, a);
        c = c->super ? *(c->super) : NULL;
    }
    grib_context_free_persistent(context, a);
}

const char* grib_arguments_get_name(grib_handle* h, grib_arguments* args, int n)
{
    grib_expression* e = NULL;
    while (args && n-- > 0) {
        args = args->next;
    }

    if (!args)
        return NULL;

    e = args->expression;
    return e ? grib_expression_get_name(e) : NULL;
}

void grib_buffer_delete(const grib_context* c, grib_buffer* b)
{
    if (b->property == GRIB_MY_BUFFER)
        grib_context_free(c, b->data);
    b->length  = 0;
    b->ulength = 0;
    grib_context_free(c, b);
}

void grib_context_set_handle_total_count(grib_context* c, int new_count)
{
    if (!c)
        c = grib_context_get_default();
    GRIB_MUTEX_INIT_ONCE(&once, &init);
    GRIB_MUTEX_LOCK(&mutex_c);
    c->handle_total_count = new_count;
    GRIB_MUTEX_UNLOCK(&mutex_c);
}

void grib_get_buffer_ownership(const grib_context* c, grib_buffer* b)
{
    unsigned char* newdata;
    if (b->property == GRIB_MY_BUFFER)
        return;

    newdata = (unsigned char*)grib_context_malloc(c, b->length);
    memcpy(newdata, b->data, b->length);
    b->data     = newdata;
    b->property = GRIB_MY_BUFFER;
}

int grib_get_string(const grib_handle* h, const char* name, char* val, size_t* length)
{
    grib_accessor* a        = NULL;
    grib_accessors_list* al = NULL;
    int ret                 = 0;

    if (name[0] == '/') {
        al = grib_find_accessors_list(h, name);
        if (!al)
            return GRIB_NOT_FOUND;
        ret = grib_unpack_string(al->accessor, val, length);
        grib_context_free(h->context, al);
        return ret;
    }
    else {
        a = grib_find_accessor(h, name);
        if (!a)
            return GRIB_NOT_FOUND;
        return grib_unpack_string(a, val, length);
    }
}

static void grib_grow_buffer_to(const grib_context* c, grib_buffer* b, size_t ns)
{
    unsigned char* newdata;

    if (ns > b->length) {
        grib_get_buffer_ownership(c, b);
        newdata = (unsigned char*)grib_context_malloc_clear(c, ns);
        memcpy(newdata, b->data, b->length);
        grib_context_free(c, b->data);
        b->data   = newdata;
        b->length = ns;
    }
}

void grib_grow_buffer(const grib_context* c, grib_buffer* b, size_t new_size)
{
    if (new_size > b->length) {
        size_t len = 0;
        size_t inc = b->length > 2048 ? b->length : 2048;
        len        = ((new_size + 2 * inc) / 1024) * 1024;
        grib_grow_buffer_to(c, b, len);
    }
}

void grib_accessor_delete(grib_context* ct, grib_accessor* a)
{
    grib_accessor_class* c = a->cclass;
    while (c) {
        grib_accessor_class* s = c->super ? *(c->super) : NULL;
        /*printf("grib_accessor_delete: before destroy a=%p c->name=%s ==> a->name=%s\n", (void*)a, c->name, a->name);*/
        if (c->destroy) {
            c->destroy(ct, a);
        }
        c = s;
    }
    /*printf("grib_accessor_delete before free a=%p\n", (void*)a);*/
    grib_context_free(ct, a);
}

int grib_handle_delete(grib_handle* h)
{
    if (h != NULL) {
        grib_context* ct   = h->context;
        grib_dependency* d = h->dependencies;
        grib_dependency* n;

        if (h->kid != NULL)
            return GRIB_INTERNAL_ERROR;

        while (d) {
            n = d->next;
            grib_context_free(ct, d);
            d = n;
        }
        h->dependencies = 0;

        grib_buffer_delete(ct, h->buffer);
        grib_section_delete(ct, h->root);
        grib_context_free(ct, h->gts_header);

        grib_context_log(ct, GRIB_LOG_DEBUG, "grib_handle_delete: deleting handle %p", (void*)h);
        grib_context_free(ct, h);
        h = NULL;
    }
    return GRIB_SUCCESS;
}

void grib_context_free_persistent(const grib_context* c, void* p)
{
    if (!c)
        c = grib_context_get_default();
    if (p)
        c->free_persistent_mem(c, p);
}

void grib_context_increment_handle_file_count(grib_context* c)
{
    if (!c)
        c = grib_context_get_default();
    GRIB_MUTEX_INIT_ONCE(&once, &init);
    GRIB_MUTEX_LOCK(&mutex_c);
    c->handle_file_count++;
    GRIB_MUTEX_UNLOCK(&mutex_c);
}

void grib_context_log(const grib_context* c, int level, const char* fmt, ...)
{
    /* Save some CPU */
    if ((level == GRIB_LOG_DEBUG && c->debug < 1) ||
        (level == GRIB_LOG_WARNING && c->debug < 2)) {
        return;
    }
    else {
        char msg[1024];
        va_list list;
        const int errsv = errno;

        va_start(list, fmt);
        vsprintf(msg, fmt, list);
        va_end(list);

        if (level & GRIB_LOG_PERROR) {
            level = level & ~GRIB_LOG_PERROR;

            /* #if HAS_STRERROR */
    #if 1
            strcat(msg, " (");
            strcat(msg, strerror(errsv));
            strcat(msg, ")");
    #else
            if (errsv > 0 && errsv < sys_nerr) {
                strcat(msg, " (");
                strcat(msg, sys_errlist[errsv]);
                strcat(msg, " )");
            }
    #endif
        }

        if (c->output_log)
            c->output_log(c, level, msg);
    }
}

char* grib_context_strdup(const grib_context* c, const char* s)
{
    char* dup = 0;
    if (s) {
        dup = (char*)grib_context_malloc(c, (strlen(s) * sizeof(char)) + 1);
        if (dup)
            strcpy(dup, s);
    }
    return dup;
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

off_t grib_context_tell(const grib_context* c, void* stream)
{
    if (!c)
        c = grib_context_get_default();
    return c->tell(c, stream);
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

void grib_context_free(const grib_context* c, void* p)
{
    if (!c)
        c = grib_context_get_default();
    if (p)
        c->free_mem(c, p);
}

void* grib_context_malloc(const grib_context* c, size_t size)
{
    void* p = NULL;
    if (!c)
        c = grib_context_get_default();
    if (size == 0)
        return p;
    else
        p = c->alloc_mem(c, size);
    if (!p) {
        grib_context_log(c, GRIB_LOG_FATAL, "grib_context_malloc: error allocating %lu bytes", (unsigned long)size);
        Assert(0);
    }
    return p;
}

void* grib_context_malloc_clear(const grib_context* c, size_t size)
{
    void* p = grib_context_malloc(c, size);
    if (p)
        memset(p, 0, size);
    return p;
}

void* grib_context_malloc_clear_persistent(const grib_context* c, size_t size)
{
    void* p = grib_context_malloc_persistent(c, size);
    if (p)
        memset(p, 0, size);
    return p;
}

void* grib_context_malloc_persistent(const grib_context* c, size_t size)
{
    void* p = c->alloc_persistent_mem(c, size);
    if (!p) {
        grib_context_log(c, GRIB_LOG_FATAL,
                         "grib_context_malloc_persistent: error allocating %lu bytes", (unsigned long)size);
        Assert(0);
    }
    return p;
}

char* grib_context_strdup_persistent(const grib_context* c, const char* s)
{
    char* dup = (char*)grib_context_malloc_persistent(c, (strlen(s) * sizeof(char)) + 1);
    if (dup)
        strcpy(dup, s);
    return dup;
}

int grib_create_accessor(grib_section* p, grib_action* a, grib_loader* h)
{
    grib_action_class* c = a->cclass;
    init(c);
    while (c) {
        if (c->create_accessor) {
            int ret;
            /* ECC-604: Do not lock excessively */
            /*GRIB_MUTEX_INIT_ONCE(&once,&init_mutex);*/
            /*GRIB_MUTEX_LOCK(&mutex1);*/
            ret = c->create_accessor(p, a, h);
            /*GRIB_MUTEX_UNLOCK(&mutex1);*/
            return ret;
        }
        c = c->super ? *(c->super) : NULL;
    }
    fprintf(stderr, "Cannot create accessor %s %s\n", a->name, a->cclass->name);
    DebugAssert(0);
    return 0;
}

void grib_empty_section(grib_context* c, grib_section* b)
{
    grib_accessor* current = NULL;
    if (!b)
        return;

    b->aclength = NULL;

    current = b->block->first;

    while (current) {
        grib_accessor* next = current->next;
        if (current->sub_section) {
            grib_section_delete(c, current->sub_section);
            current->sub_section = 0;
        }
        grib_accessor_delete(c, current);
        current = next;
    }
    b->block->first = b->block->last = 0;
}

const char* grib_expression_get_name(grib_expression* g)
{
    grib_expression_class* c = g->cclass;
    while (c) {
        if (c->get_name)
            return c->get_name(g);
        c = c->super ? *(c->super) : NULL;
    }
    if (g->cclass) printf("No expression_get_name() in %s\n", g->cclass->name);
    Assert(1 == 0);
    return 0;
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

int _grib_get_double_array_internal(const grib_handle* h, grib_accessor* a, double* val, size_t buffer_len, size_t* decoded_length)
{
    if (a) {
        int err = _grib_get_double_array_internal(h, a->same, val, buffer_len, decoded_length);

        if (err == GRIB_SUCCESS) {
            size_t len = buffer_len - *decoded_length;
            err        = grib_unpack_double(a, val + *decoded_length, &len);
            *decoded_length += len;
        }

        return err;
    }
    else {
        return GRIB_SUCCESS;
    }
}

int grib_get_double_array_internal(const grib_handle* h, const char* name, double* val, size_t* length)
{
    int ret = grib_get_double_array(h, name, val, length);

    if (ret != GRIB_SUCCESS)
        grib_context_log(h->context, GRIB_LOG_ERROR,
                         "unable to get %s as double array (%s)",
                         name, grib_get_error_message(ret));

    return ret;
}

int grib_get_double_array(const grib_handle* h, const char* name, double* val, size_t* length)
{
    size_t len              = *length;
    grib_accessor* a        = NULL;
    grib_accessors_list* al = NULL;
    int ret                 = 0;

    if (name[0] == '/') {
        al = grib_find_accessors_list(h, name);
        if (!al)
            return GRIB_NOT_FOUND;
        ret = grib_accessors_list_unpack_double(al, val, length);
        grib_accessors_list_delete(h->context, al);
        return ret;
    }
    else {
        a = grib_find_accessor(h, name);
        if (!a)
            return GRIB_NOT_FOUND;
        if (name[0] == '#') {
            return grib_unpack_double(a, val, length);
        }
        else {
            *length = 0;
            return _grib_get_double_array_internal(h, a, val, len, length);
        }
    }
}

const char* grib_get_error_message(int code)
{
    code = -code;
    if (code < 0 || code >= NUMBER(errors)) {
        static char mess[64];
        sprintf(mess,"Unknown error %d",code);
        return mess;
    }
    return errors[code];
}

int grib_get_length(const grib_handle* h, const char* name, size_t* length)
{
    return grib_get_string_length(h, name, length);
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

int grib_get_long_internal(grib_handle* h, const char* name, long* val)
{
    int ret = grib_get_long(h, name, val);

    if (ret != GRIB_SUCCESS) {
        grib_context_log(h->context, GRIB_LOG_ERROR,
                         "unable to get %s as long (%s)",
                         name, grib_get_error_message(ret));
    }

    return ret;
}

int _grib_get_string_length(grib_accessor* a, size_t* size)
{
    size_t s = 0;

    *size = 0;
    while (a) {
        s = grib_string_length(a);
        if (s > *size)
            *size = s;
        a = a->same;
    }
    (*size) += 1;

    return GRIB_SUCCESS;
}

int grib_get_string_length(const grib_handle* h, const char* name, size_t* size)
{
    grib_accessor* a        = NULL;
    grib_accessors_list* al = NULL;
    int ret                 = 0;

    if (name[0] == '/') {
        al = grib_find_accessors_list(h, name);
        if (!al)
            return GRIB_NOT_FOUND;
        ret = _grib_get_string_length(al->accessor, size);
        grib_context_free(h->context, al);
        return ret;
    }
    else {
        a = grib_find_accessor(h, name);
        if (!a)
            return GRIB_NOT_FOUND;
        return _grib_get_string_length(a, size);
    }
}

char* grib_context_full_defs_path(grib_context* c, const char* basename)
{
    int err         = 0;
    char full[1024] = {0,};
    grib_string_list* dir      = NULL;
    grib_string_list* fullpath = 0;
    if (!c)
        c = grib_context_get_default();

    GRIB_MUTEX_INIT_ONCE(&once, &init);

    if (*basename == '/' || *basename == '.') {
        return (char*)basename;
    }
    else {
        GRIB_MUTEX_LOCK(&mutex_c); /* See ECC-604 */
        fullpath = (grib_string_list*)grib_trie_get(c->def_files, basename);
        GRIB_MUTEX_UNLOCK(&mutex_c);
        if (fullpath != NULL) {
            return fullpath->value;
        }
        if (!c->grib_definition_files_dir) {
            err = init_definition_files_dir(c);
        }

        if (err != GRIB_SUCCESS) {
            grib_context_log(c, GRIB_LOG_ERROR,
                             "Unable to find definition files directory");
            return NULL;
        }

        dir = c->grib_definition_files_dir;

        while (dir) {
            sprintf(full, "%s/%s", dir->value, basename);
            if (!codes_access(full, F_OK)) {
                fullpath = (grib_string_list*)grib_context_malloc_clear_persistent(c, sizeof(grib_string_list));
                Assert(fullpath);
                fullpath->value = grib_context_strdup(c, full);
                GRIB_MUTEX_LOCK(&mutex_c);
                grib_trie_insert(c->def_files, basename, fullpath);
                grib_context_log(c, GRIB_LOG_DEBUG, "Found def file %s", full);
                GRIB_MUTEX_UNLOCK(&mutex_c);
                return fullpath->value;
            }
            dir = dir->next;
        }
    }

    GRIB_MUTEX_LOCK(&mutex_c);
    /* Store missing files so we don't check for them again and again */
    grib_string_list grib_file_not_found;
    grib_trie_insert(c->def_files, basename, &grib_file_not_found);
    /*grib_context_log(c,GRIB_LOG_ERROR,"Def file \"%s\" not found",basename);*/
    GRIB_MUTEX_UNLOCK(&mutex_c);
    full[0] = 0;
    return NULL;
}

void grib_context_increment_handle_total_count(grib_context* c)
{
    if (!c)
        c = grib_context_get_default();
    GRIB_MUTEX_INIT_ONCE(&once, &init);
    GRIB_MUTEX_LOCK(&mutex_c);
    c->handle_total_count++;
    GRIB_MUTEX_UNLOCK(&mutex_c);
}

size_t grib_context_read(const grib_context* c, void* ptr, size_t size, void* stream)
{
    if (!c)
        c = grib_context_get_default();
    return c->read(c, ptr, size, stream);
}

int grib_context_seek(const grib_context* c, off_t offset, int whence, void* stream)
{
    if (!c)
        c = grib_context_get_default();
    return c->seek(c, offset, whence, stream);
}

unsigned long grib_decode_unsigned_byte_long(const unsigned char* p, long o, int l)
{
    long accum      = 0;
    int i           = 0;
    unsigned char b = p[o++];

    Assert(l <= max_nbits);

    accum <<= 8;
    accum |= b;

    for (i = 1; i < l; i++) {
        b = p[o++];
        accum <<= 8;
        accum |= b;
    }
    return accum;
}

int grib_encode_unsigned_long(unsigned char* p, unsigned long val, long* bitp, long nbits)
{
    long len          = nbits;
    int s             = *bitp % 8;
    int n             = 8 - s;
    unsigned char tmp = 0; /*for temporary results*/

    if (nbits > max_nbits) {
        /* TODO: Do some real code here, to support long longs */
        int bits  = nbits;
        int mod   = bits % max_nbits;
        long zero = 0;

        if (mod != 0) {
            int e = grib_encode_unsigned_long(p, zero, bitp, mod);
            /* printf(" -> : encoding %ld bits=%ld %ld\n",zero,(long)mod,*bitp); */
            Assert(e == 0);
            bits -= mod;
        }

        while (bits > max_nbits) {
            int e = grib_encode_unsigned_long(p, zero, bitp, max_nbits);
            /* printf(" -> : encoding %ld bits=%ld %ld\n",zero,(long)max_nbits,*bitp); */
            Assert(e == 0);
            bits -= max_nbits;
        }

        /* printf(" -> : encoding %ld bits=%ld %ld\n",val,(long)bits,*bitp); */
        return grib_encode_unsigned_long(p, val, bitp, bits);
    }

    p += (*bitp >> 3); /* skip the bytes */

    /* head */
    if (s) {
        len -= n;
        if (len < 0) {
            tmp = ((val << -len) | ((*p) & dmasks[n]));
        }
        else {
            tmp = ((val >> len) | ((*p) & dmasks[n]));
        }
        *p++ = tmp;
    }

    /*  write the middle words */
    while (len >= 8) {
        len -= 8;
        *p++ = (val >> len);
    }

    /*  write the end bits */
    if (len)
        *p = (val << (8 - len));

    *bitp += nbits;
    return GRIB_SUCCESS;
}

static void grib_find_same_and_push(grib_accessors_list* al, grib_accessor* a)
{
    if (a) {
        grib_find_same_and_push(al, a->same);
        grib_accessors_list_push(al, a, al->rank);
    }
}

int grib_hash_keys_get_id(grib_itrie* t, const char* key)
{
    const struct grib_keys_hash* hash = grib_keys_hash_get(key, strlen(key));

    if (hash) {
        /* printf("%s found %s (%d)\n",key,hash->name,hash->id); */
        return hash->id;
    }

    /* printf("+++ \"%s\"\n",key); */
    {
        const char* k    = key;
        grib_itrie* last = t;

        GRIB_MUTEX_INIT_ONCE(&once, &init);
        GRIB_MUTEX_LOCK(&mutex);

        while (*k && t)
            t = t->next[mapping[(int)*k++]];

        if (t != NULL && t->id != -1) {
            GRIB_MUTEX_UNLOCK(&mutex);
            return t->id + TOTAL_KEYWORDS + 1;
        }
        else {
            int ret = grib_hash_keys_insert(last, key);
            GRIB_MUTEX_UNLOCK(&mutex);
            return ret + TOTAL_KEYWORDS + 1;
        }
    }
}

static int grib_hash_keys_insert(grib_itrie* t, const char* key)
{
    const char* k    = key;
    grib_itrie* last = t;
    int* count;

    GRIB_MUTEX_INIT_ONCE(&once, &init);
    GRIB_MUTEX_LOCK(&mutex);

    Assert(t);
    if (!t) return -1;

    count = t->count;

    while (*k && t) {
        last = t;
        t    = t->next[mapping[(int)*k]];
        if (t)
            k++;
    }

    if (*k != 0) {
        t = last;
        while (*k) {
            int j      = mapping[(int)*k++];
            t->next[j] = grib_hash_keys_new(t->context, count);
            t          = t->next[j];
        }
    }
    if (*(t->count) + TOTAL_KEYWORDS < ACCESSORS_ARRAY_SIZE) {
        t->id = *(t->count);
        (*(t->count))++;
    }
    else {
        grib_context_log(t->context, GRIB_LOG_ERROR,
                         "grib_hash_keys_insert: too many accessors, increase ACCESSORS_ARRAY_SIZE\n");
        Assert(*(t->count) + TOTAL_KEYWORDS < ACCESSORS_ARRAY_SIZE);
    }

    GRIB_MUTEX_UNLOCK(&mutex);

    /*printf("grib_hash_keys_get_id: %s -> %d\n",key,t->id);*/

    return t->id;
}

int  grib_is_defined(const grib_handle* h, const char* name)
{
    grib_accessor* a = grib_find_accessor(h, name);
    return (a ? 1 : 0);
}

int grib_iterator_delete(grib_iterator* i)
{
    if (i) {
        grib_iterator_class* c = i->cclass;
        while (c) {
            grib_iterator_class* s = c->super ? *(c->super) : NULL;
            if (c->destroy)
                c->destroy(i);
            c = s;
        }
        /* This should go in a top class */
        grib_context_free(i->h->context, i);
    }
    else {
        return GRIB_INVALID_ARGUMENT;
    }
    return 0;
}

int grib_iterator_init(grib_iterator* i, grib_handle* h, grib_arguments* args)
{
    int r = 0;
    GRIB_MUTEX_INIT_ONCE(&once, &init_mutex);
    GRIB_MUTEX_LOCK(&mutex);
    r = init_iterator(i->cclass, i, h, args);
    GRIB_MUTEX_UNLOCK(&mutex);
    return r;
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

long grib_byte_offset(grib_accessor* a)
{
    grib_accessor_class* c = NULL;
    if (a)
        c = a->cclass;

    while (c) {
        if (c->byte_offset)
            return c->byte_offset(a);
        c = c->super ? *(c->super) : NULL;
    }
    DebugAssert(0);
    return 0;
}

int _grib_get_size(const grib_handle* h, grib_accessor* a, size_t* size)
{
    long count = 0;
    int err    = 0;

    if (!a)
        return GRIB_NOT_FOUND;

    *size = 0;
    while (a) {
        if (err == 0) {
            err = grib_value_count(a, &count);
            if (err)
                return err;
            *size += count;
        }
        a = a->same;
    }
    return GRIB_SUCCESS;
}

int grib_get_size(const grib_handle* ch, const char* name, size_t* size)
{
    grib_handle* h          = (grib_handle*)ch;
    grib_accessor* a        = NULL;
    grib_accessors_list* al = NULL;
    int ret                 = 0;
    *size                   = 0;

    if (name[0] == '/') {
        al = grib_find_accessors_list(h, name);
        if (!al)
            return GRIB_NOT_FOUND;
        ret = grib_accessors_list_value_count(al, size);
        grib_accessors_list_delete(h->context, al);
        return ret;
    }
    else {
        a = grib_find_accessor(h, name);
        if (!a)
            return GRIB_NOT_FOUND;
        if (name[0] == '#') {
            long count = *size;
            ret        = grib_value_count(a, &count);
            *size      = count;
            return ret;
        }
        else
            return _grib_get_size(h, a, size);
    }
}

void* grib_oarray_get(grib_oarray* v, int i)
{
    if (v == NULL || i > v->n - 1)
        return NULL;
    return v->v[i];
}

int grib_pack_long(grib_accessor* a, const long* v, size_t* len)
{
    grib_accessor_class* c = a->cclass;
    /*grib_context_log(a->context, GRIB_LOG_DEBUG, "(%s)%s is packing (long) %d",(a->parent->owner)?(a->parent->owner->name):"root", a->name ,v?(*v):0); */
    while (c) {
        if (c->pack_long) {
            return c->pack_long(a, v, len);
        }
        c = c->super ? *(c->super) : NULL;
    }
    DebugAssert(0);
    return 0;
}

static void grib_push_action_file(grib_action_file* af, grib_action_file_list* afl)
{
    if (!afl->first)
        afl->first = afl->last = af;
    else
        afl->last->next = af;
    afl->last = af;
}

int grib_unpack_double(grib_accessor* a, double* v, size_t* len)
{
    grib_accessor_class* c = a->cclass;
    /*grib_context_log(a->context, GRIB_LOG_DEBUG, "(%s)%s is unpacking (double)",(a->parent->owner)?(a->parent->owner->name):"root", a->name ); */
    while (c) {
        if (c->unpack_double) {
            return c->unpack_double(a, v, len);
        }
        c = c->super ? *(c->super) : NULL;
    }
    DebugAssert(0);
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

int grib_unpack_string(grib_accessor* a, char* v, size_t* len)
{
    grib_accessor_class* c = a->cclass;
    /* grib_context_log(a->context, GRIB_LOG_DEBUG, "(%s)%s is unpacking (string)",(a->parent->owner)?(a->parent->owner->name):"root", a->name ); */
    while (c) {
        if (c->unpack_string) {
            return c->unpack_string(a, v, len);
        }
        c = c->super ? *(c->super) : NULL;
    }
    DebugAssert(0);
    return 0;
}

void* grib_trie_get(grib_trie* t, const char* key)
{
    const char* k = key;
    GRIB_MUTEX_INIT_ONCE(&once, &init);
    GRIB_MUTEX_LOCK(&mutex);

    while (*k && t) {
        DebugCheckBounds((int)*k, key);
        t = t->next[mapping[(int)*k++]];
    }

    if (*k == 0 && t != NULL && t->data != NULL) {
        GRIB_MUTEX_UNLOCK(&mutex);
        return t->data;
    }
    GRIB_MUTEX_UNLOCK(&mutex);
    return NULL;
}

void* grib_trie_insert(grib_trie* t, const char* key, void* data)
{
    grib_trie* last = t;
    const char* k   = key;
    void* old       = NULL;

    if (!t) {
        Assert(!"grib_trie_insert: grib_trie==NULL");
        return NULL;
    }

    GRIB_MUTEX_INIT_ONCE(&once, &init);
    GRIB_MUTEX_LOCK(&mutex);

    while (*k && t) {
        last = t;
        DebugCheckBounds((int)*k, key);
        t = t->next[mapping[(int)*k]];
        if (t)
            k++;
    }

    if (*k == 0) {
        old     = t->data;
        t->data = data;
    }
    else {
        t = last;
        while (*k) {
            int j = 0;
            DebugCheckBounds((int)*k, key);
            j = mapping[(int)*k++];
            if (j < t->first)
                t->first = j;
            if (j > t->last)
                t->last = j;
            t = t->next[j] = grib_trie_new(t->context);
        }
        old     = t->data;
        t->data = data;
    }
    GRIB_MUTEX_UNLOCK(&mutex);
    return data == old ? NULL : old;
}

void* grib_trie_with_rank_get(grib_trie_with_rank* t, const char* key, int rank)
{
    const char* k = key;
    void* data;
    GRIB_MUTEX_INIT_ONCE(&once, &init);

    if (rank < 0)
        return NULL;

    GRIB_MUTEX_LOCK(&mutex);

    while (*k && t) {
        DebugCheckBounds((int)*k, key);
        t = t->next[mapping[(int)*k++]];
    }

    if (*k == 0 && t != NULL) {
        data = grib_oarray_get(t->objs, rank - 1);
        GRIB_MUTEX_UNLOCK(&mutex);
        return data;
    }
    GRIB_MUTEX_UNLOCK(&mutex);
    return NULL;
}

int grib_section_adjust_sizes(grib_section* s, int update, int depth)
{
    int err          = 0;
    grib_accessor* a = s ? s->block->first : NULL;
    size_t length    = update ? 0 : (s ? s->padding : 0);
    size_t offset    = (s && s->owner) ? s->owner->offset : 0;
    int force_update = update > 1;

    while (a) {
        register long l;
        /* grib_section_adjust_sizes(grib_get_sub_section(a),update,depth+1); */
        err = grib_section_adjust_sizes(a->sub_section, update, depth + 1);
        if (err)
            return err;
        /*grib_context_log(a->context,GRIB_LOG_DEBUG,"grib_section_adjust_sizes: %s %ld [len=%ld] (depth=%d)",a->name,(long)a->offset,(long)a->length,depth);*/

        l = a->length;

        if (offset != a->offset) {
            grib_context_log(a->context, GRIB_LOG_ERROR,
                             "Offset mismatch %s A->offset %ld offset %ld\n", a->name, (long)a->offset, (long)offset);
            a->offset = offset;
            return GRIB_DECODING_ERROR;
        }
        length += l;
        offset += l;
        a = a->next;
    }

    if (s) {
        if (s->aclength) {
            size_t len = 1;
            long plen  = 0;
            int lret   = grib_unpack_long(s->aclength, &plen, &len);
            Assert(lret == GRIB_SUCCESS);
            /* This happens when there is some padding */
            if ((plen != length) || force_update) {
                if (update) {
                    plen = length;
                    lret = grib_pack_long(s->aclength, &plen, &len);
                    Assert(lret == GRIB_SUCCESS);
                    s->padding = 0;
                }
                else {
                    if (!s->h->partial) {
                        if (length >= plen) {
                            if (s->owner) {
                                grib_context_log(s->h->context, GRIB_LOG_ERROR, "Invalid size %ld found for %s, assuming %ld",
                                             (long)plen, s->owner->name, (long)length);
                            }
                            plen = length;
                        }
                        s->padding = plen - length;
                    }
                    length = plen;
                }
            }
        }

        if (s->owner) {
            /*grib_context_log(s->owner->context,GRIB_LOG_DEBUG,"grib_section_adjust_sizes: updating owner (%s->length old=%ld new=%ld)",s->owner->name,(long)s->owner->length,(long)length);*/
            s->owner->length = length;
        }
        s->length = length;
    }
    return err;
}

void grib_section_delete(grib_context* c, grib_section* b)
{
    if (!b)
        return;

    grib_empty_section(c, b);
    grib_context_free(c, b->block);
    /* printf("++++ deleted %p\n",b); */
    grib_context_free(c, b);
}

void grib_section_post_init(grib_section* s)
{
    grib_accessor* a = s ? s->block->first : NULL;

    while (a) {
        grib_accessor_class* c = a->cclass;
        if (c->post_init)
            c->post_init(a);
        if (a->sub_section)
            grib_section_post_init(a->sub_section);
        a = a->next;
    }
}

char* grib_split_name_attribute(grib_context* c, const char* name, char* attribute_name)
{
    /*returns accessor name and attribute*/
    size_t size         = 0;
    char* accessor_name = NULL;
    char* p             = strstr((char*)name, "->");
    if (!p) {
        *attribute_name = 0;
        return (char*)name;
    }
    size          = p - name;
    accessor_name = (char*)grib_context_malloc_clear(c, size + 1);
    accessor_name = (char*)memcpy(accessor_name, name, size);
    p += 2;
    strcpy(attribute_name, p);
    return accessor_name;
}

long grib_string_length(grib_accessor* a)
{
    grib_accessor_class* c = NULL;
    if (a)
        c = a->cclass;

    while (c) {
        if (c->string_length)
            return c->string_length(a);
        c = c->super ? *(c->super) : NULL;
    }
    DebugAssert(0);
    return 0;
}

int grib_value_count(grib_accessor* a, long* count)
{
    grib_accessor_class* c = NULL;
    int err                = 0;
    if (a)
        c = a->cclass;

    while (c) {
        if (c->value_count) {
            err = c->value_count(a, count);
            return err;
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

void default_free(const grib_context* c, void* p)
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

static void* allocate_buffer(void* data, size_t* length, int* err)
{
    alloc_buffer* u = (alloc_buffer*)data;
    u->buffer       = malloc(*length);
    u->size         = *length;
    if (u->buffer == NULL)
        *err = GRIB_OUT_OF_MEMORY; /* Cannot allocate buffer */
    return u->buffer;
}

static int matching(grib_accessor* a, const char* name, const char* name_space)
{
    int i = 0;
    while (i < MAX_ACCESSOR_NAMES) {
        if (a->all_names[i] == 0)
            return 0;

        if ((grib_inline_strcmp(name, a->all_names[i]) == 0) &&
            ((name_space == NULL) || (a->all_name_spaces[i] != NULL &&
                                      grib_inline_strcmp(a->all_name_spaces[i], name_space) == 0)))
            return 1;
        i++;
    }
    return 0;
}

static void init(grib_action_class* c)
{
    if (!c)
        return;

    GRIB_MUTEX_INIT_ONCE(&once, &init_mutex);
    GRIB_MUTEX_LOCK(&mutex1);
    if (!c->inited) {
        if (c->super) {
            init(*(c->super));
        }
        c->init_class_gac(c);
        c->inited = 1;
    }
    GRIB_MUTEX_UNLOCK(&mutex1);
}

static void init_class_gac (grib_action_class* c) {

}

static void dump_gac (grib_action* act, FILE*f, int lvl) {

}

static void destroy_gac (grib_context* context, grib_action* act)
{
    grib_context_free_persistent(context, act->name);
    grib_context_free_persistent(context, act->op);
}

static void xref_gac (grib_action* d, FILE* f, const char* path)
{
}

static int execute_gac (grib_action* act, grib_handle* h)
{
    return 0;
}

static int init_2(grib_iterator* iter, grib_handle* h, grib_arguments* args)
{
    grib_iterator_gen* self = (grib_iterator_gen*)iter;
    size_t dli              = 0;
    int err                 = GRIB_SUCCESS;
    const char* s_rawData   = NULL;
    const char* s_numPoints = NULL;
    long numberOfPoints     = 0;
    self->carg              = 1;

    s_numPoints        = grib_arguments_get_name(h, args, self->carg++);
    self->missingValue = grib_arguments_get_name(h, args, self->carg++);
    s_rawData          = grib_arguments_get_name(h, args, self->carg++);

    iter->h    = h; /* We may not need to keep them */
    iter->args = args;
    if ((err = grib_get_size(h, s_rawData, &dli)) != GRIB_SUCCESS)
        return err;

    if ((err = grib_get_long_internal(h, s_numPoints, &numberOfPoints)) != GRIB_SUCCESS)
        return err;

    if (numberOfPoints != dli) {
        grib_context_log(h->context, GRIB_LOG_ERROR, "Geoiterator: %s != size(%s) (%ld!=%ld)",
                         s_numPoints, s_rawData, numberOfPoints, dli);
        return GRIB_WRONG_GRID;
    }
    iter->nv = dli;
    if (iter->nv == 0) {
        grib_context_log(h->context, GRIB_LOG_ERROR, "Geoiterator: size(%s) is %ld", s_rawData, dli);
        return GRIB_WRONG_GRID;
    }
    iter->data = (double*)grib_context_malloc(h->context, (iter->nv) * sizeof(double));

    if ((err = grib_get_double_array_internal(h, s_rawData, iter->data, &(iter->nv))))
        return err;

    iter->e = -1;

    return err;
}

static int init_definition_files_dir(grib_context* c)
{
    int err = 0;
    char path[ECC_PATH_MAXLEN];
    char* p                = NULL;
    grib_string_list* next = NULL;

    if (!c)
        c = grib_context_get_default();

    if (c->grib_definition_files_dir)
        return 0;
    if (!c->grib_definition_files_path)
        return GRIB_NO_DEFINITIONS;

    /* Note: strtok modifies its first argument so we copy */
    strncpy(path, c->grib_definition_files_path, ECC_PATH_MAXLEN-1);

    GRIB_MUTEX_INIT_ONCE(&once, &init);
    GRIB_MUTEX_LOCK(&mutex_c);

    p = path;

    while (*p != ECC_PATH_DELIMITER_CHAR && *p != '\0')
        p++;

    if (*p != ECC_PATH_DELIMITER_CHAR) {
        /* No delimiter found so this is a single directory */
        c->grib_definition_files_dir        = (grib_string_list*)grib_context_malloc_clear_persistent(c, sizeof(grib_string_list));
        c->grib_definition_files_dir->value = codes_resolve_path(c, path);
    }
    else {
        /* Definitions path contains multiple directories */
        char* dir = NULL;
        dir       = strtok(path, ECC_PATH_DELIMITER_STR);

        while (dir != NULL) {
            if (next) {
                next->next = (grib_string_list*)grib_context_malloc_clear_persistent(c, sizeof(grib_string_list));
                next       = next->next;
            }
            else {
                c->grib_definition_files_dir = (grib_string_list*)grib_context_malloc_clear_persistent(c, sizeof(grib_string_list));
                next                         = c->grib_definition_files_dir;
            }
            next->value = codes_resolve_path(c, dir);
            dir         = strtok(NULL, ECC_PATH_DELIMITER_STR);
        }
    }

    GRIB_MUTEX_UNLOCK(&mutex_c);

    return err;
}

static int init_iterator(grib_iterator_class* c, grib_iterator* i, grib_handle* h, grib_arguments* args)
{
    if (c) {
        int ret                = GRIB_SUCCESS;
        grib_iterator_class* s = c->super ? *(c->super) : NULL;
        if (!c->inited) {
            if (c->init_class)
                c->init_class(c);
            c->inited = 1;
        }
        if (s)
            ret = init_iterator(s, i, h, args);

        if (ret != GRIB_SUCCESS)
            return ret;

        if (c->init)
            return c->init(i, h, args);
    }
    return GRIB_INTERNAL_ERROR;
}


//extern FILE* grib_yyin;
FILE* grib_yyin;
const char* parse_file = 0;
grib_context* grib_parser_context = 0;
static int top = 0;
// extern int grib_yylineno;
int grib_yylineno;
static context stack[MAXINCLUDE];
// extern int grib_yyparse(void);
int grib_yyparse(void);
// extern void grib_yyrestart(FILE*);
void grib_yyrestart(FILE*);

static int parse(grib_context* gc, const char* filename)
{
    int err = 0;
    GRIB_MUTEX_INIT_ONCE(&once, &init);
    GRIB_MUTEX_LOCK(&mutex_parse);

#ifdef YYDEBUG
    {
        //extern int grib_yydebug;
        int grib_yydebug;
        grib_yydebug = getenv("YYDEBUG") != NULL;
    }
#endif

    gc = gc ? gc : grib_context_get_default();

    grib_yyin  = NULL;
    top        = 0;
    parse_file = 0;
    grib_parser_include(filename);
    if (!grib_yyin) {
        /* Could not read from file */
        parse_file = 0;
        GRIB_MUTEX_UNLOCK(&mutex_parse);
        return GRIB_FILE_NOT_FOUND;
    }
    err        = grib_yyparse();
    parse_file = 0;

    if (err)
        grib_context_log(gc, GRIB_LOG_ERROR, "Parsing error: %s, file: %s\n",
                grib_get_error_message(err), filename);

    GRIB_MUTEX_UNLOCK(&mutex_parse);
    return err;
}

void grib_parser_include(const char* included_fname)
{
    FILE* f         = NULL;
    char* io_buffer = 0;
    /* int i; */
    Assert(top < MAXINCLUDE);
    Assert(included_fname);
    if (!included_fname)
        return;

    if (parse_file == 0) {
        parse_file = included_fname;
        Assert(top == 0);
    }
    else {
        /* When parse_file is not NULL, it's the path of the parent file (includer) */
        /* and 'included_fname' is the name of the file being included (includee) */

        /* GRIB-796: Search for the included file in ECCODES_DEFINITION_PATH */
        char* new_path = NULL;
        Assert(*included_fname != '/');
        new_path = grib_context_full_defs_path(grib_parser_context, included_fname);
        if (!new_path) {
            const char* ver = "eccodes_version_str";
            fprintf(stderr, "ecCodes Version:       %s\nDefinition files path: %s\n",
                    ver,
                    grib_parser_context->grib_definition_files_path);

            grib_context_log(grib_parser_context, GRIB_LOG_FATAL,
                             "grib_parser_include: Could not resolve '%s' (included in %s)", included_fname, parse_file);

            return;
        }
        parse_file = new_path;
    }

    if (strcmp(parse_file, "-") == 0) {
        grib_context_log(grib_parser_context, GRIB_LOG_DEBUG, "parsing standard input");
        f = stdin; /* read from std input */
    }
    else {
        grib_context_log(grib_parser_context, GRIB_LOG_DEBUG, "parsing include file %s", parse_file);
        f = codes_fopen(parse_file, "r");
    }
    /* for(i = 0; i < top ; i++) printf("   "); */
    /* printf("PARSING %s\n",parse_file); */

    if (f == NULL) {
        char buffer[1024];
        grib_context_log(grib_parser_context, (GRIB_LOG_ERROR) | (GRIB_LOG_PERROR), "grib_parser_include: cannot open: '%s'", parse_file);
        sprintf(buffer, "Cannot include file: '%s'", parse_file);
        grib_yyerror(buffer);
    }
    else {
        /*
        c=grib_context_get_default();
        if (c->io_buffer_size) {
            if (posix_memalign(&(io_buffer),sysconf(_SC_PAGESIZE),c->io_buffer_size) ) {
                        grib_context_log(c,GRIB_LOG_FATAL,"grib_parser_include: posix_memalign unable to allocate io_buffer\n");
            }
            setvbuf(f,io_buffer,_IOFBF,c->io_buffer_size);
        }
        */
        grib_yyin            = f;
        stack[top].file      = f;
        stack[top].io_buffer = io_buffer;
        stack[top].name      = grib_context_strdup(grib_parser_context, parse_file);
        parse_file           = stack[top].name;
        stack[top].line      = grib_yylineno;
        grib_yylineno        = 0;
        top++;
        /* grib_yyrestart(f); */
    }
}

static int _read_any(reader* r, int grib_ok, int bufr_ok, int hdf5_ok, int wrap_ok)
{
    unsigned char c;
    int err             = 0;
    unsigned long magic = 0;

    while (r->read(r->read_data, &c, 1, &err) == 1 && err == 0) {
        magic <<= 8;
        magic |= c;

        switch (magic & 0xffffffff) {
            case GRIB:
                if (grib_ok) {
                    err = read_GRIB(r);
                    return err == GRIB_END_OF_FILE ? GRIB_PREMATURE_END_OF_FILE : err; /* Premature EOF */
                }
                break;

            case BUFR:
                if (bufr_ok) {
                    err = read_BUFR(r);
                    return err == GRIB_END_OF_FILE ? GRIB_PREMATURE_END_OF_FILE : err; /* Premature EOF */
                }
                break;

            case HDF5:
                if (hdf5_ok) {
                    err = read_HDF5(r);
                    return err == GRIB_END_OF_FILE ? GRIB_PREMATURE_END_OF_FILE : err; /* Premature EOF */
                }
                break;

            case WRAP:
                if (wrap_ok) {
                    err = read_WRAP(r);
                    return err == GRIB_END_OF_FILE ? GRIB_PREMATURE_END_OF_FILE : err; /* Premature EOF */
                }
                break;

            case BUDG:
                if (grib_ok) {
                    err = read_PSEUDO(r, "BUDG");
                    return err == GRIB_END_OF_FILE ? GRIB_PREMATURE_END_OF_FILE : err; /* Premature EOF */
                }
                break;
            case DIAG:
                if (grib_ok) {
                    err = read_PSEUDO(r, "DIAG");
                    return err == GRIB_END_OF_FILE ? GRIB_PREMATURE_END_OF_FILE : err; /* Premature EOF */
                }
                break;
            case TIDE:
                if (grib_ok) {
                    err = read_PSEUDO(r, "TIDE");
                    return err == GRIB_END_OF_FILE ? GRIB_PREMATURE_END_OF_FILE : err; /* Premature EOF */
                }
                break;
        }
    }

    return err;
}

static int read_any(reader* r, int grib_ok, int bufr_ok, int hdf5_ok, int wrap_ok)
{
    int result = 0;
    result = _read_any(r, grib_ok, bufr_ok, hdf5_ok, wrap_ok);
    return result;
}

static void rebuild_hash_keys(grib_handle* h, grib_section* s)
{
    grib_accessor* a = s ? s->block->first : NULL;

    while (a) {
        grib_section* sub = a->sub_section;
        int i             = 0;
        int id            = -1;
        const char* p;
        DebugAssert(h == grib_handle_of_accessor(a));

        while (i < MAX_ACCESSOR_NAMES && ((p = a->all_names[i]) != NULL)) {
            if (*p != '_') {
                id = grib_hash_keys_get_id(a->context->keys, p);

                if (a->same != a && i == 0) {
                    grib_handle* hand   = grib_handle_of_accessor(a);
                    a->same             = hand->accessors[id];
                    hand->accessors[id] = a;
                    DebugAssert(a->same != a);
                }
            }
            i++;
        }
        rebuild_hash_keys(h, sub);
        a = a->next;
    }
}

static void search_from_accessors_list(grib_accessors_list* al, const grib_accessors_list* end, const char* name, grib_accessors_list* result)
{
    char attribute_name[200] = {0,};
    grib_accessor* accessor_result = 0;
    grib_context* c = al->accessor->context;
    int doFree = 1;

    char* accessor_name = grib_split_name_attribute(c, name, attribute_name);
    if (*attribute_name == 0) doFree = 0;

    while (al && al != end && al->accessor) {
        if (grib_inline_strcmp(al->accessor->name, accessor_name) == 0) {
            if (attribute_name[0]) {
                accessor_result = grib_accessor_get_attribute(al->accessor, attribute_name);
            }
            else {
                accessor_result = al->accessor;
            }
            if (accessor_result) {
                grib_accessors_list_push(result, accessor_result, al->rank);
            }
        }
        al = al->next;
    }
    if (al == end && al->accessor) {
        if (grib_inline_strcmp(al->accessor->name, accessor_name) == 0) {
            if (attribute_name[0]) {
                accessor_result = grib_accessor_get_attribute(al->accessor, attribute_name);
            }
            else {
                accessor_result = al->accessor;
            }
            if (accessor_result) {
                grib_accessors_list_push(result, accessor_result, al->rank);
            }
        }
    }
    if (doFree) grib_context_free(c, accessor_name);
}

static void search_accessors_list_by_condition(grib_accessors_list* al, const char* name, codes_condition* condition, grib_accessors_list* result)
{
    grib_accessors_list* start = NULL;
    grib_accessors_list* end   = NULL;

    while (al) {
        if (!grib_inline_strcmp(al->accessor->name, condition->left)) {
            if (start == NULL && condition_true(al->accessor, condition))
                start = al;
            if (start && !condition_true(al->accessor, condition))
                end = al;
        }
        if (start != NULL && (end != NULL || al->next == NULL)) {
            if (end == NULL)
                end = al;
            search_from_accessors_list(start, end, name, result);
            al    = end;
            start = NULL;
            end   = NULL;
        }
        al = al->next;
    }
}

size_t stdio_read(void* data, void* buf, size_t len, int* err)
{
    FILE* f = (FILE*)data;
    size_t n;
    /* char iobuf[1024*1024]; */

    if (len == 0)
        return 0;

    /* setvbuf(f,iobuf,_IOFBF,sizeof(iobuf)); */
    n = fread(buf, 1, len, f);
    /* fprintf(stderr,"read %d = %x %c\n",1,(int)buf[0],buf[0]); */
    if (n != len) {
        /* fprintf(stderr,"Failed to read %d, only got %d\n",len,n); */
        *err = GRIB_IO_PROBLEM;
        if (feof(f))
            *err = GRIB_END_OF_FILE;
        if (ferror(f))
            *err = GRIB_IO_PROBLEM;
    }
    return n;
}

int stdio_seek_from_start(void* data, off_t len)
{
    FILE* f = (FILE*)data;
    int err = 0;
    if (fseeko(f, len, SEEK_SET))
        err = GRIB_IO_PROBLEM;
    return err;
}

int stdio_seek(void* data, off_t len)
{
    FILE* f = (FILE*)data;
    int err = 0;
    if (fseeko(f, len, SEEK_CUR))
        err = GRIB_IO_PROBLEM;
    return err;
}

off_t stdio_tell(void* data)
{
    FILE* f = (FILE*)data;
    return ftello(f);
}

static unsigned int hash_keys (register const char *str, register size_t len)
{
  static const unsigned short asso_values[] =
    {
      32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423,
      32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423,
      32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423,
      32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423,
      32423, 32423,     1, 32423, 32423,     1, 32423, 32423,   100,  2542,
       2012,  2507,  1904,  3737,  1317,   921,   233,     6,     5,     1,
          1, 32423, 32423, 32423, 32423,  2617,  4123,  2527,   159,  1640,
         52,  5304,  2521,   684,    43,   193,   551,   292,  1641,   211,
       1969,    64,  1061,   161,    85,  4435,  2022,  3043,    60,  4866,
          6,     1,     1, 32423, 32423,  1548, 32423,     5,   552,    54,
          1,     2,   196,   180,   109,    10,  2716,  4017,    71,     7,
          1,    20,    29,  1211,     1,     8,     4,    65,   258,   230,
        764,     7,   784,    55,  1795,     2, 32423, 32423, 32423, 32423,
      32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423,
      32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423,
      32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423,
      32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423,
      32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423,
      32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423,
      32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423,
      32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423,
      32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423,
      32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423,
      32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423,
      32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423,
      32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423, 32423
    };
  register unsigned int hval = len;

  switch (hval)
    {
      default:
        hval += asso_values[(unsigned char)str[27]];
      /*FALLTHROUGH*/
      case 27:
      case 26:
        hval += asso_values[(unsigned char)str[25]];
      /*FALLTHROUGH*/
      case 25:
        hval += asso_values[(unsigned char)str[24]];
      /*FALLTHROUGH*/
      case 24:
        hval += asso_values[(unsigned char)str[23]];
      /*FALLTHROUGH*/
      case 23:
        hval += asso_values[(unsigned char)str[22]];
      /*FALLTHROUGH*/
      case 22:
      case 21:
      case 20:
        hval += asso_values[(unsigned char)str[19]];
      /*FALLTHROUGH*/
      case 19:
        hval += asso_values[(unsigned char)str[18]];
      /*FALLTHROUGH*/
      case 18:
      case 17:
      case 16:
        hval += asso_values[(unsigned char)str[15]+3];
      /*FALLTHROUGH*/
      case 15:
        hval += asso_values[(unsigned char)str[14]];
      /*FALLTHROUGH*/
      case 14:
        hval += asso_values[(unsigned char)str[13]];
      /*FALLTHROUGH*/
      case 13:
        hval += asso_values[(unsigned char)str[12]];
      /*FALLTHROUGH*/
      case 12:
        hval += asso_values[(unsigned char)str[11]+3];
      /*FALLTHROUGH*/
      case 11:
        hval += asso_values[(unsigned char)str[10]+3];
      /*FALLTHROUGH*/
      case 10:
        hval += asso_values[(unsigned char)str[9]];
      /*FALLTHROUGH*/
      case 9:
        hval += asso_values[(unsigned char)str[8]];
      /*FALLTHROUGH*/
      case 8:
        hval += asso_values[(unsigned char)str[7]];
      /*FALLTHROUGH*/
      case 7:
        hval += asso_values[(unsigned char)str[6]];
      /*FALLTHROUGH*/
      case 6:
        hval += asso_values[(unsigned char)str[5]];
      /*FALLTHROUGH*/
      case 5:
        hval += asso_values[(unsigned char)str[4]];
      /*FALLTHROUGH*/
      case 4:
        hval += asso_values[(unsigned char)str[3]];
      /*FALLTHROUGH*/
      case 3:
        hval += asso_values[(unsigned char)str[2]];
      /*FALLTHROUGH*/
      case 2:
        hval += asso_values[(unsigned char)str[1]];
      /*FALLTHROUGH*/
      case 1:
        hval += asso_values[(unsigned char)str[0]];
        break;
    }
  return hval + asso_values[(unsigned char)str[len - 1]];
}

void* wmo_read_grib_from_file_malloc(FILE* f, int headers_only, size_t* size, off_t* offset, int* err)
{
    return _wmo_read_any_from_file_malloc(f, err, size, offset, 1, 0, 0, 0, headers_only);
}

static void* _wmo_read_any_from_file_malloc(FILE* f, int* err, size_t* size, off_t* offset,
                                            int grib_ok, int bufr_ok, int hdf5_ok, int wrap_ok, int headers_only)
{
    alloc_buffer u;
    reader r;

    u.buffer = NULL;
    u.size   = 0;

    r.message_size    = 0;
    r.read_data       = f;
    r.read            = &stdio_read;
    r.seek            = &stdio_seek;
    r.seek_from_start = &stdio_seek_from_start;
    r.tell            = &stdio_tell;
    r.alloc_data      = &u;
    r.alloc           = &allocate_buffer;
    r.headers_only    = headers_only;
    r.offset          = 0;

    *err = read_any(&r, grib_ok, bufr_ok, hdf5_ok, wrap_ok);

    *size   = r.message_size;
    *offset = r.offset;

    return u.buffer;
}

static int read_HDF5_offset(reader* r, int length, unsigned long* v, unsigned char* tmp, int* i)
{
    unsigned char buf[8];
    int j, k;
    int err = 0;


    if ((r->read(r->read_data, buf, length, &err) != length) || err) {
        return err;
    }

    k = *i;
    for (j = 0; j < length; j++) {
        tmp[k++] = buf[j];
    }
    *i = k;

    *v = 0;
    for (j = length - 1; j >= 0; j--) {
        *v <<= 8;
        *v |= buf[j];
    }

    return 0;
}

static int read_HDF5(reader* r)
{
    /* 
     * See: http://www.hdfgroup.org/HDF5/doc/H5.format.html#Superblock
     */
    unsigned char tmp[49]; /* Should be enough */
    unsigned char buf[4];

    unsigned char version_of_superblock, size_of_offsets, size_of_lengths, consistency_flags;
    unsigned long base_address, superblock_extension_address, end_of_file_address;

    int i           = 0, j;
    int err         = 0;
    grib_context* c = grib_context_get_default();

    tmp[i++] = 137;
    tmp[i++] = 'H';
    tmp[i++] = 'D';
    tmp[i++] = 'F';

    if ((r->read(r->read_data, buf, 4, &err) != 4) || err) {
        return err;
    }

    if (!(buf[0] == '\r' && buf[1] == '\n' && buf[2] == 26 && buf[3] == '\n')) {
        /* Invalid magic, we should not use grib_context_log without a context */
        grib_context_log(c, GRIB_LOG_ERROR, "read_HDF5: invalid signature");
        return GRIB_INVALID_MESSAGE;
    }

    for (j = 0; j < 4; j++) {
        tmp[i++] = buf[j];
    }

    if ((r->read(r->read_data, &version_of_superblock, 1, &err) != 1) || err) {
        return err;
    }

    tmp[i++] = version_of_superblock;

    if (version_of_superblock == 2 || version_of_superblock == 3) {
        if ((r->read(r->read_data, &size_of_offsets, 1, &err) != 1) || err) {
            return err;
        }

        tmp[i++] = size_of_offsets;

        if (size_of_offsets > 8) {
            grib_context_log(c, GRIB_LOG_ERROR, "read_HDF5: invalid size_of_offsets: %ld, only <= 8 is supported", (long)size_of_offsets);
            return GRIB_NOT_IMPLEMENTED;
        }

        if ((r->read(r->read_data, &size_of_lengths, 1, &err) != 1) || err) {
            return err;
        }

        tmp[i++] = size_of_lengths;

        if ((r->read(r->read_data, &consistency_flags, 1, &err) != 1) || err) {
            return err;
        }

        tmp[i++] = consistency_flags;

        err = read_HDF5_offset(r, size_of_offsets, &base_address, tmp, &i);
        if (err) {
            return err;
        }

        err = read_HDF5_offset(r, size_of_offsets, &superblock_extension_address, tmp, &i);
        if (err) {
            return err;
        }

        err = read_HDF5_offset(r, size_of_offsets, &end_of_file_address, tmp, &i);
        if (err) {
            return err;
        }
    }
    else if (version_of_superblock == 0 || version_of_superblock == 1) {
        char skip[4];
        unsigned long file_free_space_info;
        unsigned char version_of_file_free_space, version_of_root_group_symbol_table, version_number_shared_header, ch;

        if ((r->read(r->read_data, &version_of_file_free_space, 1, &err) != 1) || err)
            return err;
        tmp[i++] = version_of_file_free_space;

        if ((r->read(r->read_data, &version_of_root_group_symbol_table, 1, &err) != 1) || err)
            return err;
        tmp[i++] = version_of_root_group_symbol_table;

        if ((r->read(r->read_data, &ch, 1, &err) != 1) || err)
            return err; /* reserved */
        tmp[i++] = ch;

        if ((r->read(r->read_data, &version_number_shared_header, 1, &err) != 1) || err)
            return err;
        tmp[i++] = version_number_shared_header;

        if ((r->read(r->read_data, &size_of_offsets, 1, &err) != 1) || err)
            return err;
        tmp[i++] = size_of_offsets;
        if (size_of_offsets > 8) {
            grib_context_log(c, GRIB_LOG_ERROR, "read_HDF5: invalid size_of_offsets: %ld, only <= 8 is supported", (long)size_of_offsets);
            return GRIB_NOT_IMPLEMENTED;
        }

        if ((r->read(r->read_data, &size_of_lengths, 1, &err) != 1) || err)
            return err;
        tmp[i++] = size_of_lengths;

        if ((r->read(r->read_data, &ch, 1, &err) != 1) || err)
            return err; /*reserved*/
        tmp[i++] = ch;

        if ((r->read(r->read_data, &skip, 4, &err) != 4) || err)
            return err; /* Group Leaf/Internal Node K: 4 bytes */
        tmp[i++] = skip[0];
        tmp[i++] = skip[1];
        tmp[i++] = skip[2];
        tmp[i++] = skip[3];

        if ((r->read(r->read_data, &skip, 4, &err) != 4) || err)
            return err; /* consistency_flags: 4 bytes */
        tmp[i++] = skip[0];
        tmp[i++] = skip[1];
        tmp[i++] = skip[2];
        tmp[i++] = skip[3];

        if (version_of_superblock == 1) {
            /* Indexed storage internal node K and reserved: only in version 1 of superblock */
            if ((r->read(r->read_data, &skip, 4, &err) != 4) || err)
                return err;
            tmp[i++] = skip[0];
            tmp[i++] = skip[1];
            tmp[i++] = skip[2];
            tmp[i++] = skip[3];
        }

        err = read_HDF5_offset(r, size_of_offsets, &base_address, tmp, &i);
        if (err)
            return err;

        err = read_HDF5_offset(r, size_of_offsets, &file_free_space_info, tmp, &i);
        if (err)
            return err;

        err = read_HDF5_offset(r, size_of_offsets, &end_of_file_address, tmp, &i);
        if (err)
            return err;
    }
    else {
        grib_context_log(c, GRIB_LOG_ERROR, "read_HDF5: invalid version of superblock: %ld", (long)version_of_superblock);
        return GRIB_NOT_IMPLEMENTED;
    }

    Assert(i <= sizeof(tmp));
    return read_the_rest(r, end_of_file_address, tmp, i, 0);
}

static int read_WRAP(reader* r)
{
    /*
     * See: http://www.hdfgroup.org/HDF5/doc/H5.format.html#Superblock
     */
    unsigned char tmp[36]; /* Should be enough */
    unsigned char buf[8];

    unsigned long long length = 0;

    int i   = 0, j;
    int err = 0;

    tmp[i++] = 'W';
    tmp[i++] = 'R';
    tmp[i++] = 'A';
    tmp[i++] = 'P';

    if ((r->read(r->read_data, buf, 8, &err) != 8) || err) {
        printf("error\n");
        return err;
    }

    for (j = 0; j < 8; j++) {
        length <<= 8;
        length |= buf[j];
        tmp[i++] = buf[j];
    }

    Assert(i <= sizeof(tmp));
    return read_the_rest(r, length, tmp, i, 1);
}

static int read_BUFR(reader* r)
{
    /* unsigned char tmp[65536];*/ /* Should be enough */
    size_t length      = 0;
    long edition       = 0;
    int err            = 0;
    int i              = 0, j;
    size_t buflen      = 2048;
    unsigned char* tmp = NULL;
    grib_context* c    = NULL;
    grib_buffer* buf   = NULL;

    /*TODO proper context*/
    c   = grib_context_get_default();
    tmp = (unsigned char*)malloc(buflen);
    if (!tmp)
        return GRIB_OUT_OF_MEMORY;
    buf           = grib_new_buffer(c, tmp, buflen);
    buf->property = GRIB_MY_BUFFER;
    r->offset     = r->tell(r->read_data) - 4;

    tmp[i++] = 'B';
    tmp[i++] = 'U';
    tmp[i++] = 'F';
    tmp[i++] = 'R';

    for (j = 0; j < 3; j++) {
        if (r->read(r->read_data, &tmp[i], 1, &err) != 1 || err)
            return err;

        length <<= 8;
        length |= tmp[i];
        i++;
    }

    if (length == 0) {
        grib_buffer_delete(c, buf);
        return GRIB_INVALID_MESSAGE;
    }

    /* Edition number */
    if (r->read(r->read_data, &tmp[i], 1, &err) != 1 || err)
        return err;

    edition = tmp[i++];

    /* Assert(edition != 1); */

    switch (edition) {
        case 0:
        case 1: {
            int n;
            size_t sec1len = 0;
            size_t sec2len = 0;
            size_t sec3len = 0;
            size_t sec4len = 0;
            unsigned long flags;

            sec1len = length;

            /* table version */
            if (r->read(r->read_data, &tmp[i++], 1, &err) != 1 || err)
                return err;
            /* center */
            if (r->read(r->read_data, &tmp[i++], 1, &err) != 1 || err)
                return err;
            /* update */
            if (r->read(r->read_data, &tmp[i++], 1, &err) != 1 || err)
                return err;
            /* flags */
            if (r->read(r->read_data, &tmp[i], 1, &err) != 1 || err)
                return err;
            flags = tmp[i++];


            GROW_BUF_IF_REQUIRED(sec1len + 4 + 3);

            /* Read section 1. 3 = length, 5 = table,center,process,flags */

            n = sec1len - 8; /* Just a guess */
            if ((r->read(r->read_data, tmp + i, n, &err) != n) || err)
                return err;

            i += n;

            if (flags & (1 << 7)) {
                /* Section 2 */
                for (j = 0; j < 3; j++) {
                    if (r->read(r->read_data, &tmp[i], 1, &err) != 1 || err)
                        return err;

                    sec2len <<= 8;
                    sec2len |= tmp[i];
                    i++;
                }

                GROW_BUF_IF_REQUIRED(sec1len + sec2len + 4 + 3);

                /* Read section 2 */
                if ((r->read(r->read_data, tmp + i, sec2len - 3, &err) != sec2len - 3) || err)
                    return err;
                i += sec2len - 3;
            }


            /* Section 3 */
            for (j = 0; j < 3; j++) {
                if (r->read(r->read_data, &tmp[i], 1, &err) != 1 || err)
                    return err;

                sec3len <<= 8;
                sec3len |= tmp[i];
                i++;
            }

            GROW_BUF_IF_REQUIRED(sec1len + sec2len + sec3len + 4 + 3);

            /* Read section 3 */
            if ((r->read(r->read_data, tmp + i, sec3len - 3, &err) != sec3len - 3) || err)
                return err;
            i += sec3len - 3;

            for (j = 0; j < 3; j++) {
                if (r->read(r->read_data, &tmp[i], 1, &err) != 1 || err)
                    return err;

                sec4len <<= 8;
                sec4len |= tmp[i];
                i++;
            }

            /* fprintf(stderr," sec1len=%d sec2len=%d sec3len=%d sec4len=%d\n",sec1len, sec2len,sec3len,sec4len); */
            length = 4 + sec1len + sec2len + sec3len + sec4len + 4;
            /* fprintf(stderr,"length = %d i = %d\n",length,i); */
        } break;
        case 2:
        case 3:
        case 4:
            break;
        default:
            r->seek_from_start(r->read_data, r->offset + 4);
            grib_buffer_delete(c, buf);
            return GRIB_UNSUPPORTED_EDITION;
    }

    /* Assert(i <= sizeof(tmp)); */
    err = read_the_rest(r, length, tmp, i, 1);
    if (err)
        r->seek_from_start(r->read_data, r->offset + 4);

    grib_buffer_delete(c, buf);

    return err;
}

static int read_PSEUDO(reader* r, const char* type)
{
    unsigned char tmp[32]; /* Should be enough */
    size_t sec1len = 0;
    size_t sec4len = 0;
    int err        = 0;
    int i = 0, j = 0;

    Assert(strlen(type) == 4);
    for (j = 0; j < 4; j++) {
        tmp[i] = type[i];
        i++;
    }

    for (j = 0; j < 3; j++) {
        if (r->read(r->read_data, &tmp[i], 1, &err) != 1 || err)
            return err;

        sec1len <<= 8;
        sec1len |= tmp[i];
        i++;
    }

    /* fprintf(stderr,"%s sec1len=%d i=%d\n",type,sec1len,i); */
    CHECK_TMP_SIZE(sec1len + 4 + 3);

    /* Read sectoin1 */
    if ((r->read(r->read_data, tmp + i, sec1len - 3, &err) != sec1len - 3) || err)
        return err;

    i += sec1len - 3;

    for (j = 0; j < 3; j++) {
        if (r->read(r->read_data, &tmp[i], 1, &err) != 1 || err)
            return err;

        sec4len <<= 8;
        sec4len |= tmp[i];
        i++;
    }

    /* fprintf(stderr,"%s sec4len=%d i=%d l=%d\n",type,sec4len,i,4+sec1len+sec4len+4); */

    Assert(i <= sizeof(tmp));
    return read_the_rest(r, 4 + sec1len + sec4len + 4, tmp, i, 1);
}

static int read_GRIB(reader* r)
{
    unsigned char* tmp  = NULL;
    size_t length       = 0;
    size_t total_length = 0;
    long edition        = 0;
    int err             = 0;
    int i               = 0, j;
    size_t sec1len      = 0;
    size_t sec2len      = 0;
    size_t sec3len      = 0;
    size_t sec4len      = 0;
    unsigned long flags;
    size_t buflen = 32768; /* See ECC-515: was 16368 */
    grib_context* c;
    grib_buffer* buf;

    /*TODO proper context*/
    c   = grib_context_get_default();
    tmp = (unsigned char*)malloc(buflen);
    if (!tmp)
        return GRIB_OUT_OF_MEMORY;
    buf           = grib_new_buffer(c, tmp, buflen);
    buf->property = GRIB_MY_BUFFER;

    tmp[i++] = 'G';
    tmp[i++] = 'R';
    tmp[i++] = 'I';
    tmp[i++] = 'B';

    r->offset = r->tell(r->read_data) - 4;

    if (r->read(r->read_data, &tmp[i], 3, &err) != 3 || err)
        return err;

    length = UINT3(tmp[i], tmp[i + 1], tmp[i + 2]);
    i += 3;

    /* Edition number */
    if (r->read(r->read_data, &tmp[i], 1, &err) != 1 || err)
        return err;

    edition = tmp[i++];
    switch (edition) {
        case 1:
            if (r->headers_only) {
                /* Read section 1 length */
                if (r->read(r->read_data, &tmp[i], 3, &err) != 3 || err)
                    return err;

                sec1len = UINT3(tmp[i], tmp[i + 1], tmp[i + 2]);
                i += 3;
                /* Read section 1. 3 = length */
                if ((r->read(r->read_data, tmp + i, sec1len - 3, &err) != sec1len - 3) || err)
                    return err;
                flags = tmp[15];

                i += sec1len - 3;

                GROW_BUF_IF_REQUIRED(i + 3);

                if (flags & (1 << 7)) {
                    /* Section 2 */
                    if (r->read(r->read_data, &tmp[i], 3, &err) != 3 || err)
                        return err;

                    sec2len = UINT3(tmp[i], tmp[i + 1], tmp[i + 2]);
                    GROW_BUF_IF_REQUIRED(i + sec2len);
                    i += 3;
                    /* Read section 2 */
                    if ((r->read(r->read_data, tmp + i, sec2len - 3, &err) != sec2len - 3) || err)
                        return err;
                    i += sec2len - 3;
                }


                if (flags & (1 << 6)) {
                    /* Section 3 */
                    GROW_BUF_IF_REQUIRED(i + 3);
                    for (j = 0; j < 3; j++) {
                        if (r->read(r->read_data, &tmp[i], 1, &err) != 1 || err)
                            return err;

                        sec3len <<= 8;
                        sec3len |= tmp[i];
                        i++;
                    }

                    /* Read section 3 */
                    GROW_BUF_IF_REQUIRED(i + sec3len);
                    if ((r->read(r->read_data, tmp + i, sec3len - 3, &err) != sec3len - 3) || err)
                        return err;
                    i += sec3len - 3;
                }

                GROW_BUF_IF_REQUIRED(i + 11);

                /* Section 4 */
                for (j = 0; j < 3; j++) {
                    if (r->read(r->read_data, &tmp[i], 1, &err) != 1 || err)
                        return err;

                    sec4len <<= 8;
                    sec4len |= tmp[i];
                    i++;
                }

                /* we don't read the data, only headers */
                if ((r->read(r->read_data, tmp + i, 8, &err) != 8) || err)
                    return err;

                i += 8;

                total_length = length;
                /* length=8+sec1len + sec2len+sec3len+11; */
                length = i;
                err    = r->seek(r->read_data, total_length - length - 1);
            }
            else if (length & 0x800000) {
                /* Large GRIBs */

                /* Read section 1 length */
                for (j = 0; j < 3; j++) {
                    if (r->read(r->read_data, &tmp[i], 1, &err) != 1 || err)
                        return err;

                    sec1len <<= 8;
                    sec1len |= tmp[i];
                    i++;
                }

                /* table version */
                if (r->read(r->read_data, &tmp[i++], 1, &err) != 1 || err)
                    return err;
                /* center */
                if (r->read(r->read_data, &tmp[i++], 1, &err) != 1 || err)
                    return err;
                /* process */
                if (r->read(r->read_data, &tmp[i++], 1, &err) != 1 || err)
                    return err;
                /* grid */
                if (r->read(r->read_data, &tmp[i++], 1, &err) != 1 || err)
                    return err;
                /* flags */
                if (r->read(r->read_data, &tmp[i], 1, &err) != 1 || err)
                    return err;
                flags = tmp[i++];

                /* fprintf(stderr," sec1len=%d i=%d flags=%x\n",sec1len,i,flags); */

                GROW_BUF_IF_REQUIRED(8 + sec1len + 4 + 3);

                /* Read section 1. 3 = length, 5 = table,center,process,grid,flags */
                if ((r->read(r->read_data, tmp + i, sec1len - 3 - 5, &err) != sec1len - 3 - 5) || err)
                    return err;

                i += sec1len - 3 - 5;

                if (flags & (1 << 7)) {
                    /* Section 2 */
                    for (j = 0; j < 3; j++) {
                        if (r->read(r->read_data, &tmp[i], 1, &err) != 1 || err)
                            return err;

                        sec2len <<= 8;
                        sec2len |= tmp[i];
                        i++;
                    }
                    /* Read section 2 */
                    GROW_BUF_IF_REQUIRED(i + sec2len);
                    if ((r->read(r->read_data, tmp + i, sec2len - 3, &err) != sec2len - 3) || err)
                        return err;
                    i += sec2len - 3;
                }

                GROW_BUF_IF_REQUIRED(sec1len + sec2len + 4 + 3);

                if (flags & (1 << 6)) {
                    /* Section 3 */
                    for (j = 0; j < 3; j++) {
                        if (r->read(r->read_data, &tmp[i], 1, &err) != 1 || err)
                            return err;

                        sec3len <<= 8;
                        sec3len |= tmp[i];
                        i++;
                    }

                    /* Read section 3 */
                    GROW_BUF_IF_REQUIRED(sec1len + sec2len + sec3len + 4 + 3);
                    if ((r->read(r->read_data, tmp + i, sec3len - 3, &err) != sec3len - 3) || err)
                        return err;
                    i += sec3len - 3;
                }

                /* fprintf(stderr,"%s sec1len=%d i=%d\n",type,sec1len,i); */

                GROW_BUF_IF_REQUIRED(sec1len + sec2len + sec3len + 4 + 3);


                for (j = 0; j < 3; j++) {
                    if (r->read(r->read_data, &tmp[i], 1, &err) != 1 || err)
                        return err;

                    sec4len <<= 8;
                    sec4len |= tmp[i];
                    i++;
                }

                if (sec4len < 120) {
                    /* Special coding */
                    length &= 0x7fffff;
                    length *= 120;
                    length -= sec4len;
                    length += 4;
                }
                else {
                    /* length is already set to the right value */
                }
            }
            break;

        case 2:
        case 3:
            length = 0;

            if (sizeof(long) >= 8) {
                for (j = 0; j < 8; j++) {
                    if (r->read(r->read_data, &tmp[i], 1, &err) != 1 || err)
                        return err;

                    length <<= 8;
                    length |= tmp[i];
                    i++;
                }
            }
            else {
                /* Check if the length fits in a long */
                for (j = 0; j < 4; j++) {
                    if (r->read(r->read_data, &tmp[i], 1, &err) != 1 || err)
                        return err;

                    length <<= 8;
                    length |= tmp[i];
                    i++;
                }

                if (length)
                    return GRIB_MESSAGE_TOO_LARGE; /* Message too large */

                for (j = 0; j < 4; j++) {
                    if (r->read(r->read_data, &tmp[i], 1, &err) != 1 || err)
                        return err;

                    length <<= 8;
                    length |= tmp[i];
                    i++;
                }
            }
            break;

        default:
            r->seek_from_start(r->read_data, r->offset + 4);
            grib_buffer_delete(c, buf);
            return GRIB_UNSUPPORTED_EDITION;
            break;
    }

    err = read_the_rest(r, length, tmp, i, 1);
    if (err)
        r->seek_from_start(r->read_data, r->offset + 4);

    grib_buffer_delete(c, buf);

    return err;
}

static int read_the_rest(reader* r, size_t message_length, unsigned char* tmp, int already_read, int check7777)
{
    int err = GRIB_SUCCESS;
    size_t buffer_size;
    size_t rest;
    unsigned char* buffer;
    grib_context* c = grib_context_get_default();

    if (message_length == 0)
        return GRIB_BUFFER_TOO_SMALL;

    buffer_size     = message_length;
    rest            = message_length - already_read;
    r->message_size = message_length;
    buffer          = (unsigned char*)r->alloc(r->alloc_data, &buffer_size, &err);
    if (err)
        return err;

    if (buffer == NULL || (buffer_size < message_length)) {
        return GRIB_BUFFER_TOO_SMALL;
    }

    memcpy(buffer, tmp, already_read);

    if ((r->read(r->read_data, buffer + already_read, rest, &err) != rest) || err) {
        /*fprintf(stderr, "read_the_rest: r->read failed: %s\n", grib_get_error_message(err));*/
        if (c->debug)
            fprintf(stderr, "ECCODES DEBUG read_the_rest: Read failed (Coded length=%lu, Already read=%d)\n",
                    message_length, already_read);
        return err;
    }

    if (check7777 && !r->headers_only &&
        (buffer[message_length - 4] != '7' ||
         buffer[message_length - 3] != '7' ||
         buffer[message_length - 2] != '7' ||
         buffer[message_length - 1] != '7'))
    {
        if (c->debug)
            fprintf(stderr, "ECCODES DEBUG read_the_rest: No final 7777 at expected location (Coded length=%lu)\n", message_length);
        return GRIB_WRONG_LENGTH;
    }

    return GRIB_SUCCESS;
}

#ifdef HAVE_MEMFS
/* These two functions are implemented in the generated C file memfs_gen_final.c in the build area */
/* See the memfs.py Python generator */
int codes_memfs_exists(const char* path);
FILE* codes_memfs_open(const char* path);

FILE* codes_fopen(const char* name, const char* mode)
{
    FILE* f;

    if (strcmp(mode, "r") != 0) { /* Not reading */
        return fopen(name, mode);
    }

    f = codes_memfs_open(name); /* Load from memory */
    if (f) {
        return f;
    }

    return fopen(name, mode);
}

int codes_access(const char* name, int mode)
{
    /* F_OK tests for the existence of the file  */
    if (mode != F_OK) {
        return access(name, mode);
    }

    if (codes_memfs_exists(name)) { /* Check memory */
        return 0;
    }

    return access(name, mode);
}

#else
/* No MEMFS */
FILE* codes_fopen(const char* name, const char* mode)
{
    return fopen(name, mode);
}

#endif

int grib_yyerror(const char* msg)
{
    grib_context_log(grib_parser_context, GRIB_LOG_ERROR,
                     "grib_parser: %s at line %d of %s", msg, grib_yylineno + 1, parse_file);
    const char* ver = "ECCODES_VERSION_STR";
    grib_context_log(grib_parser_context, GRIB_LOG_ERROR,
                     "ecCodes Version: %s", ver);
    error = 1;
    return 1;
}

long grib_op_neg(long a)
{
    return -a;
}

double grib_op_neg_d(double a)
{
    return -a;
}

long grib_op_pow(long a, long b)
{
    /* Note: This is actually 'a' to the power 'b' */
    return grib_power(b, a);
}

double grib_op_mul_d(double a, double b)
{
    return a * b;
}

long grib_op_mul(long a, long b)
{
    return a * b;
}

long grib_op_div(long a, long b)
{
    return a / b;
}

double grib_op_div_d(double a, double b)
{
    return a / b;
}

long grib_op_modulo(long a, long b)
{
    return a % b;
}

long grib_op_bit(long a, long b)
{
    return a & (1 << b);
}

long grib_op_bitoff(long a, long b)
{
    return !grib_op_bit(a, b);
}

long grib_op_not(long a)
{
    return !a;
}

double grib_op_ne_d(double a, double b)
{
    return a != b;
}

long grib_op_ne(long a, long b)
{
    return a != b;
}

long grib_op_le(long a, long b)
{
    return a <= b;
}

double grib_op_le_d(double a, double b)
{
    return a <= b;
}

long grib_op_ge(long a, long b)
{
    return a >= b;
}

double grib_op_ge_d(double a, double b)
{
    return a >= b;
}

long grib_op_lt(long a, long b)
{
    return a < b;
}

double grib_op_lt_d(double a, double b)
{
    return a < b;
}

long grib_op_eq(long a, long b)
{
    return a == b;
}

double grib_op_eq_d(double a, double b)
{
    return a == b;
}

long grib_op_gt(long a, long b)
{
    return a > b;
}

double grib_op_gt_d(double a, double b)
{
    return a > b;
}

long grib_op_sub(long a, long b)
{
    return a - b;
}

double grib_op_sub_d(double a, double b)
{
    return a - b;
}

long grib_op_add(long a, long b)
{
    return a + b;
}

double grib_op_add_d(double a, double b)
{
    return a + b;
}
double grib_power(long s, long n)
{
    double divisor = 1.0;
    if (s == 0)
        return 1.0;
    if (s == 1)
        return n;
    while (s < 0) {
        divisor /= n;
        s++;
    }
    while (s > 0) {
        divisor *= n;
        s--;
    }
    return divisor;
}


static grib_file_pool file_pool = {
    0,                    /* grib_context* context;*/
    0,                    /* grib_file* first;*/
    0,                    /* grib_file* current; */
    0,                    /* size_t size;*/
    0,                    /* int number_of_opened_files;*/
    GRIB_MAX_OPENED_FILES /* int max_opened_files; */
};
static short next_id = 0;
grib_hash_array_value* grib_integer_hash_array_value_new(grib_context* c, const char* name, grib_iarray* array)
{
    grib_hash_array_value* v = (grib_hash_array_value*)grib_context_malloc_clear_persistent(c, sizeof(grib_hash_array_value));

    v->name   = grib_context_strdup_persistent(c, name);
    v->type   = GRIB_HASH_ARRAY_TYPE_INTEGER;
    v->iarray = array;
    return v;
}
grib_concept_condition* grib_concept_condition_new(grib_context* c, const char* name, grib_expression* expression, grib_iarray* iarray)
{
    grib_concept_condition* v = (grib_concept_condition*)grib_context_malloc_clear_persistent(c, sizeof(grib_concept_condition));
    v->name                   = grib_context_strdup_persistent(c, name);
    v->expression             = expression;
    v->iarray                 = iarray;
    return v;
}
grib_concept_value* grib_concept_value_new(grib_context* c, const char* name, grib_concept_condition* conditions)
{
    grib_concept_value* v = (grib_concept_value*)grib_context_malloc_clear_persistent(c, sizeof(grib_concept_value));
    v->name               = grib_context_strdup_persistent(c, name);
    v->conditions         = conditions;
    return v;
}
grib_case* grib_case_new(grib_context* c, grib_arguments* values, grib_action* action)
{
    grib_case* Case = (grib_case*)grib_context_malloc_clear_persistent(c, sizeof(grib_case));

    Case->values = values;
    Case->action = action;

    return Case;
}
grib_arguments* grib_arguments_new(grib_context* c, grib_expression* g, grib_arguments* n)
{
    grib_arguments* l = (grib_arguments*)grib_context_malloc_clear_persistent(c, sizeof(grib_arguments));
    l->expression     = g;
    l->next           = n;
    return l;
}
grib_iarray* grib_iarray_push(grib_iarray* v, long val)
{
    size_t start_size    = 100;
    size_t start_incsize = 100;

    if (!v)
        v = grib_iarray_new(0, start_size, start_incsize);

    if (v->n >= v->size - v->number_of_pop_front)
        v = grib_iarray_resize(v);

    v->v[v->n] = val;
    v->n++;
    return v;
}
grib_sarray* grib_sarray_push(grib_context* c, grib_sarray* v, char* val)
{
    size_t start_size    = 100;
    size_t start_incsize = 100;
    if (!v)
        v = grib_sarray_new(c, start_size, start_incsize);

    if (v->n >= v->size)
        v = grib_sarray_resize(v);
    v->v[v->n] = val;
    v->n++;
    return v;
}
grib_darray* grib_darray_push(grib_context* c, grib_darray* v, double val)
{
    size_t start_size    = 100;
    size_t start_incsize = 100;
    if (!v)
        v = grib_darray_new(c, start_size, start_incsize);

    if (v->n >= v->size)
        v = grib_darray_resize(v);
    v->v[v->n] = val;
    v->n++;
    return v;
}

int grib_is_missing(const grib_handle* h, const char* name, int* err)
{
    grib_accessor* a = grib_find_accessor(h, name);
    return grib_accessor_is_missing(a, err);
}
int grib_get_double_internal(grib_handle* h, const char* name, double* val)
{
    int ret = grib_get_double(h, name, val);

    if (ret != GRIB_SUCCESS)
        grib_context_log(h->context, GRIB_LOG_ERROR,
                         "unable to get %s as double (%s)",
                         name, grib_get_error_message(ret));

    return ret;
}
void unrotate(const double inlat, const double inlon,
              const double angleOfRot, const double southPoleLat, const double southPoleLon,
              double* outlat, double* outlon)
{
    const double lon_x = inlon;
    const double lat_y = inlat;
    /* First convert the data point from spherical lat lon to (x',y',z') */
    double latr = lat_y * DEG2RAD;
    double lonr = lon_x * DEG2RAD;
    double xd   = cos(lonr) * cos(latr);
    double yd   = sin(lonr) * cos(latr);
    double zd   = sin(latr);

    double t = -(90.0 + southPoleLat);
    double o = -southPoleLon;

    double sin_t = sin(DEG2RAD * t);
    double cos_t = cos(DEG2RAD * t);
    double sin_o = sin(DEG2RAD * o);
    double cos_o = cos(DEG2RAD * o);

    double x = cos_t * cos_o * xd + sin_o * yd + sin_t * cos_o * zd;
    double y = -cos_t * sin_o * xd + cos_o * yd - sin_t * sin_o * zd;
    double z = -sin_t * xd + cos_t * zd;

    double ret_lat = 0, ret_lon = 0;

    /* Then convert back to 'normal' (lat,lon)
     * Uses arcsin, to convert back to degrees, put in range -1 to 1 in case of slight rounding error
     * avoid error on calculating e.g. asin(1.00000001) */
    if (z > 1.0)
        z = 1.0;
    if (z < -1.0)
        z = -1.0;

    ret_lat = asin(z) * RAD2DEG;
    ret_lon = atan2(y, x) * RAD2DEG;

    /* Still get a very small rounding error, round to 6 decimal places */
    ret_lat = roundf(ret_lat * 1000000.0) / 1000000.0;
    ret_lon = roundf(ret_lon * 1000000.0) / 1000000.0;

    ret_lon -= angleOfRot;

    /* Make sure ret_lon is in range*/
    /*
    while (ret_lon < lonmin_) ret_lon += 360.0;
    while (ret_lon >= lonmax_) ret_lon -= 360.0;
     */
    *outlat = ret_lat;
    *outlon = ret_lon;
}
int transform_iterator_data(grib_context* context, double* data,
                            long iScansNegatively, long jScansPositively,
                            long jPointsAreConsecutive, long alternativeRowScanning,
                            size_t numPoints, long nx, long ny)
{
    double* data2;
    double *pData0, *pData1, *pData2;
    unsigned long ix, iy;

    if (!iScansNegatively && jScansPositively && !jPointsAreConsecutive && !alternativeRowScanning) {
        /* Already +i and +j. No need to change */
        return GRIB_SUCCESS;
    }

    if (!context) context = grib_context_get_default();

    if (!iScansNegatively && !jScansPositively && !jPointsAreConsecutive && !alternativeRowScanning &&
        nx > 0 && ny > 0) {
        /* Regular grid +i -j: convert from we:ns to we:sn */
        size_t row_size = ((size_t)nx) * sizeof(double);
        data2           = (double*)grib_context_malloc(context, row_size);
        if (!data2) {
            grib_context_log(context, GRIB_LOG_ERROR, "Geoiterator data: Error allocating %ld bytes", row_size);
            return GRIB_OUT_OF_MEMORY;
        }
        for (iy = 0; iy < ny / 2; iy++) {
            memcpy(data2, data + ((size_t)iy) * nx, row_size);
            memcpy(data + iy * nx, data + (ny - 1 - iy) * ((size_t)nx), row_size);
            memcpy(data + (ny - 1 - iy) * ((size_t)nx), data2, row_size);
        }
        grib_context_free(context, data2);
        return GRIB_SUCCESS;
    }

    if (nx < 1 || ny < 1) {
        grib_context_log(context, GRIB_LOG_ERROR, "Geoiterator data: Invalid values for Nx and/or Ny");
        return GRIB_GEOCALCULUS_PROBLEM;
    }
    data2 = (double*)grib_context_malloc(context, numPoints * sizeof(double));
    if (!data2) {
        grib_context_log(context, GRIB_LOG_ERROR, "Geoiterator data: Error allocating %ld bytes", numPoints * sizeof(double));
        return GRIB_OUT_OF_MEMORY;
    }
    pData0 = data2;
    for (iy = 0; iy < ny; iy++) {
        long deltaX = 0;
        pData1 = pointer_to_data(0, iy, iScansNegatively, jScansPositively, jPointsAreConsecutive, alternativeRowScanning, nx, ny, data);
        if (!pData1) {
            grib_context_free(context, data2);
            return GRIB_GEOCALCULUS_PROBLEM;
        }
        pData2 = pointer_to_data(1, iy, iScansNegatively, jScansPositively, jPointsAreConsecutive, alternativeRowScanning, nx, ny, data);
        if (!pData2) {
            grib_context_free(context, data2);
            return GRIB_GEOCALCULUS_PROBLEM;
        }
        deltaX = pData2 - pData1;
        for (ix = 0; ix < nx; ix++) {
            *pData0++ = *pData1;
            pData1 += deltaX;
        }
    }
    memcpy(data, data2, ((size_t)numPoints) * sizeof(double));
    grib_context_free(context, data2);

    return GRIB_SUCCESS;
}
int grib_is_earth_oblate(grib_handle* h)
{
    long oblate = 0;
    int err     = grib_get_long(h, "earthIsOblate", &oblate);
    if (!err && oblate == 1) {
        return 1;
    }
    return 0;
}
double normalise_longitude_in_degrees(double lon)
{
    while (lon < 0)
        lon += 360;
    while (lon > 360)
        lon -= 360;
    return lon;
}
void grib_dependency_add(grib_accessor* observer, grib_accessor* observed)
{
    grib_handle* h        = NULL;
    grib_dependency* d    = NULL;
    grib_dependency* last = NULL;

    /*printf("grib_dependency_add: observe %p %p observed=%s observer=%s\n",
           (void*)observed, (void*)observer,
           observed ? observed->name : "NULL",
           observer ? observer->name : "NULL");*/

    if (!observer || !observed) {
        return;
    }
    h = handle_of(observed);
    d = h->dependencies;

    /* Assert(h == handle_of(observer)); */

    /* Check if already in list */
    while (d) {
        if (d->observer == observer && d->observed == observed)
            return;
        last = d;
        d    = d->next;
    }

    d = (grib_dependency*)grib_context_malloc_clear(h->context, sizeof(grib_dependency));
    Assert(d);

    d->observed = observed;
    d->observer = observer;
    d->next     = 0;

    if (last)
        last->next = d;
    else
        h->dependencies = d;
}
int grib_get_string_internal(grib_handle* h, const char* name, char* val, size_t* length)
{
    int ret = grib_get_string(h, name, val, length);

    if (ret != GRIB_SUCCESS)
        grib_context_log(h->context, GRIB_LOG_ERROR,
                         "unable to get %s as string (%s)",
                         name, grib_get_error_message(ret));

    return ret;
}
int grib_get_native_type(const grib_handle* h, const char* name, int* type)
{
    grib_accessors_list* al = NULL;
    grib_accessor* a        = NULL;
    *type                   = GRIB_TYPE_UNDEFINED;

    DebugAssert(name != NULL && strlen(name) > 0);

    if (name[0] == '/') {
        al = grib_find_accessors_list(h, name);
        if (!al)
            return GRIB_NOT_FOUND;
        *type = grib_accessor_get_native_type(al->accessor);
        grib_context_free(h->context, al);
    }
    else {
        a = grib_find_accessor(h, name);
        if (!a)
            return GRIB_NOT_FOUND;
        *type = grib_accessor_get_native_type(a);
    }

    return GRIB_SUCCESS;
}
int grib_expression_native_type(grib_handle* h, grib_expression* g)
{
    grib_expression_class* c = g->cclass;
    while (c) {
        if (c->native_type)
            return c->native_type(g, h);
        c = c->super ? *(c->super) : NULL;
    }
    if (g->cclass)
        grib_context_log(h->context, GRIB_LOG_ERROR, "No native_type() in %s\n", g->cclass->name);
    Assert(1 == 0);
    return 0;
}
void grib_dependency_observe_expression(grib_accessor* observer, grib_expression* e)
{
    grib_expression_add_dependency(e, observer);
}
void grib_expression_add_dependency(grib_expression* e, grib_accessor* observer)
{
    grib_expression_class* c = e->cclass;
    while (c) {
        if (c->add_dependency) {
            c->add_dependency(e, observer);
            return;
        }
        c = c->super ? *(c->super) : NULL;
    }
    Assert(1 == 0);
}
void grib_expression_free(grib_context* ctx, grib_expression* g)
{
    if (g) {
        grib_expression_class* c = g->cclass;
        while (c) {
            if (c->destroy)
                c->destroy(ctx, g);
            c = c->super ? *(c->super) : NULL;
        }
        grib_context_free_persistent(ctx, g);
    }
}
void grib_expression_print(grib_context* ctx, grib_expression* g, grib_handle* f)
{
    grib_expression_class* c = g->cclass;
    while (c) {
        if (c->print) {
            c->print(ctx, g, f);
            return;
        }
        c = c->super ? *(c->super) : NULL;
    }
    Assert(1 == 0);
}
const char* grib_expression_evaluate_string(grib_handle* h, grib_expression* g, char* buf, size_t* size, int* err)
{
    grib_expression_class* c = g->cclass;
    while (c) {
        if (c->evaluate_string)
            return c->evaluate_string(g, h, buf, size, err);
        c = c->super ? *(c->super) : NULL;
    }
    if (g->cclass)
        grib_context_log(h->context, GRIB_LOG_ERROR, "No evaluate_string() in %s\n", g->cclass->name);
    *err = GRIB_INVALID_TYPE;

    return 0;
}
int grib_expression_evaluate_double(grib_handle* h, grib_expression* g, double* result)
{
    grib_expression_class* c = g->cclass;
    while (c) {
        if (c->evaluate_double)
            return c->evaluate_double(g, h, result);
        c = c->super ? *(c->super) : NULL;
    }
    return GRIB_INVALID_TYPE;
}
int grib_expression_evaluate_long(grib_handle* h, grib_expression* g, long* result)
{
    grib_expression_class* c = g->cclass;
    while (c) {
        if (c->evaluate_long)
            return c->evaluate_long(g, h, result);
        c = c->super ? *(c->super) : NULL;
    }
    return GRIB_INVALID_TYPE;
}
void grib_dependency_observe_arguments(grib_accessor* observer, grib_arguments* a)
{
    while (a) {
        grib_dependency_observe_expression(observer, a->expression);
        a = a->next;
    }
}
void grib_arguments_free(grib_context* c, grib_arguments* g)
{
    if (g) {
        grib_arguments_free(c, g->next);
        grib_expression_free(c, g->expression);
        grib_context_free_persistent(c, g);
    }
}
int grib_action_execute(grib_action* a, grib_handle* h)
{
    grib_action_class* c = a->cclass;
    init(c);
    while (c) {
        if (c->execute_gac)
            return c->execute_gac(a, h);
        c = c->super ? *(c->super) : NULL;
    }
    DebugAssert(0);
    return 0;
}
void* grib_trie_insert_no_replace(grib_trie* t, const char* key, void* data)
{
    grib_trie* last = t;
    const char* k   = key;

    if (!t) {
        Assert(!"grib_trie_insert_no_replace: grib_trie==NULL");
        return NULL;
    }

    while (*k && t) {
        last = t;
        DebugCheckBounds((int)*k, key);
        t = t->next[mapping[(int)*k]];
        if (t)
            k++;
    }

    if (*k != 0) {
        t = last;
        while (*k) {
            int j = 0;
            DebugCheckBounds((int)*k, key);
            j = mapping[(int)*k++];
            if (j < t->first)
                t->first = j;
            if (j > t->last)
                t->last = j;
            t = t->next[j] = grib_trie_new(t->context);
        }
    }

    if (!t->data)
        t->data = data;

    return t->data;
}
int grib_itrie_get_id(grib_itrie* t, const char* key)
{
    const char* k    = key;
    grib_itrie* last = t;
    if (!t) {
        Assert(!"grib_itrie_get_id: grib_trie==NULL");
        return -1;
    }

    GRIB_MUTEX_INIT_ONCE(&once, &init);
    GRIB_MUTEX_LOCK(&mutex);

    while (*k && t)
        t = t->next[mapping[(int)*k++]];

    if (t != NULL && t->id != -1) {
        GRIB_MUTEX_UNLOCK(&mutex);
        return t->id;
    }
    else {
        int ret = grib_itrie_insert(last, key);
        GRIB_MUTEX_UNLOCK(&mutex);
        return ret;
    }
}
int grib_recompose_name(grib_handle* h, grib_accessor* observer, const char* uname, char* fname, int fail)
{
    grib_accessor* a;
    char loc[1024] = {0,};
    int i          = 0;
    int ret        = 0;
    int mode       = -1;
    char val[1024] = {0,};
    double dval        = 0;
    long lval          = 0;
    int type           = GRIB_TYPE_STRING;
    size_t replen      = 0;
    char* ptrEnd_fname = NULL; /* Maintain ptr to end of fname string */

    loc[0]       = 0;
    fname[0]     = 0;
    ptrEnd_fname = fname;

    /* uname is a string like "grib[GRIBEditionNumber:l]/boot.def". The result fname will be grib2/boot.def */
    while (uname[i] != '\0') {
        if (mode > -1) {
            if (uname[i] == ':') {
                type = grib_type_to_int(uname[i + 1]);
                i++;
            }
            else if (uname[i] == ']') {
                loc[mode] = 0;
                mode      = -1;
                a         = grib_find_accessor(h, loc);
                if (!a) {
                    if (!fail) {
                        sprintf(val, "undef");
                    }
                    else {
                        grib_context_log(h->context, GRIB_LOG_WARNING, "grib_recompose_name: Problem to recompose filename with : %s ( %s no accessor found)", uname, loc);
                        return GRIB_NOT_FOUND;
                    }
                }
                else {
                    switch (type) {
                        case GRIB_TYPE_STRING:
                            replen = 1024;
                            ret    = grib_unpack_string(a, val, &replen);
                            break;
                        case GRIB_TYPE_DOUBLE:
                            replen = 1;
                            ret    = grib_unpack_double(a, &dval, &replen);
                            sprintf(val, "%.12g", dval);
                            break;
                        case GRIB_TYPE_LONG:
                            replen = 1;
                            ret    = grib_unpack_long(a, &lval, &replen);
                            sprintf(val, "%d", (int)lval);
                            break;
                        default:
                            grib_context_log(h->context, GRIB_LOG_WARNING, "grib_recompose_name: Problem to recompose filename with : %s, invalid type %d", loc, type);
                            break;
                    }

                    grib_dependency_add(observer, a);

                    if ((ret != GRIB_SUCCESS)) {
                        grib_context_log(h->context, GRIB_LOG_ERROR, "grib_recompose_name: Could not recompose filename : %s", uname);
                        return ret;
                    }
                }
                {
                    char* pc = fname;
                    while (*pc != '\0')
                        pc++;
                    strcpy(pc, val);
                    ptrEnd_fname = pc + strlen(val); /* Update ptr to end of fname */
                }

                loc[0] = 0;
            }
            else
                loc[mode++] = uname[i];
        }
        else if (uname[i] == '[')
            mode = 0;
        else {

            /* Performance: faster to avoid call to strlen. Append to end */
            *ptrEnd_fname++ = uname[i];
            *ptrEnd_fname   = '\0';
            type = GRIB_TYPE_STRING;
        }
        i++;
    }
    /*fprintf(stdout,"parsed > %s\n",fname);*/
    return GRIB_SUCCESS;
}
void grib_hash_array_value_delete(grib_context* c, grib_hash_array_value* v)
{
    switch (v->type) {
        case GRIB_HASH_ARRAY_TYPE_INTEGER:
            grib_iarray_delete(v->iarray);
            break;
        case GRIB_HASH_ARRAY_TYPE_DOUBLE:
            grib_darray_delete(c, v->darray);
            break;
        default:
            grib_context_log(c, GRIB_LOG_ERROR,
                             "wrong type in grib_hash_array_value_delete");
    }
    grib_context_free_persistent(c, v->name);
    grib_context_free_persistent(c, v);
}
void grib_trie_delete(grib_trie* t)
{
    GRIB_MUTEX_INIT_ONCE(&once, &init);
    GRIB_MUTEX_LOCK(&mutex);
    if (t) {
        int i;
        for (i = t->first; i <= t->last; i++)
            if (t->next[i]) {
                grib_context_free(t->context, t->next[i]->data);
                grib_trie_delete(t->next[i]);
            }
#ifdef RECYCLE_TRIE
        grib_context_free_persistent(t->context, t);
#else
        grib_context_free(t->context, t);
#endif
    }
    GRIB_MUTEX_UNLOCK(&mutex);
}
void grib_context_print(const grib_context* c, void* descriptor, const char* fmt, ...)
{
    char msg[1024];
    va_list list;
    va_start(list, fmt);
    vsprintf(msg, fmt, list);
    va_end(list);
    c->print(c, descriptor, msg);
}
int grib_get_double(const grib_handle* h, const char* name, double* val)
{
    size_t length           = 1;
    grib_accessor* a        = NULL;
    grib_accessors_list* al = NULL;
    int ret                 = 0;

    if (name[0] == '/') {
        al = grib_find_accessors_list(h, name);
        if (!al)
            return GRIB_NOT_FOUND;
        ret = grib_unpack_double(al->accessor, val, &length);
        grib_context_free(h->context, al);
    }
    else {
        a = grib_find_accessor(h, name);
        if (!a)
            return GRIB_NOT_FOUND;
        ret = grib_unpack_double(a, val, &length);
    }
    return ret;
}
void grib_concept_value_delete(grib_context* c, grib_concept_value* v)
{
    grib_concept_condition* e = v->conditions;
    while (e) {
        grib_concept_condition* n = e->next;
        grib_concept_condition_delete(c, e);
        e = n;
    }
    grib_context_free_persistent(c, v->name);
    grib_context_free_persistent(c, v);
}
void grib_trie_delete_container(grib_trie* t)
{
    GRIB_MUTEX_INIT_ONCE(&once, &init);
    GRIB_MUTEX_LOCK(&mutex);
    if (t) {
        int i;
        for (i = t->first; i <= t->last; i++)
            if (t->next[i]) {
                grib_trie_delete_container(t->next[i]);
            }
#ifdef RECYCLE_TRIE
        grib_context_free_persistent(t->context, t);
#else
        grib_context_free(t->context, t);
#endif
    }
    GRIB_MUTEX_UNLOCK(&mutex);
}
void grib_push_accessor(grib_accessor* a, grib_block_of_accessors* l)
{
    int id;
    grib_handle* hand = grib_handle_of_accessor(a);
    if (!l->first)
        l->first = l->last = a;
    else {
        l->last->next = a;
        a->previous   = l->last;
    }
    l->last = a;

    if (hand->use_trie) {
        DebugAssert( a->all_names[0] );
        if (*(a->all_names[0]) != '_') {
            id = grib_hash_keys_get_id(a->context->keys, a->all_names[0]);

            DebugAssert(id >= 0 && id < ACCESSORS_ARRAY_SIZE);

            a->same = hand->accessors[id];
            link_same_attributes(a, a->same);
            hand->accessors[id] = a;

            if (a->same == a) {
                fprintf(stderr, "---> %s\n", a->name);
                Assert(a->same != a);
            }
        }
    }
}
void grib_dump_action_branch(FILE* out, grib_action* a, int decay)
{
    while (a) {
        grib_dump(a, out, decay);
        a = a->next;
    }
}
int grib_set_expression(grib_handle* h, const char* name, grib_expression* e)
{
    grib_accessor* a = grib_find_accessor(h, name);
    int ret          = GRIB_SUCCESS;

    if (a) {
        if (a->flags & GRIB_ACCESSOR_FLAG_READ_ONLY)
            return GRIB_READ_ONLY;

        ret = grib_pack_expression(a, e);
        if (ret == GRIB_SUCCESS) {
            return grib_dependency_notify_change(a);
        }
        return ret;
    }
    return GRIB_NOT_FOUND;
}
int grib_recompose_print(grib_handle* h, grib_accessor* observer, const char* uname, int fail, FILE* out)
{
    grib_accessors_list* al = NULL;
    char loc[1024];
    int i           = 0;
    int ret         = 0;
    int mode        = -1;
    char* pp        = NULL;
    char* format    = NULL;
    int type        = -1;
    char* separator = NULL;
    int l;
    char buff[10] = {0,};
    char buff1[1024] = {0,};
    int maxcolsd = 8;
    int maxcols;
    long numcols           = 0;
    int newline            = 1;
    const size_t uname_len = strlen(uname);

    maxcols = maxcolsd;
    loc[0]  = 0;
    for (i = 0; i < uname_len; i++) {
        if (mode > -1) {
            switch (uname[i]) {
                case ':':
                    type = grib_type_to_int(uname[i + 1]);
                    i++;
                    break;
                case '\'':
                    pp = (char*)(uname + i + 1);
                    while (*pp != '%' && *pp != '!' && *pp != ']' && *pp != ':' && *pp != '\'')
                        pp++;
                    l = pp - uname - i;
                    if (*pp == '\'')
                        separator = strncpy(buff1, uname + i + 1, l - 1);
                    i += l;
                    break;
                case '%':
                    pp = (char*)(uname + i + 1);
                    while (*pp != '%' && *pp != '!' && *pp != ']' && *pp != ':' && *pp != '\'')
                        pp++;
                    l      = pp - uname - i;
                    format = strncpy(buff, uname + i, l);
                    i += l - 1;
                    break;
                case '!':
                    pp = (char*)uname;
                    if (string_to_long(uname + i + 1, &numcols) == GRIB_SUCCESS) {
                        maxcols = (int)numcols;
                    }
                    else {
                        /* Columns specification is invalid integer */
                        maxcols = maxcolsd;
                    }
                    strtol(uname + i + 1, &pp, 10);
                    while (pp && *pp != '%' && *pp != '!' && *pp != ']' && *pp != ':' && *pp != '\'')
                        pp++;
                    i += pp - uname - i - 1;
                    break;
                case ']':
                    loc[mode] = 0;
                    mode      = -1;
                    if (al) grib_accessors_list_delete(h->context, al);
                    al        = grib_find_accessors_list(h, loc); /* This allocates memory */
                    if (!al) {
                        if (!fail) {
                            fprintf(out, "undef");
                            ret = GRIB_NOT_FOUND;
                        }
                        else {
                            grib_context_log(h->context, GRIB_LOG_WARNING, "grib_recompose_print: Problem to recompose print with : %s, no accessor found", loc);
                            return GRIB_NOT_FOUND;
                        }
                    }
                    else {
                        ret = grib_accessors_list_print(h, al, loc, type, format, separator, maxcols, &newline, out);

                        if (ret != GRIB_SUCCESS) {
                            /* grib_context_log(h->context, GRIB_LOG_ERROR,"grib_recompose_print: Could not recompose print : %s", uname); */
                            grib_accessors_list_delete(h->context, al);
                            return ret;
                        }
                    }
                    loc[0] = 0;
                    break;
                default:
                    loc[mode++] = uname[i];
                    break;
            }
        }
        else if (uname[i] == '[') {
            mode = 0;
        }
        else {
            fprintf(out, "%c", uname[i]);
            type = -1;
        }
    }
    if (newline)
        fprintf(out, "\n");

    grib_accessors_list_delete(h->context, al);
    return ret;
}
void grib_file_close(const char* filename, int force, int* err)
{
    grib_file* file       = NULL;
    grib_context* context = grib_context_get_default();

    /* Performance: keep the files open to avoid opening and closing files when writing the output. */
    /* So only call fclose() when too many files are open. */
    /* Also see ECC-411 */
    int do_close = (file_pool.number_of_opened_files > context->file_pool_max_opened_files);
    if (force == 1)
        do_close = 1; /* Can be overridden with the force argument */

    if (do_close) {
        /*printf("+++++++++++++ closing file %s (n=%d)\n",filename, file_pool.number_of_opened_files);*/
        GRIB_MUTEX_INIT_ONCE(&once, &init);
        GRIB_MUTEX_LOCK(&mutex1);
        file = grib_get_file(filename, err);
        if (file->handle) {
            if (fclose(file->handle) != 0) {
                *err = GRIB_IO_PROBLEM;
            }
            if (file->buffer) {
                free(file->buffer);
                file->buffer = 0;
            }
            file->handle = NULL;
            file_pool.number_of_opened_files--;
        }
        GRIB_MUTEX_UNLOCK(&mutex1);
    }
}
grib_file* grib_file_open(const char* filename, const char* mode, int* err)
{
    grib_file *file = 0, *prev = 0;
    int same_mode = 0;
    int is_new    = 0;
    GRIB_MUTEX_INIT_ONCE(&once, &init);

    if (!file_pool.context)
        file_pool.context = grib_context_get_default();

    if (file_pool.current && !grib_inline_strcmp(filename, file_pool.current->name)) {
        file = file_pool.current;
    }
    else {
        GRIB_MUTEX_LOCK(&mutex1);
        file = file_pool.first;
        while (file) {
            if (!grib_inline_strcmp(filename, file->name))
                break;
            prev = file;
            file = file->next;
        }
        if (!file) {
            is_new = 1;
            file   = grib_file_new(file_pool.context, filename, err);
            if (prev)
                prev->next = file;
            file_pool.current = file;
            if (!prev)
                file_pool.first = file;
            file_pool.size++;
        }
        GRIB_MUTEX_UNLOCK(&mutex1);
    }

    if (file->mode)
        same_mode = grib_inline_strcmp(mode, file->mode) ? 0 : 1;
    if (file->handle && same_mode) {
        *err = 0;
        return file;
    }

    GRIB_MUTEX_LOCK(&mutex1);
    if (!same_mode && file->handle) {
        fclose(file->handle);
    }

    if (!file->handle) {
        if (!is_new && *mode == 'w') {
            file->handle = fopen(file->name, "a");
        }
        else {
            file->handle = fopen(file->name, mode);
        }

        if (!file->handle) {
            grib_context_log(file->context, GRIB_LOG_PERROR, "grib_file_open: cannot open file %s", file->name);
            *err = GRIB_IO_PROBLEM;
            GRIB_MUTEX_UNLOCK(&mutex1);
            return NULL;
        }
        if (file->mode) free(file->mode);
        file->mode = strdup(mode);
        if (file_pool.context->io_buffer_size) {
#ifdef POSIX_MEMALIGN
            if (posix_memalign((void**)&(file->buffer), sysconf(_SC_PAGESIZE), file_pool.context->io_buffer_size)) {
                grib_context_log(file->context, GRIB_LOG_FATAL, "posix_memalign unable to allocate io_buffer\n");
            }
#else
            file->buffer = (char*)malloc(file_pool.context->io_buffer_size);
            if (!file->buffer) {
                grib_context_log(file->context, GRIB_LOG_FATAL, "Unable to allocate io_buffer\n");
            }
#endif
            setvbuf(file->handle, file->buffer, _IOFBF, file_pool.context->io_buffer_size);
        }

        file_pool.number_of_opened_files++;
    }

    GRIB_MUTEX_UNLOCK(&mutex1);
    return file;
}
int grib_get_message(const grib_handle* ch, const void** msg, size_t* size)
{
    long totalLength = 0;
    int ret          = 0;
    grib_handle* h   = (grib_handle*)ch;
    *msg             = h->buffer->data;
    *size            = h->buffer->ulength;

    ret = grib_get_long(h, "totalLength", &totalLength);
    if (!ret)
        *size = totalLength;

    if (h->context->gts_header_on && h->gts_header) {
        char strbuf[10];
        sprintf(strbuf, "%.8d", (int)(h->buffer->ulength + h->gts_header_len - 6));
        memcpy(h->gts_header, strbuf, 8);
    }
    return 0;
}
void grib_file_pool_delete_file(grib_file* file)
{
    grib_file* prev = NULL;
    GRIB_MUTEX_INIT_ONCE(&once, &init);
    GRIB_MUTEX_LOCK(&mutex1);

    if (file == file_pool.first) {
        file_pool.first   = file->next;
        file_pool.current = file->next;
    }
    else {
        prev              = file_pool.first;
        file_pool.current = file_pool.first;
        while (prev) {
            if (prev->next == file)
                break;
            prev = prev->next;
        }
        DebugAssert(prev);
        if (prev) {
            prev->next = file->next;
        }
    }

    if (file->handle) {
        file_pool.number_of_opened_files--;
    }
    grib_file_delete(file);
    GRIB_MUTEX_UNLOCK(&mutex1);
}
grib_file* grib_get_file(const char* filename, int* err)
{
    grib_file* file = NULL;

    if (file_pool.current->name && !grib_inline_strcmp(filename, file_pool.current->name)) {
        return file_pool.current;
    }

    file = file_pool.first;
    while (file) {
        if (!grib_inline_strcmp(filename, file->name))
            break;
        file = file->next;
    }
    if (!file)
        file = grib_file_new(0, filename, err);

    return file;
}
void grib_sarray_delete(grib_context* c, grib_sarray* v)
{
    if (!v)
        return;
    if (!c)
        c = grib_context_get_default();
    if (v->v)
        grib_context_free(c, v->v);
    grib_context_free(c, v);
}
int grib_set_string_array(grib_handle* h, const char* name, const char** val, size_t length)
{
    int ret = 0;
    grib_accessor* a;

    a = grib_find_accessor(h, name);

    if (h->context->debug) {
        fprintf(stderr, "ECCODES DEBUG grib_set_string_array key=%s %ld values\n", name, (long)length);
    }

    if (a) {
        if (a->flags & GRIB_ACCESSOR_FLAG_READ_ONLY)
            return GRIB_READ_ONLY;

        ret = grib_pack_string_array(a, val, &length);
        if (ret == GRIB_SUCCESS) {
            return grib_dependency_notify_change(a);
        }
        return ret;
    }
    return GRIB_NOT_FOUND;
}
void grib_darray_delete(grib_context* c, grib_darray* v)
{
    if (!v)
        return;
    if (!c)
        c = grib_context_get_default();
    if (v->v)
        grib_context_free(c, v->v);
    grib_context_free(c, v);
}
int grib_set_double_array(grib_handle* h, const char* name, const double* val, size_t length)
{
    return __grib_set_double_array(h, name, val, length, /*check=*/1);
}
int grib_set_missing(grib_handle* h, const char* name)
{
    int ret          = 0;
    grib_accessor* a = NULL;

    a = grib_find_accessor(h, name);

    if (a) {
        if (a->flags & GRIB_ACCESSOR_FLAG_READ_ONLY)
            return GRIB_READ_ONLY;

        if (a->flags & GRIB_ACCESSOR_FLAG_CAN_BE_MISSING) {
            if (h->context->debug)
                fprintf(stderr, "ECCODES DEBUG grib_set_missing %s\n", name);

            ret = grib_pack_missing(a);
            if (ret == GRIB_SUCCESS)
                return grib_dependency_notify_change(a);
        }
        else
            ret = GRIB_VALUE_CANNOT_BE_MISSING;

        grib_context_log(h->context, GRIB_LOG_ERROR, "unable to set %s=missing (%s)",
                         name, grib_get_error_message(ret));
        return ret;
    }

    grib_context_log(h->context, GRIB_LOG_ERROR, "unable to find accessor %s", name);
    return GRIB_NOT_FOUND;
}
grib_accessor* grib_find_accessor_fast(grib_handle* h, const char* name)
{
    grib_accessor* a = NULL;
    char* p          = NULL;
    DebugAssert(name);

    p = strchr((char*)name, '.');
    if (p) {
        int i = 0, len = 0;
        char name_space[MAX_NAMESPACE_LEN];
        p--;
        len = p - name + 1;

        for (i = 0; i < len; i++)
            name_space[i] = *(name + i);

        name_space[len] = '\0';

        a = h->accessors[grib_hash_keys_get_id(h->context->keys, name)];
        if (a && !matching(a, name, name_space))
            a = NULL;
    }
    else {
        a = h->accessors[grib_hash_keys_get_id(h->context->keys, name)];
    }

    if (a == NULL && h->main)
        a = grib_find_accessor_fast(h->main, name);

    return a;
}
int grib_pack_expression(grib_accessor* a, grib_expression* e)
{
    grib_accessor_class* c = a->cclass;
    /*grib_context_log(a->context, GRIB_LOG_DEBUG, "(%s)%s is packing (double) %g",(a->parent->owner)?(a->parent->owner->name):"root", a->name ,v?(*v):0); */
    while (c) {
        if (c->pack_expression) {
            return c->pack_expression(a, e);
        }
        c = c->super ? *(c->super) : NULL;
    }
    DebugAssert(0);
    return 0;
}
grib_expression* grib_arguments_get_expression(grib_handle* h, grib_arguments* args, int n)
{
    while (args && n-- > 0) {
        args = args->next;
    }

    if (!args)
        return 0;

    return args->expression;
}
int grib_pack_double(grib_accessor* a, const double* v, size_t* len)
{
    grib_accessor_class* c = a->cclass;
    /*grib_context_log(a->context, GRIB_LOG_DEBUG, "(%s)%s is packing (double) %g",(a->parent->owner)?(a->parent->owner->name):"root", a->name ,v?(*v):0); */
    while (c) {
        if (c->pack_double) {
            return c->pack_double(a, v, len);
        }
        c = c->super ? *(c->super) : NULL;
    }
    DebugAssert(0);
    return 0;
}
size_t grib_darray_used_size(grib_darray* v)
{
    return v->n;
}
long grib_get_next_position_offset(grib_accessor* a)
{
    grib_accessor_class* c = NULL;
    /*grib_context_log(a->context, GRIB_LOG_DEBUG, "(%s)%s is checking next (long)",(a->parent->owner)?(a->parent->owner->name):"root", a->name ); */
    if (a)
        c = a->cclass;

    while (c) {
        if (c->next_offset)
            return c->next_offset(a);
        c = c->super ? *(c->super) : NULL;
    }
    DebugAssert(0);
    return 0;
}
void grib_init_accessor(grib_accessor* a, const long len, grib_arguments* args)
{
    init_accessor(a->cclass, a, len, args);
}
static void init_accessor(grib_accessor_class* c, grib_accessor* a, const long len, grib_arguments* args)
{
    if (c) {
        grib_accessor_class* s = c->super ? *(c->super) : NULL;
        init_accessor(s, a, len, args);
        if (c->init)
            c->init(a, len, args);
    }
}
int grib_dependency_notify_change(grib_accessor* observed)
{
    grib_handle* h     = handle_of(observed);
    grib_dependency* d = h->dependencies;
    int ret            = GRIB_SUCCESS;

    /*Do a two pass mark&sweep, in case some dependencies are added while we notify*/
    while (d) {
        d->run = (d->observed == observed && d->observer != 0);
        d      = d->next;
    }

    d = h->dependencies;
    while (d) {
        if (d->run) {
            /*printf("grib_dependency_notify_change %s %s %p\n", observed->name, d->observer ? d->observer->name : "?", (void*)d->observer);*/
            if (d->observer && (ret = grib_accessor_notify_change(d->observer, observed)) != GRIB_SUCCESS)
                return ret;
        }
        d = d->next;
    }
    return ret;
}
int grib_pack_missing(grib_accessor* a)
{
    grib_accessor_class* c = a->cclass;
    /*grib_context_log(a->context, GRIB_LOG_DEBUG, "(%s)%s is packing (double) %g",(a->parent->owner)?(a->parent->owner->name):"root", a->name ,v?(*v):0); */
    while (c) {
        if (c->pack_missing) {
            return c->pack_missing(a);
        }
        c = c->super ? *(c->super) : NULL;
    }
    DebugAssert(0);
    return 0;
}
int grib_accessor_notify_change(grib_accessor* a, grib_accessor* changed)
{
    grib_accessor_class* c = NULL;
    if (a)
        c = a->cclass;

    while (c) {
        if (c->notify_change)
            return c->notify_change(a, changed);
        c = c->super ? *(c->super) : NULL;
    }
    if (a && a->cclass)
        printf("notify_change not implemented for %s %s\n", a->cclass->name, a->name);
    DebugAssert(0);
    return 0;
}
static grib_handle* handle_of(grib_accessor* observed)
{
    grib_handle* h = NULL;
    DebugAssert(observed);
    /* printf("+++++ %s->parent = %p\n",observed->name,observed->parent); */
    /* printf("+++++ %s = %p\n",observed->name,observed); */
    /* printf("+++++       h=%p\n",observed->h); */
    /* special case for BUFR attributes parentless */
    if (observed->parent == NULL) {
        return observed->h;
    }
    h = observed->parent->h;
    while (h->main)
        h = h->main;
    return h;
}
static int __grib_set_double_array(grib_handle* h, const char* name, const double* val, size_t length, int check)
{
    double v = 0;
    size_t i = 0;

    if (h->context->debug) {
        print_debug_info__set_double_array(h, "__grib_set_double_array", name, val, length);
    }

    if (length == 0) {
        grib_accessor* a = grib_find_accessor(h, name);
        return grib_pack_double(a, val, &length);
    }

    /*second order doesn't have a proper representation for constant fields
      the best is not to do the change of packing type if the field is constant
     */
    if (!strcmp(name, "values") || !strcmp(name, "codedValues")) {
        double missingValue;
        int ret      = 0;
        int constant = 0;

        ret = grib_get_double(h, "missingValue", &missingValue);
        if (ret)
            missingValue = 9999;

        v        = missingValue;
        constant = 1;
        for (i = 0; i < length; i++) {
            if (val[i] != missingValue) {
                if (v == missingValue) {
                    v = val[i];
                }
                else if (v != val[i]) {
                    constant = 0;
                    break;
                }
            }
        }
        if (constant) {
            char packingType[50] = {0,};
            size_t slen = 50;

            grib_get_string(h, "packingType", packingType, &slen);
            if (!strcmp(packingType, "grid_second_order") ||
                !strcmp(packingType, "grid_second_order_no_SPD") ||
                !strcmp(packingType, "grid_second_order_SPD1") ||
                !strcmp(packingType, "grid_second_order_SPD2") ||
                !strcmp(packingType, "grid_second_order_SPD3")) {
                slen = 11; /*length of 'grid_simple' */
                if (h->context->debug) {
                    fprintf(stderr, "ECCODES DEBUG __grib_set_double_array: Cannot use second order packing for constant fields. Using simple packing\n");
                }
                ret = grib_set_string(h, "packingType", "grid_simple", &slen);
                if (ret != GRIB_SUCCESS) {
                    if (h->context->debug) {
                        fprintf(stderr, "ECCODES DEBUG __grib_set_double_array: could not switch to simple packing!\n");
                    }
                }
            }
        }
    }

    return _grib_set_double_array(h, name, val, length, check);
}
static int _grib_set_double_array(grib_handle* h, const char* name,
                                  const double* val, size_t length, int check)
{
    size_t encoded   = 0;
    grib_accessor* a = grib_find_accessor(h, name);
    int err          = 0;

    if (!a)
        return GRIB_NOT_FOUND;
    if (name[0] == '/' || name[0] == '#') {
        if (check && (a->flags & GRIB_ACCESSOR_FLAG_READ_ONLY))
            return GRIB_READ_ONLY;
        err     = grib_pack_double(a, val, &length);
        encoded = length;
    }
    else
        err = _grib_set_double_array_internal(h, a, val, length, &encoded, check);

    if (err == GRIB_SUCCESS && length > encoded)
        err = GRIB_ARRAY_TOO_SMALL;

    if (err == GRIB_SUCCESS)
        return _grib_dependency_notify_change(h, a); /* See ECC-778 */

    return err;
}
int _grib_dependency_notify_change(grib_handle* h, grib_accessor* observed)
{
    grib_dependency* d = h->dependencies;
    int ret            = GRIB_SUCCESS;

    /*Do a two pass mark&sweep, in case some dependencies are added while we notify*/
    while (d) {
        d->run = (d->observed == observed && d->observer != 0);
        d      = d->next;
    }

    d = h->dependencies;
    while (d) {
        if (d->run) {
            /*printf("grib_dependency_notify_change %s %s %p\n",observed->name,d->observer ? d->observer->name : "?", (void*)d->observer);*/
            if (d->observer && (ret = grib_accessor_notify_change(d->observer, observed)) != GRIB_SUCCESS)
                return ret;
        }
        d = d->next;
    }
    return ret;
}
static int _grib_set_double_array_internal(grib_handle* h, grib_accessor* a,
                                           const double* val, size_t buffer_len, size_t* encoded_length, int check)
{
    if (a) {
        int err = _grib_set_double_array_internal(h, a->same, val, buffer_len, encoded_length, check);

        if (check && (a->flags & GRIB_ACCESSOR_FLAG_READ_ONLY))
            return GRIB_READ_ONLY;

        if (err == GRIB_SUCCESS) {
            size_t len = buffer_len - *encoded_length;
            if (len) {
                err = grib_pack_double(a, val + *encoded_length, &len);
                *encoded_length += len;
                if (err == GRIB_SUCCESS) {
                    /* See ECC-778 */
                    return _grib_dependency_notify_change(h, a);
                }
            }
            else {
                grib_get_size(h, a->name, encoded_length);
                err = GRIB_WRONG_ARRAY_SIZE;
            }
        }

        return err;
    }
    else {
        return GRIB_SUCCESS;
    }
}
int grib_set_string(grib_handle* h, const char* name, const char* val, size_t* length)
{
    int ret          = 0;
    grib_accessor* a = NULL;

    int processed = process_packingType_change(h, name, val);
    if (processed)
        return GRIB_SUCCESS;  /* Dealt with - no further action needed */

    a = grib_find_accessor(h, name);

    if (a) {
        if (h->context->debug) {
            if (strcmp(name, a->name)!=0)
                fprintf(stderr, "ECCODES DEBUG grib_set_string %s=|%s| (a->name=%s)\n", name, val, a->name);
            else
                fprintf(stderr, "ECCODES DEBUG grib_set_string %s=|%s|\n", name, val);
        }

        if (a->flags & GRIB_ACCESSOR_FLAG_READ_ONLY)
            return GRIB_READ_ONLY;

        ret = grib_pack_string(a, val, length);
        if (ret == GRIB_SUCCESS) {
            return grib_dependency_notify_change(a);
        }
        return ret;
    }
    return GRIB_NOT_FOUND;
}
int grib_pack_string(grib_accessor* a, const char* v, size_t* len)
{
    grib_accessor_class* c = a->cclass;
    /*grib_context_log(a->context, GRIB_LOG_DEBUG, "(%s)%s is packing (string) %s",(a->parent->owner)?(a->parent->owner->name):"root", a->name ,v?v:"(null)");*/
    while (c) {
        if (c->pack_string) {
            return c->pack_string(a, v, len);
        }
        c = c->super ? *(c->super) : NULL;
    }
    DebugAssert(0);
    return 0;
}
static int process_packingType_change(grib_handle* h, const char* keyname, const char* keyval)
{
    int err = 0;
    char input_packing_type[100] = {0,};
    size_t len = sizeof(input_packing_type);

    if (grib_inline_strcmp(keyname, "packingType") == 0) {
        /* Second order doesn't have a proper representation for constant fields.
           So best not to do the change of packing type.
           Use strncmp to catch all flavours of 2nd order packing e.g. grid_second_order_boustrophedonic */
        if (strncmp(keyval, "grid_second_order", 17) == 0) {
            long bitsPerValue   = 0;
            size_t numCodedVals = 0;
            err = grib_get_long(h, "bitsPerValue", &bitsPerValue);
            if (!err && bitsPerValue == 0) {
                /* ECC-1219: packingType conversion from grid_ieee to grid_second_order.
                 * Normally having a bitsPerValue of 0 means a constant field but this is 
                 * not so for IEEE packing which can be non-constant but always has bitsPerValue==0! */
                len = sizeof(input_packing_type);
                grib_get_string(h, "packingType", input_packing_type, &len);
                if (strcmp(input_packing_type, "grid_ieee") != 0) {
                    /* Not IEEE, so bitsPerValue==0 really means constant field */
                    if (h->context->debug) {
                        fprintf(stderr, "ECCODES DEBUG grib_set_string packingType: "
                                "Constant field cannot be encoded in second order. Packing not changed\n");
                    }
                    return 1; /* Dealt with - no further action needed */
                }
            }
            /* GRIB-883: check if there are enough coded values */
            err = grib_get_size(h, "codedValues", &numCodedVals);
            if (!err && numCodedVals < 3) {
                if (h->context->debug) {
                    fprintf(stderr, "ECCODES DEBUG grib_set_string packingType: "
                            "Not enough coded values for second order. Packing not changed\n");
                }
                return 1; /* Dealt with - no further action needed */
            }
        }

        /* ECC-1407: Are we changing from IEEE to CCSDS or Simple? */
        if (strcmp(keyval, "grid_simple")==0 || strcmp(keyval, "grid_ccsds")==0) {
            grib_get_string(h, "packingType", input_packing_type, &len);
            if (strcmp(input_packing_type, "grid_ieee") == 0) {
                const long max_bpv = 32; /* Cannot do any higher */
                grib_set_long(h, "bitsPerValue", max_bpv);
                /*
                long accuracy = 0;
                err = grib_get_long(h, "accuracy", &accuracy);
                if (!err) {
                    grib_set_long(h, "bitsPerValue", accuracy);
                } */
            }
        }
    }
    return 0;  /* Further action is needed */
}
int grib_set_long(grib_handle* h, const char* name, long val)
{
    int ret          = GRIB_SUCCESS;
    grib_accessor* a = NULL;
    size_t l         = 1;

    a = grib_find_accessor(h, name);

    if (a) {
        if (h->context->debug) {
            if (strcmp(name, a->name)!=0)
                fprintf(stderr, "ECCODES DEBUG grib_set_long %s=%ld (a->name=%s)\n", name, (long)val, a->name);
            else
                fprintf(stderr, "ECCODES DEBUG grib_set_long %s=%ld\n", name, (long)val);
        }

        if (a->flags & GRIB_ACCESSOR_FLAG_READ_ONLY)
            return GRIB_READ_ONLY;

        ret = grib_pack_long(a, &val, &l);
        if (ret == GRIB_SUCCESS)
            return grib_dependency_notify_change(a);

        return ret;
    }
    return GRIB_NOT_FOUND;
}
static void print_debug_info__set_double_array(grib_handle* h, const char* func, const char* name, const double* val, size_t length)
{
    size_t N = 7, i = 0;
    double minVal = DBL_MAX, maxVal = -DBL_MAX;
    Assert( h->context->debug );

    if (length <= N)
        N = length;
    fprintf(stderr, "ECCODES DEBUG %s key=%s %lu values (", func, name, (unsigned long)length);
    for (i = 0; i < N; ++i) {
        if (i != 0) fprintf(stderr,", ");
        fprintf(stderr, "%.10g", val[i]);
    }
    if (N >= length) fprintf(stderr, ") ");
    else fprintf(stderr, "...) ");
    for (i = 0; i < length; ++i) {
        if (val[i] < minVal) minVal = val[i];
        if (val[i] > maxVal) maxVal = val[i];
    }
    fprintf(stderr, "min=%.10g, max=%.10g\n",minVal,maxVal);
}
int grib_pack_string_array(grib_accessor* a, const char** v, size_t* len)
{
    grib_accessor_class* c = a->cclass;
    /*grib_context_log(a->context, GRIB_LOG_DEBUG, "(%s)%s is packing (string) %s",(a->parent->owner)?(a->parent->owner->name):"root", a->name ,v?v:"(null)");*/
    while (c) {
        if (c->pack_string_array) {
            return c->pack_string_array(a, v, len);
        }
        c = c->super ? *(c->super) : NULL;
    }
    DebugAssert(0);
    return 0;
}
grib_file* grib_file_new(grib_context* c, const char* name, int* err)
{
    grib_file* file;

    if (!c)
        c = grib_context_get_default();

    file = (grib_file*)grib_context_malloc_clear(c, sizeof(grib_file));

    if (!file) {
        grib_context_log(c, GRIB_LOG_ERROR, "grib_file_new: unable to allocate memory");
        *err = GRIB_OUT_OF_MEMORY;
        return NULL;
    }
    GRIB_MUTEX_INIT_ONCE(&once, &init);

    file->name = strdup(name);
    file->id   = next_id;

    GRIB_MUTEX_LOCK(&mutex1);
    next_id++;
    GRIB_MUTEX_UNLOCK(&mutex1);

    file->mode     = 0;
    file->handle   = 0;
    file->refcount = 0;
    file->context  = c;
    file->next     = 0;
    file->buffer   = 0;
    return file;
}
void grib_file_delete(grib_file* file)
{
    {
        if (!file)
            return;
    }
    GRIB_MUTEX_INIT_ONCE(&once, &init);
    GRIB_MUTEX_LOCK(&mutex1);
    if (file->name)
        free(file->name);
    if (file->mode)
        free(file->mode);

    if (file->buffer) {
        free(file->buffer);
    }
    grib_context_free(file->context, file);
    /* file = NULL; */
    GRIB_MUTEX_UNLOCK(&mutex1);
}
int grib_accessors_list_print(grib_handle* h, grib_accessors_list* al, const char* name,
                              int type, const char* format, const char* separator, int maxcols, int* newline, FILE* out)
{
    size_t size = 0, len = 0, replen = 0, j = 0;
    unsigned char* bval      = NULL;
    double* dval             = 0;
    long* lval               = 0;
    char** cvals             = NULL;
    int ret                  = 0;
    char* myformat           = NULL;
    char* myseparator        = NULL;
    char double_format[]     = "%.12g"; /* default format for printing double keys */
    char long_format[]       = "%ld";   /* default format for printing integer keys */
    char default_separator[] = " ";
    grib_accessor* a         = al->accessor;
    DebugAssert(a);

    /* Number of columns specified as 0 means print on ONE line i.e. num cols = infinity */
    if (maxcols == 0)
        maxcols = INT_MAX;

    if (type == -1)
        type = grib_accessor_get_native_type(al->accessor);
    grib_accessors_list_value_count(al, &size);
    switch (type) {
        case GRIB_TYPE_STRING:
            myseparator = separator ? (char*)separator : default_separator;
            if (size == 1) {
                char sbuf[1024] = {0,};
                len = sizeof(sbuf);
                ret = grib_unpack_string(al->accessor, sbuf, &len);
                if (grib_is_missing_string(al->accessor, (unsigned char*)sbuf, len)) {
                    fprintf(out, "%s", "MISSING");
                }
                else {
                    fprintf(out, "%s", sbuf);
                }
            }
            else {
                int cols = 0;
                j = 0;
                cvals    = (char**)grib_context_malloc_clear(h->context, sizeof(char*) * size);
                grib_accessors_list_unpack_string(al, cvals, &size);
                for (j = 0; j < size; j++) {
                    *newline = 1;
                    fprintf(out, "%s", cvals[j]);
                    if (j < size - 1)
                        fprintf(out, "%s", myseparator);
                    cols++;
                    if (cols >= maxcols) {
                        fprintf(out, "\n");
                        *newline = 1;
                        cols     = 0;
                    }
                    grib_context_free(h->context, cvals[j]);
                }
            }
            grib_context_free(h->context, cvals);
            break;
        case GRIB_TYPE_DOUBLE:
            myformat    = format ? (char*)format : double_format;
            myseparator = separator ? (char*)separator : default_separator;
            dval        = (double*)grib_context_malloc_clear(h->context, sizeof(double) * size);
            ret         = grib_accessors_list_unpack_double(al, dval, &size);
            if (size == 1)
                fprintf(out, myformat, dval[0]);
            else {
                int cols = 0;
                j = 0;
                for (j = 0; j < size; j++) {
                    *newline = 1;
                    fprintf(out, myformat, dval[j]);
                    if (j < size - 1)
                        fprintf(out, "%s", myseparator);
                    cols++;
                    if (cols >= maxcols) {
                        fprintf(out, "\n");
                        *newline = 1;
                        cols     = 0;
                    }
                }
            }
            grib_context_free(h->context, dval);
            break;
        case GRIB_TYPE_LONG:
            myformat    = format ? (char*)format : long_format;
            myseparator = separator ? (char*)separator : default_separator;
            lval        = (long*)grib_context_malloc_clear(h->context, sizeof(long) * size);
            ret         = grib_accessors_list_unpack_long(al, lval, &size);
            if (size == 1)
                fprintf(out, myformat, lval[0]);
            else {
                int cols = 0;
                j = 0;
                for (j = 0; j < size; j++) {
                    *newline = 1;
                    fprintf(out, myformat, lval[j]);
                    if (j < size - 1)
                        fprintf(out, "%s", myseparator);
                    cols++;
                    if (cols >= maxcols) {
                        fprintf(out, "\n");
                        *newline = 1;
                        cols     = 0;
                    }
                }
            }
            grib_context_free(h->context, lval);
            break;
        case GRIB_TYPE_BYTES:
            replen = a->length;
            bval   = (unsigned char*)grib_context_malloc(h->context, replen * sizeof(unsigned char));
            ret    = grib_unpack_bytes(al->accessor, bval, &replen);
            for (j = 0; j < replen; j++) {
                fprintf(out, "%02x", bval[j]);
            }
            grib_context_free(h->context, bval);
            *newline = 1;
            break;
        default:
            grib_context_log(h->context, GRIB_LOG_WARNING,
                             "grib_accessor_print: Problem printing \"%s\", invalid type %d", a->name, grib_get_type_name(type));
    }
    return ret;
}
int string_to_long(const char* input, long* output)
{
    const int base = 10;
    char* endptr;
    long val = 0;

    if (!input)
        return GRIB_INVALID_ARGUMENT;

    errno = 0;
    val   = strtol(input, &endptr, base);
    if ((errno == ERANGE && (val == LONG_MAX || val == LONG_MIN)) ||
        (errno != 0 && val == 0)) {
        /*perror("strtol");*/
        return GRIB_INVALID_ARGUMENT;
    }
    if (endptr == input) {
        /*fprintf(stderr, "No digits were found. EXIT_FAILURE\n");*/
        return GRIB_INVALID_ARGUMENT;
    }
    *output = val;
    return GRIB_SUCCESS;
}
int grib_type_to_int(char id)
{
    switch (id) {
        case 'd':
            return GRIB_TYPE_DOUBLE;
            break;
        case 'f':
            return GRIB_TYPE_DOUBLE;
            break;
        case 'l':
            return GRIB_TYPE_LONG;
            break;
        case 'i':
            return GRIB_TYPE_LONG;
            break;
        case 's':
            return GRIB_TYPE_STRING;
            break;
    }
    return GRIB_TYPE_UNDEFINED;
}
void grib_dump(grib_action* a, FILE* f, int l)
{
    grib_action_class* c = a->cclass;
    init(c);

    while (c) {
        if (c->dump_gac) {
            c->dump_gac(a, f, l);
            return;
        }
        c = c->super ? *(c->super) : NULL;
    }
    DebugAssert(0);
}
const char* grib_get_type_name(int type)
{
    switch (type) {
        case GRIB_TYPE_LONG:
            return "long";
        case GRIB_TYPE_STRING:
            return "string";
        case GRIB_TYPE_BYTES:
            return "bytes";
        case GRIB_TYPE_DOUBLE:
            return "double";
        case GRIB_TYPE_LABEL:
            return "label";
        case GRIB_TYPE_SECTION:
            return "section";
    }
    return "unknown";
}
int grib_unpack_bytes(grib_accessor* a, unsigned char* v, size_t* len)
{
    grib_accessor_class* c = a->cclass;
    /*grib_context_log(a->context, GRIB_LOG_DEBUG, "(%s)%s is unpacking (bytes)",(a->parent->owner)?(a->parent->owner->name):"root", a->name ); */
    while (c) {
        if (c->unpack_bytes) {
            return c->unpack_bytes(a, v, len);
        }
        c = c->super ? *(c->super) : NULL;
    }
    DebugAssert(0);
    return 0;
}
int grib_accessors_list_unpack_long(grib_accessors_list* al, long* val, size_t* buffer_len)
{
    int err             = GRIB_SUCCESS;
    size_t unpacked_len = 0;
    size_t len          = 0;

    while (al && err == GRIB_SUCCESS) {
        len = *buffer_len - unpacked_len;
        err = grib_unpack_long(al->accessor, val + unpacked_len, &len);
        unpacked_len += len;
        al = al->next;
    }

    *buffer_len = unpacked_len;
    return err;
}
int grib_accessors_list_unpack_string(grib_accessors_list* al, char** val, size_t* buffer_len)
{
    int err             = GRIB_SUCCESS;
    size_t unpacked_len = 0;
    size_t len          = 0;

    while (al && err == GRIB_SUCCESS) {
        len = *buffer_len - unpacked_len;
        err = grib_unpack_string_array(al->accessor, val + unpacked_len, &len);
        unpacked_len += len;
        al = al->next;
    }

    *buffer_len = unpacked_len;
    return err;
}
int grib_is_missing_string(grib_accessor* a, const unsigned char* x, size_t len)
{
    /* For a string value to be missing, every character has to be */
    /* all 1's (i.e. 0xFF) */
    /* Note: An empty string is also classified as missing */
    int ret;
    size_t i = 0;

    if (len == 0)
        return 1; /* empty string */
    ret = 1;
    for (i = 0; i < len; i++) {
        if (x[i] != 0xFF) {
            ret = 0;
            break;
        }
    }

    if (!a) return ret;

    ret = ( ((a->flags & GRIB_ACCESSOR_FLAG_CAN_BE_MISSING) && ret == 1) ) ? 1 : 0;
    return ret;
}
int grib_unpack_string_array(grib_accessor* a, char** v, size_t* len)
{
    grib_accessor_class* c = a->cclass;
    while (c) {
        if (c->unpack_string_array) {
            return c->unpack_string_array(a, v, len);
        }
        c = c->super ? *(c->super) : NULL;
    }
    DebugAssert(0);
    return 0;
}
long grib_accessor_get_native_type(grib_accessor* a)
{
    grib_accessor_class* c = NULL;
    if (a)
        c = a->cclass;

    while (c) {
        if (c->get_native_type)
            return c->get_native_type(a);
        c = c->super ? *(c->super) : NULL;
    }
    DebugAssert(0);
    return 0;
}
static void link_same_attributes(grib_accessor* a, grib_accessor* b)
{
    int i                     = 0;
    int idx                   = 0;
    grib_accessor* bAttribute = NULL;
    if (a == NULL || b == NULL)
        return;
    if (!grib_accessor_has_attributes(b))
        return;
    while (i < MAX_ACCESSOR_ATTRIBUTES && a->attributes[i]) {
        bAttribute = _grib_accessor_get_attribute(b, a->attributes[i]->name, &idx);
        if (bAttribute)
            a->attributes[i]->same = bAttribute;
        i++;
    }
}
int grib_accessor_has_attributes(grib_accessor* a)
{
    return a->attributes[0] ? 1 : 0;
}
void grib_concept_condition_delete(grib_context* c, grib_concept_condition* v)
{
    grib_expression_free(c, v->expression);
    grib_context_free_persistent(c, v->name);
    grib_context_free_persistent(c, v);
}
void grib_iarray_delete(grib_iarray* v)
{
    grib_context* c;

    if (!v)
        return;
    c = v->context;

    grib_iarray_delete_array(v);

    grib_context_free(c, v);
}
void grib_iarray_delete_array(grib_iarray* v)
{
    grib_context* c;

    if (!v)
        return;
    c = v->context;

    if (v->v) {
        long* vv = v->v - v->number_of_pop_front;
        grib_context_free(c, vv);
    }
}
int grib_itrie_insert(grib_itrie* t, const char* key)
{
    const char* k    = key;
    grib_itrie* last = t;
    int* count;

    if (!t) {
        Assert(!"grib_itrie_insert: grib_trie==NULL");
        return -1;
    }

    GRIB_MUTEX_INIT_ONCE(&once, &init);
    GRIB_MUTEX_LOCK(&mutex);

    count = t->count;

    while (*k && t) {
        last = t;
        t    = t->next[mapping[(int)*k]];
        if (t)
            k++;
    }

    if (*k != 0) {
        t = last;
        while (*k) {
            int j      = mapping[(int)*k++];
            t->next[j] = grib_itrie_new(t->context, count);
            t          = t->next[j];
        }
    }
    if (*(t->count) < MAX_NUM_CONCEPTS) {
        t->id = *(t->count);
        (*(t->count))++;
    }
    else {
        grib_context_log(t->context, GRIB_LOG_ERROR,
                         "grib_itrie_insert: too many accessors, increase MAX_NUM_CONCEPTS\n");
        Assert(*(t->count) < MAX_NUM_CONCEPTS);
    }

    GRIB_MUTEX_UNLOCK(&mutex);

    /*printf("grib_itrie_get_id: %s -> %d\n",key,t->id);*/

    return t->id;
}
static double* pointer_to_data(unsigned int i, unsigned int j,
                               long iScansNegatively, long jScansPositively,
                               long jPointsAreConsecutive, long alternativeRowScanning,
                               unsigned int nx, unsigned int ny, double* data)
{
    /* Regular grid */
    if (nx > 0 && ny > 0) {
        if (i >= nx || j >= ny)
            return NULL;
        j = (jScansPositively) ? j : ny - 1 - j;
        i = ((alternativeRowScanning) && (j % 2 == 1)) ? nx - 1 - i : i;
        i = (iScansNegatively) ? nx - 1 - i : i;

        return (jPointsAreConsecutive) ? data + j + i * ny : data + i + nx * j;
    }

    /* Reduced or other data not on a grid */
    return NULL;
}
int grib_accessor_is_missing(grib_accessor* a, int* err)
{
    *err = GRIB_SUCCESS;
    if (a) {
        if (a->flags & GRIB_ACCESSOR_FLAG_CAN_BE_MISSING)
            return grib_is_missing_internal(a);
        else
            return 0;
    }
    else {
        *err = GRIB_NOT_FOUND;
        return 1;
    }
}
int grib_is_missing_internal(grib_accessor* a)
{
    grib_accessor_class* c = a->cclass;
    /*grib_context_log(a->context, GRIB_LOG_DEBUG, "(%s)%s is packing (double) %g",(a->parent->owner)?(a->parent->owner->name):"root", a->name ,v?(*v):0); */
    while (c) {
        if (c->is_missing) {
            return c->is_missing(a);
        }
        c = c->super ? *(c->super) : NULL;
    }
    DebugAssert(0);
    return 0;
}
static grib_darray* grib_darray_resize(grib_darray* v)
{
    const size_t newsize = v->incsize + v->size;
    grib_context* c = v->context;
    if (!c)
        c = grib_context_get_default();

    v->v    = (double*)grib_context_realloc(c, v->v, newsize * sizeof(double));
    v->size = newsize;
    if (!v->v) {
        grib_context_log(c, GRIB_LOG_ERROR,
                         "grib_darray_resize unable to allocate %ld bytes\n", sizeof(double) * newsize);
        return NULL;
    }
    return v;
}
grib_darray* grib_darray_new(grib_context* c, size_t size, size_t incsize)
{
    grib_darray* v = NULL;
    if (!c)
        c = grib_context_get_default();
    v = (grib_darray*)grib_context_malloc_clear(c, sizeof(grib_darray));
    if (!v) {
        grib_context_log(c, GRIB_LOG_ERROR,
                         "grib_darray_new unable to allocate %ld bytes\n", sizeof(grib_darray));
        return NULL;
    }
    v->size    = size;
    v->n       = 0;
    v->incsize = incsize;
    v->context = c;
    v->v       = (double*)grib_context_malloc_clear(c, sizeof(double) * size);
    if (!v->v) {
        grib_context_log(c, GRIB_LOG_ERROR,
                         "grib_darray_new unable to allocate %ld bytes\n", sizeof(double) * size);
        return NULL;
    }
    return v;
}
static grib_sarray* grib_sarray_resize(grib_sarray* v)
{
    const size_t newsize = v->incsize + v->size;
    grib_context* c = v->context;
    if (!c)
        c = grib_context_get_default();

    v->v    = (char**)grib_context_realloc(c, v->v, newsize * sizeof(char*));
    v->size = newsize;
    if (!v->v) {
        grib_context_log(c, GRIB_LOG_ERROR,
                         "grib_sarray_resize unable to allocate %ld bytes\n", sizeof(char*) * newsize);
        return NULL;
    }
    return v;
}
grib_sarray* grib_sarray_new(grib_context* c, size_t size, size_t incsize)
{
    grib_sarray* v = NULL;
    if (!c)
        c = grib_context_get_default();
    v = (grib_sarray*)grib_context_malloc_clear(c, sizeof(grib_sarray));
    if (!v) {
        grib_context_log(c, GRIB_LOG_ERROR,
                         "grib_sarray_new unable to allocate %ld bytes\n", sizeof(grib_sarray));
        return NULL;
    }
    v->size    = size;
    v->n       = 0;
    v->incsize = incsize;
    v->context = c;
    v->v       = (char**)grib_context_malloc_clear(c, sizeof(char*) * size);
    if (!v->v) {
        grib_context_log(c, GRIB_LOG_ERROR,
                         "grib_sarray_new unable to allocate %ld bytes\n", sizeof(char*) * size);
        return NULL;
    }
    return v;
}
static grib_iarray* grib_iarray_resize_to(grib_iarray* v, size_t newsize)
{
    long* newv;
    size_t i;
    grib_context* c = v->context;

    if (newsize < v->size)
        return v;

    if (!c)
        c = grib_context_get_default();

    newv = (long*)grib_context_malloc_clear(c, newsize * sizeof(long));
    if (!newv) {
        grib_context_log(c, GRIB_LOG_ERROR,
                         "grib_iarray_resize unable to allocate %ld bytes\n", sizeof(long) * newsize);
        return NULL;
    }

    for (i = 0; i < v->n; i++)
        newv[i] = v->v[i];

    v->v -= v->number_of_pop_front;
    grib_context_free(c, v->v);

    v->v                   = newv;
    v->size                = newsize;
    v->number_of_pop_front = 0;

    return v;
}
static grib_iarray* grib_iarray_resize(grib_iarray* v)
{
    const size_t newsize = v->incsize + v->size;
    return grib_iarray_resize_to(v, newsize);
}
grib_iarray* grib_iarray_new(grib_context* c, size_t size, size_t incsize)
{
    grib_iarray* v = NULL;

    if (!c)
        c = grib_context_get_default();

    v = (grib_iarray*)grib_context_malloc(c, sizeof(grib_iarray));
    if (!v) {
        grib_context_log(c, GRIB_LOG_ERROR,
                         "grib_iarray_new unable to allocate %ld bytes\n", sizeof(grib_iarray));
        return NULL;
    }
    v->context             = c;
    v->size                = size;
    v->n                   = 0;
    v->incsize             = incsize;
    v->v                   = (long*)grib_context_malloc(c, sizeof(long) * size);
    v->number_of_pop_front = 0;
    if (!v->v) {
        grib_context_log(c, GRIB_LOG_ERROR,
                         "grib_iarray_new unable to allocate %ld bytes\n", sizeof(long) * size);
        return NULL;
    }
    return v;
}
void* grib_context_realloc(const grib_context* c, void* p, size_t size)
{
    void* q;
    if (!c)
        c = grib_context_get_default();
    q = c->realloc_mem(c, p, size);
    if (!q) {
        grib_context_log(c, GRIB_LOG_FATAL, "grib_context_realloc: error allocating %lu bytes", (unsigned long)size);
        return NULL;
    }
    return q;
}


grib_nearest_class* grib_nearest_class_gen;
grib_nearest_class* grib_nearest_class_lambert_azimuthal_equal_area;
grib_nearest_class* grib_nearest_class_lambert_conformal;
grib_nearest_class* grib_nearest_class_latlon_reduced;
grib_nearest_class* grib_nearest_class_mercator;
grib_nearest_class* grib_nearest_class_polar_stereographic;
grib_nearest_class* grib_nearest_class_reduced;
grib_nearest_class* grib_nearest_class_regular;
grib_nearest_class* grib_nearest_class_sh;
grib_nearest_class* grib_nearest_class_space_view;
static const struct nearest_table_entry nearest_table[] = {
{ "gen", &grib_nearest_class_gen, },
{ "lambert_azimuthal_equal_area", &grib_nearest_class_lambert_azimuthal_equal_area, },
{ "lambert_conformal", &grib_nearest_class_lambert_conformal, },
{ "latlon_reduced", &grib_nearest_class_latlon_reduced, },
{ "mercator", &grib_nearest_class_mercator, },
{ "polar_stereographic", &grib_nearest_class_polar_stereographic, },
{ "reduced", &grib_nearest_class_reduced, },
{ "regular", &grib_nearest_class_regular, },
{ "sh", &grib_nearest_class_sh, },
{ "space_view", &grib_nearest_class_space_view, },
};

static void gaussian_reduced_row(
    long long Ni_globe,    /*plj*/
    const Fraction_type w, /*lon_first*/
    const Fraction_type e, /*lon_last*/
    long long* pNi,        /*npoints*/
    double* pLon1,
    double* pLon2)
{
    Fraction_value_type Nw, Ne;
    Fraction_type inc, Nw_inc, Ne_inc;
    inc = fraction_construct(360ll, Ni_globe);

    /* auto Nw = (w / inc).integralPart(); */
    Nw     = fraction_integralPart(fraction_operator_divide(w, inc));
    Nw_inc = fraction_operator_multiply_n_Frac(Nw, inc);

    Assert(Ni_globe > 1);
    /*if (Nw * inc < w) {*/
    if (fraction_operator_less_than(Nw_inc, w)) {
        Nw += 1;
    }

    /*auto Ne = (e / inc).integralPart();*/
    Ne     = fraction_integralPart(fraction_operator_divide(e, inc));
    Ne_inc = fraction_operator_multiply_n_Frac(Ne, inc);
    /* if (Ne * inc > e) */
    if (fraction_operator_greater_than(Ne_inc, e)) {
        Ne -= 1;
    }
    if (Nw > Ne) {
        *pNi   = 0;          /* no points on this latitude */
        *pLon1 = *pLon2 = 0; /* dummy - unused */
    }
    else {
        *pNi = get_min(Ni_globe, Ne - Nw + 1);

        Nw_inc = fraction_operator_multiply_n_Frac(Nw, inc);
        *pLon1 = fraction_operator_double(Nw_inc);
        Ne_inc = fraction_operator_multiply_n_Frac(Ne, inc);
        *pLon2 = fraction_operator_double(Ne_inc);
    }
}

void grib_binary_search(const double xx[], size_t n, double x, size_t* ju, size_t* jl)
{
    size_t jm     = 0;
    int ascending = 0;
    *jl           = 0;
    *ju           = n;
    ascending     = (xx[n] >= xx[0]);
    while (*ju - *jl > 1) {
        jm = (*ju + *jl) >> 1;
        if ((x >= xx[jm]) == ascending)
            *jl = jm;
        else
            *ju = jm;
    }
}

double geographic_distance_spherical(double radius, double lon1, double lat1, double lon2, double lat2)
{
    double rlat1 = RADIAN(lat1);
    double rlat2 = RADIAN(lat2);
    double rlon1 = lon1;
    double rlon2 = lon2;
    double a;

    if (lat1 == lat2 && lon1 == lon2) {
        return 0.0; /* the two points are identical */
    }
    if (rlon1 >= 360) rlon1 -= 360.0;
    rlon1 = RADIAN(rlon1);
    if (rlon2 >= 360) rlon2 -= 360.0;
    rlon2 = RADIAN(rlon2);

    a = sin(rlat1) * sin(rlat2) + cos(rlat1) * cos(rlat2) * cos(rlon2 - rlon1);
    /* ECC-1258: sometimes 'a' can be very slightly outside the range [-1,1] */
    if (a > 1.0) a = 1.0;
    if (a < -1.0) a = -1.0;

    return radius * acos(a);
}
int grib_get_double_element_set_internal(grib_handle* h, const char* name, const size_t* index_array, size_t len, double* val_array)
{
    int ret = grib_get_double_element_set(h, name, index_array, len, val_array);

    if (ret != GRIB_SUCCESS)
        grib_context_log(h->context, GRIB_LOG_ERROR,
                         "unable to get %s as double element set (%s)",
                         name, grib_get_error_message(ret));

    return ret;
}
int grib_get_double_element_set(const grib_handle* h, const char* name, const size_t* index_array, size_t len, double* val_array)
{
    grib_accessor* acc = grib_find_accessor(h, name);

    if (acc) {
        return grib_unpack_double_element_set(acc, index_array, len, val_array);
    }
    return GRIB_NOT_FOUND;
}
grib_nearest* grib_nearest_factory(grib_handle* h, grib_arguments* args)
{
    int i;
    int ret    = GRIB_SUCCESS;
    char* type = (char*)grib_arguments_get_name(h, args, 0);

    for (i = 0; i < NUMBER(nearest_table); i++)
        if (strcmp(type, nearest_table[i].type) == 0) {
            grib_nearest_class* c = *(nearest_table[i].cclass);
            grib_nearest* it      = (grib_nearest*)grib_context_malloc_clear(h->context, c->size);
            it->cclass            = c;
            ret                   = grib_nearest_init(it, h, args);
            if (ret == GRIB_SUCCESS)
                return it;
            grib_context_log(h->context, GRIB_LOG_ERROR, "grib_nearest_factory: error %d instantiating nearest %s", ret, nearest_table[i].type);
            grib_nearest_delete(it);
            return NULL;
        }

    grib_context_log(h->context, GRIB_LOG_ERROR, "grib_nearest_factory : Unknown type : %s for nearest", type);

    return NULL;
}
void grib_dump_label(grib_dumper* d, grib_accessor* a, const char* comment)
{
    grib_dumper_class* c = d->cclass;
    while (c) {
        if (c->dump_label) {
            c->dump_label(d, a, comment);
            return;
        }
        c = c->super ? *(c->super) : NULL;
    }
    Assert(0);
}
int grib_nearest_delete(grib_nearest* i)
{
    grib_nearest_class* c = NULL;
    if (!i)
        return GRIB_INVALID_ARGUMENT;
    c = i->cclass;
    while (c) {
        grib_nearest_class* s = c->super ? *(c->super) : NULL;
        if (c->destroy)
            c->destroy(i);
        c = s;
    }
    return 0;
}
static int init_nearest(grib_nearest_class* c, grib_nearest* i, grib_handle* h, grib_arguments* args)
{
    if (c) {
        int ret               = GRIB_SUCCESS;
        grib_nearest_class* s = c->super ? *(c->super) : NULL;
        if (!c->inited) {
            if (c->init_class)
                c->init_class(c);
            c->inited = 1;
        }
        if (s)
            ret = init_nearest(s, i, h, args);

        if (ret != GRIB_SUCCESS)
            return ret;

        if (c->init)
            return c->init(i, h, args);
    }
    return GRIB_INTERNAL_ERROR;
}
int grib_nearest_init(grib_nearest* i, grib_handle* h, grib_arguments* args)
{
    return init_nearest(i->cclass, i, h, args);
}

int grib_nearest_find_generic(
    grib_nearest* nearest, grib_handle* h,
    double inlat, double inlon, unsigned long flags,

    const char* values_keyname,
    const char* Ni_keyname,
    const char* Nj_keyname,
    double** out_lats,
    int* out_lats_count,
    double** out_lons,
    int* out_lons_count,
    double** out_distances,

    double* outlats, double* outlons,
    double* values, double* distances, int* indexes, size_t* len)
{
    int ret = 0, i = 0;
    size_t nvalues = 0, nneighbours = 0;
    double radiusInKm;
    grib_iterator* iter = NULL;
    double lat = 0, lon = 0;

    /* array of candidates for nearest neighbours */
    PointStore* neighbours = NULL;

    inlon = normalise_longitude_in_degrees(inlon);

    if ((ret = grib_get_size(h, values_keyname, &nvalues)) != GRIB_SUCCESS)
        return ret;
    nearest->values_count = nvalues;

    if ((ret = grib_nearest_get_radius(h, &radiusInKm)) != GRIB_SUCCESS)
        return ret;

    neighbours = (PointStore*)grib_context_malloc(nearest->context, nvalues * sizeof(PointStore));
    for (i = 0; i < nvalues; ++i) {
        neighbours[i].m_dist  = 1e10; /* set all distances to large number to begin with */
        neighbours[i].m_lat   = 0;
        neighbours[i].m_lon   = 0;
        neighbours[i].m_value = 0;
        neighbours[i].m_index = 0;
    }

    /* GRIB_NEAREST_SAME_GRID not yet implemented */
    {
        double the_value = 0;
        double min_dist  = 1e10;
        size_t the_index = 0;
        int ilat = 0, ilon = 0;
        size_t idx_upper = 0, idx_lower = 0;
        double lat1 = 0, lat2 = 0;     /* inlat will be between these */
        const double LAT_DELTA = 10.0; /* in degrees */

        /* Note: If this is being called for a REDUCED grid, its Ni will be missing */

        if (grib_is_missing(h, Nj_keyname, &ret)) {
            grib_context_log(h->context, GRIB_LOG_DEBUG, "Key '%s' is missing", Nj_keyname);
            return ret ? ret : GRIB_GEOCALCULUS_PROBLEM;
        }

        *out_lons_count = nvalues; /* Maybe overestimate but safe */
        *out_lats_count = nvalues;

        if (*out_lats)
            grib_context_free(nearest->context, *out_lats);
        *out_lats = (double*)grib_context_malloc(nearest->context, nvalues * sizeof(double));
        if (!*out_lats)
            return GRIB_OUT_OF_MEMORY;

        if (*out_lons)
            grib_context_free(nearest->context, *out_lons);
        *out_lons = (double*)grib_context_malloc(nearest->context, nvalues * sizeof(double));
        if (!*out_lons)
            return GRIB_OUT_OF_MEMORY;

        iter = grib_iterator_new(h, 0, &ret);
        if (ret)
            return ret;
        /* First pass: collect all latitudes and longitudes */
        while (grib_iterator_next(iter, &lat, &lon, &the_value)) {
            ++the_index;
            Assert(ilat < *out_lats_count);
            Assert(ilon < *out_lons_count);
            (*out_lats)[ilat++] = lat;
            (*out_lons)[ilon++] = lon;
        }

        /* See between which 2 latitudes our point lies */
        qsort(*out_lats, nvalues, sizeof(double), &compare_doubles_ascending);
        grib_binary_search(*out_lats, *out_lats_count - 1, inlat, &idx_upper, &idx_lower);
        lat2 = (*out_lats)[idx_upper];
        lat1 = (*out_lats)[idx_lower];
        Assert(lat1 <= lat2);

        /* Second pass: Iterate again and collect candidate neighbours */
        grib_iterator_reset(iter);
        the_index = 0;
        i         = 0;
        while (grib_iterator_next(iter, &lat, &lon, &the_value)) {
            if (lat > lat2 + LAT_DELTA || lat < lat1 - LAT_DELTA) {
                /* Ignore latitudes too far from our point */
            }
            else {
                double dist = geographic_distance_spherical(radiusInKm, inlon, inlat, lon, lat);
                if (dist < min_dist)
                    min_dist = dist;
                /*printf("Candidate: lat=%.5f lon=%.5f dist=%f Idx=%ld Val=%f\n",lat,lon,dist,the_index,the_value);*/
                /* store this candidate point */
                neighbours[i].m_dist  = dist;
                neighbours[i].m_index = the_index;
                neighbours[i].m_lat   = lat;
                neighbours[i].m_lon   = lon;
                neighbours[i].m_value = the_value;
                i++;
            }
            ++the_index;
        }
        nneighbours = i;
        /* Sort the candidate neighbours in ascending order of distance */
        /* The first 4 entries will now be the closest 4 neighbours */
        qsort(neighbours, nneighbours, sizeof(PointStore), &compare_points);

        grib_iterator_delete(iter);
    }
    nearest->h = h;

    /* Sanity check for sorting */
#ifdef DEBUG
    for (i = 0; i < nneighbours - 1; ++i) {
        Assert(neighbours[i].m_dist <= neighbours[i + 1].m_dist);
    }
#endif

    /* GRIB_NEAREST_SAME_XXX not yet implemented */
    if (!*out_distances) {
        *out_distances = (double*)grib_context_malloc(nearest->context, 4 * sizeof(double));
    }
    (*out_distances)[0] = neighbours[0].m_dist;
    (*out_distances)[1] = neighbours[1].m_dist;
    (*out_distances)[2] = neighbours[2].m_dist;
    (*out_distances)[3] = neighbours[3].m_dist;

    for (i = 0; i < 4; ++i) {
        distances[i] = neighbours[i].m_dist;
        outlats[i]   = neighbours[i].m_lat;
        outlons[i]   = neighbours[i].m_lon;
        indexes[i]   = neighbours[i].m_index;
        values[i]    = neighbours[i].m_value;
        /*printf("(%f,%f)  i=%d  d=%f  v=%f\n",outlats[i],outlons[i],indexes[i],distances[i],values[i]);*/
    }

    free(neighbours);
    return GRIB_SUCCESS;
}






grib_accessor* grib_accessor_factory(grib_section* p, grib_action* creator,
                                     const long len, grib_arguments* params)
{
    grib_accessor_class* c = NULL;
    grib_accessor* a       = NULL;
    size_t size            = 0;

#ifdef ACCESSOR_FACTORY_USE_TRIE
    c = get_class(p->h->context, creator->op);
#else
    /* Use the hash table built with gperf (See make_accessor_class_hash.sh) */
    c = *((grib_accessor_classes_hash(creator->op, strlen(creator->op)))->cclass);
#endif

    a = (grib_accessor*)grib_context_malloc_clear(p->h->context, c->size);

    a->name       = creator->name;
    a->name_space = creator->name_space;

    a->all_names[0]       = creator->name;
    a->all_name_spaces[0] = creator->name_space;

    a->creator  = creator;
    a->context  = p->h->context;
    a->h        = NULL;
    a->next     = NULL;
    a->previous = NULL;
    a->parent   = p;
    a->length   = 0;
    a->offset   = 0;
    a->flags    = creator->flags;
    a->set      = creator->set;

    if (p->block->last) {
        a->offset = grib_get_next_position_offset(p->block->last);
#if 0
        printf("offset: p->block->last %s %s %ld %ld\n",
                p->block->last->cclass->name,
                p->block->last->name,(long)p->block->last->offset,(long)p->block->last->length);
#endif
    }
    else {
        if (p->owner) {
            a->offset = p->owner->offset;
        }
        else
            a->offset = 0;
    }

    a->cclass = c;

    grib_init_accessor(a, len, params);
    size = grib_get_next_position_offset(a);

    if (size > p->h->buffer->ulength) {
        if (!p->h->buffer->growable) {
            if (!p->h->partial)
                grib_context_log(p->h->context, GRIB_LOG_ERROR,
                                 "Creating (%s)%s of %s at offset %ld-%ld over message boundary (%lu)",
                                 p->owner ? p->owner->name : "", a->name,
                                 creator->op, a->offset,
                                 a->offset + a->length,
                                 p->h->buffer->ulength);

            grib_accessor_delete(p->h->context, a);
            return NULL;
        }
        else {
            grib_context_log(p->h->context, GRIB_LOG_DEBUG,
                             "CREATE: name=%s class=%s offset=%ld length=%ld action=",
                             a->name, a->cclass->name, a->offset, a->length);

            grib_grow_buffer(p->h->context, p->h->buffer, size);
            p->h->buffer->ulength = size;
        }
    }

    if (p->h->context->debug == 1) {
        if (p->owner)
            grib_context_log(p->h->context, GRIB_LOG_DEBUG,
                             "Creating (%s)%s of %s at offset %d [len=%d]",
                             p->owner->name, a->name, creator->op, a->offset, len, p->block);
        else
            grib_context_log(p->h->context, GRIB_LOG_DEBUG,
                             "Creating root %s of %s at offset %d [len=%d]",
                             a->name, creator->op, a->offset, len, p->block);
    }

    return a;
}
grib_concept_value* grib_parser_concept       = 0;
grib_concept_value* grib_parse_concept_file(grib_context* gc, const char* filename)
{
    GRIB_MUTEX_INIT_ONCE(&once, &init);
    GRIB_MUTEX_LOCK(&mutex_file);

    gc                  = gc ? gc : grib_context_get_default();
    grib_parser_context = gc;

    if (parse(gc, filename) == 0) {
        GRIB_MUTEX_UNLOCK(&mutex_file);
        return grib_parser_concept;
    }
    else {
        GRIB_MUTEX_UNLOCK(&mutex_file);
        return NULL;
    }
}
grib_hash_array_value* grib_parser_hash_array = 0;
grib_hash_array_value* grib_parse_hash_array_file(grib_context* gc, const char* filename)
{
    GRIB_MUTEX_INIT_ONCE(&once, &init);
    GRIB_MUTEX_LOCK(&mutex_file);

    gc                  = gc ? gc : grib_context_get_default();
    grib_parser_context = gc;

    if (parse(gc, filename) == 0) {
        GRIB_MUTEX_UNLOCK(&mutex_file);
        return grib_parser_hash_array;
    }
    else {
        GRIB_MUTEX_UNLOCK(&mutex_file);
        return NULL;
    }
}

// extern grib_action_class* grib_action_class_gen;
// grib_action_class* grib_action_class_gen;
grib_action_class* grib_action_class_section;

static grib_action_class _grib_action_class_template = {
    &grib_action_class_section,                              /* super                     */
    "action_class_template",                              /* name                      */
    sizeof(grib_action_template),            /* size                      */
    0,                                   /* inited */
    &init_class_template,                         /* init_class */
    0,                               /* init                      */
    &destroy_template,                            /* destroy */

    &dump_template,                               /* dump                      */
    0,                               /* xref                      */

    &create_accessor_template,             /* create_accessor*/

    0,                            /* notify_change */
    &reparse_template,                            /* reparse */
    0,                            /* execute */
};
grib_action_class* grib_action_class_template = &_grib_action_class_template;

static void init_class_template(grib_action_class* c)
{
    c->xref_gac    =    (*(c->super))->xref_gac;
    c->notify_change    =    (*(c->super))->notify_change;
    c->execute_gac    =    (*(c->super))->execute_gac;
}

grib_action* grib_action_create_template(grib_context* context, int nofail, const char* name, const char* arg1)
{
    grib_action_template* a;
    grib_action_class* c = grib_action_class_template;
    grib_action* act     = (grib_action*)grib_context_malloc_clear_persistent(context, c->size);
    act->name            = grib_context_strdup_persistent(context, name);
    act->op              = grib_context_strdup_persistent(context, "section");
    act->cclass          = c;
    act->next            = NULL;
    act->context         = context;
    a                    = (grib_action_template*)act;
    a->nofail            = nofail;
    if (arg1)
        a->arg = grib_context_strdup_persistent(context, arg1);
    else
        a->arg = NULL;

    return act;
}

static void dump_template(grib_action* act, FILE* f, int lvl)
{
    grib_action_template* a = (grib_action_template*)act;
    int i                   = 0;
    for (i = 0; i < lvl; i++)
        grib_context_print(act->context, f, "     ");
    grib_context_print(act->context, f, "Template %s  %s\n", act->name, a->arg);
}

grib_action* get_empty_template(grib_context* c, int* err)
{
    char fname[] = "empty_template.def";
    char* path   = 0;

    path = grib_context_full_defs_path(c, fname);
    if (path) {
        *err = GRIB_SUCCESS;
        return grib_parse_file(c, path);
    }
    else {
        *err = GRIB_INTERNAL_ERROR;
        grib_context_log(c, GRIB_LOG_ERROR, "get_empty_template: unable to get template %s", fname);
        return NULL;
    }
}

static int create_accessor_template(grib_section* p, grib_action* act, grib_loader* h)
{
    int ret                 = GRIB_SUCCESS;
    grib_action_template* a = (grib_action_template*)act;
    grib_action* la         = NULL;
    grib_action* next       = NULL;
    grib_accessor* as       = NULL;
    grib_section* gs        = NULL;

    char fname[1024] = {0,};
    char* fpath = 0;

    as = grib_accessor_factory(p, act, 0, NULL);

    if (!as)
        return GRIB_INTERNAL_ERROR;
    if (a->arg) {
        ret = grib_recompose_name(p->h, as, a->arg, fname, 1);

        if ((fpath = grib_context_full_defs_path(p->h->context, fname)) == NULL) {
            if (!a->nofail) {
                grib_context_log(p->h->context, GRIB_LOG_ERROR,
                                 "Unable to find template %s from %s ", act->name, fname);
                return GRIB_FILE_NOT_FOUND;
            }
            la = get_empty_template(p->h->context, &ret);
            if (ret)
                return ret;
        }
        else
            la = grib_parse_file(p->h->context, fpath);
    }
    as->flags |= GRIB_ACCESSOR_FLAG_HIDDEN;
    gs         = as->sub_section;
    gs->branch = la; /* Will be used to prevent unnecessary reparse */

    grib_push_accessor(as, p->block);

    if (la) {
        next = la;

        while (next) {
            ret = grib_create_accessor(gs, next, h);
            if (ret != GRIB_SUCCESS) {
                if (p->h->context->debug) {
                    grib_context_log(p->h->context, GRIB_LOG_ERROR,
                                     "Error processing template %s: %s [%s] %04lx",
                                     fname, grib_get_error_message(ret), next->name, next->flags);
                }
                return ret;
            }
            next = next->next;
        }
    }
    return GRIB_SUCCESS;
}

static grib_action* reparse_template(grib_action* a, grib_accessor* acc, int* doit)
{
    grib_action_template* self = (grib_action_template*)a;
    char* fpath                = 0;

    if (self->arg) {
        char fname[1024];
        grib_recompose_name(grib_handle_of_accessor(acc), NULL, self->arg, fname, 1);

        if ((fpath = grib_context_full_defs_path(acc->context, fname)) == NULL) {
            if (!self->nofail) {
                grib_context_log(acc->context, GRIB_LOG_ERROR,
                                 "Unable to find template %s from %s ", a->name, fname);
                return NULL;
            }
            return a;
        }

        /* printf("REPARSE %s\n",fpath); */
        return grib_parse_file(acc->context, fpath);
    }

    return NULL;
}

static void destroy_template(grib_context* context, grib_action* act)
{
    grib_action_template* a = (grib_action_template*)act;

    grib_context_free_persistent(context, a->arg);
    grib_context_free_persistent(context, act->name);
    grib_context_free_persistent(context, act->op);
}

static grib_action_class _grib_action_class_gen = {
    0,                              /* super                     */
    "action_class_gen",                              /* name                      */
    sizeof(grib_action_gen),            /* size                      */
    0,                                   /* inited */
    &init_class_gen,                         /* init_class */
    0,                               /* init                      */
    &destroy_gen,                            /* destroy */

    &dump_gen,                               /* dump                      */
    &xref_gen,                               /* xref                      */

    &create_accessor_gen,             /* create_accessor*/

    &notify_change_gen,                            /* notify_change */
    0,                            /* reparse */
    0,                            /* execute */
};
grib_action_class* grib_action_class_gen = &_grib_action_class_gen;
static void init_class_gen(grib_action_class* c)
{
}
grib_action* grib_action_create_gen(grib_context* context, const char* name, const char* op, const long len,
                                    grib_arguments* params, grib_arguments* default_value, int flags, const char* name_space, const char* set)
{
    grib_action_gen* a   = NULL;
    grib_action_class* c = grib_action_class_gen;
    grib_action* act     = (grib_action*)grib_context_malloc_clear_persistent(context, c->size);
    act->next            = NULL;
    act->name            = grib_context_strdup_persistent(context, name);
    act->op              = grib_context_strdup_persistent(context, op);
    if (name_space)
        act->name_space = grib_context_strdup_persistent(context, name_space);
    act->cclass  = c;
    act->context = context;
    act->flags   = flags;
#ifdef CHECK_LOWERCASE_AND_STRING_TYPE
    {
        int flag_lowercase=0, flag_stringtype=0;
        if (flags & GRIB_ACCESSOR_FLAG_LOWERCASE)
            flag_lowercase = 1;
        if (flags & GRIB_ACCESSOR_FLAG_STRING_TYPE)
            flag_stringtype = 1;
        if (flag_lowercase && !flag_stringtype) {
            printf("grib_action_create_gen name=%s. Has lowercase but not string_type\n", name);
            Assert(0);
        }
    }
#endif
    a            = (grib_action_gen*)act;

    a->len = len;

    a->params = params;
    if (set)
        act->set = grib_context_strdup_persistent(context, set);
    act->default_value = default_value;

    return act;
}

static void dump_gen(grib_action* act, FILE* f, int lvl)
{
    grib_action_gen* a = (grib_action_gen*)act;
    int i              = 0;
    for (i = 0; i < lvl; i++)
        grib_context_print(act->context, f, "     ");
    grib_context_print(act->context, f, "%s[%d] %s \n", act->op, a->len, act->name);
}

static void xref_gen(grib_action* act, FILE* f, const char* path)
{
    Assert(!"xref is disabled");
}

static int create_accessor_gen(grib_section* p, grib_action* act, grib_loader* loader)
{
    grib_action_gen* a = (grib_action_gen*)act;
    grib_accessor* ga  = NULL;

    ga = grib_accessor_factory(p, act, a->len, a->params);
    if (!ga)
        return GRIB_INTERNAL_ERROR;

    grib_push_accessor(ga, p->block);

    if (ga->flags & GRIB_ACCESSOR_FLAG_CONSTRAINT)
        grib_dependency_observe_arguments(ga, act->default_value);

    if (loader == NULL)
        return GRIB_SUCCESS;
    else
        return loader->init_accessor(loader, ga, act->default_value);
}

static int notify_change_gen(grib_action* act, grib_accessor* notified, grib_accessor* changed)
{
    if (act->default_value)
        return grib_pack_expression(notified, grib_arguments_get_expression(grib_handle_of_accessor(notified), act->default_value, 0));
    return GRIB_SUCCESS;
}

static void destroy_gen(grib_context* context, grib_action* act)
{
    grib_action_gen* a = (grib_action_gen*)act;

    if (a->params != act->default_value)
        grib_arguments_free(context, a->params);
    grib_arguments_free(context, act->default_value);

    grib_context_free_persistent(context, act->name);
    grib_context_free_persistent(context, act->op);
    if(act->name_space) {
        grib_context_free_persistent(context, act->name_space);
    }
    if (act->set)
        grib_context_free_persistent(context, act->set);
    if (act->defaultkey) {
        grib_context_free_persistent(context, act->defaultkey);
    }
}

/* These are the subfunctions that grib_yacc.c uses */
static grib_action_class _grib_action_class_variable = {
    &grib_action_class_gen,                              /* super                     */
    "action_class_variable",                              /* name                      */
    sizeof(grib_action_variable),            /* size                      */
    0,                                   /* inited */
    &init_class_var,                         /* init_class */
    0,                               /* init                      */
    0,                            /* destroy */

    0,                               /* dump                      */
    0,                               /* xref                      */

    0,             /* create_accessor*/

    0,                            /* notify_change */
    0,                            /* reparse */
    &execute_var,                            /* execute */
};
grib_action_class* grib_action_class_variable = &_grib_action_class_variable;
static void init_class_var(grib_action_class* c)
{
    c->dump_gac    =    (*(c->super))->dump_gac;
    c->xref_gac    =    (*(c->super))->xref_gac;
    c->create_accessor    =    (*(c->super))->create_accessor;
    c->notify_change    =    (*(c->super))->notify_change;
    c->reparse    =    (*(c->super))->reparse;
}

grib_action* grib_action_create_variable(grib_context* context, const char* name, const char* op, const long len, grib_arguments* params, grib_arguments* default_value, int flags, const char* name_space)
{
    grib_action_variable* a = NULL;
    grib_action_class* c    = grib_action_class_variable;
    grib_action* act        = (grib_action*)grib_context_malloc_clear_persistent(context, c->size);
    act->next               = NULL;
    act->name               = grib_context_strdup_persistent(context, name);
    if (name_space)
        act->name_space = grib_context_strdup_persistent(context, name_space);
    act->op            = grib_context_strdup_persistent(context, op);
    act->cclass        = c;
    act->context       = context;
    act->flags         = flags;
    a                  = (grib_action_variable*)act;
    a->len             = len;
    a->params          = params;
    act->default_value = default_value;

    /* printf("CREATE %s\n",name); */

    return act;
}

static int execute_var(grib_action* a, grib_handle* h)
{
    return grib_create_accessor(h->root, a, NULL);
}


static grib_action_class _grib_action_class_transient_darray = {
    &grib_action_class_gen,                              /* super                     */
    "action_class_transient_darray",                              /* name                      */
    sizeof(grib_action_transient_darray),            /* size                      */
    0,                                   /* inited */
    &init_class_transient_darray,                         /* init_class */
    0,                               /* init                      */
    &destroy_transient_darray,                            /* destroy */

    &dump_transient_darray,                               /* dump                      */
    &xref_transient_darray,                               /* xref                      */

    0,             /* create_accessor*/

    0,                            /* notify_change */
    0,                            /* reparse */
    &execute_transient_darray,                            /* execute */
};

grib_action_class* grib_action_class_transient_darray = &_grib_action_class_transient_darray;

static void init_class_transient_darray(grib_action_class* c)
{
    c->create_accessor    =    (*(c->super))->create_accessor;
    c->notify_change    =    (*(c->super))->notify_change;
    c->reparse    =    (*(c->super))->reparse;
}

grib_action* grib_action_create_transient_darray(grib_context* context, const char* name, grib_darray* darray, int flags)
{
    grib_action_transient_darray* a = NULL;
    grib_action_class* c            = grib_action_class_transient_darray;
    grib_action* act                = (grib_action*)grib_context_malloc_clear_persistent(context, c->size);
    act->op                         = grib_context_strdup_persistent(context, "transient_darray");

    act->cclass  = c;
    a            = (grib_action_transient_darray*)act;
    act->context = context;
    act->flags   = flags;

    a->darray = darray;
    a->name   = grib_context_strdup_persistent(context, name);

    act->name = grib_context_strdup_persistent(context, name);

    return act;
}

static int execute_transient_darray(grib_action* act, grib_handle* h)
{
    grib_action_transient_darray* self = (grib_action_transient_darray*)act;
    size_t len                         = grib_darray_used_size(self->darray);
    grib_accessor* a                   = NULL;
    grib_section* p                    = h->root;

    a = grib_accessor_factory(p, act, self->len, self->params);
    if (!a)
        return GRIB_INTERNAL_ERROR;

    grib_push_accessor(a, p->block);

    if (a->flags & GRIB_ACCESSOR_FLAG_CONSTRAINT)
        grib_dependency_observe_arguments(a, act->default_value);

    return grib_pack_double(a, self->darray->v, &len);
}

static void dump_transient_darray(grib_action* act, FILE* f, int lvl)
{
    int i                              = 0;
    grib_action_transient_darray* self = (grib_action_transient_darray*)act;
    for (i = 0; i < lvl; i++)
        grib_context_print(act->context, f, "     ");
    grib_context_print(act->context, f, self->name);
    printf("\n");
}

static void destroy_transient_darray(grib_context* context, grib_action* act)
{
    grib_action_transient_darray* a = (grib_action_transient_darray*)act;

    grib_context_free_persistent(context, a->name);
    grib_darray_delete(context, a->darray);
}

static void xref_transient_darray(grib_action* d, FILE* f, const char* path)
{
}


static grib_action_class _grib_action_class_alias = {
    0,                              /* super                     */
    "action_class_alias",                              /* name                      */
    sizeof(grib_action_alias),            /* size                      */
    0,                                   /* inited */
    &init_class_alias,                         /* init_class */
    0,                               /* init                      */
    &destroy_alias,                            /* destroy */

    &dump_alias,                               /* dump                      */
    &xref_alias,                               /* xref                      */

    &create_accessor_alias,             /* create_accessor*/

    0,                            /* notify_change */
    0,                            /* reparse */
    0,                            /* execute */
};

grib_action_class* grib_action_class_alias = &_grib_action_class_alias;

static void init_class_alias(grib_action_class* c)
{
}

/* Note: A fast cut-down version of grib_inline_strcmp which does NOT return -1 */
/* 0 means input strings are equal and 1 means not equal */
static int grib_inline_strcmp(const char* a, const char* b)
{
    if (*a != *b)
        return 1;
    while ((*a != 0 && *b != 0) && *(a) == *(b)) {
        a++;
        b++;
    }
    return (*a == 0 && *b == 0) ? 0 : 1;
}


grib_action* grib_action_create_alias(grib_context* context, const char* name, const char* arg1, const char* name_space, int flags)
{
    grib_action_alias* a;
    grib_action_class* c = grib_action_class_alias;
    grib_action* act     = (grib_action*)grib_context_malloc_clear_persistent(context, c->size);

    act->context = context;

    act->op   = NULL;
    act->name = grib_context_strdup_persistent(context, name);
    if (name_space)
        act->name_space = grib_context_strdup_persistent(context, name_space);

    act->cclass = c;
    act->flags  = flags;
    a           = (grib_action_alias*)act;
    a->target   = arg1 ? grib_context_strdup_persistent(context, arg1) : NULL;

    return act;
}

static int same(const char* a, const char* b)
{
    if (a == b)
        return 1;
    if (a && b)
        return (grib_inline_strcmp(a, b) == 0);
    return 0;
}

static int create_accessor_alias(grib_section* p, grib_action* act, grib_loader* h)
{
    int i, j, id;
    grib_action_alias* self = (grib_action_alias*)act;
    grib_accessor* x        = NULL;
    grib_accessor* y        = NULL;
    grib_handle* hand       = NULL;

    /*if alias and target have the same name add only the namespace */
    if (self->target && !grib_inline_strcmp(act->name, self->target) && act->name_space != NULL) {
        x = grib_find_accessor_fast(p->h, self->target);
        if (x == NULL) {
            grib_context_log(p->h->context, GRIB_LOG_DEBUG, "alias %s: cannot find %s (part 1)",
                             act->name, self->target);
            grib_context_log(p->h->context, GRIB_LOG_WARNING, "alias %s: cannot find %s",
                             act->name, self->target);
            return GRIB_SUCCESS;
        }

        if (x->name_space == NULL)
            x->name_space = act->name_space;

        grib_context_log(p->h->context, GRIB_LOG_DEBUG, "alias: add only namespace: %s.%s",
                         act->name_space, act->name);
        i = 0;
        while (i < MAX_ACCESSOR_NAMES) {
            if (x->all_names[i] != NULL && !grib_inline_strcmp(x->all_names[i], act->name)) {
                if (x->all_name_spaces[i] == NULL) {
                    x->all_name_spaces[i] = act->name_space;
                    return GRIB_SUCCESS;
                }
                else if (!grib_inline_strcmp(x->all_name_spaces[i], act->name_space)) {
                    return GRIB_SUCCESS;
                }
            }
            i++;
        }
        i = 0;
        while (i < MAX_ACCESSOR_NAMES) {
            if (x->all_names[i] == NULL) {
                x->all_names[i]       = act->name;
                x->all_name_spaces[i] = act->name_space;
                return GRIB_SUCCESS;
            }
            i++;
        }
        grib_context_log(p->h->context, GRIB_LOG_FATAL,
                         "unable to alias %s : increase MAX_ACCESSOR_NAMES", act->name);

        return GRIB_INTERNAL_ERROR;
    }

    y = grib_find_accessor_fast(p->h, act->name);

    /* delete old alias if already defined */
    if (y != NULL) {
        i = 0;
        while (i < MAX_ACCESSOR_NAMES && y->all_names[i]) {
            if (same(y->all_names[i], act->name) && same(y->all_name_spaces[i], act->name_space)) {
                grib_context_log(p->h->context, GRIB_LOG_DEBUG, "alias %s.%s already defined for %s. Deleting old alias",
                                 act->name_space, act->name, y->name);
                /* printf("[%s %s]\n",y->all_names[i], y->all_name_spaces[i]); */

                while (i < MAX_ACCESSOR_NAMES - 1) {
                    y->all_names[i]       = y->all_names[i + 1];
                    y->all_name_spaces[i] = y->all_name_spaces[i + 1];
                    i++;
                }

                y->all_names[MAX_ACCESSOR_NAMES - 1]       = NULL;
                y->all_name_spaces[MAX_ACCESSOR_NAMES - 1] = NULL;

                break;
            }
            i++;
        }

        if (self->target == NULL)
            return GRIB_SUCCESS;
    }

    if (!self->target)
        return GRIB_SUCCESS;

    x = grib_find_accessor_fast(p->h, self->target);
    if (x == NULL) {
        grib_context_log(p->h->context, GRIB_LOG_DEBUG, "alias %s: cannot find %s (part 2)",
                         act->name, self->target);
        grib_context_log(p->h->context, GRIB_LOG_WARNING, "alias %s: cannot find %s",
                         act->name, self->target);
        return GRIB_SUCCESS;
    }

    hand = grib_handle_of_accessor(x);
    if (hand->use_trie) {
        id = grib_hash_keys_get_id(x->context->keys, act->name);
        hand->accessors[id] = x;

        /*
         if (hand->accessors[id] != x) {
           x->same=hand->accessors[id];
           hand->accessors[id] = x;
         }
        */
    }

    i = 0;
    while (i < MAX_ACCESSOR_NAMES) {
        if (x->all_names[i] == NULL) {
            /* Only add entries if not already there */
            int found = 0;
            for (j = 0; j < i && !found; ++j) {
                int nameSame      = same(x->all_names[j], act->name);
                int namespaceSame = same(x->all_name_spaces[j], act->name_space);
                if (nameSame && namespaceSame) {
                    found = 1;
                }
            }
            if (!found) { /* Not there. So add them */
                x->all_names[i]       = act->name;
                x->all_name_spaces[i] = act->name_space;
                grib_context_log(p->h->context, GRIB_LOG_DEBUG, "alias %s.%s added (%s)",
                                 act->name_space, act->name, self->target);
            }
            return GRIB_SUCCESS;
        }
        i++;
    }

    for (i = 0; i < MAX_ACCESSOR_NAMES; i++)
        grib_context_log(p->h->context, GRIB_LOG_ERROR, "alias %s= ( %s already bound to %s )",
                         act->name, self->target, x->all_names[i]);

    return GRIB_SUCCESS;
}

static void dump_alias(grib_action* act, FILE* f, int lvl)
{
    grib_action_alias* a = (grib_action_alias*)act;
    int i                = 0;
    for (i = 0; i < lvl; i++)
        grib_context_print(act->context, f, "     ");
    if (a->target)
        grib_context_print(act->context, f, " alias %s  %s \n", act->name, a->target);
    else
        grib_context_print(act->context, f, " unalias %s  \n", act->name);
}

static void xref_alias(grib_action* act, FILE* f, const char* path)
{
    Assert(!"xref is disabled");
}

static void destroy_alias(grib_context* context, grib_action* act)
{
    grib_action_alias* a = (grib_action_alias*)act;

    if (a->target)
        grib_context_free_persistent(context, a->target);

    grib_context_free_persistent(context, act->name);
    grib_context_free_persistent(context, act->op);
    grib_context_free_persistent(context, act->name_space);
}


static grib_action_class _grib_action_class_meta = {
    &grib_action_class_gen,                              /* super                     */
    "action_class_meta",                              /* name                      */
    sizeof(grib_action_meta),            /* size                      */
    0,                                   /* inited */
    &init_class_meta,                         /* init_class */
    0,                               /* init                      */
    0,                            /* destroy */

    &dump_meta,                               /* dump                      */
    0,                               /* xref                      */

    0,             /* create_accessor*/

    0,                            /* notify_change */
    0,                            /* reparse */
    &execute_meta,                            /* execute */
};

grib_action_class* grib_action_class_meta = &_grib_action_class_meta;

static void init_class_meta(grib_action_class* c)
{
    c->xref_gac   =    (*(c->super))->xref_gac;
    c->create_accessor    =    (*(c->super))->create_accessor;
    c->notify_change    =    (*(c->super))->notify_change;
    c->reparse    =    (*(c->super))->reparse;
}

grib_action* grib_action_create_meta(grib_context* context, const char* name, const char* op,
                                     grib_arguments* params, grib_arguments* default_value, unsigned long flags, const char* name_space)
{
    grib_action_meta* a = (grib_action_meta*)grib_context_malloc_clear_persistent(context, sizeof(grib_action_meta));
    grib_action* act    = (grib_action*)a;
    act->next           = NULL;
    act->name           = grib_context_strdup_persistent(context, name);
    act->op             = grib_context_strdup_persistent(context, op);
    if (name_space)
        act->name_space = grib_context_strdup_persistent(context, name_space);
    act->cclass        = grib_action_class_meta;
    act->context       = context;
    act->flags         = flags;
    a->params          = params;
    act->default_value = default_value;
    a->len             = 0;

    /* grib_arguments_print(context,a->params,0); printf("\n"); */

    return act;
}

static void dump_meta(grib_action* act, FILE* f, int lvl)
{
    int i = 0;
    for (i = 0; i < lvl; i++)
        grib_context_print(act->context, f, "     ");
    grib_context_print(act->context, f, " meta %s \n", act->name);
}

static int execute_meta(grib_action* act, grib_handle* h)
{
    grib_action_class* super = *(act->cclass)->super;
    return super->create_accessor(h->root, act, NULL);
}

static grib_action_class _grib_action_class_put = {
    0,                              /* super                     */
    "action_class_put",                              /* name                      */
    sizeof(grib_action_put),            /* size                      */
    0,                                   /* inited */
    &init_class_put,                         /* init_class */
    0,                               /* init                      */
    &destroy_put,                            /* destroy */

    &dump_put,                               /* dump                      */
    0,                               /* xref                      */

    &create_accessor_put,             /* create_accessor*/

    0,                            /* notify_change */
    0,                            /* reparse */
    0,                            /* execute */
};
grib_action_class* grib_action_class_put = &_grib_action_class_put;
static void init_class_put(grib_action_class* c)
{
}

grib_action* grib_action_create_put(grib_context* context, const char* name, grib_arguments* args)
{
    grib_action_put* a   = NULL;
    grib_action_class* c = grib_action_class_put;
    grib_action* act     = (grib_action*)grib_context_malloc_clear_persistent(context, c->size);
    act->next            = NULL;
    act->name            = grib_context_strdup_persistent(context, name);
    act->op              = grib_context_strdup_persistent(context, "forward");
    act->cclass          = c;
    act->context         = context;
    a                    = (grib_action_put*)act;
    a->args              = args;
    return act;
}

static int create_accessor_put(grib_section* p, grib_action* act, grib_loader* h)
{
    grib_action_put* a = (grib_action_put*)act;

    grib_section* ts = NULL;

    grib_accessor* ga = NULL;

    ga = grib_find_accessor(p->h, grib_arguments_get_name(p->h, a->args, 1));
    if (ga)
        ts = ga->sub_section;
    /* ts = grib_get_sub_section(ga); */
    else
        return GRIB_BUFFER_TOO_SMALL;

    if (ts) {
        ga = grib_accessor_factory(ts, act, 0, a->args);
        if (ga)
            grib_push_accessor(ga, ts->block);
        else
            return GRIB_BUFFER_TOO_SMALL;
    }
    else {
        grib_context_log(act->context, GRIB_LOG_ERROR, "Action_class_put  : create_accessor_buffer : No Section named %s to export %s ", grib_arguments_get_name(p->h, a->args, 1), grib_arguments_get_name(p->h, a->args, 0));
    }
    return GRIB_SUCCESS;
}

static void dump_put(grib_action* act, FILE* f, int lvl)
{
    grib_action_put* a = (grib_action_put*)act;

    int i = 0;

    for (i = 0; i < lvl; i++)
        grib_context_print(act->context, f, "     ");

    grib_context_print(act->context, f, "put %s as %s in %s\n", grib_arguments_get_name(0, a->args, 0), act->name, grib_arguments_get_name(0, a->args, 1));
}

static void destroy_put(grib_context* context, grib_action* act)
{
    grib_action_put* a = (grib_action_put*)act;

    grib_arguments_free(context, a->args);
    grib_context_free_persistent(context, act->name);
    grib_context_free_persistent(context, act->op);
}


static grib_action_class _grib_action_class_remove = {
    0,                              /* super                     */
    "action_class_remove",                              /* name                      */
    sizeof(grib_action_remove),            /* size                      */
    0,                                   /* inited */
    &init_class_remove,                         /* init_class */
    0,                               /* init                      */
    &destroy_remove,                            /* destroy */

    &dump_remove,                               /* dump                      */
    &xref_remove,                               /* xref                      */

    &create_accessor_remove,             /* create_accessor*/

    0,                            /* notify_change */
    0,                            /* reparse */
    0,                            /* execute */
};

grib_action_class* grib_action_class_remove = &_grib_action_class_remove;

static void init_class_remove(grib_action_class* c)
{
}

grib_action* grib_action_create_remove(grib_context* context, grib_arguments* args)
{
    grib_action_remove* a = NULL;
    grib_action_class* c  = grib_action_class_remove;
    grib_action* act      = (grib_action*)grib_context_malloc_clear_persistent(context, c->size);
    act->next             = NULL;
    act->name             = grib_context_strdup_persistent(context, "DELETE");
    act->op               = grib_context_strdup_persistent(context, "remove");
    act->cclass           = c;
    act->context          = context;
    a                     = (grib_action_remove*)act;
    a->args               = args;
    return act;
}

static void remove_accessor(grib_accessor* a)
{
    grib_section* s = NULL;
    int id;

    if (!a || !a->previous)
        return;
    s = a->parent;

    if (grib_handle_of_accessor(a)->use_trie && *(a->all_names[0]) != '_') {
        id = grib_hash_keys_get_id(a->context->keys, a->all_names[0]);
        grib_handle_of_accessor(a)->accessors[id] = NULL;
    }

    if (a->next)
        a->previous->next = a->next;
    else
        return;

    a->next->previous = a->previous;

    grib_accessor_delete(s->h->context, a);

    return;
}

static int create_accessor_remove(grib_section* p, grib_action* act, grib_loader* h)
{
    grib_action_remove* a = (grib_action_remove*)act;

    grib_accessor* ga = NULL;

    ga = grib_find_accessor(p->h, grib_arguments_get_name(p->h, a->args, 0));

    if (ga) {
        remove_accessor(ga);
    } else {
        grib_context_log(act->context, GRIB_LOG_DEBUG, 
                         "Action_class_remove: create_accessor: No accessor named %s to remove", grib_arguments_get_name(p->h, a->args, 0));
    }
    return GRIB_SUCCESS;
}

static void dump_remove(grib_action* act, FILE* f, int lvl)
{
    grib_action_remove* a = (grib_action_remove*)act;

    int i = 0;

    for (i = 0; i < lvl; i++)
        grib_context_print(act->context, f, "     ");

    grib_context_print(act->context, f, "remove %s as %s in %s\n", grib_arguments_get_name(0, a->args, 0), act->name, grib_arguments_get_name(0, a->args, 1));
}

static void destroy_remove(grib_context* context, grib_action* act)
{
    grib_action_remove* a = (grib_action_remove*)act;

    grib_arguments_free(context, a->args);
    grib_context_free_persistent(context, act->name);
    grib_context_free_persistent(context, act->op);
}

static void xref_remove(grib_action* d, FILE* f, const char* path)
{
}



static grib_action_class _grib_action_class_rename = {
    0,                              /* super                     */
    "action_class_rename",                              /* name                      */
    sizeof(grib_action_rename),            /* size                      */
    0,                                   /* inited */
    &init_class_rename,                         /* init_class */
    0,                               /* init                      */
    &destroy_rename,                            /* destroy */

    &dump_rename,                               /* dump                      */
    &xref_rename,                               /* xref                      */

    &create_accessor_rename,             /* create_accessor*/

    0,                            /* notify_change */
    0,                            /* reparse */
    0,                            /* execute */
};

grib_action_class* grib_action_class_rename = &_grib_action_class_rename;

static void init_class_rename(grib_action_class* c)
{
}

grib_action* grib_action_create_rename(grib_context* context, char* the_old, char* the_new)
{
    grib_action_rename* a = NULL;
    grib_action_class* c  = grib_action_class_rename;
    grib_action* act      = (grib_action*)grib_context_malloc_clear_persistent(context, c->size);
    act->next             = NULL;
    act->name             = grib_context_strdup_persistent(context, "RENAME");
    act->op               = grib_context_strdup_persistent(context, "rename");
    act->cclass           = c;
    act->context          = context;
    a                     = (grib_action_rename*)act;
    a->the_old            = grib_context_strdup_persistent(context, the_old);
    a->the_new            = grib_context_strdup_persistent(context, the_new);
    return act;
}

static void rename_accessor(grib_accessor* a, char* name)
{
    int id;
    char* the_old = (char*)a->all_names[0];

    if (grib_handle_of_accessor(a)->use_trie && *(a->all_names[0]) != '_') {
        id                                        = grib_hash_keys_get_id(a->context->keys, a->all_names[0]);
        grib_handle_of_accessor(a)->accessors[id] = NULL;
        id                                        = grib_hash_keys_get_id(a->context->keys, name);
        grib_handle_of_accessor(a)->accessors[id] = a;
    }
    a->all_names[0] = grib_context_strdup_persistent(a->context, name);
    a->name         = a->all_names[0];
    grib_context_log(a->context, GRIB_LOG_DEBUG, "Renaming %s to %s", the_old, name);
    /* grib_context_free(a->context,the_old); */
}

static int create_accessor_rename(grib_section* p, grib_action* act, grib_loader* h)
{
    grib_action_rename* a = (grib_action_rename*)act;
    grib_accessor* ga     = NULL;

    ga = grib_find_accessor(p->h, a->the_old);

    if (ga) {
        rename_accessor(ga, a->the_new);
    }
    else {
        grib_context_log(act->context, GRIB_LOG_DEBUG, "Action_class_rename  : create_accessor_buffer : No accessor named %s to rename ", a->the_old);
    }

    return GRIB_SUCCESS;
}

static void dump_rename(grib_action* act, FILE* f, int lvl)
{
    grib_action_rename* a = (grib_action_rename*)act;

    int i = 0;

    for (i = 0; i < lvl; i++)
        grib_context_print(act->context, f, "     ");

    grib_context_print(act->context, f, "rename %s as %s in %s\n", a->the_old, act->name, a->the_new);
}

static void destroy_rename(grib_context* context, grib_action* act)
{
    grib_action_rename* a = (grib_action_rename*)act;

    grib_context_free_persistent(context, a->the_old);
    grib_context_free_persistent(context, a->the_new);
    grib_context_free_persistent(context, act->name);
    grib_context_free_persistent(context, act->op);
}

static void xref_rename(grib_action* d, FILE* f, const char* path)
{
}

static grib_action_class _grib_action_class_assert = {
    0,                              /* super                     */
    "action_class_assert",                              /* name                      */
    sizeof(grib_action_assert),            /* size                      */
    0,                                   /* inited */
    &init_class_assert,                         /* init_class */
    0,                               /* init                      */
    &destroy_assert,                            /* destroy */

    &dump_assert,                               /* dump                      */
    0,                               /* xref                      */

    &create_accessor_assert,             /* create_accessor*/

    &notify_change_assert,                            /* notify_change */
    0,                            /* reparse */
    &execute_assert,                            /* execute */
};
grib_action_class* grib_action_class_assert = &_grib_action_class_assert;

static void init_class_assert(grib_action_class* c)
{
}

grib_action* grib_action_create_assert(grib_context* context, grib_expression* expression)
{
    grib_action_assert* a = NULL;
    grib_action_class* c  = grib_action_class_assert;
    grib_action* act      = (grib_action*)grib_context_malloc_clear_persistent(context, c->size);
    act->next             = NULL;
    act->name             = grib_context_strdup_persistent(context, "assertion");
    act->op               = grib_context_strdup_persistent(context, "evaluate");
    act->cclass           = c;
    act->context          = context;
    a                     = (grib_action_assert*)act;
    a->expression         = expression;
    return act;
}

static int create_accessor_assert(grib_section* p, grib_action* act, grib_loader* h)
{
    grib_action_assert* self = (grib_action_assert*)act;
    grib_accessor* as        = grib_accessor_factory(p, act, 0, NULL);
    if (!as)
        return GRIB_INTERNAL_ERROR;
    grib_dependency_observe_expression(as, self->expression);

    grib_push_accessor(as, p->block);

    return GRIB_SUCCESS;
}

static void dump_assert(grib_action* act, FILE* f, int lvl)
{
    int i                    = 0;
    grib_action_assert* self = (grib_action_assert*)act;
    for (i = 0; i < lvl; i++)
        grib_context_print(act->context, f, "     ");
    grib_expression_print(act->context, self->expression, 0);
    printf("\n");
}

static void destroy_assert(grib_context* context, grib_action* act)
{
    grib_action_assert* a = (grib_action_assert*)act;
    grib_expression_free(context, a->expression);
    grib_context_free_persistent(context, act->name);
    grib_context_free_persistent(context, act->op);
}

static int execute_assert(grib_action* a, grib_handle* h)
{
    int ret                  = 0;
    double res               = 0;
    grib_action_assert* self = (grib_action_assert*)a;

    if ((ret = grib_expression_evaluate_double(h, self->expression, &res)) != GRIB_SUCCESS)
        return ret;

    if (res != 0) {
        return GRIB_SUCCESS;
    }
    else {
        grib_context_log(h->context, GRIB_LOG_ERROR, "Assertion failure: ");
        grib_expression_print(h->context, self->expression, h);
        printf("\n");
        return GRIB_ASSERTION_FAILURE;
    }
}

static int notify_change_assert(grib_action* a, grib_accessor* observer, grib_accessor* observed)
{
    grib_action_assert* self = (grib_action_assert*)a;

    int ret = GRIB_SUCCESS;
    long lres;

    if ((ret = grib_expression_evaluate_long(grib_handle_of_accessor(observed), self->expression, &lres)) != GRIB_SUCCESS)
        return ret;

    if (lres != 0)
        return GRIB_SUCCESS;
    else
        return GRIB_ASSERTION_FAILURE;
}

static grib_action_class _grib_action_class_modify = {
    0,                              /* super                     */
    "action_class_modify",                              /* name                      */
    sizeof(grib_action_modify),            /* size                      */
    0,                                   /* inited */
    &init_class_modify,                         /* init_class */
    0,                               /* init                      */
    &destroy_modify,                            /* destroy */

    &dump_modify,                               /* dump                      */
    &xref_modify,                               /* xref                      */

    &create_accessor_modify,             /* create_accessor*/

    0,                            /* notify_change */
    0,                            /* reparse */
    0,                            /* execute */
};
grib_action_class* grib_action_class_modify = &_grib_action_class_modify;
static void init_class_modify(grib_action_class* c)
{
}

grib_action* grib_action_create_modify(grib_context* context,
                                       const char* name,
                                       long flags)
{
    grib_action_modify* a;
    grib_action_class* c = grib_action_class_modify;
    grib_action* act     = (grib_action*)grib_context_malloc_clear_persistent(context, c->size);
    act->op              = grib_context_strdup_persistent(context, "section");

    act->cclass  = c;
    a            = (grib_action_modify*)act;
    act->context = context;

    a->flags = flags;
    a->name  = grib_context_strdup_persistent(context, name);


    act->name = grib_context_strdup_persistent(context, "flags");

    return act;
}

static void dump_modify(grib_action* act, FILE* f, int lvl)
{
}

static int create_accessor_modify(grib_section* p, grib_action* act, grib_loader* h)
{
    grib_action_modify* a = (grib_action_modify*)act;
    grib_accessor* ga     = NULL;

    ga = grib_find_accessor(p->h, a->name);

    if (ga)
        ga->flags = a->flags;

    else {
        grib_context_log(act->context, GRIB_LOG_DEBUG, "action_class_modify: create_accessor_buffer : No accessor named %s to modify.", a->name);
    }
    return GRIB_SUCCESS;
}

static void destroy_modify(grib_context* context, grib_action* act)
{
    grib_action_modify* a = (grib_action_modify*)act;

    grib_context_free_persistent(context, a->name);
    grib_context_free_persistent(context, act->name);
    grib_context_free_persistent(context, act->op);
}

static void xref_modify(grib_action* d, FILE* f, const char* path)
{
}


static grib_action_class _grib_action_class_set_missing = {
    0,                              /* super                     */
    "action_class_set_missing",                              /* name                      */
    sizeof(grib_action_set_missing),            /* size                      */
    0,                                   /* inited */
    &init_class_missing,                         /* init_class */
    0,                               /* init                      */
    &destroy_missing,                            /* destroy */

    &dump_missing,                               /* dump                      */
    0,                               /* xref                      */

    0,             /* create_accessor*/

    0,                            /* notify_change */
    0,                            /* reparse */
    &execute_missing,                            /* execute */
};
grib_action_class* grib_action_class_set_missing = &_grib_action_class_set_missing;

static void init_class_missing(grib_action_class* c)
{
}

grib_action* grib_action_create_set_missing(grib_context* context,
                                            const char* name)
{
    char buf[1024];

    grib_action_set_missing* a;
    grib_action_class* c = grib_action_class_set_missing;
    grib_action* act     = (grib_action*)grib_context_malloc_clear_persistent(context, c->size);
    act->op              = grib_context_strdup_persistent(context, "set_missing");

    act->cclass  = c;
    a            = (grib_action_set_missing*)act;
    act->context = context;

    a->name = grib_context_strdup_persistent(context, name);

    sprintf(buf, "set_missing_%s", name);

    act->name = grib_context_strdup_persistent(context, buf);

    return act;
}

static int execute_missing(grib_action* a, grib_handle* h)
{
    grib_action_set_missing* self = (grib_action_set_missing*)a;

    return grib_set_missing(h, self->name);
}

static void dump_missing(grib_action* act, FILE* f, int lvl)
{
    int i                         = 0;
    grib_action_set_missing* self = (grib_action_set_missing*)act;
    for (i = 0; i < lvl; i++)
        grib_context_print(act->context, f, "     ");
    grib_context_print(act->context, f, self->name);
    printf("\n");
}

static void destroy_missing(grib_context* context, grib_action* act)
{
    grib_action_set_missing* a = (grib_action_set_missing*)act;

    grib_context_free_persistent(context, a->name);
    grib_context_free_persistent(context, act->name);
    grib_context_free_persistent(context, act->op);
}


static grib_action_class _grib_action_class_set_darray = {
    0,                              /* super                     */
    "action_class_set_darray",                              /* name                      */
    sizeof(grib_action_set_darray),            /* size                      */
    0,                                   /* inited */
    &init_class_darray,                         /* init_class */
    0,                               /* init                      */
    &destroy_darray,                            /* destroy */

    &dump_darray,                               /* dump                      */
    &xref_darray,                               /* xref                      */

    0,             /* create_accessor*/

    0,                            /* notify_change */
    0,                            /* reparse */
    &execute_darray,                            /* execute */
};
grib_action_class* grib_action_class_set_darray = &_grib_action_class_set_darray;
static void init_class_darray(grib_action_class* c)
{
}

grib_action* grib_action_create_set_darray(grib_context* context,
                                           const char* name,
                                           grib_darray* darray)
{
    char buf[1024];

    grib_action_set_darray* a;
    grib_action_class* c = grib_action_class_set_darray;
    grib_action* act     = (grib_action*)grib_context_malloc_clear_persistent(context, c->size);
    act->op              = grib_context_strdup_persistent(context, "section");

    act->cclass  = c;
    a            = (grib_action_set_darray*)act;
    act->context = context;

    a->darray = darray;
    a->name   = grib_context_strdup_persistent(context, name);


    sprintf(buf, "set_darray%p", (void*)darray);

    act->name = grib_context_strdup_persistent(context, buf);

    return act;
}

static int execute_darray(grib_action* a, grib_handle* h)
{
    grib_action_set_darray* self = (grib_action_set_darray*)a;

    return grib_set_double_array(h, self->name, self->darray->v, self->darray->n);
}

static void dump_darray(grib_action* act, FILE* f, int lvl)
{
    int i                        = 0;
    grib_action_set_darray* self = (grib_action_set_darray*)act;
    for (i = 0; i < lvl; i++)
        grib_context_print(act->context, f, "     ");
    grib_context_print(act->context, f, self->name);
    printf("\n");
}

static void destroy_darray(grib_context* context, grib_action* act)
{
    grib_action_set_darray* a = (grib_action_set_darray*)act;

    grib_context_free_persistent(context, a->name);
    grib_darray_delete(context, a->darray);
    grib_context_free_persistent(context, act->name);
    grib_context_free_persistent(context, act->op);
}

static void xref_darray(grib_action* d, FILE* f, const char* path)
{
}



static grib_action_class _grib_action_class_set_sarray = {
    0,                              /* super                     */
    "action_class_set_sarray",                              /* name                      */
    sizeof(grib_action_set_sarray),            /* size                      */
    0,                                   /* inited */
    &init_class_sarray,                         /* init_class */
    0,                               /* init                      */
    &destroy_sarray,                            /* destroy */

    &dump_sarray,                               /* dump                      */
    &xref_sarray,                               /* xref                      */

    0,             /* create_accessor*/

    0,                            /* notify_change */
    0,                            /* reparse */
    &execute_sarray,                            /* execute */
};
grib_action_class* grib_action_class_set_sarray = &_grib_action_class_set_sarray;
static void init_class_sarray(grib_action_class* c)
{
}

grib_action* grib_action_create_set_sarray(grib_context* context,
                                           const char* name,
                                           grib_sarray* sarray)
{
    char buf[1024];

    grib_action_set_sarray* a;
    grib_action_class* c = grib_action_class_set_sarray;
    grib_action* act     = (grib_action*)grib_context_malloc_clear_persistent(context, c->size);
    act->op              = grib_context_strdup_persistent(context, "section");

    act->cclass  = c;
    a            = (grib_action_set_sarray*)act;
    act->context = context;

    a->sarray = sarray;
    a->name   = grib_context_strdup_persistent(context, name);


    sprintf(buf, "set_sarray%p", (void*)sarray);

    act->name = grib_context_strdup_persistent(context, buf);

    return act;
}

static int execute_sarray(grib_action* a, grib_handle* h)
{
    grib_action_set_sarray* self = (grib_action_set_sarray*)a;

    return grib_set_string_array(h, self->name, (const char**)self->sarray->v, self->sarray->n);
}

static void dump_sarray(grib_action* act, FILE* f, int lvl)
{
    int i                        = 0;
    grib_action_set_sarray* self = (grib_action_set_sarray*)act;
    for (i = 0; i < lvl; i++)
        grib_context_print(act->context, f, "     ");
    grib_context_print(act->context, f, self->name);
    printf("\n");
}

static void destroy_sarray(grib_context* context, grib_action* act)
{
    grib_action_set_sarray* a = (grib_action_set_sarray*)act;

    grib_context_free_persistent(context, a->name);
    grib_sarray_delete(context, a->sarray);
    grib_context_free_persistent(context, act->name);
    grib_context_free_persistent(context, act->op);
}

static void xref_sarray(grib_action* d, FILE* f, const char* path)
{
}


static grib_action_class _grib_action_class_close = {
    0,                              /* super                     */
    "action_class_close",                              /* name                      */
    sizeof(grib_action_close),            /* size                      */
    0,                                   /* inited */
    &init_class_close,                         /* init_class */
    0,                               /* init                      */
    &destroy_close,                            /* destroy */

    &dump_close,                               /* dump                      */
    0,                               /* xref                      */

    0,             /* create_accessor*/

    0,                            /* notify_change */
    0,                            /* reparse */
    &execute_close,                            /* execute */
};

grib_action_class* grib_action_class_close = &_grib_action_class_close;

static void init_class_close(grib_action_class* c)
{
}

grib_action* grib_action_create_close(grib_context* context, char* filename)
{
    char buf[1024];

    grib_action_close* a;
    grib_action_class* c = grib_action_class_close;
    grib_action* act     = (grib_action*)grib_context_malloc_clear_persistent(context, c->size);
    act->op              = grib_context_strdup_persistent(context, "section");

    act->cclass  = c;
    a            = (grib_action_close*)act;
    act->context = context;

    a->filename = grib_context_strdup_persistent(context, filename);

    sprintf(buf, "close_%p", (void*)a->filename);

    act->name = grib_context_strdup_persistent(context, buf);

    return act;
}

static int execute_close(grib_action* act, grib_handle* h)
{
    char filename[2048] = {0,};
    size_t len              = 2048;
    grib_action_close* self = (grib_action_close*)act;
    int err                 = 0;
    grib_file* file         = 0;

    err = grib_get_string(h, self->filename, filename, &len);
    /* fprintf(stderr,"++++ name %s\n",filename); */
    if (err)
        return err;
    /* grib_file_close(filename,1,&err); */
    file = grib_get_file(filename, &err);
    if (err)
        return err;
    if (file)
        grib_file_pool_delete_file(file);

    return GRIB_SUCCESS;
}

static void dump_close(grib_action* act, FILE* f, int lvl)
{
}

static void destroy_close(grib_context* context, grib_action* act)
{
    grib_action_close* a = (grib_action_close*)act;

    grib_context_free_persistent(context, a->filename);
    grib_context_free_persistent(context, act->name);
    grib_context_free_persistent(context, act->op);
}


static grib_action_class _grib_action_class_write = {
    0,                              /* super                     */
    "action_class_write",                              /* name                      */
    sizeof(grib_action_write),            /* size                      */
    0,                                   /* inited */
    &init_class_write,                         /* init_class */
    0,                               /* init                      */
    &destroy_write,                            /* destroy */

    &dump_write,                               /* dump                      */
    0,                               /* xref                      */

    0,             /* create_accessor*/

    0,                            /* notify_change */
    0,                            /* reparse */
    &execute_write,                            /* execute */
};
grib_action_class* grib_action_class_write = &_grib_action_class_write;

static void init_class_write(grib_action_class* c)
{
}

grib_action* grib_action_create_write(grib_context* context, const char* name, int append, int padtomultiple)
{
    char buf[1024];

    grib_action_write* a = NULL;
    grib_action_class* c = grib_action_class_write;
    grib_action* act     = (grib_action*)grib_context_malloc_clear_persistent(context, c->size);
    act->op              = grib_context_strdup_persistent(context, "section");

    act->cclass  = c;
    a            = (grib_action_write*)act;
    act->context = context;

    a->name = grib_context_strdup_persistent(context, name);

    sprintf(buf, "write%p", (void*)a->name);

    act->name        = grib_context_strdup_persistent(context, buf);
    a->append        = append;
    a->padtomultiple = padtomultiple;

    return act;
}

static int execute_write(grib_action* act, grib_handle* h)
{
    grib_action_write* a = (grib_action_write*)act;
    int err              = GRIB_SUCCESS;
    size_t size;
    const void* buffer   = NULL;
    const char* filename = NULL;
    char string[1024]    = {0,};

    grib_file* of = NULL;

    if ((err = grib_get_message(h, &buffer, &size)) != GRIB_SUCCESS) {
        grib_context_log(act->context, GRIB_LOG_ERROR, "unable to get message\n");
        return err;
    }

    if (strlen(a->name) != 0) {
        err      = grib_recompose_name(h, NULL, a->name, string, 0);
        filename = string;
    }
    else {
        if (act->context->outfilename) {
            filename = act->context->outfilename;
            err      = grib_recompose_name(h, NULL, act->context->outfilename, string, 0);
            if (!err)
                filename = string;
        }
        else {
            filename = "filter.out";
        }
    }

    Assert(filename);
    if (a->append)
        of = grib_file_open(filename, "a", &err);
    else
        of = grib_file_open(filename, "w", &err);

    if (!of || !of->handle) {
        grib_context_log(act->context, GRIB_LOG_ERROR, "unable to open file %s\n", filename);
        return GRIB_IO_PROBLEM;
    }

    if (h->gts_header) {
        if (fwrite(h->gts_header, 1, h->gts_header_len, of->handle) != h->gts_header_len) {
            grib_context_log(act->context, (GRIB_LOG_ERROR) | (GRIB_LOG_PERROR),
                             "Error writing GTS header to %s", filename);
            return GRIB_IO_PROBLEM;
        }
    }

    if (fwrite(buffer, 1, size, of->handle) != size) {
        grib_context_log(act->context, (GRIB_LOG_ERROR) | (GRIB_LOG_PERROR),
                         "Error writing to %s", filename);
        return GRIB_IO_PROBLEM;
    }

    if (a->padtomultiple) {
        char* zeros;
        size_t padding = a->padtomultiple - size % a->padtomultiple;
        /* printf("XXX padding=%d size=%d padtomultiple=%d\n",padding,size,a->padtomultiple); */
        zeros = (char*)calloc(padding, 1);
        Assert(zeros);
        if (fwrite(zeros, 1, padding, of->handle) != padding) {
            grib_context_log(act->context, (GRIB_LOG_ERROR) | (GRIB_LOG_PERROR),
                             "Error writing to %s", filename);
            free(zeros);
            return GRIB_IO_PROBLEM;
        }
        free(zeros);
    }

    if (h->gts_header) {
        char gts_trailer[4] = { '\x0D', '\x0D', '\x0A', '\x03' };
        if (fwrite(gts_trailer, 1, 4, of->handle) != 4) {
            grib_context_log(act->context, (GRIB_LOG_ERROR) | (GRIB_LOG_PERROR),
                             "Error writing GTS trailer to %s", filename);
            return GRIB_IO_PROBLEM;
        }
    }

    grib_file_close(filename, 0, &err);
    if (err != GRIB_SUCCESS) {
        grib_context_log(act->context, GRIB_LOG_ERROR, "unable to write message\n");
        return err;
    }

    return err;
}

static void dump_write(grib_action* act, FILE* f, int lvl)
{
}

static void destroy_write(grib_context* context, grib_action* act)
{
    grib_action_write* a = (grib_action_write*)act;

    grib_context_free_persistent(context, a->name);
    grib_context_free_persistent(context, act->name);
    grib_context_free_persistent(context, act->op);
}

static grib_action_class _grib_action_class_print = {
    0,                              /* super                     */
    "action_class_print",                              /* name                      */
    sizeof(grib_action_print),            /* size                      */
    0,                                   /* inited */
    &init_class_print,                         /* init_class */
    0,                               /* init                      */
    &destroy_print,                            /* destroy */

    &dump_print,                               /* dump                      */
    0,                               /* xref                      */

    0,             /* create_accessor*/

    0,                            /* notify_change */
    0,                            /* reparse */
    &execute_print,                            /* execute */
};
grib_action_class* grib_action_class_print = &_grib_action_class_print;
static void init_class_print(grib_action_class* c)
{
}

grib_action* grib_action_create_print(grib_context* context, const char* name, char* outname)
{
    char buf[1024];

    grib_action_print* a;
    grib_action_class* c = grib_action_class_print;
    grib_action* act     = (grib_action*)grib_context_malloc_clear_persistent(context, c->size);
    act->op              = grib_context_strdup_persistent(context, "section");

    act->cclass  = c;
    a            = (grib_action_print*)act;
    act->context = context;

    a->name = grib_context_strdup_persistent(context, name);

    if (outname) {
        FILE* out  = NULL;
        int ioerr  = 0;
        a->outname = grib_context_strdup_persistent(context, outname);
        out        = fopen(outname, "w");
        ioerr      = errno;
        if (!out) {
            grib_context_log(act->context, (GRIB_LOG_ERROR) | (GRIB_LOG_PERROR),
                             "IO ERROR: %s: %s", strerror(ioerr), outname);
        }
        if (out)
            fclose(out);
    }

    sprintf(buf, "print%p", (void*)a->name);

    act->name = grib_context_strdup_persistent(context, buf);

    return act;
}

static int execute_print(grib_action* act, grib_handle* h)
{
    grib_action_print* self = (grib_action_print*)act;
    int err                 = 0;
    FILE* out               = NULL;
    int ioerr               = 0;

    if (self->outname) {
        out   = fopen(self->outname, "a");
        ioerr = errno;
        if (!out) {
            grib_context_log(act->context, (GRIB_LOG_ERROR) | (GRIB_LOG_PERROR),
                             "IO ERROR: %s: %s", strerror(ioerr), self->outname);
            return GRIB_IO_PROBLEM;
        }
    }
    else {
        out = stdout;
    }

    err = grib_recompose_print(h, NULL, self->name, 0, out);

    if (self->outname)
        fclose(out);

    return err;
}

static void dump_print(grib_action* act, FILE* f, int lvl)
{
}

static void destroy_print(grib_context* context, grib_action* act)
{
    grib_action_print* a = (grib_action_print*)act;

    grib_context_free_persistent(context, a->name);
    grib_context_free_persistent(context, act->name);
    grib_context_free_persistent(context, act->op);
}


static grib_action_class _grib_action_class_if = {
    &grib_action_class_section,                              /* super                     */
    "action_class_if",                              /* name                      */
    sizeof(grib_action_if),            /* size                      */
    0,                                   /* inited */
    &init_class_if,                         /* init_class */
    0,                               /* init                      */
    &destroy_if,                            /* destroy */

    &dump_if,                               /* dump                      */
    &xref_if,                               /* xref                      */

    &create_accessor_if,             /* create_accessor*/

    0,                            /* notify_change */
    &reparse_if,                            /* reparse */
    &execute_if,                           /* execute */
};
grib_action_class* grib_action_class_if = &_grib_action_class_if;
static void init_class_if(grib_action_class* c)
{
    c->notify_change    =    (*(c->super))->notify_change;
}

grib_action* grib_action_create_if(grib_context* context,
                                   grib_expression* expression,
                                   grib_action* block_true, grib_action* block_false, int transient,
                                   int lineno, char* file_being_parsed)
{
    char name[1024];
    grib_action_if* a;
    grib_action_class* c = grib_action_class_if;
    grib_action* act     = (grib_action*)grib_context_malloc_clear_persistent(context, c->size);
    act->op              = grib_context_strdup_persistent(context, "section");

    act->cclass  = c;
    a            = (grib_action_if*)act;
    act->context = context;

    a->expression  = expression;
    a->block_true  = block_true;
    a->block_false = block_false;
    a->transient   = transient;

    if (transient)
        sprintf(name, "__if%p", (void*)a);
    else
        sprintf(name, "_if%p", (void*)a);

    act->name       = grib_context_strdup_persistent(context, name);
    act->debug_info = NULL;
    if (context->debug > 0 && file_being_parsed) {
        /* Construct debug information showing definition file and line */
        /* number of IF statement */
        char debug_info[1024];
        sprintf(debug_info, "File=%s line=%d", file_being_parsed, lineno);
        act->debug_info = grib_context_strdup_persistent(context, debug_info);
    }

    return act;
}

static int create_accessor_if(grib_section* p, grib_action* act, grib_loader* h)
{
    grib_action_if* a = (grib_action_if*)act;
    grib_action* next = NULL;
    int ret           = 0;
    long lres         = 0;

    grib_accessor* as = NULL;
    grib_section* gs  = NULL;

    as = grib_accessor_factory(p, act, 0, NULL);
    if (!as)
        return GRIB_INTERNAL_ERROR;
    gs = as->sub_section;
    grib_push_accessor(as, p->block);

    if ((ret = grib_expression_evaluate_long(p->h, a->expression, &lres)) != GRIB_SUCCESS)
        return ret;

    if (lres)
        next = a->block_true;
    else
        next = a->block_false;

    if (p->h->context->debug > 1) {
        printf("EVALUATE create_accessor_handle ");
        grib_expression_print(p->h->context, a->expression, p->h);
        printf(" [%s][_if%p]\n", (next == a->block_true ? "true" : "false"), (void*)a);

        /*grib_dump_action_branch(stdout,next,5);*/
    }

    gs->branch = next;
    grib_dependency_observe_expression(as, a->expression);

    while (next) {
        ret = grib_create_accessor(gs, next, h);
        if (ret != GRIB_SUCCESS)
            return ret;
        next = next->next;
    }

    return GRIB_SUCCESS;
}

static void print_expression_debug_info(grib_context* ctx, grib_expression* exp, grib_handle* h)
{
    grib_expression_print(ctx, exp, h); /* writes to stdout without a newline */
    printf("\n");
}

static int execute_if(grib_action* act, grib_handle* h)
{
    grib_action_if* a = (grib_action_if*)act;
    grib_action* next = NULL;
    grib_context* ctx = h->context;
    int ret           = 0;
    long lres         = 0;

    /* See GRIB-394 */
    int type = grib_expression_native_type(h, a->expression);
    if (type != GRIB_TYPE_DOUBLE) {
        if ((ret = grib_expression_evaluate_long(h, a->expression, &lres)) != GRIB_SUCCESS) {
            if (ret == GRIB_NOT_FOUND)
                lres = 0;
            else {
                if (ctx->debug)
                    print_expression_debug_info(ctx, a->expression, h);
                return ret;
            }
        }
    }
    else {
        double dres = 0.0;
        ret         = grib_expression_evaluate_double(h, a->expression, &dres);
        lres        = (long)dres;
        if (ret != GRIB_SUCCESS) {
            if (ret == GRIB_NOT_FOUND)
                lres = 0;
            else {
                if (ctx->debug)
                    print_expression_debug_info(ctx, a->expression, h);
                return ret;
            }
        }
    }

    if (lres)
        next = a->block_true;
    else
        next = a->block_false;

    while (next) {
        ret = grib_action_execute(next, h);
        if (ret != GRIB_SUCCESS)
            return ret;
        next = next->next;
    }

    return GRIB_SUCCESS;
}

static void dump_if(grib_action* act, FILE* f, int lvl)
{
    grib_action_if* a = (grib_action_if*)act;
    int i             = 0;

    for (i = 0; i < lvl; i++)
        grib_context_print(act->context, f, "     ");

    printf("if(%s) { ", act->name);
    grib_expression_print(act->context, a->expression, 0);
    printf("\n");

    if (a->block_true) {
        /*      grib_context_print(act->context,f,"IF \t TODO \n");  TODO */
        grib_dump_action_branch(f, a->block_true, lvl + 1);
    }
    if (a->block_false) {
        printf("}\n");
        for (i = 0; i < lvl; i++)
            grib_context_print(act->context, f, "     ");
        printf("else(%s) { ", act->name);
        grib_expression_print(act->context, a->expression, 0);
        /*     grib_context_print(act->context,f,"ELSE \n" );*/
        grib_dump_action_branch(f, a->block_false, lvl + 1);
    }
    for (i = 0; i < lvl; i++)
        grib_context_print(act->context, f, "     ");
    printf("}\n");
}

static grib_action* reparse_if(grib_action* a, grib_accessor* acc, int* doit)
{
    int ret              = 0;
    long lres            = 0;
    grib_action_if* self = (grib_action_if*)a;

    /* printf("reparse %s %s\n",a->name,acc->name); */

    if ((ret = grib_expression_evaluate_long(grib_handle_of_accessor(acc), self->expression, &lres)) != GRIB_SUCCESS)
        grib_context_log(acc->context,
                         GRIB_LOG_ERROR, "if reparse  grib_expression_evaluate_long %s",
                         grib_get_error_message(ret));

    if (lres)
        return self->block_true;
    else
        return self->block_false;
}

static void destroy_if(grib_context* context, grib_action* act)
{
    grib_action_if* a = (grib_action_if*)act;
    grib_action* t    = a->block_true;
    grib_action* f    = a->block_false;

    while (t) {
        grib_action* nt = t->next;
        grib_action_delete(context, t);
        t = nt;
    }

    while (f) {
        grib_action* nf = f->next;
        grib_action_delete(context, f);
        f = nf;
    }

    grib_expression_free(context, a->expression);

    grib_context_free_persistent(context, act->name);
    grib_context_free_persistent(context, act->debug_info);
    grib_context_free_persistent(context, act->op);
}

static void xref_if(grib_action* d, FILE* f, const char* path)
{
}


static grib_action_class _grib_action_class_when = {
    0,                              /* super                     */
    "action_class_when",                              /* name                      */
    sizeof(grib_action_when),            /* size                      */
    0,                                   /* inited */
    &init_class_when,                         /* init_class */
    0,                               /* init                      */
    &destroy_when,                            /* destroy */

    &dump_when,                               /* dump                      */
    &xref_when,                               /* xref                      */

    &create_accessor_when,             /* create_accessor*/

    &notify_change_when,                            /* notify_change */
    0,                            /* reparse */
    0,                            /* execute */
};
grib_action_class* grib_action_class_when = &_grib_action_class_when;
static void init_class_when(grib_action_class* c)
{
}
#if defined(DEBUG) && GRIB_PTHREADS == 0 && GRIB_OMP_THREADS == 0
#define CHECK_LOOP 1
#endif
grib_action* grib_action_create_when(grib_context* context,
                                     grib_expression* expression,
                                     grib_action* block_true, grib_action* block_false)
{
    char name[1024];

    grib_action_when* a;
    grib_action_class* c = grib_action_class_when;
    grib_action* act     = (grib_action*)grib_context_malloc_clear_persistent(context, c->size);
    act->op              = grib_context_strdup_persistent(context, "when");

    act->cclass  = c;
    a            = (grib_action_when*)act;
    act->context = context;

    a->expression  = expression;
    a->block_true  = block_true;
    a->block_false = block_false;

    sprintf(name, "_when%p", (void*)expression);

    act->name = grib_context_strdup_persistent(context, name);

    return act;
}

static int create_accessor_when(grib_section* p, grib_action* act, grib_loader* h)
{
    grib_action_when* self = (grib_action_when*)act;
    grib_accessor* as      = grib_accessor_factory(p, act, 0, 0);
    if (!as)
        return GRIB_INTERNAL_ERROR;

    grib_dependency_observe_expression(as, self->expression);

    grib_push_accessor(as, p->block);

    return GRIB_SUCCESS;
}

static void dump_when(grib_action* act, FILE* f, int lvl)
{
    grib_action_when* a = (grib_action_when*)act;
    int i               = 0;

    for (i = 0; i < lvl; i++)
        grib_context_print(act->context, f, "     ");

    printf("when(%s) { ", act->name);
    grib_expression_print(act->context, a->expression, 0);
    printf("\n");

    grib_dump_action_branch(f, a->block_true, lvl + 1);

    for (i = 0; i < lvl; i++)
        grib_context_print(act->context, f, "     ");
    printf("}");

    if (a->block_false) {
        printf(" else { ");

        grib_dump_action_branch(f, a->block_true, lvl + 1);

        for (i = 0; i < lvl; i++)
            grib_context_print(act->context, f, "     ");
        printf("}");
    }
    printf("\n");
}

#ifdef CHECK_LOOP
#define SET_LOOP(self, v) self->loop = v;
#else
#define SET_LOOP(self, v)
#endif

static int notify_change_when(grib_action* a, grib_accessor* observer, grib_accessor* observed)
{
    grib_action_when* self = (grib_action_when*)a;
    grib_action* b         = NULL;
    int ret                = GRIB_SUCCESS;
    long lres;

    /* ECC-974: observed->parent will change as a result of the execute
     * so must store the handle once here (in 'hand') rather than call
     * grib_handle_of_accessor(observed) later
     */
    grib_handle* hand = grib_handle_of_accessor(observed);

    if ((ret = grib_expression_evaluate_long(hand, self->expression, &lres)) != GRIB_SUCCESS)
        return ret;
#ifdef CHECK_LOOP
    if (self->loop) {
        printf("LOOP detected...\n");
        printf("WHEN triggered by %s %ld\n", observed->name, lres);
        grib_expression_print(observed->context, self->expression, 0);
        printf("\n");
        return ret;
    }
#endif
    SET_LOOP(self, 1);

    if (lres)
        b = self->block_true;
    else
        b = self->block_false;

    while (b) {
        ret = grib_action_execute(b, hand);
        if (ret != GRIB_SUCCESS) {
            SET_LOOP(self, 0);
            return ret;
        }
        b = b->next;
    }

    SET_LOOP(self, 0);

    return GRIB_SUCCESS;
}

static void destroy_when(grib_context* context, grib_action* act)
{
    grib_action_when* self = (grib_action_when*)act;
    grib_action* t         = self->block_true;

    while (t) {
        grib_action* nt = t->next;
        grib_action_delete(context, t);
        t = nt;
    }

    t = self->block_false;
    while (t) {
        grib_action* nt = t->next;
        grib_action_delete(context, t);
        t = nt;
    }

    grib_expression_free(context, self->expression);

    grib_context_free_persistent(context, act->name);
    grib_context_free_persistent(context, act->op);
}

static void xref_when(grib_action* d, FILE* f, const char* path)
{
}

static grib_action_class _grib_action_class_set = {
    0,                              /* super                     */
    "action_class_set",                              /* name                      */
    sizeof(grib_action_set),            /* size                      */
    0,                                   /* inited */
    &init_class_cset,                         /* init_class */
    0,                               /* init                      */
    &destroy_cset,                            /* destroy */

    &dump_cset,                               /* dump                      */
    &xref_cset,                               /* xref                      */

    0,             /* create_accessor*/

    0,                            /* notify_change */
    0,                            /* reparse */
    &execute_cset,                            /* execute */
};
grib_action_class* grib_action_class_set = &_grib_action_class_set;
static void init_class_cset(grib_action_class* c)
{
}

grib_action* grib_action_create_set(grib_context* context,
                                    const char* name, grib_expression* expression, int nofail)
{
    char buf[1024];

    grib_action_set* a;
    grib_action_class* c = grib_action_class_set;
    grib_action* act     = (grib_action*)grib_context_malloc_clear_persistent(context, c->size);
    act->op              = grib_context_strdup_persistent(context, "section");

    act->cclass  = c;
    a            = (grib_action_set*)act;
    act->context = context;

    a->expression = expression;
    a->name       = grib_context_strdup_persistent(context, name);
    a->nofail     = nofail;


    sprintf(buf, "set%p", (void*)expression);

    act->name = grib_context_strdup_persistent(context, buf);

    return act;
}

static int execute_cset(grib_action* a, grib_handle* h)
{
    int ret               = 0;
    grib_action_set* self = (grib_action_set*)a;
    ret                   = grib_set_expression(h, self->name, self->expression);
    if (self->nofail)
        return 0;
    if (ret != GRIB_SUCCESS) {
        grib_context_log(h->context, GRIB_LOG_ERROR, "Error while setting key %s (%s)",
                         self->name, grib_get_error_message(ret));
    }
    return ret;
}

static void dump_cset(grib_action* act, FILE* f, int lvl)
{
    int i                 = 0;
    grib_action_set* self = (grib_action_set*)act;
    for (i = 0; i < lvl; i++)
        grib_context_print(act->context, f, "     ");
    grib_context_print(act->context, f, self->name);
    printf("\n");
}

static void destroy_cset(grib_context* context, grib_action* act)
{
    grib_action_set* a = (grib_action_set*)act;

    grib_context_free_persistent(context, a->name);
    grib_expression_free(context, a->expression);
    grib_context_free_persistent(context, act->name);
    grib_context_free_persistent(context, act->op);
}

static void xref_cset(grib_action* d, FILE* f, const char* path)
{
}


static grib_action_class _grib_action_class_list = {
    &grib_action_class_section,                              /* super                     */
    "action_class_list",                              /* name                      */
    sizeof(grib_action_list),            /* size                      */
    0,                                   /* inited */
    &init_class_gacl,                         /* init_class */
    0,                               /* init                      */
    &destroy_gacl,                            /* destroy */

    &dump_gacl,                               /* dump                      */
    0,                               /* xref                      */

    &create_accessor_gacl,             /* create_accessor*/

    0,                            /* notify_change */
    &reparse_gacl,                            /* reparse */
    0,                            /* execute */
};
grib_action_class* grib_action_class_list = &_grib_action_class_list;
static void init_class_gacl(grib_action_class* c)
{
    c->xref_gac    =    (*(c->super))->xref_gac;
    c->notify_change    =    (*(c->super))->notify_change;
    c->execute_gac    =    (*(c->super))->execute_gac;
}

static void dump_gacl(grib_action* act, FILE* f, int lvl)
{
    grib_action_list* a = (grib_action_list*)act;
    int i               = 0;
    for (i = 0; i < lvl; i++)
        grib_context_print(act->context, f, "     ");
    grib_context_print(act->context, f, "Loop   %s\n", act->name);
    grib_dump_action_branch(f, a->block_list, lvl + 1);
}

static int create_accessor_gacl(grib_section* p, grib_action* act, grib_loader* h)
{
    grib_action_list* a = (grib_action_list*)act;

    grib_accessor* ga = NULL;
    grib_section* gs  = NULL;
    grib_action* la   = NULL;
    grib_action* next = NULL;
    int ret           = 0;
    long val          = 0;

    if ((ret = grib_expression_evaluate_long(p->h, a->expression, &val)) != GRIB_SUCCESS) {
        grib_context_log(p->h->context, GRIB_LOG_DEBUG, "List %s creating %d values unable to evaluate long", act->name, val);
        return ret;
    }

    grib_context_log(p->h->context, GRIB_LOG_DEBUG, "List %s creating %d values", act->name, val);

    ga = grib_accessor_factory(p, act, 0, NULL);
    if (!ga)
        return GRIB_BUFFER_TOO_SMALL;
    gs       = ga->sub_section;
    ga->loop = val;

    grib_push_accessor(ga, p->block);

    la = a->block_list;

    gs->branch = la;
    grib_dependency_observe_expression(ga, a->expression);

    while (val--) {
        next = la;
        while (next) {
            ret = grib_create_accessor(gs, next, h);
            if (ret != GRIB_SUCCESS)
                return ret;
            next = next->next;
        }
    }
    return GRIB_SUCCESS;
}

grib_action* grib_action_create_list(grib_context* context, const char* name, grib_expression* expression, grib_action* block)
{
    grib_action_list* a;
    grib_action_class* c = grib_action_class_list;
    grib_action* act     = (grib_action*)grib_context_malloc_clear_persistent(context, c->size);
    act->cclass          = c;
    act->context         = context;
    a                    = (grib_action_list*)act;
    act->next            = NULL;
    act->name            = grib_context_strdup_persistent(context, name);
    act->op              = grib_context_strdup_persistent(context, "section");
    a->expression        = expression;

    a->block_list = block;

    grib_context_log(context, GRIB_LOG_DEBUG, " Action List %s is created  \n", act->name);
    return act;
}

static grib_action* reparse_gacl(grib_action* a, grib_accessor* acc, int* doit)
{
    grib_action_list* self = (grib_action_list*)a;

    int ret  = 0;
    long val = 0;

    if ((ret = grib_expression_evaluate_long(grib_handle_of_accessor(acc), self->expression, &val)) != GRIB_SUCCESS) {
        grib_context_log(acc->context, GRIB_LOG_ERROR,
                "List %s creating %ld values, unable to evaluate long", acc->name, val);
    }

    *doit = (val != acc->loop);

    return self->block_list;
}

static void destroy_gacl(grib_context* context, grib_action* act)
{
    grib_action_list* self = (grib_action_list*)act;
    grib_action* a         = self->block_list;

    while (a) {
        grib_action* na = a->next;
        grib_action_delete(context, a);
        a = na;
    }

    grib_context_free_persistent(context, act->name);
    grib_context_free_persistent(context, act->op);
    grib_expression_free(context, self->expression);
}



static grib_action_class _grib_action_class_while = {
    &grib_action_class_section,                              /* super                     */
    "action_class_while",                              /* name                      */
    sizeof(grib_action_while),            /* size                      */
    0,                                   /* inited */
    &init_class_while,                         /* init_class */
    0,                               /* init                      */
    &destroy_while,                            /* destroy */

    &dump_while,                               /* dump                      */
    0,                               /* xref                      */

    &create_accessor_while,             /* create_accessor*/

    0,                            /* notify_change */
    0,                            /* reparse */
    0,                            /* execute */
};

grib_action_class* grib_action_class_while = &_grib_action_class_while;

static void init_class_while(grib_action_class* c)
{
    c->xref_gac    =    (*(c->super))->xref_gac;
    c->notify_change    =    (*(c->super))->notify_change;
    c->reparse    =    (*(c->super))->reparse;
    c->execute_gac    =    (*(c->super))->execute_gac;
}

static void dump_while(grib_action* act, FILE* f, int lvl)
{
    grib_action_while* a = (grib_action_while*)act;
    int i                = 0;
    for (i = 0; i < lvl; i++)
        grib_context_print(act->context, f, "     ");
    grib_context_print(act->context, f, "Loop   %s\n", act->name);
    grib_dump_action_branch(f, a->block_while, lvl + 1);
}

static int create_accessor_while(grib_section* p, grib_action* act, grib_loader* h)
{
    grib_action_while* a = (grib_action_while*)act;

    grib_accessor* ga = NULL;
    grib_section* gs  = NULL;
    grib_action* la   = NULL;
    grib_action* next = NULL;
    int ret           = 0;
    /* long n = 0; */

    ga = grib_accessor_factory(p, act, 0, NULL);
    if (!ga)
        return GRIB_BUFFER_TOO_SMALL;
    gs = ga->sub_section;

    grib_push_accessor(ga, p->block);

    la = a->block_while;

    for (;;) {
        long val = 0;

        if ((ret = grib_expression_evaluate_long(p->h, a->expression, &val)) != GRIB_SUCCESS) {
            grib_context_log(p->h->context, GRIB_LOG_DEBUG, " List %s creating %d values unable to evaluate long \n", act->name, val);
            return ret;
        }

        /* printf("val=%ld %ld\n",val,n++); */

        if (!val)
            break;


        next = la;
        while (next) {
            ret = grib_create_accessor(gs, next, h);
            if (ret != GRIB_SUCCESS)
                return ret;
            next = next->next;
        }
    }
    return GRIB_SUCCESS;
}

grib_action* grib_action_create_while(grib_context* context, grib_expression* expression, grib_action* block)
{
    char name[80];
    grib_action_while* a;
    grib_action_class* c = grib_action_class_while;
    grib_action* act     = (grib_action*)grib_context_malloc_clear_persistent(context, c->size);
    act->cclass          = c;
    act->context         = context;
    a                    = (grib_action_while*)act;
    act->next            = NULL;


    sprintf(name, "_while%p", (void*)a);
    act->name     = grib_context_strdup_persistent(context, name);
    act->op       = grib_context_strdup_persistent(context, "section");
    a->expression = expression;

    a->block_while = block;

    grib_context_log(context, GRIB_LOG_DEBUG, " Action List %s is created  \n", act->name);
    return act;
}

static void destroy_while(grib_context* context, grib_action* act)
{
    grib_action_while* self = (grib_action_while*)act;
    grib_action* a          = self->block_while;

    while (a) {
        grib_action* na = a->next;
        grib_action_delete(context, a);
        a = na;
    }

    grib_context_free_persistent(context, act->name);
    grib_context_free_persistent(context, act->op);
    grib_expression_free(context, self->expression);
}

static grib_action_class _grib_action_class_trigger = {
    &grib_action_class_section,                              /* super                     */
    "action_class_trigger",                              /* name                      */
    sizeof(grib_action_trigger),            /* size                      */
    0,                                   /* inited */
    &init_class_trigger,                         /* init_class */
    0,                               /* init                      */
    &destroy_trigger,                            /* destroy */

    &dump_trigger,                               /* dump                      */
    0,                               /* xref                      */

    &create_accessor_trigger,             /* create_accessor*/

    0,                            /* notify_change */
    &reparse_trigger,                            /* reparse */
    0,                            /* execute */
};
grib_action_class* grib_action_class_trigger = &_grib_action_class_trigger;
static void init_class_trigger(grib_action_class* c)
{
    c->xref_gac    =    (*(c->super))->xref_gac;
    c->notify_change    =    (*(c->super))->notify_change;
    c->execute_gac    =    (*(c->super))->execute_gac;
}

grib_action* grib_action_create_trigger(grib_context* context, grib_arguments* args, grib_action* block)
{
    char name[1024];

    grib_action_trigger* a = 0;
    grib_action_class* c   = grib_action_class_trigger;
    grib_action* act       = (grib_action*)grib_context_malloc_clear_persistent(context, c->size);

    sprintf(name, "_trigger%p", (void*)act);

    act->name    = grib_context_strdup_persistent(context, name);
    act->op      = grib_context_strdup_persistent(context, "section");
    act->cclass  = c;
    act->next    = NULL;
    act->context = context;

    a             = (grib_action_trigger*)act;
    a->trigger_on = args;
    a->block      = block;

    return act;
}

static void dump_trigger(grib_action* act, FILE* f, int lvl)
{
    /* grib_action_trigger* a = ( grib_action_trigger*)act; */
    int i = 0;
    for (i = 0; i < lvl; i++)
        grib_context_print(act->context, f, "     ");
    grib_context_print(act->context, f, "Trigger\n");
}

static int create_accessor_trigger(grib_section* p, grib_action* act, grib_loader* h)
{
    int ret                = GRIB_SUCCESS;
    grib_action_trigger* a = (grib_action_trigger*)act;
    grib_action* next      = NULL;
    grib_accessor* as      = NULL;
    grib_section* gs       = NULL;


    as = grib_accessor_factory(p, act, 0, NULL);

    if (!as)
        return GRIB_INTERNAL_ERROR;

    gs         = as->sub_section;
    gs->branch = 0; /* Force a reparse each time */

    grib_push_accessor(as, p->block);
    grib_dependency_observe_arguments(as, a->trigger_on);

    next = a->block;

    while (next) {
        ret = grib_create_accessor(gs, next, h);
        if (ret != GRIB_SUCCESS)
            return ret;
        next = next->next;
    }

    return GRIB_SUCCESS;
}

static grib_action* reparse_trigger(grib_action* a, grib_accessor* acc, int* doit)
{
    grib_action_trigger* self = (grib_action_trigger*)a;
    return self->block;
}

static void destroy_trigger(grib_context* context, grib_action* act)
{
    grib_action_trigger* a = (grib_action_trigger*)act;

    grib_action* b = a->block;

    while (b) {
        grib_action* n = b->next;
        grib_action_delete(context, b);
        b = n;
    }

    grib_arguments_free(context, a->trigger_on);
    grib_context_free_persistent(context, act->name);
    grib_context_free_persistent(context, act->op);
}


static grib_action_class _grib_action_class_concept = {
    &grib_action_class_gen,                              /* super                     */
    "action_class_concept",                              /* name                      */
    sizeof(grib_action_concept),            /* size                      */
    0,                                   /* inited */
    &init_class_concept,                         /* init_class */
    0,                               /* init                      */
    &destroy_concept,                            /* destroy */

    &dump_concept,                               /* dump                      */
    0,                               /* xref                      */

    0,             /* create_accessor*/

    0,                            /* notify_change */
    0,                            /* reparse */
    0,                            /* execute */
};
grib_action_class* grib_action_class_concept = &_grib_action_class_concept;

grib_action* grib_action_create_concept(grib_context* context,
                                        const char* name,
                                        grib_concept_value* concept,
                                        const char* basename, const char* name_space, const char* defaultkey,
                                        const char* masterDir, const char* localDir, const char* ecmfDir, int flags, int nofail)
{
    grib_action_concept* a = NULL;
    grib_action_class* c   = grib_action_class_concept;
    grib_action* act       = (grib_action*)grib_context_malloc_clear_persistent(context, c->size);
    act->op                = grib_context_strdup_persistent(context, "concept");

    act->cclass  = c;
    a            = (grib_action_concept*)act;
    act->context = context;
    act->flags   = flags;

    if (name_space)
        act->name_space = grib_context_strdup_persistent(context, name_space);

    if (basename)
        a->basename = grib_context_strdup_persistent(context, basename);
    else
        a->basename = NULL;

    if (masterDir)
        a->masterDir = grib_context_strdup_persistent(context, masterDir);
    else
        a->masterDir = NULL;

    if (localDir)
        a->localDir = grib_context_strdup_persistent(context, localDir);
    else
        a->localDir = NULL;

    if (defaultkey)
        act->defaultkey = grib_context_strdup_persistent(context, defaultkey);

    a->concept = concept;
    if (concept) {
        grib_concept_value* conc_val = concept;
        grib_trie* index             = grib_trie_new(context);
        while (conc_val) {
            conc_val->index = index;
            grib_trie_insert_no_replace(index, conc_val->name, conc_val);
            conc_val = conc_val->next;
        }
    }
    act->name = grib_context_strdup_persistent(context, name);

    a->nofail = nofail;

    return act;
}

static void init_class_concept(grib_action_class* c)
{
    c->xref_gac    =    (*(c->super))->xref_gac;
    c->create_accessor    =    (*(c->super))->create_accessor;
    c->notify_change    =    (*(c->super))->notify_change;
    c->reparse    =    (*(c->super))->reparse;
    c->execute_gac    =    (*(c->super))->execute_gac;
}

grib_concept_value* action_concept_get_concept(grib_accessor* a)
{
    return get_concept(grib_handle_of_accessor(a), (grib_action_concept*)a->creator);
}

int action_concept_get_nofail(grib_accessor* a)
{
    grib_action_concept* self = (grib_action_concept*)a->creator;
    return self->nofail;
}

static void dump_concept(grib_action* act, FILE* f, int lvl)
{
    int i = 0;

    for (i = 0; i < lvl; i++)
        grib_context_print(act->context, f, "     ");

    printf("concept(%s) { ", act->name);
    printf("\n");

    for (i = 0; i < lvl; i++)
        grib_context_print(act->context, f, "     ");
    printf("}\n");
}

static void destroy_concept(grib_context* context, grib_action* act)
{
    grib_action_concept* self = (grib_action_concept*)act;

    grib_concept_value* v = self->concept;
    if (v) {
        grib_trie_delete_container(v->index);
    }
    while (v) {
        grib_concept_value* n = v->next;
        grib_concept_value_delete(context, v);
        v = n;
    }
    grib_context_free_persistent(context, self->masterDir);
    grib_context_free_persistent(context, self->localDir);
    grib_context_free_persistent(context, self->basename);
}

static grib_concept_value* get_concept_impl(grib_handle* h, grib_action_concept* self)
{
    char buf[4096] = {0,};
    char master[1024] = {0,};
    char local[1024] = {0,};
    char masterDir[1024] = {0,};
    size_t lenMasterDir = 1024;
    char key[4096]      = {0,};
    char* full = 0;
    int id;

    grib_context* context = ((grib_action*)self)->context;
    grib_concept_value* c = NULL;

    if (self->concept != NULL)
        return self->concept;

    Assert(self->masterDir);
    grib_get_string(h, self->masterDir, masterDir, &lenMasterDir);

    sprintf(buf, "%s/%s", masterDir, self->basename);

    grib_recompose_name(h, NULL, buf, master, 1);

    if (self->localDir) {
        char localDir[1024] = {0,};
        size_t lenLocalDir = 1024;
        grib_get_string(h, self->localDir, localDir, &lenLocalDir);
        sprintf(buf, "%s/%s", localDir, self->basename);
        grib_recompose_name(h, NULL, buf, local, 1);
    }

    sprintf(key, "%s%s", master, local);

    id = grib_itrie_get_id(h->context->concepts_index, key);
    if ((c = h->context->concepts[id]) != NULL)
        return c;

    if (*local && (full = grib_context_full_defs_path(context, local)) != NULL) {
        c = grib_parse_concept_file(context, full);
        grib_context_log(h->context, GRIB_LOG_DEBUG,
                         "Loading concept %s from %s", ((grib_action*)self)->name, full);
    }

    full = grib_context_full_defs_path(context, master);

    if (c) {
        grib_concept_value* last = c;
        while (last->next)
            last = last->next;
        if (full) {
            last->next = grib_parse_concept_file(context, full);
        }
    }
    else if (full) {
        c = grib_parse_concept_file(context, full);
    }
    else {
        grib_context_log(context, GRIB_LOG_FATAL,
                         "unable to find definition file %s in %s:%s\nDefinition files path=\"%s\"",
                         self->basename, master, local, context->grib_definition_files_path);
        return NULL;
    }

    if (full) {
        grib_context_log(h->context, GRIB_LOG_DEBUG,
                         "Loading concept %s from %s", ((grib_action*)self)->name, full);
    }

    h->context->concepts[id] = c;
    if (c) {
        grib_trie* index = grib_trie_new(context);
        while (c) {
            c->index = index;
            grib_trie_insert_no_replace(index, c->name, c);
            c = c->next;
        }
    }

    return h->context->concepts[id];
}

static grib_concept_value* get_concept(grib_handle* h, grib_action_concept* self)
{
    grib_concept_value* result = NULL;
    GRIB_MUTEX_INIT_ONCE(&once, &init)
    GRIB_MUTEX_LOCK(&mutex);

    result = get_concept_impl(h, self);

    GRIB_MUTEX_UNLOCK(&mutex);
    return result;
}

static int concept_condition_expression_true(grib_handle* h, grib_concept_condition* c, char* exprVal)
{
    long lval;
    long lres      = 0;
    int ok         = 0;
    int err        = 0;
    const int type = grib_expression_native_type(h, c->expression);

    switch (type) {
        case GRIB_TYPE_LONG:
            grib_expression_evaluate_long(h, c->expression, &lres);
            ok = (grib_get_long(h, c->name, &lval) == GRIB_SUCCESS) &&
                 (lval == lres);
            if (ok)
                sprintf(exprVal, "%ld", lres);
            break;

        case GRIB_TYPE_DOUBLE: {
            double dval;
            double dres = 0.0;
            grib_expression_evaluate_double(h, c->expression, &dres);
            ok = (grib_get_double(h, c->name, &dval) == GRIB_SUCCESS) &&
                 (dval == dres);
            if (ok)
                sprintf(exprVal, "%g", dres);
            break;
        }

        case GRIB_TYPE_STRING: {
            const char* cval;
            char buf[80];
            char tmp[80];
            size_t len  = sizeof(buf);
            size_t size = sizeof(tmp);

            ok = (grib_get_string(h, c->name, buf, &len) == GRIB_SUCCESS) &&
                 ((cval = grib_expression_evaluate_string(h, c->expression, tmp, &size, &err)) != NULL) &&
                 (err == 0) && (strcmp(buf, cval) == 0);
            if (ok)
                sprintf(exprVal, "%s", cval);
            break;
        }

        default:
            /* TODO: */
            break;
    }
    return ok;
}

int get_concept_condition_string(grib_handle* h, const char* key, const char* value, char* result)
{
    int err         = 0;
    int length      = 0;
    char strVal[64] = {0,};
    char exprVal[256] = {0,};
    const char* pValue                = value;
    size_t len                        = sizeof(strVal);
    grib_concept_value* concept_value = NULL;
    grib_accessor* acc                = grib_find_accessor(h, key);
    if (!acc)
        return GRIB_NOT_FOUND;

    if (!value) {
        err = grib_get_string(h, key, strVal, &len);
        if (err)
            return GRIB_INTERNAL_ERROR;
        pValue = strVal;
    }

    concept_value = action_concept_get_concept(acc);
    while (concept_value) {
        grib_concept_condition* concept_condition = concept_value->conditions;
        if (strcmp(pValue, concept_value->name) == 0) {
            while (concept_condition) {
                grib_expression* expression = concept_condition->expression;
                const char* condition_name  = concept_condition->name;
                Assert(expression);
                if (concept_condition_expression_true(h, concept_condition, exprVal) && strcmp(condition_name, "one") != 0) {
                    length += sprintf(result + length, "%s%s=%s",
                                      (length == 0 ? "" : ","), condition_name, exprVal);
                }
                concept_condition = concept_condition->next;
            }
        }

        concept_value = concept_value->next;
    }
    if (length == 0)
        return GRIB_CONCEPT_NO_MATCH;
    return GRIB_SUCCESS;
}

static grib_action_class _grib_action_class_hash_array = {
    &grib_action_class_gen,                              /* super                     */
    "action_class_hash_array",                              /* name                      */
    sizeof(grib_action_hash_array),            /* size                      */
    0,                                   /* inited */
    &init_class_hash,                         /* init_class */
    0,                               /* init                      */
    &destroy_hash,                            /* destroy */

    &dump_hash,                               /* dump                      */
    0,                               /* xref                      */

    0,             /* create_accessor*/

    0,                            /* notify_change */
    0,                            /* reparse */
    0,                            /* execute */
};

grib_action_class* grib_action_class_hash_array = &_grib_action_class_hash_array;
grib_action* grib_action_create_hash_array(grib_context* context,
                                           const char* name,
                                           grib_hash_array_value* hash_array,
                                           const char* basename, const char* name_space, const char* defaultkey,
                                           const char* masterDir, const char* localDir, const char* ecmfDir, int flags, int nofail)
{
    grib_action_hash_array* a = NULL;
    grib_action_class* c      = grib_action_class_hash_array;
    grib_action* act          = (grib_action*)grib_context_malloc_clear_persistent(context, c->size);
    act->op                   = grib_context_strdup_persistent(context, "hash_array");

    act->cclass  = c;
    a            = (grib_action_hash_array*)act;
    act->context = context;
    act->flags   = flags;

    if (name_space)
        act->name_space = grib_context_strdup_persistent(context, name_space);

    if (basename)
        a->basename = grib_context_strdup_persistent(context, basename);
    else
        a->basename = NULL;

    if (masterDir)
        a->masterDir = grib_context_strdup_persistent(context, masterDir);
    else
        a->masterDir = NULL;

    if (localDir)
        a->localDir = grib_context_strdup_persistent(context, localDir);
    else
        a->localDir = NULL;

    if (ecmfDir)
        a->ecmfDir = grib_context_strdup_persistent(context, ecmfDir);
    else
        a->ecmfDir = NULL;

    if (defaultkey)
        act->defaultkey = grib_context_strdup_persistent(context, defaultkey);

    a->hash_array = hash_array;
    if (hash_array) {
        grib_hash_array_value* ha = hash_array;
        grib_trie* index          = grib_trie_new(context);
        while (ha) {
            ha->index = index;
            grib_trie_insert_no_replace(index, ha->name, ha);
            ha = ha->next;
        }
    }
    act->name = grib_context_strdup_persistent(context, name);

    a->nofail = nofail;

    return act;
}

static void init_class_hash(grib_action_class* c)
{
    c->xref_gac    =    (*(c->super))->xref_gac;
    c->create_accessor    =    (*(c->super))->create_accessor;
    c->notify_change    =    (*(c->super))->notify_change;
    c->reparse    =    (*(c->super))->reparse;
    c->execute_gac    =    (*(c->super))->execute_gac;
}

static void dump_hash(grib_action* act, FILE* f, int lvl)
{
    int i = 0;

    for (i = 0; i < lvl; i++)
        grib_context_print(act->context, f, "     ");

    printf("hash_array(%s) { ", act->name);
    printf("\n");

    for (i = 0; i < lvl; i++)
        grib_context_print(act->context, f, "     ");
    printf("}\n");
}

static void destroy_hash(grib_context* context, grib_action* act)
{
    grib_action_hash_array* self = (grib_action_hash_array*)act;

    grib_hash_array_value* v = self->hash_array;
    if (v)
        grib_trie_delete(v->index);
    while (v) {
        grib_hash_array_value* n = v->next;
        grib_hash_array_value_delete(context, v);
        v = n;
    }

    grib_context_free_persistent(context, self->masterDir);
    grib_context_free_persistent(context, self->localDir);
    grib_context_free_persistent(context, self->ecmfDir);
    grib_context_free_persistent(context, self->basename);
}


static grib_hash_array_value* get_hash_array_impl(grib_handle* h, grib_action* a)
{
    char buf[4096] = {0,};
    char master[1024] = {0,};
    char local[1024] = {0,};
    char ecmf[1024] = {0,};
    char masterDir[1024] = {0,};
    size_t lenMasterDir = 1024;
    char localDir[1024] = {0,};
    size_t lenLocalDir = 1024;
    char ecmfDir[1024] = {0,};
    size_t lenEcmfDir = 1024;
    char key[4096]    = {0,};
    char* full = 0;
    int id;
    int err;
    grib_action_hash_array* self = (grib_action_hash_array*)a;

    grib_context* context    = ((grib_action*)self)->context;
    grib_hash_array_value* c = NULL;

    if (self->hash_array != NULL)
        return self->hash_array;

    Assert(self->masterDir);
    grib_get_string(h, self->masterDir, masterDir, &lenMasterDir);

    sprintf(buf, "%s/%s", masterDir, self->basename);

    err = grib_recompose_name(h, NULL, buf, master, 1);
    if (err) {
        grib_context_log(context, GRIB_LOG_ERROR,
                         "unable to build name of directory %s", self->masterDir);
        return NULL;
    }

    if (self->localDir) {
        grib_get_string(h, self->localDir, localDir, &lenLocalDir);
        sprintf(buf, "%s/%s", localDir, self->basename);
        grib_recompose_name(h, NULL, buf, local, 1);
    }

    if (self->ecmfDir) {
        grib_get_string(h, self->ecmfDir, ecmfDir, &lenEcmfDir);
        sprintf(buf, "%s/%s", ecmfDir, self->basename);
        grib_recompose_name(h, NULL, buf, ecmf, 1);
    }

    sprintf(key, "%s%s%s", master, local, ecmf);

    id = grib_itrie_get_id(h->context->hash_array_index, key);
    if ((c = h->context->hash_array[id]) != NULL)
        return c;

    if (*local && (full = grib_context_full_defs_path(context, local)) != NULL) {
        c = grib_parse_hash_array_file(context, full);
        grib_context_log(h->context, GRIB_LOG_DEBUG,
                         "Loading hash_array %s from %s", ((grib_action*)self)->name, full);
    }
    else if (*ecmf && (full = grib_context_full_defs_path(context, ecmf)) != NULL) {
        c = grib_parse_hash_array_file(context, full);
        grib_context_log(h->context, GRIB_LOG_DEBUG,
                         "Loading hash_array %s from %s", ((grib_action*)self)->name, full);
    }

    full = grib_context_full_defs_path(context, master);

    if (c) {
        grib_hash_array_value* last = c;
        while (last->next)
            last = last->next;
        last->next = grib_parse_hash_array_file(context, full);
    }
    else if (full) {
        c = grib_parse_hash_array_file(context, full);
    }
    else {
        grib_context_log(context, GRIB_LOG_ERROR,
                         "unable to find definition file %s in %s:%s:%s\nDefinition files path=\"%s\"",
                         self->basename, master, ecmf, local, context->grib_definition_files_path);
        return NULL;
    }

    grib_context_log(h->context, GRIB_LOG_DEBUG,
                     "Loading hash_array %s from %s", ((grib_action*)self)->name, full);

    h->context->hash_array[id] = c;
    if (c) {
        grib_trie* index = grib_trie_new(context);
        while (c) {
            c->index = index;
            grib_trie_insert_no_replace(index, c->name, c);
            c = c->next;
        }
    }

    return h->context->hash_array[id];
}

grib_hash_array_value* get_hash_array(grib_handle* h, grib_action* a)
{
    grib_hash_array_value* result = NULL;
    GRIB_MUTEX_INIT_ONCE(&once, &init);
    GRIB_MUTEX_LOCK(&mutex);

    result = get_hash_array_impl(h, a);

    GRIB_MUTEX_UNLOCK(&mutex);
    return result;
}

static grib_action_class _grib_action_class_noop = {
    0,                              /* super                     */
    "action_class_noop",                              /* name                      */
    sizeof(grib_action_noop),            /* size                      */
    0,                                   /* inited */
    &init_class_noop,                         /* init_class */
    0,                               /* init                      */
    &destroy_noop,                            /* destroy */

    &dump_noop,                               /* dump                      */
    &xref_noop,                               /* xref                      */

    0,             /* create_accessor*/

    0,                            /* notify_change */
    0,                            /* reparse */
    &execute_noop,                            /* execute */
};

grib_action_class* grib_action_class_noop = &_grib_action_class_noop;
grib_action* grib_action_create_noop(grib_context* context, const char* fname)
{
    char buf[1024];

    grib_action_noop* a;
    grib_action_class* c = grib_action_class_noop;
    grib_action* act     = (grib_action*)grib_context_malloc_clear_persistent(context, c->size);
    act->op              = grib_context_strdup_persistent(context, "section");

    act->cclass  = c;
    a            = (grib_action_noop*)act;
    act->context = context;

    sprintf(buf, "_noop%p", (void*)a);

    act->name = grib_context_strdup_persistent(context, buf);

    return act;
}

static void init_class_noop(grib_action_class* c)
{
}

static void dump_noop(grib_action* act, FILE* f, int lvl)
{
}

static void destroy_noop(grib_context* context, grib_action* act)
{
    grib_context_free_persistent(context, act->name);
    grib_context_free_persistent(context, act->op);
}

static void xref_noop(grib_action* d, FILE* f, const char* path)
{
}

static int execute_noop(grib_action* act, grib_handle* h)
{
    return 0;
}

extern grib_action_class* grib_action_class_section;
static grib_action_class _grib_action_class_switch = {
    &grib_action_class_section,                              /* super                     */
    "action_class_switch",                              /* name                      */
    sizeof(grib_action_switch),            /* size                      */
    0,                                   /* inited */
    &init_class_switch,                         /* init_class */
    0,                               /* init                      */
    &destroy_switch,                            /* destroy */

    0,                               /* dump                      */
    &xref_switch,                               /* xref                      */

    0,             /* create_accessor*/

    0,                            /* notify_change */
    0,                            /* reparse */
    &execute_switch,                            /* execute */
};

grib_action_class* grib_action_class_switch = &_grib_action_class_switch;
grib_action* grib_action_create_switch(grib_context* context, grib_arguments* args,
                                       grib_case* Case, grib_action* Default)
{
    char name[1024];
    grib_action_switch* a;
    grib_action_class* c = grib_action_class_switch;
    grib_action* act     = (grib_action*)grib_context_malloc_clear_persistent(context, c->size);
    act->op              = grib_context_strdup_persistent(context, "section");

    act->cclass  = c;
    a            = (grib_action_switch*)act;
    act->context = context;

    a->args    = args;
    a->Case    = Case;
    a->Default = Default;

    sprintf(name, "_switch%p", (void*)a);

    act->name = grib_context_strdup_persistent(context, name);

    return act;
}
static void init_class_switch(grib_action_class* c)
{
    c->dump_gac    =    (*(c->super))->dump_gac;
    c->create_accessor    =    (*(c->super))->create_accessor;
    c->notify_change    =    (*(c->super))->notify_change;
    c->reparse    =    (*(c->super))->reparse;
}
static int execute_switch(grib_action* act, grib_handle* h)
{
    grib_action_switch* a = (grib_action_switch*)act;
    grib_case* c          = a->Case;
    grib_action* next     = a->Default;
    grib_arguments* args  = a->args;
    grib_arguments* values;
    grib_expression* e;
    grib_expression* value;
    int ret     = 0;
    long lres   = 0;
    double dres = 0;
    long lval   = 0;
    double dval = 0;
    int type    = 0;
    int ok      = 0;
    const char* cval;
    const char* cres;
    char buf[80];
    char tmp[80];
    size_t len  = sizeof(buf);
    size_t size = sizeof(tmp);
    int err     = 0;

    Assert(args);

    while (c) {
        e      = args->expression;
        values = c->values;
        value  = values->expression;
        ok     = 0;
        while (e && value) {
            if (!strcmp(value->cclass->name, "true"))
                ok = 1;
            else {
                type = grib_expression_native_type(h, value);

                switch (type) {
                    case GRIB_TYPE_LONG:
                        ok = (grib_expression_evaluate_long(h, value, &lres) == GRIB_SUCCESS) &&
                             (grib_expression_evaluate_long(h, e, &lval) == GRIB_SUCCESS) &&
                             (lval == lres);
                        break;

                    case GRIB_TYPE_DOUBLE:
                        ok = (grib_expression_evaluate_double(h, value, &dres) == GRIB_SUCCESS) &&
                             (grib_expression_evaluate_double(h, e, &dval) == GRIB_SUCCESS) &&
                             (dval == dres);
                        break;

                    case GRIB_TYPE_STRING:
                        len  = sizeof(buf);
                        size = sizeof(tmp);
                        ok   = ((cres = grib_expression_evaluate_string(h, e, buf, &len, &err)) != NULL) &&
                             (err == 0) && ((cval = grib_expression_evaluate_string(h, value, tmp, &size, &err)) != NULL) &&
                             (err == 0) && ((strcmp(buf, cval) == 0) || (strcmp(cval, "*") == 0));
                        break;

                    default:
                        /* TODO: */
                        break;
                }
            }
            if (!ok)
                break;

            args = args->next;
            if (args)
                e = args->expression;
            else
                e = NULL;

            values = values->next;
            if (values)
                value = values->expression;
            else
                value = NULL;
        }

        if (ok) {
            next = c->action;
            break;
        }

        c = c->next;
    }

    if (!next)
        return GRIB_SWITCH_NO_MATCH;

    while (next) {
        ret = grib_action_execute(next, h);
        if (ret != GRIB_SUCCESS)
            return ret;
        next = next->next;
    }

    return GRIB_SUCCESS;
}
static void destroy_switch(grib_context* context, grib_action* act)
{
    grib_action_switch* a = (grib_action_switch*)act;
    grib_case* t          = a->Case;

    while (t) {
        grib_case* nt = t->next;
        grib_action_delete(context, t->action);
        grib_arguments_free(context, t->values);
        grib_context_free(context, t);
        t = nt;
    }

    grib_action_delete(context, a->Default);

    grib_context_free_persistent(context, act->name);
    grib_context_free_persistent(context, act->op);
}
static void xref_switch(grib_action* d, FILE* f, const char* path)
{
}

static grib_expression_class _grib_expression_class_accessor = {
    0,                    /* super                     */
    "accessor",                    /* name                      */
    sizeof(grib_expression_accessor),/* size of instance          */
    0,                           /* inited */
    &init_class_acc,                 /* init_class */
    0,                     /* constructor               */
    &destroy_acc,                  /* destructor                */
    &print_acc,                 
    &add_dependency_acc,       

	&native_type_acc,
	&get_name_acc,

	&evaluate_long_acc,
	&evaluate_double_acc,
	&evaluate_string_acc,
};
grib_expression_class* grib_expression_class_accessor = &_grib_expression_class_accessor;
grib_expression* new_accessor_expression(grib_context* c, const char* name, long start, size_t length)
{
    grib_expression_accessor* e = (grib_expression_accessor*)grib_context_malloc_clear_persistent(c, sizeof(grib_expression_accessor));
    e->base.cclass              = grib_expression_class_accessor;
    e->name                     = grib_context_strdup_persistent(c, name);
    e->start                    = start;
    e->length                   = length;
    return (grib_expression*)e;
}
static void init_class_acc(grib_expression_class* c)
{
}

static const char* get_name_acc(grib_expression* g)
{
    grib_expression_accessor* e = (grib_expression_accessor*)g;
    return e->name;
}

static int evaluate_long_acc(grib_expression* g, grib_handle* h, long* result)
{
    grib_expression_accessor* e = (grib_expression_accessor*)g;
    return grib_get_long_internal(h, e->name, result);
}

static int evaluate_double_acc(grib_expression* g, grib_handle* h, double* result)
{
    grib_expression_accessor* e = (grib_expression_accessor*)g;
    return grib_get_double_internal(h, e->name, result);
}

static string evaluate_string_acc(grib_expression* g, grib_handle* h, char* buf, size_t* size, int* err)
{
    grib_expression_accessor* e = (grib_expression_accessor*)g;
    char mybuf[1024]            = {0,};
    long start = e->start;
    if (e->length > sizeof(mybuf)) {
        *err = GRIB_INVALID_ARGUMENT;
        return NULL;
    }

    Assert(buf);
    if ((*err = grib_get_string_internal(h, e->name, mybuf, size)) != GRIB_SUCCESS)
        return NULL;

    if (e->start < 0)
        start += *size;

    if (e->length != 0) {
        if (start >= 0)
            memcpy(buf, mybuf + start, e->length);
        buf[e->length] = 0;
    }
    else {
        memcpy(buf, mybuf, *size);
        if (*size == 1024)
            *size = *size - 1; /* ECC-336 */
        buf[*size] = 0;
    }
    return buf;
}

static void print_acc(grib_context* c, grib_expression* g, grib_handle* f)
{
    grib_expression_accessor* e = (grib_expression_accessor*)g;
    printf("access('%s", e->name);
    if (f) {
        long s = 0;
        grib_get_long(f, e->name, &s);
        printf("=%ld", s);
    }
    printf("')");
}

static void destroy_acc(grib_context* c, grib_expression* g)
{
    grib_expression_accessor* e = (grib_expression_accessor*)g;
    grib_context_free_persistent(c, e->name);
}

static void add_dependency_acc(grib_expression* g, grib_accessor* observer)
{
    grib_expression_accessor* e = (grib_expression_accessor*)g;
    grib_accessor* observed     = grib_find_accessor(grib_handle_of_accessor(observer), e->name);

    if (!observed) {
        /* grib_context_log(observer->context, GRIB_LOG_ERROR, */
        /* "Error in accessor_add_dependency: cannot find [%s]", e->name); */
        /* Assert(observed); */
        return;
    }

    grib_dependency_add(observer, observed);
}

static int native_type_acc(grib_expression* g, grib_handle* h)
{
    grib_expression_accessor* e = (grib_expression_accessor*)g;
    int type                    = 0;
    int err;
    if ((err = grib_get_native_type(h, e->name, &type)) != GRIB_SUCCESS)
        grib_context_log(h->context, GRIB_LOG_ERROR,
                         "Error in native_type %s : %s", e->name, grib_get_error_message(err));
    return type;
}


static grib_expression_class _grib_expression_class_sub_string = {
    0,                    /* super                     */
    "sub_string",                    /* name                      */
    sizeof(grib_expression_sub_string),/* size of instance          */
    0,                           /* inited */
    &init_class_substr,                 /* init_class */
    0,                     /* constructor               */
    &destroy_substr,                  /* destructor                */
    &print_substr,                 
    &add_dependency_substr,       

	&native_type_substr,
	0,

	0,
	0,
	&evaluate_string_substr,
};
grib_expression_class* grib_expression_class_sub_string = &_grib_expression_class_sub_string;
grib_expression* new_sub_string_expression(grib_context* c, const char* value, size_t start, size_t length)
{
    char v[1024] = {0,};
    grib_expression_sub_string* e = (grib_expression_sub_string*)grib_context_malloc_clear_persistent(c, sizeof(grib_expression_sub_string));
    const size_t slen             = strlen(value);
    /* if (start<0) start+=strlen(value);  */

    if (length == 0) {
        grib_context_log(c, GRIB_LOG_ERROR, "Invalid substring: length must be > 0");
        grib_context_free_persistent(c, e);
        return NULL;
    }
    if (start > slen) { /* to catch a -ve number passed to start */
        grib_context_log(c, GRIB_LOG_ERROR, "Invalid substring: start=%lu", start);
        grib_context_free_persistent(c, e);
        return NULL;
    }
    if (start + length > slen) {
        grib_context_log(c, GRIB_LOG_ERROR, "Invalid substring: start(=%lu)+length(=%lu) > length('%s'))", start, length, value);
        grib_context_free_persistent(c, e);
        return NULL;
    }

    memcpy(v, value + start, length);
    e->base.cclass = grib_expression_class_sub_string;
    e->value       = grib_context_strdup_persistent(c, v);
    return (grib_expression*)e;
}
static void init_class_substr(grib_expression_class* c)
{
}

static const char* evaluate_string_substr(grib_expression* g, grib_handle* h, char* buf, size_t* size, int* err)
{
    grib_expression_sub_string* e = (grib_expression_sub_string*)g;
    *err                          = 0;
    return e->value;
}

static void print_substr(grib_context* c, grib_expression* g, grib_handle* f)
{
    grib_expression_sub_string* e = (grib_expression_sub_string*)g;
    printf("string('%s')", e->value);
}

static void destroy_substr(grib_context* c, grib_expression* g)
{
    grib_expression_sub_string* e = (grib_expression_sub_string*)g;
    grib_context_free_persistent(c, e->value);
}

static void add_dependency_substr(grib_expression* g, grib_accessor* observer)
{
    /* grib_expression_sub_string* e = (grib_expression_sub_string*)g; */
}

static int native_type_substr(grib_expression* g, grib_handle* h)
{
    return GRIB_TYPE_STRING;
}

static grib_expression_class _grib_expression_class_string = {
    0,                    /* super                     */
    "string",                    /* name                      */
    sizeof(grib_expression_string),/* size of instance          */
    0,                           /* inited */
    &init_class_string,                 /* init_class */
    0,                     /* constructor               */
    &destroy_string,                  /* destructor                */
    &print_string,                 
    &add_dependency_string,       

	&native_type_string,
	0,

	0,
	0,
	&evaluate_string_string,
};
grib_expression_class* grib_expression_class_string = &_grib_expression_class_string;
grib_expression* new_string_expression(grib_context* c, const char* value)
{
    grib_expression_string* e = (grib_expression_string*)grib_context_malloc_clear_persistent(c, sizeof(grib_expression_string));
    e->base.cclass            = grib_expression_class_string;
    e->value                  = grib_context_strdup_persistent(c, value);
    return (grib_expression*)e;
}

static void init_class_string(grib_expression_class* c)
{
}

static const char* evaluate_string_string(grib_expression* g, grib_handle* h, char* buf, size_t* size, int* err)
{
    grib_expression_string* e = (grib_expression_string*)g;
    *err                      = 0;
    return e->value;
}

static void print_string(grib_context* c, grib_expression* g, grib_handle* f)
{
    grib_expression_string* e = (grib_expression_string*)g;
    printf("string('%s')", e->value);
}

static void destroy_string(grib_context* c, grib_expression* g)
{
    grib_expression_string* e = (grib_expression_string*)g;
    grib_context_free_persistent(c, e->value);
}

static void add_dependency_string(grib_expression* g, grib_accessor* observer)
{
    /* grib_expression_string* e = (grib_expression_string*)g; */
}

static int native_type_string(grib_expression* g, grib_handle* h)
{
    return GRIB_TYPE_STRING;
}


static grib_expression_class _grib_expression_class_long = {
    0,                    /* super                     */
    "long",                    /* name                      */
    sizeof(grib_expression_long),/* size of instance          */
    0,                           /* inited */
    &init_class_long,                 /* init_class */
    0,                     /* constructor               */
    &destroy_long,                  /* destructor                */
    &print_long,                 
    &add_dependency_long,       

	&native_type_long,
	0,

	&evaluate_long_long,
	&evaluate_double_long,
	0,
};
grib_expression_class* grib_expression_class_long = &_grib_expression_class_long;
grib_expression* new_long_expression(grib_context* c, long value)
{
    grib_expression_long* e = (grib_expression_long*)grib_context_malloc_clear_persistent(c, sizeof(grib_expression_long));
    e->base.cclass          = grib_expression_class_long;
    e->value                = value;
    return (grib_expression*)e;
}
static void init_class_long(grib_expression_class* c)
{
}

static int evaluate_long_long(grib_expression* g, grib_handle* h, long* lres)
{
    grib_expression_long* e = (grib_expression_long*)g;
    *lres                   = e->value;
    return GRIB_SUCCESS;
}

static int evaluate_double_long(grib_expression* g, grib_handle* h, double* dres)
{
    grib_expression_long* e = (grib_expression_long*)g;
    *dres                   = e->value;
    return GRIB_SUCCESS;
}

static void print_long(grib_context* c, grib_expression* g, grib_handle* f)
{
    grib_expression_long* e = (grib_expression_long*)g;
    printf("long(%ld)", e->value);
}

static void destroy_long(grib_context* c, grib_expression* g)
{
    /* grib_expression_long* e = (grib_expression_long*)g; */
}

static void add_dependency_long(grib_expression* g, grib_accessor* observer)
{
    /* grib_expression_long* e = (grib_expression_long*)g; */
}

static int native_type_long(grib_expression* g, grib_handle* h)
{
    return GRIB_TYPE_LONG;
}


static grib_expression_class _grib_expression_class_double = {
    0,                    /* super                     */
    "double",                    /* name                      */
    sizeof(grib_expression_double),/* size of instance          */
    0,                           /* inited */
    &init_class_double,                 /* init_class */
    0,                     /* constructor               */
    &destroy_double,                  /* destructor                */
    &print_double,                 
    &add_dependency_double,       

	&native_type_double,
	0,

	&evaluate_long_double,
	&evaluate_double_double,
	0,
};

grib_expression_class* grib_expression_class_double = &_grib_expression_class_double;
grib_expression* new_double_expression(grib_context* c, double value)
{
    grib_expression_double* e = (grib_expression_double*)grib_context_malloc_clear_persistent(c, sizeof(grib_expression_double));
    e->base.cclass            = grib_expression_class_double;
    e->value                  = value;
    return (grib_expression*)e;
}
static void init_class_double(grib_expression_class* c)
{
}

static int evaluate_long_double(grib_expression* g, grib_handle* h, long* lres)
{
    grib_expression_double* e = (grib_expression_double*)g;
    *lres                     = e->value;
    return GRIB_SUCCESS;
}

static int evaluate_double_double(grib_expression* g, grib_handle* h, double* dres)
{
    grib_expression_double* e = (grib_expression_double*)g;
    *dres                     = e->value;
    return GRIB_SUCCESS;
}

static void print_double(grib_context* c, grib_expression* g, grib_handle* f)
{
    grib_expression_double* e = (grib_expression_double*)g;
    printf("double(%g)", e->value);
}

static void destroy_double(grib_context* c, grib_expression* g)
{
    /* grib_expression_double* e = (grib_expression_double*)g; */
}

static void add_dependency_double(grib_expression* g, grib_accessor* observer)
{
    /* grib_expression_double* e = (grib_expression_double*)g; */
}

static int native_type_double(grib_expression* g, grib_handle* h)
{
    return GRIB_TYPE_DOUBLE;
}


static grib_expression_class _grib_expression_class_true = {
    0,                    /* super                     */
    "true",                    /* name                      */
    sizeof(grib_expression_true),/* size of instance          */
    0,                           /* inited */
    &init_class_true,                 /* init_class */
    0,                     /* constructor               */
    &destroy_true,                  /* destructor                */
    &print_true,                 
    &add_dependency_true,       

	&native_type_true,
	0,

	&evaluate_long_true,
	&evaluate_double_true,
	0,
};
grib_expression_class* grib_expression_class_true = &_grib_expression_class_true;
grib_expression* new_true_expression(grib_context* c)
{
    grib_expression_true* e = (grib_expression_true*)grib_context_malloc_clear_persistent(c, sizeof(grib_expression_true));
    e->base.cclass          = grib_expression_class_true;
    return (grib_expression*)e;
}
static void init_class_true(grib_expression_class* c)
{
}

static int evaluate_long_true(grib_expression* g, grib_handle* h, long* lres)
{
    *lres = 1;
    return GRIB_SUCCESS;
}

static int evaluate_double_true(grib_expression* g, grib_handle* h, double* dres)
{
    *dres = 1;
    return GRIB_SUCCESS;
}

static void print_true(grib_context* c, grib_expression* g, grib_handle* f)
{
    printf("true(");
    printf(")");
}

static void destroy_true(grib_context* c, grib_expression* g)
{
}

static void add_dependency_true(grib_expression* g, grib_accessor* observer)
{
}

static int native_type_true(grib_expression* g, grib_handle* h)
{
    return GRIB_TYPE_LONG;
}

static grib_expression_class _grib_expression_class_functor = {
    0,                    /* super                     */
    "functor",                    /* name                      */
    sizeof(grib_expression_functor),/* size of instance          */
    0,                           /* inited */
    &init_class_functor,                 /* init_class */
    0,                     /* constructor               */
    &destroy_functor,                  /* destructor                */
    &print_functor,                 
    &add_dependency_functor,       

	&native_type_functor,
	0,

	&evaluate_long_functor,
	0,
	0,
};

grib_expression_class* grib_expression_class_functor = &_grib_expression_class_functor;

grib_expression* new_func_expression(grib_context* c, const char* name, grib_arguments* args)
{
    grib_expression_functor* e = (grib_expression_functor*)grib_context_malloc_clear_persistent(c, sizeof(grib_expression_functor));
    e->base.cclass             = grib_expression_class_functor;
    e->name                    = grib_context_strdup_persistent(c, name);
    e->args                    = args;
    return (grib_expression*)e;
}

static void init_class_functor(grib_expression_class* c)
{
}
/* END_CLASS_IMP */

static int evaluate_long_functor(grib_expression* g, grib_handle* h, long* lres)
{
    grib_expression_functor* e = (grib_expression_functor*)g;

    /*
    TODO: needs OO code here
     */
    if (strcmp(e->name, "lookup") == 0) {
        return GRIB_SUCCESS;
    }

    if (strcmp(e->name, "new") == 0) {
        *lres = h->loader != NULL;
        return GRIB_SUCCESS;
    }

    if (strcmp(e->name, "missing") == 0) {
        const char* p = grib_arguments_get_name(h, e->args, 0);
        if (p) {
            long val = 0;
            int err  = 0;
            if (h->product_kind == PRODUCT_BUFR) {
                int ismiss = grib_is_missing(h, p, &err);
                if (err) return err;
                *lres = ismiss;
                return GRIB_SUCCESS;
            }
            err = grib_get_long_internal(h, p, &val);
            if (err) return err;
            /* Note: This does not cope with keys like typeOfSecondFixedSurface
             * which are codetable entries with values like 255: this value is
             * not classed as 'missing'!
             * (See ECC-594)
             */
            *lres = (val == GRIB_MISSING_LONG);
            return GRIB_SUCCESS;
        }
        else {
            /* No arguments means return the actual integer missing value */
            *lres = GRIB_MISSING_LONG;
        }
        return GRIB_SUCCESS;
    }

    if (strcmp(e->name, "defined") == 0) {
        const char* p = grib_arguments_get_name(h, e->args, 0);

        if (p) {
            grib_accessor* a = grib_find_accessor(h, p);
            *lres            = a != NULL ? 1 : 0;
            return GRIB_SUCCESS;
        }
        *lres = 0;
        return GRIB_SUCCESS;
    }

    if (strcmp(e->name, "changed") == 0) {
        *lres = 1;
        return GRIB_SUCCESS;
    }

    if (strcmp(e->name, "gribex_mode_on") == 0) {
        *lres = h->context->gribex_mode_on ? 1 : 0;
        return GRIB_SUCCESS;
    }

    return GRIB_NOT_IMPLEMENTED;
}

static void print_functor(grib_context* c, grib_expression* g, grib_handle* f)
{
    grib_expression_functor* e = (grib_expression_functor*)g;
    printf("%s(", e->name);
    /*grib_expression_print(c,e->args,f);*/
    printf(")");
}

static void destroy_functor(grib_context* c, grib_expression* g)
{
    grib_expression_functor* e = (grib_expression_functor*)g;
    grib_context_free_persistent(c, e->name);
    grib_arguments_free(c, e->args);
}

static void add_dependency_functor(grib_expression* g, grib_accessor* observer)
{
    grib_expression_functor* e = (grib_expression_functor*)g;
    if (strcmp(e->name, "defined"))
        grib_dependency_observe_arguments(observer, e->args);
}

static int native_type_functor(grib_expression* g, grib_handle* h)
{
    return GRIB_TYPE_LONG;
}


grib_rule* grib_new_rule(grib_context* c, grib_expression* condition, grib_rule_entry* entries)
{
    grib_rule* r = (grib_rule*)grib_context_malloc_clear_persistent(c, sizeof(grib_rule));
    r->condition = condition;
    r->entries   = entries;
    return r;
}

grib_rule_entry* grib_new_rule_entry(grib_context* c, const char* name, grib_expression* expression)
{
    grib_rule_entry* e = (grib_rule_entry*)grib_context_malloc_clear_persistent(c, sizeof(grib_rule_entry));
    e->name            = grib_context_strdup_persistent(c, name);
    e->value           = expression;
    return e;
}

static grib_expression_class _grib_expression_class_logical_or = {
    0,                    /* super                     */
    "logical_or",                    /* name                      */
    sizeof(grib_expression_logical_or),/* size of instance          */
    0,                           /* inited */
    &init_class_log_or,                 /* init_class */
    0,                     /* constructor               */
    &destroy_log_or,                  /* destructor                */
    &print_log_or,                 
    &add_dependency_log_or,       

	&native_type_log_or,
	0,

	&evaluate_long_log_or,
	&evaluate_double_log_or,
	0,
};

grib_expression_class* grib_expression_class_logical_or = &_grib_expression_class_logical_or;

grib_expression* new_logical_or_expression(grib_context* c, grib_expression* left, grib_expression* right)
{
    grib_expression_logical_or* e = (grib_expression_logical_or*)grib_context_malloc_clear_persistent(c, sizeof(grib_expression_logical_or));
    e->base.cclass                = grib_expression_class_logical_or;
    e->left                       = left;
    e->right                      = right;
    return (grib_expression*)e;
}

static void init_class_log_or(grib_expression_class* c)
{
}

static void destroy_log_or(grib_context* c, grib_expression* g)
{
    grib_expression_logical_or* e = (grib_expression_logical_or*)g;
    grib_expression_free(c, e->left);
    grib_expression_free(c, e->right);
}

static void print_log_or(grib_context* c, grib_expression* g, grib_handle* f)
{
    grib_expression_logical_or* e = (grib_expression_logical_or*)g;
    printf("(");
    grib_expression_print(c, e->left, f);
    printf(" && ");
    grib_expression_print(c, e->right, f);
    printf(")");
}

static void add_dependency_log_or(grib_expression* g, grib_accessor* observer)
{
    grib_expression_logical_or* e = (grib_expression_logical_or*)g;
    grib_dependency_observe_expression(observer, e->left);
    grib_dependency_observe_expression(observer, e->right);
}

static int native_type_log_or(grib_expression* g, grib_handle* h)
{
    return GRIB_TYPE_LONG;
}

static int evaluate_long_log_or(grib_expression* g, grib_handle* h, long* lres)
{
    long v1    = 0;
    long v2    = 0;
    double dv1 = 0;
    double dv2 = 0;
    int ret;
    grib_expression_logical_or* e = (grib_expression_logical_or*)g;


    switch (grib_expression_native_type(h, e->left)) {
        case GRIB_TYPE_LONG:
            ret = grib_expression_evaluate_long(h, e->left, &v1);
            if (ret != GRIB_SUCCESS)
                return ret;
            if (v1 != 0) {
                *lres = 1;
                return ret;
            }
            break;
        case GRIB_TYPE_DOUBLE:
            ret = grib_expression_evaluate_double(h, e->left, &dv1);
            if (ret != GRIB_SUCCESS)
                return ret;
            if (dv1 != 0) {
                *lres = 1;
                return ret;
            }
            break;
        default:
            return GRIB_INVALID_TYPE;
    }

    switch (grib_expression_native_type(h, e->right)) {
        case GRIB_TYPE_LONG:
            ret = grib_expression_evaluate_long(h, e->right, &v2);
            if (ret != GRIB_SUCCESS)
                return ret;
            *lres = v2 ? 1 : 0;
            break;
        case GRIB_TYPE_DOUBLE:
            ret = grib_expression_evaluate_double(h, e->right, &dv2);
            if (ret != GRIB_SUCCESS)
                return ret;
            *lres = dv2 ? 1 : 0;
            break;
        default:
            return GRIB_INVALID_TYPE;
    }

    return GRIB_SUCCESS;
}

static int evaluate_double_log_or(grib_expression* g, grib_handle* h, double* dres)
{
    long lres = 0;
    int ret   = 0;

    ret   = evaluate_long_log_or(g, h, &lres);
    *dres = (double)lres;

    return ret;
}

static grib_expression_class _grib_expression_class_unop = {
    0,                    /* super                     */
    "unop",                    /* name                      */
    sizeof(grib_expression_unop),/* size of instance          */
    0,                           /* inited */
    &init_class_unop,                 /* init_class */
    0,                     /* constructor               */
    &destroy_unop,                  /* destructor                */
    &print_unop,                 
    &add_dependency_unop,       

	&native_type_unop,
	0,

	&evaluate_long_unop,
	&evaluate_double_unop,
	0,
};

grib_expression_class* grib_expression_class_unop = &_grib_expression_class_unop;

grib_expression* new_unop_expression(grib_context* c,
                                     grib_unop_long_proc long_func,
                                     grib_unop_double_proc double_func,
                                     grib_expression* exp)
{
    grib_expression_unop* e = (grib_expression_unop*)grib_context_malloc_clear_persistent(c, sizeof(grib_expression_unop));
    e->base.cclass          = grib_expression_class_unop;
    e->exp                  = exp;
    e->long_func            = long_func;
    e->double_func          = double_func;
    return (grib_expression*)e;
}

static void init_class_unop(grib_expression_class* c)
{
}

static int evaluate_long_unop(grib_expression* g, grib_handle* h, long* lres)
{
    int ret;
    long v                  = 0;
    grib_expression_unop* e = (grib_expression_unop*)g;
    ret                     = grib_expression_evaluate_long(h, e->exp, &v);
    if (ret != GRIB_SUCCESS)
        return ret;
    *lres = e->long_func(v);
    return GRIB_SUCCESS;
}

static int evaluate_double_unop(grib_expression* g, grib_handle* h, double* dres)
{
    int ret;
    double v                = 0;
    grib_expression_unop* e = (grib_expression_unop*)g;
    ret                     = grib_expression_evaluate_double(h, e->exp, &v);
    if (ret != GRIB_SUCCESS)
        return ret;
    *dres = e->double_func ? e->double_func(v) : e->long_func(v);
    return GRIB_SUCCESS;
}

static void print_unop(grib_context* c, grib_expression* g, grib_handle* f)
{
    grib_expression_unop* e = (grib_expression_unop*)g;
    printf("unop(");
    grib_expression_print(c, e->exp, f);
    printf(")");
}

static void destroy_unop(grib_context* c, grib_expression* g)
{
    grib_expression_unop* e = (grib_expression_unop*)g;
    grib_expression_free(c, e->exp);
}


static void add_dependency_unop(grib_expression* g, grib_accessor* observer)
{
    grib_expression_unop* e = (grib_expression_unop*)g;
    grib_dependency_observe_expression(observer, e->exp);
}

static grib_expression_class _grib_expression_class_logical_and = {
    0,                    /* super                     */
    "logical_and",                    /* name                      */
    sizeof(grib_expression_logical_and),/* size of instance          */
    0,                           /* inited */
    &init_class_logand,                 /* init_class */
    0,                     /* constructor               */
    &destroy_logand,                  /* destructor                */
    &print_logand,                 
    &add_dependency_logand,       

	&native_type_logand,
	0,

	&evaluate_long_logand,
	&evaluate_double_logand,
	0,
};

grib_expression_class* grib_expression_class_logical_and = &_grib_expression_class_logical_and;

grib_expression* new_logical_and_expression(grib_context* c, grib_expression* left, grib_expression* right)
{
    grib_expression_logical_and* e = (grib_expression_logical_and*)grib_context_malloc_clear_persistent(c, sizeof(grib_expression_logical_and));
    e->base.cclass                 = grib_expression_class_logical_and;
    e->left                        = left;
    e->right                       = right;
    return (grib_expression*)e;
}

static void init_class_logand(grib_expression_class* c)
{
}

static int native_type_logand(grib_expression* g, grib_handle* h)
{
    return GRIB_TYPE_LONG;
}

static int evaluate_long_logand(grib_expression* g, grib_handle* h, long* lres)
{
    long v1    = 0;
    long v2    = 0;
    double dv1 = 0;
    double dv2 = 0;
    int ret;
    grib_expression_logical_and* e = (grib_expression_logical_and*)g;


    switch (grib_expression_native_type(h, e->left)) {
        case GRIB_TYPE_LONG:
            ret = grib_expression_evaluate_long(h, e->left, &v1);
            if (ret != GRIB_SUCCESS)
                return ret;
            if (v1 == 0) {
                *lres = 0;
                return ret;
            }
            break;
        case GRIB_TYPE_DOUBLE:
            ret = grib_expression_evaluate_double(h, e->left, &dv1);
            if (ret != GRIB_SUCCESS)
                return ret;
            if (dv1 == 0) {
                *lres = 0;
                return ret;
            }
            break;
        default:
            return GRIB_INVALID_TYPE;
    }

    switch (grib_expression_native_type(h, e->right)) {
        case GRIB_TYPE_LONG:
            ret = grib_expression_evaluate_long(h, e->right, &v2);
            if (ret != GRIB_SUCCESS)
                return ret;
            *lres = v2 ? 1 : 0;
            break;
        case GRIB_TYPE_DOUBLE:
            ret = grib_expression_evaluate_double(h, e->right, &dv2);
            if (ret != GRIB_SUCCESS)
                return ret;
            *lres = dv2 ? 1 : 0;
            break;
        default:
            return GRIB_INVALID_TYPE;
    }

    return GRIB_SUCCESS;
}

static int evaluate_double_logand(grib_expression* g, grib_handle* h, double* dres)
{
    long lres = 0;
    int ret   = 0;

    ret   = evaluate_long_logand(g, h, &lres);
    *dres = (double)lres;

    return ret;
}

static void print_logand(grib_context* c, grib_expression* g, grib_handle* f)
{
    grib_expression_logical_and* e = (grib_expression_logical_and*)g;
    printf("(");
    grib_expression_print(c, e->left, f);
    printf(" && ");
    grib_expression_print(c, e->right, f);
    printf(")");
}

static void destroy_logand(grib_context* c, grib_expression* g)
{
    grib_expression_logical_and* e = (grib_expression_logical_and*)g;
    grib_expression_free(c, e->left);
    grib_expression_free(c, e->right);
}

static void add_dependency_logand(grib_expression* g, grib_accessor* observer)
{
    grib_expression_logical_and* e = (grib_expression_logical_and*)g;
    grib_dependency_observe_expression(observer, e->left);
    grib_dependency_observe_expression(observer, e->right);
}

static grib_expression_class _grib_expression_class_string_compare = {
    0,                    /* super                     */
    "string_compare",                    /* name                      */
    sizeof(grib_expression_string_compare),/* size of instance          */
    0,                           /* inited */
    &init_class_strcmp,                 /* init_class */
    0,                     /* constructor               */
    &destroy_strcmp,                  /* destructor                */
    &print_strcmp,                 
    &add_dependency_strcmp,       

	&native_type_strcmp,
	0,

	&evaluate_long_strcmp,
	&evaluate_double_strcmp,
	0,
};

grib_expression_class* grib_expression_class_string_compare = &_grib_expression_class_string_compare;
grib_expression* new_string_compare_expression(grib_context* c,
                                               grib_expression* left, grib_expression* right)
{
    grib_expression_string_compare* e = (grib_expression_string_compare*)grib_context_malloc_clear_persistent(c, sizeof(grib_expression_string_compare));
    e->base.cclass                    = grib_expression_class_string_compare;
    e->left                           = left;
    e->right                          = right;
    return (grib_expression*)e;
}

static void init_class_strcmp(grib_expression_class* c)
{
}

static int evaluate_long_strcmp(grib_expression* g, grib_handle* h, long* lres)
{
    int ret = 0;
    char b1[1024];
    size_t l1 = sizeof(b1);
    char b2[1024];
    size_t l2 = sizeof(b2);
    const char* v1 = NULL;
    const char* v2 = NULL;

    grib_expression_string_compare* e = (grib_expression_string_compare*)g;

    v1 = grib_expression_evaluate_string(h, e->left, b1, &l1, &ret);
    if (!v1 || ret) {
        *lres = 0;
        return ret;
    }

    v2 = grib_expression_evaluate_string(h, e->right, b2, &l2, &ret);
    if (!v2 || ret) {
        *lres = 0;
        return ret;
    }

    *lres = (grib_inline_strcmp(v1, v2) == 0);
    return GRIB_SUCCESS;
}

static int evaluate_double_strcmp(grib_expression* g, grib_handle* h, double* dres)
{
    long n;
    int ret = evaluate_long_strcmp(g, h, &n);
    *dres   = n;
    return ret;
}

static void print_strcmp(grib_context* c, grib_expression* g, grib_handle* f)
{
    grib_expression_string_compare* e = (grib_expression_string_compare*)g;
    printf("string_compare(");
    grib_expression_print(c, e->left, f);
    printf(",");
    grib_expression_print(c, e->right, f);
    printf(")");
}

static void destroy_strcmp(grib_context* c, grib_expression* g)
{
    grib_expression_string_compare* e = (grib_expression_string_compare*)g;
    grib_expression_free(c, e->left);
    grib_expression_free(c, e->right);
}

static void add_dependency_strcmp(grib_expression* g, grib_accessor* observer)
{
    grib_expression_string_compare* e = (grib_expression_string_compare*)g;
    grib_dependency_observe_expression(observer, e->left);
    grib_dependency_observe_expression(observer, e->right);
}

static int native_type_strcmp(grib_expression* g, grib_handle* h)
{
    return GRIB_TYPE_LONG;
}


static grib_expression_class _grib_expression_class_binop = {
    0,                    /* super                     */
    "binop",                    /* name                      */
    sizeof(grib_expression_binop),/* size of instance          */
    0,                           /* inited */
    &init_class_binop,                 /* init_class */
    0,                     /* constructor               */
    &destroy_binop,                  /* destructor                */
    &print_binop,                 
    &add_dependency_binop,       

	&native_type_binop,
	0,

	&evaluate_long_binop,
	&evaluate_double_binop,
	0,
};

grib_expression_class* grib_expression_class_binop = &_grib_expression_class_binop;

grib_expression* new_binop_expression(grib_context* c,
                                      grib_binop_long_proc long_func,
                                      grib_binop_double_proc double_func,
                                      grib_expression* left, grib_expression* right)
{
    grib_expression_binop* e = (grib_expression_binop*)grib_context_malloc_clear_persistent(c, sizeof(grib_expression_binop));
    e->base.cclass           = grib_expression_class_binop;
    e->left                  = left;
    e->right                 = right;
    e->long_func             = long_func;
    e->double_func           = double_func;
    return (grib_expression*)e;
}

static void init_class_binop(grib_expression_class* c)
{
}

static int evaluate_long_binop(grib_expression* g, grib_handle* h, long* lres)
{
    int ret = 0;
    char b1[1024];
    size_t l1 = sizeof(b1);
    char b2[1024];
    size_t l2 = sizeof(b2);
    const char* v1 = NULL;
    const char* v2 = NULL;

    grib_expression_string_compare* e = (grib_expression_string_compare*)g;

    v1 = grib_expression_evaluate_string(h, e->left, b1, &l1, &ret);
    if (!v1 || ret) {
        *lres = 0;
        return ret;
    }

    v2 = grib_expression_evaluate_string(h, e->right, b2, &l2, &ret);
    if (!v2 || ret) {
        *lres = 0;
        return ret;
    }

    *lres = (grib_inline_strcmp(v1, v2) == 0);
    return GRIB_SUCCESS;
}

static int evaluate_double_binop(grib_expression* g, grib_handle* h, double* dres)
{
    long n;
    int ret = evaluate_long_binop(g, h, &n);
    *dres   = n;
    return ret;
}

static void print_binop(grib_context* c, grib_expression* g, grib_handle* f)
{
    grib_expression_string_compare* e = (grib_expression_string_compare*)g;
    printf("string_compare(");
    grib_expression_print(c, e->left, f);
    printf(",");
    grib_expression_print(c, e->right, f);
    printf(")");
}

static void destroy_binop(grib_context* c, grib_expression* g)
{
    grib_expression_string_compare* e = (grib_expression_string_compare*)g;
    grib_expression_free(c, e->left);
    grib_expression_free(c, e->right);
}

static void add_dependency_binop(grib_expression* g, grib_accessor* observer)
{
    grib_expression_string_compare* e = (grib_expression_string_compare*)g;
    grib_dependency_observe_expression(observer, e->left);
    grib_dependency_observe_expression(observer, e->right);
}

static int native_type_binop(grib_expression* g, grib_handle* h)
{
    return GRIB_TYPE_LONG;
}

static grib_expression_class _grib_expression_class_is_integer = {
    0,                    /* super                     */
    "is_integer",                    /* name                      */
    sizeof(grib_expression_is_integer),/* size of instance          */
    0,                           /* inited */
    &init_class_int,                 /* init_class */
    0,                     /* constructor               */
    &destroy_int,                  /* destructor                */
    &print_int,                 
    &add_dependency_int,       

	&native_type_int,
	&get_name_int,

	&evaluate_long_int,
	&evaluate_double_int,
	&evaluate_string_int,
};

grib_expression_class* grib_expression_class_is_integer = &_grib_expression_class_is_integer;

grib_expression* new_is_integer_expression(grib_context* c, const char* name, int start, int length)
{
    grib_expression_is_integer* e = (grib_expression_is_integer*)grib_context_malloc_clear_persistent(c, sizeof(grib_expression_is_integer));
    e->base.cclass                = grib_expression_class_is_integer;
    e->name                       = grib_context_strdup_persistent(c, name);
    e->start                      = start;
    e->length                     = length;
    return (grib_expression*)e;
}

static void init_class_int(grib_expression_class* c)
{
}

static const char* get_name_int(grib_expression* g)
{
    grib_expression_is_integer* e = (grib_expression_is_integer*)g;
    return e->name;
}

static int evaluate_long_int(grib_expression* g, grib_handle* h, long* result)
{
    grib_expression_is_integer* e = (grib_expression_is_integer*)g;
    int err                       = 0;
    char mybuf[1024]              = {0,};
    size_t size = 1024;
    char* p     = 0;
    long val    = 0;
    char* start = 0;

    if ((err = grib_get_string_internal(h, e->name, mybuf, &size)) != GRIB_SUCCESS)
        return err;

    start = mybuf + e->start;

    if (e->length > 0)
        start[e->length] = 0;

    val = strtol(start, &p, 10);

    if (*p != 0)
        *result = 0;
    else
        *result = 1;

    (void)val;
    return err;
}

static int evaluate_double_int(grib_expression* g, grib_handle* h, double* result)
{
    int err      = 0;
    long lresult = 0;

    err     = evaluate_long_int(g, h, &lresult);
    *result = lresult;
    return err;
}

static string evaluate_string_int(grib_expression* g, grib_handle* h, char* buf, size_t* size, int* err)
{
    long lresult   = 0;
    double dresult = 0.0;

    switch (grib_expression_native_type(h, g)) {
        case GRIB_TYPE_LONG:
            *err = evaluate_long_int(g, h, &lresult);
            sprintf(buf, "%ld", lresult);
            break;
        case GRIB_TYPE_DOUBLE:
            *err = evaluate_double_int(g, h, &dresult);
            sprintf(buf, "%g", dresult);
            break;
    }
    return buf;
}

static void print_int(grib_context* c, grib_expression* g, grib_handle* f)
{
    grib_expression_is_integer* e = (grib_expression_is_integer*)g;
    printf("access('%s", e->name);
    if (f) {
        long s = 0;
        grib_get_long(f, e->name, &s);
        printf("=%ld", s);
    }
    printf("')");
}

static void destroy_int(grib_context* c, grib_expression* g)
{
    grib_expression_is_integer* e = (grib_expression_is_integer*)g;
    grib_context_free_persistent(c, e->name);
}

static void add_dependency_int(grib_expression* g, grib_accessor* observer)
{
    grib_expression_is_integer* e = (grib_expression_is_integer*)g;
    grib_accessor* observed       = grib_find_accessor(grib_handle_of_accessor(observer), e->name);

    if (!observed) {
        /* grib_context_log(observer->context, GRIB_LOG_ERROR, */
        /* "Error in accessor_add_dependency: cannot find [%s]", e->name); */
        /* Assert(observed); */
        return;
    }

    grib_dependency_add(observer, observed);
}

static int native_type_int(grib_expression* g, grib_handle* h)
{
    return GRIB_TYPE_LONG;
}


static grib_expression_class _grib_expression_class_is_in_dict = {
    0,                    /* super                     */
    "is_in_dict",                    /* name                      */
    sizeof(grib_expression_is_in_dict),/* size of instance          */
    0,                           /* inited */
    &init_class_dict,                 /* init_class */
    0,                     /* constructor               */
    0,                  /* destructor                */
    &print_dict,                 
    &add_dependency_dict,       

	&native_type_dict,
	&get_name_dict,

	&evaluate_long_dict,
	&evaluate_double_dict,
	&evaluate_string_dict,
};

grib_expression_class* grib_expression_class_is_in_dict = &_grib_expression_class_is_in_dict;

grib_expression* new_is_in_dict_expression(grib_context* c, const char* name, const char* list)
{
    grib_expression_is_in_dict* e = (grib_expression_is_in_dict*)grib_context_malloc_clear_persistent(c, sizeof(grib_expression_is_in_dict));
    e->base.cclass                = grib_expression_class_is_in_dict;
    e->key                        = grib_context_strdup_persistent(c, name);
    e->dictionary                 = grib_context_strdup_persistent(c, list);
    return (grib_expression*)e;
}

static void init_class_dict(grib_expression_class* c)
{
}

static grib_trie* load_dictionary_dict(grib_context* c, grib_expression* e, int* err)
{
    grib_expression_is_in_dict* self = (grib_expression_is_in_dict*)e;

    char* filename  = NULL;
    char line[1024] = {0,};
    char key[1024] = {0,};
    char* list            = 0;
    grib_trie* dictionary = NULL;
    FILE* f               = NULL;
    int i                 = 0;

    *err = GRIB_SUCCESS;

    filename = grib_context_full_defs_path(c, self->dictionary);
    if (!filename) {
        grib_context_log(c, GRIB_LOG_ERROR, "unable to find def file %s", self->dictionary);
        *err = GRIB_FILE_NOT_FOUND;
        return NULL;
    }
    else {
        grib_context_log(c, GRIB_LOG_DEBUG, "is_in_dict: found def file %s", filename);
    }
    dictionary = (grib_trie*)grib_trie_get(c->lists, filename);
    if (dictionary) {
        grib_context_log(c, GRIB_LOG_DEBUG, "using dictionary %s from cache", self->dictionary);
        return dictionary;
    }
    else {
        grib_context_log(c, GRIB_LOG_DEBUG, "using dictionary %s from file %s", self->dictionary, filename);
    }

    f = codes_fopen(filename, "r");
    if (!f) {
        *err = GRIB_IO_PROBLEM;
        return NULL;
    }

    dictionary = grib_trie_new(c);

    while (fgets(line, sizeof(line) - 1, f)) {
        i = 0;
        while (line[i] != '|' && line[i] != 0) {
            key[i] = line[i];
            i++;
        }
        key[i] = 0;
        list   = (char*)grib_context_malloc_clear(c, strlen(line) + 1);
        memcpy(list, line, strlen(line));
        grib_trie_insert(dictionary, key, list);
    }

    grib_trie_insert(c->lists, filename, dictionary);

    fclose(f);

    return dictionary;
}

static const char* get_name_dict(grib_expression* g)
{
    grib_expression_is_in_dict* e = (grib_expression_is_in_dict*)g;
    return e->key;
}

static int evaluate_long_dict(grib_expression* g, grib_handle* h, long* result)
{
    grib_expression_is_in_dict* e = (grib_expression_is_in_dict*)g;
    int err                       = 0;
    char mybuf[1024]              = {0,};
    size_t size = 1024;

    grib_trie* dict = load_dictionary_dict(h->context, g, &err);

    if ((err = grib_get_string_internal(h, e->key, mybuf, &size)) != GRIB_SUCCESS)
        return err;

    if (grib_trie_get(dict, mybuf))
        *result = 1;
    else
        *result = 0;

    return err;
}

static int evaluate_double_dict(grib_expression* g, grib_handle* h, double* result)
{
    grib_expression_is_in_dict* e = (grib_expression_is_in_dict*)g;
    int err                       = 0;
    char mybuf[1024]              = {0,};
    size_t size = 1024;

    grib_trie* list = load_dictionary_dict(h->context, g, &err);

    if ((err = grib_get_string_internal(h, e->key, mybuf, &size)) != GRIB_SUCCESS)
        return err;

    if (grib_trie_get(list, mybuf))
        *result = 1;
    else
        *result = 0;

    return err;
}

static string evaluate_string_dict(grib_expression* g, grib_handle* h, char* buf, size_t* size, int* err)
{
    grib_expression_is_in_dict* e = (grib_expression_is_in_dict*)g;
    char mybuf[1024]              = {0,};
    size_t sizebuf = 1024;
    long result;

    grib_trie* list = load_dictionary_dict(h->context, g, err);

    if ((*err = grib_get_string_internal(h, e->key, mybuf, &sizebuf)) != GRIB_SUCCESS)
        return NULL;

    if (grib_trie_get(list, mybuf))
        result = 1;
    else
        result = 0;

    sprintf(buf, "%ld", result);
    *size = strlen(buf);
    return buf;
}

static void print_dict(grib_context* c, grib_expression* g, grib_handle* f)
{
    grib_expression_is_in_dict* e = (grib_expression_is_in_dict*)g;
    printf("access('%s", e->key);
    if (f) {
        long s = 0;
        grib_get_long(f, e->key, &s);
        printf("=%ld", s);
    }
    printf("')");
}

static int native_type_dict(grib_expression* g, grib_handle* h)
{
    return GRIB_TYPE_LONG;
}

static void add_dependency_dict(grib_expression* g, grib_accessor* observer)
{
    grib_expression_is_in_dict* e = (grib_expression_is_in_dict*)g;
    grib_accessor* observed       = grib_find_accessor(grib_handle_of_accessor(observer), e->key);

    if (!observed) {
        /* grib_context_log(observer->context, GRIB_LOG_ERROR, */
        /* "Error in accessor_add_dependency: cannot find [%s]", e->name); */
        /* Assert(observed); */
        return;
    }

    grib_dependency_add(observer, observed);
}



static grib_expression_class _grib_expression_class_is_in_list = {
    0,                    /* super                     */
    "is_in_list",                    /* name                      */
    sizeof(grib_expression_is_in_list),/* size of instance          */
    0,                           /* inited */
    &init_class_list,                 /* init_class */
    0,                     /* constructor               */
    &destroy_list,                  /* destructor                */
    &print_list,                 
    &add_dependency_list,       

	&native_type_list,
	&get_name_list,

	&evaluate_long_list,
	&evaluate_double_list,
	&evaluate_string_list,
};

grib_expression_class* grib_expression_class_is_in_list = &_grib_expression_class_is_in_list;

grib_expression* new_is_in_list_expression(grib_context* c, const char* name, const char* list)
{
    grib_expression_is_in_list* e = (grib_expression_is_in_list*)grib_context_malloc_clear_persistent(c, sizeof(grib_expression_is_in_list));
    e->base.cclass                = grib_expression_class_is_in_list;
    e->name                       = grib_context_strdup_persistent(c, name);
    e->list                       = grib_context_strdup_persistent(c, list);
    return (grib_expression*)e;
}

static void init_class_list(grib_expression_class* c)
{
}

static grib_trie* load_list_list(grib_context* c, grib_expression* e, int* err)
{
    grib_expression_is_in_list* self = (grib_expression_is_in_list*)e;

    char* filename  = NULL;
    char line[1024] = {0,};
    grib_trie* list = NULL;
    FILE* f         = NULL;

    *err = GRIB_SUCCESS;

    filename = grib_context_full_defs_path(c, self->list);
    if (!filename) {
        grib_context_log(c, GRIB_LOG_ERROR, "unable to find def file %s", self->list);
        *err = GRIB_FILE_NOT_FOUND;
        return NULL;
    }
    else {
        grib_context_log(c, GRIB_LOG_DEBUG, "is_in_list: found def file %s", filename);
    }
    list = (grib_trie*)grib_trie_get(c->lists, filename);
    if (list) {
        grib_context_log(c, GRIB_LOG_DEBUG, "using list %s from cache", self->list);
        return list;
    }
    else {
        grib_context_log(c, GRIB_LOG_DEBUG, "using list %s from file %s", self->list, filename);
    }

    f = codes_fopen(filename, "r");
    if (!f) {
        *err = GRIB_IO_PROBLEM;
        return NULL;
    }

    list = grib_trie_new(c);

    while (fgets(line, sizeof(line) - 1, f)) {
        unsigned char* p = (unsigned char*)line;
        while (*p != 0) {
            if (*p < 33) {
                *p = 0;
                break;
            }
            p++;
        }
        grib_trie_insert(list, line, line);
    }

    grib_trie_insert(c->lists, filename, list);

    fclose(f);

    return list;
}

static const char* get_name_list(grib_expression* g)
{
    grib_expression_is_in_list* e = (grib_expression_is_in_list*)g;
    return e->name;
}

static int evaluate_long_list(grib_expression* g, grib_handle* h, long* result)
{
    grib_expression_is_in_list* e = (grib_expression_is_in_list*)g;
    int err                       = 0;
    char mybuf[1024]              = {0,};
    size_t size = 1024;

    grib_trie* list = load_list_list(h->context, g, &err);

    if ((err = grib_get_string_internal(h, e->name, mybuf, &size)) != GRIB_SUCCESS)
        return err;

    if (grib_trie_get(list, mybuf))
        *result = 1;
    else
        *result = 0;

    return err;
}

static int evaluate_double_list(grib_expression* g, grib_handle* h, double* result)
{
    grib_expression_is_in_list* e = (grib_expression_is_in_list*)g;
    int err                       = 0;
    char mybuf[1024]              = {0,};
    size_t size = 1024;

    grib_trie* list = load_list_list(h->context, g, &err);

    if ((err = grib_get_string_internal(h, e->name, mybuf, &size)) != GRIB_SUCCESS)
        return err;

    if (grib_trie_get(list, mybuf))
        *result = 1;
    else
        *result = 0;

    return err;
}

static string evaluate_string_list(grib_expression* g, grib_handle* h, char* buf, size_t* size, int* err)
{
    grib_expression_is_in_list* e = (grib_expression_is_in_list*)g;
    char mybuf[1024]              = {0,};
    size_t sizebuf = 1024;
    long result;

    grib_trie* list = load_list_list(h->context, g, err);

    if ((*err = grib_get_string_internal(h, e->name, mybuf, &sizebuf)) != GRIB_SUCCESS)
        return NULL;

    if (grib_trie_get(list, mybuf))
        result = 1;
    else
        result = 0;

    sprintf(buf, "%ld", result);
    *size = strlen(buf);
    return buf;
}

static void print_list(grib_context* c, grib_expression* g, grib_handle* f)
{
    grib_expression_is_in_list* e = (grib_expression_is_in_list*)g;
    printf("access('%s", e->name);
    if (f) {
        long s = 0;
        grib_get_long(f, e->name, &s);
        printf("=%ld", s);
    }
    printf("')");
}

static void destroy_list(grib_context* c, grib_expression* g)
{
}

static void add_dependency_list(grib_expression* g, grib_accessor* observer)
{
    grib_expression_is_in_list* e = (grib_expression_is_in_list*)g;
    grib_accessor* observed       = grib_find_accessor(grib_handle_of_accessor(observer), e->name);

    if (!observed) {
        /* grib_context_log(observer->context, GRIB_LOG_ERROR, */
        /* "Error in accessor_add_dependency: cannot find [%s]", e->name); */
        /* Assert(observed); */
        return;
    }

    grib_dependency_add(observer, observed);
}

static int native_type_list(grib_expression* g, grib_handle* h)
{
    grib_expression_is_in_list* e = (grib_expression_is_in_list*)g;
    int type                      = 0;
    int err;
    if ((err = grib_get_native_type(h, e->name, &type)) != GRIB_SUCCESS)
        grib_context_log(h->context, GRIB_LOG_ERROR,
                         "Error in native_type %s : %s", e->name, grib_get_error_message(err));
    return type;
}


static grib_expression_class _grib_expression_class_length = {
    0,                    /* super                     */
    "length",                    /* name                      */
    sizeof(grib_expression_length),/* size of instance          */
    0,                           /* inited */
    &init_class_length,                 /* init_class */
    0,                     /* constructor               */
    &destroy_length,                  /* destructor                */
    &print_length,                 
    &add_dependency_length,       

	&native_type_length,
	&get_name_length,

	&evaluate_long_length,
	&evaluate_double_length,
	&evaluate_string_length,
};

grib_expression_class* grib_expression_class_length = &_grib_expression_class_length;

grib_expression* new_length_expression(grib_context* c, const char* name)
{
    grib_expression_length* e = (grib_expression_length*)grib_context_malloc_clear_persistent(c, sizeof(grib_expression_length));
    e->base.cclass            = grib_expression_class_length;
    e->name                   = grib_context_strdup_persistent(c, name);
    return (grib_expression*)e;
}

static void init_class_length(grib_expression_class* c)
{
}

static const char* get_name_length(grib_expression* g)
{
    grib_expression_length* e = (grib_expression_length*)g;
    return e->name;
}

static int evaluate_long_length(grib_expression* g, grib_handle* h, long* result)
{
    grib_expression_length* e = (grib_expression_length*)g;
    int err                   = 0;
    char mybuf[1024]          = {0,};
    size_t size = 1024;
    if ((err = grib_get_string_internal(h, e->name, mybuf, &size)) != GRIB_SUCCESS)
        return err;

    *result = strlen(mybuf);
    return err;
}

static int evaluate_double_length(grib_expression* g, grib_handle* h, double* result)
{
    grib_expression_length* e = (grib_expression_length*)g;
    char mybuf[1024]          = {0,};
    size_t size = 1024;
    int err     = 0;
    if ((err = grib_get_string_internal(h, e->name, mybuf, &size)) != GRIB_SUCCESS)
        return err;

    *result = strlen(mybuf);
    return err;
}

static string evaluate_string_length(grib_expression* g, grib_handle* h, char* buf, size_t* size, int* err)
{
    grib_expression_length* e = (grib_expression_length*)g;
    char mybuf[1024]          = {0,};
    Assert(buf);
    if ((*err = grib_get_string_internal(h, e->name, mybuf, size)) != GRIB_SUCCESS)
        return NULL;

    sprintf(buf, "%ld", (long)strlen(mybuf));
    return buf;
}

static void print_length(grib_context* c, grib_expression* g, grib_handle* f)
{
    grib_expression_length* e = (grib_expression_length*)g;
    printf("access('%s", e->name);
    if (f) {
        long s = 0;
        grib_get_long(f, e->name, &s);
        printf("=%ld", s);
    }
    printf("')");
}

static void destroy_length(grib_context* c, grib_expression* g)
{
    grib_expression_length* e = (grib_expression_length*)g;
    grib_context_free_persistent(c, e->name);
}

static void add_dependency_length(grib_expression* g, grib_accessor* observer)
{
    grib_expression_length* e = (grib_expression_length*)g;
    grib_accessor* observed   = grib_find_accessor(grib_handle_of_accessor(observer), e->name);

    if (!observed) {
        /* grib_context_log(observer->context, GRIB_LOG_ERROR, */
        /* "Error in accessor_add_dependency: cannot find [%s]", e->name); */
        /* Assert(observed); */
        return;
    }

    grib_dependency_add(observer, observed);
}

static int native_type_length(grib_expression* g, grib_handle* h)
{
    return GRIB_TYPE_LONG;
}

