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
void rotate(const double inlat, const double inlon,
            const double angleOfRot, const double southPoleLat, const double southPoleLon,
            double* outlat, double* outlon)
{
    double PYROT, PXROT, ZCYROT, ZCXROT, ZSXROT;
    const double ZSYCEN = sin(DEG2RAD * (southPoleLat + 90.));
    const double ZCYCEN = cos(DEG2RAD * (southPoleLat + 90.));
    const double ZXMXC  = DEG2RAD * (inlon - southPoleLon);
    const double ZSXMXC = sin(ZXMXC);
    const double ZCXMXC = cos(ZXMXC);
    const double ZSYREG = sin(DEG2RAD * inlat);
    const double ZCYREG = cos(DEG2RAD * inlat);
    double ZSYROT       = ZCYCEN * ZSYREG - ZSYCEN * ZCYREG * ZCXMXC;

    ZSYROT = MAX(MIN(ZSYROT, +1.0), -1.0);

    PYROT = asin(ZSYROT) * RAD2DEG;

    ZCYROT = cos(PYROT * DEG2RAD);
    ZCXROT = (ZCYCEN * ZCYREG * ZCXMXC + ZSYCEN * ZSYREG) / ZCYROT;
    ZCXROT = MAX(MIN(ZCXROT, +1.0), -1.0);
    ZSXROT = ZCYREG * ZSXMXC / ZCYROT;

    PXROT = acos(ZCXROT) * RAD2DEG;

    if (ZSXROT < 0.0)
        PXROT = -PXROT;

    *outlat = PYROT;
    *outlon = PXROT;
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

int grib_nearest_get_radius(grib_handle* h, double* radiusInKm)
{
    int err = 0;
    long lRadiusInMetres;
    double result = 0;
    const char* s_radius = "radius";
    const char* s_minor = "earthMinorAxisInMetres";
    const char* s_major = "earthMajorAxisInMetres";

    if ((err = grib_get_long(h, s_radius, &lRadiusInMetres)) == GRIB_SUCCESS) {
        if (grib_is_missing(h, s_radius, &err) || lRadiusInMetres == GRIB_MISSING_LONG) {
            grib_context_log(h->context, GRIB_LOG_DEBUG, "Key 'radius' is missing");
            return GRIB_GEOCALCULUS_PROBLEM;
        }
        result = ((double)lRadiusInMetres) / 1000.0;
    }
    else {
        double minor = 0, major = 0;
        if ((err = grib_get_double_internal(h, s_minor, &minor)) != GRIB_SUCCESS) return err;
        if ((err = grib_get_double_internal(h, s_major, &major)) != GRIB_SUCCESS) return err;
        if (grib_is_missing(h, s_minor, &err)) return GRIB_GEOCALCULUS_PROBLEM;
        if (grib_is_missing(h, s_major, &err)) return GRIB_GEOCALCULUS_PROBLEM;
        result = (major + minor) / 2.0;
        result = result / 1000.0;
    }
    *radiusInKm = result;
    return GRIB_SUCCESS;
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

int _grib_get_long_array_internal(const grib_handle* h, grib_accessor* a, long* val, size_t buffer_len, size_t* decoded_length)
{
    if (a) {
        int err = _grib_get_long_array_internal(h, a->same, val, buffer_len, decoded_length);

        if (err == GRIB_SUCCESS) {
            size_t len = buffer_len - *decoded_length;
            err        = grib_unpack_long(a, val + *decoded_length, &len);
            *decoded_length += len;
        }

        return err;
    }
    else {
        return GRIB_SUCCESS;
    }
}

int grib_get_long_array_internal(grib_handle* h, const char* name, long* val, size_t* length)
{
    int ret = grib_get_long_array(h, name, val, length);

    if (ret != GRIB_SUCCESS)
        grib_context_log(h->context, GRIB_LOG_ERROR,
                         "unable to get %s as long array (%s)",
                         name, grib_get_error_message(ret));

    return ret;
}

int grib_get_long_array(const grib_handle* h, const char* name, long* val, size_t* length)
{
    size_t len              = *length;
    grib_accessor* a        = NULL;
    grib_accessors_list* al = NULL;
    int ret                 = 0;

    if (name[0] == '/') {
        al = grib_find_accessors_list(h, name);
        if (!al)
            return GRIB_NOT_FOUND;
        ret = grib_accessors_list_unpack_long(al, val, length);
        grib_context_free(h->context, al);
    }
    else {
        a = grib_find_accessor(h, name);
        if (!a)
            return GRIB_NOT_FOUND;
        if (name[0] == '#') {
            return grib_unpack_long(a, val, length);
        }
        else {
            *length = 0;
            return _grib_get_long_array_internal(h, a, val, len, length);
        }
    }
    return ret;
}

int grib_get_double_element_internal(grib_handle* h, const char* name, int i, double* val)
{
    int ret = grib_get_double_element(h, name, i, val);

    if (ret != GRIB_SUCCESS)
        grib_context_log(h->context, GRIB_LOG_ERROR,
                         "unable to get %s as double element (%s)",
                         name, grib_get_error_message(ret));

    return ret;
}
int is_gaussian_global(
    double lat1, double lat2, double lon1, double lon2, /* bounding box*/
    long num_points_equator,                            /* num points on latitude at equator */
    const double* latitudes,                            /* array of Gaussian latitudes (size 2*N) */
    double angular_precision                            /* tolerance for angle comparison */
)
{
    int global         = 1;
    const double d     = fabs(latitudes[0] - latitudes[1]);
    const double delta = 360.0 / num_points_equator;
    /* Compute the expected last longitude for a global field */
    const double lon2_global = 360.0 - delta;
    /* Compute difference between expected longitude and actual one */
    const double lon2_diff = fabs(lon2 - lon2_global) - delta;

    /*
    {
        grib_context* c=grib_context_get_default();
        if (c->debug) {
            fprintf(stderr,"ECCODES DEBUG is_gaussian_global: lat1=%f, lat2=%f, glat0=%f, d=%f\n", lat1, lat2, latitudes[0], d);
            fprintf(stderr,"ECCODES DEBUG is_gaussian_global: lon1=%f, lon2=%f, glon2=%f, delta=%f\n", lon1, lon2, lon2_global, delta);
        }
    }
    */

    /* Note: final gaussian latitude = -first latitude */
    if ((fabs(lat1 - latitudes[0]) >= d) ||
        (fabs(lat2 + latitudes[0]) >= d) ||
        lon1 != 0 ||
        lon2_diff > angular_precision) {
        global = 0; /* sub area */
    }
    return global;
}
int grib_get_gaussian_latitudes(long trunc, double* lats)
{
    if (trunc == 1280)
        return get_precomputed_latitudes_N1280(lats);
    if (trunc == 640)
        return get_precomputed_latitudes_N640(lats);
    else
        return _grib_get_gaussian_latitudes(trunc, lats);
}
void grib_get_reduced_row_p(long pl, double lon_first, double lon_last, long* npoints, double* olon_first, double* olon_last)
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
    *olon_first = the_lon1;
    *olon_last  = the_lon2;
}
size_t sum_of_pl_array(const long* pl, size_t plsize)
{
    long i, count = 0;
    for (i = 0; i < plsize; i++) {
        count += pl[i];
    }
    return count;
}
int grib_get_double_element(const grib_handle* h, const char* name, int i, double* val)
{
    grib_accessor* act = grib_find_accessor(h, name);

    if (act) {
        return grib_unpack_double_element(act, i, val);
    }
    return GRIB_NOT_FOUND;
}
int grib_iterator_reset(grib_iterator* i)
{
    grib_iterator_class* c = i->cclass;
    while (c) {
        grib_iterator_class* s = c->super ? *(c->super) : NULL;
        if (c->reset)
            return c->reset(i);
        c = s;
    }
    Assert(0);
    return 0;
}
int grib_unpack_double_element(grib_accessor* a, size_t i, double* v)
{
    grib_accessor_class* c = a->cclass;
    while (c) {
        if (c->unpack_double_element) {
            return c->unpack_double_element(a, i, v);
        }
        c = c->super ? *(c->super) : NULL;
    }
    return GRIB_NOT_IMPLEMENTED;
}
const Fraction_value_type MAX_DENOM = 3037000499; /* sqrt(LLONG_MAX) */
static Fraction_type fraction_construct_from_double(double x)
{
    Fraction_type result;
    double value             = x;
    Fraction_value_type sign = 1;
    Fraction_value_type m00 = 1, m11 = 1, m01 = 0, m10 = 0;
    Fraction_value_type a = x;
    Fraction_value_type t2, top, bottom, g;
    size_t cnt = 0;

    /*Assert(x != NAN);*/
    Assert(fabs(x) < 1e30);

    if (x < 0) {
        sign = -sign;
        x    = -x;
    }

    t2 = m10 * a + m11;

    while (t2 <= MAX_DENOM) {
        Fraction_value_type t1 = m00 * a + m01;
        m01                    = m00;
        m00                    = t1;

        m11 = m10;
        m10 = t2;

        if (x == a) {
            break;
        }

        x = 1.0 / (x - a);

        if (x > LLONG_MAX) {
            break;
        }

        a  = x;
        t2 = m10 * a + m11;

        if (cnt++ > 10000) {
            fprintf(stderr, "Cannot compute fraction from %g\n", value);
        }
    }

    while (m10 >= MAX_DENOM || m00 >= MAX_DENOM) {
        m00 >>= 1;
        m10 >>= 1;
    }

    top    = m00;
    bottom = m10;

    g      = fraction_gcd(top, bottom);
    top    = top / g;
    bottom = bottom / g;

    result.top_    = sign * top;
    result.bottom_ = bottom;
    return result;
}
static int _grib_get_gaussian_latitudes(long trunc, double* lats)
{
    long jlat, iter, legi;
    double rad2deg, convval, root, legfonc = 0;
    double mem1, mem2, conv;
    double denom     = 0.0;
    double precision = 1.0E-14;
    const long nlat  = trunc * 2;

    rad2deg = 180.0 / M_PI;

    convval = (1.0 - ((2.0 / M_PI) * (2.0 / M_PI)) * 0.25);

    gauss_first_guess(trunc, lats);
    denom = sqrt(((((double)nlat) + 0.5) * (((double)nlat) + 0.5)) + convval);

    for (jlat = 0; jlat < trunc; jlat++) {
        /*   First approximation for root      */
        root = cos(lats[jlat] / denom);

        /*   Perform loop of Newton iterations  */
        iter = 0;
        conv = 1;

        while (fabs(conv) >= precision) {
            mem2 = 1.0;
            mem1 = root;

            /*  Compute Legendre polynomial  */
            for (legi = 0; legi < nlat; legi++) {
                legfonc = ((2.0 * (legi + 1) - 1.0) * root * mem1 - legi * mem2) / ((double)(legi + 1));
                mem2    = mem1;
                mem1    = legfonc;
            }

            /*  Perform Newton iteration  */
            conv = legfonc / ((((double)nlat) * (mem2 - root * legfonc)) / (1.0 - (root * root)));
            root -= conv;

            /*  Routine fails if no convergence after MAXITER iterations  */
            if (iter++ > MAXITER) {
                return GRIB_GEOCALCULUS_PROBLEM;
            }
        }

        /*   Set North and South values using symmetry */
        lats[jlat]            = asin(root) * rad2deg;
        lats[nlat - 1 - jlat] = -lats[jlat];
    }

    if (nlat != (trunc * 2))
        lats[trunc + 1] = 0.0;
    return GRIB_SUCCESS;
}
static int get_precomputed_latitudes_N1280(double* lats)
{
    lats[0]=89.946187715665616;
    lats[1]=89.876478353332288;
    lats[2]=89.806357319542244;
    lats[3]=89.736143271609578;
    lats[4]=89.6658939412157;
    lats[5]=89.595627537554492;
    lats[6]=89.525351592371393;
    lats[7]=89.45506977912261;
    lats[8]=89.3847841013921;
    lats[9]=89.314495744374256;
    lats[10]=89.24420545380525;
    lats[11]=89.173913722284126;
    lats[12]=89.103620888238879;
    lats[13]=89.033327191845927;
    lats[14]=88.96303280826325;
    lats[15]=88.892737868230952;
    lats[16]=88.822442471310097;
    lats[17]=88.752146694650691;
    lats[18]=88.681850598961759;
    lats[19]=88.611554232668382;
    lats[20]=88.541257634868515;
    lats[21]=88.470960837474877;
    lats[22]=88.40066386679355;
    lats[23]=88.330366744702559;
    lats[24]=88.26006948954614;
    lats[25]=88.189772116820762;
    lats[26]=88.119474639706425;
    lats[27]=88.049177069484486;
    lats[28]=87.978879415867283;
    lats[29]=87.908581687261687;
    lats[30]=87.838283890981543;
    lats[31]=87.767986033419561;
    lats[32]=87.697688120188062;
    lats[33]=87.627390156234085;
    lats[34]=87.557092145935584;
    lats[35]=87.486794093180748;
    lats[36]=87.416496001434894;
    lats[37]=87.346197873795816;
    lats[38]=87.275899713041966;
    lats[39]=87.205601521672108;
    lats[40]=87.135303301939786;
    lats[41]=87.065005055882821;
    lats[42]=86.994706785348129;
    lats[43]=86.924408492014166;
    lats[44]=86.854110177408927;
    lats[45]=86.783811842927179;
    lats[46]=86.713513489844246;
    lats[47]=86.643215119328573;
    lats[48]=86.572916732453024;
    lats[49]=86.502618330203831;
    lats[50]=86.432319913489792;
    lats[51]=86.362021483149363;
    lats[52]=86.291723039957418;
    lats[53]=86.221424584631109;
    lats[54]=86.151126117835304;
    lats[55]=86.080827640187209;
    lats[56]=86.010529152260403;
    lats[57]=85.940230654588888;
    lats[58]=85.869932147670127;
    lats[59]=85.799633631968391;
    lats[60]=85.729335107917464;
    lats[61]=85.659036575922883;
    lats[62]=85.588738036364362;
    lats[63]=85.518439489597966;
    lats[64]=85.448140935957483;
    lats[65]=85.377842375756586;
    lats[66]=85.307543809290152;
    lats[67]=85.237245236835548;
    lats[68]=85.16694665865414;
    lats[69]=85.09664807499216;
    lats[70]=85.026349486081983;
    lats[71]=84.95605089214304;
    lats[72]=84.885752293382765;
    lats[73]=84.81545368999717;
    lats[74]=84.745155082171991;
    lats[75]=84.674856470082915;
    lats[76]=84.604557853896708;
    lats[77]=84.534259233771479;
    lats[78]=84.463960609857125;
    lats[79]=84.393661982296322;
    lats[80]=84.323363351224444;
    lats[81]=84.253064716770425;
    lats[82]=84.18276607905679;
    lats[83]=84.112467438200326;
    lats[84]=84.042168794312317;
    lats[85]=83.971870147498763;
    lats[86]=83.901571497860914;
    lats[87]=83.831272845495249;
    lats[88]=83.760974190494011;
    lats[89]=83.690675532945292;
    lats[90]=83.620376872933264;
    lats[91]=83.550078210538487;
    lats[92]=83.479779545838113;
    lats[93]=83.409480878905782;
    lats[94]=83.339182209812321;
    lats[95]=83.268883538625232;
    lats[96]=83.198584865409657;
    lats[97]=83.128286190227698;
    lats[98]=83.057987513139125;
    lats[99]=82.987688834201322;
    lats[100]=82.917390153469313;
    lats[101]=82.84709147099602;
    lats[102]=82.77679278683226;
    lats[103]=82.706494101026948;
    lats[104]=82.63619541362705;
    lats[105]=82.56589672467787;
    lats[106]=82.495598034222837;
    lats[107]=82.425299342304029;
    lats[108]=82.355000648961692;
    lats[109]=82.284701954234833;
    lats[110]=82.214403258160871;
    lats[111]=82.144104560776;
    lats[112]=82.073805862115165;
    lats[113]=82.003507162211946;
    lats[114]=81.933208461098829;
    lats[115]=81.862909758807191;
    lats[116]=81.792611055367345;
    lats[117]=81.722312350808508;
    lats[118]=81.652013645158945;
    lats[119]=81.581714938445955;
    lats[120]=81.511416230696042;
    lats[121]=81.441117521934686;
    lats[122]=81.370818812186627;
    lats[123]=81.300520101475826;
    lats[124]=81.230221389825374;
    lats[125]=81.159922677257711;
    lats[126]=81.089623963794551;
    lats[127]=81.019325249456955;
    lats[128]=80.949026534265244;
    lats[129]=80.878727818239184;
    lats[130]=80.808429101397948;
    lats[131]=80.73813038376008;
    lats[132]=80.667831665343556;
    lats[133]=80.59753294616587;
    lats[134]=80.527234226243991;
    lats[135]=80.456935505594302;
    lats[136]=80.386636784232863;
    lats[137]=80.316338062175078;
    lats[138]=80.246039339436052;
    lats[139]=80.175740616030438;
    lats[140]=80.105441891972376;
    lats[141]=80.035143167275749;
    lats[142]=79.9648444419539;
    lats[143]=79.894545716019948;
    lats[144]=79.824246989486554;
    lats[145]=79.753948262366038;
    lats[146]=79.683649534670437;
    lats[147]=79.61335080641139;
    lats[148]=79.543052077600308;
    lats[149]=79.472753348248219;
    lats[150]=79.402454618365894;
    lats[151]=79.332155887963822;
    lats[152]=79.261857157052191;
    lats[153]=79.191558425640977;
    lats[154]=79.121259693739859;
    lats[155]=79.050960961358285;
    lats[156]=78.980662228505423;
    lats[157]=78.910363495190211;
    lats[158]=78.840064761421445;
    lats[159]=78.769766027207638;
    lats[160]=78.699467292557102;
    lats[161]=78.629168557477882;
    lats[162]=78.558869821977908;
    lats[163]=78.488571086064923;
    lats[164]=78.418272349746417;
    lats[165]=78.347973613029708;
    lats[166]=78.277674875922045;
    lats[167]=78.207376138430348;
    lats[168]=78.137077400561424;
    lats[169]=78.066778662322022;
    lats[170]=77.996479923718596;
    lats[171]=77.926181184757539;
    lats[172]=77.855882445445019;
    lats[173]=77.785583705787161;
    lats[174]=77.71528496578982;
    lats[175]=77.644986225458879;
    lats[176]=77.574687484799924;
    lats[177]=77.504388743818524;
    lats[178]=77.434090002520122;
    lats[179]=77.363791260909963;
    lats[180]=77.293492518993247;
    lats[181]=77.22319377677502;
    lats[182]=77.15289503426024;
    lats[183]=77.082596291453768;
    lats[184]=77.012297548360323;
    lats[185]=76.941998804984564;
    lats[186]=76.871700061330955;
    lats[187]=76.801401317404;
    lats[188]=76.731102573208048;
    lats[189]=76.660803828747362;
    lats[190]=76.59050508402602;
    lats[191]=76.520206339048215;
    lats[192]=76.449907593817869;
    lats[193]=76.379608848338933;
    lats[194]=76.3093101026152;
    lats[195]=76.239011356650423;
    lats[196]=76.16871261044831;
    lats[197]=76.098413864012443;
    lats[198]=76.028115117346374;
    lats[199]=75.957816370453543;
    lats[200]=75.887517623337317;
    lats[201]=75.81721887600105;
    lats[202]=75.746920128447996;
    lats[203]=75.67662138068134;
    lats[204]=75.60632263270422;
    lats[205]=75.536023884519707;
    lats[206]=75.465725136130786;
    lats[207]=75.395426387540439;
    lats[208]=75.325127638751567;
    lats[209]=75.254828889766983;
    lats[210]=75.184530140589501;
    lats[211]=75.114231391221821;
    lats[212]=75.043932641666672;
    lats[213]=74.973633891926625;
    lats[214]=74.903335142004323;
    lats[215]=74.833036391902269;
    lats[216]=74.762737641622991;
    lats[217]=74.692438891168877;
    lats[218]=74.622140140542356;
    lats[219]=74.551841389745761;
    lats[220]=74.481542638781434;
    lats[221]=74.411243887651622;
    lats[222]=74.340945136358584;
    lats[223]=74.270646384904481;
    lats[224]=74.200347633291472;
    lats[225]=74.13004888152166;
    lats[226]=74.059750129597163;
    lats[227]=73.98945137751997;
    lats[228]=73.919152625292114;
    lats[229]=73.848853872915541;
    lats[230]=73.778555120392184;
    lats[231]=73.70825636772399;
    lats[232]=73.637957614912779;
    lats[233]=73.567658861960396;
    lats[234]=73.497360108868662;
    lats[235]=73.427061355639339;
    lats[236]=73.356762602274188;
    lats[237]=73.2864638487749;
    lats[238]=73.216165095143182;
    lats[239]=73.145866341380668;
    lats[240]=73.075567587489019;
    lats[241]=73.005268833469799;
    lats[242]=72.934970079324657;
    lats[243]=72.864671325055056;
    lats[244]=72.794372570662574;
    lats[245]=72.724073816148703;
    lats[246]=72.653775061514935;
    lats[247]=72.583476306762691;
    lats[248]=72.513177551893421;
    lats[249]=72.442878796908545;
    lats[250]=72.3725800418094;
    lats[251]=72.302281286597392;
    lats[252]=72.231982531273843;
    lats[253]=72.161683775840089;
    lats[254]=72.091385020297409;
    lats[255]=72.02108626464711;
    lats[256]=71.950787508890414;
    lats[257]=71.880488753028587;
    lats[258]=71.810189997062835;
    lats[259]=71.739891240994368;
    lats[260]=71.669592484824364;
    lats[261]=71.599293728553988;
    lats[262]=71.528994972184378;
    lats[263]=71.458696215716685;
    lats[264]=71.388397459152031;
    lats[265]=71.318098702491469;
    lats[266]=71.247799945736105;
    lats[267]=71.177501188887007;
    lats[268]=71.107202431945211;
    lats[269]=71.036903674911756;
    lats[270]=70.966604917787635;
    lats[271]=70.896306160573886;
    lats[272]=70.826007403271475;
    lats[273]=70.755708645881384;
    lats[274]=70.685409888404578;
    lats[275]=70.615111130841967;
    lats[276]=70.544812373194532;
    lats[277]=70.474513615463138;
    lats[278]=70.404214857648739;
    lats[279]=70.333916099752187;
    lats[280]=70.263617341774406;
    lats[281]=70.193318583716191;
    lats[282]=70.123019825578467;
    lats[283]=70.052721067362043;
    lats[284]=69.982422309067744;
    lats[285]=69.912123550696421;
    lats[286]=69.841824792248843;
    lats[287]=69.771526033725834;
    lats[288]=69.701227275128161;
    lats[289]=69.630928516456592;
    lats[290]=69.560629757711908;
    lats[291]=69.490330998894862;
    lats[292]=69.420032240006194;
    lats[293]=69.349733481046613;
    lats[294]=69.279434722016902;
    lats[295]=69.209135962917699;
    lats[296]=69.138837203749759;
    lats[297]=69.068538444513763;
    lats[298]=68.998239685210365;
    lats[299]=68.927940925840304;
    lats[300]=68.85764216640419;
    lats[301]=68.787343406902693;
    lats[302]=68.717044647336493;
    lats[303]=68.646745887706189;
    lats[304]=68.576447128012447;
    lats[305]=68.506148368255865;
    lats[306]=68.435849608437067;
    lats[307]=68.365550848556666;
    lats[308]=68.295252088615257;
    lats[309]=68.224953328613438;
    lats[310]=68.154654568551791;
    lats[311]=68.084355808430871;
    lats[312]=68.014057048251274;
    lats[313]=67.943758288013555;
    lats[314]=67.873459527718282;
    lats[315]=67.803160767365966;
    lats[316]=67.732862006957205;
    lats[317]=67.662563246492482;
    lats[318]=67.592264485972336;
    lats[319]=67.521965725397308;
    lats[320]=67.451666964767895;
    lats[321]=67.381368204084609;
    lats[322]=67.311069443347961;
    lats[323]=67.240770682558434;
    lats[324]=67.170471921716526;
    lats[325]=67.100173160822706;
    lats[326]=67.029874399877471;
    lats[327]=66.95957563888129;
    lats[328]=66.889276877834618;
    lats[329]=66.818978116737924;
    lats[330]=66.748679355591662;
    lats[331]=66.678380594396273;
    lats[332]=66.608081833152212;
    lats[333]=66.537783071859891;
    lats[334]=66.467484310519808;
    lats[335]=66.397185549132331;
    lats[336]=66.326886787697887;
    lats[337]=66.256588026216932;
    lats[338]=66.186289264689833;
    lats[339]=66.115990503117033;
    lats[340]=66.045691741498899;
    lats[341]=65.975392979835888;
    lats[342]=65.905094218128355;
    lats[343]=65.834795456376696;
    lats[344]=65.764496694581283;
    lats[345]=65.694197932742526;
    lats[346]=65.623899170860767;
    lats[347]=65.553600408936404;
    lats[348]=65.483301646969792;
    lats[349]=65.413002884961315;
    lats[350]=65.342704122911286;
    lats[351]=65.272405360820116;
    lats[352]=65.202106598688133;
    lats[353]=65.131807836515677;
    lats[354]=65.061509074303089;
    lats[355]=64.991210312050711;
    lats[356]=64.920911549758912;
    lats[357]=64.850612787427963;
    lats[358]=64.780314025058246;
    lats[359]=64.710015262650074;
    lats[360]=64.639716500203733;
    lats[361]=64.569417737719576;
    lats[362]=64.499118975197902;
    lats[363]=64.428820212639039;
    lats[364]=64.358521450043284;
    lats[365]=64.288222687410922;
    lats[366]=64.21792392474228;
    lats[367]=64.147625162037642;
    lats[368]=64.07732639929732;
    lats[369]=64.00702763652157;
    lats[370]=63.93672887371072;
    lats[371]=63.866430110865004;
    lats[372]=63.796131347984762;
    lats[373]=63.725832585070251;
    lats[374]=63.655533822121711;
    lats[375]=63.585235059139464;
    lats[376]=63.514936296123757;
    lats[377]=63.444637533074854;
    lats[378]=63.374338769993031;
    lats[379]=63.304040006878537;
    lats[380]=63.23374124373165;
    lats[381]=63.163442480552604;
    lats[382]=63.093143717341647;
    lats[383]=63.022844954099064;
    lats[384]=62.952546190825068;
    lats[385]=62.882247427519928;
    lats[386]=62.811948664183866;
    lats[387]=62.741649900817137;
    lats[388]=62.67135113741999;
    lats[389]=62.60105237399263;
    lats[390]=62.530753610535321;
    lats[391]=62.460454847048261;
    lats[392]=62.3901560835317;
    lats[393]=62.319857319985871;
    lats[394]=62.249558556410982;
    lats[395]=62.179259792807258;
    lats[396]=62.108961029174914;
    lats[397]=62.038662265514176;
    lats[398]=61.968363501825259;
    lats[399]=61.898064738108381;
    lats[400]=61.827765974363729;
    lats[401]=61.757467210591535;
    lats[402]=61.687168446791986;
    lats[403]=61.616869682965287;
    lats[404]=61.546570919111666;
    lats[405]=61.476272155231321;
    lats[406]=61.405973391324409;
    lats[407]=61.335674627391185;
    lats[408]=61.265375863431785;
    lats[409]=61.195077099446451;
    lats[410]=61.124778335435344;
    lats[411]=61.054479571398652;
    lats[412]=60.984180807336578;
    lats[413]=60.913882043249295;
    lats[414]=60.843583279137007;
    lats[415]=60.773284514999872;
    lats[416]=60.702985750838074;
    lats[417]=60.632686986651805;
    lats[418]=60.562388222441243;
    lats[419]=60.492089458206543;
    lats[420]=60.421790693947884;
    lats[421]=60.35149192966545;
    lats[422]=60.28119316535939;
    lats[423]=60.21089440102989;
    lats[424]=60.140595636677112;
    lats[425]=60.070296872301235;
    lats[426]=59.999998107902378;
    lats[427]=59.929699343480763;
    lats[428]=59.859400579036503;
    lats[429]=59.78910181456979;
    lats[430]=59.718803050080759;
    lats[431]=59.64850428556958;
    lats[432]=59.578205521036402;
    lats[433]=59.507906756481383;
    lats[434]=59.43760799190467;
    lats[435]=59.3673092273064;
    lats[436]=59.29701046268675;
    lats[437]=59.226711698045854;
    lats[438]=59.156412933383855;
    lats[439]=59.086114168700909;
    lats[440]=59.015815403997145;
    lats[441]=58.945516639272725;
    lats[442]=58.875217874527763;
    lats[443]=58.804919109762423;
    lats[444]=58.73462034497684;
    lats[445]=58.664321580171141;
    lats[446]=58.594022815345468;
    lats[447]=58.523724050499972;
    lats[448]=58.453425285634758;
    lats[449]=58.383126520749968;
    lats[450]=58.312827755845746;
    lats[451]=58.242528990922203;
    lats[452]=58.172230225979497;
    lats[453]=58.101931461017728;
    lats[454]=58.031632696037022;
    lats[455]=57.961333931037537;
    lats[456]=57.891035166019364;
    lats[457]=57.820736400982646;
    lats[458]=57.75043763592749;
    lats[459]=57.680138870854037;
    lats[460]=57.60984010576238;
    lats[461]=57.539541340652676;
    lats[462]=57.469242575525016;
    lats[463]=57.398943810379521;
    lats[464]=57.328645045216312;
    lats[465]=57.258346280035504;
    lats[466]=57.188047514837208;
    lats[467]=57.117748749621541;
    lats[468]=57.047449984388614;
    lats[469]=56.977151219138541;
    lats[470]=56.90685245387143;
    lats[471]=56.836553688587379;
    lats[472]=56.766254923286517;
    lats[473]=56.695956157968951;
    lats[474]=56.625657392634771;
    lats[475]=56.555358627284086;
    lats[476]=56.485059861917016;
    lats[477]=56.41476109653366;
    lats[478]=56.34446233113411;
    lats[479]=56.274163565718467;
    lats[480]=56.203864800286865;
    lats[481]=56.133566034839362;
    lats[482]=56.063267269376091;
    lats[483]=55.992968503897131;
    lats[484]=55.922669738402583;
    lats[485]=55.852370972892551;
    lats[486]=55.782072207367136;
    lats[487]=55.711773441826416;
    lats[488]=55.641474676270505;
    lats[489]=55.571175910699488;
    lats[490]=55.500877145113449;
    lats[491]=55.430578379512511;
    lats[492]=55.360279613896743;
    lats[493]=55.289980848266232;
    lats[494]=55.219682082621084;
    lats[495]=55.149383316961377;
    lats[496]=55.07908455128721;
    lats[497]=55.008785785598668;
    lats[498]=54.938487019895831;
    lats[499]=54.868188254178797;
    lats[500]=54.797889488447652;
    lats[501]=54.727590722702473;
    lats[502]=54.657291956943347;
    lats[503]=54.586993191170357;
    lats[504]=54.516694425383605;
    lats[505]=54.446395659583146;
    lats[506]=54.376096893769081;
    lats[507]=54.305798127941479;
    lats[508]=54.235499362100448;
    lats[509]=54.165200596246031;
    lats[510]=54.094901830378333;
    lats[511]=54.024603064497434;
    lats[512]=53.954304298603383;
    lats[513]=53.884005532696307;
    lats[514]=53.813706766776235;
    lats[515]=53.743408000843282;
    lats[516]=53.673109234897495;
    lats[517]=53.602810468938962;
    lats[518]=53.53251170296776;
    lats[519]=53.462212936983953;
    lats[520]=53.391914170987633;
    lats[521]=53.321615404978871;
    lats[522]=53.251316638957725;
    lats[523]=53.181017872924265;
    lats[524]=53.110719106878584;
    lats[525]=53.040420340820731;
    lats[526]=52.970121574750792;
    lats[527]=52.899822808668837;
    lats[528]=52.829524042574917;
    lats[529]=52.759225276469131;
    lats[530]=52.688926510351514;
    lats[531]=52.618627744222159;
    lats[532]=52.548328978081123;
    lats[533]=52.478030211928477;
    lats[534]=52.407731445764284;
    lats[535]=52.337432679588609;
    lats[536]=52.26713391340153;
    lats[537]=52.196835147203096;
    lats[538]=52.126536380993372;
    lats[539]=52.056237614772435;
    lats[540]=51.985938848540336;
    lats[541]=51.915640082297152;
    lats[542]=51.845341316042933;
    lats[543]=51.775042549777737;
    lats[544]=51.704743783501634;
    lats[545]=51.634445017214695;
    lats[546]=51.56414625091697;
    lats[547]=51.493847484608516;
    lats[548]=51.423548718289396;
    lats[549]=51.353249951959683;
    lats[550]=51.282951185619417;
    lats[551]=51.21265241926865;
    lats[552]=51.14235365290746;
    lats[553]=51.072054886535909;
    lats[554]=51.001756120154049;
    lats[555]=50.931457353761914;
    lats[556]=50.86115858735959;
    lats[557]=50.790859820947119;
    lats[558]=50.720561054524559;
    lats[559]=50.650262288091959;
    lats[560]=50.579963521649397;
    lats[561]=50.509664755196901;
    lats[562]=50.439365988734544;
    lats[563]=50.369067222262359;
    lats[564]=50.298768455780426;
    lats[565]=50.228469689288779;
    lats[566]=50.158170922787484;
    lats[567]=50.087872156276575;
    lats[568]=50.017573389756123;
    lats[569]=49.947274623226157;
    lats[570]=49.876975856686762;
    lats[571]=49.80667709013796;
    lats[572]=49.736378323579807;
    lats[573]=49.66607955701236;
    lats[574]=49.595780790435676;
    lats[575]=49.525482023849783;
    lats[576]=49.455183257254745;
    lats[577]=49.384884490650613;
    lats[578]=49.314585724037435;
    lats[579]=49.244286957415234;
    lats[580]=49.173988190784094;
    lats[581]=49.103689424144044;
    lats[582]=49.03339065749514;
    lats[583]=48.963091890837418;
    lats[584]=48.892793124170929;
    lats[585]=48.822494357495721;
    lats[586]=48.752195590811837;
    lats[587]=48.681896824119335;
    lats[588]=48.611598057418242;
    lats[589]=48.541299290708608;
    lats[590]=48.47100052399049;
    lats[591]=48.400701757263917;
    lats[592]=48.330402990528938;
    lats[593]=48.260104223785596;
    lats[594]=48.189805457033941;
    lats[595]=48.119506690274015;
    lats[596]=48.049207923505868;
    lats[597]=47.978909156729507;
    lats[598]=47.908610389945018;
    lats[599]=47.838311623152421;
    lats[600]=47.76801285635176;
    lats[601]=47.697714089543084;
    lats[602]=47.627415322726435;
    lats[603]=47.557116555901828;
    lats[604]=47.486817789069342;
    lats[605]=47.416519022228997;
    lats[606]=47.346220255380835;
    lats[607]=47.275921488524894;
    lats[608]=47.205622721661214;
    lats[609]=47.13532395478984;
    lats[610]=47.065025187910805;
    lats[611]=46.994726421024154;
    lats[612]=46.924427654129929;
    lats[613]=46.85412888722815;
    lats[614]=46.783830120318882;
    lats[615]=46.713531353402139;
    lats[616]=46.643232586477971;
    lats[617]=46.572933819546414;
    lats[618]=46.502635052607502;
    lats[619]=46.432336285661272;
    lats[620]=46.362037518707766;
    lats[621]=46.291738751747012;
    lats[622]=46.221439984779053;
    lats[623]=46.151141217803925;
    lats[624]=46.080842450821663;
    lats[625]=46.01054368383231;
    lats[626]=45.94024491683588;
    lats[627]=45.869946149832437;
    lats[628]=45.799647382821995;
    lats[629]=45.729348615804589;
    lats[630]=45.659049848780263;
    lats[631]=45.588751081749038;
    lats[632]=45.51845231471097;
    lats[633]=45.448153547666081;
    lats[634]=45.377854780614399;
    lats[635]=45.30755601355596;
    lats[636]=45.237257246490813;
    lats[637]=45.166958479418959;
    lats[638]=45.096659712340461;
    lats[639]=45.026360945255341;
    lats[640]=44.956062178163634;
    lats[641]=44.885763411065362;
    lats[642]=44.81546464396056;
    lats[643]=44.745165876849271;
    lats[644]=44.674867109731515;
    lats[645]=44.604568342607337;
    lats[646]=44.534269575476756;
    lats[647]=44.463970808339802;
    lats[648]=44.39367204119651;
    lats[649]=44.323373274046915;
    lats[650]=44.253074506891046;
    lats[651]=44.182775739728925;
    lats[652]=44.112476972560586;
    lats[653]=44.042178205386072;
    lats[654]=43.971879438205391;
    lats[655]=43.9015806710186;
    lats[656]=43.831281903825705;
    lats[657]=43.760983136626741;
    lats[658]=43.690684369421732;
    lats[659]=43.620385602210717;
    lats[660]=43.550086834993728;
    lats[661]=43.479788067770777;
    lats[662]=43.409489300541907;
    lats[663]=43.339190533307139;
    lats[664]=43.26889176606651;
    lats[665]=43.19859299882004;
    lats[666]=43.128294231567757;
    lats[667]=43.057995464309691;
    lats[668]=42.987696697045862;
    lats[669]=42.917397929776307;
    lats[670]=42.847099162501053;
    lats[671]=42.776800395220121;
    lats[672]=42.706501627933541;
    lats[673]=42.63620286064134;
    lats[674]=42.565904093343548;
    lats[675]=42.495605326040177;
    lats[676]=42.425306558731272;
    lats[677]=42.355007791416853;
    lats[678]=42.284709024096927;
    lats[679]=42.214410256771551;
    lats[680]=42.144111489440725;
    lats[681]=42.073812722104492;
    lats[682]=42.003513954762873;
    lats[683]=41.933215187415882;
    lats[684]=41.862916420063563;
    lats[685]=41.792617652705921;
    lats[686]=41.722318885343;
    lats[687]=41.6520201179748;
    lats[688]=41.581721350601363;
    lats[689]=41.511422583222718;
    lats[690]=41.441123815838885;
    lats[691]=41.370825048449873;
    lats[692]=41.300526281055724;
    lats[693]=41.230227513656445;
    lats[694]=41.159928746252085;
    lats[695]=41.089629978842645;
    lats[696]=41.01933121142816;
    lats[697]=40.949032444008644;
    lats[698]=40.878733676584126;
    lats[699]=40.808434909154634;
    lats[700]=40.738136141720176;
    lats[701]=40.667837374280786;
    lats[702]=40.597538606836487;
    lats[703]=40.527239839387299;
    lats[704]=40.456941071933244;
    lats[705]=40.386642304474343;
    lats[706]=40.316343537010617;
    lats[707]=40.246044769542102;
    lats[708]=40.175746002068806;
    lats[709]=40.105447234590748;
    lats[710]=40.035148467107952;
    lats[711]=39.964849699620437;
    lats[712]=39.894550932128247;
    lats[713]=39.824252164631375;
    lats[714]=39.753953397129855;
    lats[715]=39.683654629623703;
    lats[716]=39.613355862112947;
    lats[717]=39.543057094597607;
    lats[718]=39.472758327077692;
    lats[719]=39.402459559553229;
    lats[720]=39.332160792024254;
    lats[721]=39.261862024490775;
    lats[722]=39.191563256952804;
    lats[723]=39.121264489410365;
    lats[724]=39.050965721863491;
    lats[725]=38.980666954312184;
    lats[726]=38.910368186756479;
    lats[727]=38.840069419196389;
    lats[728]=38.769770651631937;
    lats[729]=38.699471884063136;
    lats[730]=38.629173116490001;
    lats[731]=38.558874348912568;
    lats[732]=38.488575581330842;
    lats[733]=38.418276813744846;
    lats[734]=38.347978046154608;
    lats[735]=38.277679278560143;
    lats[736]=38.20738051096145;
    lats[737]=38.137081743358586;
    lats[738]=38.066782975751536;
    lats[739]=37.99648420814033;
    lats[740]=37.926185440524989;
    lats[741]=37.855886672905527;
    lats[742]=37.785587905281965;
    lats[743]=37.715289137654317;
    lats[744]=37.644990370022605;
    lats[745]=37.574691602386856;
    lats[746]=37.504392834747065;
    lats[747]=37.434094067103274;
    lats[748]=37.363795299455489;
    lats[749]=37.293496531803719;
    lats[750]=37.223197764147997;
    lats[751]=37.152898996488332;
    lats[752]=37.082600228824752;
    lats[753]=37.012301461157264;
    lats[754]=36.942002693485883;
    lats[755]=36.871703925810628;
    lats[756]=36.801405158131523;
    lats[757]=36.731106390448581;
    lats[758]=36.660807622761808;
    lats[759]=36.590508855071242;
    lats[760]=36.520210087376888;
    lats[761]=36.449911319678755;
    lats[762]=36.379612551976876;
    lats[763]=36.309313784271254;
    lats[764]=36.239015016561908;
    lats[765]=36.16871624884886;
    lats[766]=36.098417481132117;
    lats[767]=36.028118713411708;
    lats[768]=35.957819945687639;
    lats[769]=35.887521177959933;
    lats[770]=35.817222410228595;
    lats[771]=35.746923642493655;
    lats[772]=35.676624874755113;
    lats[773]=35.606326107012997;
    lats[774]=35.536027339267314;
    lats[775]=35.465728571518085;
    lats[776]=35.395429803765317;
    lats[777]=35.325131036009047;
    lats[778]=35.254832268249267;
    lats[779]=35.184533500486005;
    lats[780]=35.114234732719261;
    lats[781]=35.043935964949064;
    lats[782]=34.973637197175435;
    lats[783]=34.903338429398374;
    lats[784]=34.833039661617903;
    lats[785]=34.762740893834028;
    lats[786]=34.692442126046771;
    lats[787]=34.622143358256153;
    lats[788]=34.551844590462188;
    lats[789]=34.481545822664863;
    lats[790]=34.411247054864234;
    lats[791]=34.340948287060286;
    lats[792]=34.270649519253041;
    lats[793]=34.200350751442521;
    lats[794]=34.130051983628725;
    lats[795]=34.059753215811682;
    lats[796]=33.989454447991392;
    lats[797]=33.919155680167876;
    lats[798]=33.848856912341155;
    lats[799]=33.778558144511237;
    lats[800]=33.708259376678136;
    lats[801]=33.637960608841851;
    lats[802]=33.567661841002426;
    lats[803]=33.497363073159853;
    lats[804]=33.42706430531414;
    lats[805]=33.356765537465314;
    lats[806]=33.286466769613391;
    lats[807]=33.216168001758369;
    lats[808]=33.145869233900278;
    lats[809]=33.075570466039117;
    lats[810]=33.005271698174909;
    lats[811]=32.934972930307666;
    lats[812]=32.864674162437396;
    lats[813]=32.794375394564113;
    lats[814]=32.724076626687825;
    lats[815]=32.653777858808567;
    lats[816]=32.583479090926325;
    lats[817]=32.513180323041112;
    lats[818]=32.442881555152965;
    lats[819]=32.372582787261891;
    lats[820]=32.302284019367875;
    lats[821]=32.231985251470959;
    lats[822]=32.161686483571145;
    lats[823]=32.091387715668439;
    lats[824]=32.021088947762863;
    lats[825]=31.950790179854422;
    lats[826]=31.880491411943137;
    lats[827]=31.810192644029012;
    lats[828]=31.739893876112063;
    lats[829]=31.669595108192297;
    lats[830]=31.599296340269738;
    lats[831]=31.528997572344384;
    lats[832]=31.458698804416255;
    lats[833]=31.388400036485361;
    lats[834]=31.318101268551715;
    lats[835]=31.247802500615318;
    lats[836]=31.177503732676204;
    lats[837]=31.107204964734358;
    lats[838]=31.036906196789811;
    lats[839]=30.966607428842572;
    lats[840]=30.896308660892647;
    lats[841]=30.826009892940046;
    lats[842]=30.755711124984781;
    lats[843]=30.685412357026873;
    lats[844]=30.615113589066322;
    lats[845]=30.544814821103138;
    lats[846]=30.47451605313735;
    lats[847]=30.404217285168947;
    lats[848]=30.333918517197947;
    lats[849]=30.263619749224372;
    lats[850]=30.19332098124822;
    lats[851]=30.123022213269511;
    lats[852]=30.052723445288244;
    lats[853]=29.98242467730444;
    lats[854]=29.91212590931811;
    lats[855]=29.841827141329258;
    lats[856]=29.771528373337894;
    lats[857]=29.701229605344039;
    lats[858]=29.630930837347698;
    lats[859]=29.560632069348884;
    lats[860]=29.490333301347597;
    lats[861]=29.420034533343859;
    lats[862]=29.349735765337677;
    lats[863]=29.279436997329057;
    lats[864]=29.209138229318015;
    lats[865]=29.138839461304556;
    lats[866]=29.068540693288696;
    lats[867]=28.998241925270449;
    lats[868]=28.927943157249814;
    lats[869]=28.857644389226806;
    lats[870]=28.787345621201432;
    lats[871]=28.717046853173709;
    lats[872]=28.646748085143642;
    lats[873]=28.576449317111244;
    lats[874]=28.506150549076519;
    lats[875]=28.435851781039485;
    lats[876]=28.365553013000145;
    lats[877]=28.29525424495851;
    lats[878]=28.224955476914594;
    lats[879]=28.154656708868405;
    lats[880]=28.084357940819952;
    lats[881]=28.014059172769244;
    lats[882]=27.94376040471629;
    lats[883]=27.873461636661098;
    lats[884]=27.803162868603682;
    lats[885]=27.732864100544052;
    lats[886]=27.662565332482213;
    lats[887]=27.592266564418171;
    lats[888]=27.521967796351948;
    lats[889]=27.451669028283543;
    lats[890]=27.381370260212968;
    lats[891]=27.311071492140236;
    lats[892]=27.240772724065348;
    lats[893]=27.170473955988321;
    lats[894]=27.100175187909159;
    lats[895]=27.029876419827872;
    lats[896]=26.959577651744471;
    lats[897]=26.889278883658971;
    lats[898]=26.818980115571364;
    lats[899]=26.748681347481678;
    lats[900]=26.678382579389908;
    lats[901]=26.608083811296069;
    lats[902]=26.53778504320017;
    lats[903]=26.467486275102218;
    lats[904]=26.397187507002222;
    lats[905]=26.326888738900195;
    lats[906]=26.256589970796135;
    lats[907]=26.186291202690064;
    lats[908]=26.115992434581983;
    lats[909]=26.045693666471902;
    lats[910]=25.975394898359827;
    lats[911]=25.90509613024577;
    lats[912]=25.834797362129745;
    lats[913]=25.764498594011751;
    lats[914]=25.694199825891793;
    lats[915]=25.623901057769892;
    lats[916]=25.553602289646051;
    lats[917]=25.483303521520277;
    lats[918]=25.413004753392578;
    lats[919]=25.342705985262967;
    lats[920]=25.272407217131445;
    lats[921]=25.202108448998025;
    lats[922]=25.13180968086272;
    lats[923]=25.061510912725527;
    lats[924]=24.991212144586456;
    lats[925]=24.920913376445526;
    lats[926]=24.850614608302738;
    lats[927]=24.780315840158096;
    lats[928]=24.710017072011613;
    lats[929]=24.639718303863294;
    lats[930]=24.569419535713152;
    lats[931]=24.499120767561195;
    lats[932]=24.428821999407425;
    lats[933]=24.358523231251851;
    lats[934]=24.288224463094483;
    lats[935]=24.217925694935328;
    lats[936]=24.1476269267744;
    lats[937]=24.077328158611696;
    lats[938]=24.007029390447226;
    lats[939]=23.936730622281004;
    lats[940]=23.866431854113038;
    lats[941]=23.796133085943328;
    lats[942]=23.725834317771888;
    lats[943]=23.655535549598721;
    lats[944]=23.585236781423838;
    lats[945]=23.514938013247242;
    lats[946]=23.444639245068949;
    lats[947]=23.374340476888957;
    lats[948]=23.304041708707278;
    lats[949]=23.233742940523921;
    lats[950]=23.163444172338895;
    lats[951]=23.0931454041522;
    lats[952]=23.022846635963852;
    lats[953]=22.952547867773848;
    lats[954]=22.882249099582204;
    lats[955]=22.811950331388925;
    lats[956]=22.741651563194019;
    lats[957]=22.671352794997489;
    lats[958]=22.60105402679935;
    lats[959]=22.530755258599601;
    lats[960]=22.460456490398254;
    lats[961]=22.390157722195315;
    lats[962]=22.319858953990789;
    lats[963]=22.249560185784691;
    lats[964]=22.179261417577013;
    lats[965]=22.108962649367779;
    lats[966]=22.038663881156989;
    lats[967]=21.968365112944642;
    lats[968]=21.898066344730758;
    lats[969]=21.827767576515338;
    lats[970]=21.757468808298391;
    lats[971]=21.687170040079913;
    lats[972]=21.616871271859928;
    lats[973]=21.546572503638437;
    lats[974]=21.47627373541544;
    lats[975]=21.40597496719095;
    lats[976]=21.335676198964972;
    lats[977]=21.265377430737512;
    lats[978]=21.195078662508585;
    lats[979]=21.124779894278181;
    lats[980]=21.054481126046323;
    lats[981]=20.984182357813012;
    lats[982]=20.913883589578251;
    lats[983]=20.843584821342048;
    lats[984]=20.773286053104417;
    lats[985]=20.702987284865355;
    lats[986]=20.632688516624874;
    lats[987]=20.562389748382977;
    lats[988]=20.492090980139672;
    lats[989]=20.421792211894967;
    lats[990]=20.35149344364887;
    lats[991]=20.28119467540138;
    lats[992]=20.210895907152516;
    lats[993]=20.140597138902272;
    lats[994]=20.070298370650661;
    lats[995]=19.999999602397686;
    lats[996]=19.929700834143357;
    lats[997]=19.859402065887682;
    lats[998]=19.789103297630657;
    lats[999]=19.718804529372303;
    lats[1000]=19.648505761112613;
    lats[1001]=19.578206992851602;
    lats[1002]=19.507908224589269;
    lats[1003]=19.437609456325632;
    lats[1004]=19.367310688060684;
    lats[1005]=19.297011919794439;
    lats[1006]=19.226713151526898;
    lats[1007]=19.15641438325807;
    lats[1008]=19.086115614987968;
    lats[1009]=19.015816846716586;
    lats[1010]=18.945518078443939;
    lats[1011]=18.875219310170031;
    lats[1012]=18.804920541894862;
    lats[1013]=18.734621773618446;
    lats[1014]=18.664323005340787;
    lats[1015]=18.594024237061891;
    lats[1016]=18.523725468781763;
    lats[1017]=18.453426700500408;
    lats[1018]=18.383127932217832;
    lats[1019]=18.312829163934047;
    lats[1020]=18.242530395649048;
    lats[1021]=18.172231627362851;
    lats[1022]=18.101932859075458;
    lats[1023]=18.031634090786874;
    lats[1024]=17.96133532249711;
    lats[1025]=17.89103655420616;
    lats[1026]=17.820737785914044;
    lats[1027]=17.75043901762076;
    lats[1028]=17.680140249326314;
    lats[1029]=17.60984148103071;
    lats[1030]=17.539542712733962;
    lats[1031]=17.469243944436066;
    lats[1032]=17.39894517613704;
    lats[1033]=17.328646407836878;
    lats[1034]=17.258347639535586;
    lats[1035]=17.188048871233182;
    lats[1036]=17.117750102929655;
    lats[1037]=17.04745133462502;
    lats[1038]=16.977152566319283;
    lats[1039]=16.906853798012452;
    lats[1040]=16.836555029704527;
    lats[1041]=16.766256261395515;
    lats[1042]=16.69595749308542;
    lats[1043]=16.625658724774254;
    lats[1044]=16.555359956462013;
    lats[1045]=16.485061188148713;
    lats[1046]=16.41476241983435;
    lats[1047]=16.344463651518936;
    lats[1048]=16.274164883202477;
    lats[1049]=16.203866114884974;
    lats[1050]=16.133567346566434;
    lats[1051]=16.063268578246863;
    lats[1052]=15.992969809926265;
    lats[1053]=15.922671041604652;
    lats[1054]=15.852372273282016;
    lats[1055]=15.78207350495838;
    lats[1056]=15.711774736633735;
    lats[1057]=15.641475968308091;
    lats[1058]=15.571177199981456;
    lats[1059]=15.500878431653829;
    lats[1060]=15.430579663325226;
    lats[1061]=15.360280894995643;
    lats[1062]=15.289982126665089;
    lats[1063]=15.219683358333569;
    lats[1064]=15.149384590001089;
    lats[1065]=15.07908582166765;
    lats[1066]=15.008787053333259;
    lats[1067]=14.938488284997929;
    lats[1068]=14.868189516661655;
    lats[1069]=14.797890748324447;
    lats[1070]=14.727591979986309;
    lats[1071]=14.657293211647247;
    lats[1072]=14.586994443307265;
    lats[1073]=14.516695674966371;
    lats[1074]=14.446396906624567;
    lats[1075]=14.376098138281863;
    lats[1076]=14.305799369938256;
    lats[1077]=14.23550060159376;
    lats[1078]=14.165201833248371;
    lats[1079]=14.0949030649021;
    lats[1080]=14.024604296554955;
    lats[1081]=13.954305528206934;
    lats[1082]=13.884006759858046;
    lats[1083]=13.813707991508297;
    lats[1084]=13.743409223157688;
    lats[1085]=13.673110454806226;
    lats[1086]=13.602811686453919;
    lats[1087]=13.532512918100766;
    lats[1088]=13.46221414974678;
    lats[1089]=13.391915381391959;
    lats[1090]=13.32161661303631;
    lats[1091]=13.251317844679837;
    lats[1092]=13.181019076322551;
    lats[1093]=13.110720307964451;
    lats[1094]=13.040421539605545;
    lats[1095]=12.970122771245832;
    lats[1096]=12.899824002885323;
    lats[1097]=12.829525234524022;
    lats[1098]=12.759226466161934;
    lats[1099]=12.688927697799061;
    lats[1100]=12.618628929435411;
    lats[1101]=12.548330161070988;
    lats[1102]=12.478031392705796;
    lats[1103]=12.407732624339841;
    lats[1104]=12.337433855973126;
    lats[1105]=12.267135087605659;
    lats[1106]=12.196836319237443;
    lats[1107]=12.126537550868482;
    lats[1108]=12.056238782498781;
    lats[1109]=11.985940014128348;
    lats[1110]=11.915641245757183;
    lats[1111]=11.845342477385294;
    lats[1112]=11.775043709012685;
    lats[1113]=11.704744940639358;
    lats[1114]=11.634446172265324;
    lats[1115]=11.564147403890583;
    lats[1116]=11.493848635515141;
    lats[1117]=11.423549867139002;
    lats[1118]=11.35325109876217;
    lats[1119]=11.282952330384653;
    lats[1120]=11.212653562006453;
    lats[1121]=11.142354793627575;
    lats[1122]=11.072056025248026;
    lats[1123]=11.001757256867807;
    lats[1124]=10.931458488486923;
    lats[1125]=10.861159720105382;
    lats[1126]=10.790860951723188;
    lats[1127]=10.720562183340341;
    lats[1128]=10.65026341495685;
    lats[1129]=10.579964646572719;
    lats[1130]=10.509665878187954;
    lats[1131]=10.439367109802557;
    lats[1132]=10.369068341416533;
    lats[1133]=10.298769573029887;
    lats[1134]=10.228470804642624;
    lats[1135]=10.158172036254747;
    lats[1136]=10.087873267866264;
    lats[1137]=10.017574499477174;
    lats[1138]=9.9472757310874869;
    lats[1139]=9.8769769626972046;
    lats[1140]=9.8066781943063344;
    lats[1141]=9.7363794259148779;
    lats[1142]=9.6660806575228388;
    lats[1143]=9.5957818891302242;
    lats[1144]=9.5254831207370376;
    lats[1145]=9.4551843523432826;
    lats[1146]=9.3848855839489662;
    lats[1147]=9.3145868155540921;
    lats[1148]=9.2442880471586619;
    lats[1149]=9.1739892787626829;
    lats[1150]=9.1036905103661585;
    lats[1151]=9.0333917419690941;
    lats[1152]=8.963092973571495;
    lats[1153]=8.8927942051733631;
    lats[1154]=8.8224954367747017;
    lats[1155]=8.7521966683755217;
    lats[1156]=8.6818978999758194;
    lats[1157]=8.6115991315756055;
    lats[1158]=8.5413003631748801;
    lats[1159]=8.4710015947736537;
    lats[1160]=8.4007028263719228;
    lats[1161]=8.3304040579696963;
    lats[1162]=8.2601052895669778;
    lats[1163]=8.1898065211637725;
    lats[1164]=8.1195077527600841;
    lats[1165]=8.049208984355916;
    lats[1166]=7.9789102159512737;
    lats[1167]=7.9086114475461606;
    lats[1168]=7.8383126791405831;
    lats[1169]=7.7680139107345463;
    lats[1170]=7.6977151423280494;
    lats[1171]=7.6274163739210996;
    lats[1172]=7.557117605513703;
    lats[1173]=7.4868188371058624;
    lats[1174]=7.4165200686975803;
    lats[1175]=7.3462213002888648;
    lats[1176]=7.2759225318797176;
    lats[1177]=7.2056237634701441;
    lats[1178]=7.1353249950601469;
    lats[1179]=7.0650262266497315;
    lats[1180]=6.994727458238903;
    lats[1181]=6.924428689827665;
    lats[1182]=6.8541299214160212;
    lats[1183]=6.7838311530039768;
    lats[1184]=6.7135323845915353;
    lats[1185]=6.6432336161787013;
    lats[1186]=6.5729348477654792;
    lats[1187]=6.5026360793518734;
    lats[1188]=6.4323373109378874;
    lats[1189]=6.3620385425235257;
    lats[1190]=6.2917397741087928;
    lats[1191]=6.2214410056936931;
    lats[1192]=6.151142237278231;
    lats[1193]=6.0808434688624091;
    lats[1194]=6.0105447004462347;
    lats[1195]=5.9402459320297085;
    lats[1196]=5.869947163612836;
    lats[1197]=5.7996483951956233;
    lats[1198]=5.729349626778073;
    lats[1199]=5.6590508583601888;
    lats[1200]=5.5887520899419751;
    lats[1201]=5.5184533215234373;
    lats[1202]=5.4481545531045787;
    lats[1203]=5.3778557846854023;
    lats[1204]=5.3075570162659149;
    lats[1205]=5.2372582478461194;
    lats[1206]=5.1669594794260192;
    lats[1207]=5.0966607110056197;
    lats[1208]=5.0263619425849244;
    lats[1209]=4.9560631741639369;
    lats[1210]=4.8857644057426626;
    lats[1211]=4.8154656373211049;
    lats[1212]=4.7451668688992683;
    lats[1213]=4.6748681004771564;
    lats[1214]=4.6045693320547736;
    lats[1215]=4.5342705636321252;
    lats[1216]=4.4639717952092139;
    lats[1217]=4.3936730267860451;
    lats[1218]=4.3233742583626205;
    lats[1219]=4.2530754899389471;
    lats[1220]=4.1827767215150269;
    lats[1221]=4.1124779530908659;
    lats[1222]=4.0421791846664661;
    lats[1223]=3.9718804162418326;
    lats[1224]=3.90158164781697;
    lats[1225]=3.8312828793918823;
    lats[1226]=3.7609841109665734;
    lats[1227]=3.6906853425410477;
    lats[1228]=3.6203865741153085;
    lats[1229]=3.5500878056893601;
    lats[1230]=3.4797890372632065;
    lats[1231]=3.4094902688368531;
    lats[1232]=3.339191500410303;
    lats[1233]=3.2688927319835597;
    lats[1234]=3.1985939635566285;
    lats[1235]=3.1282951951295126;
    lats[1236]=3.0579964267022164;
    lats[1237]=2.9876976582747439;
    lats[1238]=2.9173988898470999;
    lats[1239]=2.8471001214192873;
    lats[1240]=2.7768013529913107;
    lats[1241]=2.7065025845631743;
    lats[1242]=2.6362038161348824;
    lats[1243]=2.5659050477064382;
    lats[1244]=2.4956062792778466;
    lats[1245]=2.4253075108491116;
    lats[1246]=2.3550087424202366;
    lats[1247]=2.2847099739912267;
    lats[1248]=2.2144112055620848;
    lats[1249]=2.1441124371328155;
    lats[1250]=2.0738136687034232;
    lats[1251]=2.0035149002739114;
    lats[1252]=1.9332161318442849;
    lats[1253]=1.8629173634145471;
    lats[1254]=1.792618594984702;
    lats[1255]=1.7223198265547539;
    lats[1256]=1.6520210581247066;
    lats[1257]=1.5817222896945646;
    lats[1258]=1.5114235212643317;
    lats[1259]=1.4411247528340119;
    lats[1260]=1.3708259844036093;
    lats[1261]=1.300527215973128;
    lats[1262]=1.2302284475425722;
    lats[1263]=1.1599296791119456;
    lats[1264]=1.0896309106812523;
    lats[1265]=1.0193321422504964;
    lats[1266]=0.949033373819682;
    lats[1267]=0.87873460538881287;
    lats[1268]=0.80843583695789356;
    lats[1269]=0.73813706852692773;
    lats[1270]=0.66783830009591949;
    lats[1271]=0.59753953166487306;
    lats[1272]=0.52724076323379232;
    lats[1273]=0.45694199480268116;
    lats[1274]=0.3866432263715438;
    lats[1275]=0.31634445794038429;
    lats[1276]=0.24604568950920663;
    lats[1277]=0.17574692107801482;
    lats[1278]=0.10544815264681295;
    lats[1279]=0.035149384215604956;
    lats[1280]=-0.035149384215604956;
    lats[1281]=-0.10544815264681295;
    lats[1282]=-0.17574692107801482;
    lats[1283]=-0.24604568950920663;
    lats[1284]=-0.31634445794038429;
    lats[1285]=-0.3866432263715438;
    lats[1286]=-0.45694199480268116;
    lats[1287]=-0.52724076323379232;
    lats[1288]=-0.59753953166487306;
    lats[1289]=-0.66783830009591949;
    lats[1290]=-0.73813706852692773;
    lats[1291]=-0.80843583695789356;
    lats[1292]=-0.87873460538881287;
    lats[1293]=-0.949033373819682;
    lats[1294]=-1.0193321422504964;
    lats[1295]=-1.0896309106812523;
    lats[1296]=-1.1599296791119456;
    lats[1297]=-1.2302284475425722;
    lats[1298]=-1.300527215973128;
    lats[1299]=-1.3708259844036093;
    lats[1300]=-1.4411247528340119;
    lats[1301]=-1.5114235212643317;
    lats[1302]=-1.5817222896945646;
    lats[1303]=-1.6520210581247066;
    lats[1304]=-1.7223198265547539;
    lats[1305]=-1.792618594984702;
    lats[1306]=-1.8629173634145471;
    lats[1307]=-1.9332161318442849;
    lats[1308]=-2.0035149002739114;
    lats[1309]=-2.0738136687034232;
    lats[1310]=-2.1441124371328155;
    lats[1311]=-2.2144112055620848;
    lats[1312]=-2.2847099739912267;
    lats[1313]=-2.3550087424202366;
    lats[1314]=-2.4253075108491116;
    lats[1315]=-2.4956062792778466;
    lats[1316]=-2.5659050477064382;
    lats[1317]=-2.6362038161348824;
    lats[1318]=-2.7065025845631743;
    lats[1319]=-2.7768013529913107;
    lats[1320]=-2.8471001214192873;
    lats[1321]=-2.9173988898470999;
    lats[1322]=-2.9876976582747439;
    lats[1323]=-3.0579964267022164;
    lats[1324]=-3.1282951951295126;
    lats[1325]=-3.1985939635566285;
    lats[1326]=-3.2688927319835597;
    lats[1327]=-3.339191500410303;
    lats[1328]=-3.4094902688368531;
    lats[1329]=-3.4797890372632065;
    lats[1330]=-3.5500878056893601;
    lats[1331]=-3.6203865741153085;
    lats[1332]=-3.6906853425410477;
    lats[1333]=-3.7609841109665734;
    lats[1334]=-3.8312828793918823;
    lats[1335]=-3.90158164781697;
    lats[1336]=-3.9718804162418326;
    lats[1337]=-4.0421791846664661;
    lats[1338]=-4.1124779530908659;
    lats[1339]=-4.1827767215150269;
    lats[1340]=-4.2530754899389471;
    lats[1341]=-4.3233742583626205;
    lats[1342]=-4.3936730267860451;
    lats[1343]=-4.4639717952092139;
    lats[1344]=-4.5342705636321252;
    lats[1345]=-4.6045693320547736;
    lats[1346]=-4.6748681004771564;
    lats[1347]=-4.7451668688992683;
    lats[1348]=-4.8154656373211049;
    lats[1349]=-4.8857644057426626;
    lats[1350]=-4.9560631741639369;
    lats[1351]=-5.0263619425849244;
    lats[1352]=-5.0966607110056197;
    lats[1353]=-5.1669594794260192;
    lats[1354]=-5.2372582478461194;
    lats[1355]=-5.3075570162659149;
    lats[1356]=-5.3778557846854023;
    lats[1357]=-5.4481545531045787;
    lats[1358]=-5.5184533215234373;
    lats[1359]=-5.5887520899419751;
    lats[1360]=-5.6590508583601888;
    lats[1361]=-5.729349626778073;
    lats[1362]=-5.7996483951956233;
    lats[1363]=-5.869947163612836;
    lats[1364]=-5.9402459320297085;
    lats[1365]=-6.0105447004462347;
    lats[1366]=-6.0808434688624091;
    lats[1367]=-6.151142237278231;
    lats[1368]=-6.2214410056936931;
    lats[1369]=-6.2917397741087928;
    lats[1370]=-6.3620385425235257;
    lats[1371]=-6.4323373109378874;
    lats[1372]=-6.5026360793518734;
    lats[1373]=-6.5729348477654792;
    lats[1374]=-6.6432336161787013;
    lats[1375]=-6.7135323845915353;
    lats[1376]=-6.7838311530039768;
    lats[1377]=-6.8541299214160212;
    lats[1378]=-6.924428689827665;
    lats[1379]=-6.994727458238903;
    lats[1380]=-7.0650262266497315;
    lats[1381]=-7.1353249950601469;
    lats[1382]=-7.2056237634701441;
    lats[1383]=-7.2759225318797176;
    lats[1384]=-7.3462213002888648;
    lats[1385]=-7.4165200686975803;
    lats[1386]=-7.4868188371058624;
    lats[1387]=-7.557117605513703;
    lats[1388]=-7.6274163739210996;
    lats[1389]=-7.6977151423280494;
    lats[1390]=-7.7680139107345463;
    lats[1391]=-7.8383126791405831;
    lats[1392]=-7.9086114475461606;
    lats[1393]=-7.9789102159512737;
    lats[1394]=-8.049208984355916;
    lats[1395]=-8.1195077527600841;
    lats[1396]=-8.1898065211637725;
    lats[1397]=-8.2601052895669778;
    lats[1398]=-8.3304040579696963;
    lats[1399]=-8.4007028263719228;
    lats[1400]=-8.4710015947736537;
    lats[1401]=-8.5413003631748801;
    lats[1402]=-8.6115991315756055;
    lats[1403]=-8.6818978999758194;
    lats[1404]=-8.7521966683755217;
    lats[1405]=-8.8224954367747017;
    lats[1406]=-8.8927942051733631;
    lats[1407]=-8.963092973571495;
    lats[1408]=-9.0333917419690941;
    lats[1409]=-9.1036905103661585;
    lats[1410]=-9.1739892787626829;
    lats[1411]=-9.2442880471586619;
    lats[1412]=-9.3145868155540921;
    lats[1413]=-9.3848855839489662;
    lats[1414]=-9.4551843523432826;
    lats[1415]=-9.5254831207370376;
    lats[1416]=-9.5957818891302242;
    lats[1417]=-9.6660806575228388;
    lats[1418]=-9.7363794259148779;
    lats[1419]=-9.8066781943063344;
    lats[1420]=-9.8769769626972046;
    lats[1421]=-9.9472757310874869;
    lats[1422]=-10.017574499477174;
    lats[1423]=-10.087873267866264;
    lats[1424]=-10.158172036254747;
    lats[1425]=-10.228470804642624;
    lats[1426]=-10.298769573029887;
    lats[1427]=-10.369068341416533;
    lats[1428]=-10.439367109802557;
    lats[1429]=-10.509665878187954;
    lats[1430]=-10.579964646572719;
    lats[1431]=-10.65026341495685;
    lats[1432]=-10.720562183340341;
    lats[1433]=-10.790860951723188;
    lats[1434]=-10.861159720105382;
    lats[1435]=-10.931458488486923;
    lats[1436]=-11.001757256867807;
    lats[1437]=-11.072056025248026;
    lats[1438]=-11.142354793627575;
    lats[1439]=-11.212653562006453;
    lats[1440]=-11.282952330384653;
    lats[1441]=-11.35325109876217;
    lats[1442]=-11.423549867139002;
    lats[1443]=-11.493848635515141;
    lats[1444]=-11.564147403890583;
    lats[1445]=-11.634446172265324;
    lats[1446]=-11.704744940639358;
    lats[1447]=-11.775043709012685;
    lats[1448]=-11.845342477385294;
    lats[1449]=-11.915641245757183;
    lats[1450]=-11.985940014128348;
    lats[1451]=-12.056238782498781;
    lats[1452]=-12.126537550868482;
    lats[1453]=-12.196836319237443;
    lats[1454]=-12.267135087605659;
    lats[1455]=-12.337433855973126;
    lats[1456]=-12.407732624339841;
    lats[1457]=-12.478031392705796;
    lats[1458]=-12.548330161070988;
    lats[1459]=-12.618628929435411;
    lats[1460]=-12.688927697799061;
    lats[1461]=-12.759226466161934;
    lats[1462]=-12.829525234524022;
    lats[1463]=-12.899824002885323;
    lats[1464]=-12.970122771245832;
    lats[1465]=-13.040421539605545;
    lats[1466]=-13.110720307964451;
    lats[1467]=-13.181019076322551;
    lats[1468]=-13.251317844679837;
    lats[1469]=-13.32161661303631;
    lats[1470]=-13.391915381391959;
    lats[1471]=-13.46221414974678;
    lats[1472]=-13.532512918100766;
    lats[1473]=-13.602811686453919;
    lats[1474]=-13.673110454806226;
    lats[1475]=-13.743409223157688;
    lats[1476]=-13.813707991508297;
    lats[1477]=-13.884006759858046;
    lats[1478]=-13.954305528206934;
    lats[1479]=-14.024604296554955;
    lats[1480]=-14.0949030649021;
    lats[1481]=-14.165201833248371;
    lats[1482]=-14.23550060159376;
    lats[1483]=-14.305799369938256;
    lats[1484]=-14.376098138281863;
    lats[1485]=-14.446396906624567;
    lats[1486]=-14.516695674966371;
    lats[1487]=-14.586994443307265;
    lats[1488]=-14.657293211647247;
    lats[1489]=-14.727591979986309;
    lats[1490]=-14.797890748324447;
    lats[1491]=-14.868189516661655;
    lats[1492]=-14.938488284997929;
    lats[1493]=-15.008787053333259;
    lats[1494]=-15.07908582166765;
    lats[1495]=-15.149384590001089;
    lats[1496]=-15.219683358333569;
    lats[1497]=-15.289982126665089;
    lats[1498]=-15.360280894995643;
    lats[1499]=-15.430579663325226;
    lats[1500]=-15.500878431653829;
    lats[1501]=-15.571177199981456;
    lats[1502]=-15.641475968308091;
    lats[1503]=-15.711774736633735;
    lats[1504]=-15.78207350495838;
    lats[1505]=-15.852372273282016;
    lats[1506]=-15.922671041604652;
    lats[1507]=-15.992969809926265;
    lats[1508]=-16.063268578246863;
    lats[1509]=-16.133567346566434;
    lats[1510]=-16.203866114884974;
    lats[1511]=-16.274164883202477;
    lats[1512]=-16.344463651518936;
    lats[1513]=-16.41476241983435;
    lats[1514]=-16.485061188148713;
    lats[1515]=-16.555359956462013;
    lats[1516]=-16.625658724774254;
    lats[1517]=-16.69595749308542;
    lats[1518]=-16.766256261395515;
    lats[1519]=-16.836555029704527;
    lats[1520]=-16.906853798012452;
    lats[1521]=-16.977152566319283;
    lats[1522]=-17.04745133462502;
    lats[1523]=-17.117750102929655;
    lats[1524]=-17.188048871233182;
    lats[1525]=-17.258347639535586;
    lats[1526]=-17.328646407836878;
    lats[1527]=-17.39894517613704;
    lats[1528]=-17.469243944436066;
    lats[1529]=-17.539542712733962;
    lats[1530]=-17.60984148103071;
    lats[1531]=-17.680140249326314;
    lats[1532]=-17.75043901762076;
    lats[1533]=-17.820737785914044;
    lats[1534]=-17.89103655420616;
    lats[1535]=-17.96133532249711;
    lats[1536]=-18.031634090786874;
    lats[1537]=-18.101932859075458;
    lats[1538]=-18.172231627362851;
    lats[1539]=-18.242530395649048;
    lats[1540]=-18.312829163934047;
    lats[1541]=-18.383127932217832;
    lats[1542]=-18.453426700500408;
    lats[1543]=-18.523725468781763;
    lats[1544]=-18.594024237061891;
    lats[1545]=-18.664323005340787;
    lats[1546]=-18.734621773618446;
    lats[1547]=-18.804920541894862;
    lats[1548]=-18.875219310170031;
    lats[1549]=-18.945518078443939;
    lats[1550]=-19.015816846716586;
    lats[1551]=-19.086115614987968;
    lats[1552]=-19.15641438325807;
    lats[1553]=-19.226713151526898;
    lats[1554]=-19.297011919794439;
    lats[1555]=-19.367310688060684;
    lats[1556]=-19.437609456325632;
    lats[1557]=-19.507908224589269;
    lats[1558]=-19.578206992851602;
    lats[1559]=-19.648505761112613;
    lats[1560]=-19.718804529372303;
    lats[1561]=-19.789103297630657;
    lats[1562]=-19.859402065887682;
    lats[1563]=-19.929700834143357;
    lats[1564]=-19.999999602397686;
    lats[1565]=-20.070298370650661;
    lats[1566]=-20.140597138902272;
    lats[1567]=-20.210895907152516;
    lats[1568]=-20.28119467540138;
    lats[1569]=-20.35149344364887;
    lats[1570]=-20.421792211894967;
    lats[1571]=-20.492090980139672;
    lats[1572]=-20.562389748382977;
    lats[1573]=-20.632688516624874;
    lats[1574]=-20.702987284865355;
    lats[1575]=-20.773286053104417;
    lats[1576]=-20.843584821342048;
    lats[1577]=-20.913883589578251;
    lats[1578]=-20.984182357813012;
    lats[1579]=-21.054481126046323;
    lats[1580]=-21.124779894278181;
    lats[1581]=-21.195078662508585;
    lats[1582]=-21.265377430737512;
    lats[1583]=-21.335676198964972;
    lats[1584]=-21.40597496719095;
    lats[1585]=-21.47627373541544;
    lats[1586]=-21.546572503638437;
    lats[1587]=-21.616871271859928;
    lats[1588]=-21.687170040079913;
    lats[1589]=-21.757468808298391;
    lats[1590]=-21.827767576515338;
    lats[1591]=-21.898066344730758;
    lats[1592]=-21.968365112944642;
    lats[1593]=-22.038663881156989;
    lats[1594]=-22.108962649367779;
    lats[1595]=-22.179261417577013;
    lats[1596]=-22.249560185784691;
    lats[1597]=-22.319858953990789;
    lats[1598]=-22.390157722195315;
    lats[1599]=-22.460456490398254;
    lats[1600]=-22.530755258599601;
    lats[1601]=-22.60105402679935;
    lats[1602]=-22.671352794997489;
    lats[1603]=-22.741651563194019;
    lats[1604]=-22.811950331388925;
    lats[1605]=-22.882249099582204;
    lats[1606]=-22.952547867773848;
    lats[1607]=-23.022846635963852;
    lats[1608]=-23.0931454041522;
    lats[1609]=-23.163444172338895;
    lats[1610]=-23.233742940523921;
    lats[1611]=-23.304041708707278;
    lats[1612]=-23.374340476888957;
    lats[1613]=-23.444639245068949;
    lats[1614]=-23.514938013247242;
    lats[1615]=-23.585236781423838;
    lats[1616]=-23.655535549598721;
    lats[1617]=-23.725834317771888;
    lats[1618]=-23.796133085943328;
    lats[1619]=-23.866431854113038;
    lats[1620]=-23.936730622281004;
    lats[1621]=-24.007029390447226;
    lats[1622]=-24.077328158611696;
    lats[1623]=-24.1476269267744;
    lats[1624]=-24.217925694935328;
    lats[1625]=-24.288224463094483;
    lats[1626]=-24.358523231251851;
    lats[1627]=-24.428821999407425;
    lats[1628]=-24.499120767561195;
    lats[1629]=-24.569419535713152;
    lats[1630]=-24.639718303863294;
    lats[1631]=-24.710017072011613;
    lats[1632]=-24.780315840158096;
    lats[1633]=-24.850614608302738;
    lats[1634]=-24.920913376445526;
    lats[1635]=-24.991212144586456;
    lats[1636]=-25.061510912725527;
    lats[1637]=-25.13180968086272;
    lats[1638]=-25.202108448998025;
    lats[1639]=-25.272407217131445;
    lats[1640]=-25.342705985262967;
    lats[1641]=-25.413004753392578;
    lats[1642]=-25.483303521520277;
    lats[1643]=-25.553602289646051;
    lats[1644]=-25.623901057769892;
    lats[1645]=-25.694199825891793;
    lats[1646]=-25.764498594011751;
    lats[1647]=-25.834797362129745;
    lats[1648]=-25.90509613024577;
    lats[1649]=-25.975394898359827;
    lats[1650]=-26.045693666471902;
    lats[1651]=-26.115992434581983;
    lats[1652]=-26.186291202690064;
    lats[1653]=-26.256589970796135;
    lats[1654]=-26.326888738900195;
    lats[1655]=-26.397187507002222;
    lats[1656]=-26.467486275102218;
    lats[1657]=-26.53778504320017;
    lats[1658]=-26.608083811296069;
    lats[1659]=-26.678382579389908;
    lats[1660]=-26.748681347481678;
    lats[1661]=-26.818980115571364;
    lats[1662]=-26.889278883658971;
    lats[1663]=-26.959577651744471;
    lats[1664]=-27.029876419827872;
    lats[1665]=-27.100175187909159;
    lats[1666]=-27.170473955988321;
    lats[1667]=-27.240772724065348;
    lats[1668]=-27.311071492140236;
    lats[1669]=-27.381370260212968;
    lats[1670]=-27.451669028283543;
    lats[1671]=-27.521967796351948;
    lats[1672]=-27.592266564418171;
    lats[1673]=-27.662565332482213;
    lats[1674]=-27.732864100544052;
    lats[1675]=-27.803162868603682;
    lats[1676]=-27.873461636661098;
    lats[1677]=-27.94376040471629;
    lats[1678]=-28.014059172769244;
    lats[1679]=-28.084357940819952;
    lats[1680]=-28.154656708868405;
    lats[1681]=-28.224955476914594;
    lats[1682]=-28.29525424495851;
    lats[1683]=-28.365553013000145;
    lats[1684]=-28.435851781039485;
    lats[1685]=-28.506150549076519;
    lats[1686]=-28.576449317111244;
    lats[1687]=-28.646748085143642;
    lats[1688]=-28.717046853173709;
    lats[1689]=-28.787345621201432;
    lats[1690]=-28.857644389226806;
    lats[1691]=-28.927943157249814;
    lats[1692]=-28.998241925270449;
    lats[1693]=-29.068540693288696;
    lats[1694]=-29.138839461304556;
    lats[1695]=-29.209138229318015;
    lats[1696]=-29.279436997329057;
    lats[1697]=-29.349735765337677;
    lats[1698]=-29.420034533343859;
    lats[1699]=-29.490333301347597;
    lats[1700]=-29.560632069348884;
    lats[1701]=-29.630930837347698;
    lats[1702]=-29.701229605344039;
    lats[1703]=-29.771528373337894;
    lats[1704]=-29.841827141329258;
    lats[1705]=-29.91212590931811;
    lats[1706]=-29.98242467730444;
    lats[1707]=-30.052723445288244;
    lats[1708]=-30.123022213269511;
    lats[1709]=-30.19332098124822;
    lats[1710]=-30.263619749224372;
    lats[1711]=-30.333918517197947;
    lats[1712]=-30.404217285168947;
    lats[1713]=-30.47451605313735;
    lats[1714]=-30.544814821103138;
    lats[1715]=-30.615113589066322;
    lats[1716]=-30.685412357026873;
    lats[1717]=-30.755711124984781;
    lats[1718]=-30.826009892940046;
    lats[1719]=-30.896308660892647;
    lats[1720]=-30.966607428842572;
    lats[1721]=-31.036906196789811;
    lats[1722]=-31.107204964734358;
    lats[1723]=-31.177503732676204;
    lats[1724]=-31.247802500615318;
    lats[1725]=-31.318101268551715;
    lats[1726]=-31.388400036485361;
    lats[1727]=-31.458698804416255;
    lats[1728]=-31.528997572344384;
    lats[1729]=-31.599296340269738;
    lats[1730]=-31.669595108192297;
    lats[1731]=-31.739893876112063;
    lats[1732]=-31.810192644029012;
    lats[1733]=-31.880491411943137;
    lats[1734]=-31.950790179854422;
    lats[1735]=-32.021088947762863;
    lats[1736]=-32.091387715668439;
    lats[1737]=-32.161686483571145;
    lats[1738]=-32.231985251470959;
    lats[1739]=-32.302284019367875;
    lats[1740]=-32.372582787261891;
    lats[1741]=-32.442881555152965;
    lats[1742]=-32.513180323041112;
    lats[1743]=-32.583479090926325;
    lats[1744]=-32.653777858808567;
    lats[1745]=-32.724076626687825;
    lats[1746]=-32.794375394564113;
    lats[1747]=-32.864674162437396;
    lats[1748]=-32.934972930307666;
    lats[1749]=-33.005271698174909;
    lats[1750]=-33.075570466039117;
    lats[1751]=-33.145869233900278;
    lats[1752]=-33.216168001758369;
    lats[1753]=-33.286466769613391;
    lats[1754]=-33.356765537465314;
    lats[1755]=-33.42706430531414;
    lats[1756]=-33.497363073159853;
    lats[1757]=-33.567661841002426;
    lats[1758]=-33.637960608841851;
    lats[1759]=-33.708259376678136;
    lats[1760]=-33.778558144511237;
    lats[1761]=-33.848856912341155;
    lats[1762]=-33.919155680167876;
    lats[1763]=-33.989454447991392;
    lats[1764]=-34.059753215811682;
    lats[1765]=-34.130051983628725;
    lats[1766]=-34.200350751442521;
    lats[1767]=-34.270649519253041;
    lats[1768]=-34.340948287060286;
    lats[1769]=-34.411247054864234;
    lats[1770]=-34.481545822664863;
    lats[1771]=-34.551844590462188;
    lats[1772]=-34.622143358256153;
    lats[1773]=-34.692442126046771;
    lats[1774]=-34.762740893834028;
    lats[1775]=-34.833039661617903;
    lats[1776]=-34.903338429398374;
    lats[1777]=-34.973637197175435;
    lats[1778]=-35.043935964949064;
    lats[1779]=-35.114234732719261;
    lats[1780]=-35.184533500486005;
    lats[1781]=-35.254832268249267;
    lats[1782]=-35.325131036009047;
    lats[1783]=-35.395429803765317;
    lats[1784]=-35.465728571518085;
    lats[1785]=-35.536027339267314;
    lats[1786]=-35.606326107012997;
    lats[1787]=-35.676624874755113;
    lats[1788]=-35.746923642493655;
    lats[1789]=-35.817222410228595;
    lats[1790]=-35.887521177959933;
    lats[1791]=-35.957819945687639;
    lats[1792]=-36.028118713411708;
    lats[1793]=-36.098417481132117;
    lats[1794]=-36.16871624884886;
    lats[1795]=-36.239015016561908;
    lats[1796]=-36.309313784271254;
    lats[1797]=-36.379612551976876;
    lats[1798]=-36.449911319678755;
    lats[1799]=-36.520210087376888;
    lats[1800]=-36.590508855071242;
    lats[1801]=-36.660807622761808;
    lats[1802]=-36.731106390448581;
    lats[1803]=-36.801405158131523;
    lats[1804]=-36.871703925810628;
    lats[1805]=-36.942002693485883;
    lats[1806]=-37.012301461157264;
    lats[1807]=-37.082600228824752;
    lats[1808]=-37.152898996488332;
    lats[1809]=-37.223197764147997;
    lats[1810]=-37.293496531803719;
    lats[1811]=-37.363795299455489;
    lats[1812]=-37.434094067103274;
    lats[1813]=-37.504392834747065;
    lats[1814]=-37.574691602386856;
    lats[1815]=-37.644990370022605;
    lats[1816]=-37.715289137654317;
    lats[1817]=-37.785587905281965;
    lats[1818]=-37.855886672905527;
    lats[1819]=-37.926185440524989;
    lats[1820]=-37.99648420814033;
    lats[1821]=-38.066782975751536;
    lats[1822]=-38.137081743358586;
    lats[1823]=-38.20738051096145;
    lats[1824]=-38.277679278560143;
    lats[1825]=-38.347978046154608;
    lats[1826]=-38.418276813744846;
    lats[1827]=-38.488575581330842;
    lats[1828]=-38.558874348912568;
    lats[1829]=-38.629173116490001;
    lats[1830]=-38.699471884063136;
    lats[1831]=-38.769770651631937;
    lats[1832]=-38.840069419196389;
    lats[1833]=-38.910368186756479;
    lats[1834]=-38.980666954312184;
    lats[1835]=-39.050965721863491;
    lats[1836]=-39.121264489410365;
    lats[1837]=-39.191563256952804;
    lats[1838]=-39.261862024490775;
    lats[1839]=-39.332160792024254;
    lats[1840]=-39.402459559553229;
    lats[1841]=-39.472758327077692;
    lats[1842]=-39.543057094597607;
    lats[1843]=-39.613355862112947;
    lats[1844]=-39.683654629623703;
    lats[1845]=-39.753953397129855;
    lats[1846]=-39.824252164631375;
    lats[1847]=-39.894550932128247;
    lats[1848]=-39.964849699620437;
    lats[1849]=-40.035148467107952;
    lats[1850]=-40.105447234590748;
    lats[1851]=-40.175746002068806;
    lats[1852]=-40.246044769542102;
    lats[1853]=-40.316343537010617;
    lats[1854]=-40.386642304474343;
    lats[1855]=-40.456941071933244;
    lats[1856]=-40.527239839387299;
    lats[1857]=-40.597538606836487;
    lats[1858]=-40.667837374280786;
    lats[1859]=-40.738136141720176;
    lats[1860]=-40.808434909154634;
    lats[1861]=-40.878733676584126;
    lats[1862]=-40.949032444008644;
    lats[1863]=-41.01933121142816;
    lats[1864]=-41.089629978842645;
    lats[1865]=-41.159928746252085;
    lats[1866]=-41.230227513656445;
    lats[1867]=-41.300526281055724;
    lats[1868]=-41.370825048449873;
    lats[1869]=-41.441123815838885;
    lats[1870]=-41.511422583222718;
    lats[1871]=-41.581721350601363;
    lats[1872]=-41.6520201179748;
    lats[1873]=-41.722318885343;
    lats[1874]=-41.792617652705921;
    lats[1875]=-41.862916420063563;
    lats[1876]=-41.933215187415882;
    lats[1877]=-42.003513954762873;
    lats[1878]=-42.073812722104492;
    lats[1879]=-42.144111489440725;
    lats[1880]=-42.214410256771551;
    lats[1881]=-42.284709024096927;
    lats[1882]=-42.355007791416853;
    lats[1883]=-42.425306558731272;
    lats[1884]=-42.495605326040177;
    lats[1885]=-42.565904093343548;
    lats[1886]=-42.63620286064134;
    lats[1887]=-42.706501627933541;
    lats[1888]=-42.776800395220121;
    lats[1889]=-42.847099162501053;
    lats[1890]=-42.917397929776307;
    lats[1891]=-42.987696697045862;
    lats[1892]=-43.057995464309691;
    lats[1893]=-43.128294231567757;
    lats[1894]=-43.19859299882004;
    lats[1895]=-43.26889176606651;
    lats[1896]=-43.339190533307139;
    lats[1897]=-43.409489300541907;
    lats[1898]=-43.479788067770777;
    lats[1899]=-43.550086834993728;
    lats[1900]=-43.620385602210717;
    lats[1901]=-43.690684369421732;
    lats[1902]=-43.760983136626741;
    lats[1903]=-43.831281903825705;
    lats[1904]=-43.9015806710186;
    lats[1905]=-43.971879438205391;
    lats[1906]=-44.042178205386072;
    lats[1907]=-44.112476972560586;
    lats[1908]=-44.182775739728925;
    lats[1909]=-44.253074506891046;
    lats[1910]=-44.323373274046915;
    lats[1911]=-44.39367204119651;
    lats[1912]=-44.463970808339802;
    lats[1913]=-44.534269575476756;
    lats[1914]=-44.604568342607337;
    lats[1915]=-44.674867109731515;
    lats[1916]=-44.745165876849271;
    lats[1917]=-44.81546464396056;
    lats[1918]=-44.885763411065362;
    lats[1919]=-44.956062178163634;
    lats[1920]=-45.026360945255341;
    lats[1921]=-45.096659712340461;
    lats[1922]=-45.166958479418959;
    lats[1923]=-45.237257246490813;
    lats[1924]=-45.30755601355596;
    lats[1925]=-45.377854780614399;
    lats[1926]=-45.448153547666081;
    lats[1927]=-45.51845231471097;
    lats[1928]=-45.588751081749038;
    lats[1929]=-45.659049848780263;
    lats[1930]=-45.729348615804589;
    lats[1931]=-45.799647382821995;
    lats[1932]=-45.869946149832437;
    lats[1933]=-45.94024491683588;
    lats[1934]=-46.01054368383231;
    lats[1935]=-46.080842450821663;
    lats[1936]=-46.151141217803925;
    lats[1937]=-46.221439984779053;
    lats[1938]=-46.291738751747012;
    lats[1939]=-46.362037518707766;
    lats[1940]=-46.432336285661272;
    lats[1941]=-46.502635052607502;
    lats[1942]=-46.572933819546414;
    lats[1943]=-46.643232586477971;
    lats[1944]=-46.713531353402139;
    lats[1945]=-46.783830120318882;
    lats[1946]=-46.85412888722815;
    lats[1947]=-46.924427654129929;
    lats[1948]=-46.994726421024154;
    lats[1949]=-47.065025187910805;
    lats[1950]=-47.13532395478984;
    lats[1951]=-47.205622721661214;
    lats[1952]=-47.275921488524894;
    lats[1953]=-47.346220255380835;
    lats[1954]=-47.416519022228997;
    lats[1955]=-47.486817789069342;
    lats[1956]=-47.557116555901828;
    lats[1957]=-47.627415322726435;
    lats[1958]=-47.697714089543084;
    lats[1959]=-47.76801285635176;
    lats[1960]=-47.838311623152421;
    lats[1961]=-47.908610389945018;
    lats[1962]=-47.978909156729507;
    lats[1963]=-48.049207923505868;
    lats[1964]=-48.119506690274015;
    lats[1965]=-48.189805457033941;
    lats[1966]=-48.260104223785596;
    lats[1967]=-48.330402990528938;
    lats[1968]=-48.400701757263917;
    lats[1969]=-48.47100052399049;
    lats[1970]=-48.541299290708608;
    lats[1971]=-48.611598057418242;
    lats[1972]=-48.681896824119335;
    lats[1973]=-48.752195590811837;
    lats[1974]=-48.822494357495721;
    lats[1975]=-48.892793124170929;
    lats[1976]=-48.963091890837418;
    lats[1977]=-49.03339065749514;
    lats[1978]=-49.103689424144044;
    lats[1979]=-49.173988190784094;
    lats[1980]=-49.244286957415234;
    lats[1981]=-49.314585724037435;
    lats[1982]=-49.384884490650613;
    lats[1983]=-49.455183257254745;
    lats[1984]=-49.525482023849783;
    lats[1985]=-49.595780790435676;
    lats[1986]=-49.66607955701236;
    lats[1987]=-49.736378323579807;
    lats[1988]=-49.80667709013796;
    lats[1989]=-49.876975856686762;
    lats[1990]=-49.947274623226157;
    lats[1991]=-50.017573389756123;
    lats[1992]=-50.087872156276575;
    lats[1993]=-50.158170922787484;
    lats[1994]=-50.228469689288779;
    lats[1995]=-50.298768455780426;
    lats[1996]=-50.369067222262359;
    lats[1997]=-50.439365988734544;
    lats[1998]=-50.509664755196901;
    lats[1999]=-50.579963521649397;
    lats[2000]=-50.650262288091959;
    lats[2001]=-50.720561054524559;
    lats[2002]=-50.790859820947119;
    lats[2003]=-50.86115858735959;
    lats[2004]=-50.931457353761914;
    lats[2005]=-51.001756120154049;
    lats[2006]=-51.072054886535909;
    lats[2007]=-51.14235365290746;
    lats[2008]=-51.21265241926865;
    lats[2009]=-51.282951185619417;
    lats[2010]=-51.353249951959683;
    lats[2011]=-51.423548718289396;
    lats[2012]=-51.493847484608516;
    lats[2013]=-51.56414625091697;
    lats[2014]=-51.634445017214695;
    lats[2015]=-51.704743783501634;
    lats[2016]=-51.775042549777737;
    lats[2017]=-51.845341316042933;
    lats[2018]=-51.915640082297152;
    lats[2019]=-51.985938848540336;
    lats[2020]=-52.056237614772435;
    lats[2021]=-52.126536380993372;
    lats[2022]=-52.196835147203096;
    lats[2023]=-52.26713391340153;
    lats[2024]=-52.337432679588609;
    lats[2025]=-52.407731445764284;
    lats[2026]=-52.478030211928477;
    lats[2027]=-52.548328978081123;
    lats[2028]=-52.618627744222159;
    lats[2029]=-52.688926510351514;
    lats[2030]=-52.759225276469131;
    lats[2031]=-52.829524042574917;
    lats[2032]=-52.899822808668837;
    lats[2033]=-52.970121574750792;
    lats[2034]=-53.040420340820731;
    lats[2035]=-53.110719106878584;
    lats[2036]=-53.181017872924265;
    lats[2037]=-53.251316638957725;
    lats[2038]=-53.321615404978871;
    lats[2039]=-53.391914170987633;
    lats[2040]=-53.462212936983953;
    lats[2041]=-53.53251170296776;
    lats[2042]=-53.602810468938962;
    lats[2043]=-53.673109234897495;
    lats[2044]=-53.743408000843282;
    lats[2045]=-53.813706766776235;
    lats[2046]=-53.884005532696307;
    lats[2047]=-53.954304298603383;
    lats[2048]=-54.024603064497434;
    lats[2049]=-54.094901830378333;
    lats[2050]=-54.165200596246031;
    lats[2051]=-54.235499362100448;
    lats[2052]=-54.305798127941479;
    lats[2053]=-54.376096893769081;
    lats[2054]=-54.446395659583146;
    lats[2055]=-54.516694425383605;
    lats[2056]=-54.586993191170357;
    lats[2057]=-54.657291956943347;
    lats[2058]=-54.727590722702473;
    lats[2059]=-54.797889488447652;
    lats[2060]=-54.868188254178797;
    lats[2061]=-54.938487019895831;
    lats[2062]=-55.008785785598668;
    lats[2063]=-55.07908455128721;
    lats[2064]=-55.149383316961377;
    lats[2065]=-55.219682082621084;
    lats[2066]=-55.289980848266232;
    lats[2067]=-55.360279613896743;
    lats[2068]=-55.430578379512511;
    lats[2069]=-55.500877145113449;
    lats[2070]=-55.571175910699488;
    lats[2071]=-55.641474676270505;
    lats[2072]=-55.711773441826416;
    lats[2073]=-55.782072207367136;
    lats[2074]=-55.852370972892551;
    lats[2075]=-55.922669738402583;
    lats[2076]=-55.992968503897131;
    lats[2077]=-56.063267269376091;
    lats[2078]=-56.133566034839362;
    lats[2079]=-56.203864800286865;
    lats[2080]=-56.274163565718467;
    lats[2081]=-56.34446233113411;
    lats[2082]=-56.41476109653366;
    lats[2083]=-56.485059861917016;
    lats[2084]=-56.555358627284086;
    lats[2085]=-56.625657392634771;
    lats[2086]=-56.695956157968951;
    lats[2087]=-56.766254923286517;
    lats[2088]=-56.836553688587379;
    lats[2089]=-56.90685245387143;
    lats[2090]=-56.977151219138541;
    lats[2091]=-57.047449984388614;
    lats[2092]=-57.117748749621541;
    lats[2093]=-57.188047514837208;
    lats[2094]=-57.258346280035504;
    lats[2095]=-57.328645045216312;
    lats[2096]=-57.398943810379521;
    lats[2097]=-57.469242575525016;
    lats[2098]=-57.539541340652676;
    lats[2099]=-57.60984010576238;
    lats[2100]=-57.680138870854037;
    lats[2101]=-57.75043763592749;
    lats[2102]=-57.820736400982646;
    lats[2103]=-57.891035166019364;
    lats[2104]=-57.961333931037537;
    lats[2105]=-58.031632696037022;
    lats[2106]=-58.101931461017728;
    lats[2107]=-58.172230225979497;
    lats[2108]=-58.242528990922203;
    lats[2109]=-58.312827755845746;
    lats[2110]=-58.383126520749968;
    lats[2111]=-58.453425285634758;
    lats[2112]=-58.523724050499972;
    lats[2113]=-58.594022815345468;
    lats[2114]=-58.664321580171141;
    lats[2115]=-58.73462034497684;
    lats[2116]=-58.804919109762423;
    lats[2117]=-58.875217874527763;
    lats[2118]=-58.945516639272725;
    lats[2119]=-59.015815403997145;
    lats[2120]=-59.086114168700909;
    lats[2121]=-59.156412933383855;
    lats[2122]=-59.226711698045854;
    lats[2123]=-59.29701046268675;
    lats[2124]=-59.3673092273064;
    lats[2125]=-59.43760799190467;
    lats[2126]=-59.507906756481383;
    lats[2127]=-59.578205521036402;
    lats[2128]=-59.64850428556958;
    lats[2129]=-59.718803050080759;
    lats[2130]=-59.78910181456979;
    lats[2131]=-59.859400579036503;
    lats[2132]=-59.929699343480763;
    lats[2133]=-59.999998107902378;
    lats[2134]=-60.070296872301235;
    lats[2135]=-60.140595636677112;
    lats[2136]=-60.21089440102989;
    lats[2137]=-60.28119316535939;
    lats[2138]=-60.35149192966545;
    lats[2139]=-60.421790693947884;
    lats[2140]=-60.492089458206543;
    lats[2141]=-60.562388222441243;
    lats[2142]=-60.632686986651805;
    lats[2143]=-60.702985750838074;
    lats[2144]=-60.773284514999872;
    lats[2145]=-60.843583279137007;
    lats[2146]=-60.913882043249295;
    lats[2147]=-60.984180807336578;
    lats[2148]=-61.054479571398652;
    lats[2149]=-61.124778335435344;
    lats[2150]=-61.195077099446451;
    lats[2151]=-61.265375863431785;
    lats[2152]=-61.335674627391185;
    lats[2153]=-61.405973391324409;
    lats[2154]=-61.476272155231321;
    lats[2155]=-61.546570919111666;
    lats[2156]=-61.616869682965287;
    lats[2157]=-61.687168446791986;
    lats[2158]=-61.757467210591535;
    lats[2159]=-61.827765974363729;
    lats[2160]=-61.898064738108381;
    lats[2161]=-61.968363501825259;
    lats[2162]=-62.038662265514176;
    lats[2163]=-62.108961029174914;
    lats[2164]=-62.179259792807258;
    lats[2165]=-62.249558556410982;
    lats[2166]=-62.319857319985871;
    lats[2167]=-62.3901560835317;
    lats[2168]=-62.460454847048261;
    lats[2169]=-62.530753610535321;
    lats[2170]=-62.60105237399263;
    lats[2171]=-62.67135113741999;
    lats[2172]=-62.741649900817137;
    lats[2173]=-62.811948664183866;
    lats[2174]=-62.882247427519928;
    lats[2175]=-62.952546190825068;
    lats[2176]=-63.022844954099064;
    lats[2177]=-63.093143717341647;
    lats[2178]=-63.163442480552604;
    lats[2179]=-63.23374124373165;
    lats[2180]=-63.304040006878537;
    lats[2181]=-63.374338769993031;
    lats[2182]=-63.444637533074854;
    lats[2183]=-63.514936296123757;
    lats[2184]=-63.585235059139464;
    lats[2185]=-63.655533822121711;
    lats[2186]=-63.725832585070251;
    lats[2187]=-63.796131347984762;
    lats[2188]=-63.866430110865004;
    lats[2189]=-63.93672887371072;
    lats[2190]=-64.00702763652157;
    lats[2191]=-64.07732639929732;
    lats[2192]=-64.147625162037642;
    lats[2193]=-64.21792392474228;
    lats[2194]=-64.288222687410922;
    lats[2195]=-64.358521450043284;
    lats[2196]=-64.428820212639039;
    lats[2197]=-64.499118975197902;
    lats[2198]=-64.569417737719576;
    lats[2199]=-64.639716500203733;
    lats[2200]=-64.710015262650074;
    lats[2201]=-64.780314025058246;
    lats[2202]=-64.850612787427963;
    lats[2203]=-64.920911549758912;
    lats[2204]=-64.991210312050711;
    lats[2205]=-65.061509074303089;
    lats[2206]=-65.131807836515677;
    lats[2207]=-65.202106598688133;
    lats[2208]=-65.272405360820116;
    lats[2209]=-65.342704122911286;
    lats[2210]=-65.413002884961315;
    lats[2211]=-65.483301646969792;
    lats[2212]=-65.553600408936404;
    lats[2213]=-65.623899170860767;
    lats[2214]=-65.694197932742526;
    lats[2215]=-65.764496694581283;
    lats[2216]=-65.834795456376696;
    lats[2217]=-65.905094218128355;
    lats[2218]=-65.975392979835888;
    lats[2219]=-66.045691741498899;
    lats[2220]=-66.115990503117033;
    lats[2221]=-66.186289264689833;
    lats[2222]=-66.256588026216932;
    lats[2223]=-66.326886787697887;
    lats[2224]=-66.397185549132331;
    lats[2225]=-66.467484310519808;
    lats[2226]=-66.537783071859891;
    lats[2227]=-66.608081833152212;
    lats[2228]=-66.678380594396273;
    lats[2229]=-66.748679355591662;
    lats[2230]=-66.818978116737924;
    lats[2231]=-66.889276877834618;
    lats[2232]=-66.95957563888129;
    lats[2233]=-67.029874399877471;
    lats[2234]=-67.100173160822706;
    lats[2235]=-67.170471921716526;
    lats[2236]=-67.240770682558434;
    lats[2237]=-67.311069443347961;
    lats[2238]=-67.381368204084609;
    lats[2239]=-67.451666964767895;
    lats[2240]=-67.521965725397308;
    lats[2241]=-67.592264485972336;
    lats[2242]=-67.662563246492482;
    lats[2243]=-67.732862006957205;
    lats[2244]=-67.803160767365966;
    lats[2245]=-67.873459527718282;
    lats[2246]=-67.943758288013555;
    lats[2247]=-68.014057048251274;
    lats[2248]=-68.084355808430871;
    lats[2249]=-68.154654568551791;
    lats[2250]=-68.224953328613438;
    lats[2251]=-68.295252088615257;
    lats[2252]=-68.365550848556666;
    lats[2253]=-68.435849608437067;
    lats[2254]=-68.506148368255865;
    lats[2255]=-68.576447128012447;
    lats[2256]=-68.646745887706189;
    lats[2257]=-68.717044647336493;
    lats[2258]=-68.787343406902693;
    lats[2259]=-68.85764216640419;
    lats[2260]=-68.927940925840304;
    lats[2261]=-68.998239685210365;
    lats[2262]=-69.068538444513763;
    lats[2263]=-69.138837203749759;
    lats[2264]=-69.209135962917699;
    lats[2265]=-69.279434722016902;
    lats[2266]=-69.349733481046613;
    lats[2267]=-69.420032240006194;
    lats[2268]=-69.490330998894862;
    lats[2269]=-69.560629757711908;
    lats[2270]=-69.630928516456592;
    lats[2271]=-69.701227275128161;
    lats[2272]=-69.771526033725834;
    lats[2273]=-69.841824792248843;
    lats[2274]=-69.912123550696421;
    lats[2275]=-69.982422309067744;
    lats[2276]=-70.052721067362043;
    lats[2277]=-70.123019825578467;
    lats[2278]=-70.193318583716191;
    lats[2279]=-70.263617341774406;
    lats[2280]=-70.333916099752187;
    lats[2281]=-70.404214857648739;
    lats[2282]=-70.474513615463138;
    lats[2283]=-70.544812373194532;
    lats[2284]=-70.615111130841967;
    lats[2285]=-70.685409888404578;
    lats[2286]=-70.755708645881384;
    lats[2287]=-70.826007403271475;
    lats[2288]=-70.896306160573886;
    lats[2289]=-70.966604917787635;
    lats[2290]=-71.036903674911756;
    lats[2291]=-71.107202431945211;
    lats[2292]=-71.177501188887007;
    lats[2293]=-71.247799945736105;
    lats[2294]=-71.318098702491469;
    lats[2295]=-71.388397459152031;
    lats[2296]=-71.458696215716685;
    lats[2297]=-71.528994972184378;
    lats[2298]=-71.599293728553988;
    lats[2299]=-71.669592484824364;
    lats[2300]=-71.739891240994368;
    lats[2301]=-71.810189997062835;
    lats[2302]=-71.880488753028587;
    lats[2303]=-71.950787508890414;
    lats[2304]=-72.02108626464711;
    lats[2305]=-72.091385020297409;
    lats[2306]=-72.161683775840089;
    lats[2307]=-72.231982531273843;
    lats[2308]=-72.302281286597392;
    lats[2309]=-72.3725800418094;
    lats[2310]=-72.442878796908545;
    lats[2311]=-72.513177551893421;
    lats[2312]=-72.583476306762691;
    lats[2313]=-72.653775061514935;
    lats[2314]=-72.724073816148703;
    lats[2315]=-72.794372570662574;
    lats[2316]=-72.864671325055056;
    lats[2317]=-72.934970079324657;
    lats[2318]=-73.005268833469799;
    lats[2319]=-73.075567587489019;
    lats[2320]=-73.145866341380668;
    lats[2321]=-73.216165095143182;
    lats[2322]=-73.2864638487749;
    lats[2323]=-73.356762602274188;
    lats[2324]=-73.427061355639339;
    lats[2325]=-73.497360108868662;
    lats[2326]=-73.567658861960396;
    lats[2327]=-73.637957614912779;
    lats[2328]=-73.70825636772399;
    lats[2329]=-73.778555120392184;
    lats[2330]=-73.848853872915541;
    lats[2331]=-73.919152625292114;
    lats[2332]=-73.98945137751997;
    lats[2333]=-74.059750129597163;
    lats[2334]=-74.13004888152166;
    lats[2335]=-74.200347633291472;
    lats[2336]=-74.270646384904481;
    lats[2337]=-74.340945136358584;
    lats[2338]=-74.411243887651622;
    lats[2339]=-74.481542638781434;
    lats[2340]=-74.551841389745761;
    lats[2341]=-74.622140140542356;
    lats[2342]=-74.692438891168877;
    lats[2343]=-74.762737641622991;
    lats[2344]=-74.833036391902269;
    lats[2345]=-74.903335142004323;
    lats[2346]=-74.973633891926625;
    lats[2347]=-75.043932641666672;
    lats[2348]=-75.114231391221821;
    lats[2349]=-75.184530140589501;
    lats[2350]=-75.254828889766983;
    lats[2351]=-75.325127638751567;
    lats[2352]=-75.395426387540439;
    lats[2353]=-75.465725136130786;
    lats[2354]=-75.536023884519707;
    lats[2355]=-75.60632263270422;
    lats[2356]=-75.67662138068134;
    lats[2357]=-75.746920128447996;
    lats[2358]=-75.81721887600105;
    lats[2359]=-75.887517623337317;
    lats[2360]=-75.957816370453543;
    lats[2361]=-76.028115117346374;
    lats[2362]=-76.098413864012443;
    lats[2363]=-76.16871261044831;
    lats[2364]=-76.239011356650423;
    lats[2365]=-76.3093101026152;
    lats[2366]=-76.379608848338933;
    lats[2367]=-76.449907593817869;
    lats[2368]=-76.520206339048215;
    lats[2369]=-76.59050508402602;
    lats[2370]=-76.660803828747362;
    lats[2371]=-76.731102573208048;
    lats[2372]=-76.801401317404;
    lats[2373]=-76.871700061330955;
    lats[2374]=-76.941998804984564;
    lats[2375]=-77.012297548360323;
    lats[2376]=-77.082596291453768;
    lats[2377]=-77.15289503426024;
    lats[2378]=-77.22319377677502;
    lats[2379]=-77.293492518993247;
    lats[2380]=-77.363791260909963;
    lats[2381]=-77.434090002520122;
    lats[2382]=-77.504388743818524;
    lats[2383]=-77.574687484799924;
    lats[2384]=-77.644986225458879;
    lats[2385]=-77.71528496578982;
    lats[2386]=-77.785583705787161;
    lats[2387]=-77.855882445445019;
    lats[2388]=-77.926181184757539;
    lats[2389]=-77.996479923718596;
    lats[2390]=-78.066778662322022;
    lats[2391]=-78.137077400561424;
    lats[2392]=-78.207376138430348;
    lats[2393]=-78.277674875922045;
    lats[2394]=-78.347973613029708;
    lats[2395]=-78.418272349746417;
    lats[2396]=-78.488571086064923;
    lats[2397]=-78.558869821977908;
    lats[2398]=-78.629168557477882;
    lats[2399]=-78.699467292557102;
    lats[2400]=-78.769766027207638;
    lats[2401]=-78.840064761421445;
    lats[2402]=-78.910363495190211;
    lats[2403]=-78.980662228505423;
    lats[2404]=-79.050960961358285;
    lats[2405]=-79.121259693739859;
    lats[2406]=-79.191558425640977;
    lats[2407]=-79.261857157052191;
    lats[2408]=-79.332155887963822;
    lats[2409]=-79.402454618365894;
    lats[2410]=-79.472753348248219;
    lats[2411]=-79.543052077600308;
    lats[2412]=-79.61335080641139;
    lats[2413]=-79.683649534670437;
    lats[2414]=-79.753948262366038;
    lats[2415]=-79.824246989486554;
    lats[2416]=-79.894545716019948;
    lats[2417]=-79.9648444419539;
    lats[2418]=-80.035143167275749;
    lats[2419]=-80.105441891972376;
    lats[2420]=-80.175740616030438;
    lats[2421]=-80.246039339436052;
    lats[2422]=-80.316338062175078;
    lats[2423]=-80.386636784232863;
    lats[2424]=-80.456935505594302;
    lats[2425]=-80.527234226243991;
    lats[2426]=-80.59753294616587;
    lats[2427]=-80.667831665343556;
    lats[2428]=-80.73813038376008;
    lats[2429]=-80.808429101397948;
    lats[2430]=-80.878727818239184;
    lats[2431]=-80.949026534265244;
    lats[2432]=-81.019325249456955;
    lats[2433]=-81.089623963794551;
    lats[2434]=-81.159922677257711;
    lats[2435]=-81.230221389825374;
    lats[2436]=-81.300520101475826;
    lats[2437]=-81.370818812186627;
    lats[2438]=-81.441117521934686;
    lats[2439]=-81.511416230696042;
    lats[2440]=-81.581714938445955;
    lats[2441]=-81.652013645158945;
    lats[2442]=-81.722312350808508;
    lats[2443]=-81.792611055367345;
    lats[2444]=-81.862909758807191;
    lats[2445]=-81.933208461098829;
    lats[2446]=-82.003507162211946;
    lats[2447]=-82.073805862115165;
    lats[2448]=-82.144104560776;
    lats[2449]=-82.214403258160871;
    lats[2450]=-82.284701954234833;
    lats[2451]=-82.355000648961692;
    lats[2452]=-82.425299342304029;
    lats[2453]=-82.495598034222837;
    lats[2454]=-82.56589672467787;
    lats[2455]=-82.63619541362705;
    lats[2456]=-82.706494101026948;
    lats[2457]=-82.77679278683226;
    lats[2458]=-82.84709147099602;
    lats[2459]=-82.917390153469313;
    lats[2460]=-82.987688834201322;
    lats[2461]=-83.057987513139125;
    lats[2462]=-83.128286190227698;
    lats[2463]=-83.198584865409657;
    lats[2464]=-83.268883538625232;
    lats[2465]=-83.339182209812321;
    lats[2466]=-83.409480878905782;
    lats[2467]=-83.479779545838113;
    lats[2468]=-83.550078210538487;
    lats[2469]=-83.620376872933264;
    lats[2470]=-83.690675532945292;
    lats[2471]=-83.760974190494011;
    lats[2472]=-83.831272845495249;
    lats[2473]=-83.901571497860914;
    lats[2474]=-83.971870147498763;
    lats[2475]=-84.042168794312317;
    lats[2476]=-84.112467438200326;
    lats[2477]=-84.18276607905679;
    lats[2478]=-84.253064716770425;
    lats[2479]=-84.323363351224444;
    lats[2480]=-84.393661982296322;
    lats[2481]=-84.463960609857125;
    lats[2482]=-84.534259233771479;
    lats[2483]=-84.604557853896708;
    lats[2484]=-84.674856470082915;
    lats[2485]=-84.745155082171991;
    lats[2486]=-84.81545368999717;
    lats[2487]=-84.885752293382765;
    lats[2488]=-84.95605089214304;
    lats[2489]=-85.026349486081983;
    lats[2490]=-85.09664807499216;
    lats[2491]=-85.16694665865414;
    lats[2492]=-85.237245236835548;
    lats[2493]=-85.307543809290152;
    lats[2494]=-85.377842375756586;
    lats[2495]=-85.448140935957483;
    lats[2496]=-85.518439489597966;
    lats[2497]=-85.588738036364362;
    lats[2498]=-85.659036575922883;
    lats[2499]=-85.729335107917464;
    lats[2500]=-85.799633631968391;
    lats[2501]=-85.869932147670127;
    lats[2502]=-85.940230654588888;
    lats[2503]=-86.010529152260403;
    lats[2504]=-86.080827640187209;
    lats[2505]=-86.151126117835304;
    lats[2506]=-86.221424584631109;
    lats[2507]=-86.291723039957418;
    lats[2508]=-86.362021483149363;
    lats[2509]=-86.432319913489792;
    lats[2510]=-86.502618330203831;
    lats[2511]=-86.572916732453024;
    lats[2512]=-86.643215119328573;
    lats[2513]=-86.713513489844246;
    lats[2514]=-86.783811842927179;
    lats[2515]=-86.854110177408927;
    lats[2516]=-86.924408492014166;
    lats[2517]=-86.994706785348129;
    lats[2518]=-87.065005055882821;
    lats[2519]=-87.135303301939786;
    lats[2520]=-87.205601521672108;
    lats[2521]=-87.275899713041966;
    lats[2522]=-87.346197873795816;
    lats[2523]=-87.416496001434894;
    lats[2524]=-87.486794093180748;
    lats[2525]=-87.557092145935584;
    lats[2526]=-87.627390156234085;
    lats[2527]=-87.697688120188062;
    lats[2528]=-87.767986033419561;
    lats[2529]=-87.838283890981543;
    lats[2530]=-87.908581687261687;
    lats[2531]=-87.978879415867283;
    lats[2532]=-88.049177069484486;
    lats[2533]=-88.119474639706425;
    lats[2534]=-88.189772116820762;
    lats[2535]=-88.26006948954614;
    lats[2536]=-88.330366744702559;
    lats[2537]=-88.40066386679355;
    lats[2538]=-88.470960837474877;
    lats[2539]=-88.541257634868515;
    lats[2540]=-88.611554232668382;
    lats[2541]=-88.681850598961759;
    lats[2542]=-88.752146694650691;
    lats[2543]=-88.822442471310097;
    lats[2544]=-88.892737868230952;
    lats[2545]=-88.96303280826325;
    lats[2546]=-89.033327191845927;
    lats[2547]=-89.103620888238879;
    lats[2548]=-89.173913722284126;
    lats[2549]=-89.24420545380525;
    lats[2550]=-89.314495744374256;
    lats[2551]=-89.3847841013921;
    lats[2552]=-89.45506977912261;
    lats[2553]=-89.525351592371393;
    lats[2554]=-89.595627537554492;
    lats[2555]=-89.6658939412157;
    lats[2556]=-89.736143271609578;
    lats[2557]=-89.806357319542244;
    lats[2558]=-89.876478353332288;
    lats[2559]=-89.946187715665616;
    return GRIB_SUCCESS;
}
static int get_precomputed_latitudes_N640(double* lats)
{
    lats[0]    = 89.892396445590066;
    lats[1]    = 89.753004943174034;
    lats[2]    = 89.612790258599077;
    lats[3]    = 89.472389582061126;
    lats[4]    = 89.331918354381827;
    lats[5]    = 89.191412986832432;
    lats[6]    = 89.050888539966436;
    lats[7]    = 88.91035235926023;
    lats[8]    = 88.76980845110036;
    lats[9]    = 88.629259185411627;
    lats[10]   = 88.488706053376362;
    lats[11]   = 88.348150039999084;
    lats[12]   = 88.207591822004105;
    lats[13]   = 88.067031879650926;
    lats[14]   = 87.926470563186442;
    lats[15]   = 87.785908134040668;
    lats[16]   = 87.645344791295628;
    lats[17]   = 87.504780689222315;
    lats[18]   = 87.364215949214667;
    lats[19]   = 87.223650668104085;
    lats[20]   = 87.083084924070917;
    lats[21]   = 86.942518780928566;
    lats[22]   = 86.801952291278369;
    lats[23]   = 86.661385498868242;
    lats[24]   = 86.520818440379529;
    lats[25]   = 86.380251146798656;
    lats[26]   = 86.239683644481104;
    lats[27]   = 86.0991159559849;
    lats[28]   = 85.958548100730781;
    lats[29]   = 85.817980095529578;
    lats[30]   = 85.677411955006008;
    lats[31]   = 85.536843691942948;
    lats[32]   = 85.396275317562669;
    lats[33]   = 85.255706841757572;
    lats[34]   = 85.115138273281829;
    lats[35]   = 84.974569619910426;
    lats[36]   = 84.834000888572191;
    lats[37]   = 84.693432085462035;
    lats[38]   = 84.552863216135577;
    lats[39]   = 84.412294285589354;
    lats[40]   = 84.271725298329656;
    lats[41]   = 84.131156258431133;
    lats[42]   = 83.990587169587158;
    lats[43]   = 83.850018035153667;
    lats[44]   = 83.709448858186462;
    lats[45]   = 83.568879641474325;
    lats[46]   = 83.428310387567549;
    lats[47]   = 83.287741098802584;
    lats[48]   = 83.147171777324388;
    lats[49]   = 83.006602425105484;
    lats[50]   = 82.866033043962815;
    lats[51]   = 82.725463635573107;
    lats[52]   = 82.584894201485696;
    lats[53]   = 82.444324743134914;
    lats[54]   = 82.303755261850071;
    lats[55]   = 82.163185758865239;
    lats[56]   = 82.022616235327504;
    lats[57]   = 81.882046692304485;
    lats[58]   = 81.741477130791196;
    lats[59]   = 81.600907551715878;
    lats[60]   = 81.460337955945846;
    lats[61]   = 81.319768344292086;
    lats[62]   = 81.179198717514012;
    lats[63]   = 81.038629076323318;
    lats[64]   = 80.898059421387785;
    lats[65]   = 80.757489753334553;
    lats[66]   = 80.616920072753146;
    lats[67]   = 80.47635038019834;
    lats[68]   = 80.335780676192584;
    lats[69]   = 80.195210961228469;
    lats[70]   = 80.054641235770603;
    lats[71]   = 79.914071500257819;
    lats[72]   = 79.773501755104689;
    lats[73]   = 79.632932000703448;
    lats[74]   = 79.492362237425226;
    lats[75]   = 79.351792465621628;
    lats[76]   = 79.211222685625927;
    lats[77]   = 79.070652897754229;
    lats[78]   = 78.930083102306568;
    lats[79]   = 78.789513299567957;
    lats[80]   = 78.648943489809355;
    lats[81]   = 78.508373673288318;
    lats[82]   = 78.367803850250056;
    lats[83]   = 78.227234020928066;
    lats[84]   = 78.086664185544819;
    lats[85]   = 77.946094344312371;
    lats[86]   = 77.805524497433041;
    lats[87]   = 77.664954645099883;
    lats[88]   = 77.524384787497311;
    lats[89]   = 77.383814924801513;
    lats[90]   = 77.243245057180829;
    lats[91]   = 77.102675184796354;
    lats[92]   = 76.962105307802219;
    lats[93]   = 76.821535426345932;
    lats[94]   = 76.680965540568806;
    lats[95]   = 76.540395650606285;
    lats[96]   = 76.399825756588143;
    lats[97]   = 76.259255858638895;
    lats[98]   = 76.118685956877997;
    lats[99]   = 75.978116051420102;
    lats[100]  = 75.837546142375359;
    lats[101]  = 75.69697622984954;
    lats[102]  = 75.556406313944308;
    lats[103]  = 75.41583639475742;
    lats[104]  = 75.275266472382896;
    lats[105]  = 75.134696546911186;
    lats[106]  = 74.994126618429377;
    lats[107]  = 74.853556687021296;
    lats[108]  = 74.712986752767719;
    lats[109]  = 74.57241681574645;
    lats[110]  = 74.431846876032495;
    lats[111]  = 74.291276933698185;
    lats[112]  = 74.150706988813226;
    lats[113]  = 74.010137041445006;
    lats[114]  = 73.869567091658411;
    lats[115]  = 73.728997139516167;
    lats[116]  = 73.588427185078871;
    lats[117]  = 73.447857228405013;
    lats[118]  = 73.307287269551111;
    lats[119]  = 73.166717308571819;
    lats[120]  = 73.026147345520002;
    lats[121]  = 72.885577380446747;
    lats[122]  = 72.745007413401481;
    lats[123]  = 72.604437444432065;
    lats[124]  = 72.463867473584784;
    lats[125]  = 72.323297500904502;
    lats[126]  = 72.182727526434604;
    lats[127]  = 72.042157550217183;
    lats[128]  = 71.901587572292982;
    lats[129]  = 71.761017592701492;
    lats[130]  = 71.620447611481026;
    lats[131]  = 71.47987762866866;
    lats[132]  = 71.339307644300462;
    lats[133]  = 71.198737658411332;
    lats[134]  = 71.058167671035164;
    lats[135]  = 70.917597682204899;
    lats[136]  = 70.777027691952398;
    lats[137]  = 70.636457700308753;
    lats[138]  = 70.495887707304007;
    lats[139]  = 70.355317712967462;
    lats[140]  = 70.214747717327526;
    lats[141]  = 70.074177720411782;
    lats[142]  = 69.933607722247146;
    lats[143]  = 69.793037722859665;
    lats[144]  = 69.65246772227475;
    lats[145]  = 69.511897720517084;
    lats[146]  = 69.37132771761064;
    lats[147]  = 69.230757713578825;
    lats[148]  = 69.090187708444333;
    lats[149]  = 68.949617702229318;
    lats[150]  = 68.809047694955296;
    lats[151]  = 68.668477686643286;
    lats[152]  = 68.52790767731365;
    lats[153]  = 68.387337666986312;
    lats[154]  = 68.246767655680657;
    lats[155]  = 68.106197643415527;
    lats[156]  = 67.965627630209354;
    lats[157]  = 67.825057616080073;
    lats[158]  = 67.684487601045149;
    lats[159]  = 67.543917585121662;
    lats[160]  = 67.403347568326168;
    lats[161]  = 67.262777550674912;
    lats[162]  = 67.122207532183722;
    lats[163]  = 66.981637512867991;
    lats[164]  = 66.841067492742795;
    lats[165]  = 66.700497471822814;
    lats[166]  = 66.559927450122359;
    lats[167]  = 66.41935742765547;
    lats[168]  = 66.278787404435761;
    lats[169]  = 66.138217380476604;
    lats[170]  = 65.997647355791017;
    lats[171]  = 65.85707733039176;
    lats[172]  = 65.716507304291198;
    lats[173]  = 65.575937277501538;
    lats[174]  = 65.435367250034616;
    lats[175]  = 65.294797221902016;
    lats[176]  = 65.154227193115119;
    lats[177]  = 65.013657163684968;
    lats[178]  = 64.873087133622406;
    lats[179]  = 64.732517102938033;
    lats[180]  = 64.591947071642196;
    lats[181]  = 64.451377039745026;
    lats[182]  = 64.310807007256443;
    lats[183]  = 64.170236974186125;
    lats[184]  = 64.029666940543564;
    lats[185]  = 63.889096906338061;
    lats[186]  = 63.748526871578648;
    lats[187]  = 63.607956836274255;
    lats[188]  = 63.467386800433559;
    lats[189]  = 63.326816764065093;
    lats[190]  = 63.186246727177178;
    lats[191]  = 63.045676689778013;
    lats[192]  = 62.905106651875542;
    lats[193]  = 62.764536613477638;
    lats[194]  = 62.62396657459194;
    lats[195]  = 62.483396535225978;
    lats[196]  = 62.342826495387122;
    lats[197]  = 62.202256455082583;
    lats[198]  = 62.061686414319418;
    lats[199]  = 61.921116373104539;
    lats[200]  = 61.780546331444761;
    lats[201]  = 61.639976289346727;
    lats[202]  = 61.499406246816953;
    lats[203]  = 61.358836203861841;
    lats[204]  = 61.21826616048768;
    lats[205]  = 61.077696116700601;
    lats[206]  = 60.937126072506608;
    lats[207]  = 60.796556027911663;
    lats[208]  = 60.655985982921543;
    lats[209]  = 60.515415937541938;
    lats[210]  = 60.374845891778421;
    lats[211]  = 60.234275845636503;
    lats[212]  = 60.093705799121537;
    lats[213]  = 59.953135752238794;
    lats[214]  = 59.812565704993467;
    lats[215]  = 59.671995657390596;
    lats[216]  = 59.531425609435225;
    lats[217]  = 59.390855561132213;
    lats[218]  = 59.250285512486386;
    lats[219]  = 59.10971546350244;
    lats[220]  = 58.96914541418505;
    lats[221]  = 58.828575364538722;
    lats[222]  = 58.688005314567938;
    lats[223]  = 58.547435264277105;
    lats[224]  = 58.406865213670514;
    lats[225]  = 58.266295162752428;
    lats[226]  = 58.125725111526968;
    lats[227]  = 57.985155059998249;
    lats[228]  = 57.844585008170284;
    lats[229]  = 57.704014956047033;
    lats[230]  = 57.563444903632337;
    lats[231]  = 57.422874850930043;
    lats[232]  = 57.282304797943887;
    lats[233]  = 57.141734744677549;
    lats[234]  = 57.001164691134662;
    lats[235]  = 56.860594637318769;
    lats[236]  = 56.720024583233375;
    lats[237]  = 56.579454528881925;
    lats[238]  = 56.438884474267795;
    lats[239]  = 56.29831441939433;
    lats[240]  = 56.157744364264779;
    lats[241]  = 56.017174308882367;
    lats[242]  = 55.876604253250278;
    lats[243]  = 55.736034197371588;
    lats[244]  = 55.595464141249401;
    lats[245]  = 55.45489408488671;
    lats[246]  = 55.314324028286471;
    lats[247]  = 55.173753971451625;
    lats[248]  = 55.033183914385013;
    lats[249]  = 54.892613857089486;
    lats[250]  = 54.752043799567822;
    lats[251]  = 54.611473741822735;
    lats[252]  = 54.470903683856939;
    lats[253]  = 54.330333625673063;
    lats[254]  = 54.189763567273758;
    lats[255]  = 54.049193508661538;
    lats[256]  = 53.90862344983897;
    lats[257]  = 53.768053390808532;
    lats[258]  = 53.627483331572677;
    lats[259]  = 53.486913272133812;
    lats[260]  = 53.346343212494332;
    lats[261]  = 53.205773152656562;
    lats[262]  = 53.065203092622802;
    lats[263]  = 52.924633032395342;
    lats[264]  = 52.784062971976404;
    lats[265]  = 52.643492911368206;
    lats[266]  = 52.502922850572908;
    lats[267]  = 52.362352789592649;
    lats[268]  = 52.221782728429538;
    lats[269]  = 52.081212667085637;
    lats[270]  = 51.940642605563028;
    lats[271]  = 51.800072543863692;
    lats[272]  = 51.659502481989627;
    lats[273]  = 51.518932419942786;
    lats[274]  = 51.378362357725095;
    lats[275]  = 51.237792295338465;
    lats[276]  = 51.097222232784773;
    lats[277]  = 50.956652170065858;
    lats[278]  = 50.81608210718354;
    lats[279]  = 50.675512044139623;
    lats[280]  = 50.534941980935862;
    lats[281]  = 50.39437191757402;
    lats[282]  = 50.253801854055808;
    lats[283]  = 50.113231790382912;
    lats[284]  = 49.972661726557028;
    lats[285]  = 49.832091662579785;
    lats[286]  = 49.691521598452823;
    lats[287]  = 49.550951534177734;
    lats[288]  = 49.410381469756118;
    lats[289]  = 49.269811405189529;
    lats[290]  = 49.129241340479489;
    lats[291]  = 48.988671275627539;
    lats[292]  = 48.848101210635171;
    lats[293]  = 48.707531145503857;
    lats[294]  = 48.56696108023506;
    lats[295]  = 48.42639101483023;
    lats[296]  = 48.285820949290759;
    lats[297]  = 48.145250883618075;
    lats[298]  = 48.004680817813544;
    lats[299]  = 47.864110751878535;
    lats[300]  = 47.723540685814392;
    lats[301]  = 47.582970619622444;
    lats[302]  = 47.442400553303997;
    lats[303]  = 47.301830486860368;
    lats[304]  = 47.161260420292813;
    lats[305]  = 47.020690353602596;
    lats[306]  = 46.880120286790955;
    lats[307]  = 46.73955021985914;
    lats[308]  = 46.598980152808338;
    lats[309]  = 46.458410085639763;
    lats[310]  = 46.317840018354602;
    lats[311]  = 46.177269950954006;
    lats[312]  = 46.036699883439134;
    lats[313]  = 45.896129815811136;
    lats[314]  = 45.755559748071114;
    lats[315]  = 45.614989680220205;
    lats[316]  = 45.474419612259481;
    lats[317]  = 45.333849544190024;
    lats[318]  = 45.193279476012933;
    lats[319]  = 45.052709407729239;
    lats[320]  = 44.912139339339987;
    lats[321]  = 44.771569270846214;
    lats[322]  = 44.630999202248923;
    lats[323]  = 44.490429133549149;
    lats[324]  = 44.349859064747854;
    lats[325]  = 44.209288995846045;
    lats[326]  = 44.068718926844674;
    lats[327]  = 43.928148857744716;
    lats[328]  = 43.787578788547094;
    lats[329]  = 43.64700871925276;
    lats[330]  = 43.506438649862638;
    lats[331]  = 43.365868580377636;
    lats[332]  = 43.225298510798666;
    lats[333]  = 43.0847284411266;
    lats[334]  = 42.944158371362349;
    lats[335]  = 42.803588301506764;
    lats[336]  = 42.663018231560706;
    lats[337]  = 42.522448161525034;
    lats[338]  = 42.381878091400594;
    lats[339]  = 42.241308021188203;
    lats[340]  = 42.100737950888686;
    lats[341]  = 41.960167880502873;
    lats[342]  = 41.819597810031553;
    lats[343]  = 41.679027739475522;
    lats[344]  = 41.538457668835562;
    lats[345]  = 41.397887598112455;
    lats[346]  = 41.257317527306981;
    lats[347]  = 41.116747456419873;
    lats[348]  = 40.976177385451912;
    lats[349]  = 40.835607314403816;
    lats[350]  = 40.695037243276325;
    lats[351]  = 40.554467172070169;
    lats[352]  = 40.41389710078608;
    lats[353]  = 40.273327029424742;
    lats[354]  = 40.132756957986885;
    lats[355]  = 39.992186886473185;
    lats[356]  = 39.851616814884331;
    lats[357]  = 39.711046743220997;
    lats[358]  = 39.570476671483874;
    lats[359]  = 39.429906599673615;
    lats[360]  = 39.289336527790894;
    lats[361]  = 39.148766455836338;
    lats[362]  = 39.008196383810613;
    lats[363]  = 38.867626311714339;
    lats[364]  = 38.727056239548169;
    lats[365]  = 38.5864861673127;
    lats[366]  = 38.44591609500857;
    lats[367]  = 38.305346022636385;
    lats[368]  = 38.164775950196741;
    lats[369]  = 38.02420587769025;
    lats[370]  = 37.883635805117493;
    lats[371]  = 37.743065732479067;
    lats[372]  = 37.602495659775542;
    lats[373]  = 37.461925587007492;
    lats[374]  = 37.321355514175501;
    lats[375]  = 37.180785441280122;
    lats[376]  = 37.040215368321896;
    lats[377]  = 36.899645295301404;
    lats[378]  = 36.759075222219167;
    lats[379]  = 36.618505149075737;
    lats[380]  = 36.477935075871656;
    lats[381]  = 36.33736500260742;
    lats[382]  = 36.196794929283605;
    lats[383]  = 36.056224855900687;
    lats[384]  = 35.9156547824592;
    lats[385]  = 35.775084708959632;
    lats[386]  = 35.634514635402525;
    lats[387]  = 35.493944561788332;
    lats[388]  = 35.353374488117588;
    lats[389]  = 35.21280441439076;
    lats[390]  = 35.072234340608333;
    lats[391]  = 34.931664266770788;
    lats[392]  = 34.79109419287861;
    lats[393]  = 34.650524118932253;
    lats[394]  = 34.509954044932208;
    lats[395]  = 34.369383970878907;
    lats[396]  = 34.228813896772813;
    lats[397]  = 34.088243822614395;
    lats[398]  = 33.9476737484041;
    lats[399]  = 33.807103674142361;
    lats[400]  = 33.66653359982962;
    lats[401]  = 33.525963525466317;
    lats[402]  = 33.385393451052892;
    lats[403]  = 33.244823376589757;
    lats[404]  = 33.104253302077339;
    lats[405]  = 32.963683227516071;
    lats[406]  = 32.823113152906366;
    lats[407]  = 32.682543078248621;
    lats[408]  = 32.541973003543255;
    lats[409]  = 32.401402928790681;
    lats[410]  = 32.260832853991289;
    lats[411]  = 32.120262779145477;
    lats[412]  = 31.979692704253651;
    lats[413]  = 31.839122629316183;
    lats[414]  = 31.698552554333489;
    lats[415]  = 31.55798247930592;
    lats[416]  = 31.417412404233875;
    lats[417]  = 31.276842329117731;
    lats[418]  = 31.136272253957859;
    lats[419]  = 30.99570217875463;
    lats[420]  = 30.855132103508407;
    lats[421]  = 30.71456202821955;
    lats[422]  = 30.573991952888438;
    lats[423]  = 30.433421877515418;
    lats[424]  = 30.292851802100841;
    lats[425]  = 30.152281726645064;
    lats[426]  = 30.011711651148435;
    lats[427]  = 29.87114157561129;
    lats[428]  = 29.730571500033992;
    lats[429]  = 29.590001424416862;
    lats[430]  = 29.449431348760253;
    lats[431]  = 29.308861273064483;
    lats[432]  = 29.168291197329893;
    lats[433]  = 29.027721121556816;
    lats[434]  = 28.887151045745565;
    lats[435]  = 28.746580969896474;
    lats[436]  = 28.606010894009859;
    lats[437]  = 28.465440818086037;
    lats[438]  = 28.324870742125327;
    lats[439]  = 28.184300666128038;
    lats[440]  = 28.043730590094491;
    lats[441]  = 27.903160514024975;
    lats[442]  = 27.762590437919812;
    lats[443]  = 27.622020361779295;
    lats[444]  = 27.481450285603731;
    lats[445]  = 27.340880209393415;
    lats[446]  = 27.200310133148644;
    lats[447]  = 27.05974005686971;
    lats[448]  = 26.919169980556905;
    lats[449]  = 26.778599904210516;
    lats[450]  = 26.638029827830831;
    lats[451]  = 26.497459751418134;
    lats[452]  = 26.356889674972713;
    lats[453]  = 26.216319598494842;
    lats[454]  = 26.075749521984797;
    lats[455]  = 25.935179445442859;
    lats[456]  = 25.794609368869299;
    lats[457]  = 25.654039292264386;
    lats[458]  = 25.513469215628398;
    lats[459]  = 25.3728991389616;
    lats[460]  = 25.232329062264245;
    lats[461]  = 25.091758985536615;
    lats[462]  = 24.951188908778963;
    lats[463]  = 24.810618831991551;
    lats[464]  = 24.670048755174633;
    lats[465]  = 24.529478678328466;
    lats[466]  = 24.388908601453309;
    lats[467]  = 24.248338524549407;
    lats[468]  = 24.107768447617016;
    lats[469]  = 23.96719837065638;
    lats[470]  = 23.826628293667756;
    lats[471]  = 23.686058216651375;
    lats[472]  = 23.545488139607492;
    lats[473]  = 23.404918062536346;
    lats[474]  = 23.264347985438178;
    lats[475]  = 23.123777908313219;
    lats[476]  = 22.98320783116171;
    lats[477]  = 22.84263775398389;
    lats[478]  = 22.70206767677999;
    lats[479]  = 22.561497599550243;
    lats[480]  = 22.420927522294875;
    lats[481]  = 22.280357445014126;
    lats[482]  = 22.139787367708202;
    lats[483]  = 21.999217290377352;
    lats[484]  = 21.858647213021786;
    lats[485]  = 21.718077135641735;
    lats[486]  = 21.577507058237412;
    lats[487]  = 21.436936980809044;
    lats[488]  = 21.296366903356844;
    lats[489]  = 21.155796825881037;
    lats[490]  = 21.015226748381831;
    lats[491]  = 20.874656670859444;
    lats[492]  = 20.734086593314085;
    lats[493]  = 20.593516515745968;
    lats[494]  = 20.452946438155308;
    lats[495]  = 20.312376360542309;
    lats[496]  = 20.171806282907177;
    lats[497]  = 20.031236205250121;
    lats[498]  = 19.890666127571347;
    lats[499]  = 19.750096049871054;
    lats[500]  = 19.609525972149449;
    lats[501]  = 19.468955894406733;
    lats[502]  = 19.328385816643106;
    lats[503]  = 19.187815738858767;
    lats[504]  = 19.04724566105391;
    lats[505]  = 18.906675583228736;
    lats[506]  = 18.766105505383443;
    lats[507]  = 18.625535427518219;
    lats[508]  = 18.484965349633256;
    lats[509]  = 18.344395271728757;
    lats[510]  = 18.203825193804899;
    lats[511]  = 18.063255115861882;
    lats[512]  = 17.922685037899889;
    lats[513]  = 17.782114959919113;
    lats[514]  = 17.641544881919739;
    lats[515]  = 17.500974803901951;
    lats[516]  = 17.360404725865926;
    lats[517]  = 17.219834647811862;
    lats[518]  = 17.079264569739937;
    lats[519]  = 16.938694491650331;
    lats[520]  = 16.798124413543224;
    lats[521]  = 16.657554335418794;
    lats[522]  = 16.516984257277226;
    lats[523]  = 16.376414179118694;
    lats[524]  = 16.235844100943371;
    lats[525]  = 16.09527402275144;
    lats[526]  = 15.954703944543072;
    lats[527]  = 15.814133866318445;
    lats[528]  = 15.673563788077727;
    lats[529]  = 15.532993709821094;
    lats[530]  = 15.392423631548718;
    lats[531]  = 15.251853553260768;
    lats[532]  = 15.111283474957411;
    lats[533]  = 14.970713396638821;
    lats[534]  = 14.830143318305167;
    lats[535]  = 14.689573239956617;
    lats[536]  = 14.549003161593328;
    lats[537]  = 14.408433083215476;
    lats[538]  = 14.267863004823225;
    lats[539]  = 14.127292926416734;
    lats[540]  = 13.986722847996173;
    lats[541]  = 13.8461527695617;
    lats[542]  = 13.705582691113481;
    lats[543]  = 13.565012612651675;
    lats[544]  = 13.424442534176441;
    lats[545]  = 13.283872455687943;
    lats[546]  = 13.143302377186339;
    lats[547]  = 13.002732298671786;
    lats[548]  = 12.862162220144443;
    lats[549]  = 12.72159214160447;
    lats[550]  = 12.58102206305202;
    lats[551]  = 12.440451984487247;
    lats[552]  = 12.299881905910311;
    lats[553]  = 12.159311827321366;
    lats[554]  = 12.018741748720567;
    lats[555]  = 11.878171670108063;
    lats[556]  = 11.73760159148401;
    lats[557]  = 11.597031512848561;
    lats[558]  = 11.456461434201868;
    lats[559]  = 11.315891355544077;
    lats[560]  = 11.175321276875344;
    lats[561]  = 11.034751198195819;
    lats[562]  = 10.894181119505649;
    lats[563]  = 10.753611040804984;
    lats[564]  = 10.613040962093971;
    lats[565]  = 10.472470883372759;
    lats[566]  = 10.331900804641496;
    lats[567]  = 10.191330725900327;
    lats[568]  = 10.050760647149401;
    lats[569]  = 9.9101905683888614;
    lats[570]  = 9.7696204896188554;
    lats[571]  = 9.6290504108395272;
    lats[572]  = 9.4884803320510205;
    lats[573]  = 9.3479102532534792;
    lats[574]  = 9.2073401744470491;
    lats[575]  = 9.0667700956318686;
    lats[576]  = 8.9262000168080871;
    lats[577]  = 8.7856299379758411;
    lats[578]  = 8.645059859135273;
    lats[579]  = 8.5044897802865282;
    lats[580]  = 8.3639197014297419;
    lats[581]  = 8.223349622565058;
    lats[582]  = 8.0827795436926184;
    lats[583]  = 7.9422094648125583;
    lats[584]  = 7.8016393859250206;
    lats[585]  = 7.661069307030143;
    lats[586]  = 7.5204992281280649;
    lats[587]  = 7.3799291492189223;
    lats[588]  = 7.2393590703028563;
    lats[589]  = 7.098788991380002;
    lats[590]  = 6.9582189124504987;
    lats[591]  = 6.8176488335144816;
    lats[592]  = 6.6770787545720891;
    lats[593]  = 6.5365086756234554;
    lats[594]  = 6.3959385966687181;
    lats[595]  = 6.2553685177080123;
    lats[596]  = 6.1147984387414738;
    lats[597]  = 5.9742283597692367;
    lats[598]  = 5.833658280791437;
    lats[599]  = 5.6930882018082087;
    lats[600]  = 5.5525181228196869;
    lats[601]  = 5.4119480438260039;
    lats[602]  = 5.2713779648272956;
    lats[603]  = 5.1308078858236934;
    lats[604]  = 4.9902378068153324;
    lats[605]  = 4.8496677278023448;
    lats[606]  = 4.7090976487848639;
    lats[607]  = 4.5685275697630221;
    lats[608]  = 4.4279574907369508;
    lats[609]  = 4.2873874117067841;
    lats[610]  = 4.1468173326726534;
    lats[611]  = 4.0062472536346903;
    lats[612]  = 3.8656771745930261;
    lats[613]  = 3.7251070955477918;
    lats[614]  = 3.5845370164991213;
    lats[615]  = 3.4439669374471427;
    lats[616]  = 3.3033968583919884;
    lats[617]  = 3.1628267793337885;
    lats[618]  = 3.0222567002726746;
    lats[619]  = 2.8816866212087762;
    lats[620]  = 2.7411165421422243;
    lats[621]  = 2.6005464630731496;
    lats[622]  = 2.4599763840016813;
    lats[623]  = 2.3194063049279499;
    lats[624]  = 2.1788362258520855;
    lats[625]  = 2.0382661467742174;
    lats[626]  = 1.8976960676944756;
    lats[627]  = 1.7571259886129893;
    lats[628]  = 1.6165559095298885;
    lats[629]  = 1.4759858304453026;
    lats[630]  = 1.3354157513593612;
    lats[631]  = 1.194845672272193;
    lats[632]  = 1.0542755931839276;
    lats[633]  = 0.91370551409469447;
    lats[634]  = 0.77313543500462234;
    lats[635]  = 0.63256535591384055;
    lats[636]  = 0.49199527682247807;
    lats[637]  = 0.351425197730664;
    lats[638]  = 0.21085511863852741;
    lats[639]  = 0.070285039546197275;
    lats[640]  = -0.070285039546197275;
    lats[641]  = -0.21085511863852741;
    lats[642]  = -0.351425197730664;
    lats[643]  = -0.49199527682247807;
    lats[644]  = -0.63256535591384055;
    lats[645]  = -0.77313543500462234;
    lats[646]  = -0.91370551409469447;
    lats[647]  = -1.0542755931839276;
    lats[648]  = -1.194845672272193;
    lats[649]  = -1.3354157513593612;
    lats[650]  = -1.4759858304453026;
    lats[651]  = -1.6165559095298885;
    lats[652]  = -1.7571259886129893;
    lats[653]  = -1.8976960676944756;
    lats[654]  = -2.0382661467742174;
    lats[655]  = -2.1788362258520855;
    lats[656]  = -2.3194063049279499;
    lats[657]  = -2.4599763840016813;
    lats[658]  = -2.6005464630731496;
    lats[659]  = -2.7411165421422243;
    lats[660]  = -2.8816866212087762;
    lats[661]  = -3.0222567002726746;
    lats[662]  = -3.1628267793337885;
    lats[663]  = -3.3033968583919884;
    lats[664]  = -3.4439669374471427;
    lats[665]  = -3.5845370164991213;
    lats[666]  = -3.7251070955477918;
    lats[667]  = -3.8656771745930261;
    lats[668]  = -4.0062472536346903;
    lats[669]  = -4.1468173326726534;
    lats[670]  = -4.2873874117067841;
    lats[671]  = -4.4279574907369508;
    lats[672]  = -4.5685275697630221;
    lats[673]  = -4.7090976487848639;
    lats[674]  = -4.8496677278023448;
    lats[675]  = -4.9902378068153324;
    lats[676]  = -5.1308078858236934;
    lats[677]  = -5.2713779648272956;
    lats[678]  = -5.4119480438260039;
    lats[679]  = -5.5525181228196869;
    lats[680]  = -5.6930882018082087;
    lats[681]  = -5.833658280791437;
    lats[682]  = -5.9742283597692367;
    lats[683]  = -6.1147984387414738;
    lats[684]  = -6.2553685177080123;
    lats[685]  = -6.3959385966687181;
    lats[686]  = -6.5365086756234554;
    lats[687]  = -6.6770787545720891;
    lats[688]  = -6.8176488335144816;
    lats[689]  = -6.9582189124504987;
    lats[690]  = -7.098788991380002;
    lats[691]  = -7.2393590703028563;
    lats[692]  = -7.3799291492189223;
    lats[693]  = -7.5204992281280649;
    lats[694]  = -7.661069307030143;
    lats[695]  = -7.8016393859250206;
    lats[696]  = -7.9422094648125583;
    lats[697]  = -8.0827795436926184;
    lats[698]  = -8.223349622565058;
    lats[699]  = -8.3639197014297419;
    lats[700]  = -8.5044897802865282;
    lats[701]  = -8.645059859135273;
    lats[702]  = -8.7856299379758411;
    lats[703]  = -8.9262000168080871;
    lats[704]  = -9.0667700956318686;
    lats[705]  = -9.2073401744470491;
    lats[706]  = -9.3479102532534792;
    lats[707]  = -9.4884803320510205;
    lats[708]  = -9.6290504108395272;
    lats[709]  = -9.7696204896188554;
    lats[710]  = -9.9101905683888614;
    lats[711]  = -10.050760647149401;
    lats[712]  = -10.191330725900327;
    lats[713]  = -10.331900804641496;
    lats[714]  = -10.472470883372759;
    lats[715]  = -10.613040962093971;
    lats[716]  = -10.753611040804984;
    lats[717]  = -10.894181119505649;
    lats[718]  = -11.034751198195819;
    lats[719]  = -11.175321276875344;
    lats[720]  = -11.315891355544077;
    lats[721]  = -11.456461434201868;
    lats[722]  = -11.597031512848561;
    lats[723]  = -11.73760159148401;
    lats[724]  = -11.878171670108063;
    lats[725]  = -12.018741748720567;
    lats[726]  = -12.159311827321366;
    lats[727]  = -12.299881905910311;
    lats[728]  = -12.440451984487247;
    lats[729]  = -12.58102206305202;
    lats[730]  = -12.72159214160447;
    lats[731]  = -12.862162220144443;
    lats[732]  = -13.002732298671786;
    lats[733]  = -13.143302377186339;
    lats[734]  = -13.283872455687943;
    lats[735]  = -13.424442534176441;
    lats[736]  = -13.565012612651675;
    lats[737]  = -13.705582691113481;
    lats[738]  = -13.8461527695617;
    lats[739]  = -13.986722847996173;
    lats[740]  = -14.127292926416734;
    lats[741]  = -14.267863004823225;
    lats[742]  = -14.408433083215476;
    lats[743]  = -14.549003161593328;
    lats[744]  = -14.689573239956617;
    lats[745]  = -14.830143318305167;
    lats[746]  = -14.970713396638821;
    lats[747]  = -15.111283474957411;
    lats[748]  = -15.251853553260768;
    lats[749]  = -15.392423631548718;
    lats[750]  = -15.532993709821094;
    lats[751]  = -15.673563788077727;
    lats[752]  = -15.814133866318445;
    lats[753]  = -15.954703944543072;
    lats[754]  = -16.09527402275144;
    lats[755]  = -16.235844100943371;
    lats[756]  = -16.376414179118694;
    lats[757]  = -16.516984257277226;
    lats[758]  = -16.657554335418794;
    lats[759]  = -16.798124413543224;
    lats[760]  = -16.938694491650331;
    lats[761]  = -17.079264569739937;
    lats[762]  = -17.219834647811862;
    lats[763]  = -17.360404725865926;
    lats[764]  = -17.500974803901951;
    lats[765]  = -17.641544881919739;
    lats[766]  = -17.782114959919113;
    lats[767]  = -17.922685037899889;
    lats[768]  = -18.063255115861882;
    lats[769]  = -18.203825193804899;
    lats[770]  = -18.344395271728757;
    lats[771]  = -18.484965349633256;
    lats[772]  = -18.625535427518219;
    lats[773]  = -18.766105505383443;
    lats[774]  = -18.906675583228736;
    lats[775]  = -19.04724566105391;
    lats[776]  = -19.187815738858767;
    lats[777]  = -19.328385816643106;
    lats[778]  = -19.468955894406733;
    lats[779]  = -19.609525972149449;
    lats[780]  = -19.750096049871054;
    lats[781]  = -19.890666127571347;
    lats[782]  = -20.031236205250121;
    lats[783]  = -20.171806282907177;
    lats[784]  = -20.312376360542309;
    lats[785]  = -20.452946438155308;
    lats[786]  = -20.593516515745968;
    lats[787]  = -20.734086593314085;
    lats[788]  = -20.874656670859444;
    lats[789]  = -21.015226748381831;
    lats[790]  = -21.155796825881037;
    lats[791]  = -21.296366903356844;
    lats[792]  = -21.436936980809044;
    lats[793]  = -21.577507058237412;
    lats[794]  = -21.718077135641735;
    lats[795]  = -21.858647213021786;
    lats[796]  = -21.999217290377352;
    lats[797]  = -22.139787367708202;
    lats[798]  = -22.280357445014126;
    lats[799]  = -22.420927522294875;
    lats[800]  = -22.561497599550243;
    lats[801]  = -22.70206767677999;
    lats[802]  = -22.84263775398389;
    lats[803]  = -22.98320783116171;
    lats[804]  = -23.123777908313219;
    lats[805]  = -23.264347985438178;
    lats[806]  = -23.404918062536346;
    lats[807]  = -23.545488139607492;
    lats[808]  = -23.686058216651375;
    lats[809]  = -23.826628293667756;
    lats[810]  = -23.96719837065638;
    lats[811]  = -24.107768447617016;
    lats[812]  = -24.248338524549407;
    lats[813]  = -24.388908601453309;
    lats[814]  = -24.529478678328466;
    lats[815]  = -24.670048755174633;
    lats[816]  = -24.810618831991551;
    lats[817]  = -24.951188908778963;
    lats[818]  = -25.091758985536615;
    lats[819]  = -25.232329062264245;
    lats[820]  = -25.3728991389616;
    lats[821]  = -25.513469215628398;
    lats[822]  = -25.654039292264386;
    lats[823]  = -25.794609368869299;
    lats[824]  = -25.935179445442859;
    lats[825]  = -26.075749521984797;
    lats[826]  = -26.216319598494842;
    lats[827]  = -26.356889674972713;
    lats[828]  = -26.497459751418134;
    lats[829]  = -26.638029827830831;
    lats[830]  = -26.778599904210516;
    lats[831]  = -26.919169980556905;
    lats[832]  = -27.05974005686971;
    lats[833]  = -27.200310133148644;
    lats[834]  = -27.340880209393415;
    lats[835]  = -27.481450285603731;
    lats[836]  = -27.622020361779295;
    lats[837]  = -27.762590437919812;
    lats[838]  = -27.903160514024975;
    lats[839]  = -28.043730590094491;
    lats[840]  = -28.184300666128038;
    lats[841]  = -28.324870742125327;
    lats[842]  = -28.465440818086037;
    lats[843]  = -28.606010894009859;
    lats[844]  = -28.746580969896474;
    lats[845]  = -28.887151045745565;
    lats[846]  = -29.027721121556816;
    lats[847]  = -29.168291197329893;
    lats[848]  = -29.308861273064483;
    lats[849]  = -29.449431348760253;
    lats[850]  = -29.590001424416862;
    lats[851]  = -29.730571500033992;
    lats[852]  = -29.87114157561129;
    lats[853]  = -30.011711651148435;
    lats[854]  = -30.152281726645064;
    lats[855]  = -30.292851802100841;
    lats[856]  = -30.433421877515418;
    lats[857]  = -30.573991952888438;
    lats[858]  = -30.71456202821955;
    lats[859]  = -30.855132103508407;
    lats[860]  = -30.99570217875463;
    lats[861]  = -31.136272253957859;
    lats[862]  = -31.276842329117731;
    lats[863]  = -31.417412404233875;
    lats[864]  = -31.55798247930592;
    lats[865]  = -31.698552554333489;
    lats[866]  = -31.839122629316183;
    lats[867]  = -31.979692704253651;
    lats[868]  = -32.120262779145477;
    lats[869]  = -32.260832853991289;
    lats[870]  = -32.401402928790681;
    lats[871]  = -32.541973003543255;
    lats[872]  = -32.682543078248621;
    lats[873]  = -32.823113152906366;
    lats[874]  = -32.963683227516071;
    lats[875]  = -33.104253302077339;
    lats[876]  = -33.244823376589757;
    lats[877]  = -33.385393451052892;
    lats[878]  = -33.525963525466317;
    lats[879]  = -33.66653359982962;
    lats[880]  = -33.807103674142361;
    lats[881]  = -33.9476737484041;
    lats[882]  = -34.088243822614395;
    lats[883]  = -34.228813896772813;
    lats[884]  = -34.369383970878907;
    lats[885]  = -34.509954044932208;
    lats[886]  = -34.650524118932253;
    lats[887]  = -34.79109419287861;
    lats[888]  = -34.931664266770788;
    lats[889]  = -35.072234340608333;
    lats[890]  = -35.21280441439076;
    lats[891]  = -35.353374488117588;
    lats[892]  = -35.493944561788332;
    lats[893]  = -35.634514635402525;
    lats[894]  = -35.775084708959632;
    lats[895]  = -35.9156547824592;
    lats[896]  = -36.056224855900687;
    lats[897]  = -36.196794929283605;
    lats[898]  = -36.33736500260742;
    lats[899]  = -36.477935075871656;
    lats[900]  = -36.618505149075737;
    lats[901]  = -36.759075222219167;
    lats[902]  = -36.899645295301404;
    lats[903]  = -37.040215368321896;
    lats[904]  = -37.180785441280122;
    lats[905]  = -37.321355514175501;
    lats[906]  = -37.461925587007492;
    lats[907]  = -37.602495659775542;
    lats[908]  = -37.743065732479067;
    lats[909]  = -37.883635805117493;
    lats[910]  = -38.02420587769025;
    lats[911]  = -38.164775950196741;
    lats[912]  = -38.305346022636385;
    lats[913]  = -38.44591609500857;
    lats[914]  = -38.5864861673127;
    lats[915]  = -38.727056239548169;
    lats[916]  = -38.867626311714339;
    lats[917]  = -39.008196383810613;
    lats[918]  = -39.148766455836338;
    lats[919]  = -39.289336527790894;
    lats[920]  = -39.429906599673615;
    lats[921]  = -39.570476671483874;
    lats[922]  = -39.711046743220997;
    lats[923]  = -39.851616814884331;
    lats[924]  = -39.992186886473185;
    lats[925]  = -40.132756957986885;
    lats[926]  = -40.273327029424742;
    lats[927]  = -40.41389710078608;
    lats[928]  = -40.554467172070169;
    lats[929]  = -40.695037243276325;
    lats[930]  = -40.835607314403816;
    lats[931]  = -40.976177385451912;
    lats[932]  = -41.116747456419873;
    lats[933]  = -41.257317527306981;
    lats[934]  = -41.397887598112455;
    lats[935]  = -41.538457668835562;
    lats[936]  = -41.679027739475522;
    lats[937]  = -41.819597810031553;
    lats[938]  = -41.960167880502873;
    lats[939]  = -42.100737950888686;
    lats[940]  = -42.241308021188203;
    lats[941]  = -42.381878091400594;
    lats[942]  = -42.522448161525034;
    lats[943]  = -42.663018231560706;
    lats[944]  = -42.803588301506764;
    lats[945]  = -42.944158371362349;
    lats[946]  = -43.0847284411266;
    lats[947]  = -43.225298510798666;
    lats[948]  = -43.365868580377636;
    lats[949]  = -43.506438649862638;
    lats[950]  = -43.64700871925276;
    lats[951]  = -43.787578788547094;
    lats[952]  = -43.928148857744716;
    lats[953]  = -44.068718926844674;
    lats[954]  = -44.209288995846045;
    lats[955]  = -44.349859064747854;
    lats[956]  = -44.490429133549149;
    lats[957]  = -44.630999202248923;
    lats[958]  = -44.771569270846214;
    lats[959]  = -44.912139339339987;
    lats[960]  = -45.052709407729239;
    lats[961]  = -45.193279476012933;
    lats[962]  = -45.333849544190024;
    lats[963]  = -45.474419612259481;
    lats[964]  = -45.614989680220205;
    lats[965]  = -45.755559748071114;
    lats[966]  = -45.896129815811136;
    lats[967]  = -46.036699883439134;
    lats[968]  = -46.177269950954006;
    lats[969]  = -46.317840018354602;
    lats[970]  = -46.458410085639763;
    lats[971]  = -46.598980152808338;
    lats[972]  = -46.73955021985914;
    lats[973]  = -46.880120286790955;
    lats[974]  = -47.020690353602596;
    lats[975]  = -47.161260420292813;
    lats[976]  = -47.301830486860368;
    lats[977]  = -47.442400553303997;
    lats[978]  = -47.582970619622444;
    lats[979]  = -47.723540685814392;
    lats[980]  = -47.864110751878535;
    lats[981]  = -48.004680817813544;
    lats[982]  = -48.145250883618075;
    lats[983]  = -48.285820949290759;
    lats[984]  = -48.42639101483023;
    lats[985]  = -48.56696108023506;
    lats[986]  = -48.707531145503857;
    lats[987]  = -48.848101210635171;
    lats[988]  = -48.988671275627539;
    lats[989]  = -49.129241340479489;
    lats[990]  = -49.269811405189529;
    lats[991]  = -49.410381469756118;
    lats[992]  = -49.550951534177734;
    lats[993]  = -49.691521598452823;
    lats[994]  = -49.832091662579785;
    lats[995]  = -49.972661726557028;
    lats[996]  = -50.113231790382912;
    lats[997]  = -50.253801854055808;
    lats[998]  = -50.39437191757402;
    lats[999]  = -50.534941980935862;
    lats[1000] = -50.675512044139623;
    lats[1001] = -50.81608210718354;
    lats[1002] = -50.956652170065858;
    lats[1003] = -51.097222232784773;
    lats[1004] = -51.237792295338465;
    lats[1005] = -51.378362357725095;
    lats[1006] = -51.518932419942786;
    lats[1007] = -51.659502481989627;
    lats[1008] = -51.800072543863692;
    lats[1009] = -51.940642605563028;
    lats[1010] = -52.081212667085637;
    lats[1011] = -52.221782728429538;
    lats[1012] = -52.362352789592649;
    lats[1013] = -52.502922850572908;
    lats[1014] = -52.643492911368206;
    lats[1015] = -52.784062971976404;
    lats[1016] = -52.924633032395342;
    lats[1017] = -53.065203092622802;
    lats[1018] = -53.205773152656562;
    lats[1019] = -53.346343212494332;
    lats[1020] = -53.486913272133812;
    lats[1021] = -53.627483331572677;
    lats[1022] = -53.768053390808532;
    lats[1023] = -53.90862344983897;
    lats[1024] = -54.049193508661538;
    lats[1025] = -54.189763567273758;
    lats[1026] = -54.330333625673063;
    lats[1027] = -54.470903683856939;
    lats[1028] = -54.611473741822735;
    lats[1029] = -54.752043799567822;
    lats[1030] = -54.892613857089486;
    lats[1031] = -55.033183914385013;
    lats[1032] = -55.173753971451625;
    lats[1033] = -55.314324028286471;
    lats[1034] = -55.45489408488671;
    lats[1035] = -55.595464141249401;
    lats[1036] = -55.736034197371588;
    lats[1037] = -55.876604253250278;
    lats[1038] = -56.017174308882367;
    lats[1039] = -56.157744364264779;
    lats[1040] = -56.29831441939433;
    lats[1041] = -56.438884474267795;
    lats[1042] = -56.579454528881925;
    lats[1043] = -56.720024583233375;
    lats[1044] = -56.860594637318769;
    lats[1045] = -57.001164691134662;
    lats[1046] = -57.141734744677549;
    lats[1047] = -57.282304797943887;
    lats[1048] = -57.422874850930043;
    lats[1049] = -57.563444903632337;
    lats[1050] = -57.704014956047033;
    lats[1051] = -57.844585008170284;
    lats[1052] = -57.985155059998249;
    lats[1053] = -58.125725111526968;
    lats[1054] = -58.266295162752428;
    lats[1055] = -58.406865213670514;
    lats[1056] = -58.547435264277105;
    lats[1057] = -58.688005314567938;
    lats[1058] = -58.828575364538722;
    lats[1059] = -58.96914541418505;
    lats[1060] = -59.10971546350244;
    lats[1061] = -59.250285512486386;
    lats[1062] = -59.390855561132213;
    lats[1063] = -59.531425609435225;
    lats[1064] = -59.671995657390596;
    lats[1065] = -59.812565704993467;
    lats[1066] = -59.953135752238794;
    lats[1067] = -60.093705799121537;
    lats[1068] = -60.234275845636503;
    lats[1069] = -60.374845891778421;
    lats[1070] = -60.515415937541938;
    lats[1071] = -60.655985982921543;
    lats[1072] = -60.796556027911663;
    lats[1073] = -60.937126072506608;
    lats[1074] = -61.077696116700601;
    lats[1075] = -61.21826616048768;
    lats[1076] = -61.358836203861841;
    lats[1077] = -61.499406246816953;
    lats[1078] = -61.639976289346727;
    lats[1079] = -61.780546331444761;
    lats[1080] = -61.921116373104539;
    lats[1081] = -62.061686414319418;
    lats[1082] = -62.202256455082583;
    lats[1083] = -62.342826495387122;
    lats[1084] = -62.483396535225978;
    lats[1085] = -62.62396657459194;
    lats[1086] = -62.764536613477638;
    lats[1087] = -62.905106651875542;
    lats[1088] = -63.045676689778013;
    lats[1089] = -63.186246727177178;
    lats[1090] = -63.326816764065093;
    lats[1091] = -63.467386800433559;
    lats[1092] = -63.607956836274255;
    lats[1093] = -63.748526871578648;
    lats[1094] = -63.889096906338061;
    lats[1095] = -64.029666940543564;
    lats[1096] = -64.170236974186125;
    lats[1097] = -64.310807007256443;
    lats[1098] = -64.451377039745026;
    lats[1099] = -64.591947071642196;
    lats[1100] = -64.732517102938033;
    lats[1101] = -64.873087133622406;
    lats[1102] = -65.013657163684968;
    lats[1103] = -65.154227193115119;
    lats[1104] = -65.294797221902016;
    lats[1105] = -65.435367250034616;
    lats[1106] = -65.575937277501538;
    lats[1107] = -65.716507304291198;
    lats[1108] = -65.85707733039176;
    lats[1109] = -65.997647355791017;
    lats[1110] = -66.138217380476604;
    lats[1111] = -66.278787404435761;
    lats[1112] = -66.41935742765547;
    lats[1113] = -66.559927450122359;
    lats[1114] = -66.700497471822814;
    lats[1115] = -66.841067492742795;
    lats[1116] = -66.981637512867991;
    lats[1117] = -67.122207532183722;
    lats[1118] = -67.262777550674912;
    lats[1119] = -67.403347568326168;
    lats[1120] = -67.543917585121662;
    lats[1121] = -67.684487601045149;
    lats[1122] = -67.825057616080073;
    lats[1123] = -67.965627630209354;
    lats[1124] = -68.106197643415527;
    lats[1125] = -68.246767655680657;
    lats[1126] = -68.387337666986312;
    lats[1127] = -68.52790767731365;
    lats[1128] = -68.668477686643286;
    lats[1129] = -68.809047694955296;
    lats[1130] = -68.949617702229318;
    lats[1131] = -69.090187708444333;
    lats[1132] = -69.230757713578825;
    lats[1133] = -69.37132771761064;
    lats[1134] = -69.511897720517084;
    lats[1135] = -69.65246772227475;
    lats[1136] = -69.793037722859665;
    lats[1137] = -69.933607722247146;
    lats[1138] = -70.074177720411782;
    lats[1139] = -70.214747717327526;
    lats[1140] = -70.355317712967462;
    lats[1141] = -70.495887707304007;
    lats[1142] = -70.636457700308753;
    lats[1143] = -70.777027691952398;
    lats[1144] = -70.917597682204899;
    lats[1145] = -71.058167671035164;
    lats[1146] = -71.198737658411332;
    lats[1147] = -71.339307644300462;
    lats[1148] = -71.47987762866866;
    lats[1149] = -71.620447611481026;
    lats[1150] = -71.761017592701492;
    lats[1151] = -71.901587572292982;
    lats[1152] = -72.042157550217183;
    lats[1153] = -72.182727526434604;
    lats[1154] = -72.323297500904502;
    lats[1155] = -72.463867473584784;
    lats[1156] = -72.604437444432065;
    lats[1157] = -72.745007413401481;
    lats[1158] = -72.885577380446747;
    lats[1159] = -73.026147345520002;
    lats[1160] = -73.166717308571819;
    lats[1161] = -73.307287269551111;
    lats[1162] = -73.447857228405013;
    lats[1163] = -73.588427185078871;
    lats[1164] = -73.728997139516167;
    lats[1165] = -73.869567091658411;
    lats[1166] = -74.010137041445006;
    lats[1167] = -74.150706988813226;
    lats[1168] = -74.291276933698185;
    lats[1169] = -74.431846876032495;
    lats[1170] = -74.57241681574645;
    lats[1171] = -74.712986752767719;
    lats[1172] = -74.853556687021296;
    lats[1173] = -74.994126618429377;
    lats[1174] = -75.134696546911186;
    lats[1175] = -75.275266472382896;
    lats[1176] = -75.41583639475742;
    lats[1177] = -75.556406313944308;
    lats[1178] = -75.69697622984954;
    lats[1179] = -75.837546142375359;
    lats[1180] = -75.978116051420102;
    lats[1181] = -76.118685956877997;
    lats[1182] = -76.259255858638895;
    lats[1183] = -76.399825756588143;
    lats[1184] = -76.540395650606285;
    lats[1185] = -76.680965540568806;
    lats[1186] = -76.821535426345932;
    lats[1187] = -76.962105307802219;
    lats[1188] = -77.102675184796354;
    lats[1189] = -77.243245057180829;
    lats[1190] = -77.383814924801513;
    lats[1191] = -77.524384787497311;
    lats[1192] = -77.664954645099883;
    lats[1193] = -77.805524497433041;
    lats[1194] = -77.946094344312371;
    lats[1195] = -78.086664185544819;
    lats[1196] = -78.227234020928066;
    lats[1197] = -78.367803850250056;
    lats[1198] = -78.508373673288318;
    lats[1199] = -78.648943489809355;
    lats[1200] = -78.789513299567957;
    lats[1201] = -78.930083102306568;
    lats[1202] = -79.070652897754229;
    lats[1203] = -79.211222685625927;
    lats[1204] = -79.351792465621628;
    lats[1205] = -79.492362237425226;
    lats[1206] = -79.632932000703448;
    lats[1207] = -79.773501755104689;
    lats[1208] = -79.914071500257819;
    lats[1209] = -80.054641235770603;
    lats[1210] = -80.195210961228469;
    lats[1211] = -80.335780676192584;
    lats[1212] = -80.47635038019834;
    lats[1213] = -80.616920072753146;
    lats[1214] = -80.757489753334553;
    lats[1215] = -80.898059421387785;
    lats[1216] = -81.038629076323318;
    lats[1217] = -81.179198717514012;
    lats[1218] = -81.319768344292086;
    lats[1219] = -81.460337955945846;
    lats[1220] = -81.600907551715878;
    lats[1221] = -81.741477130791196;
    lats[1222] = -81.882046692304485;
    lats[1223] = -82.022616235327504;
    lats[1224] = -82.163185758865239;
    lats[1225] = -82.303755261850071;
    lats[1226] = -82.444324743134914;
    lats[1227] = -82.584894201485696;
    lats[1228] = -82.725463635573107;
    lats[1229] = -82.866033043962815;
    lats[1230] = -83.006602425105484;
    lats[1231] = -83.147171777324388;
    lats[1232] = -83.287741098802584;
    lats[1233] = -83.428310387567549;
    lats[1234] = -83.568879641474325;
    lats[1235] = -83.709448858186462;
    lats[1236] = -83.850018035153667;
    lats[1237] = -83.990587169587158;
    lats[1238] = -84.131156258431133;
    lats[1239] = -84.271725298329656;
    lats[1240] = -84.412294285589354;
    lats[1241] = -84.552863216135577;
    lats[1242] = -84.693432085462035;
    lats[1243] = -84.834000888572191;
    lats[1244] = -84.974569619910426;
    lats[1245] = -85.115138273281829;
    lats[1246] = -85.255706841757572;
    lats[1247] = -85.396275317562669;
    lats[1248] = -85.536843691942948;
    lats[1249] = -85.677411955006008;
    lats[1250] = -85.817980095529578;
    lats[1251] = -85.958548100730781;
    lats[1252] = -86.0991159559849;
    lats[1253] = -86.239683644481104;
    lats[1254] = -86.380251146798656;
    lats[1255] = -86.520818440379529;
    lats[1256] = -86.661385498868242;
    lats[1257] = -86.801952291278369;
    lats[1258] = -86.942518780928566;
    lats[1259] = -87.083084924070917;
    lats[1260] = -87.223650668104085;
    lats[1261] = -87.364215949214667;
    lats[1262] = -87.504780689222315;
    lats[1263] = -87.645344791295628;
    lats[1264] = -87.785908134040668;
    lats[1265] = -87.926470563186442;
    lats[1266] = -88.067031879650926;
    lats[1267] = -88.207591822004105;
    lats[1268] = -88.348150039999084;
    lats[1269] = -88.488706053376362;
    lats[1270] = -88.629259185411627;
    lats[1271] = -88.76980845110036;
    lats[1272] = -88.91035235926023;
    lats[1273] = -89.050888539966436;
    lats[1274] = -89.191412986832432;
    lats[1275] = -89.331918354381827;
    lats[1276] = -89.472389582061126;
    lats[1277] = -89.612790258599077;
    lats[1278] = -89.753004943174034;
    lats[1279] = -89.892396445590066;
    return GRIB_SUCCESS;
}

static void gauss_first_guess(long trunc, double* vals)
{
    long i                = 0, numVals;
    static double gvals[] = {
        2.4048255577E0,
        5.5200781103E0,
        8.6537279129E0,
        11.7915344391E0,
        14.9309177086E0,
        18.0710639679E0,
        21.2116366299E0,
        24.3524715308E0,
        27.4934791320E0,
        30.6346064684E0,
        33.7758202136E0,
        36.9170983537E0,
        40.0584257646E0,
        43.1997917132E0,
        46.3411883717E0,
        49.4826098974E0,
        52.6240518411E0,
        55.7655107550E0,
        58.9069839261E0,
        62.0484691902E0,
        65.1899648002E0,
        68.3314693299E0,
        71.4729816036E0,
        74.6145006437E0,
        77.7560256304E0,
        80.8975558711E0,
        84.0390907769E0,
        87.1806298436E0,
        90.3221726372E0,
        93.4637187819E0,
        96.6052679510E0,
        99.7468198587E0,
        102.8883742542E0,
        106.0299309165E0,
        109.1714896498E0,
        112.3130502805E0,
        115.4546126537E0,
        118.5961766309E0,
        121.7377420880E0,
        124.8793089132E0,
        128.0208770059E0,
        131.1624462752E0,
        134.3040166383E0,
        137.4455880203E0,
        140.5871603528E0,
        143.7287335737E0,
        146.8703076258E0,
        150.0118824570E0,
        153.1534580192E0,
        156.2950342685E0,
    };

    numVals = NUMBER(gvals);
    for (i = 0; i < trunc; i++) {
        if (i < numVals)
            vals[i] = gvals[i];
        else
            vals[i] = vals[i - 1] + M_PI;
    }
}
static Fraction_value_type fraction_gcd(Fraction_value_type a, Fraction_value_type b)
{
    while (b != 0) {
        Fraction_value_type r = a % b;
        a                     = b;
        b                     = r;
    }
    return a;
}
static int compare_points(const void* a, const void* b)
{
    PointStore* pA = (PointStore*)a;
    PointStore* pB = (PointStore*)b;

    if (pA->m_dist < pB->m_dist) return -1;
    if (pA->m_dist > pB->m_dist) return 1;
    return 0;
}
static int compare_doubles_ascending(const void* a, const void* b)
{
    return compare_doubles(a, b, 1);
}
int grib_unpack_double_element_set(grib_accessor* a, const size_t* index_array, size_t len, double* val_array)
{
    grib_accessor_class* c = a->cclass;
    DebugAssert(len > 0);
    while (c) {
        if (c->unpack_double_element_set) {
            return c->unpack_double_element_set(a, index_array, len, val_array);
        }
        c = c->super ? *(c->super) : NULL;
    }
    return GRIB_NOT_IMPLEMENTED;
}
static int compare_doubles(const void* a, const void* b, int ascending)
{
    /* ascending is a boolean: 0 or 1 */
    double* arg1 = (double*)a;
    double* arg2 = (double*)b;
    if (ascending) {
        if (*arg1 < *arg2)
            return -1; /*Smaller values come before larger ones*/
    }
    else {
        if (*arg1 > *arg2)
            return -1; /*Larger values come before smaller ones*/
    }
    if (*arg1 == *arg2)
        return 0;
    else
        return 1;
}
static double fraction_operator_double(Fraction_type self)
{
    return (double)self.top_ / (double)self.bottom_;
}
static Fraction_value_type get_min(Fraction_value_type a, Fraction_value_type b)
{
    return ((a < b) ? a : b);
}
static int fraction_operator_greater_than(Fraction_type self, Fraction_type other)
{
    int overflow = 0;
    int result   = fraction_mul(&overflow, self.top_, other.bottom_) > fraction_mul(&overflow, other.top_, self.bottom_);
    if (overflow) {
        double d1 = fraction_operator_double(self);
        double d2 = fraction_operator_double(other);
        return (d1 > d2);
        /* return double(*this) > double(other);*/
    }
    return result;
}
static Fraction_value_type fraction_mul(int* overflow, Fraction_value_type a, Fraction_value_type b)
{
    if (*overflow) {
        return 0;
    }

    if (b != 0) {
        *overflow = llabs(a) > (ULLONG_MAX / llabs(b));
    }
    return a * b;
}
static int fraction_operator_less_than(Fraction_type self, Fraction_type other)
{
    int overflow = 0;
    int result   = fraction_mul(&overflow, self.top_, other.bottom_) < fraction_mul(&overflow, other.top_, self.bottom_);
    if (overflow) {
        double d1 = fraction_operator_double(self);
        double d2 = fraction_operator_double(other);
        return (d1 < d2);
        /* return double(*this) < double(other); */
    }
    return result;
}
static Fraction_type fraction_operator_multiply_n_Frac(Fraction_value_type n, Fraction_type f)
{
    Fraction_type ft     = fraction_construct_from_long_long(n);
    Fraction_type result = fraction_operator_multiply(ft, f);
    return result;
    /*return Fraction(n) * f;*/
}
static Fraction_value_type fraction_integralPart(const Fraction_type frac)
{
    Assert(frac.bottom_);
    if (frac.bottom_ == 0) return frac.top_;
    return frac.top_ / frac.bottom_;
}
static Fraction_type fraction_construct_from_long_long(long long n)
{
    Fraction_type result;
    result.top_    = n;
    result.bottom_ = 1;
    return result;
}
static Fraction_type fraction_operator_multiply(Fraction_type self, Fraction_type other)
{
    int overflow = 0; /*boolean*/

    Fraction_value_type top    = fraction_mul(&overflow, self.top_, other.top_);
    Fraction_value_type bottom = fraction_mul(&overflow, self.bottom_, other.bottom_);

    if (!overflow) {
        return fraction_construct(top, bottom);
    }
    else {
        /* Fallback option */
        /*return Fraction(double(*this) * double(other));*/
        double d1        = fraction_operator_double(self);
        double d2        = fraction_operator_double(other);
        Fraction_type f1 = fraction_construct_from_double(d1 * d2);
        return f1;
    }
}
static Fraction_type fraction_construct(Fraction_value_type top, Fraction_value_type bottom)
{
    Fraction_type result;

    /* @note in theory we also assume that numerator and denominator are both representable in
     * double without loss
     *   ASSERT(top == Fraction_value_type(double(top)));
     *   ASSERT(bottom == Fraction_value_type(double(bottom)));
     */
    Fraction_value_type g;
    Fraction_value_type sign = 1;
    Assert(bottom != 0);
    if (top < 0) {
        top  = -top;
        sign = -sign;
    }

    if (bottom < 0) {
        bottom = -bottom;
        sign   = -sign;
    }

    g = fraction_gcd(top, bottom);
    if (g != 0) {
        top    = top / g;
        bottom = bottom / g;
    }

    result.top_    = sign * top;
    result.bottom_ = bottom;
    return result;
}
static Fraction_type fraction_operator_divide(Fraction_type self, Fraction_type other)
{
    int overflow = 0; /*boolean*/

    Fraction_value_type top    = fraction_mul(&overflow, self.top_, other.bottom_);
    Fraction_value_type bottom = fraction_mul(&overflow, self.bottom_, other.top_);

    if (!overflow) {
        return fraction_construct(top, bottom);
    }
    else {
        /*Fallback option*/
        /*return Fraction(double(*this) / double(other));*/
        double d1        = fraction_operator_double(self);
        double d2        = fraction_operator_double(other);
        Fraction_type f1 = fraction_construct_from_double(d1 / d2);
        return f1;
    }
}
int grib_action_notify_change(grib_action* a, grib_accessor* observer, grib_accessor* observed)
{
    grib_action_class* c = a->cclass;

    /*GRIB_MUTEX_INIT_ONCE(&once,&init_mutex);*/
    /*GRIB_MUTEX_LOCK(&mutex1);*/

    init(c);
    while (c) {
        if (c->notify_change) {
            int result = c->notify_change(a, observer, observed);
            /*GRIB_MUTEX_UNLOCK(&mutex1);*/
            return result;
        }
        c = c->super ? *(c->super) : NULL;
    }
    /*GRIB_MUTEX_UNLOCK(&mutex1);*/
    DebugAssert(0);
    return 0;
}
void grib_dependency_remove_observer(grib_accessor* observer)
{
    grib_handle* h     = NULL;
    grib_dependency* d = NULL;

    if (!observer)
        return;

    h = handle_of(observer);
    d = h->dependencies;

    while (d) {
        if (d->observer == observer) {
            d->observer = 0;
        }
        d = d->next;
    }
}
void grib_dependency_remove_observed(grib_accessor* observed)
{
    grib_handle* h     = handle_of(observed);
    grib_dependency* d = h->dependencies;
    /* printf("%s\n",observed->name); */

    while (d) {
        if (d->observed == observed) {
            /*  TODO: Notify observer...*/
            d->observed = 0; /*printf("grib_dependency_remove_observed %s\n",observed->name); */
        }
        d = d->next;
    }
}
void grib_buffer_replace(grib_accessor* a, const unsigned char* data,
                         size_t newsize, int update_lengths, int update_paddings)
{
    size_t offset = a->offset;
    long oldsize  = grib_get_next_position_offset(a) - offset;
    long increase = (long)newsize - (long)oldsize;

    grib_buffer* buffer   = grib_handle_of_accessor(a)->buffer;
    size_t message_length = buffer->ulength;

    grib_context_log(a->context, GRIB_LOG_DEBUG,
                     "grib_buffer_replace %s offset=%ld oldsize=%ld newsize=%ld message_length=%ld update_paddings=%d",
                     a->name, (long)offset, oldsize, (long)newsize, (long)message_length, update_paddings);

    grib_buffer_set_ulength(a->context,
                            buffer,
                            buffer->ulength + increase);

    /* move the end */
    if (increase)
        memmove(
            buffer->data + offset + newsize,
            buffer->data + offset + oldsize,
            message_length - offset - oldsize);

    /* copy new data */
    DebugAssert(buffer->data + offset);
    DebugAssert(data || (newsize == 0)); /* if data==NULL then newsize must be 0 */
    if (data) {
        /* Note: memcpy behaviour is undefined if either dest or src is NULL */
        memcpy(buffer->data + offset, data, newsize);
    }

    if (increase) {
        update_offsets_after(a, increase);
        if (update_lengths) {
            grib_update_size(a, newsize);
            grib_section_adjust_sizes(grib_handle_of_accessor(a)->root, 1, 0);
            if (update_paddings)
                grib_update_paddings(grib_handle_of_accessor(a)->root);
        }
    }
}
long grib_byte_count(grib_accessor* a)
{
    grib_accessor_class* c = NULL;
    if (a)
        c = a->cclass;

    while (c) {
        if (c->byte_count)
            return c->byte_count(a);
        c = c->super ? *(c->super) : NULL;
    }
    DebugAssert(0);
    return 0;
}
void grib_dump_bytes(grib_dumper* d, grib_accessor* a, const char* comment)
{
    grib_dumper_class* c = d->cclass;
    while (c) {
        if (c->dump_bytes) {
            c->dump_bytes(d, a, comment);
            return;
        }
        c = c->super ? *(c->super) : NULL;
    }
    Assert(0);
}
void grib_dump_long(grib_dumper* d, grib_accessor* a, const char* comment)
{
    grib_dumper_class* c = d->cclass;
    while (c) {
        if (c->dump_long) {
            c->dump_long(d, a, comment);
            return;
        }
        c = c->super ? *(c->super) : NULL;
    }
    Assert(0);
}
void grib_dump_double(grib_dumper* d, grib_accessor* a, const char* comment)
{
    grib_dumper_class* c = d->cclass;
    while (c) {
        if (c->dump_double) {
            c->dump_double(d, a, comment);
            return;
        }
        c = c->super ? *(c->super) : NULL;
    }
    Assert(0);
}
void grib_dump_string(grib_dumper* d, grib_accessor* a, const char* comment)
{
    grib_dumper_class* c = d->cclass;
    while (c) {
        if (c->dump_string) {
            c->dump_string(d, a, comment);
            return;
        }
        c = c->super ? *(c->super) : NULL;
    }
    Assert(0);
}
void grib_update_paddings(grib_section* s)
{
    grib_accessor* last = NULL;
    grib_accessor* changed;

    /* while((changed = find_paddings(s)) != NULL) */
    while ((changed = find_paddings(s->h->root)) != NULL) {
        Assert(changed != last);
        grib_resize(changed, grib_preferred_size(changed, 0));
        last = changed;
    }
}
grib_accessor* find_paddings(grib_section* s)
{
    grib_accessor* a = s ? s->block->first : NULL;

    while (a) {
        /* grib_accessor* p = find_paddings(grib_get_sub_section(a)); */
        grib_accessor* p = find_paddings(a->sub_section);
        if (p)
            return p;

        if (grib_preferred_size(a, 0) != a->length)
            return a;

        a = a->next;
    }

    return NULL;
}
void grib_update_size(grib_accessor* a, size_t len)
{
    grib_accessor_class* c = a->cclass;
    /*grib_context_log(a->context, GRIB_LOG_DEBUG, "(%s)%s is packing (double) %g",(a->parent->owner)?(a->parent->owner->name):"root", a->name ,v?(*v):0); */
    while (c) {
        if (c->update_size) {
            c->update_size(a, len);
            return;
        }
        c = c->super ? *(c->super) : NULL;
    }
    DebugAssert(0);
}
static void update_offsets(grib_accessor* a, long len)
{
    while (a) {
        grib_section* s = a->sub_section;
        a->offset += len;
        grib_context_log(a->context, GRIB_LOG_DEBUG, "::::: grib_buffer : accessor %s is moving by %d bytes to %ld", a->name, len, a->offset);
        if (s)
            update_offsets(s->block->first, len);
        a = a->next;
    }
}

static void update_offsets_after(grib_accessor* a, long len)
{
    while (a) {
        update_offsets(a->next, len);
        a = a->parent->owner;
    }
}
size_t grib_preferred_size(grib_accessor* a, int from_handle)
{
    grib_accessor_class* c = a->cclass;
    /*grib_context_log(a->context, GRIB_LOG_DEBUG, "(%s)%s is packing (long) %d",(a->parent->owner)?(a->parent->owner->name):"root", a->name ,v?(*v):0); */
    while (c) {
        if (c->preferred_size) {
            return c->preferred_size(a, from_handle);
        }
        c = c->super ? *(c->super) : NULL;
    }
    DebugAssert(0);
    return 0;
}
void grib_resize(grib_accessor* a, size_t new_size)
{
    grib_accessor_class* c = a->cclass;
    /*grib_context_log(a->context, GRIB_LOG_DEBUG, "(%s)%s is packing (long) %d",(a->parent->owner)?(a->parent->owner->name):"root", a->name ,v?(*v):0); */
    while (c) {
        if (c->resize) {
            c->resize(a, new_size);
            return;
        }
        c = c->super ? *(c->super) : NULL;
    }
    DebugAssert(0);
    return;
}
void grib_buffer_set_ulength(const grib_context* c, grib_buffer* b, size_t length)
{
    grib_grow_buffer(c, b, length);
    b->ulength      = length;
    b->ulength_bits = length * 8;
}

