#include "header.h"


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

static grib_action_class def_grib_action_class_noop = {
        0,                              /* super                     */
    "action_class_noop",                              /* name                      */
    sizeof(grib_action_noop),            /* size                      */
    0,                                   /* inited */
    &init_class_gac,                         /* init_class */
    0,                               /* init                      */
    &destroy_gac,                            /* destroy */

    &dump_gac,                               /* dump                      */
    &xref_gac,                               /* xref                      */

    0,             /* create_accessor*/

    0,                            /* notify_change */
    0,                            /* reparse */
    &execute_gac,                            /* execute */
};
grib_action_class* grib_action_class_noop = &def_grib_action_class_noop;
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
static const struct table_entry table[] = {
{ "gaussian", &grib_iterator_class_gaussian, },
{ "gen", &grib_iterator_class_gen, },
{ "lambert_conformal", &grib_iterator_class_lambert_conformal, },
{ "latlon", &grib_iterator_class_latlon, },
{ "regular", &grib_iterator_class_regular, },
};

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
    for (i = 0; i < NUMBER(table); i++)
        if (strcmp(type, table[i].type) == 0) {
            grib_iterator_class* c = *(table[i].cclass);
            grib_iterator* it      = (grib_iterator*)grib_context_malloc_clear(h->context, c->size);
            it->cclass             = c;
            it->flags              = flags;
            *ret                   = GRIB_SUCCESS;
            *ret                   = grib_iterator_init(it, h, args);
            if (*ret == GRIB_SUCCESS)
                return it;
            grib_context_log(h->context, GRIB_LOG_ERROR, "Geoiterator factory: Error instantiating iterator %s (%s)",
                             table[i].type, grib_get_error_message(*ret));
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

static GRIB_INLINE int grib_inline_strcmp(const char* a, const char* b)
{
    if (*a != *b)
        return 1;
    while ((*a != 0 && *b != 0) && *(a) == *(b)) {
        a++;
        b++;
    }
    return (*a == 0 && *b == 0) ? 0 : 1;
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

int grib_is_defined(const grib_handle* h, const char* name)
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

static int parse(grib_context* gc, const char* filename)
{
    int err = 0;
    GRIB_MUTEX_INIT_ONCE(&once, &init);
    GRIB_MUTEX_LOCK(&mutex_parse);

    #ifdef YYDEBUG
    {
        extern int grib_yydebug;
        grib_yydebug = getenv("YYDEBUG") != NULL;
    }
    #endif

    gc = gc ? gc : grib_context_get_default();

    // grib_yyin  = NULL;
    FILE* grib_yyin = NULL;
    top        = 0;
    // parse_file = 0;
    const char* parse_file = 0;
    // grib_parser_include(filename);
    // if (!grib_yyin) {
    //     /* Could not read from file */
    //     parse_file = 0;
    //     GRIB_MUTEX_UNLOCK(&mutex_parse);
    //     return GRIB_FILE_NOT_FOUND;
    // }
    // TODO: may need to change this. function was very complex
    //       in grib_yacc.c
    // err        = grib_yyparse();
    err = 0;
    //  parse_file = 0;

    if (err)
        grib_context_log(gc, GRIB_LOG_ERROR, "Parsing error: %s, file: %s\n",
                grib_get_error_message(err), filename);

    GRIB_MUTEX_UNLOCK(&mutex_parse);
    return err;
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


