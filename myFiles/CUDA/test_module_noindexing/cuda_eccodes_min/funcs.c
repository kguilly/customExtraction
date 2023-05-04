#include "header.h"


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <stdarg.h>
#include <errno.h>

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

grib_iterator* grib_iterator_factory(grib_handle* h, grib_arguments* args, unsigned long flags, int* ret)
{
    int i;
    const char* type = (char*)grib_arguments_get_name(h, args, 0);

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

int codes_get_long(const grib_handle* h, const char* key, long* value)
{
    return grib_get_long(h, key, value);
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

void grib_context_increment_handle_file_count(grib_context* c)
{
    if (!c)
        c = grib_context_get_default();
    GRIB_MUTEX_INIT_ONCE(&once, &init);
    GRIB_MUTEX_LOCK(&mutex_c);
    c->handle_file_count++;
    GRIB_MUTEX_UNLOCK(&mutex_c);
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

void grib_section_delete(grib_context* c, grib_section* b)
{
    if (!b)
        return;

    grib_empty_section(c, b);
    grib_context_free(c, b->block);
    /* printf("++++ deleted %p\n",b); */
    grib_context_free(c, b);
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

static void* allocate_buffer(void* data, size_t* length, int* err)
{
    alloc_buffer* u = (alloc_buffer*)data;
    u->buffer       = malloc(*length);
    u->size         = *length;
    if (u->buffer == NULL)
        *err = GRIB_OUT_OF_MEMORY; /* Cannot allocate buffer */
    return u->buffer;
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

static int _read_any(reader* r, int grib_ok, int bufr_ok, int hdf5_ok, int wrap_ok)
{
    unsigned char c;
    int err             = 0;
    unsigned long magic = 0;

    while (r->read(r->read_data, &c, 1, &err) == 1 && err == 0) {
        magic <<= 8;
        magic |= c;

        if (grib_ok) {
            err = read_GRIB(r);
            return err == GRIB_END_OF_FILE ? GRIB_PREMATURE_END_OF_FILE : err; /* Premature EOF */
        }
        break;
    }

    return err;
}

static int read_any(reader* r, int grib_ok, int bufr_ok, int hdf5_ok, int wrap_ok)
{
    int result = 0;
    result = _read_any(r, grib_ok, bufr_ok, hdf5_ok, wrap_ok);
    return result;
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

    /* Assert(i <= buf->length); */
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


