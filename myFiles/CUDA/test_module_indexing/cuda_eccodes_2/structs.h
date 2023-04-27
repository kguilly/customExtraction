
#ifndef STRUCTS_H
#define STRUCTS_H

#include <stdio.h>

#define MAX_SET_VALUES 10
#define ACCESSORS_ARRAY_SIZE 5000
#define MAX_NUM_SECTIONS 12
#define STRING_VALUE_LEN 100



typedef struct grib_handle grib_handle;
typedef struct grib_field_tree grib_field_tree;
typedef struct grib_index_key grib_index_key;
typedef struct grib_field_list grib_field_list;
typedef struct grib_index grib_index;
typedef struct grib_field grib_field;
typedef struct grib_file grib_file;
typedef struct grib_context grib_context;



typedef enum ProductKind
{
    PRODUCT_ANY,
    PRODUCT_GRIB,
    PRODUCT_BUFR,
    PRODUCT_METAR,
    PRODUCT_GTS,
    PRODUCT_TAF
} ProductKind;

struct grib_file
{
    grib_context* context;
    char* name;
    FILE* handle;
    char* mode;
    char* buffer;
    long refcount;
    grib_file* next;
    short id;
};

struct grib_field
{
    grib_file* file;
    off_t offset;
    long length;
    grib_field* next;
};

struct grib_handle
{
    grib_context* context;         /** < context attached to this handle    */
    grib_buffer* buffer;           /** < buffer attached to the handle      */
    grib_section* root;            /**  the root      section*/
    grib_section* asserts;         /** the assertion section*/
    grib_section* rules;           /** the rules     section*/
    grib_dependency* dependencies; /** List of dependencies */
    grib_handle* main;             /** Used during reparsing */
    grib_handle* kid;              /** Used during reparsing */
    grib_loader* loader;           /** Used during reparsing */
    int values_stack;
    const grib_values* values[MAX_SET_VALUES]; /** Used when setting multiple values at once */
    size_t values_count[MAX_SET_VALUES];       /** Used when setting multiple values at once */
    int dont_trigger;                          /** Don't notify triggers */
    int partial;                               /** Not a complete message (just headers) */
    int header_mode;                           /** Header not jet complete */
    char* gts_header;
    size_t gts_header_len;
    int use_trie;
    int trie_invalid;
    grib_accessor* accessors[ACCESSORS_ARRAY_SIZE];
    char* section_offset[MAX_NUM_SECTIONS];
    char* section_length[MAX_NUM_SECTIONS];
    int sections_count;
    off_t offset;
    long bufr_subset_number; /* bufr subset number */
    long bufr_group_number;  /* used in bufr */
    /* grib_accessor* groups[MAX_NUM_GROUPS]; */
    long missingValueLong;
    double missingValueDouble;
    ProductKind product_kind;
    /* grib_trie* bufr_elements_table; */
};

struct grib_context
{
    int inited;
    int debug;
    int write_on_fail;
    int no_abort;
    int io_buffer_size;
    int no_big_group_split;
    int no_spd;
    int keep_matrix;
    char* grib_definition_files_path;
    char* grib_samples_path;
    char* grib_concept_path;

    grib_action_file_list* grib_reader;
    void* user_data;
    int real_mode;

    grib_free_proc free_mem;
    grib_malloc_proc alloc_mem;
    grib_realloc_proc realloc_mem;

    grib_free_proc free_persistent_mem;
    grib_malloc_proc alloc_persistent_mem;

    grib_free_proc free_buffer_mem;
    grib_malloc_proc alloc_buffer_mem;
    grib_realloc_proc realloc_buffer_mem;

    grib_data_read_proc read;
    grib_data_write_proc write;
    grib_data_tell_proc tell;
    grib_data_seek_proc seek;
    grib_data_eof_proc eof;

    grib_log_proc output_log;
    grib_print_proc print;

    grib_codetable* codetable;
    grib_smart_table* smart_table;
    char* outfilename;
    int multi_support_on;
    grib_multi_support* multi_support;
    grib_string_list* grib_definition_files_dir;
    int handle_file_count;
    int handle_total_count;
    off_t message_file_offset;
    int no_fail_on_wrong_length;
    int gts_header_on;
    int gribex_mode_on;
    int large_constant_fields;
    grib_itrie* keys;
    int keys_count;
    grib_itrie* concepts_index;
    int concepts_count;
    grib_concept_value* concepts[MAX_NUM_CONCEPTS];
    grib_itrie* hash_array_index;
    int hash_array_count;
    grib_hash_array_value* hash_array[MAX_NUM_HASH_ARRAY];
    grib_trie* def_files;
    grib_string_list* blocklist;
    int ieee_packing; /* 32 or 64 */
    int bufrdc_mode;
    int bufr_set_to_missing_if_out_of_range;
    int bufr_multi_element_constant_arrays;
    int grib_data_quality_checks;
    FILE* log_stream;
    grib_trie* classes;
    grib_trie* lists;
    grib_trie* expanded_descriptors;
    int file_pool_max_opened_files;
#if GRIB_PTHREADS
    pthread_mutex_t mutex;
#elif GRIB_OMP_THREADS
    omp_nest_lock_t mutex;
#endif
};

struct grib_field_tree
{
    grib_field* field;
    char* value;
    grib_field_tree* next;
    grib_field_tree* next_level;
};

struct grib_index_key
{
    char* name;
    int type;
    char value[STRING_VALUE_LEN];
    grib_string_list* values;
    grib_string_list* current;
    int values_count;
    int count;
    grib_index_key* next;
};

struct grib_field_list
{
    grib_field* field;
    grib_field_list* next;
};
struct grib_index
{
    grib_context* context;
    grib_index_key* keys;
    int rewind;
    int orderby;
    grib_index_key* orederby_keys;
    grib_field_tree* fields;
    grib_field_list* fieldset;
    grib_field_list* current;
    grib_file* files;
    int count;
    ProductKind product_kind;
    int unpack_bufr; /* Only meaningful for product_kind of BUFR */
};


#endif

