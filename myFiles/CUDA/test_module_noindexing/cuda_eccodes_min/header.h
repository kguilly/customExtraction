#ifndef HEADER_H
#define HEADER_H


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>


#define ACCESSORS_ARRAY_SIZE 5000
#define Assert(a) \
    do {                                                          \
        if (!(a)) codes_assertion_failed(#a, __FILE__, __LINE__); \
    } while (0)
#define DebugAssert(a) Assert(a)
#define DebugAssert(a)
#define DEFAULT_FILE_POOL_MAX_OPENED_FILES 0
#define ECC_PATH_DELIMITER_CHAR ';'
#define ECC_PATH_MAXLEN 8192
#define GRIB_DECODING_ERROR -13
#define GRIB_IO_PROBLEM -11
#define GRIB_END_OF_FILE -1
#define GRIB_LOG_DEBUG 4
#define GRIB_LOG_ERROR 2
#define GRIB_LOG_FATAL 3
#define GRIB_LOG_INFO 0
#define GRIB_LOG_WARNING 1
#define GRIB_MUTEX_INIT_ONCE(a, b)
#define GRIB_MUTEX_LOCK(a)
#define GRIB_MUTEX_UNLOCK(a)
#define GRIB_MY_BUFFER 0
#define GRIB_NOT_FOUND -10
#define GRIB_REAL_MODE8 8
#define GRIB_SUCCESS 0
#define GRIB_WRONG_LENGTH -23
#define ITRIE_SIZE 40
#define MAX_ACCESSOR_NAMES 20
#define MAX_ACCESSOR_ATTRIBUTES 20
#define MAX_NUM_CONCEPTS 2000
#define MAX_NUM_HASH_ARRAY 2000
#define MAX_NUM_SECTIONS 12
#define MAX_SET_VALUES 10
#define MAX_SMART_TABLE_COLUMNS 20
#define STRING_VALUE_LEN 100
#define TRIE_SIZE 39

/* structs & enums */
typedef enum ProductKind
{
    PRODUCT_ANY,
    PRODUCT_GRIB,
    PRODUCT_BUFR,
    PRODUCT_METAR,
    PRODUCT_GTS,
    PRODUCT_TAF
} ProductKind;

typedef struct codes_condition codes_condition;
typedef struct code_table_entry code_table_entry;
typedef struct grib_accessors_list grib_accessors_list;
typedef struct grib_accessor grib_accessor;
typedef struct grib_accessor_class grib_accessor_class;
typedef struct grib_action grib_action;
typedef struct grib_action_class grib_action_class;
typedef struct grib_action_file grib_action_file;
typedef struct grib_action_file_list grib_action_file_list;
typedef struct grib_arguments grib_arguments;
typedef struct grib_block_of_accessors grib_block_of_accessors;
typedef struct grib_buffer grib_buffer;
typedef struct grib_codetable grib_codetable;
typedef struct grib_concept_condition grib_concept_condition;
typedef struct grib_concept_value grib_concept_value;
typedef struct grib_context grib_context;
typedef struct grib_dependency grib_dependency;
typedef struct grib_dumper grib_dumper;
typedef struct grib_dumper_class grib_dumper_class;
typedef struct grib_expression grib_expression;
typedef struct grib_expression_class grib_expression_class;
typedef struct grib_handle grib_handle;
typedef struct grib_hash_array_value grib_hash_array_value;
typedef struct grib_iarray grib_iarray;
typedef struct grib_iterator grib_iterator;
typedef struct grib_iterator_class grib_iterator_class;
typedef struct grib_itrie grib_itrie;
typedef struct grib_loader grib_loader;
typedef struct grib_multi_support grib_multi_support;
typedef struct grib_section grib_section;
typedef struct grib_smart_table grib_smart_table;
typedef struct grib_smart_table_entry grib_smart_table_entry;
typedef struct grib_string_list grib_string_list;
typedef struct grib_trie grib_trie;
typedef struct grib_values grib_values;
typedef struct grib_virtual_value grib_virtual_value;



struct codes_condition
{
    char* left;
    int rightType;
    char* rightString;
    long rightLong;
    double rightDouble;
};

struct code_table_entry
{
    char* abbreviation;
    char* title;
    char* units;
};

struct grib_accessors_list
{
    grib_accessor* accessor;
    int rank;
    grib_accessors_list* next;
    grib_accessors_list* prev;
    grib_accessors_list* last;
};

struct grib_accessor
{
    const char* name;       /** < name of the accessor                       */
    const char* name_space; /** < namespace to which the accessor belongs    */
    grib_context* context;
    grib_handle* h;
    grib_action* creator;        /** < action that created the accessor           */
    long length;                 /** < byte length of the accessor                */
    long offset;                 /** < offset of the data in the buffer           */
    grib_section* parent;        /** < section to which the accessor is attached  */
    grib_accessor* next;         /** < next accessor in list                      */
    grib_accessor* previous;     /** < next accessor in list                      */
    grib_accessor_class* cclass; /** < behaviour of the accessor                  */
    unsigned long flags;         /** < Various flags                              */
    grib_section* sub_section;

    const char* all_names[MAX_ACCESSOR_NAMES];       /** < name of the accessor */
    const char* all_name_spaces[MAX_ACCESSOR_NAMES]; /** < namespace to which the accessor belongs */
    int dirty;

    grib_accessor* same;        /** < accessors with the same name */
    long loop;                  /** < used in lists */
    long bufr_subset_number;    /** < bufr subset (bufr data accessors belong to different subsets)*/
    long bufr_group_number;     /** < used in bufr */
    grib_virtual_value* vvalue; /** < virtual value used when transient flag on **/
    const char* set;
    grib_accessor* attributes[MAX_ACCESSOR_ATTRIBUTES]; /** < attributes are accessors */
    grib_accessor* parent_as_attribute;
};

struct grib_accessor_class
{
    grib_accessor_class** super;
    const char* name;
    size_t size;

    int inited;
    accessor_init_class_proc init_class;

    accessor_init_proc init;
    accessor_post_init_proc post_init;
    accessor_destroy_proc destroy;

    accessor_dump_proc dump;
    accessor_value_proc next_offset;

    accessor_string_proc string_length;
    accessor_value_with_ret_proc value_count;

    accessor_value_proc byte_count;
    accessor_value_proc byte_offset;

    accessor_get_native_type_proc get_native_type;

    accessor_sub_section_proc sub_section;

    accessor_pack_missing_proc pack_missing;
    accessor_pack_is_missing_proc is_missing;

    accessor_pack_long_proc pack_long;
    accessor_unpack_long_proc unpack_long;

    accessor_pack_double_proc pack_double;
    accessor_unpack_double_proc unpack_double;

    accessor_pack_string_proc pack_string;
    accessor_unpack_string_proc unpack_string;

    accessor_pack_string_array_proc pack_string_array;
    accessor_unpack_string_array_proc unpack_string_array;

    accessor_pack_bytes_proc pack_bytes;
    accessor_unpack_bytes_proc unpack_bytes;

    accessor_pack_expression_proc pack_expression;

    accessor_notify_change_proc notify_change;
    accessor_update_size_proc update_size;

    accessor_preferred_size_proc preferred_size;
    accessor_resize_proc resize;

    accessor_nearest_proc nearest_smaller_value;
    accessor_next_proc next;
    accessor_compare_proc compare;
    accessor_unpack_double_element_proc unpack_double_element;
    accessor_unpack_double_element_set_proc unpack_double_element_set;
    accessor_unpack_double_subarray_proc unpack_double_subarray;
    accessor_clear_proc clear;
    accessor_clone_proc make_clone;
};

struct grib_action
{
    char* name;                /**  name of the definition statement */
    char* op;                  /**  operator of the definition statement */
    char* name_space;          /**  namespace of the definition statement */
    grib_action* next;         /**  next action in the list */
    grib_action_class* cclass; /**  link to the structure containing a specific behaviour */
    grib_context* context;     /**  Context */
    unsigned long flags;
    char* defaultkey;              /** name of the key used as default if not found */
    grib_arguments* default_value; /** default expression as in .def file */
    char* set;
    char* debug_info; /** purely for debugging and tracing */
};

struct grib_action_class
{
    grib_action_class** super; /** < link to a more general behaviour */
    const char* name;          /** < name of the behaviour class */
    size_t size;               /** < size in bytes of the structure */

    int inited;
    action_init_class_proc init_class;

    action_init_proc init;
    action_destroy_proc destroy; /** < destructor method to release the memory */

    grib_dump_proc dump;                                 /** < dump method of the action  */
    grib_xref_proc xref;                                 /** < dump method of the action  */
    action_create_accessors_handle_proc create_accessor; /** < method to create the corresponding accessor from a handle*/
    action_notify_change_proc notify_change;             /** < method to create the corresponding accessor from a handle*/

    action_reparse_proc reparse;
    action_execute_proc execute;
};

struct grib_action_file
{
    char* filename;
    grib_action* root;
    grib_action_file* next;
};

struct grib_action_file_list
{
    grib_action_file* first;
    grib_action_file* last;
};

struct grib_arguments
{
    struct grib_arguments* next;
    grib_expression* expression;
};

struct grib_block_of_accessors
{
    grib_accessor* first;
    grib_accessor* last;
};

struct grib_buffer
{
    int property;        /** < property parameter of buffer         */
    int validity;        /** < validity parameter of buffer         */
    int growable;        /** < buffer can be grown                  */
    size_t length;       /** < Buffer length                        */
    size_t ulength;      /** < length used of the buffer            */
    size_t ulength_bits; /** < length used of the buffer in bits  */
    unsigned char* data; /** < the data byte array                  */
};

struct grib_codetable
{
    char* filename[2];
    char* recomposed_name[2];
    grib_codetable* next;
    size_t size;
    code_table_entry entries[1];
};

struct grib_concept_condition
{
    grib_concept_condition* next;
    char* name;
    grib_expression* expression;
    grib_iarray* iarray;
};

struct grib_concept_value
{
    grib_concept_value* next;
    char* name;
    grib_concept_condition* conditions;
    grib_trie* index;
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
};

struct grib_darray
{
    double* v;
    size_t size; /* capacity */
    size_t n;    /* used size */
    size_t incsize;
    grib_context* context;
};

struct grib_dependency
{
    grib_dependency* next;
    grib_accessor* observed;
    grib_accessor* observer;
    int run;
};

struct grib_dumper
{
    FILE* out;
    unsigned long option_flags;
    void* arg;
    int depth;
    long count;
    grib_context* context;
    grib_dumper_class* cclass;
};

struct grib_dumper_class
{
    grib_dumper_class** super;
    const char* name;
    size_t size;
    int inited;
    dumper_init_class_proc init_class;
    dumper_init_proc init;
    dumper_destroy_proc destroy;
    dumper_dump_proc dump_long;
    dumper_dump_proc dump_double;
    dumper_dump_proc dump_string;
    dumper_dump_proc dump_string_array;
    dumper_dump_proc dump_label;
    dumper_dump_proc dump_bytes;
    dumper_dump_proc dump_bits;
    dumper_dump_section_proc dump_section;
    dumper_dump_values_proc dump_values;
    dumper_header_proc header;
    dumper_footer_proc footer;
};

struct grib_expression
{
    grib_expression_class* cclass;
};

struct grib_expression_class
{
    grib_expression_class** super;
    const char* name;
    size_t size;
    int inited;

    expression_class_init_proc init_class;
    expression_init_proc init;
    expression_destroy_proc destroy;


    expression_print_proc print;
    expression_add_dependency_proc add_dependency;

    expression_native_type_proc native_type;
    expression_get_name_proc get_name;

    expression_evaluate_long_proc evaluate_long;
    expression_evaluate_double_proc evaluate_double;
    expression_evaluate_string_proc evaluate_string;
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
    // long bufr_subset_number; /* bufr subset number */
    // long bufr_group_number;  /* used in bufr */
    /* grib_accessor* groups[MAX_NUM_GROUPS]; */
    long missingValueLong;
    double missingValueDouble;
    ProductKind product_kind;
    /* grib_trie* bufr_elements_table; */
};

struct grib_hash_array_value
{
    grib_hash_array_value* next;
    char* name;
    int type;
    grib_iarray* iarray;
    grib_darray* darray;
    grib_trie* index;
};

struct grib_iarray
{
    long* v;
    size_t size; /* capacity */
    size_t n;    /* used size */
    size_t incsize;
    size_t number_of_pop_front;
    grib_context* context;
};

struct grib_iterator
{
    grib_arguments* args; /**  args of iterator   */
    grib_handle* h;
    long e;       /**  current element    */
    size_t nv;    /**  number of values   */
    double* data; /**  data values        */
    grib_iterator_class* cclass;
    unsigned long flags;
};

struct grib_iterator_class
{
    grib_iterator_class** super;
    const char* name;
    size_t size;

    int inited;
    iterator_init_class_proc init_class;

    iterator_init_proc init;
    iterator_destroy_proc destroy;

    iterator_next_proc next;
    iterator_previous_proc previous;
    iterator_reset_proc reset;
    iterator_has_next_proc has_next;
};

struct grib_itrie
{
    grib_itrie* next[ITRIE_SIZE];
    grib_context* context;
    int id;
    int* count;
};

struct grib_multi_support
{
    FILE* file;
    size_t offset;
    unsigned char* message;
    size_t message_length;
    unsigned char* sections[8];
    unsigned char* bitmap_section;
    size_t bitmap_section_length;
    size_t sections_length[9];
    int section_number;
    grib_multi_support* next;
};

struct grib_loader
{
    void* data;
    grib_loader_init_accessor_proc init_accessor;
    grib_loader_lookup_long_proc lookup_long;
    int list_is_resized; /** will be true if we resize a list */
    int changing_edition;
};

struct grib_section
{
    grib_accessor* owner;
    grib_handle* h;                 /** < Handles of all accessors and buffer  */
    grib_accessor* aclength;        /** < block of the length of the block     */
    grib_block_of_accessors* block; /** < block                                */
    grib_action* branch;            /** < branch that created the block        */
    size_t length;
    size_t padding;
};

struct grib_smart_table
{
    char* filename[3];
    char* recomposed_name[3];
    grib_smart_table* next;
    size_t numberOfEntries;
    grib_smart_table_entry* entries;
};

struct grib_smart_table_entry
{
    /*int   code;*/
    char* abbreviation;
    char* column[MAX_SMART_TABLE_COLUMNS];
};

struct grib_string_list
{
    char* value;
    int count;
    grib_string_list* next;
};

struct grib_trie
{
    grib_trie* next[TRIE_SIZE];
    grib_context* context;
    int first;
    int last;
    void* data;
};

struct grib_values
{
    const char* name;
    int type;
    long long_value;
    double double_value;
    const char* string_value;
    int error;
    int has_value;
    int equal;
    grib_values* next;
};

struct grib_virtual_value
{
    long lval;
    double dval;
    char* cval;
    int missing;
    int length;
    int type;
};


/* initialization headers */
typedef int (*action_create_accessors_handle_proc)(grib_section*, grib_action*, grib_loader*);
typedef void (*action_destroy_proc)(grib_context*, grib_action*);
typedef int (*action_execute_proc)(grib_action*, grib_handle*);
typedef void (*action_init_class_proc)(grib_action_class*);
typedef void (*action_init_proc)(grib_action*);
typedef int (*action_notify_change_proc)(grib_action*, grib_accessor*, grib_accessor*);
typedef grib_action* (*action_reparse_proc)(grib_action*, grib_accessor*, int*);

typedef int (*accessor_clear_proc)(grib_accessor*);
typedef grib_accessor* (*accessor_clone_proc)(grib_accessor*, grib_section*, int*);
typedef int (*accessor_compare_proc)(grib_accessor*, grib_accessor*);
typedef void (*accessor_destroy_proc)(grib_context*, grib_accessor*);
typedef void (*accessor_dump_proc)(grib_accessor*, grib_dumper*);
typedef int (*accessor_get_native_type_proc)(grib_accessor*);
typedef void (*accessor_init_proc)(grib_accessor*, const long, grib_arguments*);
typedef void (*accessor_init_class_proc)(grib_accessor_class*);
typedef int (*accessor_nearest_proc)(grib_accessor*, double, double*);
typedef grib_accessor* (*accessor_next_proc)(grib_accessor*, int);
typedef int (*accessor_notify_change_proc)(grib_accessor*, grib_accessor*);
typedef int (*accessor_pack_bytes_proc)(grib_accessor*, const unsigned char*, size_t*);
typedef int (*accessor_pack_double_proc)(grib_accessor*, const double*, size_t*);
typedef int (*accessor_pack_expression_proc)(grib_accessor*, grib_expression*);
typedef int (*accessor_pack_is_missing_proc)(grib_accessor*);
typedef int (*accessor_pack_long_proc)(grib_accessor*, const long*, size_t*);
typedef int (*accessor_pack_missing_proc)(grib_accessor*);
typedef int (*accessor_pack_string_array_proc)(grib_accessor*, const char**, size_t*);
typedef int (*accessor_pack_string_proc)(grib_accessor*, const char*, size_t*);
typedef void (*accessor_post_init_proc)(grib_accessor*);
typedef size_t (*accessor_preferred_size_proc)(grib_accessor*, int);
typedef int (*accessor_unpack_bytes_proc)(grib_accessor*, unsigned char*, size_t*);
typedef int (*accessor_unpack_double_element_proc)(grib_accessor*, size_t, double*);
typedef int (*accessor_unpack_double_element_set_proc)(grib_accessor*, const size_t*, size_t, double*);
typedef int (*accessor_unpack_double_proc)(grib_accessor*, double*, size_t*);
typedef int (*accessor_unpack_double_subarray_proc)(grib_accessor*, double*, size_t, size_t);
typedef int (*accessor_unpack_long_proc)(grib_accessor*, long*, size_t*);
typedef int (*accessor_unpack_string_array_proc)(grib_accessor*, char**, size_t*);
typedef int (*accessor_unpack_string_proc)(grib_accessor*, char*, size_t*);
typedef void (*accessor_update_size_proc)(grib_accessor*, size_t);
typedef void (*accessor_resize_proc)(grib_accessor*, size_t);
typedef size_t (*accessor_string_proc)(grib_accessor*);
typedef grib_section* (*accessor_sub_section_proc)(grib_accessor*);
typedef long (*accessor_value_proc)(grib_accessor*);
typedef int (*accessor_value_with_ret_proc)(grib_accessor*, long*);

typedef void (*codes_assertion_failed_proc)(const char*);

typedef int (*dumper_destroy_proc)(grib_dumper*);
typedef void (*dumper_dump_proc)(grib_dumper*, grib_accessor*, const char*);
typedef void (*dumper_dump_section_proc)(grib_dumper*, grib_accessor*, grib_block_of_accessors*);
typedef void (*dumper_dump_values_proc)(grib_dumper*, grib_accessor*);
typedef void (*dumper_footer_proc)(grib_dumper*, grib_handle*);
typedef void (*dumper_header_proc)(grib_dumper*, grib_handle*);
typedef int (*dumper_init_proc)(grib_dumper*);
typedef void (*dumper_init_class_proc)(grib_dumper_class*);

typedef void (*expression_add_dependency_proc)(grib_expression*, grib_accessor*);
typedef void (*expression_class_init_proc)(grib_expression_class*);
typedef void (*expression_destroy_proc)(grib_context*, grib_expression*);
typedef int (*expression_evaluate_double_proc)(grib_expression*, grib_handle*, double*);
typedef int (*expression_evaluate_long_proc)(grib_expression*, grib_handle*, long*);
typedef const char* (*expression_evaluate_string_proc)(grib_expression*, grib_handle*, char*, size_t*, int*);
typedef const char* (*expression_get_name_proc)(grib_expression*);
typedef void (*expression_init_proc)(grib_expression*);
typedef int (*expression_native_type_proc)(grib_expression*, grib_handle*);
typedef void (*expression_print_proc)(grib_context*, grib_expression*, grib_handle*);

typedef int (*grib_data_eof_proc)(const grib_context*, void*);
typedef off_t (*grib_data_tell_proc)(const grib_context*, void*);
typedef off_t (*grib_data_seek_proc)(const grib_context*, off_t, int, void*);
typedef size_t (*grib_data_read_proc)(const grib_context*, void*, size_t, void*);
typedef size_t (*grib_data_write_proc)(const grib_context*, const void*, size_t, void*);
typedef void (*grib_dump_proc)(grib_action*, FILE*, int);
typedef int (*grib_loader_init_accessor_proc)(grib_loader*, grib_accessor*, grib_arguments*);
typedef int (*grib_loader_lookup_long_proc)(grib_context*, grib_loader*, const char*, long*);
typedef void (*grib_log_proc)(const grib_context*, int, const char*);
typedef void (*grib_print_proc)(const grib_context*, void*, const char*);
typedef void (*grib_free_proc)(const grib_context*, void*);
typedef void* (*grib_malloc_proc)(const grib_context*, size_t);
typedef void* (*grib_realloc_proc)(const grib_context*, void*, size_t);
typedef void (*grib_xref_proc)(grib_action*, FILE*, const char*);

typedef int (*iterator_destroy_proc)(grib_iterator* );
typedef long (*iterator_has_next_proc)(grib_iterator* );
typedef void (*iterator_init_class_proc)(grib_iterator_class*);
typedef int (*iterator_init_proc)(grib_iterator*, grib_handle*, grib_arguments*);
typedef int (*iterator_next_proc)(grib_iterator*, double*, double*, double*);
typedef int (*iterator_previous_proc)(grib_iterator*, double*, double*, double*);
typedef int (*iterator_reset_proc)(grib_iterator*);

static codes_assertion_failed_proc assertion = NULL;
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



/* function headers */
void codes_assertion_failed(const char*, const char*, int);
void codes_check(const char*, const char*, int, int, const char*);

grib_accessor* grib_find_accessor(const grib_handle*, const char*);
grib_accessors_list* grib_find_accessors_list(const grib_handle*, const char*);

grib_context* grib_context_get_default();

grib_handle* grib_handle_new_from_file(grib_context*, FILE*, int*);
grib_handle* grib_new_from_file(grib_context*, FILE*, int*);
static grib_handle* grib_handle_new_from_file_no_multi(grib_context*, FILE*, int, int*);

int codes_get_long(const grib_handle*, const char*, long*);
void grib_context_set_handle_file_count(grib_context*, int);
void grib_check(const char*, const char*, int, int, const char*);
int grib_get_data(const grib_handle*, double*, double*, double*);
int grib_get_long(const grib_handle*, const char*, long*);
int grib_iterator_next(grib_iterator*, double*, double*, double*);
int grib_unpack_long(grib_accessor*, long*, size_t*);

static void default_buffer_free(const grib_context*, void*);
static void* default_buffer_malloc(const grib_context*, size_t);
static void* default_buffer_realloc(const grib_context*, void*, size_t);
static void default_log(const grib_context*, int, const char*);
static void default_long_lasting_free(const grib_context*, void*);
static void* default_long_lasting_malloc(const grib_context*, size_t);
static int default_feof(const grib_context*, void*);
static void default_free(const grib_context*, void*);
static void* default_malloc(const grib_context*, size_t);
static void default_print(const grib_context*, void*, const char*);
static size_t default_read(const grib_context*, void*, size_t, void*);
static void* default_realloc(const grib_context*, void*, size_t);
static off_t default_seek(const grib_context*, off_t, int, void*);
static off_t default_tell(const grib_context*, void*);
static size_t default_write(const grib_context*, const void*, size_t, void*);





#endif