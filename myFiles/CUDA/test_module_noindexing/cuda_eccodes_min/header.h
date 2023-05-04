#ifndef HEADER_H
#define HEADER_H


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <string.h>


#define ACCESSORS_ARRAY_SIZE 5000
#define Assert(a) \
    do {                                                          \
        if (!(a)) codes_assertion_failed(#a, __FILE__, __LINE__); \
    } while (0)
#define BIT_MASK(x) \
(((x) == max_nbits) ? (unsigned long)-1UL : (1UL << (x)) - 1)
#define DebugAssert(a) Assert(a)
#define DebugAssert(a)
#define DEFAULT_FILE_POOL_MAX_OPENED_FILES 0
#define ECC_PATH_DELIMITER_CHAR ';'
#define ECC_PATH_MAXLEN 8192
#define GRIB_7777_NOT_FOUND -5
#define GRIB_BUFFER_TOO_SMALL -3
#define GROW_BUF_IF_REQUIRED(desired_length)      \
    if (buf->length < (desired_length)) {         \
        grib_grow_buffer(c, buf, desired_length); \
        tmp = buf->data;                          \
    }
#define GRIB_DECODING_ERROR -13
#define GRIB_END_OF_FILE -1
#define GRIB_INLINE
#define GRIB_INTERNAL_ERROR -2
#define GRIB_INVALID_ARGUMENT -19
#define GRIB_IO_PROBLEM -11
#define GRIB_LOG_DEBUG 4
#define GRIB_LOG_ERROR 2
#define GRIB_LOG_FATAL 3
#define GRIB_LOG_INFO 0
#define GRIB_LOG_PERROR (1 << 10)
#define GRIB_LOG_WARNING 1
#define GRIB_MESSAGE_TOO_LARGE -47
#define GRIB_MUTEX_INIT_ONCE(a, b)
#define GRIB_MUTEX_LOCK(a)
#define GRIB_MUTEX_UNLOCK(a)
#define GRIB_MY_BUFFER 0
#define GRIB_NOT_FOUND -10
#define GRIB_NOT_IMPLEMENTED -4
#define GRIB_OUT_OF_MEMORY -17
#define GRIB_PREMATURE_END_OF_FILE -45
#define GRIB_REAL_MODE8 8
#define GRIB_SUCCESS 0
#define GRIB_UNSUPPORTED_EDITION -64
#define GRIB_USER_BUFFER 1
#define GRIB_WRONG_LENGTH -23
#define ITRIE_SIZE 40
#define MAX_ACCESSOR_NAMES 20
#define MAX_ACCESSOR_ATTRIBUTES 20
#define MAX_NUM_CONCEPTS 2000
#define MAX_NUM_HASH_ARRAY 2000
#define MAX_NUM_SECTIONS 12
#define MAX_SET_VALUES 10
#define MAX_SMART_TABLE_COLUMNS 20
#define NUMBER(x) (sizeof(x) / sizeof(x[0]))
#define STRING_VALUE_LEN 100
#define TRIE_SIZE 39
#define UINT3(a, b, c) (size_t)((a << 16) + (b << 8) + c);

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

typedef struct alloc_buffer alloc_buffer;
typedef struct codes_condition codes_condition;
typedef struct code_table_entry code_table_entry;
typedef struct grib_accessors_list grib_accessors_list;
typedef struct grib_accessor grib_accessor;
typedef struct grib_accessor_class grib_accessor_class;
typedef struct grib_accessor_iterator grib_accessor_iterator;
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
typedef struct grib_darray grib_darray;
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
typedef struct reader reader;
typedef struct table_entry table_entry;

/* funnction pointer typedefs */
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

typedef void* (*allocproc)(void*, size_t*, int*);
typedef size_t (*readproc)(void*, void*, size_t, int*);
typedef int (*seekproc)(void*, off_t);
typedef off_t (*tellproc)(void*);



struct alloc_buffer
{
    size_t size;
    void* buffer;
};

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

struct grib_accessor_iterator 
{
    grib_accessor att;
    grib_arguments* args;
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

struct reader
{
    void* read_data;
    readproc read;

    void* alloc_data;
    allocproc alloc;
    int headers_only;

    seekproc seek;
    seekproc seek_from_start;
    tellproc tell;
    off_t offset;

    size_t message_size;

};

struct table_entry
{
    const char* type;
    grib_iterator_class** cclass;
};


static codes_assertion_failed_proc assertion = NULL;
static const char *errors[] = {
"No error",		/* 0 GRIB_SUCCESS */
"End of resource reached",		/* -1 GRIB_END_OF_FILE */
"Internal error",		/* -2 GRIB_INTERNAL_ERROR */
"Passed buffer is too small",		/* -3 GRIB_BUFFER_TOO_SMALL */
"Function not yet implemented",		/* -4 GRIB_NOT_IMPLEMENTED */
"Missing 7777 at end of message",		/* -5 GRIB_7777_NOT_FOUND */
"Passed array is too small",		/* -6 GRIB_ARRAY_TOO_SMALL */
"File not found",		/* -7 GRIB_FILE_NOT_FOUND */
"Code not found in code table",		/* -8 GRIB_CODE_NOT_FOUND_IN_TABLE */
"Array size mismatch",		/* -9 GRIB_WRONG_ARRAY_SIZE */
"Key/value not found",		/* -10 GRIB_NOT_FOUND */
"Input output problem",		/* -11 GRIB_IO_PROBLEM */
"Message invalid",		/* -12 GRIB_INVALID_MESSAGE */
"Decoding invalid",		/* -13 GRIB_DECODING_ERROR */
"Encoding invalid",		/* -14 GRIB_ENCODING_ERROR */
"Code cannot unpack because of string too small",		/* -15 GRIB_NO_MORE_IN_SET */
"Problem with calculation of geographic attributes",		/* -16 GRIB_GEOCALCULUS_PROBLEM */
"Memory allocation error",		/* -17 GRIB_OUT_OF_MEMORY */
"Value is read only",		/* -18 GRIB_READ_ONLY */
"Invalid argument",		/* -19 GRIB_INVALID_ARGUMENT */
"Null handle",		/* -20 GRIB_NULL_HANDLE */
"Invalid section number",		/* -21 GRIB_INVALID_SECTION_NUMBER */
"Value cannot be missing",		/* -22 GRIB_VALUE_CANNOT_BE_MISSING */
"Wrong message length",		/* -23 GRIB_WRONG_LENGTH */
"Invalid key type",		/* -24 GRIB_INVALID_TYPE */
"Unable to set step",		/* -25 GRIB_WRONG_STEP */
"Wrong units for step (step must be integer)",		/* -26 GRIB_WRONG_STEP_UNIT */
"Invalid file id",		/* -27 GRIB_INVALID_FILE */
"Invalid grib id",		/* -28 GRIB_INVALID_GRIB */
"Invalid index id",		/* -29 GRIB_INVALID_INDEX */
"Invalid iterator id",		/* -30 GRIB_INVALID_ITERATOR */
"Invalid keys iterator id",		/* -31 GRIB_INVALID_KEYS_ITERATOR */
"Invalid nearest id",		/* -32 GRIB_INVALID_NEAREST */
"Invalid order by",		/* -33 GRIB_INVALID_ORDERBY */
"Missing a key from the fieldset",		/* -34 GRIB_MISSING_KEY */
"The point is out of the grid area",		/* -35 GRIB_OUT_OF_AREA */
"Concept no match",		/* -36 GRIB_CONCEPT_NO_MATCH */
"Hash array no match",		/* -37 GRIB_HASH_ARRAY_NO_MATCH */
"Definitions files not found",		/* -38 GRIB_NO_DEFINITIONS */
"Wrong type while packing",		/* -39 GRIB_WRONG_TYPE */
"End of resource",		/* -40 GRIB_END */
"Unable to code a field without values",		/* -41 GRIB_NO_VALUES */
"Grid description is wrong or inconsistent",		/* -42 GRIB_WRONG_GRID */
"End of index reached",		/* -43 GRIB_END_OF_INDEX */
"Null index",		/* -44 GRIB_NULL_INDEX */
"End of resource reached when reading message",		/* -45 GRIB_PREMATURE_END_OF_FILE */
"An internal array is too small",		/* -46 GRIB_INTERNAL_ARRAY_TOO_SMALL */
"Message is too large for the current architecture",		/* -47 GRIB_MESSAGE_TOO_LARGE */
"Constant field",		/* -48 GRIB_CONSTANT_FIELD */
"Switch unable to find a matching case",		/* -49 GRIB_SWITCH_NO_MATCH */
"Underflow",		/* -50 GRIB_UNDERFLOW */
"Message malformed",		/* -51 GRIB_MESSAGE_MALFORMED */
"Index is corrupted",		/* -52 GRIB_CORRUPTED_INDEX */
"Invalid number of bits per value",		/* -53 GRIB_INVALID_BPV */
"Edition of two messages is different",		/* -54 GRIB_DIFFERENT_EDITION */
"Value is different",		/* -55 GRIB_VALUE_DIFFERENT */
"Invalid key value",		/* -56 GRIB_INVALID_KEY_VALUE */
"String is smaller than requested",		/* -57 GRIB_STRING_TOO_SMALL */
"Wrong type conversion",		/* -58 GRIB_WRONG_CONVERSION */
"Missing BUFR table entry for descriptor",		/* -59 GRIB_MISSING_BUFR_ENTRY */
"Null pointer",		/* -60 GRIB_NULL_POINTER */
"Attribute is already present, cannot add",		/* -61 GRIB_ATTRIBUTE_CLASH */
"Too many attributes. Increase MAX_ACCESSOR_ATTRIBUTES",		/* -62 GRIB_TOO_MANY_ATTRIBUTES */
"Attribute not found.",		/* -63 GRIB_ATTRIBUTE_NOT_FOUND */
"Edition not supported.",		/* -64 GRIB_UNSUPPORTED_EDITION */
"Value out of coding range",		/* -65 GRIB_OUT_OF_RANGE */
"Size of bitmap is incorrect",		/* -66 GRIB_WRONG_BITMAP_SIZE */
"Functionality not enabled",		/* -67 GRIB_FUNCTIONALITY_NOT_ENABLED */
"Value mismatch",		/* 1 GRIB_VALUE_MISMATCH */
"double values are different",		/* 2 GRIB_DOUBLE_VALUE_MISMATCH */
"long values are different",		/* 3 GRIB_LONG_VALUE_MISMATCH */
"byte values are different",		/* 4 GRIB_BYTE_VALUE_MISMATCH */
"string values are different",		/* 5 GRIB_STRING_VALUE_MISMATCH */
"Offset mismatch",		/* 6 GRIB_OFFSET_MISMATCH */
"Count mismatch",		/* 7 GRIB_COUNT_MISMATCH */
"Name mismatch",		/* 8 GRIB_NAME_MISMATCH */
"Type mismatch",		/* 9 GRIB_TYPE_MISMATCH */
"Type and value mismatch",		/* 10 GRIB_TYPE_AND_VALUE_MISMATCH */
"Unable to compare accessors",		/* 11 GRIB_UNABLE_TO_COMPARE_ACCESSORS */
"Unable to reset iterator",		/* 12 GRIB_UNABLE_TO_RESET_ITERATOR */
"Assertion failure",		/* 13 GRIB_ASSERTION_FAILURE */
};
static const unsigned long dmasks[] = {
    0xFF,
    0xFE,
    0xFC,
    0xF8,
    0xF0,
    0xE0,
    0xC0,
    0x80,
    0x00,
};
static const int max_nbits        = sizeof(unsigned long) * 8;
static const int max_nbits_size_t = sizeof(size_t) * 8;
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

static const struct table_entry table[] = {
#include "subfuncs/grib_iterator_factory.h"
};


/* function headers */
void codes_assertion_failed(const char*, const char*, int);
void codes_check(const char*, const char*, int, int, const char*);

grib_accessor* grib_find_accessor(const grib_handle*, const char*);
grib_accessors_list* grib_find_accessors_list(const grib_handle*, const char*);

grib_buffer* grib_new_buffer(const grib_context* c, const unsigned char* data, size_t buflen);
grib_context* grib_context_get_default();

static grib_handle* grib_handle_create(grib_handle* gl, grib_context* c, const void* data, size_t buflen);
grib_handle* grib_handle_new_from_file(grib_context*, FILE*, int*);
grib_handle* grib_new_from_file(grib_context*, FILE*, int, int*);
grib_handle* grib_handle_new_from_message(grib_context* c, const void* data, size_t buflen);
grib_handle* grib_handle_new_from_partial_message(grib_context* c, const void* data, size_t buflen);
static grib_handle* grib_handle_new_from_file_no_multi(grib_context*, FILE*, int, int*);
static grib_handle* grib_handle_new_from_file_multi(grib_context* c, FILE* f, int* error);
grib_handle* grib_new_handle(grib_context* c);

grib_iterator* grib_iterator_factory(grib_handle* h, grib_arguments* args, unsigned long flags, int* ret);
grib_iterator* grib_iterator_new(const grib_handle*, unsigned long flags, int*);

static void init_class(grib_iterator_class*);
static int init(grib_iterator* i, grib_handle*, grib_arguments*);
static int destroy(grib_iterator* i);
static int reset(grib_iterator* i);
static long has_next(grib_iterator* i);

int codes_get_long(const grib_handle*, const char*, long*);
static int determine_product_kind(grib_handle* h, ProductKind* prod_kind);
static void grib2_build_message(grib_context* context, unsigned char* sections[], size_t sections_len[], void** data, size_t* len);
static int grib2_get_next_section(unsigned char* msgbegin, size_t msglen, unsigned char** secbegin, size_t* seclen, int* secnum, int* err);
static int grib2_has_next_section(unsigned char* msgbegin, size_t msglen, unsigned char* secbegin, size_t seclen, int* err);
const char* grib_arguments_get_name(grib_handle* h, grib_arguments* args, int n);
void grib_buffer_delete(const grib_context* c, grib_buffer* b);
void grib_context_set_handle_total_count(grib_context* c, int new_count);
void grib_get_buffer_ownership(const grib_context* c, grib_buffer* b);
int grib_get_string(const grib_handle* h, const char* name, char* val, size_t* length);
static void grib_grow_buffer_to(const grib_context* c, grib_buffer* b, size_t ns);
void grib_grow_buffer(const grib_context* c, grib_buffer* b, size_t new_size);
void grib_accessor_delete(grib_context* ct, grib_accessor* a);
int grib_handle_delete(grib_handle* h);
void* grib_context_malloc(const grib_context* c, size_t size);
void* grib_context_malloc_clear(const grib_context* c, size_t size);
void grib_empty_section(grib_context* c, grib_section* b);
const char* grib_expression_get_name(grib_expression*);
void grib_context_set_handle_file_count(grib_context*, int);
off_t grib_context_tell(const grib_context* c, void* stream);
void grib_check(const char*, const char*, int, int, const char*);
void grib_context_free(const grib_context* c, void* p);
void grib_context_increment_handle_file_count(grib_context* c);
void grib_context_log(const grib_context*, int, const char*, ...);
int grib_get_data(const grib_handle*, double*, double*, double*);
const char* grib_get_error_message(int code);
int grib_get_length(const grib_handle* h, const char* name, size_t* length);
int grib_get_long(const grib_handle*, const char*, long*);
int _grib_get_string_length(grib_accessor* a, size_t* size);
int grib_get_string_length(const grib_handle* h, const char* name, size_t* size);
void grib_context_increment_handle_total_count(grib_context* c);
size_t grib_context_read(const grib_context* c, void* ptr, size_t size, void* stream);
static GRIB_INLINE int grib_inline_strcmp(const char* a, const char* b);
int grib_context_seek(const grib_context* c, off_t offset, int whence, void* stream);
int grib_encode_unsigned_long(unsigned char* p, unsigned long val, long* bitp, long nbits);
int grib_is_defined(const grib_handle* h, const char* name);
int grib_iterator_delete(grib_iterator* i);
int grib_iterator_init(grib_iterator* i, grib_handle* h, grib_arguments* args);
int grib_iterator_next(grib_iterator*, double*, double*, double*);
long grib_byte_offset(grib_accessor* a);
int grib_unpack_long(grib_accessor*, long*, size_t*);
int grib_unpack_string(grib_accessor* a, char* v, size_t* len);
long grib_string_length(grib_accessor* a);
void grib_section_delete(grib_context* c, grib_section* b);

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

static void* allocate_buffer(void* data, size_t* length, int* err);
static int read_any(reader* r, int grib_ok, int bufr_ok, int hdf5_ok, int wrap_ok);
static int init_iterator(grib_iterator_class* c, grib_iterator* i, grib_handle* h, grib_arguments* args);
size_t stdio_read(void* data, void* buf, size_t len, int* err);
int stdio_seek_from_start(void* data, off_t len);
int stdio_seek(void* data, off_t len);
off_t stdio_tell(void* data);
void* wmo_read_grib_from_file_malloc(FILE* f, int headers_only, size_t* size, off_t* offset, int* err);
static void* _wmo_read_any_from_file_malloc(FILE* f, int* err, size_t* size, off_t* offset,
                                            int grib_ok, int bufr_ok, int hdf5_ok, int wrap_ok, int headers_only);

static int read_GRIB(reader* r);
static int read_the_rest(reader* r, size_t message_length, unsigned char* tmp, int already_read, int check7777);



#endif