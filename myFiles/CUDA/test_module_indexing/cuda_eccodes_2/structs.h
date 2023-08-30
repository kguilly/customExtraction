
#ifndef STRUCTS_H
#define STRUCTS_H

#include <stdio.h>
#include <unistd.h>


#define MAX_SET_VALUES 10
#define ACCESSORS_ARRAY_SIZE 5000
#define MAX_NUM_SECTIONS 12
#define STRING_VALUE_LEN 100
#define MAX_NUM_CONCEPTS 2000
#define MAX_NUM_HASH_ARRAY 2000
#define MAX_ACCESSOR_NAMES 20
#define MAX_ACCESSOR_ATTRIBUTES 20

#if GRIB_PTHREADS
#include <pthread.h>
#define GRIB_MUTEX_INIT_ONCE(a, b) pthread_once(a, b);
#define GRIB_MUTEX_LOCK(a) pthread_mutex_lock(a);
#define GRIB_MUTEX_UNLOCK(a) pthread_mutex_unlock(a);
/*
 #define GRIB_MUTEX_LOCK(a) {pthread_mutex_lock(a); printf("MUTEX LOCK %p %s line %d\n",(void*)a,__FILE__,__LINE__);}
 #define GRIB_MUTEX_UNLOCK(a) {pthread_mutex_unlock(a);printf("MUTEX UNLOCK %p %s line %d\n",(void*)a,__FILE__,__LINE__);}
 */
#elif GRIB_OMP_THREADS
#include <omp.h>
#ifdef _MSC_VER
#define GRIB_OMP_CRITICAL(a) __pragma(omp critical(a))
#else
#define GRIB_OMP_STR(a) #a
#define GRIB_OMP_XSTR(a) GRIB_OMP_STR(a)
#define GRIB_OMP_CRITICAL(a) _Pragma(GRIB_OMP_XSTR(omp critical(a)))
#endif
#define GRIB_MUTEX_INIT_ONCE(a, b) (*(b))();
#define GRIB_MUTEX_LOCK(a) omp_set_nest_lock(a);
#define GRIB_MUTEX_UNLOCK(a) omp_unset_nest_lock(a);
#else
#define GRIB_MUTEX_INIT_ONCE(a, b)
#define GRIB_MUTEX_LOCK(a)
#define GRIB_MUTEX_UNLOCK(a)
#endif


typedef struct grib_handle grib_handle;
typedef struct grib_field_tree grib_field_tree;
typedef struct grib_index_key grib_index_key;
typedef struct grib_field_list grib_field_list;
typedef struct grib_index grib_index;
typedef struct grib_field grib_field;
typedef struct grib_file grib_file;
typedef struct grib_context grib_context;
typedef struct grib_string_list grib_string_list;
typedef struct grib_buffer grib_buffer;
typedef struct grib_section grib_section;
typedef struct grib_accessor grib_accessor;
typedef struct grib_action grib_action;
typedef struct grib_action_class grib_action_class;
typedef struct grib_arguments grib_arguments;
typedef struct grib_trie grib_trie;
typedef struct grib_dependency grib_dependency;
typedef struct grib_loader grib_loader;
typedef struct grib_values grib_values;

typedef enum ProductKind
{
    PRODUCT_ANY,
    PRODUCT_GRIB,
    PRODUCT_BUFR,
    PRODUCT_METAR,
    PRODUCT_GTS,
    PRODUCT_TAF
} ProductKind;

#define SIZE 39
struct grib_trie
{
    grib_trie* next[SIZE];
    grib_context* context;
    int first;
    int last;
    void* data;
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

struct grib_arguments
{
    struct grib_arguments* next;
    grib_expression* expression;
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

struct grib_string_list
{
    char* value;
    int count;
    grib_string_list* next;
};

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

