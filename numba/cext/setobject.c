#include "setobject.h"
/* The following is adapted from CPython3.11.
The exact commit is:

- https://github.com/python/cpython/blob/4b8d2a1b40b88e4c658b3f5f450c146c78f2e6bd/Objects/setobject.c

   set object implementation

   Written and maintained by Raymond D. Hettinger <python@rcn.com>
   Derived from Lib/sets.py and Objects/setobject.c.

   The basic lookup function used by all operations.
   This is based on Algorithm D from Knuth Vol. 3, Sec. 6.4.

   The initial probe index is computed as hash mod the table size.
   Subsequent probe indices are computed as explained in Objects/setobject.c.

   To improve cache locality, each probe inspects a series of consecutive
   nearby entries before moving on to probes elsewhere in memory.  This leaves
   us with a hybrid of linear probing and randomized probing.  The linear probing
   reduces the cost of hash collisions because consecutive memory accesses
   tend to be much cheaper than scattered probes.  After LINEAR_PROBES steps,
   we then use more of the upper bits from the hash value and apply a simple
   linear congruential random number genearator.  This helps break-up long
   chains of collisions.

   All arithmetic on hash should ignore overflow.

   Unlike the dictionary implementation, the lookkey function can return
   NULL if the rich comparison returns an error.

   Use cases for sets differ considerably from dictionaries where looked-up
   keys are more likely to be present.  In contrast, sets are primarily
   about membership testing where the presence of an element is not known in
   advance.  Accordingly, the set implementation needs to optimize for both
   the found and not-found case.
*/

typedef enum {
    ENTRY_PRESENT = 1,
    OK = 0,
    ERR_KEY_NOT_FOUND = -1,
    ERR_SET_MUTATED = -2,
    ERR_ITER_EXHAUSTED = -3,
    ERR_SET_EMPTY = -4,
    ERR_CMP_FAILED = -5,
} SetStatus;

static Py_ssize_t EMPTY_HASH = -9;

/* Object used as dummy key to filled deleted entries */
static char *_dummy_struct;

#define dummy (&_dummy_struct)

/* Returns -1 for error; 0 for not equal; 1 for equal */
static int
key_equal(NB_Set *setp, const char *lhs, const char *rhs) {
    if ( setp->methods.key_equal ) {
        return setp->methods.key_equal(lhs, rhs);
    } else {
        return memcmp(lhs, rhs, setp->key_size) == 0;
    }
}

static void
set_incref_key(NB_Set *setp, const char *key) {
    if ( setp->methods.key_incref ) {
        setp->methods.key_incref(key);
    }
}

static void
set_decref_key(NB_Set *setp, const char *key) {
    if ( setp->methods.key_decref ) {
        setp->methods.key_decref(key);
    }
}


void
numba_set_free(NB_Set *setp) {
    if(setp->smalltable != setp->table)
        free(setp->table);
    free(setp);
}

char * alloc_table(Py_ssize_t table_size, Py_ssize_t key_size, Py_ssize_t hash_size){
    char *table;

    Py_ssize_t entry_size = key_size + hash_size;
    table = calloc(table_size, entry_size);

    char *entry = table;
    for (int i=0;i<table_size;i++) {
        memcpy(entry, (char*)&EMPTY_HASH, hash_size);
        entry += entry_size;
    }

    return table;
}

/* Allocate new set */
int
numba_set_new(NB_Set **out, Py_ssize_t key_size, Py_ssize_t size) {
    NB_Set *setp;

    setp = malloc(sizeof(NB_Set));
    /* Ensure that the method table is all nulls */
    memset(&setp->methods, 0x00, sizeof(type_based_methods_table));

    setp->key_size = key_size;
    setp->hash_size = sizeof(Py_ssize_t);
    setp->entry_size = setp->key_size + setp->hash_size;
    setp->filled = 0;
    setp->used = 0;
    setp->size = SET_MINSIZE;
    setp->mask = SET_MINSIZE - 1;
    setp->table = setp->smalltable = alloc_table(SET_MINSIZE, setp->key_size, setp->hash_size);

    *out = setp;
    return OK;
}

Py_ssize_t
numba_set_length(NB_Set *setp) {
    return setp->used;
}

void numba_set_dump(NB_Set *setp){
    printf("Printing Set:\n");

    printf("Hashtable size:%ld\nActive Entries: %ld\nDummy Entries: %ld\nEmpty Slots: %ld\n\n",
            setp->size, setp->used, setp->filled - setp->used, setp->size-setp->filled);

    int i, j;
    char *entry;
    char *entry_key;
    Py_ssize_t entry_hash;

    printf("Hashtable entries as follows:\n");
    for(i=0;i<setp->size;i++){
        entry = setp->table + (i * setp->entry_size);
        entry_hash = get_entry_hash(entry);
        entry_key = entry + setp->hash_size;

        if(entry_key == NULL){
            printf("%d - No Entry Found\n", i);
        }else if(entry_key == (char *)dummy){
            printf("%d - Dummy Entry Found\n", i);
        }else{
            printf("%d - Key Found: ", i);
            for(j=0;j<setp->key_size;j++){
                printf("%c", *(entry_key+j));
            }
            printf(" with hash %ld\n", entry_hash);
        }
    }
    printf("Hashtable ends\n\n");
}

/* ======================================================================== */
/* ======= Begin logic for probing the hash table ========================= */

/* Set this to zero to turn-off linear probing */
#ifndef LINEAR_PROBES
#define LINEAR_PROBES 9
#endif

/* This must be >= 1 */
#define PERTURB_SHIFT 5

static char *
numba_set_lookkey(NB_Set *setp, char *key, Py_ssize_t hash)
{
    char *table;
    char *entry;
    size_t perturb;
    size_t mask = setp->mask;
    size_t i = (size_t)hash & mask; /* Unsigned for defined overflow behavior */
    size_t j;

    Py_ssize_t entry_hash;
    char *entry_key;

    entry = setp->table + (i * setp->entry_size);
    entry_hash = get_entry_hash(entry);
    entry_key = entry + setp->hash_size;

    if (entry_hash == EMPTY_HASH)
        return entry;

    perturb = hash;

    while (1) {
        if (entry_hash == hash) {
            char *startkey = NULL;
            startkey = entry_key;
            table = setp->table;
            if (key_equal(setp, startkey, key))
                return entry;
            if (table != setp->table || entry_key != startkey)
                return numba_set_lookkey(setp, key, hash);
            mask = setp->mask;
        }

        if (i + LINEAR_PROBES <= mask) {
            for (j = 0 ; j < LINEAR_PROBES ; j++) {
                entry += setp->entry_size;
                entry_hash = get_entry_hash(entry);
                entry_key = entry + setp->hash_size;
    
                if (entry_hash == EMPTY_HASH)
                    return entry;
                if (entry_hash == hash) {
                    char *startkey;
                    startkey = entry_key;
                    assert(startkey != dummy);
                    table = setp->table;
                    if (key_equal(setp, startkey, key))
                        return entry;
                    if (table != setp->table || entry_key != startkey)
                        return numba_set_lookkey(setp, key, hash);
                    mask = setp->mask;
                }
            }
        }

        perturb >>= PERTURB_SHIFT;
        i = (i * 5 + 1 + perturb) & mask;

        entry = setp->table + (i * setp->entry_size);
        entry_hash = get_entry_hash(entry);
        entry_key = entry + setp->hash_size;
        if (entry_hash == EMPTY_HASH)
            return entry;
    }
}

static void
numba_set_add_clean(char *table, size_t mask, char *key, Py_hash_t hash, Py_ssize_t key_size)
{
    char *entry;
    size_t perturb = hash;
    size_t i = (size_t)hash & mask;
    size_t j;
    Py_ssize_t hash_size = sizeof(Py_ssize_t);
    Py_ssize_t entry_size = hash_size + key_size;

    Py_ssize_t entry_hash;
    char *entry_key;

    while (1) {
        entry = table + (i * entry_size);
        entry_hash = get_entry_hash(entry);
        entry_key = entry + hash_size;

        if (entry_hash == EMPTY_HASH)
            goto found_null;
        if (i + LINEAR_PROBES <= mask) {
            for (j = 0; j < LINEAR_PROBES; j++) {
                entry += entry_size;
                entry_hash = get_entry_hash(entry);
                entry_key = entry + hash_size;

                if (entry_hash == EMPTY_HASH)
                    goto found_null;
            }
        }
        perturb >>= PERTURB_SHIFT;
        i = (i * 5 + 1 + perturb) & mask;
    }

  found_null:
    memcpy(entry, (char*)&hash, hash_size);
    memcpy(entry + hash_size, key, key_size);
}

static int
numba_set_table_resize(NB_Set *setp, Py_ssize_t minused)
{
    char *oldtable, *newtable, *entry, *temp_entry, *entry_key;
    Py_ssize_t oldmask = setp->mask;
    Py_ssize_t entry_hash;
    size_t newmask;
    int is_oldtable_malloced;

    char *small_copy = alloc_table(SET_MINSIZE, setp->key_size, setp->hash_size);

    assert(minused >= 0);

    /* Find the smallest table size > minused. */
    /* XXX speed-up with intrinsics */
    size_t newsize = SET_MINSIZE;
    while (newsize <= (size_t)minused) {
        newsize <<= 1; // The largest possible value is PY_SSIZE_T_MAX + 1.
    }
    setp->size = newsize;

    /* Get space for a new table. */
    oldtable = setp->table;
    assert(oldtable != NULL);
    is_oldtable_malloced = oldtable != setp->smalltable;

    if (newsize == SET_MINSIZE) {
        /* A large table is shrinking, or we can't get any smaller. */
        newtable = setp->smalltable;
        if (newtable == oldtable) {
            if (setp->filled == setp->used) {
                /* No dummies, setp no point doing anything. */
                return 0;
            }
            /* We're not going to resize it, but rebuild the
               table anyway to purge old dummy entries.
               Subtle:  This is *necessary* if filled==size,
               as set_lookkey needs at least one virgin slot to
               terminate failing searches.  If filled < size, it's
               merely desirable, as dummies slow searches. */
            assert(setp->filled > setp->used);
            memcpy(small_copy, oldtable, setp->entry_size * SET_MINSIZE);
            oldtable = small_copy;
        }
    }
    else {
        newtable = alloc_table(newsize, setp->key_size, setp->hash_size);
        if (newtable == NULL) {
            return -1;
        }
    }

    /* Make the set empty, using the new table. */
    assert(newtable != oldtable);

    setp->mask = newsize - 1;
    setp->table = newtable;

    /* Copy the data over; this is refcount-neutral for active entries;
       dummy entries aren't copied over, of course */

    newmask = (size_t)setp->mask;
    if (setp->filled == setp->used) {
        for (entry = oldtable; entry <= oldtable + (oldmask * setp->entry_size); entry+=setp->entry_size) {
            entry_hash = get_entry_hash(entry);
            entry_key = entry + setp->hash_size;
            if (entry_hash != EMPTY_HASH) {
                numba_set_add_clean(newtable, newmask, entry_key, entry_hash, setp->key_size);
            }
        }
    } else {
        setp->filled = setp->used;
        for (entry = oldtable; entry <= oldtable + (oldmask * setp->entry_size); entry+=setp->entry_size) {
            entry_hash = get_entry_hash(entry);
            entry_key = entry + setp->hash_size;
            if (entry_hash != EMPTY_HASH && entry_key != (char *)dummy) {
                numba_set_add_clean(newtable, newmask, entry_key, entry_hash, setp->key_size);
            }
        }
    }

    if (is_oldtable_malloced)
        free(oldtable);
    return 0;
}

void
numba_set_set_method_table(NB_Set *setp, set_type_based_methods_table *methods)
{
    memcpy(&setp->methods, methods, sizeof(set_type_based_methods_table));
}

static int
numba_set_found_unused(NB_Set *setp, char *key, char *entry, Py_ssize_t hash, size_t mask){
    setp->filled++;
    setp->used++;

    memcpy(entry, (char*)&hash, setp->hash_size);
    memcpy(entry + setp->hash_size, key, setp->key_size);

    if ((size_t)setp->filled*5 < mask*3)
        return OK;
    return numba_set_table_resize(setp, setp->used>50000 ? setp->used*2 : setp->used*4);
}

static int
numba_set_found_unused_or_dummy(NB_Set *setp, char *key, char *entry, char *freeslot, Py_ssize_t hash, size_t mask){
    if (get_entry_hash(freeslot) == 0)
        return numba_set_found_unused(setp, key, entry, hash, mask);
    setp->used++;

    memcpy(freeslot, (char*)&hash, setp->hash_size);
    memcpy(freeslot + setp->hash_size, key, setp->key_size);

    set_incref_key(setp, key);

    return OK;
}

Py_ssize_t get_entry_hash(char *entry) {
    return ((Py_ssize_t*)entry)[0];
}

int
numba_set_add(NB_Set *setp, char *key, Py_ssize_t hash)
{
    char *table;
    char *freeslot;
    char *entry;
    size_t perturb;
    size_t mask;
    size_t i;                       /* Unsigned for defined overflow behavior */
    size_t j;

    Py_ssize_t entry_hash;
    char *entry_key;

  restart:
    mask = setp->mask;
    i = (size_t)hash & mask;

    entry = setp->table + (i * setp->entry_size);
    entry_hash = get_entry_hash(entry);
    entry_key = entry + setp->hash_size;

    if (entry_hash == EMPTY_HASH)
        return numba_set_found_unused(setp, key, entry, hash, mask);

    freeslot = NULL;
    perturb = hash;

    while (1) {
        if (entry_hash == hash) {
            char *startkey = entry_key;
            /* startkey cannot be a dummy because the dummy hash field is -1 */
            assert(startkey != dummy);
            table = setp->table;
            if (key_equal(setp, startkey, key))                                        /* likely */
                return OK;
            /* Continuing the search from the current entry only makes
               sense if the table and entry are unchanged; otherwise,
               we have to restart from the beginning */
            if (table != setp->table || entry_key != startkey)
                goto restart;
            mask = setp->mask;                 /* help avoid a register spill */
        }
        else if (entry_hash == -1)
            freeslot = entry;

        if (i + LINEAR_PROBES <= mask) {
            for (j = 0 ; j < LINEAR_PROBES ; j++) {
                entry += setp->entry_size;
                entry_hash = get_entry_hash(entry);
                entry_key = entry + setp->hash_size;
    
                if (entry_hash == 0 && entry_key == NULL)
                    return numba_set_found_unused_or_dummy(setp, key, entry, freeslot, hash, mask);
                if (entry_hash == hash) {
                    char *startkey = entry_key;
                    assert(startkey != dummy);
                    table = setp->table;
                    if (key_equal(setp, startkey, key))
                        return OK;
                    if (table != setp->table || entry_key != startkey)
                        goto restart;
                    mask = setp->mask;
                }
                else if (entry_hash == -1)
                    freeslot = entry;
            }
        }

        perturb >>= PERTURB_SHIFT;
        i = (i * 5 + 1 + perturb) & mask;

        entry = setp->table + (i * setp->entry_size);
        entry_hash = get_entry_hash(entry);
        entry_key = entry + setp->hash_size;
        if (entry_hash == EMPTY_HASH)
            return numba_set_found_unused_or_dummy(setp, key, entry, freeslot, hash, mask);
    }
}


int
numba_set_contains(NB_Set *setp, char *key, Py_ssize_t hash)
{
    char *entry;
    Py_ssize_t entry_hash;
    char *entry_key;

    entry = numba_set_lookkey(setp, key, hash);
    entry_hash = get_entry_hash(entry);
    entry_key = entry + setp->hash_size;
    if (entry_hash != EMPTY_HASH)
        return entry_key != NULL;          /* Returns 1 only for a valid entry  */
    return entry_hash != EMPTY_HASH;
}

int
numba_set_discard(NB_Set *setp, char *key, Py_hash_t hash)
{
    char *entry;

    Py_ssize_t entry_hash;
    char *entry_key;

    entry = numba_set_lookkey(setp, key, hash);
    entry_hash = get_entry_hash(entry);
    entry_key = entry + setp->hash_size;

    if (entry == NULL)
        return ERR_KEY_NOT_FOUND;
    if (entry_key == NULL)
        return ERR_KEY_NOT_FOUND;

    memcpy(entry, (char*)&EMPTY_HASH, setp->hash_size);
    memcpy(entry + setp->hash_size, (char *)dummy, sizeof(dummy));

    setp->used--;
    return OK;
}


size_t
numba_set_iter_sizeof() {
    return sizeof(NB_SetIter);
}

void
numba_set_iter(NB_SetIter *it, NB_Set *setp) {
    it->parent = setp;
    it->table = setp->table;
    it->table_size = setp->size;
    it->num_keys = setp->used;
    it->pos = 0;
    it->hash_size = setp->hash_size;
    it->key_size = setp->key_size;
    it->entry_size = setp->entry_size;
}


int
numba_set_iter_next(NB_SetIter *it, const char **set_ptr) {
    /* Detect set mutation during iteration */
    char *entry_table_ptr;
    if (it->parent->table != it->table ||
        it->parent->used != it->num_keys) {
        return ERR_SET_MUTATED;
    }
    entry_table_ptr = it->table;
    while ( it->pos < it->table_size ) {
        char *entry = (entry_table_ptr + (it->pos++) * it->entry_size);
        Py_ssize_t entry_hash = get_entry_hash(entry);;
        char *entry_key = entry + it->hash_size;

        if ( entry_hash != EMPTY_HASH && entry_key != NULL && entry_key != (char *)dummy) {
            *set_ptr = entry_key;
            return OK;
        }
    }
    return ERR_ITER_EXHAUSTED;
}

// remove me PySet_Getsize is basically return setp->used setp can directly check that
// #define PySet_GET_SIZE(setp) (assert(PyAnySet_Check(setp)),(((PySetObject *)(setp))->used))

#define CHECK(CASE) {                                                   \
    if ( !(CASE) ) {                                                    \
        printf("'%s' failed file %s:%d\n", #CASE, __FILE__, __LINE__);   \
    }                                                                   \
}

void _verify_slots(NB_Set *setp, int size, int active_slots, int dummy_slots){
    // Check total size is as expected
    CHECK (setp->size == size);
    Py_ssize_t entry_hash;
    char *entry_key;

    int _active_slots = 0, _dummy_slots = 0;
    char *entry = setp->table;

    for(int i=0;i<size;i++){          
        entry_hash = get_entry_hash(entry);
        entry_key = entry + setp->hash_size;
        if(entry_key != NULL){
            if(entry_key == (char *)dummy){
                _dummy_slots++;
            }else{
                _active_slots++;
            }
        }
        entry += setp->entry_size;
    }

    CHECK (active_slots == _active_slots);
    CHECK (dummy_slots == _dummy_slots);
}

/* Basic C based tests for sets.
 */
int
numba_test_set(void) {
    NB_Set *setp = NULL;
    char *set_entry;
    NB_SetIter iter;
    Py_ssize_t it_count;
    const char *it_val;

    int status, ix, has_entry;
    puts("test_set");

    status = numba_set_new(&setp, 4, SET_MINSIZE);
    CHECK(status == OK);
    // TODO: Check if initialized correctly

    set_entry = numba_set_lookkey(setp, "befz", 0xbeef);
    CHECK (get_entry_hash(set_entry) == EMPTY_HASH);

    // insert 1st key
    status = numba_set_add(setp, "befz", 0xbeef);
    CHECK (status == OK);
    _verify_slots(setp, 8, 1, 0);

    // insert same key
    status = numba_set_add(setp, "befz", 0xbeef);
    CHECK (status == OK);
    _verify_slots(setp, 8, 1, 0);

    // insert 2nd key
    status = numba_set_add(setp, "begz", 0xbeef);
    CHECK (status == OK);
    _verify_slots(setp, 8, 2, 0);

    // insert 3rd key
    status = numba_set_add(setp, "behz", 0xcafe);
    CHECK (status == OK);
    _verify_slots(setp, 8, 3, 0);

    // insert 4th key
    status = numba_set_add(setp, "beiz", 0xcafe);
    CHECK (status == OK);
    _verify_slots(setp, 8, 4, 0);

    // insert 5th key; triggers resize
    status = numba_set_add(setp, "bejz", 0xcafe);
    CHECK (status == OK);
    _verify_slots(setp, 32, 5, 0);

    // insert 6th key
    status = numba_set_add(setp, "bekz", 0xcafe);
    CHECK (status == OK);
    _verify_slots(setp, 32, 6, 0);

    // Make sure everything are still in there
    // Check 1st key
    has_entry = numba_set_contains(setp, "befz", 0xbeef);
    CHECK (has_entry == ENTRY_PRESENT);

    // Check 2nd key
    has_entry = numba_set_contains(setp, "begz", 0xbeef);
    CHECK (has_entry == ENTRY_PRESENT);

    // Check 3rd key
    has_entry = numba_set_contains(setp, "behz", 0xcafe);
    CHECK (has_entry == ENTRY_PRESENT);

    // Check 4th key
    has_entry = numba_set_contains(setp, "beiz", 0xcafe);
    CHECK (has_entry == ENTRY_PRESENT);

    // Check 5th key
    has_entry = numba_set_contains(setp, "bejz", 0xcafe);
    CHECK (has_entry == ENTRY_PRESENT);

    // Check 6th key
    has_entry = numba_set_contains(setp, "bekz", 0xcafe);
    CHECK (has_entry == ENTRY_PRESENT);

    // Test delete
    ix = numba_set_discard(setp, "begz", 0xbeef);
    CHECK (ix == OK);
    // 6 slots in use, 5 active and 1 dummy
    _verify_slots(setp, 32, 5, 1);

    ix = numba_set_discard(setp, "begz", 0xbeef);
    CHECK (ix == ERR_KEY_NOT_FOUND); // not found
    // 6 slots in use, 5 active and 1 dummy
    _verify_slots(setp, 32, 5, 1);

    ix = numba_set_discard(setp, "befz", 0xbeef);
    CHECK (ix == OK);
    // 6 slots in use, 4 active and 2 dummy
    _verify_slots(setp, 32, 4, 2);

    ix = numba_set_discard(setp, "behz", 0xcafe);
    CHECK (ix == OK);
    // 6 slots in use, 3 active and 3 dummy
    _verify_slots(setp, 32, 3, 3);

    // Test iterator
    CHECK( setp->used > 0 );
    numba_set_iter(&iter, setp);
    it_count = 0;
    while ( (status = numba_set_iter_next(&iter, &it_val)) == OK) {
        it_count += 1;  // valid items
        CHECK(it_val != NULL);
    }

    CHECK(status == ERR_ITER_EXHAUSTED);
    CHECK(setp->used == it_count);

    numba_set_free(setp);
    return 0;
}
