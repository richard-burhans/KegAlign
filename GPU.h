#include "DRAM.h"
#include <vector>
#include <string.h>
#include <stdio.h>
#include <mutex>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <stdlib.h>

#define MAX_BLOCKS 1<<10
#define MAX_THREADS 1024 
#define TILE_SIZE 32
#define MAX_HITS 100000000 
#define MAX_SEEDS 13*524288  //make it 13*chunk_size
#define BLOCK_SIZE 128 
#define NUM_WARPS 4

struct hsp {
    uint32_t ref_start;
    uint32_t query_start;
    uint32_t len;
    uint32_t score;
};

typedef size_t(*InitializeProcessor_ptr)();
typedef void(*SendSeedPosTable_ptr)(uint32_t* index_table, uint32_t index_table_size, uint32_t* pos_table, uint32_t ref_size);
typedef std::vector<hsp> (*SeedAndFilter_ptr)(std::vector<uint64_t> seed_offset_vector, bool rev, uint32_t buffer);
typedef void(*ShutdownProcessor_ptr)();
typedef void(*SendRefWriteRequest_ptr)(size_t addr, size_t len);
typedef void(*SendQueryWriteRequest_ptr)(size_t addr, size_t len, uint32_t buffer);

extern DRAM *g_DRAM;
    
extern InitializeProcessor_ptr g_InitializeProcessor;
extern SendSeedPosTable_ptr g_SendSeedPosTable;
extern SeedAndFilter_ptr g_SeedAndFilter;
extern SendRefWriteRequest_ptr g_SendRefWriteRequest;
extern SendQueryWriteRequest_ptr g_SendQueryWriteRequest;
extern ShutdownProcessor_ptr g_ShutdownProcessor;      
