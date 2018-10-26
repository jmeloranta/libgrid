#include <cuda_runtime_api.h>
#include <cuda.h>

#define GRID_CUDA_SKIP_TRANSFER 0
#define GRID_CUDA_TRANSFER_DATA 1

// TODO: Could add flag to make sure that the block never gets synced.
struct gpu_mem_block {
  void *gpu_mem;             /* GPU memory pointer */
  void *host_mem;            /* Corresponding host memory pointer */
  size_t length;             /* Memory block length */
  time_t created;            /* time(0) value of block creation */
  time_t last_used;          /* time(0) value of last access */
  long access_count;         /* How many times has the data been accessed */
  char locked;               /* Page locked in GPU memory? (1 = yes, 0 = no) */
  char id[32];               /* String ID describing the block */
  struct gpu_mem_block *next;/* Next block in linked list (NULL terminated list) */
};

typedef struct gpu_mem_block gpu_mem_block; 
