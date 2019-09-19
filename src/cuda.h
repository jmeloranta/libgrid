#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cudalibxt.h>
#include <cufft.h>
#include <cufftXt.h>

#define MAX_GPU 8   /* Set to zero to disable multi-GPU code */

#define GRID_CUDA_SKIP_TRANSFER 0
#define GRID_CUDA_TRANSFER_DATA 1

struct gpu_mem_block {
  void *host_mem;              /* Host memory pointer */
  cudaLibXtDesc *gpu_info;     /* GPU memory information pointer (NULL if not in GPU memory) */
  cufftHandle cufft_handle;    /* cufft handle (-1 = when not in use) */
  time_t created;              /* time(0) value of block creation */
  time_t last_used;            /* time(0) value of last access */
  long access_count;           /* How many times has the data been accessed */
  char locked;                 /* Page locked in GPU memory? (1 = yes, 0 = no) */
  char id[32];                 /* String ID describing the block */
  struct gpu_mem_block *next;  /* Next block in linked list (NULL terminated list) */
};

typedef struct gpu_mem_block gpu_mem_block; 

void grid_cufft_make_plan(cufftHandle *, cufftType, INT, INT, INT);

