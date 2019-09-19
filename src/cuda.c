/*
 * General CUDA memory management routines.
 *
 */

#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>
#include "cuda.h"

/* Debugging */
/* #define CUDA_DEBUG */

/* Lock all pages in memory (for debugging) */
/* #define CUDA_LOCK_BLOCKS */

static gpu_mem_block *gpu_blocks_head = NULL;
static long gpu_memory_hits = 0, gpu_memory_misses = 0;
static char enable_cuda = 0;
static size_t total_alloc = 0;
static int *use_gpus, use_ngpus = 0;
static char grid_cufft_workarea;  /* gpu block memory holder */
static cufftHandle grid_cufft_highest_plan = -1;

#include "cuda-private.h"

#define EXPORT

/*
 * CUDA error checking.
 *
 * Calls abort() if something is wrong, so that the stack trace
 * can be used to locate the problem.
 *
 * NOTE: This will also do device synchronize! (= can slow things down)
 *
 */

EXPORT inline void cuda_error_check() {

  cudaError_t err;
  int i;
  
  for(i = 0; i < use_ngpus; i++) {
    cudaSetDevice(use_gpus[i]);
    if((err = cudaGetLastError()) != cudaSuccess) {
      fprintf(stderr, "libgrid(cuda): Error check for gpu %d: %s\n", i, cudaGetErrorString(err));
      abort();
    }
#ifdef CUDA_DEBUG
    if((err = cudaDeviceSynchronize()) != cudaSuccess) {
      fprintf(stderr, "libgrid(cuda): Error check (sync) for gpu %d: %s\n", i, cudaGetErrorString(err));
      abort();
    }
#endif
  }
}

/*
 * CUFFT error check.
 *
 * value = Error code (cufftResult; input).
 *
 */

EXPORT void cufft_error_check(int value) {  /* Should be cufftResult enum but this would fail on CPU-only systems */

  switch(value) {
  case CUFFT_SUCCESS:
    return;
  case CUFFT_INVALID_PLAN:
    fprintf(stderr, "Invalid plan.\n");
    break;
  case CUFFT_ALLOC_FAILED:
    fprintf(stderr, "GPU memory allocation failed.\n");
    break;
  case CUFFT_INVALID_TYPE:
    fprintf(stderr, "Invalid parameter types (invalid type).\n");
    break;
  case CUFFT_INVALID_VALUE:
    fprintf(stderr, "Invalid parameter values (invalid value).\n");
    break;
  case CUFFT_INTERNAL_ERROR:
    fprintf(stderr, "Internal driver error.\n");
    break;
  case CUFFT_EXEC_FAILED:
    fprintf(stderr, "Exec failed.\n");
    break;
  case CUFFT_SETUP_FAILED:
    fprintf(stderr, "Library failed to initialize.\n");
    break;
  case CUFFT_INVALID_SIZE:
    fprintf(stderr, "Dimension of nx, ny, or nz not supported (invalid size).\n");
    break;
  case CUFFT_UNALIGNED_DATA:
    fprintf(stderr, "Unaligned data.\n");
    break;
  case CUFFT_INCOMPLETE_PARAMETER_LIST:
    fprintf(stderr, "Incomplete parameter list.\n");
    break;
  case CUFFT_PARSE_ERROR:
    fprintf(stderr, "Parse error.\n");
    break;
  case CUFFT_NO_WORKSPACE:
    fprintf(stderr, "No workspace.\n");
    break;
  case CUFFT_NOT_IMPLEMENTED:
    fprintf(stderr, "Not implemented.\n");
    break;
  case CUFFT_LICENSE_ERROR:
    fprintf(stderr, "License error.\n");
    break;
  case CUFFT_NOT_SUPPORTED:
    fprintf(stderr, "Not supported.\n");
    break;
  default:
    fprintf(stderr, "Unknown cufft error code.\n");    
  }
  abort();
}

/*
 * Return GPU usage info.
 *
 */

EXPORT int cuda_ngpus() {

  return use_ngpus;
}

/*
 * Return GPUs to use.
 *
 */

EXPORT int *cuda_gpus() {

  return use_gpus;
}

/*
 * Allocate GPUs.
 *
 * ngpus = Number of GPUs to allocate (int). If ngpus = 0, attempt to allocate all available GPUs. In this case, gpus variable is not accessed.
 * gpus  = GPU numbers to use (int *).
 *
 * No return value.
 *
 */

EXPORT void cuda_alloc_gpus(int ngpus, int *gpus) {

  int i;

  if(!ngpus) {
    if(cudaGetDeviceCount(&ngpus) != cudaSuccess) {
      fprintf(stderr, "libgrid(cuda): Cannot get device count.\n");
      cuda_error_check();
      return;
    }
    fprintf(stderr, "libgrid(cuda): Allocating all GPUs (%d).\n", ngpus);
    if(!(use_gpus = (int *) malloc(sizeof(int) * (size_t) use_ngpus))) {
      fprintf(stderr, "libgrid(cuda): Out of memory in cuda_alloc_gpus().\n");
      abort();
    }
    for(i = 0; i < ngpus; i++) use_gpus[i] = i;
  } else {
    if(!(use_gpus = (int *) malloc(sizeof(int) * (size_t) use_ngpus))) {
      fprintf(stderr, "libgrid(cuda): Out of memory in cuda_alloc_gpus().\n");
      abort();
    }
    for(i = 0; i < ngpus; i++) use_gpus[i] = gpus[i];
  }
  use_ngpus = ngpus;
  fprintf(stderr, "libgrid(cuda): Initialized with %d GPU(s).\n", ngpus);
}

/*
 * Returns the total amount (= all allocated GPUs combined) of free GPU memory (in bytes).
 *
 */

EXPORT size_t cuda_memory() {

  int i;
  size_t free = 0, total = 0, tmp1, tmp2;

  for(i = 0; i < use_ngpus; i++) { 
    if(cudaSetDevice(use_gpus[i]) != cudaSuccess) {
      fprintf(stderr, "libgrid(cuda): Error getting memory info (setdevice).\n");
      cuda_error_check();
      return 0;
    }    
    if(cudaMemGetInfo(&tmp1, &tmp2) != cudaSuccess) {
      fprintf(stderr, "libgrid(cuda): Error getting memory info (getmeminfo).\n");
      cuda_error_check();
      return 0;
    }
    free += tmp1;
    total += tmp2;
  }
  return free;
}

/*
 * Transfer data from host memory to GPUs.
 *
 * block = Memory block to syncronize from host to GPU (gpu_mem_block *; input/output).
 *
 * Return value: 0 = OK, -1 = error.
 *
 */

EXPORT inline int cuda_mem2gpu(gpu_mem_block *block) {

  int i;
  size_t st = 0;

#ifdef CUDA_DEBUG
  fprintf(stderr, "cuda: mem2gpu from host to GPU mem (%s).\n", block->id);
#endif

  if(!block->gpu_info) {
    fprintf(stderr, "libgrid(cuda): Internal error - mem2gpu called for non-GPU resident block.\n");
    abort();
  }

  if(block->cufft_handle == -1) {
    fprintf(stderr, "libgrid(cuda): mem2gpu called for non-cufft capable block.\n");
    abort();
  }

  for(i = 0; i < use_ngpus; i++) {
    if(cudaSetDevice(block->gpu_info->descriptor->GPUs[i]) != cudaSuccess) {
      fprintf(stderr, "libgrid(cuda): mem2gpu copy error (set device).\n");
      cuda_error_check();
      abort();
    }        
    if(cudaMemcpy(block->gpu_info->descriptor->data[i], &(((char *) block->host_mem)[st]), block->gpu_info->descriptor->size[i], cudaMemcpyHostToDevice) != cudaSuccess) {
      fprintf(stderr, "libgrid(cuda): mem2gpu copy error.\n");
      cuda_error_check();
      abort();
    }        
    st += block->gpu_info->descriptor->size[i];
  }    
  cuda_error_check();

  return 0;
}

/*
 * Transfer data from GPU to host memory.
 *
 * block = Memory block to syncronize from GPU to host (gpu_mem_block *; input/output).
 *
 * Return value: 0 = OK, -1 = error.
 *
 */

EXPORT inline int cuda_gpu2mem(gpu_mem_block *block) {

  int i;
  size_t st = 0;

#ifdef CUDA_DEBUG
  fprintf(stderr, "cuda: gpu2mem from GPU to host mem (%s).\n", block->id);
#endif

  if(!block->gpu_info) {
    fprintf(stderr, "libgrid(cuda): Internal error - gpu2mem called for non-GPU resident block.\n");
    abort();
  }

  if(block->cufft_handle == -1) {
    fprintf(stderr, "libgrid(cuda): gpu2mem called for non-cufft capable block.\n");
    abort();
  }

  for(i = 0; i < use_ngpus; i++) {
    if(cudaSetDevice(block->gpu_info->descriptor->GPUs[i]) != cudaSuccess) {
      fprintf(stderr, "libgrid(cuda): gpu2mem copy error (set device).\n");
      cuda_error_check();
      abort();
    }        
    if(cudaMemcpy(&(((char *) block->host_mem)[st]), block->gpu_info->descriptor->data[i], block->gpu_info->descriptor->size[i], cudaMemcpyDeviceToHost) != cudaSuccess) {
      fprintf(stderr, "libgrid(cuda): gpu2mem copy error.\n");
      cuda_error_check();
      abort();
    }        
    st += block->gpu_info->descriptor->size[i];
  }    
  cuda_error_check();

  return 0;
}

/*
 * Transfer data from one area in GPU to another
 *
 * dst     = (destination) GPU buffer (gpu_mem_block *; output).
 * src     = (source) GPU buffer (gpu_mem_block *; input).
 *
 * Return value: 0 = OK, -1 = error.
 *
 */

EXPORT inline int cuda_gpu2gpu(gpu_mem_block *dst, gpu_mem_block *src) {

  size_t i;

#ifdef CUDA_DEBUG
  fprintf(stderr, "cuda: gpu2gpu from GPU (%s) to GPU (%s).\n", src->id, dst->id);
#endif

  if(!src->gpu_info || !dst->gpu_info) {
    fprintf(stderr, "libgrid(cuda): Internal error - gpu2gpu called for non-GPU resident block(s).\n");
    abort();
  }

  for(i = 0; i < dst->gpu_info->descriptor->nGPUs; i++) {
    if(cudaSetDevice(dst->gpu_info->descriptor->GPUs[i]) != cudaSuccess) {
      fprintf(stderr, "libgrid(cuda): gpu2gpu copy error (set device).\n");
      cuda_error_check();
      return -1;
    }        
    if(cudaMemcpy(dst->gpu_info->descriptor->data[i], src->gpu_info->descriptor->data[i], src->gpu_info->descriptor->size[i], cudaMemcpyDeviceToDevice) != cudaSuccess) {
      fprintf(stderr, "libgrid(cuda): gpu2gpu copy error.\n");
      cuda_error_check();
      return -1;
    }
  }
  dst->gpu_info->subFormat = src->gpu_info->subFormat;
  cuda_error_check();

  return 0;
}

/*
 * Find GPU memory block based on host_mem address (or to check if host_mem block on GPU).
 *
 * host_mem = Host memory address to identify the GPU memory block (void *; input).
 *
 * Return: Pointer to gpu_mem_block or NULL (not found).
 *
 */

EXPORT gpu_mem_block *cuda_find_block(void *host_mem) {

  gpu_mem_block *ptr;

  if(!enable_cuda) return NULL;
#ifdef CUDA_DEBUG
  fprintf(stderr, "cuda: find block by host mem %lx - ", (unsigned long int) host_mem);
#endif
  for(ptr = gpu_blocks_head; ptr; ptr = ptr->next)
    if(host_mem == ptr->host_mem) {
#ifdef CUDA_DEBUG
      fprintf(stderr, "found.\n");
#endif
      return ptr;
    }
#ifdef CUDA_DEBUG
  fprintf(stderr, "not found.\n");
#endif
  return NULL;
}  

/*
 * Move memory block from GPU back to host memory.
 *
 * host_mem = Host memory address to identify the block (void *; input).
 * copy     = Synchronize the contents of the block to host memory before removing it from GPU? (1 = yes, 0 = no)
 *
 * Return value: 0 = OK, -1 = error.
 *
 */

EXPORT int cuda_remove_block(void *host_mem, char copy) {

  gpu_mem_block *ptr, *prev;
  int i;

#ifdef CUDA_DEBUG
  fprintf(stderr, "cuda: Remove block %lx with copy %d.\n", (unsigned long int) host_mem, copy);
#endif
  for(ptr = gpu_blocks_head, prev = NULL; ptr; prev = ptr, ptr = ptr->next)
    if(host_mem == ptr->host_mem) break;

  if(!ptr) {
#ifdef CUDA_DEBUG
    fprintf(stderr, "cuda: block not found.\n");
#endif
    return 0; /* not on GPU */
  }
  if(copy) {
#ifdef CUDA_DEBUG
    fprintf(stderr, "cuda: Syncing memory of the removed block from GPU to memory.\n");
#endif
    cuda_gpu2mem(ptr);
  }
#ifndef CUDA_LOCK_BLOCKS
  if(ptr->locked) {
    fprintf(stderr, "libgrid(cuda): Error - attempt to remove locked block (%s)!\n", ptr->id);
    fprintf(stderr, "libgrid(cuda): Bailing out...\n");
    abort();
  }
#else
  if(ptr->locked) return 0;  // keep it in memory
#endif
#ifdef CUDA_DEBUG
  fprintf(stderr, "cuda: removed (%s).\n", ptr->id);
#endif
  for(i = 0; i < ptr->gpu_info->descriptor->nGPUs; i++)
    total_alloc -= ptr->gpu_info->descriptor->size[i];
  if(use_ngpus == 1 || ptr->cufft_handle == -1) {
    for(i = 0; i < ptr->gpu_info->descriptor->nGPUs; i++) {
      if(cudaSetDevice(use_gpus[i]) != cudaSuccess) {
        fprintf(stderr, "libgrid(cuda): Error removing block (setdevice).\n");
        cuda_error_check();
        return 0;
      }    
      cudaFree(ptr->gpu_info->descriptor->data[i]);
    }
    cuda_error_check();
    free(ptr->gpu_info->descriptor);
    free(ptr->gpu_info);
  } else {
    cufftResult status;
    if((status = cufftXtFree(ptr->gpu_info)) != CUFFT_SUCCESS) {
      fprintf(stderr, "libgrid(cuda): Failed to free memory.\n");
      cufft_error_check(status);
    }
  }
  if(prev) prev->next = ptr->next;
  else gpu_blocks_head = ptr->next;
  free(ptr);
  return 0;
}

/*
 * Attempt to allocate memory for new block (not to be called directly).
 *
 * Returns -1 for failure or 0 for success.
 *
 */

static int alloc_mem(gpu_mem_block *block, size_t length) {

  int i;

  if(use_ngpus == 1 || block->cufft_handle == -1) {
    /* This is for GPU blocks that are not to be used with cufft */
    if(!(block->gpu_info = (cudaLibXtDesc *) malloc(sizeof(cudaLibXtDesc)))) {
      fprintf(stderr, "libgrid(cuda): Out of memory in alloc_mem().\n");
      abort();
    }
    if(!(block->gpu_info->descriptor = (cudaXtDesc *) malloc(sizeof(cudaXtDesc)))) {
      fprintf(stderr, "libgrid(cuda): Out of memory in alloc_mem().\n");
      abort();
    }
    block->gpu_info->version = 0;
    block->gpu_info->descriptor->version = CUDA_XT_DESCRIPTOR_VERSION;
    block->gpu_info->descriptor->nGPUs = use_ngpus;
    for(i = 0; i < use_ngpus; i++) {
      block->gpu_info->descriptor->GPUs[i] = use_gpus[i];
      if(cudaSetDevice(use_gpus[i]) != cudaSuccess) {
        fprintf(stderr, "libgrid(cuda): Error allocating memory (setdevice).\n");
        cuda_error_check();
        exit(1);
      }    
      if(cudaMalloc((void **) &(block->gpu_info->descriptor->data[i]), length) != cudaSuccess) {
        int j;
        for(j = 0; j < i; j++) {
           if(cudaSetDevice(use_gpus[j]) != cudaSuccess) {
            fprintf(stderr, "libgrid(cuda): Error alllocating memory (setdevice).\n");
            cuda_error_check();
            exit(0);
          }    
          cudaFree(block->gpu_info->descriptor->data[j]);
        }
        free(block->gpu_info->descriptor);
        free(block->gpu_info);
        block->gpu_info = NULL;
        return -1;
      }
      block->gpu_info->descriptor->size[i] = length;  /* Reserve length amount of memory on ALL GPUs */
    }
    block->gpu_info->descriptor->cudaXtState = NULL;
    block->gpu_info->library = LIB_FORMAT_CUFFT;
    block->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
    block->gpu_info->libDescriptor = NULL;
  } else {    /* cufft capable blocks */
    cufftResult status;
    if((status = cufftXtMalloc(block->cufft_handle, &(block->gpu_info), CUFFT_XT_FORMAT_INPLACE)) != CUFFT_SUCCESS) {
      cufft_error_check(status);
      return -1;
    }
  }
  return 0;
}

/*
 * Add block (from host memory to GPU). If there is not enough space in GPU memory,
 * this may swap out another block(s) based on their last use stamp.
 *
 * host_mem     = Host memory pointer containing the data (void *; input).
 * length       = Length of host memory data (size_t; input). If cufft_handle == -1, 
 *                allocate this amount on all GPUs, otherwise use CUFFT partitioning of data over the GPUs (length not used; this is contained in the cufft handle).
 * cufft_handle = CUFFT handle (if not known, -1; allocates length amount of memory on all GPUs). If != -1, CUFFT multi GPU routines are used for managing memory (cufftHandle).
 * id           = String describing the block contents. Useful for debugging. (char *; input).
 * copy         = Copy host_mem to gpu_mem? (1 = yes, 0 = no).
 *
 * Return value: Pointer to new gpu_block_mem or NULL = error.
 *
 */

EXPORT gpu_mem_block *cuda_add_block(void *host_mem, size_t length, cufftHandle cufft_handle, char *id, char copy) {

  gpu_mem_block *ptr, *rptr = NULL, *new;
  time_t current;
  long current_access;

  if(!enable_cuda) return NULL;
#ifdef CUDA_DEBUG
  fprintf(stderr, "cuda: Add block %lx (%s) of length %ld with copy %d.\n", (unsigned long int) host_mem, id, length, copy);
#endif
  if((ptr = cuda_find_block(host_mem))) { /* Already in GPU memory? */
    cuda_block_hit(ptr);
#ifdef CUDA_DEBUG
    fprintf(stderr, "cuda: Already in GPU memory.\n");
#endif
    return ptr;
  } else cuda_block_miss();

#ifdef CUDA_DEBUG
  fprintf(stderr, "cuda: Not in GPU memory - trying to add.\n");
#endif
  if(!(new = (gpu_mem_block *) malloc(sizeof(gpu_mem_block)))) {
    fprintf(stderr, "libgrid(cuda): Out of memory in allocating gpu_mem_block.\n");
    abort();
  }
  new->cufft_handle = cufft_handle;

  /* Data not in GPU - try to allocate & possibly swap out other blocks */
  while(alloc_mem(new, length) == -1) { // If successful, also fills out new->gpu_info
    /* search for swap out candidate (later: also consider memory block sizes) */
    current = time(0) + 1;
    current_access = 0;
    for(ptr = gpu_blocks_head, rptr = NULL; ptr; ptr = ptr->next) {
      if(!ptr->locked && (ptr->last_used < current || (ptr->last_used == current && ptr->access_count < current_access))) {
        current = ptr->last_used;
        current_access = ptr->access_count;
        rptr = ptr;
      }
    }
    if(!rptr) {
      free(new);
      return NULL; /* We ran out of blocks that could be swapped out ! */
    }
#ifdef CUDA_DEBUG
    fprintf(stderr, "cuda: Swap out block host mem: %lx (%s).\n", (unsigned long int) rptr->host_mem, rptr->id);
#endif
    cuda_remove_block(rptr->host_mem, 1);
  }
#ifdef CUDA_DEBUG
  fprintf(stderr, "cuda: Succesfully added new block to GPU.\n");
#endif
  total_alloc += length;
  new->host_mem = host_mem;
  new->next = gpu_blocks_head;
  gpu_blocks_head = new;
  new->created = new->last_used = time(0);
  new->access_count = 1;
#ifdef CUDA_LOCK_BLOCKS
  new->locked = 1;
#else
  new->locked = 0;
#endif
  strncpy(new->id, id, 32);
  new->id[31] = 0;
  if(copy) {
#ifdef CUDA_DEBUG
    fprintf(stderr, "cuda: Syncing memory of the new block to GPU.\n");
#endif
    cuda_mem2gpu(new);
  }

  return new;
}

/*
 * Lock block to GPU memory.
 * A locked memory block cannot be swapped out of GPU.
 *
 * host_mem = Host memory to be locked to GPU memory (void *; input).
 *
 * Return value: 0 = OK, -1 = error (not on GPU).
 *
 */

EXPORT int cuda_lock_block(void *host_mem) {

  gpu_mem_block *ptr;

  if(!enable_cuda) return -1;
#ifdef CUDA_DEBUG
  fprintf(stderr, "cuda: Lock block request for host mem %lx.\n", (unsigned long int) host_mem);
#endif
  if(!(ptr = cuda_find_block(host_mem))) {
#ifdef CUDA_DEBUG
    fprintf(stderr, "Block NOT FOUND.\n");
#endif
    return -1;
  }
#ifdef CUDA_DEBUG
  fprintf(stderr, "cuda: Host mem %lx (%s) locked.\n", (unsigned long int) host_mem, ptr->id);
#endif
  ptr->locked = 1;
  return 0;
}

/*
 * Unlock block from GPU memory.
 * An unlocked memory block can be swapped out of GPU.
 *
 * host_mem = Host memory to be locked to GPU memory (void *; input).
 *
 * Return value: 0 = OK, -1 = error (not on GPU).
 *
 */

EXPORT int cuda_unlock_block(void *host_mem) {

  gpu_mem_block *ptr;

  if(!enable_cuda) return -1;
#ifdef CUDA_LOCK_BLOCKS
  return 0;
#endif
#ifdef CUDA_DEBUG
  fprintf(stderr, "cuda: Unlock block reqest for host mem %lx.\n", (unsigned long int) host_mem);
#endif
  if(!(ptr = cuda_find_block(host_mem))) {
#ifdef CUDA_DEBUG
    fprintf(stderr, "Block not found.\n");
#endif
    return -1;
  }
#ifdef CUDA_DEBUG
  fprintf(stderr, "Host mem %lx (%s) unlocked.\n", (unsigned long int) host_mem, ptr->id);
#endif
  ptr->locked = 0;
  return 0;
}

/*
 * Add two blocks to GPU simulatenously (or keep both at CPU). Other blocks may be swapped out
 * or the two blocks may not fit in at the same time.
 *
 * host_mem1     = Host memory pointer 1 (void *; input).
 * length1       = Length of host memory pointer 1 (size_t; input).
 * cufft_handle1 = CUFFT handle for 1 (cufftHandle).
 * id1           = String describing block 1 contents. Useful for debugging. (char *; input).
 * copy1         = Copy contents of block 1 to GPU? 1 = copy, 0 = don't copy.
 * host_mem2     = Host memory pointer 2 (void *; input).
 * length2       = Length of host memory pointer 2 (size_t; input).
 * cufft_handle2 = CUFFT handle for 2 (cufftHandle).
 * id2           = String describing block 2 contents. Useful for debugging. (char *; input).
 * copy2         = Copy contents of block 2 to GPU? 1 = copy, 0 = don't copy.
 *
 * If both memory blocks can be allocated in GPU, their contents will
 * be transferred there and 0 is returned.
 *
 * If neither of the blocks can be allocated in GPU, both blocks will
 * be pushed back to host memory and -1 is returned.
 *
 * Note that this may end up in unresolvable situation if:
 * one of the blocks is locked to GPU and the other one does not fit there!
 *
 */

EXPORT char cuda_add_two_blocks(void *host_mem1, size_t length1, cufftHandle cufft_handle1, char *id1, char copy1, void *host_mem2, size_t length2, cufftHandle cufft_handle2, char *id2, char copy2) {

  gpu_mem_block *block1, *test;
  char l1;

  if(!enable_cuda) return -1;
  if(host_mem1 == host_mem2) return cuda_add_block(host_mem1, length1, cufft_handle1, id1, (char) (copy1 + copy2))?0:-1;

#ifdef CUDA_DEBUG
  fprintf(stderr, "cuda: Request two blocks with host mems %lx with len %ld (%s) and %lx with len %ld (%s)...\n", (unsigned long int) host_mem1, length1, id1, (unsigned long int) host_mem2, length2, id2);
#endif
  test = cuda_find_block(host_mem1);
  if(!(block1 = cuda_add_block(host_mem1, length1, cufft_handle1, id1, 0))) {
    /* both need to be in host memory */
#ifdef CUDA_DEBUG
    fprintf(stderr, "cuda: Failed to add host_mem1 to GPU, making sure host_mem2 is removed from GPU.\n");
#endif
    cuda_remove_block(host_mem2, copy2);  // remove 2 (may or may not be in GPU)
    return -1;
  }

  l1 = block1->locked;
  block1->locked = 1;
  if(!(cuda_add_block(host_mem2, length2, cufft_handle2, id2, copy2))) {
    block1->locked = l1;
    /* both need to be in host memory */
#ifdef CUDA_DEBUG
    fprintf(stderr, "cuda: Failed to add host_mem2 to GPU, removing host_mem from GPU1.\n");
#endif
    if(test) cuda_remove_block(host_mem1, copy1); // was already in GPU but now needs to be synced to host memory when removing
    else cuda_remove_block(host_mem1, 0);
    return -1;
  }
  block1->locked = l1;

  if(!test && copy1) {
#ifdef CUDA_DEBUG
    fprintf(stderr, "cuda: Syncing host_mem1 to GPU.\n");
#endif
    cuda_mem2gpu(block1);  // Now that everything is OK, we need to also sync block1 to GPU (as it wasn't there already)
  }
  /* Both blocks now successfully in GPU memory */
#ifdef CUDA_DEBUG
  fprintf(stderr, "cuda: Successful two-block add.\n");
#endif
  return 0;  
}

/*
 * Add three blocks to GPU simulatenously. Other blocks may be swapped out
 * or the three blocks may not fit in at the same time.
 *
 * host_mem1     = Host memory pointer 1 (void *; input).
 * length1       = Length of host memory pointer 1 (size_t; input).
 * cufft_handle1 = CUFFT Handle for 1 (cufftHandle).
 * id1           = String describing block 1 contents. Useful for debugging. (char *; input).
 * copy1         = Copy contents of block 1 to GPU? 1 = copy, 0 = don't copy.
 * host_mem2     = Host memory pointer 2 (void *; input).
 * length2       = Length of host memory pointer 2 (size_t; input).
 * cufft_handle2 = CUFFT Handle for 2 (cufftHandle).
 * id2           = String describing block 2 contents. Useful for debugging. (char *; input).
 * copy2         = Copy contents of block 2 to GPU? 1 = copy, 0 = don't copy.
 * host_mem3     = Host memory pointer 3 (void *; input).
 * length3       = Length of host memory pointer 3 (size_t; input).
 * cufft_handle3 = CUFFT Handle for 3 (cufftHandle).
 * id3           = String describing block 3 contents. Useful for debugging. (char *; input).
 * copy3         = Copy contents of block 3 to GPU? 1 = copy, 0 = don't copy.
 *
 * If all three memory blocks can be allocated in GPU, their contents will
 * be transferred there and 0 is returned.
 *
 * If any of the blocks can be allocated in GPU, all three blocks will
 * be pushed back to host memory and -1 is returned.
 *
 * Note that this may end up in unresolvable situation if:
 * one of the blocks is locked to GPU and the other ones do not fit there!
 *
 */

EXPORT char cuda_add_three_blocks(void *host_mem1, size_t length1, cufftHandle cufft_handle1, char *id1, char copy1, void *host_mem2, size_t length2, cufftHandle cufft_handle2, char *id2, char copy2, void *host_mem3, size_t length3, cufftHandle cufft_handle3, char *id3, char copy3) {

  gpu_mem_block *block1, *block2, *test1, *test2, *test3;
  char l1, l2;

  if(!enable_cuda) return -1;
  if(host_mem1 == host_mem2) return cuda_add_two_blocks(host_mem1, length1, cufft_handle1, id1, (char) (copy1 + copy2), host_mem3, length3, cufft_handle3, id3, copy3);
  if(host_mem1 == host_mem3) return cuda_add_two_blocks(host_mem1, length1, cufft_handle1, id1, (char) (copy1 + copy3), host_mem2, length2, cufft_handle2, id2, copy2);
  if(host_mem2 == host_mem3) return cuda_add_two_blocks(host_mem1, length1, cufft_handle1, id1, copy1, host_mem2, length2, cufft_handle2, id2, (char) (copy2 + copy3));
#ifdef CUDA_DEBUG
  fprintf(stderr, "cuda: Request three blocks with host mems %lx with len %ld (%s), %lx with len %ld (%s), and %lx with len %ld (%s)...\n", (unsigned long int) host_mem1, length1, id1, (unsigned long int) host_mem2, length2, id2, (unsigned long int) host_mem3, length3, id3);
#endif
  test1 = cuda_find_block(host_mem1);
  test2 = cuda_find_block(host_mem2);
  test3 = cuda_find_block(host_mem3);
  if(!(block1 = cuda_add_block(host_mem1, length1, cufft_handle1, id1, 0))) {
#ifdef CUDA_DEBUG
    fprintf(stderr, "cuda: Failed to add host_mem1 to GPU, making sure host_mem2 and host_mem3 are removed from GPU.\n");
#endif
    if(test2) cuda_remove_block(host_mem2, copy2);
    if(test3) cuda_remove_block(host_mem3, copy3);
    return -1;
  }
  l1 = block1->locked;
  block1->locked = 1;
  if(!(block2 = cuda_add_block(host_mem2, length2, cufft_handle2, id2, 0))) {
#ifdef CUDA_DEBUG
    fprintf(stderr, "cuda: Failed to add host_mem2 to GPU, making sure host_mem1 and host_mem3 are removed from GPU.\n");
#endif
    block1->locked = l1;
    if(test1) cuda_remove_block(host_mem1, copy1);   // was already in GPU, need to sync back to host memory
    else cuda_remove_block(host_mem1, 0);        // wasn't in GPU, no need to sync
    if(test3) cuda_remove_block(host_mem3, copy3);
    return -1;
  }
  l2 = block2->locked;
  block2->locked = 1;
  if(!(cuda_add_block(host_mem3, length3, cufft_handle3, id3, copy3))) {
#ifdef CUDA_DEBUG
    fprintf(stderr, "cuda: Failed to add host_mem3 to GPU, making sure host_mem1 and host_mem2 are removed from GPU.\n");
#endif
    block1->locked = l1;
    block2->locked = l2;
    if(test1) cuda_remove_block(host_mem1, copy1);
    else cuda_remove_block(host_mem1, 0);
    if(test2) cuda_remove_block(host_mem2, copy2);
    else cuda_remove_block(host_mem2, 0);
    return -1;
  }
  block1->locked = l1;
  block2->locked = l2;
  if(!test1 && copy1) {
#ifdef CUDA_DEBUG
    fprintf(stderr, "cuda: Syncing host_mem1 to GPU.\n");
#endif
    cuda_mem2gpu(block1);  // Now that everything is OK, we need to also sync block1
  }
  if(!test2 && copy2) {
#ifdef CUDA_DEBUG
    fprintf(stderr, "cuda: Syncing host_mem2 to GPU.\n");
#endif
    cuda_mem2gpu(block2);  // Now that everything is OK, we need to also sync block2
  }
#ifdef CUDA_DEBUG
  fprintf(stderr, "cuda: Successful three-block add.\n");
#endif
  return 0;  
}

/*
 * Add four blocks to GPU simulatenously. Other blocks may be swapped out
 * or the four blocks may not fit in at the same time.
 *
 * host_mem1     = Host memory pointer 1 (void *; input).
 * length1       = Length of host memory pointer 1 (size_t; input).
 * id1           = String describing block 1 contents. Useful for debugging. (char *; input).
 * cufft_handle1 = CUFFT Handle for 1 (cufftHandle).
 * copy1         = Copy contents of block 1 to GPU? 1 = copy, 0 = don't copy.
 * host_mem2     = Host memory pointer 2 (void *; input).
 * length2       = Length of host memory pointer 2 (size_t; input).
 * cufft_handle2 = CUFFT Handle for 2 (cufftHandle).
 * id2           = String describing block 2 contents. Useful for debugging. (char *; input).
 * copy2         = Copy contents of block 2 to GPU? 1 = copy, 0 = don't copy.
 * host_mem3     = Host memory pointer 3 (void *; input).
 * length3       = Length of host memory pointer 3 (size_t; input).
 * cufft_handle3 = CUFFT Handle for 3 (cufftHandle).
 * id3           = String describing block 3 contents. Useful for debugging. (char *; input).
 * copy3         = Copy contents of block 3 to GPU? 1 = copy, 0 = don't copy.
 * host_mem4     = Host memory pointer 4 (void *; input).
 * length4       = Length of host memory pointer 4 (size_t; input).
 * cufft_handle4 = CUFFT Handle for 4 (cufftHandle).
 * id4           = String describing block 4 contents. Useful for debugging. (char *; input).
 * copy4         = Copy contents of block 4 to GPU? 1 = copy, 0 = don't copy.
 *
 * If all four memory blocks can be allocated in GPU, their contents will
 * be transferred there and 0 is returned.
 *
 * If any of the blocks can be allocated in GPU, all four blocks will
 * be pushed back to host memory and -1 is returned.
 *
 * Note that this may end up in unresolvable situation if:
 * one of the blocks is locked to GPU and the other ones do not fit there!
 *
 */

EXPORT char cuda_add_four_blocks(void *host_mem1, size_t length1, cufftHandle cufft_handle1, char *id1, char copy1, void *host_mem2, size_t length2, cufftHandle cufft_handle2, char *id2, char copy2, void *host_mem3, size_t length3, cufftHandle cufft_handle3, char *id3, char copy3, void *host_mem4, size_t length4, cufftHandle cufft_handle4, char *id4, char copy4) {

  gpu_mem_block *block1, *block2, *block3, *test1, *test2, *test3, *test4;
  char l1, l2, l3;

  if(!enable_cuda) return -1;
  if(host_mem1 == host_mem2) return cuda_add_three_blocks(host_mem1, length1, cufft_handle1, id1, (char) (copy1 + copy2), host_mem3, length3, cufft_handle3, id3, copy3, host_mem4, length4, cufft_handle4, id4, copy4);
  if(host_mem1 == host_mem3) return cuda_add_three_blocks(host_mem1, length1, cufft_handle1, id1, (char) (copy1 + copy3), host_mem2, length2, cufft_handle2, id2, copy2, host_mem4, length4, cufft_handle4, id4, copy4);
  if(host_mem1 == host_mem4) return cuda_add_three_blocks(host_mem1, length1, cufft_handle1, id1, (char) (copy1 + copy4), host_mem2, length2, cufft_handle2, id2, copy2, host_mem3, length3, cufft_handle3, id3, copy3);
  if(host_mem2 == host_mem3) return cuda_add_three_blocks(host_mem1, length1, cufft_handle1, id1, copy1, host_mem2, length2, cufft_handle2, id2, (char) (copy2 + copy3), host_mem4, length4, cufft_handle4, id4, copy4);
  if(host_mem2 == host_mem4) return cuda_add_three_blocks(host_mem1, length1, cufft_handle1, id1, copy1, host_mem2, length2, cufft_handle2, id2, (char) (copy2 + copy4), host_mem3, length3, cufft_handle3, id3, copy3);
  if(host_mem3 == host_mem4) return cuda_add_three_blocks(host_mem1, length1, cufft_handle1, id1, copy1, host_mem2, length2, cufft_handle2, id2, copy2, host_mem3, length3, cufft_handle3, id3, (char) (copy3 + copy4));
#ifdef CUDA_DEBUG
  fprintf(stderr, "cuda: Request four blocks with host mems %lx with len %ld (%s), %lx with len %ld (%s), %lx with len %ld (%s), and %lx with len %ld (%s)...\n", (unsigned long int) host_mem1, length1, id1, (unsigned long int) host_mem2, length2, id2, (unsigned long int) host_mem3, length3, id3, (unsigned long int) host_mem4, length4, id4);
#endif
  test1 = cuda_find_block(host_mem1);
  test2 = cuda_find_block(host_mem2);
  test3 = cuda_find_block(host_mem3);
  test4 = cuda_find_block(host_mem4);
  if(!(block1 = cuda_add_block(host_mem1, length1, cufft_handle1, id1, 0))) {
#ifdef CUDA_DEBUG
    fprintf(stderr, "cuda: Failed to add host_mem1 to GPU, making sure host_mem2, host_mem3, and host_mem4 are removed from GPU.\n");
#endif
    if(test2) cuda_remove_block(host_mem2, copy2);
    if(test3) cuda_remove_block(host_mem3, copy3);
    if(test4) cuda_remove_block(host_mem4, copy4);
    return -1;
  }
  l1 = block1->locked;
  block1->locked = 1;
  if(!(block2 = cuda_add_block(host_mem2, length2, cufft_handle2, id2, 0))) {
#ifdef CUDA_DEBUG
    fprintf(stderr, "cuda: Failed to add host_mem2 to GPU, making sure host_mem1, host_mem3, and host_mem4 are removed from GPU.\n");
#endif
    block1->locked = l1;
    if(test1) cuda_remove_block(host_mem1, copy1);
    else cuda_remove_block(host_mem1, 0);
    if(test3) cuda_remove_block(host_mem3, copy3);
    if(test4) cuda_remove_block(host_mem4, copy4);
    return -1;
  }
  l2 = block2->locked;
  block2->locked = 1;
  if(!(block3 = cuda_add_block(host_mem3, length3, cufft_handle3, id3, 0))) {
#ifdef CUDA_DEBUG
    fprintf(stderr, "cuda: Failed to add host_mem3 to GPU, making sure host_mem1, host_mem2, and host_mem4 are removed from GPU.\n");
#endif
    block1->locked = l1;
    block2->locked = l2;
    if(test1) cuda_remove_block(host_mem1, copy1);
    else cuda_remove_block(host_mem1, 0);
    if(test2) cuda_remove_block(host_mem2, copy2);
    else cuda_remove_block(host_mem2, 0);
    if(test4) cuda_remove_block(host_mem4, copy4);
    return -1;
  }
  l3 = block3->locked;
  block3->locked = 1;
  if(!(cuda_add_block(host_mem4, length4, cufft_handle4, id4, copy4))) {
#ifdef CUDA_DEBUG
    fprintf(stderr, "cuda: Failed to add host_mem4 to GPU, making sure host_mem1, host_mem2, and host_mem3 are removed from GPU.\n");
#endif
    block1->locked = l1;
    block2->locked = l2;
    block3->locked = l3;
    if(test1) cuda_remove_block(host_mem1, copy1);
    else cuda_remove_block(host_mem1, 0);
    if(test2) cuda_remove_block(host_mem2, copy2);
    else cuda_remove_block(host_mem2, 0);
    if(test3) cuda_remove_block(host_mem3, copy3);
    else cuda_remove_block(host_mem3, 0);
    return -1;
  }
  block1->locked = l1;
  block2->locked = l2;
  block3->locked = l3;
  if(!test1 && copy1) {
#ifdef CUDA_DEBUG
    fprintf(stderr, "cuda: Syncing host_mem1 to GPU.\n");
#endif
    cuda_mem2gpu(block1);  // Now that everything is OK, we need to also sync block1
  }
  if(!test2 && copy2) {
#ifdef CUDA_DEBUG
    fprintf(stderr, "cuda: Syncing host_mem2 to GPU.\n");
#endif
    cuda_mem2gpu(block2);  // Now that everything is OK, we need to also sync block2
  }
  if(!test3 && copy3) {
#ifdef CUDA_DEBUG
    fprintf(stderr, "cuda: Syncing host_mem3 to GPU.\n");
#endif
    cuda_mem2gpu(block3);  // Now that everything is OK, we need to also sync block2
  }
  return 0;
}

/*
 * Fetch one element from a GPU/CPU array.
 * If the data is not on GPU, it will be retrieved from host memory instead.
 *
 * host_mem = Host memory for output (void *; input).
 * gpu      = Which GPU to access (int).
 * index    = Index for the host memory array (size_t; input).
 * size     = Size of each element in bytes for indexing (size_t; input).
 * value    = Where the value will be stored (void *; output).
 *
 * Returns -1 for error, 0 = OK.
 *
 * Avoid calling this repeatedly - VERY SLOW!
 *
 */

EXPORT int cuda_get_element(void *host_mem, int gpu, size_t index, size_t size, void *value) {

  gpu_mem_block *ptr;

#ifdef CUDA_DEBUG
  fprintf(stderr, "cuda: Request to fetch element from host mem %lx of size %ld.\n", (unsigned long int) host_mem, size);
#endif
  if(!enable_cuda || !(ptr = cuda_find_block(host_mem))) {
#ifdef CUDA_DEBUG
    fprintf(stderr, "cuda: found in host memory.\n");
#endif
    memcpy(value, &(((char *) host_mem)[index * size]), size);
    return 0;
  }
#ifdef CUDA_DEBUG
  fprintf(stderr, "cuda: found in GPU memory.\n");
#endif
  if(cudaSetDevice(gpu) != cudaSuccess) {
    fprintf(stderr, "libgrid(cuda): Error getting element (setdevice).\n");
    cuda_error_check();
    return 0;
  }    
  if(cudaMemcpy(value, &(((char *) ptr->gpu_info->descriptor->data[gpu])[index * size]), size, cudaMemcpyDeviceToHost) != cudaSuccess) {
    fprintf(stderr, "libgrid(cuda): read failed in cuda_get_element().\n");
    abort();
  }
  return 0;
}

/*
 * Set value for one element on a GPU/CPU array.
 * If the data is not on GPU, it will be set in host memory instead.
 *
 * host_mem = Host memory (void *; output).
 * gpu      = Which GPU to access (int).
 * index    = Index for the host memory array (size_t; input).
 * size     = Size of each element in bytes for indexing (size_t; input).
 * value    = The value that will be stored (void *; input).
 *
 * Returns -1 for error, 0 = OK.
 *
 * Avoid calling this repeatedly - VERY SLOW!
 *
 */

EXPORT int cuda_set_element(void *host_mem, int gpu, size_t index, size_t size, void *value) {

  gpu_mem_block *ptr;

#ifdef CUDA_DEBUG
  fprintf(stderr, "cuda: Request to set element to host mem %lx of size %ld.\n", (unsigned long int) host_mem, size);
#endif
  if(!enable_cuda || !(ptr = cuda_find_block(host_mem))) {
#ifdef CUDA_DEBUG
    fprintf(stderr, "cuda: found in host memory.\n");
#endif
    memcpy(&(((char *) host_mem)[index * size]), value, size);  // In host memory
    return 0;
  }
#ifdef CUDA_DEBUG
  fprintf(stderr, "cuda: found in GPU memory.\n");
#endif
  if(cudaSetDevice(gpu) != cudaSuccess) {
    fprintf(stderr, "libgrid(cuda): Error setting element (setdevice).\n");
    cuda_error_check();
    return 0;
  }    
  if(cudaMemcpy(&(((char *) ptr->gpu_info->descriptor->data[gpu])[index * size]), value, size, cudaMemcpyHostToDevice) != cudaSuccess) return -1;
  return 0;
}

/*
 * Free all GPU memory blocks (except locked) and optionally sync them to host memory.
 *
 * sync  = 1 sync GPU to host, 0 = no sync (char; input).
 *
 */

EXPORT void cuda_free_all_blocks(char sync) {

  gpu_mem_block *ptr, *ptr2;

  if(!enable_cuda) return;
#ifdef CUDA_DEBUG
  fprintf(stderr, "cuda: Releasing all GPU memory blocks.\n");
#endif
  for(ptr = gpu_blocks_head; ptr; ptr = ptr2) {
    ptr2 = ptr->next;
    if(!ptr->locked) cuda_remove_block(ptr->host_mem, sync);
  }
  gpu_memory_hits = gpu_memory_misses = 0;
}

/*
 * Enable/disable CUDA.
 *
 * val   = 0 to disable CUDA or val = 1 to enable CUDA. 
 * ngpus = Number of GPUs to allocate (int). If val = 0, attempt to allocate all gpus and the gpus array not accessed.
 * gpus  = GPU numbers to use (int *). If val = 0, not accessed.
 *
 * Disabling active CUDA will flush GPU memory pages to the host memory.
 *
 */

EXPORT void cuda_enable(char val, int ngpus, int *gpus) {

  static char been_here = 0;
  void cuda_gpu_info();

#ifdef CUDA_DEBUG
  fprintf(stderr, "cuda: Enable/disable %d.\n", val);
#endif
  if(enable_cuda && val == 0) cuda_free_all_blocks(1);  // disable CUDA; flush GPU memory blocks
  enable_cuda = val;
  if(val) {
    fprintf(stderr, "libgrid(cuda): CUDA enabled.\n");
    cuda_alloc_gpus(ngpus, gpus);
    if(!been_here) {
      cuda_gpu_info();
      been_here = 1;
    }
  }
}

/*
 * CUDA status.
 *
 * Returns 1 if CUDA enabled, 0 otherwise.
 *
 */

EXPORT char cuda_status() {

  return enable_cuda;
}

/*
 * Output memory allocation statustics.
 *
 * verbose =  0 = print only summary; 1 = print also GPU memory block information.
 *
 * No return value.
 *
 */

EXPORT void cuda_statistics(char verbose) {

  gpu_mem_block *ptr;
  long n = 0, nl = 0, lc = 999999, hc = 0;
  size_t total_size = 0;
  time_t oldest = time(0), newest = 0;
  int i;

  if(!enable_cuda) {
    fprintf(stderr, "CUDA not enabled.\n");
    return;
  }
  for(ptr = gpu_blocks_head; ptr; ptr = ptr->next, n++) {
    if(ptr->locked) nl++;
    for(i = 0; i < use_ngpus; i++)
      total_size += ptr->gpu_info->descriptor->size[i];
    if(ptr->last_used < oldest) oldest = ptr->last_used;
    if(ptr->last_used > newest) newest = ptr->last_used;
    if(lc > ptr->access_count) lc = ptr->access_count;
    if(hc < ptr->access_count) hc = ptr->access_count;
  }
 
  fprintf(stderr, "Total amount of GPU memory avail. : %ld MB\n", cuda_memory() / (1024 * 1024));
  fprintf(stderr, "Amount of memory used for blocks  : %ld MB.\n", total_size / (1024 * 1024));
  if(n) fprintf(stderr, "Average block size                : %ld MB.\n", total_size / (((size_t) n) * 1024 * 1024));
  fprintf(stderr, "Number of GPU memory blocks in use: %ld\n", n);
  fprintf(stderr, "Total memory allocated by libgrid : %ld MB\n", total_alloc / (1024 * 1024));
  fprintf(stderr, "Number of locked GPU memory blocks: %ld\n", nl);
  fprintf(stderr, "Oldest block time stamp           : %s", ctime(&oldest));  
  fprintf(stderr, "Most recent block time stamp      : %s", ctime(&newest));  
  fprintf(stderr, "Highest block access count        : %ld\n", hc);
  fprintf(stderr, "Lowest block access count         : %ld\n", lc);
  if(gpu_memory_hits + gpu_memory_misses) {
    fprintf(stderr, "GPU memory hits                   : %ld (%ld%%)\n", gpu_memory_hits, (100 * gpu_memory_hits) / (gpu_memory_hits + gpu_memory_misses));
    fprintf(stderr, "GPU memory misses                 : %ld (%ld%%)\n", gpu_memory_misses, (100 * gpu_memory_misses) / (gpu_memory_hits + gpu_memory_misses));
  }
  if(verbose) {
    fprintf(stderr, "Current block information:\n");
    n = 0;
    for(ptr = gpu_blocks_head; ptr; ptr = ptr->next, n++) {
      fprintf(stderr, "Block number: %ld\n", n);
      fprintf(stderr, "Block ID    : %s\n", ptr->id);
      fprintf(stderr, "Host mem    : %lx\n", (long unsigned int) ptr->host_mem);
      fprintf(stderr, "GPU blocks  : ");
      for(i = 0; i < use_ngpus; i++)
        fprintf(stderr, "%lx ", (long unsigned int) ptr->gpu_info->descriptor->data[i]);
      fprintf(stderr, "\n");
      total_size = 0;
      fprintf(stderr, "GPU sizes   : ");
      for(i = 0; i < use_ngpus; i++) {
        fprintf(stderr, "%lu ", ptr->gpu_info->descriptor->size[i]);
        total_size += ptr->gpu_info->descriptor->size[i];
      }
      fprintf(stderr, "\n");
      fprintf(stderr, "Block size  : %ld MB (%ld bytes)\n", total_size / (1024 * 1024), total_size);
      fprintf(stderr, "Created     : %s", ctime(&(ptr->created)));
      fprintf(stderr, "Last used   : %s", ctime(&(ptr->last_used)));
      fprintf(stderr, "Access cnt  : %ld\n", ptr->access_count);
      fprintf(stderr, "Lock status : %d\n", ptr->locked);
      fprintf(stderr, "================================\n");
    }
  }
}

/*
 * Return th GPU address pointer given the host memory pointer.
 *
 * host_mem = Host memory pointer (void *; input).
 *
 * Returns GPU address pointer or NULL if not found.
 *
 */

EXPORT gpu_mem_block *cuda_block_address(void *host_mem) {

  gpu_mem_block *ptr;

  if(!enable_cuda) return NULL;
#ifdef CUDA_DEBUG
  fprintf(stderr, "cuda: Request GPU address for host mem %lx.\n", (unsigned long int) host_mem);
#endif
  if(!(ptr = cuda_find_block(host_mem))) {
    fprintf(stderr, "libgrid(cuda): warning - host_mem %lx not found.\n", (long unsigned int) host_mem);
    return NULL;
  }
#ifdef CUDA_DEBUG
  fprintf(stderr, "cuda: Block found on GPU with host address %lx.\n", (unsigned long int) host_mem);
#endif
  return ptr;
}

/*
 * CUDA fft memory policy.
 *
 * host_mem     = Host memory block (void *; input).
 * length       = Host memory block length (size_t; input).
 * cufft_handle = CUFFT handle (-1 if not available; cufftHandle).
 * id           = String describing the block contents. Useful for debugging. (char *; input).
 *
 * Returns 0 if GPU operation can proceed
 * or -1 if the operation is to be carried out in host memory.
 *
 * The FFT policy is as follows:
 *
 * 1. Execute FFT on the GPU always (even when other blocks have to be swapped out).
 *
 */

EXPORT int cuda_fft_policy(void *host_mem, size_t length, cufftHandle cufft_handle, char *id) {

  gpu_mem_block *ptr;

  if(!enable_cuda) return -1;
  if(cufft_handle == -1) {
    fprintf(stderr, "libgrid(cuda): Attempt to run FFT without CUFFT handle.\n");
    abort();
  }
#ifdef CUDA_DEBUG
  fprintf(stderr, "cuda: FFT policy check for host mem %lx (%s).\n", (unsigned long int) host_mem, id);
#endif
  /* Always do FFT on GPU if possible */
  if(!(ptr = cuda_add_block(host_mem, length, cufft_handle, id, 1))) {
#ifdef CUDA_DEBUG
    fprintf(stderr, "cuda: Check result = In host memory.\n");
#endif
    return -1;
  }
#ifdef CUDA_DEBUG
  fprintf(stderr, "cuda: Check result = In GPU memory.\n");
#endif
  cuda_block_hit(ptr);
  return 0;
}

/*
 * CUDA one memory block policy.
 *
 * host_mem     = Host memory block (void *; input).
 * cufft_handle = CUFFT handle (cufft_handle).
 * length       = Host memory block length (size_t; input).
 * id           = String describing the block contents. Useful for debugging. (char *; input).
 * copy         = Copy contents of the block to GPU? 1 = copy, 0 = don't copy.
 *
 * Returns 0 if GPU operation can proceed
 * or -1 if the operation is to be carried out in host memory.
 *
 * The one block policy is as follows:
 *
 * 1. Execute the operation on the GPU if there is enough memory left.
 * 2. If the block is alreay on GPU run the operation there.
 * 3. If the block is not on GPU and there is not enough memory left, run on the host (CPU).
 *
 */

EXPORT int cuda_one_block_policy(void *host_mem, size_t length, cufftHandle cufft_handle, char *id, char copy) {

  gpu_mem_block *ptr;

  if(!enable_cuda) return -1;
#ifdef CUDA_DEBUG
  fprintf(stderr, "cuda: one block policy check for host mem %lx (%s).\n", (unsigned long int) host_mem, id);
#endif
  if(length < cuda_memory())
    return cuda_add_block(host_mem, length, cufft_handle, id, copy)?0:-1; /* but if there is enough mem free, just do it */
  /* If grid not already on GPU, use host memory */
  if(!(ptr = cuda_find_block(host_mem))) {
#ifdef CUDA_DEBUG
    fprintf(stderr, "cuda: Check result = In host memory.\n");
#endif
    return -1;
  }
#ifdef CUDA_DEBUG
  fprintf(stderr, "cuda: Check result = In GPU memory.\n");
#endif
  cuda_block_hit(ptr);  // add_block was not used, have to hit manually
  return 0;
}

/*
 * CUDA two memory block policy.
 *
 * host_mem1     = Host memory block 1 (void *; input).
 * length1       = Host memory block length 1 (size_t; input).
 * cufft_handle1 = CUFFT handle 1 (cufft_handle).
 * id1           = String describing block 1 contents. Useful for debugging. (char *; input).
 * copy1         = Copy contents of block 1 to GPU? 1 = copy, 0 = don't copy.
 * host_mem2     = Host memory block 2 (void *; input).
 * length2       = Host memory block length 2 (size_t; input).
 * cufft_handle2 = CUFFT handle 2 (cufft_handle).
 * id2           = String describing block 2 contents. Useful for debugging. (char *; input).
 * copy2         = Copy contents of block 2 to GPU? 1 = copy, 0 = don't copy.
 *
 * Returns 0 if GPU operation can proceed
 * or -1 if the operation is to be carried out in host memory.
 *
 * Note: When -1 is returned, all memory blocks will be in host memory.
 *
 * The two block policy is as follows:
 *
 * 1. Execute the operation on the GPU if there is enough memory available for both blocks.
 * 2. If one of the blocks is already on GPU run the operation there.
 * 3. If neither block is on GPU and there is not enough memory left, run on the host (CPU).
 *
 */

EXPORT int cuda_two_block_policy(void *host_mem1, size_t length1, cufftHandle cufft_handle1, char *id1, char copy1, void *host_mem2, size_t length2, cufftHandle cufft_handle2, char *id2, char copy2) {

  void *a, *b;

  if(!enable_cuda) return -1;
  if(host_mem1 == host_mem2) return cuda_one_block_policy(host_mem1, length1, cufft_handle1, id1, (char) (copy1 + copy2));
#ifdef CUDA_DEBUG
  fprintf(stderr, "cuda: two blocks policy check for host mem1 %lx (%s) and host mem2 %lx (%s).\n", (unsigned long int) host_mem1, id1, (unsigned long int) host_mem2, id2);
#endif
  if(length1 + length2 < cuda_memory())
    return cuda_add_two_blocks(host_mem1, length1, cufft_handle1, id1, copy1, host_mem2, length2, cufft_handle2, id2, copy2);

  a = cuda_find_block(host_mem1);
  b = cuda_find_block(host_mem2);

  /* If one of the grids is on GPU, use GPU. */
  if(!a && !b) {
#ifdef CUDA_DEBUG
    fprintf(stderr, "cuda: Check result = In host memory.\n");
#endif
    return -1;
  }

#ifdef CUDA_DEBUG
  fprintf(stderr, "cuda: Check result = In GPU memory.\n");
#endif
  return cuda_add_two_blocks(host_mem1, length1, cufft_handle1, id1, copy1, host_mem2, length2, cufft_handle2, id2, copy2);
}

/*
 * CUDA three memory block policy.
 *
 * host_mem1     = Host memory block 1 (void *; input).
 * length1       = Host memory block length 1 (size_t; input).
 * cufft_handle1 = CUFFT handle 1 (cufft_handle).
 * id1           = String describing block 1 contents. Useful for debugging. (char *; input).
 * copy1         = Copy contents of block 1 to GPU? 1 = copy, 0 = don't copy.
 * host_mem2     = Host memory block 2 (void *; input).
 * length2       = Host memory block length 2 (size_t; input).
 * cufft_handle2 = CUFFT handle 2 (cufft_handle).
 * id2           = String describing block 2 contents. Useful for debugging. (char *; input).
 * copy2         = Copy contents of block 2 to GPU? 1 = copy, 0 = don't copy.
 * host_mem3     = Host memory block 3 (void *; input).
 * length3       = Host memory block length 3 (size_t; input).
 * cufft_handle3 = CUFFT handle 3 (cufft_handle).
 * id3           = String describing block 3 contents. Useful for debugging. (char *; input).
 * copy3         = Copy contents of block 3 to GPU? 1 = copy, 0 = don't copy.
 *
 * Returns 0 if GPU operation can proceed
 * or -1 if the operation is to be carried out in host memory.
 *
 * Note: When -1 is returned, all memory blocks will reside in host memory.
 *
 * The three block policy is as follows:
 *
 * 1. Execute the operation on the GPU if there is enough memory left for all three blocks.
 * 2. If at least two of the blocks are already on GPU run the operation there.
 * 3. If none of the blocks are on GPU and there is not enough memory left, run on the host (CPU).
 *
 */

EXPORT int cuda_three_block_policy(void *host_mem1, size_t length1, cufftHandle cufft_handle1, char *id1, char copy1, void *host_mem2, size_t length2, cufftHandle cufft_handle2, char *id2, char copy2, void *host_mem3, size_t length3, cufftHandle cufft_handle3, char *id3, char copy3) {

  void *a, *b, *c;

  if(!enable_cuda) return -1;
  if(host_mem1 == host_mem2) return cuda_two_block_policy(host_mem1, length1, cufft_handle1, id1, (char) (copy1 + copy2), host_mem3, length3, cufft_handle3, id3, copy3);
  if(host_mem1 == host_mem3) return cuda_two_block_policy(host_mem1, length1, cufft_handle1, id1, (char) (copy1 + copy3), host_mem2, length2, cufft_handle2, id2, copy2);
  if(host_mem2 == host_mem3) return cuda_two_block_policy(host_mem1, length1, cufft_handle1, id1, copy1, host_mem2, length2, cufft_handle2, id2, (char) (copy2 + copy3));
#ifdef CUDA_DEBUG
  fprintf(stderr, "cuda: three blocks policy check for host mem1 %lx (%s), host mem2 %lx (%s), and host mem3 %lx (%s).\n", (unsigned long int) host_mem1, id1, (unsigned long int) host_mem2, id2, (unsigned long int) host_mem3, id3);
#endif
  if(length1 + length2 + length3 < cuda_memory())
    return cuda_add_three_blocks(host_mem1, length1, cufft_handle1, id1, copy1, host_mem2, length2, cufft_handle2, id2, copy2, host_mem3, length3, cufft_handle3, id3, copy3);

  a = cuda_find_block(host_mem1);
  b = cuda_find_block(host_mem2);
  c = cuda_find_block(host_mem3);

  /* At least two grids must be on GPU already before we use it */
  if((!a && !b) || (!a && !c) || (!b && !c)) {
#ifdef CUDA_DEBUG
    fprintf(stderr, "cuda: Check result = In host memory.\n");
#endif
    if(a) cuda_remove_block(host_mem1, copy1);
    if(b) cuda_remove_block(host_mem2, copy2);
    if(c) cuda_remove_block(host_mem3, copy3);
    return -1;
  }

#ifdef CUDA_DEBUG
  fprintf(stderr, "cuda: Check result = In GPU memory.\n");
#endif
  return cuda_add_three_blocks(host_mem1, length1, cufft_handle1, id1, copy1, host_mem2, length2, cufft_handle2, id2, copy2, host_mem3, length3, cufft_handle3, id3, copy3);
}

/*
 * CUDA four memory block policy.
 *
 * host_mem1     = Host memory block 1 (void *; input).
 * length1       = Host memory block length 1 (size_t; input).
 * cufft_handle1 = CUFFT handle 1 (cufft_handle).
 * id1           = String describing block 1 contents. Useful for debugging. (char *; input).
 * copy1         = Copy contents of block 1 to GPU? 1 = copy, 0 = don't copy.
 * host_mem2     = Host memory block 2 (void *; input).
 * length2       = Host memory block length 2 (size_t; input).
 * cufft_handle2 = CUFFT handle 2 (cufft_handle).
 * id2           = String describing block 2 contents. Useful for debugging. (char *; input).
 * copy2         = Copy contents of block 2 to GPU? 1 = copy, 0 = don't copy.
 * host_mem3     = Host memory block 3 (void *; input).
 * length3       = Host memory block length 3 (size_t; input).
 * cufft_handle3 = CUFFT handle 3 (cufft_handle).
 * id3           = String describing block 3 contents. Useful for debugging. (char *; input).
 * copy3         = Copy contents of block 3 to GPU? 1 = copy, 0 = don't copy.
 * host_mem4     = Host memory block 4 (void *; input).
 * length4       = Host memory block length 4 (size_t; input).
 * cufft_handle4 = CUFFT handle 4 (cufft_handle).
 * id4           = String describing block 4 contents. Useful for debugging. (char *; input).
 * copy4         = Copy contents of block 4 to GPU? 1 = copy, 0 = don't copy.
 *
 * Returns 0 if GPU operation can proceed
 * or -1 if the operation is to be carried out in host memory.
 *
 * Note: When -1 is returned, all block will be in host memory.
 *
 * The four block policy is as follows:
 *
 * 1. Execute the operation on the GPU if there is enough memory left for all four blocks.
 * 2. If at least two of the blocks are already on GPU run the operation there.
 * 3. If none of the blocks are on GPU and there is not enough memory left, run on the host (CPU).
 *
 */

EXPORT int cuda_four_block_policy(void *host_mem1, size_t length1, cufftHandle cufft_handle1, char *id1, char copy1, void *host_mem2, size_t length2, cufftHandle cufft_handle2, char *id2, char copy2, void *host_mem3, size_t length3, cufftHandle cufft_handle3, char *id3, char copy3, void *host_mem4, size_t length4, cufftHandle cufft_handle4, char *id4, char copy4) {

  void *a, *b, *c, *d;

  if(!enable_cuda) return -1;
  if(host_mem1 == host_mem2) return cuda_three_block_policy(host_mem1, length1, cufft_handle1, id1, (char) (copy1 + copy2), host_mem3, length3, cufft_handle3, id3, copy3, host_mem4, length4, cufft_handle4, id4, copy4);
  if(host_mem1 == host_mem3) return cuda_three_block_policy(host_mem1, length1, cufft_handle1, id1, (char) (copy1 + copy3), host_mem2, length2, cufft_handle2, id2, copy2, host_mem4, length4, cufft_handle4, id4, copy4);
  if(host_mem1 == host_mem4) return cuda_three_block_policy(host_mem1, length1, cufft_handle1, id1, (char) (copy1 + copy4), host_mem2, length2, cufft_handle2, id2, copy2, host_mem3, length3, cufft_handle3, id3, copy3);
  if(host_mem2 == host_mem3) return cuda_three_block_policy(host_mem1, length1, cufft_handle1, id1, copy1, host_mem2, length2, cufft_handle2, id2, (char) (copy2 + copy3), host_mem4, length4, cufft_handle4, id4, copy4);
  if(host_mem2 == host_mem4) return cuda_three_block_policy(host_mem1, length1, cufft_handle1, id1, copy1, host_mem2, length2, cufft_handle2, id2, (char) (copy2 + copy4), host_mem3, length3, cufft_handle3, id3, copy3);
  if(host_mem3 == host_mem4) return cuda_three_block_policy(host_mem1, length1, cufft_handle1, id1, copy1, host_mem2, length2, cufft_handle2, id2, copy2, host_mem3, length3, cufft_handle3, id3, (char) (copy3 + copy4));
#ifdef CUDA_DEBUG
  fprintf(stderr, "cuda: four blocks policy check for host mem1 %lx (%s), host mem2 %lx (%s), host mem3 %lx (%s), and host mem4 %lx (%s).\n", (unsigned long int) host_mem1, id1, (unsigned long int) host_mem2, id2, (unsigned long int) host_mem3, id3, (unsigned long int) host_mem4, id4);
#endif
  if(length1 + length2 + length3 + length4 < cuda_memory())
    return cuda_add_four_blocks(host_mem1, length1, cufft_handle1, id1, copy1, host_mem2, length2, cufft_handle2, id2, copy2, host_mem3, length3, cufft_handle3, id3, copy3, host_mem4, length4, cufft_handle4, id4, copy4);

  a = cuda_find_block(host_mem1);
  b = cuda_find_block(host_mem2);
  c = cuda_find_block(host_mem3);
  d = cuda_find_block(host_mem4);

  /* At least two grids must be on GPU already before we use it */
  if((!a && !b) || (!a && !c) || (!b && !c) || (!d && !a) || (!d && !b) || (!d && !c)) {
#ifdef CUDA_DEBUG
    fprintf(stderr, "cuda: Check result = In host memory.\n");
#endif
    if(a) cuda_remove_block(host_mem1, copy1);
    if(b) cuda_remove_block(host_mem2, copy2);
    if(c) cuda_remove_block(host_mem3, copy3);
    if(d) cuda_remove_block(host_mem4, copy4);
    return -1;
  }

#ifdef CUDA_DEBUG
  fprintf(stderr, "cuda: Check result = In GPU memory.\n");
#endif
  return cuda_add_four_blocks(host_mem1, length1, cufft_handle1, id1, copy1, host_mem2, length2, cufft_handle2, id2, copy2, host_mem3, length3, cufft_handle3, id3, copy3, host_mem4, length4, cufft_handle4, id4, copy4);
}

/*
 * CUDA memory copy policy. Copy host_mem2 to host_mem1.
 *
 * host_mem1     = Destination host memory block (void *; input).
 * length1       = Destination host memory block length (size_t; input).
 * cufft_handle1 = CUFFT handle 1 (cufft_handle).
 * id1           = String describing block 1 contents. Useful for debugging. (char *; input).
 * host_mem2     = Source host memory block (void *; input).
 * length2       = Source host memory block length (size_t; input).
 * cufft_handle2 = CUFFT handle 2 (cufft_handle).
 * id2           = String describing block 2 contents. Useful for debugging. (char *; input).
 *
 * Returns 0 if GPU operation can proceed
 * or -1 if the operation is to be carried out in host memory.
 *
 * Note: When -1 is returned, all blocks will be in host memory.
 *
 * The three block policy is as follows:
 *
 * 1. Execute the operation on the GPU if there is enough memory left for both blocks. The subsequent ops will be likely on GPU.
 * 2. If the source block is already on GPU run the operation there.
 * 3. If none of the blocks are on GPU and there is not enough memory left, run on the host (CPU).
 *
 */

EXPORT int cuda_copy_policy(void *host_mem1, size_t length1, cufftHandle cufft_handle1, char *id1, void *host_mem2, size_t length2, cufftHandle cufft_handle2, char *id2) {

  gpu_mem_block *ptr;
  char l;

  if(!enable_cuda) return -1;
  /* mem1 = dest, mem2 = source */
  if(host_mem1 == host_mem2) return 0;  // src and dest the same??
#ifdef CUDA_DEBUG
  fprintf(stderr, "cuda: copy policy check for host dst host_mem %lx (%s) and src host mem %lx (%s).\n", (unsigned long int) host_mem1, id1, (unsigned long int) host_mem2, id2);
#endif
  if(length1 + length2 < cuda_memory())
    return cuda_add_two_blocks(host_mem1, length1, cufft_handle1, id1, 0, host_mem2, length2, cufft_handle2, id2, 1);

  /* Proceed with GPU is the source is already on GPU */
  if(!(ptr = cuda_find_block(host_mem2))) {
#ifdef CUDA_DEBUG
    fprintf(stderr, "cuda: Check result = In host memory (host mem2).\n");
#endif
    cuda_remove_block(host_mem1, 0);  // would have been overwritten anyway
    return -1;
  }
  l = ptr->locked;
  ptr->locked = 1;
  if(cuda_add_block(host_mem1, length1, cufft_handle1, id1, 0) == NULL) {
#ifdef CUDA_DEBUG
    fprintf(stderr, "cuda: Check result = In host memory (host mem1).\n");
#endif
    ptr->locked = l;
    cuda_remove_block(host_mem2, 1);
    return -1;
  }
#ifdef CUDA_DEBUG
  fprintf(stderr, "cuda: Check result = In GPU memory.\n");
#endif
  ptr->locked = l;
  cuda_block_hit(ptr);
  return 0;
}

/*
 * Print GPU information.
 *
 * No return value.
 *
 */

EXPORT void cuda_gpu_info() {

  int ndev, i;
  struct cudaDeviceProp prop;

  if(cudaGetDeviceCount(&ndev) != cudaSuccess) {
    fprintf(stderr, "libgrid(cuda): Cannot get device count.\n");
    cuda_error_check();
    return;
  }
  fprintf(stderr, "Threads per block = %d.\n\n", CUDA_THREADS_PER_BLOCK);
  for(i = 0; i < ndev; i++) {
    fprintf(stderr, "**********************************************************************\n");
    cudaGetDeviceProperties(&prop, i);
    fprintf(stderr, "Device Number: %d\n", i);
    fprintf(stderr, "  Device name: %s\n", prop.name);
    fprintf(stderr, "  Major revision number: %d (%s)\n", prop.major, cuda_arch(prop.major));
    fprintf(stderr, "  Minor revision number: %d\n", prop.minor);
    fprintf(stderr, "  GPU Architecture string: sm_%1d%1d\n", prop.major, prop.minor);
    fprintf(stderr, "  Total global memory: %lu kB\n", prop.totalGlobalMem / 1024);
    fprintf(stderr, "  Memory Clock Rate (MHz): %d\n", prop.memoryClockRate / 1000);
    fprintf(stderr, "  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    fprintf(stderr, "  Peak Memory Bandwidth (GB/s): %f\n", 2.0 * ((float) prop.memoryClockRate) * ((float) (prop.memoryBusWidth / 8)) / 1.0e6);
    fprintf(stderr, "  Number of multiprocessors: %d\n", prop.multiProcessorCount);
    fprintf(stderr, "  Number of cores: %d\n", cuda_get_sp_cores(prop));
    fprintf(stderr, "  Total amount of shared memory per block: %lu kB\n", prop.sharedMemPerBlock / 1024);
    fprintf(stderr, "  Total registers per block: %d\n", prop.regsPerBlock);
    fprintf(stderr, "  Warp size: %d\n", prop.warpSize);
    fprintf(stderr, "  Maximum memory pitch: %lu kB\n", prop.memPitch / 1024);
    fprintf(stderr, "  Total amount of constant memory: %lu kB\n", prop.totalConstMem / 1024);
    fprintf(stderr, "**********************************************************************\n");
  }
}

/*
 * Allocate cufft plan.
 *
 * plan = Pointer to cufft plan (cufftHandle *; input/output).
 * type = cufft plan type (cufftType; input).
 * nx   = number of points along x (INT; input).
 * ny   = number of points along y (INT; input).
 * nz   = number of points along z (INT; input).
 *
 * No return value.
 *
 */

void grid_cufft_make_plan(cufftHandle *plan, cufftType type, INT nx, INT ny, INT nz) {

  size_t wrksize[MAX_GPU], maxsize = 0;
  void *gpumem[MAX_GPU];
  gpu_mem_block *block;
  int i, ngpus = cuda_ngpus(), *gpus = cuda_gpus();
  cufftResult status;

  cufftCreate(plan);
  if(*plan > grid_cufft_highest_plan) grid_cufft_highest_plan = *plan;
  cufftSetAutoAllocation(*plan, 0);

  cudaSetDevice(gpus[0]);
  if(ngpus > 1) {
    if((status = cufftXtSetGPUs(*plan, ngpus, gpus)) != CUFFT_SUCCESS) {
      fprintf(stderr, "libgrid(cuda): Error allocating GPUs in rcufft_workspace.\n");
      cufft_error_check(status);
    }    
  }

  /* Make cufft plan & get workspace sizes */
  if((status = cufftMakePlan3d(*plan, (int) nx, (int) ny, (int) nz, type, &wrksize[0])) != CUFFT_SUCCESS) {
    fprintf(stderr, "libgrid(cuda): Error in making real 3d cufft plan.\n");
    cufft_error_check(status);
    return;
  }

  /* Maximum amount of memory on GPU */
  for(i = 0; i < ngpus; i++)
    if(wrksize[i] > maxsize) maxsize = wrksize[i];

  /* Allocate the same amount on all GPUs (cuda.c restriction; blocks on each gpu must be the same length) */
  if(!(block = cuda_find_block(&grid_cufft_workarea))) {
    block = cuda_add_block(&grid_cufft_workarea, maxsize, -1, "cufft temp", 0);
    cuda_lock_block(&grid_cufft_workarea);
  } else if(maxsize > block->gpu_info->descriptor->size[0]) {
    cuda_unlock_block(&grid_cufft_workarea);
    cuda_remove_block(&grid_cufft_workarea, 0);
    block = cuda_add_block(&grid_cufft_workarea, maxsize, -1, "cufft temp", 0);
    cuda_lock_block(&grid_cufft_workarea);
  }

  /* Set up the work areas */
  for(i = 0; i < ngpus; i++)
    gpumem[i] = block->gpu_info->descriptor->data[i];

  if(ngpus == 1) {
    for(i = 0; i <= grid_cufft_highest_plan; i++)
      if((status = cufftSetWorkArea(i, gpumem[0])) != CUFFT_SUCCESS) {
        if(status != CUFFT_INVALID_PLAN) { // some plans may not be active...
          fprintf(stderr, "libgrid(cuda): CUFFT set workarea failed.\n");      
          cufft_error_check(status);
        }
      }
  } else {
    for(i = 0; i <= grid_cufft_highest_plan; i++)
      if((status = cufftXtSetWorkArea(i, &(gpumem[0]))) != CUFFT_SUCCESS) {
        if(status != CUFFT_INVALID_PLAN) { // some plans may not be active...
          fprintf(stderr, "libgrid(cuda): CUFFT set workarea failed.\n");      
          cufft_error_check(status);
        }
      }
  }
}
