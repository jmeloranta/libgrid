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
#ifdef CUDA_DEBUG
static char cuda_debug_flag = 0;
#endif

#include "cuda-private.h"

#define EXPORT

/*
 * Set GPU device to be used.
 *
 * dev = device number (int).
 *
 * No return value.
 *
 */

EXPORT void cuda_set_gpu(int dev) {

  cudaError_t err;

  if((err = cudaSetDevice(dev)) != cudaSuccess) {
    fprintf(stderr, "cuda: Error setting device %d.\n", dev);
    abort();
  }
}

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

  if((err = cudaGetLastError()) != cudaSuccess) {
    fprintf(stderr, "cuda: Error check: %s\n", cudaGetErrorString(err));
    abort();
  }
#ifdef CUDA_DEBUG
  if((err = cudaDeviceSynchronize()) != cudaSuccess) {
    fprintf(stderr, "cuda: Error check (sync): %s\n", cudaGetErrorString(err));
    abort();
  }
#endif
}

/*
 * Returns the amount of free current GPU memory (in bytes).
 * 
 */

EXPORT size_t cuda_memory() {

  size_t free, total;

  if(cudaMemGetInfo(&free, &total) != cudaSuccess) {
    fprintf(stderr, "cuda: Error getting memory info.\n");
    abort();
  }
  return free;
}

/*
 * Make at least given number of bytes available on the GPU by swapping out blocks.
 * Blocks that will be swapped out are synced.
 *
 * size = Requested size in bytes (size_t; input).
 *
 * Return value: 0 = OK, -1 = error.
 *
 */

EXPORT char cuda_freemem(size_t size) {

  size_t curr_size, dec_size;
  gpu_mem_block *ptr, *rptr = NULL;
  time_t current;
  long current_access;
  int cuda_remove_block(void *, char);

  for (curr_size = cuda_memory(); curr_size < size; curr_size -= dec_size) {
    current = time(0) + 1;
    current_access = 0;
    for(ptr = gpu_blocks_head, rptr = NULL; ptr; ptr = ptr->next) {
      if(!ptr->locked && (ptr->last_used < current || (ptr->last_used == current && ptr->access_count < current_access))) {
        current = ptr->last_used;
        current_access = ptr->access_count;
        rptr = ptr;
      }
    }
    if(!rptr) return -1;   // Nothing we can do to make more space...
    dec_size = rptr->length;
    cuda_remove_block(rptr->host_mem, 1);
  }
  return 0;
}

/*
 * Transfer data from host memory to GPU.
 *
 * block = Memory block to syncronize from host to GPU (gpu_mem_block *; input/output).
 * len   = Transfer length (0 = all) (size_t; input).
 *
 * Return value: 0 = OK, -1 = error.
 *
 */

EXPORT inline int cuda_mem2gpu(gpu_mem_block *block, size_t len) {

  if(len == 0) len = block->length;
#ifdef CUDA_DEBUG
  if(cuda_debug_flag) fprintf(stderr, "cuda: mem2gpu from host mem %lx to GPU mem %lx with length %ld (%s).\n", (long unsigned int) block->host_mem, 
    (long unsigned int) block->gpu_mem, len, block->id);
#endif
  if(cudaMemcpy(block->gpu_mem, block->host_mem, len, cudaMemcpyHostToDevice) != cudaSuccess) return -1;

  return 0;
}

/*
 * Transfer data from GPU to host memory.
 *
 * block = Memory block to syncronize from GPU to host (gpu_mem_block *; input/output).
 * len   = Transfer length (0 = full block length) (size_t; input).
 *
 * Return value: 0 = OK, -1 = error.
 *
 */

EXPORT inline int cuda_gpu2mem(gpu_mem_block *block, size_t len) {

  if(len == 0) len = block->length;
#ifdef CUDA_DEBUG
  if(cuda_debug_flag) fprintf(stderr, "cuda: gpu2mem from gpu mem %lx to host mem %lx with length %ld (%s).\n", (long unsigned int) block->gpu_mem, 
    (long unsigned int) block->host_mem, len, block->id);
#endif
  if(cudaMemcpy(block->host_mem, block->gpu_mem, len, cudaMemcpyDeviceToHost) != cudaSuccess) return -1;

  return 0;
}

/*
 * Transfer data from one area in GPU to another
 *
 * dst     = (destination) GPU buffer (gpu_mem_block *; output).
 * src     = (source) GPU buffer (gpu_mem_block *; input).
 * size    = # of bytes to transfer (INT; input). 0 = all in src block.
 *
 * Return value: 0 = OK, -1 = error.
 *
 */

EXPORT inline int cuda_gpu2gpu(gpu_mem_block *dst, gpu_mem_block *src, size_t len) {

  if(len == 0) len = src->length;
#ifdef CUDA_DEBUG
  if(cuda_debug_flag) fprintf(stderr, "cuda: gpu2gpu from gpu mem %lx (%s) to gpu mem %lx (%s) with length %ld.\n", (long unsigned int) src->gpu_mem, src->id,
    (long unsigned int) dst->gpu_mem, dst->id, len);
#endif
  if(cudaMemcpy(dst->gpu_mem, src->gpu_mem, len, cudaMemcpyDeviceToDevice) != cudaSuccess) return -1;

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
  if(cuda_debug_flag) fprintf(stderr, "cuda: find block by host mem %lx - ", (unsigned long int) host_mem);
#endif
  for(ptr = gpu_blocks_head; ptr; ptr = ptr->next)
    if(host_mem == ptr->host_mem) {
#ifdef CUDA_DEBUG
      if(cuda_debug_flag) fprintf(stderr, "found.\n");
#endif
      return ptr;
    }
#ifdef CUDA_DEBUG
  if(cuda_debug_flag) fprintf(stderr, "not found.\n");
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

#ifdef CUDA_DEBUG
  if(cuda_debug_flag) fprintf(stderr, "cuda: Remove block %lx with copy %d.\n", (unsigned long int) host_mem, copy);
#endif
  for(ptr = gpu_blocks_head, prev = NULL; ptr; prev = ptr, ptr = ptr->next)
    if(host_mem == ptr->host_mem) break;

  if(!ptr) {
#ifdef CUDA_DEBUG
    if(cuda_debug_flag) fprintf(stderr, "cuda: block not found.\n");
#endif
    return 0; /* not on GPU */
  }
  if(copy) {
#ifdef CUDA_DEBUG
    if(cuda_debug_flag) fprintf(stderr, "cuda: Syncing memory of the removed block from GPU to memory.\n");
#endif
    cuda_gpu2mem(ptr, 0);
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
  if(cuda_debug_flag) fprintf(stderr, "cuda: removed (%s).\n", ptr->id);
#endif
  total_alloc -= ptr->length;
  if(cudaFree(ptr->gpu_mem) != cudaSuccess) {
    fprintf(stderr, "libgrid(cuda): Failed to free memory.\n");
    abort();
  }
  if(prev) prev->next = ptr->next;
  else gpu_blocks_head = ptr->next;
  free(ptr);
  return 0;
}

/*
 * Add block (from host memory to GPU). If there is not enough space in GPU memory,
 * this may swap out another block(s) based on their last use stamp.
 *
 * host_mem = Host memory pointer containing the data (void *; input).
 * length   = Length of host memory data (size_t; input).
 * id       = String describing the block contents. Useful for debugging. (char *; input).
 * copy     = Copy host_mem to gpu_mem? (1 = yes, 0 = no).
 *
 * Return value: Pointer to new gpu_block_mem or NULL = error.
 *
 */

EXPORT gpu_mem_block *cuda_add_block(void *host_mem, size_t length, char *id, char copy) {

  gpu_mem_block *ptr, *rptr = NULL, *new;
  time_t current;
  long current_access;

  if(!enable_cuda) return NULL;
#ifdef CUDA_DEBUG
  if(cuda_debug_flag) fprintf(stderr, "cuda: Add block %lx (%s) of length %ld with copy %d.\n", (unsigned long int) host_mem, id, length, copy);
#endif
  if((ptr = cuda_find_block(host_mem))) { /* Already in GPU memory? */
    cuda_block_hit(ptr);
#ifdef CUDA_DEBUG
    if(cuda_debug_flag) fprintf(stderr, "cuda: Already in GPU memory.\n");
#endif
    return ptr;
  } else cuda_block_miss();

#ifdef CUDA_DEBUG
  if(cuda_debug_flag) fprintf(stderr, "cuda: Not in GPU memory - trying to add.\n");
#endif
  if(!(new = (gpu_mem_block *) malloc(sizeof(gpu_mem_block)))) {
    fprintf(stderr, "libgrid(cuda): Out of memory in allocating gpu_mem_block.\n");
    abort();
  }

  /* Data not in GPU - try to allocate & possibly swap out other blocks */

  while(cudaMalloc((void **) &(new->gpu_mem), length) != cudaSuccess) {
    (void) cudaGetLastError();  // clear cuda error so that it will not be caught by cuda_error_check()
    /* search for swap out candidate (later: also consider memory block sizes) */
    current = time(0)+1;
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
    if(cuda_debug_flag) fprintf(stderr, "cuda: Swap out block host mem: %lx and GPU mem: %lx (%s).\n", (unsigned long int) rptr->host_mem, (unsigned long int) rptr->gpu_mem, rptr->id);
#endif
    cuda_remove_block(rptr->host_mem, 1);
  }
#ifdef CUDA_DEBUG
  if(cuda_debug_flag) fprintf(stderr, "cuda: Succesfully added new block to GPU.\n");
#endif
  total_alloc += length;
  new->length = length;
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
    if(cuda_debug_flag) fprintf(stderr, "cuda: Syncing memory of the new block to GPU.\n");
#endif
    cuda_mem2gpu(new, 0);
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
  if(cuda_debug_flag) fprintf(stderr, "cuda: Lock block request for host mem %lx.\n", (unsigned long int) host_mem);
#endif
  if(!(ptr = cuda_find_block(host_mem))) {
#ifdef CUDA_DEBUG
    if(cuda_debug_flag) fprintf(stderr, "Block NOT FOUND.\n");
#endif
    return -1;
  }
#ifdef CUDA_DEBUG
  if(cuda_debug_flag) fprintf(stderr, "cuda: GPU mem %lx (%s) locked.\n", (unsigned long int) ptr->gpu_mem, ptr->id);
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
  if(cuda_debug_flag) fprintf(stderr, "cuda: Unlock block reqest for host mem %lx.\n", (unsigned long int) host_mem);
#endif
  if(!(ptr = cuda_find_block(host_mem))) {
#ifdef CUDA_DEBUG
    if(cuda_debug_flag) fprintf(stderr, "Block NOT FOUND.\n");
#endif
    return -1;
  }
#ifdef CUDA_DEBUG
  if(cuda_debug_flag) fprintf(stderr, "GPU mem %lx (%s) unlocked.\n", (unsigned long int) ptr->gpu_mem, ptr->id);
#endif
  ptr->locked = 0;
  return 0;
}

/*
 * Add two blocks to GPU simulatenously (or keep both at CPU). Other blocks may be swapped out
 * or the two blocks may not fit in at the same time.
 *
 * host_mem1 = Host memory pointer 1 (void *; input).
 * length1   = Length of host memory pointer 1 (size_t; input).
 * id1       = String describing block 1 contents. Useful for debugging. (char *; input).
 * copy1     = Copy contents of block 1 to GPU? 1 = copy, 0 = don't copy.
 * host_mem2 = Host memory pointer 2 (void *; input).
 * length2   = Length of host memory pointer 2 (size_t; input).
 * id2       = String describing block 2 contents. Useful for debugging. (char *; input).
 * copy2     = Copy contents of block 2 to GPU? 1 = copy, 0 = don't copy.
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

EXPORT char cuda_add_two_blocks(void *host_mem1, size_t length1, char *id1, char copy1, void *host_mem2, size_t length2, char *id2, char copy2) {

  gpu_mem_block *block1, *test;
  char l1;

  if(!enable_cuda) return -1;
  if(host_mem1 == host_mem2) return cuda_add_block(host_mem1, length1, id1, (char) (copy1 + copy2))?0:-1;

#ifdef CUDA_DEBUG
  if(cuda_debug_flag) fprintf(stderr, "cuda: Request two blocks with host mems %lx with len %ld (%s) and %lx with len %ld (%s)...\n", (unsigned long int) host_mem1, length1, id1, (unsigned long int) host_mem2, length2, id2);
#endif
  test = cuda_find_block(host_mem1);
  if(!(block1 = cuda_add_block(host_mem1, length1, id1, 0))) {
    /* both need to be in host memory */
#ifdef CUDA_DEBUG
    if(cuda_debug_flag) fprintf(stderr, "cuda: Failed to add host_mem1 to GPU, making sure host_mem2 is removed from GPU.\n");
#endif
    cuda_remove_block(host_mem2, copy2);  // remove 2 (may or may not be in GPU)
    return -1;
  }

  l1 = block1->locked;
  block1->locked = 1;
  if(!(cuda_add_block(host_mem2, length2, id2, copy2))) {
    block1->locked = l1;
    /* both need to be in host memory */
#ifdef CUDA_DEBUG
    if(cuda_debug_flag) fprintf(stderr, "cuda: Failed to add host_mem2 to GPU, removing host_mem from GPU1.\n");
#endif
    if(test) cuda_remove_block(host_mem1, copy1); // was already in GPU but now needs to be synced to host memory when removing
    else cuda_remove_block(host_mem1, 0);
    return -1;
  }
  block1->locked = l1;

  if(!test && copy1) {
#ifdef CUDA_DEBUG
    if(cuda_debug_flag) fprintf(stderr, "cuda: Syncing host_mem1 to GPU.\n");
#endif
    cuda_mem2gpu(block1, 0);  // Now that everything is OK, we need to also sync block1 to GPU (as it wasn't there already)
  }
  /* Both blocks now successfully in GPU memory */
#ifdef CUDA_DEBUG
  if(cuda_debug_flag) fprintf(stderr, "cuda: Successful two-block add.\n");
#endif
  return 0;  
}

/*
 * Add three blocks to GPU simulatenously. Other blocks may be swapped out
 * or the three blocks may not fit in at the same time.
 *
 * host_mem1 = Host memory pointer 1 (void *; input).
 * length1   = Length of host memory pointer 1 (size_t; input).
 * id1       = String describing block 1 contents. Useful for debugging. (char *; input).
 * copy1     = Copy contents of block 1 to GPU? 1 = copy, 0 = don't copy.
 * host_mem2 = Host memory pointer 2 (void *; input).
 * length2   = Length of host memory pointer 2 (size_t; input).
 * id2       = String describing block 2 contents. Useful for debugging. (char *; input).
 * copy2     = Copy contents of block 2 to GPU? 1 = copy, 0 = don't copy.
 * host_mem3 = Host memory pointer 3 (void *; input).
 * length3   = Length of host memory pointer 3 (size_t; input).
 * id3       = String describing block 3 contents. Useful for debugging. (char *; input).
 * copy3     = Copy contents of block 3 to GPU? 1 = copy, 0 = don't copy.
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

EXPORT char cuda_add_three_blocks(void *host_mem1, size_t length1, char *id1, char copy1, void *host_mem2, size_t length2, char *id2, char copy2, void *host_mem3, size_t length3, char *id3, char copy3) {

  gpu_mem_block *block1, *block2, *test1, *test2, *test3;
  char l1, l2;

  if(!enable_cuda) return -1;
  if(host_mem1 == host_mem2) return cuda_add_two_blocks(host_mem1, length1, id1, (char) (copy1 + copy2), host_mem3, length3, id3, copy3);
  if(host_mem1 == host_mem3) return cuda_add_two_blocks(host_mem1, length1, id1, (char) (copy1 + copy3), host_mem2, length2, id2, copy2);
  if(host_mem2 == host_mem3) return cuda_add_two_blocks(host_mem1, length1, id1, copy1, host_mem2, length2, id2, (char) (copy2 + copy3));
#ifdef CUDA_DEBUG
  if(cuda_debug_flag) fprintf(stderr, "cuda: Request three blocks with host mems %lx with len %ld (%s), %lx with len %ld (%s), and %lx with len %ld (%s)...\n", (unsigned long int) host_mem1, length1, id1, (unsigned long int) host_mem2, length2, id2, (unsigned long int) host_mem3, length3, id3);
#endif
  test1 = cuda_find_block(host_mem1);
  test2 = cuda_find_block(host_mem2);
  test3 = cuda_find_block(host_mem3);
  if(!(block1 = cuda_add_block(host_mem1, length1, id1, 0))) {
#ifdef CUDA_DEBUG
    if(cuda_debug_flag) fprintf(stderr, "cuda: Failed to add host_mem1 to GPU, making sure host_mem2 and host_mem3 are removed from GPU.\n");
#endif
    if(test2) cuda_remove_block(host_mem2, copy2);
    if(test3) cuda_remove_block(host_mem3, copy3);
    return -1;
  }
  l1 = block1->locked;
  block1->locked = 1;
  if(!(block2 = cuda_add_block(host_mem2, length2, id2, 0))) {
#ifdef CUDA_DEBUG
    if(cuda_debug_flag) fprintf(stderr, "cuda: Failed to add host_mem2 to GPU, making sure host_mem1 and host_mem3 are removed from GPU.\n");
#endif
    block1->locked = l1;
    if(test1) cuda_remove_block(host_mem1, copy1);   // was already in GPU, need to sync back to host memory
    else cuda_remove_block(host_mem1, 0);        // wasn't in GPU, no need to sync
    if(test3) cuda_remove_block(host_mem3, copy3);
    return -1;
  }
  l2 = block2->locked;
  block2->locked = 1;
  if(!(cuda_add_block(host_mem3, length3, id3, copy3))) {
#ifdef CUDA_DEBUG
    if(cuda_debug_flag) fprintf(stderr, "cuda: Failed to add host_mem3 to GPU, making sure host_mem1 and host_mem2 are removed from GPU.\n");
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
    if(cuda_debug_flag) fprintf(stderr, "cuda: Syncing host_mem1 to GPU.\n");
#endif
    cuda_mem2gpu(block1, 0);  // Now that everything is OK, we need to also sync block1
  }
  if(!test2 && copy2) {
#ifdef CUDA_DEBUG
    if(cuda_debug_flag) fprintf(stderr, "cuda: Syncing host_mem2 to GPU.\n");
#endif
    cuda_mem2gpu(block2, 0);  // Now that everything is OK, we need to also sync block2
  }
#ifdef CUDA_DEBUG
  if(cuda_debug_flag) fprintf(stderr, "cuda: Successful three-block add.\n");
#endif
  return 0;  
}

/*
 * Add four blocks to GPU simulatenously. Other blocks may be swapped out
 * or the four blocks may not fit in at the same time.
 *
 * host_mem1 = Host memory pointer 1 (void *; input).
 * length1   = Length of host memory pointer 1 (size_t; input).
 * id1       = String describing block 1 contents. Useful for debugging. (char *; input).
 * copy1     = Copy contents of block 1 to GPU? 1 = copy, 0 = don't copy.
 * host_mem2 = Host memory pointer 2 (void *; input).
 * length2   = Length of host memory pointer 2 (size_t; input).
 * id2       = String describing block 2 contents. Useful for debugging. (char *; input).
 * copy2     = Copy contents of block 2 to GPU? 1 = copy, 0 = don't copy.
 * host_mem3 = Host memory pointer 3 (void *; input).
 * length3   = Length of host memory pointer 3 (size_t; input).
 * id3       = String describing block 3 contents. Useful for debugging. (char *; input).
 * copy3     = Copy contents of block 3 to GPU? 1 = copy, 0 = don't copy.
 * host_mem4 = Host memory pointer 4 (void *; input).
 * length4   = Length of host memory pointer 4 (size_t; input).
 * id4       = String describing block 4 contents. Useful for debugging. (char *; input).
 * copy4     = Copy contents of block 4 to GPU? 1 = copy, 0 = don't copy.
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

EXPORT char cuda_add_four_blocks(void *host_mem1, size_t length1, char *id1, char copy1, void *host_mem2, size_t length2, char *id2, char copy2, void *host_mem3, size_t length3, char *id3, char copy3, void *host_mem4, size_t length4, char *id4, char copy4) {

  gpu_mem_block *block1, *block2, *block3, *test1, *test2, *test3, *test4;
  char l1, l2, l3;

  if(!enable_cuda) return -1;
  if(host_mem1 == host_mem2) return cuda_add_three_blocks(host_mem1, length1, id1, (char) (copy1 + copy2), host_mem3, length3, id3, copy3, host_mem4, length4, id4, copy4);
  if(host_mem1 == host_mem3) return cuda_add_three_blocks(host_mem1, length1, id1, (char) (copy1 + copy3), host_mem2, length2, id2, copy2, host_mem4, length4, id4, copy4);
  if(host_mem1 == host_mem4) return cuda_add_three_blocks(host_mem1, length1, id1, (char) (copy1 + copy4), host_mem2, length2, id2, copy2, host_mem3, length3, id3, copy3);
  if(host_mem2 == host_mem3) return cuda_add_three_blocks(host_mem1, length1, id1, copy1, host_mem2, length2, id2, (char) (copy2 + copy3), host_mem4, length4, id4, copy4);
  if(host_mem2 == host_mem4) return cuda_add_three_blocks(host_mem1, length1, id1, copy1, host_mem2, length2, id2, (char) (copy2 + copy4), host_mem3, length3, id3, copy3);
  if(host_mem3 == host_mem4) return cuda_add_three_blocks(host_mem1, length1, id1, copy1, host_mem2, length2, id2, copy2, host_mem3, length3, id3, (char) (copy3 + copy4));
#ifdef CUDA_DEBUG
  if(cuda_debug_flag) fprintf(stderr, "cuda: Request four blocks with host mems %lx with len %ld (%s), %lx with len %ld (%s), %lx with len %ld (%s), and %lx with len %ld (%s)...\n", (unsigned long int) host_mem1, length1, id1, (unsigned long int) host_mem2, length2, id2, (unsigned long int) host_mem3, length3, id3, (unsigned long int) host_mem4, length4, id4);
#endif
  test1 = cuda_find_block(host_mem1);
  test2 = cuda_find_block(host_mem2);
  test3 = cuda_find_block(host_mem3);
  test4 = cuda_find_block(host_mem4);
  if(!(block1 = cuda_add_block(host_mem1, length1, id1, 0))) {
#ifdef CUDA_DEBUG
    if(cuda_debug_flag) fprintf(stderr, "cuda: Failed to add host_mem1 to GPU, making sure host_mem2, host_mem3, and host_mem4 are removed from GPU.\n");
#endif
    if(test2) cuda_remove_block(host_mem2, copy2);
    if(test3) cuda_remove_block(host_mem3, copy3);
    if(test4) cuda_remove_block(host_mem4, copy4);
    return -1;
  }
  l1 = block1->locked;
  block1->locked = 1;
  if(!(block2 = cuda_add_block(host_mem2, length2, id2, 0))) {
#ifdef CUDA_DEBUG
    if(cuda_debug_flag) fprintf(stderr, "cuda: Failed to add host_mem2 to GPU, making sure host_mem1, host_mem3, and host_mem4 are removed from GPU.\n");
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
  if(!(block3 = cuda_add_block(host_mem3, length3, id3, 0))) {
#ifdef CUDA_DEBUG
    if(cuda_debug_flag) fprintf(stderr, "cuda: Failed to add host_mem3 to GPU, making sure host_mem1, host_mem2, and host_mem4 are removed from GPU.\n");
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
  if(!(cuda_add_block(host_mem4, length4, id4, copy4))) {
#ifdef CUDA_DEBUG
    if(cuda_debug_flag) fprintf(stderr, "cuda: Failed to add host_mem4 to GPU, making sure host_mem1, host_mem2, and host_mem3 are removed from GPU.\n");
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
    if(cuda_debug_flag) fprintf(stderr, "cuda: Syncing host_mem1 to GPU.\n");
#endif
    cuda_mem2gpu(block1, 0);  // Now that everything is OK, we need to also sync block1
  }
  if(!test2 && copy2) {
#ifdef CUDA_DEBUG
    if(cuda_debug_flag) fprintf(stderr, "cuda: Syncing host_mem2 to GPU.\n");
#endif
    cuda_mem2gpu(block2, 0);  // Now that everything is OK, we need to also sync block2
  }
  if(!test3 && copy3) {
#ifdef CUDA_DEBUG
    if(cuda_debug_flag) fprintf(stderr, "cuda: Syncing host_mem3 to GPU.\n");
#endif
    cuda_mem2gpu(block3, 0);  // Now that everything is OK, we need to also sync block2
  }
  return 0;
}

/*
 * Fetch one element from a GPU/CPU array.
 * If the data is not on GPU, it will be retrieved from host memory instead.
 *
 * host_mem = Host memory (void *; input).
 * index    = Index for the host memory array (size_t; input).
 * size     = Size of each element in bytes for indexing (size_t; input).
 * value    = Where the value will be stored (void *; output).
 *
 * Returns -1 for error, 0 = OK.
 *
 * Avoid calling this repeatedly - VERY SLOW!
 *
 */

EXPORT int cuda_get_element(void *host_mem, size_t index, size_t size, void *value) {

  gpu_mem_block *ptr;

#ifdef CUDA_DEBUG
  if(cuda_debug_flag) fprintf(stderr, "cuda: Request to fetch element from host mem %lx of size %ld.\n", (unsigned long int) host_mem, size);
#endif
  if(!enable_cuda || !(ptr = cuda_find_block(host_mem))) {
#ifdef CUDA_DEBUG
    if(cuda_debug_flag) fprintf(stderr, "cuda: found in host memory.\n");
#endif
    memcpy(value, &((char *) host_mem)[index * size], size);
    return 0;
  }
#ifdef CUDA_DEBUG
  if(cuda_debug_flag) fprintf(stderr, "cuda: found in GPU memory.\n");
#endif
  if(cudaMemcpy(value, &((char *) ptr->gpu_mem)[index * size], size, cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
  return 0;
}

/*
 * Set value for one element on a GPU/CPU array.
 * If the data is not on GPU, it will be set in host memory instead.
 *
 * host_mem = Host memory (void *; output).
 * index    = Index for the host memory array (size_t; input).
 * size     = Size of each element in bytes for indexing (size_t; input).
 * value    = The value that will be stored (void *; input).
 *
 * Returns -1 for error, 0 = OK.
 *
 * Avoid calling this repeatedly - VERY SLOW!
 *
 */

EXPORT int cuda_set_element(void *host_mem, size_t index, size_t size, void *value) {

  gpu_mem_block *ptr;

#ifdef CUDA_DEBUG
  if(cuda_debug_flag) fprintf(stderr, "cuda: Request to set element to host mem %lx of size %ld.\n", (unsigned long int) host_mem, size);
#endif
  if(!enable_cuda || !(ptr = cuda_find_block(host_mem))) {
#ifdef CUDA_DEBUG
    if(cuda_debug_flag) fprintf(stderr, "cuda: found in host memory.\n");
#endif
    memcpy(&((char *) host_mem)[index * size], value, size);  // In host memory
    return 0;
  }
#ifdef CUDA_DEBUG
  if(cuda_debug_flag) fprintf(stderr, "cuda: found in GPU memory.\n");
#endif
  if(cudaMemcpy(&((char *) ptr->gpu_mem)[index * size], value, size, cudaMemcpyHostToDevice) != cudaSuccess) return -1;
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
  if(cuda_debug_flag) fprintf(stderr, "cuda: Releasing all GPU memory blocks.\n");
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
 * Set val = 0 to disable CUDA or val = 1 to enable CUDA.
 *
 * Disabling active CUDA will flush GPU memory pages to the host memory.
 *
 */

EXPORT void cuda_enable(char val) {

  void cuda_gpu_info();

#ifdef CUDA_DEBUG
  if(cuda_debug_flag) fprintf(stderr, "cuda: Enable/disable %d.\n", val);
#endif
  if(enable_cuda && val == 0) cuda_free_all_blocks(1);  // disable CUDA; flush GPU memory blocks
  enable_cuda = val;
  if(val) {
    fprintf(stderr, "libgrid(cuda): CUDA enabled.\n");
    cuda_gpu_info();
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
 * Enable/disable CUDA debugging.
 *
 * val: 1 = debug, 0 = no debug (char; input).
 *
 * No return value.
 *
 */

EXPORT void cuda_debug(char val) {

#ifdef CUDA_DEBUG
  cuda_debug_flag = val;
  if(val) fprintf(stderr, "cuda: Debugging enabled.\n");
  else fprintf(stderr, "cuda: Debugging disabled.\n");
#else
  fprintf(stderr, "cuda: Debug code not complied in.\n");
#endif
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

  if(!enable_cuda) {
    fprintf(stderr, "CUDA not enabled.\n");
    return;
  }
  for(ptr = gpu_blocks_head; ptr; ptr = ptr->next, n++) {
    if(ptr->locked) nl++;
    total_size += ptr->length;
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
      fprintf(stderr, "Block mem   : %lx\n", (long unsigned int) ptr->host_mem);
      fprintf(stderr, "Block gpu   : %lx\n", (long unsigned int) ptr->gpu_mem);
      fprintf(stderr, "Block size  : %ld MB\n", ptr->length / (1024 * 1024));
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

EXPORT void *cuda_block_address(void *host_mem) {

  gpu_mem_block *ptr;

  if(!enable_cuda) return NULL;
#ifdef CUDA_DEBUG
  if(cuda_debug_flag) fprintf(stderr, "cuda: Request GPU address for host mem %lx.\n", (unsigned long int) host_mem);
#endif
  if(!(ptr = cuda_find_block(host_mem))) {
    fprintf(stderr, "libgrid(cuda): warning - host_mem %lx not found.\n", (long unsigned int) host_mem);
    return NULL;
  }
#ifdef CUDA_DEBUG
  if(cuda_debug_flag) fprintf(stderr, "cuda: Block found with GPU address %lx.\n", (unsigned long int) ptr->gpu_mem);
#endif
  return ptr->gpu_mem;
}

/*
 * CUDA fft memory policy.
 *
 * host_mem = Host memory block (void *; input).
 * length   = Host memory block length (size_t; input).
 * id       = String describing the block contents. Useful for debugging. (char *; input).
 *
 * Returns 0 if GPU operation can proceed
 * or -1 if the operation is to be carried out in host memory.
 *
 * The FFT policy is as follows:
 *
 * 1. Execute FFT on the GPU always (even when other blocks have to be swapped out).
 *
 */

EXPORT int cuda_fft_policy(void *host_mem, size_t length, char *id) {

  gpu_mem_block *ptr;

  if(!enable_cuda) return -1;
#ifdef CUDA_DEBUG
  if(cuda_debug_flag) fprintf(stderr, "cuda: FFT policy check for host mem %lx.\n", (unsigned long int) host_mem);
#endif
  /* Always do FFT on GPU if possible */
  if(!(ptr = cuda_add_block(host_mem, length, id, 1))) {
#ifdef CUDA_DEBUG
    if(cuda_debug_flag) fprintf(stderr, "cuda: Check result = In host memory.\n");
#endif
    return -1;
  }
#ifdef CUDA_DEBUG
  if(cuda_debug_flag) fprintf(stderr, "cuda: Check result = In GPU memory.\n");
#endif
  cuda_block_hit(ptr);
  return 0;
}

/*
 * CUDA one memory block policy.
 *
 * host_mem = Host memory block (void *; input).
 * length   = Host memory block length (size_t; input).
 * id       = String describing the block contents. Useful for debugging. (char *; input).
 * copy     = Copy contents of the block to GPU? 1 = copy, 0 = don't copy.
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

EXPORT int cuda_one_block_policy(void *host_mem, size_t length, char *id, char copy) {

  gpu_mem_block *ptr;

  if(!enable_cuda) return -1;
#ifdef CUDA_DEBUG
  if(cuda_debug_flag) fprintf(stderr, "cuda: one block policy check for host mem %lx.\n", (unsigned long int) host_mem);
#endif
  if(length < cuda_memory())
    return cuda_add_block(host_mem, length, id, copy)?0:-1; /* but if there is enough mem free, just do it */
  /* If grid not already on GPU, use host memory */
  if(!(ptr = cuda_find_block(host_mem))) {
#ifdef CUDA_DEBUG
    if(cuda_debug_flag) fprintf(stderr, "cuda: Check result = In host memory.\n");
#endif
    return -1;
  }
#ifdef CUDA_DEBUG
  if(cuda_debug_flag) fprintf(stderr, "cuda: Check result = In GPU memory.\n");
#endif
  cuda_block_hit(ptr);  // add_block was not used, have to hit manually
  return 0;
}

/*
 * CUDA two memory block policy.
 *
 * host_mem1 = Host memory block 1 (void *; input).
 * length1   = Host memory block length 1 (size_t; input).
 * id1       = String describing block 1 contents. Useful for debugging. (char *; input).
 * copy1     = Copy contents of block 1 to GPU? 1 = copy, 0 = don't copy.
 * host_mem2 = Host memory block 2 (void *; input).
 * length2   = Host memory block length 2 (size_t; input).
 * id2       = String describing block 2 contents. Useful for debugging. (char *; input).
 * copy2     = Copy contents of block 2 to GPU? 1 = copy, 0 = don't copy.
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

EXPORT int cuda_two_block_policy(void *host_mem1, size_t length1, char *id1, char copy1, void *host_mem2, size_t length2, char *id2, char copy2) {

  void *a, *b;

  if(!enable_cuda) return -1;
  if(host_mem1 == host_mem2) return cuda_one_block_policy(host_mem1, length1, id1, (char) (copy1 + copy2));
#ifdef CUDA_DEBUG
  if(cuda_debug_flag) fprintf(stderr, "cuda: two blocks policy check for host mem1 %lx and host mem2 %lx.\n", (unsigned long int) host_mem1, (unsigned long int) host_mem2);
#endif
  if(length1 + length2 < cuda_memory())
    return cuda_add_two_blocks(host_mem1, length1, id1, copy1, host_mem2, length2, id2, copy2);

  a = cuda_find_block(host_mem1);
  b = cuda_find_block(host_mem2);

  /* If one of the grids is on GPU, use GPU. */
  if(!a && !b) {
#ifdef CUDA_DEBUG
    if(cuda_debug_flag) fprintf(stderr, "cuda: Check result = In host memory.\n");
#endif
    return -1;
  }

#ifdef CUDA_DEBUG
  if(cuda_debug_flag) fprintf(stderr, "cuda: Check result = In GPU memory.\n");
#endif
  return cuda_add_two_blocks(host_mem1, length1, id1, copy1, host_mem2, length2, id2, copy2);
}

/*
 * CUDA three memory block policy.
 *
 * host_mem1 = Host memory block 1 (void *; input).
 * length1   = Host memory block length 1 (size_t; input).
 * id1       = String describing block 1 contents. Useful for debugging. (char *; input).
 * copy1     = Copy contents of block 1 to GPU? 1 = copy, 0 = don't copy.
 * host_mem2 = Host memory block 2 (void *; input).
 * length2   = Host memory block length 2 (size_t; input).
 * id2       = String describing block 2 contents. Useful for debugging. (char *; input).
 * copy2     = Copy contents of block 2 to GPU? 1 = copy, 0 = don't copy.
 * host_mem3 = Host memory block 3 (void *; input).
 * length3   = Host memory block length 3 (size_t; input).
 * id3       = String describing block 3 contents. Useful for debugging. (char *; input).
 * copy3     = Copy contents of block 3 to GPU? 1 = copy, 0 = don't copy.
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

EXPORT int cuda_three_block_policy(void *host_mem1, size_t length1, char *id1, char copy1, void *host_mem2, size_t length2, char *id2, char copy2, void *host_mem3, size_t length3, char *id3, char copy3) {

  void *a, *b, *c;

  if(!enable_cuda) return -1;
  if(host_mem1 == host_mem2) return cuda_two_block_policy(host_mem1, length1, id1, (char) (copy1 + copy2), host_mem3, length3, id3, copy3);
  if(host_mem1 == host_mem3) return cuda_two_block_policy(host_mem1, length1, id1, (char) (copy1 + copy3), host_mem2, length2, id2, copy2);
  if(host_mem2 == host_mem3) return cuda_two_block_policy(host_mem1, length1, id1, copy1, host_mem2, length2, id2, (char) (copy2 + copy3));
#ifdef CUDA_DEBUG
  if(cuda_debug_flag) fprintf(stderr, "cuda: three blocks policy check for host mem1 %lx, host mem2 %lx, and host mem3 %lx.\n", (unsigned long int) host_mem1, (unsigned long int) host_mem2, (unsigned long int) host_mem3);
#endif
  if(length1 + length2 + length3 < cuda_memory())
    return cuda_add_three_blocks(host_mem1, length1, id1, copy1, host_mem2, length2, id2, copy2, host_mem3, length3, id3, copy3);

  a = cuda_find_block(host_mem1);
  b = cuda_find_block(host_mem2);
  c = cuda_find_block(host_mem3);

  /* At least two grids must be on GPU already before we use it */
  if((!a && !b) || (!a && !c) || (!b && !c)) {
#ifdef CUDA_DEBUG
    if(cuda_debug_flag) fprintf(stderr, "cuda: Check result = In host memory.\n");
#endif
    if(a) cuda_remove_block(host_mem1, copy1);
    if(b) cuda_remove_block(host_mem2, copy2);
    if(c) cuda_remove_block(host_mem3, copy3);
    return -1;
  }

#ifdef CUDA_DEBUG
  if(cuda_debug_flag) fprintf(stderr, "cuda: Check result = In GPU memory.\n");
#endif
  return cuda_add_three_blocks(host_mem1, length1, id1, copy1, host_mem2, length2, id2, copy2, host_mem3, length3, id3, copy3);
}

/*
 * CUDA four memory block policy.
 *
 * host_mem1 = Host memory block 1 (void *; input).
 * length1   = Host memory block length 1 (size_t; input).
 * id1       = String describing block 1 contents. Useful for debugging. (char *; input).
 * copy1     = Copy contents of block 1 to GPU? 1 = copy, 0 = don't copy.
 * host_mem2 = Host memory block 2 (void *; input).
 * length2   = Host memory block length 2 (size_t; input).
 * id2       = String describing block 2 contents. Useful for debugging. (char *; input).
 * copy2     = Copy contents of block 2 to GPU? 1 = copy, 0 = don't copy.
 * host_mem3 = Host memory block 3 (void *; input).
 * length3   = Host memory block length 3 (size_t; input).
 * id3       = String describing block 3 contents. Useful for debugging. (char *; input).
 * copy3     = Copy contents of block 3 to GPU? 1 = copy, 0 = don't copy.
 * host_mem4 = Host memory block 4 (void *; input).
 * length4   = Host memory block length 4 (size_t; input).
 * id4       = String describing block 4 contents. Useful for debugging. (char *; input).
 * copy4     = Copy contents of block 4 to GPU? 1 = copy, 0 = don't copy.
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

EXPORT int cuda_four_block_policy(void *host_mem1, size_t length1, char *id1, char copy1, void *host_mem2, size_t length2, char *id2, char copy2, void *host_mem3, size_t length3, char *id3, char copy3, void *host_mem4, size_t length4, char *id4, char copy4) {

  void *a, *b, *c, *d;

  if(!enable_cuda) return -1;
  if(host_mem1 == host_mem2) return cuda_three_block_policy(host_mem1, length1, id1, (char) (copy1 + copy2), host_mem3, length3, id3, copy3, host_mem4, length4, id4, copy4);
  if(host_mem1 == host_mem3) return cuda_three_block_policy(host_mem1, length1, id1, (char) (copy1 + copy3), host_mem2, length2, id2, copy2, host_mem4, length4, id4, copy4);
  if(host_mem1 == host_mem4) return cuda_three_block_policy(host_mem1, length1, id1, (char) (copy1 + copy4), host_mem2, length2, id2, copy2, host_mem3, length4, id3, copy3);
  if(host_mem2 == host_mem3) return cuda_three_block_policy(host_mem1, length1, id1, copy1, host_mem2, length2, id2, (char) (copy2 + copy3), host_mem4, length4, id4, copy4);
  if(host_mem2 == host_mem4) return cuda_three_block_policy(host_mem1, length1, id1, copy1, host_mem2, length2, id2, (char) (copy2 + copy4), host_mem3, length4, id3, copy3);
  if(host_mem3 == host_mem4) return cuda_three_block_policy(host_mem1, length1, id1, copy1, host_mem2, length2, id2, copy2, host_mem3, length3, id3, (char) (copy3 + copy4));
#ifdef CUDA_DEBUG
  if(cuda_debug_flag) fprintf(stderr, "cuda: four blocks policy check for host mem1 %lx, host mem2 %lx, host mem3 %lx, and host mem4 %lx.\n", (unsigned long int) host_mem1, (unsigned long int) host_mem2, (unsigned long int) host_mem3, (unsigned long int) host_mem4);
#endif
  if(length1 + length2 + length3 + length4 < cuda_memory())
    return cuda_add_four_blocks(host_mem1, length1, id1, copy1, host_mem2, length2, id2, copy2, host_mem3, length3, id3, copy3, host_mem4, length4, id4, copy4);

  a = cuda_find_block(host_mem1);
  b = cuda_find_block(host_mem2);
  c = cuda_find_block(host_mem3);
  d = cuda_find_block(host_mem4);

  /* At least two grids must be on GPU already before we use it */
  if((!a && !b) || (!a && !c) || (!b && !c) || (!d && !a) || (!d && !b) || (!d && !c)) {
#ifdef CUDA_DEBUG
    if(cuda_debug_flag) fprintf(stderr, "cuda: Check result = In host memory.\n");
#endif
    if(a) cuda_remove_block(host_mem1, copy1);
    if(b) cuda_remove_block(host_mem2, copy2);
    if(c) cuda_remove_block(host_mem3, copy3);
    if(d) cuda_remove_block(host_mem4, copy4);
    return -1;
  }

#ifdef CUDA_DEBUG
  if(cuda_debug_flag) fprintf(stderr, "cuda: Check result = In GPU memory.\n");
#endif
  return cuda_add_four_blocks(host_mem1, length1, id1, copy1, host_mem2, length2, id2, copy2, host_mem3, length3, id3, copy3, host_mem4, length4, id4, copy4);
}

/*
 * CUDA memory copy policy.
 *
 * host_mem1 = Destination host memory block (void *; input).
 * length1   = Destination host memory block length (size_t; input).
 * id1       = String describing block 1 contents. Useful for debugging. (char *; input).
 * host_mem2 = Source host memory block (void *; input).
 * length2   = Source host memory block length (size_t; input).
 * id2       = String describing block 2 contents. Useful for debugging. (char *; input).
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

EXPORT int cuda_copy_policy(void *host_mem1, size_t length1, char *id1, void *host_mem2, size_t length2, char *id2) {

  gpu_mem_block *ptr;
  char l;

  if(!enable_cuda) return -1;
  /* mem1 = dest, mem2 = source */
  if(host_mem1 == host_mem2) return 0;  // src and dest the same??
#ifdef CUDA_DEBUG
  if(cuda_debug_flag) fprintf(stderr, "cuda: copy policy check for host dst host_mem %lx and src host mem %lx.\n", (unsigned long int) host_mem1, (unsigned long int) host_mem2);
#endif
  if(length1 + length2 < cuda_memory())
    return cuda_add_two_blocks(host_mem1, length1, id1, 0, host_mem2, length2, id2, 1);

  /* Proceed with GPU is the source is already on GPU */
  if(!(ptr = cuda_find_block(host_mem2))) {
#ifdef CUDA_DEBUG
    if(cuda_debug_flag) fprintf(stderr, "cuda: Check result = In host memory (host mem2).\n");
#endif
    cuda_remove_block(host_mem1, 0);  // would have been overwritten anyway
    return -1;
  }
  l = ptr->locked;
  ptr->locked = 1;
  if(cuda_add_block(host_mem1, length1, id1, 0) == NULL) {
#ifdef CUDA_DEBUG
    if(cuda_debug_flag) fprintf(stderr, "cuda: Check result = In host memory (host mem1).\n");
#endif
    ptr->locked = l;
    cuda_remove_block(host_mem2, 1);
    return -1;
  }
#ifdef CUDA_DEBUG
  if(cuda_debug_flag) fprintf(stderr, "cuda: Check result = In GPU memory.\n");
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
    abort();
  }
  for(i = 0; i < ndev; i++) {
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
  }
  fprintf(stderr, "**********************************************************************\n");
}
