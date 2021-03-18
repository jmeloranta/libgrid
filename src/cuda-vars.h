/*
 * CUDA variables macros.
 *
 */

/* The setup for separating between real and fourier space indexing (space: 0 = real, 1 = reciprocal) */
#define SETUP_VARIABLES(X) INT i, ngpu2 = X->gpu_info->descriptor->nGPUs, ngpu1, nnx1, nnx2, nny1, nny2;\
                        dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK); \
                        dim3 blocks1, blocks2; \
                        if(X->gpu_info->subFormat == CUFFT_XT_FORMAT_INPLACE) { \
                          ngpu1 = nx % ngpu2; \
                          nnx2 = nx / ngpu2; \
                          nnx1 = nnx2 + 1; \
                          nny1 = ny; \
                          nny2 = ny; \
                          blocks2.x = blocks1.x = (nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK; \
                          blocks2.y = blocks1.y = (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK; \
                          blocks1.z = (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK; \
                          blocks2.z = (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK; \
                        } else { \
                          ngpu1 = ny % ngpu2; \
                          nny2 = ny / ngpu2; \
                          nny1 = nny2 + 1; \
                          nnx1 = nx; \
                          nnx2 = nx; \
                          blocks2.x = blocks1.x = (nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK; \
                          blocks1.y = (nny1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK; \
                          blocks2.y = (nny2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK; \
                          blocks1.z = blocks2.z = (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK; \
                        }

/* The setup for separating between real and fourier space indexing (space: 0 = real, 1 = reciprocal) - 2D version */
#define SETUP_VARIABLES2(X) INT i, ngpu2 = X->gpu_info->descriptor->nGPUs, ngpu1, nnx1, nnx2, nny1, nny2;\
                        dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK); \
                        dim3 blocks1, blocks2; \
                        if(X->gpu_info->subFormat == CUFFT_XT_FORMAT_INPLACE) { \
                          ngpu1 = nx % ngpu2; \
                          nnx2 = nx / ngpu2; \
                          nnx1 = nnx2 + 1; \
                          nny1 = ny; \
                          nny2 = ny; \
                          blocks2.x = blocks1.x = (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK; \
                          blocks1.y = (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK; \
                          blocks2.y = (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK; \
                        } else { \
                          ngpu1 = ny % ngpu2; \
                          nny2 = ny / ngpu2; \
                          nny1 = nny2 + 1; \
                          nnx1 = nx; \
                          nnx2 = nx; \
                          blocks1.x = (nny1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK; \
                          blocks2.x = (nny2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK; \
                          blocks1.y = blocks2.y = (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK; \
                        }

/* Same as above but includes tracking of indices (for routines that need to access grid indices) */
/* dseg{x,y}{1,2} are the increments for segments such that semgent + index = absolute index */
#define SETUP_VARIABLES_SEG(X) INT i, ngpu2 = X->gpu_info->descriptor->nGPUs, ngpu1, nnx1, nnx2, nny1, nny2, dsegx1, dsegx2, dsegy1, dsegy2;\
                        dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK); \
                        dim3 blocks1, blocks2; \
                        if(X->gpu_info->subFormat == CUFFT_XT_FORMAT_INPLACE) { \
                          ngpu1 = nx % ngpu2; \
                          nnx2 = nx / ngpu2; \
                          nnx1 = nnx2 + 1; \
                          nny1 = ny; \
                          nny2 = ny; \
                          blocks2.x = blocks1.x = (nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK; \
                          blocks2.y = blocks1.y = (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK; \
                          blocks1.z = (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK; \
                          blocks2.z = (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK; \
                          dsegx1 = nnx1; \
                          dsegx2 = nnx2; \
                          dsegy1 = 0; \
                          dsegy2 = 0; \
                        } else { \
                          ngpu1 = ny % ngpu2; \
                          nny2 = ny / ngpu2; \
                          nny1 = nny2 + 1; \
                          nnx1 = nx; \
                          nnx2 = nx; \
                          blocks2.x = blocks1.x = (nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK; \
                          blocks1.y = (nny1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK; \
                          blocks2.y = (nny2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK; \
                          blocks1.z = blocks2.z = (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK; \
                          dsegx1 = 0; \
                          dsegx2 = 0; \
                          dsegy1 = nny1; \
                          dsegy2 = nny2; \
                        }
/* Setup variables for the real case (real space) */
#define SETUP_VARIABLES_REAL(X)  INT i, ngpu2 = X->gpu_info->descriptor->nGPUs, ngpu1 = nx % ngpu2, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1, nzz = 2 * (nz / 2 + 1); \
                                 dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK); \
                                 dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK, \
                                              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK, \
                                              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK); \
                                 dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK, \
                                              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK, \
                                              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

/* Setup variables for the real case (real space) - 2D */
#define SETUP_VARIABLES_REAL2(X)  INT i, ngpu2 = X->gpu_info->descriptor->nGPUs, ngpu1 = nx % ngpu2, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1, nzz = 2 * (nz / 2 + 1); \
                                 dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK); \
                                 dim3 blocks1((ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK, \
                                              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK); \
                                 dim3 blocks2((ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK, \
                                              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

/* Setup variables for the real case (reciprocal space) */
#define SETUP_VARIABLES_RECIPROCAL(X) INT i, ngpu2 = X->gpu_info->descriptor->nGPUs, ngpu1 = ny % ngpu2, nny2 = ny / ngpu2, nny1 = nny2 + 1, nzz = nz / 2 + 1; \
                                      dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK); \
                                      dim3 blocks1((nzz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK, \
                                                   (nny1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK, \
                                                   (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK); \
                                      dim3 blocks2((nzz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK, \
                                                   (nny2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK, \
                                                   (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

/* Setup variables for the real case (reciprocal space) - 2D */
#define SETUP_VARIABLES_RECIPROCAL2(X) INT i, ngpu2 = X->gpu_info->descriptor->nGPUs, ngpu1 = ny % ngpu2, nny2 = ny / ngpu2, nny1 = nny2 + 1, nzz = nz / 2 + 1; \
                                      dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK); \
                                      dim3 blocks1((nny1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK, \
                                                   (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK); \
                                      dim3 blocks2((nny2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK, \
                                                   (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

