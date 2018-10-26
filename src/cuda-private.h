/*
 * CUDA memory management private functions.
 *
 */

static inline void cuda_block_hit(gpu_mem_block *ptr) {

  ptr->last_used = time(0);
  (ptr->access_count)++;
  gpu_memory_hits++;
}

static inline void cuda_block_miss() {

  gpu_memory_misses++;
}

static char *cuda_arch(int major) {

  static char arch[32];

  switch(major) {
    case 2:
      strcpy(arch, "Fermi");
    break;
    case 3:
      strcpy(arch, "Kepler");
    break;
    case 4:
      strcpy(arch, "???");
    break;
    case 5:
      strcpy(arch, "Maxwell");
    break;
    case 6:
      strcpy(arch, "Pascal");
    break;
    case 7:
      strcpy(arch, "Volta");
    break;
  }
  return arch;
}

static int cuda_get_sp_cores(struct cudaDeviceProp prop) {  

  int cores = 0;
  int mp = prop.multiProcessorCount;

  switch (prop.major) {
    case 2: // Fermi
      if (prop.minor == 1) cores = mp * 48;
      else cores = mp * 32;
    break;
    case 3: // Kepler
      cores = mp * 192;
    break;
    case 5: // Maxwell
      cores = mp * 128;
    break;
    case 6: // Pascal
      if (prop.minor == 1) cores = mp * 128;
      else if (prop.minor == 0) cores = mp * 64;
      else fprintf(stderr, "Unknown device type\n");
    break;
    case 7: // Volta
      if (prop.minor == 0) cores = mp * 64;
      else fprintf(stderr, "Unknown device type\n");
    break;
    default:
      fprintf(stderr, "Unknown device type\n"); 
    break;
    }
  return cores;
}
