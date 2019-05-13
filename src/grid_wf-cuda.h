/*
 * Shared device code. These functions must be static - otherwise they will be multiply defined.
 *
 */

/*
 * grid_wf_absorb cuda equivalent.
 *
 * i         = Index i (x) (INT; input).
 * j         = Index j (y) (INT; input).
 * k         = Index k (z) (INT; input).
 * lx        = Lower limit index for i (x) (INT; input).
 * hx        = Upper limit index for i (x) (INT; input).
 * ly        = Lower limit index for j (y) (INT; input).
 * hy        = Upper limit index for j (y) (INT; input).
 * lz        = Lower limit index for k (z) (INT; input).
 * hz        = Upper limit index for k (z) (INT; input).
 * 
 * Returns time_step scaling (CUREAL).
 *
 */

static __device__ CUREAL grid_cuda_wf_absorb(INT i, INT j, INT k, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz) {

  CUREAL t;

  t = 0.0;

  if(i < lx) t += ((CUREAL) (lx - i)) / (CUREAL) lx;
  else if(i > hx) t += ((CUREAL) (i - hx)) / (CUREAL) lx;

  if(j < ly) t += ((CUREAL) (ly - j)) / (CUREAL) ly;
  else if(j > hy) t += ((CUREAL) (j - hy)) / (CUREAL) ly;

  if(k < lz) t += ((CUREAL) (lz - k)) / (CUREAL) lz;
  else if(k > hz) t += ((CUREAL) (k - hz)) / (CUREAL) lz;

  t *= 2.0 / 3.0;   // new
  if(t > 1.0) return 1.0; // new
  return t; // new
//  return t / 3.0; // old code
}
