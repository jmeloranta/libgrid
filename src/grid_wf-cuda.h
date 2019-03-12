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
 * amp       = Ampltude for absorption (REAL; input).
 * lx        = Lower limit index for i (x) (INT; input).
 * hx        = Upper limit index for i (x) (INT; input).
 * ly        = Lower limit index for j (y) (INT; input).
 * hy        = Upper limit index for j (y) (INT; input).
 * lz        = Lower limit index for k (z) (INT; input).
 * hz        = Upper limit index for k (z) (INT; input).
 * 
 * Returns time_step scaling (CUCOMPLEX).
 *
 */

static __device__ CUCOMPLEX grid_cuda_wf_absorb(INT i, INT j, INT k, CUREAL amp, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz) {

  CUCOMPLEX t;

//  if(i >= lx && i <= hx && j >= ly && j <= hy && k >= lz && k <= hz) return CUMAKE(1.0,0.0);

  t.x = 1.0; t.y = 0.0;

  if(i < lx) t.y -= ((CUREAL) (lx - i)) / (CUREAL) lx;
  else if(i > hx) t.y -= ((CUREAL) (i - hx)) / (CUREAL) lx;

  if(j < ly) t.y -= ((CUREAL) (ly - j)) / (CUREAL) ly;
  else if(j > hy) t.y -= ((CUREAL) (j - hy)) / (CUREAL) ly;

  if(k < lz) t.y -= ((CUREAL) (lz - k)) / (CUREAL) lz;
  else if(k > hz) t.y -= ((CUREAL) (k - hz)) / (CUREAL) lz;

  t.y *= amp / 3.0;
  return t;
}
