#ifndef __CUDA_YAOSHEN
#define __CUDA_YAOSHEN

#ifdef  __cplusplus
extern "C" {
#endif

typedef struct  KernelParams {
  int patchsize;
  int searchsize;
  float rpatchnomalize;
  int dimx,dimy,dimz;
  int blockdimx, blockdimy, blockdimz;
  float maxval;
  int blockwidth;
  int sharedwidth_x;
  int sharedwidth;
  int sharedSlice;
  int apron;
  int apronFull;
  int apronShared;
  int rician;
} FilterParam;

/* void runFilter4(float * ima_input, float * Estimate1, int f1, float * Estimate2, int f2, int v, int dimx, int dimy, int dimz, float MAX, int width2, int width, int s, int gpuid, bool rician)  */
void runFilter(float * ima, float * Estimate1, int f1, float * Estimate2, int f2, int v, int dimx, int dimy, int dimz, float MAX, int width, int width2, int s, int gpuid, int rician);

void runFilter_v(float * ima, float * Estimate1, int f1, float * Estimate2, int f2, int v, int dimx, int dimy, int dimz, float MAX, int width, int width2, int s, int gpuid, int rician);

void runFilter_s(float * ima, float * Estimate1, int f1, int v, int dimx, int dimy, int dimz, float MAX, int width2, int width, int s, int gpuid, int rician);
#ifdef  __cplusplus
}
#endif

#endif