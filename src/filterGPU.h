/*--------------------------------------------------------------------------*/
// A GPU-accelerated Adaptive Non-Local Means Filter for Denoising 3D Monte
// Carlo Photon Transport Simulations
/*--------------------------------------------------------------------------*/

// Yaoshen Yuan - yuan.yaos at husky.neu.edu
// Qianqian Fang - q.fang at neu.edu
// Computational Optics & Translational Imaging Lab
// Northeastern University

// Publication:
// Yaoshen Yuan, Leiming Yu, Zafer Dogan, and Qianqian Fang, "Graphics processing
// units-accelerated adaptive nonlocal means filter for denoising three-dimensional
// Monte Carlo photon transport simulations," J. of Biomedical Optics, 23(12), 121618 (2018).
// https://doi.org/10.1117/1.JBO.23.12.121618

// Copyright (C) 2018 Yaoshen Yuan, Qianqian Fang

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
  bool rician;
} FilterParam;

/* void runFilter4(float * ima_input, float * Estimate1, int f1, float * Estimate2, int f2, int v, int dimx, int dimy, int dimz, float MAX, int width2, int width, int s, int gpuid, bool rician)  */
void runFilter(float * ima, float * Estimate1, int f1, float * Estimate2, int f2, int v, int dimx, int dimy, int dimz, float MAX, int width, int width2, int s, int gpuid, bool rician);

void runFilter_v(float * ima, float * Estimate1, int f1, float * Estimate2, int f2, int v, int dimx, int dimy, int dimz, float MAX, int width, int width2, int s, int gpuid, bool rician);

void runFilter_s(float * ima, float * Estimate1, int f1, int v, int dimx, int dimy, int dimz, float MAX, int width2, int width, int s, int gpuid, bool rician);
#ifdef  __cplusplus
}
#endif

#endif