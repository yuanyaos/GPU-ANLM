---------------------------------------------------------------------

          GPU-accelerated adaptive non-local means filter

---------------------------------------------------------------------
Copyright (c) 2018 Yaoshen Yuan, Qianqian Fang
---------------------------------------------------------------------

Author:       Yaoshen Yuan and Qianqian Fang
Webpage:      http://mcx.space
Contact:      yuan.yaos at husky.neu.edu
              q.fang at neu.edu

Publication:

Yaoshen Yuan, Leiming Yu, Zafer Dogan, and Qianqian Fang, "Graphics processing
units-accelerated adaptive nonlocal means filter for denoising three-dimensional
Monte Carlo photon transport simulations," J. of Biomedical Optics, 23(12), 121618 (2018).
https://doi.org/10.1117/1.JBO.23.12.121618


== Contents ==

 \src
    --ANLMGPU.c
    --filterGPU.h
    --filterGPU.cu
    --filterGPU_v.cu
    --filterGPU_s.cu
    --Makefile
 \bin
    --ganlm.mexa64
 \demo
    --demo_basic.m
    --demo_MCdenoising.m
    --data.mat
 \Wave3D
 README.txt
 LICENSE.txt


== Introduction ==

The Monte Carlo (MC) photon transport is the gold standard for modeling light 
propagation inside turbid media. However, the inherent stochastic noise becomes 
dominant when using less photons or in the region far away from the source. 
Instead of lauching more photons, we can apply denoising technique to achieve  
results equivalent to lauching more photons. This software takes advantage of 
the adaptive non-local means (ANLM) filter [2] for its adaptivity to spatially 
varying noise to denoise the shot noise in the MC images while having a good 
edge preservation. However, the original CPU version is less beneficial for MC 
images due to its long run-time. This work therefore optimized the speed using 
GPU. In a previous work [3], a GPU version of ANLM filter was implemented but 
there are some simplifications and a few features missing. The comparison can 
be seen below.

_____________________________________________________________________
Main Features             CPU-ANLM         GPU-ANLM         this work
---------------------------------------------------------------------
Compute type                   CPU              GPU               GPU
Data type*                  double       short integer          float 
Block-wise update              yes               no                no
Non-local patch
pre-selection                  yes               no               yes
Adaptive to noise               3D               2D                3D
Filtering Gaussian             yes              yes               yes
Filtering Rician               yes              yes               yes
Sub-band mixing                yes               no               yes
GPU block                       -           16x16x1             8x8x8
GPU texture memory              -                no               yes
Source code            open-source     closed-source      open-source
_____________________________________________________________________

Furthermore, this software can be not only used for MC images, but also for 
denoising other volumetric images as the MR or CT 3D scans.


== References ==

If you use this filter in your research, the author of this software would like 
you to cite the below paper in your related publications [1].

[1] Yuan Y, Yu L, Doğan Z, Fang Q. Graphics processing units-accelerated adaptive
nonlocal means filter for denoising three-dimensional Monte Carlo photon transport
simulations. Journal of Biomedical Optics. 2018 Nov; 23(12): 121618.

In addition, other publications relevant to Monte Carlo photon transport and 
adaptive non-local means filter can be found below.

[2] Manjón J V, Coupé P, Martí‐Bonmatí L, et al. "Adaptive non‐local 
means denoising of MR images with spatially varying noise levels," Journal of 
Magnetic Resonance Imaging, 2010, 31(1): 192-203.

[3] Granata D, Amato U, Alfano B. "MRI denoising by nonlocal means on 
multi-GPU," Journal of Real-Time Image Processing, 2016: 1-11.

[4] Fang Q, Boas D A. "Monte Carlo simulation of photon migration in 3D turbid 
media accelerated by graphics processing units," Optics express, 2009, 
17(22): 20178-20190.