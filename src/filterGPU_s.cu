/*--------------------------------------------------------------------------*/
// A GPU-accelerated Adaptive Non-Local Means Filter for Denoising 3D Monte
// Carlo Photon Transport Simulations

// filterGPU_s.cu is the version of the single ANLM filter (B+Opt2+Opt3)
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

#include <stdio.h>
#include <math.h>       /* floor */
#include <unistd.h>
#include "filterGPU.h"
#include <time.h>
#include <cuda.h>

#define CUDA_ASSERT(a) cuda_assess((a),__FILE__,__LINE__)

void cuda_assess(cudaError_t cuerr,const char *file, const int linenum);
__constant__ FilterParam gcfg[1];
texture<float,cudaTextureType3D,cudaReadModeElementType> ima_tex;
texture<float,cudaTextureType3D,cudaReadModeElementType> means_tex;
texture<float,cudaTextureType3D,cudaReadModeElementType> variances_tex;
texture<float,cudaTextureType3D,cudaReadModeElementType> R_tex;

__device__ inline static float * distance(float *d, int x,int y,int z,int sx,int sy,int sz, float *ima_space)
{
// d=distance(ima,i,j,k,ni,nj,nk,f,cols,rows,slices);
/*
ima: the unfiltered image.
medias: the image filtered by 3x3 box filter.
x, y, z: the location of the "center of the local patch" in shared memory.
nx, ny, nz: the location of the "center of the non-local patch" in the full-image.
sx, sy, sz: the location of the "center of the non-local patch" in shared memory
f: patch size.
gcfg->dimx, gcfg->dimy, gcfg->dimz: the size of the image.
*/
	float dt,distancetotal;
	int i,j,k,ni1,nj1,nk1,ni4,nj4,nk4,f1;

	f1=gcfg->patchsize;
	distancetotal=0.f;

	for(k=-f1;k<=f1;k++)
	{
	 nk1=z+k;	// local in shared memory
	 nk4=sz+k;	// non-local in shared memory

		 for(j=-f1;j<=f1;j++)
		 {
		  nj1=y+j;
		  nj4=sy+j;

			for(i=-f1;i<=f1;i++)
			{
			  ni1=x+i;
			  ni4=sx+i;

			  d[0] = ima_space[nk1*gcfg->sharedSlice+nj1*gcfg->sharedwidth_x+ni1]-ima_space[nk4*gcfg->sharedSlice+nj4*gcfg->sharedwidth_x+ni4];
			  dt = d[0]*d[0];
			  distancetotal = distancetotal + dt;
			}
		 }
	}

	d[0]=distancetotal*gcfg->rpatchnomalize;

	return d;
}

__device__ inline static float * distance2(float *d, int x,int y,int z, int fx, int fy, int fz, int nx,int ny,int nz, int sx, int sy, int sz, float *ima_space)
{
//													 local in shared    local in full image    non-local in full image    non-local in shared
// d=distance2(ima,means,i,j,k,ni,nj,nk,f,cols,rows,slices);
/*
ima: the unfiltered image.
medias: the image filtered by 3x3 box filter.
x, y, z: the location of the "center of the local patch" in shared memory.
fx, fy, fz: the location of the "center of the local patch" in full image.
nx, ny, nz: the location of the "center of the non-local patch" in the full-image.
sx, sy, sz: the location of the "center of the non-local patch" in shared memory
f: patch size.
gcfg->dimx, gcfg->dimy, gcfg->dimz: the size of the image.
*/
	float dt,distancetotal;
	int i,j,k,ni1,nj1,nk1,ni2,nj2,nk2,ni3,nj3,nk3,ni4,nj4,nk4,f1;

	f1=gcfg->patchsize;
	distancetotal=0;
	for(k=-f1;k<=f1;k++) // 1D
	{
		 nk1=z+k;	// local in shared memory
		 nk2=nz+k;	// non-local in full image
		 nk3=fz+k;	// local in full image
		 nk4=sz+k;	// non-local in shared memory
		 for(j=-f1;j<=f1;j++) // 2D
		 {
				nj1=y+j;
				nj2=ny+j;
				nj3=fy+j;
				nj4=sy+j;
				for(i=-f1;i<=f1;i++)    // 3D
				{
					ni1=x+i;
					ni2=nx+i;
					ni3=fx+i;
					ni4=sx+i;

					// Load whole search area into shared memory
					d[0]=(ima_space[nk1*(gcfg->sharedSlice)+(nj1*gcfg->sharedwidth)+ni1]-tex3D(means_tex,ni3,nj3,nk3))-(ima_space[nk4*(gcfg->sharedSlice)+(nj4*gcfg->sharedwidth)+ni4]-tex3D(means_tex,ni2,nj2,nk2));
					dt = d[0]*d[0];
					distancetotal = distancetotal + dt;
				}
		 }
	}
	
	d[0]=distancetotal*gcfg->rpatchnomalize;
	return d;
}


__global__ static void ANLMfilter(float *Estimate)
{
/*	
	ima: the input unfiltered image.
	means: the mean value of ima by using 3x3 block filter.
	variance: the variance of ima by using 3x3 block filter.
	average: save the value of weighted summation for patch i.
	Estimate: save the sum of all the filtered values for each voxel.
	Label: save the count of how many filtered values are computed for each voxel. 
	v: the searching area.
	rows, cols, slices: the size of the image (x, y, z).
	gcfg->maxval: MAXimum value of the image.
*/
	// declare shared memory
	extern __shared__ float ima_space[];	// extern indicates the dynamic memory allocation.

	int i,j,k,rc,ii,jj,kk,ni,nj,nk,is,js,ks,istart,jstart,kstart,icount,jcount,kcount,threadIdx_x,threadIdx_y,threadIdx_z,i_Fl,j_Fl,k_Fl,i_fl,j_fl,k_fl,i_sl,j_sl,k_sl;
	float totalweight,t1,t1i,t2,w,distanciaminima,estimate,means_t,variances_t,ima_tt,means_tt,variances_tt;
	float d[2];

/*	Parameters setting	*/
	// const float pi = 3.14159265359f;
	const float mu1 = 0.95f;
	const float var1 = 0.5f;
    const float rmu1= 1.f/mu1;
    const float rvar1= 1.f/var1;
	rc=gcfg->dimy*gcfg->dimx;
	estimate = 0.0f;

	d[0]=0;
	d[1]=0;

	threadIdx_x = threadIdx.x;
	threadIdx_y = threadIdx.y;
	threadIdx_z = threadIdx.z;

	totalweight=0.0f;

	distanciaminima=100000000000000.f;
	
	i = blockIdx.x*gcfg->blockwidth+threadIdx_x;	// The coordinate of local patch in the original image (image that does NOT includes the apron)
	j = blockIdx.y*gcfg->blockwidth+threadIdx_y;
	k = blockIdx.z*gcfg->blockwidth+threadIdx_z;
	
	i_Fl = i+gcfg->apronFull;    // The coordinate of local patch in the super full-image (image that includes the apron+s)
	j_Fl = j+gcfg->apronFull;
	k_Fl = k+gcfg->apronFull;
	
	i_fl = i+gcfg->apron;    // The coordinate of local patch in the full-image (image that includes the apron)
	j_fl = j+gcfg->apron;
	k_fl = k+gcfg->apron;

	i_sl = threadIdx_x+gcfg->apronShared;		// The coordinate of local patch in the shared memory
	j_sl = threadIdx_y+gcfg->apronShared;
	k_sl = threadIdx_z+gcfg->apronShared;
	// return if the thread number exceeds the dimension
    if(i>=gcfg->dimx || j>=gcfg->dimy || k>=gcfg->dimz)
         return;

	if(threadIdx_z==0){
	    kstart = -gcfg->apronShared;
	    kcount = gcfg->apronShared+1;
	}
	else if(threadIdx_z==gcfg->blockwidth-1 || k==gcfg->dimz-1){
	    kstart = 0;
	    kcount = gcfg->apronShared+1;
	}
	else{
	    kstart = 0;
	    kcount = 1;
	} 

	if(threadIdx_y==0){
	    jstart = -gcfg->apronShared;
	    jcount = gcfg->apronShared+1;
	}
	else if(threadIdx_y==gcfg->blockwidth-1 || j==gcfg->dimy-1){
	    jstart = 0;
	    jcount = gcfg->apronShared+1;
	}
	else{
	    jstart = 0;
	    jcount = 1;
	}

	if(threadIdx_x==0){
	    istart = -gcfg->apronShared;
	    icount = gcfg->apronShared+1;
	}
	else if(threadIdx_x==gcfg->blockwidth-1 || i==gcfg->dimx-1){
	    istart = 0;
	    icount = gcfg->apronShared+1;
	}
	else{
	    istart = 0;
	    icount = 1;
	}

	/* special case */
	if(threadIdx_x==0 && i==gcfg->dimx-1){
	    istart = -gcfg->apronShared;
	    icount = 2*gcfg->apronShared+1;
	}
	
	if(threadIdx_y==0 && j==gcfg->dimy-1){
	    jstart = -gcfg->apronShared;
	    jcount = 2*gcfg->apronShared+1;
	}
	
	if(threadIdx_z==0 && k==gcfg->dimz-1){
	    kstart = -gcfg->apronShared;
	    kcount = 2*gcfg->apronShared+1;
	}

	for(ks=0;ks<kcount;ks++){
	    for(js=0;js<jcount;js++){
			for(is=0;is<icount;is++){
			    // load the image data into shared memory
			    ima_space[(k_sl+kstart+ks)*gcfg->sharedSlice+(j_sl+jstart+js)*gcfg->sharedwidth_x+(i_sl+istart+is)] = tex3D(ima_tex,i_Fl+istart+is,j_Fl+jstart+js,k_Fl+kstart+ks);
			}
	    }
	}


	__syncthreads();

	Estimate[k*rc+(j*gcfg->dimx)+i] = 0.0f;
	means_t = tex3D(means_tex,i_fl,j_fl,k_fl);
	variances_t = tex3D(variances_tex,i_fl,j_fl,k_fl);

    /*  COMPUTE ADAPTIVE PARAMTER */
    for(kk=-gcfg->searchsize; kk<=gcfg->searchsize; kk++)
    {
        nk=k_fl+kk;           // here nk, ni, nj mean the coordinates of the central voxel of patch j
        for(jj=-gcfg->searchsize; jj<=gcfg->searchsize; jj++)
        {
            nj=j_fl+jj;
            for(ii=-gcfg->searchsize; ii<=gcfg->searchsize; ii++)
            {
                ni=i_fl+ii;

                if(ii==0 && jj==0 && kk==0) continue; // Skip the patch when i==j
                if(ni-gcfg->apron>=0 && nj-gcfg->apron>=0 && nk-gcfg->apron>=0 && ni-gcfg->apron<gcfg->dimx && nj-gcfg->apron<gcfg->dimy && nk-gcfg->apron<gcfg->dimz)
                {
    				ima_tt = ima_space[(k_sl+kk)*(gcfg->sharedSlice)+((j_sl+jj)*gcfg->sharedwidth_x)+(i_sl+ii)];
                    means_tt = tex3D(means_tex,ni,nj,nk);
                    variances_tt = tex3D(variances_tex,ni,nj,nk);

					// The purpose is to set the threshold to eliminate the patches (j) that are too far away from the patch i
                    t1 = means_t/means_tt;
                    t1i= (gcfg->maxval-means_t)/(gcfg->maxval-means_tt);
                    t2 = (variances_t)/(variances_tt);

                    if( (t1>mu1 && t1<rmu1) || (t1i>mu1 && t1i<rmu1) && t2>var1 && t2<rvar1)
                    {
						// d: save Euclidean distance; coordinates in shared memory; coordinates in full image.
    					// distance2(d,i_sl,j_sl,k_sl,ni,nj,nk,i_sl+ii,j_sl+jj,k_sl+kk, ima_space);


    					distance2(d,i_sl,j_sl,k_sl,i_fl,j_fl,k_fl,ni,nj,nk,i_sl+ii,j_sl+jj,k_sl+kk,ima_space);
                        if(d[0]<distanciaminima) distanciaminima=d[0];    // Get the minimum distance in order to calculate the adaptive variance
                    }
                }
            }
        }
    }
    if(distanciaminima==0) distanciaminima=1;

    /*  FILTERING PROCESS */
    if(gcfg->rician==0)		// No rician noise
    {
	    for(kk=-gcfg->searchsize; kk<=gcfg->searchsize; kk++)
	    {
	        nk=k_fl+kk;                     // here nk, ni, nj mean the coordinates of the central voxel of patch j
	        for(jj=-gcfg->searchsize; jj<=gcfg->searchsize; jj++)
	        {
	            nj=j_fl+jj;
	            for(ii=-gcfg->searchsize; ii<=gcfg->searchsize; ii++)
	            {
	                ni=i_fl+ii;
					if(ni-gcfg->apron>=0 && nj-gcfg->apron>=0 && nk-gcfg->apron>=0 && ni-gcfg->apron<gcfg->dimx && nj-gcfg->apron<gcfg->dimy && nk-gcfg->apron<gcfg->dimz)
	                {  
	    				ima_tt = ima_space[(k_sl+kk)*(gcfg->sharedSlice)+((j_sl+jj)*gcfg->sharedwidth_x)+(i_sl+ii)];
	    				means_tt = tex3D(means_tex,ni,nj,nk);
	                    variances_tt = tex3D(variances_tex,ni,nj,nk);

						t1 = (means_t)/(means_tt);
	                    t1i= (gcfg->maxval-means_t)/(gcfg->maxval-means_tt);
	                    t2 = (variances_t)/(variances_tt);

	                    if( (t1>mu1 && t1<rmu1) || (t1i>mu1 && t1i<rmu1) && t2>var1 && t2<rvar1)
	                    {
	                        distance(d,i_sl,j_sl,k_sl,i_sl+ii,j_sl+jj,k_sl+kk, ima_space);
	    					w = expf(-d[0]/distanciaminima);
							estimate = estimate + w*ima_tt;
	                        totalweight = totalweight + w;
	                    }
	                }
	            }
	        }
	    }
	    estimate = estimate/totalweight;
	}
	else 				// Consider rician noise
	{
		for(kk=-gcfg->searchsize; kk<=gcfg->searchsize; kk++)
	    {
	        nk=k_fl+kk;                     // here nk, ni, nj mean the coordinates of the central voxel of patch j
	        for(jj=-gcfg->searchsize; jj<=gcfg->searchsize; jj++)
	        {
	            nj=j_fl+jj;
	            for(ii=-gcfg->searchsize; ii<=gcfg->searchsize; ii++)
	            {
	                ni=i_fl+ii;	                
					if(ni-gcfg->apron>=0 && nj-gcfg->apron>=0 && nk-gcfg->apron>=0 && ni-gcfg->apron<gcfg->dimx && nj-gcfg->apron<gcfg->dimy && nk-gcfg->apron<gcfg->dimz)
	                {  
	    				ima_tt = ima_space[(k_sl+kk)*(gcfg->sharedSlice)+((j_sl+jj)*gcfg->sharedwidth_x)+(i_sl+ii)];
	    				means_tt = tex3D(means_tex,ni,nj,nk);
	                    variances_tt = tex3D(variances_tex,ni,nj,nk);

						t1 = (means_t)/(means_tt);
	                    t1i= (gcfg->maxval-means_t)/(gcfg->maxval-means_tt);
	                    t2 = (variances_t)/(variances_tt);

	                    if( (t1>mu1 && t1<rmu1) || (t1i>mu1 && t1i<rmu1) && t2>var1 && t2<rvar1)
	                    {
	                        distance(d,i_sl,j_sl,k_sl,i_sl+ii,j_sl+jj,k_sl+kk,ima_space);
	    					w = expf(-d[0]/distanciaminima);
							estimate = estimate + w*ima_tt*ima_tt;
	                        totalweight = totalweight + w;
	                    }
	                }
	            }
	        }
	    }

	    estimate = estimate/totalweight;
	    estimate = estimate-2.0f*distanciaminima;
	    estimate = (estimate>0.0f)?estimate:0.0f;
	    estimate = sqrtf(estimate);
	}

	Estimate[k*rc+(j*gcfg->dimx)+i] = estimate;
	__syncthreads();

}

__global__ static void preProcess(cudaPitchedPtr mean, cudaPitchedPtr R, cudaPitchedPtr var, int dimfull_x, int dimfull_y, int dimfull_z, int s, int blockwidth)
{
	extern __shared__ float ima_shared[];

	int sharedwidthSlice, sharedwidth, threadIdx_x, threadIdx_y, threadIdx_z, istart, jstart, kstart, icount, jcount, kcount, i, j, k, ii, jj, kk, i_fl, j_fl, k_fl, i_sl, j_sl, k_sl, is, js, ks;
	int N = (2*s+1)*(2*s+1)*(2*s+1);	// size of the filter box
	
	sharedwidth = blockwidth+2*s;
	sharedwidthSlice = sharedwidth*sharedwidth;
	
	threadIdx_x = threadIdx.x;
	threadIdx_y = threadIdx.y;
	threadIdx_z = threadIdx.z;

	i = blockIdx.x*blockwidth+threadIdx_x;	// The coordinate of local patch in the original image (image that does NOT includes the apron)
	j = blockIdx.y*blockwidth+threadIdx_y;
	k = blockIdx.z*blockwidth+threadIdx_z;
	
	i_fl = i+s;    // The coordinate of local patch in the full-image (image that includes the apron)
	j_fl = j+s;
	k_fl = k+s;

	i_sl = threadIdx_x+s;		// The coordinate of local patch in the shared memory
	j_sl = threadIdx_y+s;
	k_sl = threadIdx_z+s;
	// return if the thread number exceeds the dimension
        if(i>=dimfull_x || j>=dimfull_y || k>=dimfull_z)
             return;

	/* general case */		
	if(threadIdx_z==blockwidth-1 || k==dimfull_z-1){
	    kstart = 0;
	    kcount = s+1;
	}
	else if(threadIdx_z==0){
	    kstart = -s;
	    kcount = s+1;
	}
	else{
	    kstart = 0;
	    kcount = 1;
	}
	
	if(threadIdx_y==blockwidth-1 || j==dimfull_y-1){
	    jstart = 0;
	    jcount = s+1;
	}
	else if(threadIdx_y==0){
	    jstart = -s;
	    jcount = s+1;
	}
	else{
	    jstart = 0;
	    jcount = 1;
	}
	
	if(threadIdx_x==blockwidth-1 || i==dimfull_x-1){
	    istart = 0;
	    icount = s+1;
	}
	else if(threadIdx_x==0){
	    istart = -s;
	    icount = s+1;
	}
	else{
	    istart = 0;
	    icount = 1;
	}

	/* special case */
	if(threadIdx_x==0 && i==dimfull_x-1){
	    istart = -s;
	    icount = 2*s+1;
	}
	
	if(threadIdx_y==0 && j==dimfull_y-1){
	    jstart = -s;
	    jcount = 2*s+1;
	}
	
	if(threadIdx_z==0 && k==dimfull_z-1){
	    kstart = -s;
	    kcount = 2*s+1;
	}


	for(ks=0;ks<kcount;ks++){
	    for(js=0;js<jcount;js++){
			for(is=0;is<icount;is++){
			    // load the image data into shared memory
			    ima_shared[(k_sl+kstart+ks)*sharedwidthSlice+(j_sl+jstart+js)*sharedwidth+(i_sl+istart+is)] = tex3D(ima_tex,i_fl+istart+is,j_fl+jstart+js,k_fl+kstart+ks);
			}
	    }
	}
	
	__syncthreads();

	float Mt, Vt=0;
	
	// mean matrix
	char* Ptr = (char *) mean.ptr;		// location in mean
	size_t pitch = mean.pitch;	// dimx
	size_t slicePitch = pitch*dimfull_y;	// dimx*dimy	
	char* slice = Ptr+k*slicePitch;
	float* row = (float *) (slice+j*pitch);
	
	char* Ptr2 = (char *) R.ptr;		// location in R
	char* slice2 = Ptr2+k*slicePitch;
	float* row2 = (float *) (slice2+j*pitch);

	Mt = 0;
	for(kk=-s; kk<=s; kk++)
	    for(jj=-s;jj<=s;jj++)
			for(ii=-s;ii<=s;ii++)
			{
			    Mt = Mt+ima_shared[(k_sl+kk)*sharedwidthSlice+(j_sl+jj)*sharedwidth+(i_sl+ii)];
			}
	Mt = Mt/N;	// mean value for voxel at (i,j,k)

	row[i] = Mt;	// Save mean value into global mean matrix
	row2[i] = ima_shared[k_sl*sharedwidthSlice+j_sl*sharedwidth+i_sl]-Mt;	// Save R (image-mean) value into global R memory

//	__syncthreads();


	char* Ptr3 = (char *) var.ptr;		// location in var
	char* slice3 = Ptr3+k*slicePitch;
	float* row3 = (float *) (slice3+j*pitch);

	float t = 0;
	for(kk=-s; kk<=s; kk++)
	    for(jj=-s;jj<=s;jj++)
			for(ii=-s;ii<=s;ii++)
			{
			    t = ima_shared[(k_sl+kk)*sharedwidthSlice+(j_sl+jj)*sharedwidth+(i_sl+ii)]-Mt;
			    Vt = Vt+t*t;
			}
	Vt = Vt/(N-1);	// variance value for voxel at (i,j,k)

	row3[i] = Vt;	// Save variance value into global var matrix
	
	__syncthreads();
}

void runFilter_s(float * ima_input, float * Estimate1, int f1, int v, int dimx, int dimy, int dimz, float MAX, int width2, int width, int s, int gpuid, bool rician)

{
/*
	ima_input: input image
	Estimate1: output image
	f1: patch size for the first filtering
	v: searching area
	dimx, dimy, dimz: the size of each dimension of the image
	MAX: max value of the input image
	width2: block width
	width: pre-process blockwidth
	s: patch radius for computing mean, R, variance matrix
	gpuid: GPU id
*/
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	printf("====== Designed for single filtering ======\n");
	printf("Baseline+Opt2+Opt3\n");
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,gpuid);
	printf("Device name: %s\n",prop.name);
	
	//@@@@@@@@@@@@ Pre-processing kernel starts @@@@@@@@@@@@/
	int dimPrecom_x, dimPrecom_y, dimPrecom_z, dimfull_x, dimfull_y, dimfull_z, widthPrecom, blockPrecom_x, blockPrecom_y, blockPrecom_z, sharedsizePre;
	cudaArray *ima=0, *meansArray=0, *variancesArray=0, *RArray=0;
	cudaEvent_t start, stop;
	float elapsedTime;
	
	widthPrecom = width;
	dimPrecom_x = dimx+2*(f1+v+s);	// Size of the input image
	dimPrecom_y = dimy+2*(f1+v+s);
	dimPrecom_z = dimz+2*(f1+v+s);
	dimfull_x = dimx+2*(f1+v);	// Size of the mean, R and variance matrix
	dimfull_y = dimy+2*(f1+v);
	dimfull_z = dimz+2*(f1+v);
	printf("dim_x=%d\tdim_y=%d\tdim_z=%d\n",dimx,dimy,dimz);
	// const int sizePre = (dimfull_x*dimfull_y*dimfull_z)*sizeof(float);
	
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaExtent imaSize = make_cudaExtent(dimPrecom_x,dimPrecom_y,dimPrecom_z);
	
	// Load ima_input into texture memory
	// cudaMalloc3DArray allocate cudaArray which is only for texture memory
	cudaMalloc3DArray(&ima, &channelDesc, imaSize);
	cudaMemcpy3DParms copyParams1 = {0};
	// make_cudaPitchedPtr returns cudaPitchedPtr{pitch(pitch of the pointer),ptr(pointer to the allocated mem),xsize,ysize (logical width and height)}
	copyParams1.srcPtr = make_cudaPitchedPtr((void*)ima_input, imaSize.width*sizeof(float), imaSize.width, imaSize.height);
	copyParams1.dstArray = ima;	// destination array
	copyParams1.extent = imaSize;		// dimensions of the transferred area in elements
	copyParams1.kind = cudaMemcpyHostToDevice;	// copy from host to device
	cudaMemcpy3D(&copyParams1);
	
	ima_tex.normalized = false;
	ima_tex.filterMode = cudaFilterModePoint;	//cudaFilterModePoint; cudaFilterModeLinear;
	ima_tex.addressMode[0] = cudaAddressModeClamp;
	ima_tex.addressMode[1] = cudaAddressModeClamp;
	ima_tex.addressMode[2] = cudaAddressModeClamp;
	cudaBindTextureToArray(ima_tex, ima, channelDesc);
	
	
	// Allocate global memory for mean, R and variance
	cudaExtent imaApron = make_cudaExtent(dimfull_x*sizeof(float),dimfull_y,dimfull_z);
	cudaPitchedPtr mean, R, var;
	// cudaMalloc3D allocate global memory
	cudaMalloc3D(&mean, imaApron);
	cudaMalloc3D(&R, imaApron);
	cudaMalloc3D(&var, imaApron);

	sharedsizePre = (widthPrecom+2*s)*(widthPrecom+2*s)*(widthPrecom+2*s);
	blockPrecom_x = (dimfull_x+widthPrecom-1)/widthPrecom;
	blockPrecom_y = (dimfull_y+widthPrecom-1)/widthPrecom;
	blockPrecom_z = (dimfull_z+widthPrecom-1)/widthPrecom;
	
	printf("shared size for pre-computation=%fkB\n",float(sharedsizePre*sizeof(float))/1024);
	if(sharedsizePre*sizeof(float)>48*1024){
	    printf("The memory requirement for pre-computation is larger than the size of shared memory!");
	    exit(1);
	}
	
	//@@@@@@@@@@ Pre-processing kernel time start @@@@@@@@@@/
	CUDA_ASSERT(cudaEventCreate(&start));
	CUDA_ASSERT(cudaEventRecord(start,0));
	
	dim3 dimBlockPre(widthPrecom,widthPrecom,widthPrecom);
	dim3 dimGridPre(blockPrecom_x,blockPrecom_y,blockPrecom_z);

	// preProcess(cudaPitchedPtr mean, cudaPitchedPtr R, cudaPitchedPtr var, int dimfull_x, int dimfull_y, int dimfull_z, int s, int blockwidth)
	preProcess<<<dimGridPre,dimBlockPre,sharedsizePre*sizeof(float)>>>(mean, R, var, dimfull_x, dimfull_y, dimfull_z, s, widthPrecom);
	// preProcess<<<dimGridPre,dimBlockPre>>>(f1);

	CUDA_ASSERT(cudaThreadSynchronize());	// Synchronize until all mean, R and variance are finished
	
	CUDA_ASSERT(cudaEventCreate(&stop));
	CUDA_ASSERT(cudaEventRecord(stop,0));
	CUDA_ASSERT(cudaEventSynchronize(stop));
	
	CUDA_ASSERT(cudaEventElapsedTime(&elapsedTime,start,stop));
	printf("Pre-computation kernel time: %f ms\n" ,elapsedTime);
	//@@@@@@@@@@ Pre-processing kernel time end @@@@@@@@@@/
	
	
	
/*
	//@@@@@@@@@@ Copy from device to host start @@@@@@@@@@/
	cudaExtent imaOut = make_cudaExtent(dimfull_x,dimfull_y,dimfull_z);
	cudaMemcpy3DParms copyParamsOutput1 = {0};
	// make_cudaPitchedPtr returns cudaPitchedPtr{pitch(pitch of the pointer),ptr(pointer to the allocated mem),xsize,ysize (logical width and height)}
	copyParamsOutput1.srcPtr = mean;
	copyParamsOutput1.dstPtr = make_cudaPitchedPtr((void*)Estimate1, imaOut.width*sizeof(float), imaOut.width, imaOut.height);
	copyParamsOutput1.extent = imaApron;		// dimensions of the transferred area in elements
	copyParamsOutput1.kind = cudaMemcpyDeviceToHost;	// copy from host to device
	CUDA_ASSERT(cudaMemcpy3D(&copyParamsOutput1));
	
	cudaMemcpy3DParms copyParamsOutput2 = {0};
	// make_cudaPitchedPtr returns cudaPitchedPtr{pitch(pitch of the pointer),ptr(pointer to the allocated mem),xsize,ysize (logical width and height)}
	copyParamsOutput2.srcPtr = var;
	copyParamsOutput2.dstPtr = make_cudaPitchedPtr((void*)Estimate2, imaOut.width*sizeof(float), imaOut.width, imaOut.height);
	copyParamsOutput2.extent = imaApron;		// dimensions of the transferred area in elements
	copyParamsOutput2.kind = cudaMemcpyDeviceToHost;	// copy from host to device
	CUDA_ASSERT(cudaMemcpy3D(&copyParamsOutput2));
	//@@@@@@@@@@ Copy from device to host end @@@@@@@@@@/
*/	






	//@@@@@@@@@@@@ 1st kernel finished. Binding to texture memory starts @@@@@@@@@@@@/
	CUDA_ASSERT(cudaEventCreate(&start));
	CUDA_ASSERT(cudaEventRecord(start,0));
	//@@@@@@@@@@@@ Binding pre-computed mean, R and variance to texture memory start @@@@@@@@@@@@/
	cudaExtent Size = make_cudaExtent(dimfull_x,dimfull_y,dimfull_z);	
	// mean
	cudaMalloc3DArray(&meansArray, &channelDesc, Size);
	cudaMemcpy3DParms copyParams2 = {0};
	// make_cudaPitchedPtr returns cudaPitchedPtr{pitch(pitch of the pointer),ptr(pointer to the allocated mem),xsize,ysize (logical width and height)}
	copyParams2.srcPtr = mean;
	copyParams2.dstArray = meansArray;	// destination array
	copyParams2.extent = Size;		// dimensions of the transferred area in elements
	copyParams2.kind = cudaMemcpyDeviceToDevice;
	cudaMemcpy3D(&copyParams2);
	
	means_tex.normalized = false;
	means_tex.filterMode = cudaFilterModePoint;	//cudaFilterModePoint; cudaFilterModeLinear;
	means_tex.addressMode[0] = cudaAddressModeClamp;
	means_tex.addressMode[1] = cudaAddressModeClamp;
	means_tex.addressMode[2] = cudaAddressModeClamp;
	cudaBindTextureToArray(means_tex, meansArray, channelDesc);
	
	// R
	cudaMalloc3DArray(&RArray, &channelDesc, Size);
	cudaMemcpy3DParms copyParams3 = {0};
	// make_cudaPitchedPtr returns cudaPitchedPtr{pitch(pitch of the pointer),ptr(pointer to the allocated mem),xsize,ysize (logical width and height)}
	copyParams3.srcPtr = R;
	copyParams3.dstArray = RArray;	// destination array
	copyParams3.extent = Size;		// dimensions of the transferred area in elements
	copyParams3.kind = cudaMemcpyDeviceToDevice;
	cudaMemcpy3D(&copyParams3);
	
	R_tex.normalized = false;
	R_tex.filterMode = cudaFilterModePoint;	//cudaFilterModePoint; cudaFilterModeLinear;
	R_tex.addressMode[0] = cudaAddressModeClamp;
	R_tex.addressMode[1] = cudaAddressModeClamp;
	R_tex.addressMode[2] = cudaAddressModeClamp;
	cudaBindTextureToArray(R_tex, RArray, channelDesc);
	
	// variance
	cudaMalloc3DArray(&variancesArray, &channelDesc, Size);
	cudaMemcpy3DParms copyParams4 = {0};
	// make_cudaPitchedPtr returns cudaPitchedPtr{pitch(pitch of the pointer),ptr(pointer to the allocated mem),xsize,ysize (logical width and height)}
	copyParams4.srcPtr = var;
	copyParams4.dstArray = variancesArray;	// destination array
	copyParams4.extent = Size;		// dimensions of the transferred area in elements
	copyParams4.kind = cudaMemcpyDeviceToDevice;
	cudaMemcpy3D(&copyParams4);
	
	variances_tex.normalized = false;
	variances_tex.filterMode = cudaFilterModePoint;	//cudaFilterModePoint; cudaFilterModeLinear;
	variances_tex.addressMode[0] = cudaAddressModeClamp;
	variances_tex.addressMode[1] = cudaAddressModeClamp;
	variances_tex.addressMode[2] = cudaAddressModeClamp;
	cudaBindTextureToArray(variances_tex, variancesArray, channelDesc);
	
	CUDA_ASSERT(cudaEventCreate(&stop));
	CUDA_ASSERT(cudaEventRecord(stop,0));
	CUDA_ASSERT(cudaEventSynchronize(stop));
	CUDA_ASSERT(cudaEventElapsedTime(&elapsedTime, start,stop));
	printf("Elapsed time of texture binding: %f ms\n" ,elapsedTime);
	//@@@@@@@@@@@@ Binding pre-computed mean, R and variance to texture memory end @@@@@@@@@@@@/










	//@@@@@@@@@@@@ Filtering process starts @@@@@@@@@@@@/
	FilterParam param={0,v,0.f,dimx,dimy,dimz,0,0,0,MAX,width,0,0,0,0,0,0,rician};
	
	float *EstimateKernel;
	int Ndim,sharedsize;		// The total size of the input image
	Ndim = dimx*dimy*dimz;
	const int size = Ndim*sizeof(float);

 
	CUDA_ASSERT(cudaSetDevice(gpuid));
	
	CUDA_ASSERT(cudaMalloc((void**)&EstimateKernel, size));	// copy from EstimateKernel (device) to Estimate (host)
	

	//&&&&&&&&&& Parameter setup for the 1st filtering (large) &&&&&&&&&&/
	width = width2;
	sharedsize = (width+2*(f1+v))*(width+2*(f1+v))*(width+2*(f1+v)); // The shared memory size needed for each block
	printf("Shared size for filtering=%fkB\n",float(sharedsize*sizeof(float))/1024);

	printf("width=%d\n",width);
	if(width*width*width>1024){
	    printf("Error: The number of threads in a block is larger than 1024!");
	    exit(1);
	}

	param.patchsize = f1;
	param.rpatchnomalize = 1.f/((f1<<1)+1);
	param.rpatchnomalize = param.rpatchnomalize*param.rpatchnomalize*param.rpatchnomalize;
	param.blockdimx = (dimx+width-1)/width;
	param.blockdimy = (dimy+width-1)/width;
	param.blockdimz = (dimy+width-1)/width;
	param.blockwidth = width;
	param.sharedwidth_x = width+2*(f1+v);
	param.sharedwidth = width+2*(f1+v);		// The shared width at other dimension is still not changed.
	param.sharedSlice = param.sharedwidth_x*param.sharedwidth;
	param.apron = f1+v;
	param.apronFull = f1+v+s;
	param.apronShared = f1+v;
	CUDA_ASSERT(cudaMemcpyToSymbol(gcfg, &param, sizeof(FilterParam), 0, cudaMemcpyHostToDevice));

	printf("1st: searchsize=%d\n", param.searchsize);
	printf("1st: patchsize=%d\n", param.patchsize);





	//@@@@@@@@@@ 1st filtering (large): Time for filtering kernel start @@@@@@@@@@/
	CUDA_ASSERT(cudaEventCreate(&start));
	CUDA_ASSERT(cudaEventRecord(start,0));

	dim3 dimBlock(width,width,width);
	dim3 dimGrid(param.blockdimx,param.blockdimy,param.blockdimz);

	// First filtering
	ANLMfilter<<<dimGrid, dimBlock,sharedsize*sizeof(float)>>>(EstimateKernel);
	CUDA_ASSERT(cudaThreadSynchronize());
	
	CUDA_ASSERT(cudaEventCreate(&stop));
	CUDA_ASSERT(cudaEventRecord(stop,0));
	CUDA_ASSERT(cudaEventSynchronize(stop));
	
	CUDA_ASSERT(cudaEventElapsedTime(&elapsedTime, start,stop));
	printf("1st filtering (large) kernel time: %f ms\n\n\n" ,elapsedTime);
	//@@@@@@@@@@ 1st filtering (large): Time for filtering kernel  end @@@@@@@@@@/



	//@@@@@@@@@@ Free memory @@@@@@@@@@/
	CUDA_ASSERT(cudaMemcpy(Estimate1, EstimateKernel, size, cudaMemcpyDeviceToHost));

	CUDA_ASSERT(cudaFree(EstimateKernel));

	
	// *ima=0, *means_Tex=0, *variances_Tex=0, *R_Tex=0;
	CUDA_ASSERT(cudaFreeArray(meansArray));
	CUDA_ASSERT(cudaFreeArray(variancesArray));
	CUDA_ASSERT(cudaFreeArray(RArray));


	CUDA_ASSERT(cudaFreeArray(ima));
	CUDA_ASSERT(cudaFree(mean.ptr));
	CUDA_ASSERT(cudaFree(R.ptr));
	CUDA_ASSERT(cudaFree(var.ptr));

	CUDA_ASSERT(cudaDeviceReset());
}
