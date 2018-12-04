/*--------------------------------------------------------------------------*/
// A GPU-accelerated Adaptive Non-Local Means Filter for Denoising 3D Monte
// Carlo Photon Transport Simulations

// ANLMGPU.c is an interface connecting MATLAB and cuda files
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


#include "math.h"
#include "mex.h"
#include <stdlib.h>
#include <stdio.h>
#include "matrix.h"
#include "filterGPU.h"

typedef struct
{
    int cols;
    int rows;
    int slices;
    float * in_image;
    float * means_image;
    float * var_image;
    float * estimate1;
    float * estimate2;
    int rV;		// search size
    int rPl;		// larger patch radius
    int rPs;		// smaller patch size
    float max_image;
    int blockwidth;
    int blockwidth2;
    int s;
    int gpuid;
    bool R; 		// Rician noise
    int flag;
}myargument;

void passParam(myargument *arg)
{

	if(arg->flag==0){
    	// runFilter(float * ima, float * Estimate1, int f1, float * Estimate2, int f2, int v, int dimx, int dimy, int dimz, float MAX, int width, int width2, int s, int gpuid);
    	// Baseline+Opt1+Opt2+Opt3+Opt4
    	runFilter(arg->in_image, arg->estimate1, arg->rPl, arg->estimate2, arg->rPs, arg->rV, arg->cols, arg->rows, arg->slices, arg->max_image, arg->blockwidth2, arg->blockwidth, arg->s, arg->gpuid, arg->R);
    }
    else if(arg->flag==1){
    	// runFilter_v(float * ima_input, float * Estimate1, int f1, float * Estimate2, int f2, int v, int dimx, int dimy, int dimz, float MAX, int width2, int width, int s, int gpuid, bool rician)
    	// Baseline+Opt2+Opt3+Opt4 for larger volume
    	runFilter_v(arg->in_image, arg->estimate1, arg->rPl, arg->estimate2, arg->rPs, arg->rV, arg->cols, arg->rows, arg->slices, arg->max_image, arg->blockwidth2, arg->blockwidth, arg->s, arg->gpuid, arg->R);
    }
    else{
    	// runFilter_s(float * ima_input, float * Estimate1, int f1, int v, int dimx, int dimy, int dimz, float MAX, int width2, int width, int s, int gpuid, bool rician)
    	runFilter_s(arg->in_image, arg->estimate1, arg->rPs, arg->rV, arg->cols, arg->rows, arg->slices, arg->max_image, arg->blockwidth2, arg->blockwidth, arg->s, arg->gpuid, arg->R);
    }
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* Declarations */
    mxArray *xData;
    mxArray *Mxmeans, *Mxvariances, *Mxima_full, *Mxmeans_full, *Mxvariances_full, *MxR_full;
    float *ima;
    float *means, *variances, *Estimate1, *Estimate2, *float_ima, *means_full, *variances_full, *R_full;
    mxArray *pv;
    float max;
    int i,j,k,ii,jj,kk,ni,nj,nk,v,f1,f2,ndim,indice,Nthreads,ini,fin,blockwidth,blockwidth2,s,it,jt,kt,gpuid,request,flag;
    const int *dims;
    int dimfull[3];
    bool r;

    myargument filterParam;


    // Default value
    flag = 0;
    r = 0;
    blockwidth = 8;
    blockwidth2 = 8;
    s = 1;
    gpuid = 1;
    
    switch(nrhs){
	case 7:
		/* Block width for filtering */
		pv = (mxArray*)prhs[6];
		blockwidth2 = (int)(mxGetScalar(pv));
	case 6:
		/* gpuid */
		pv = (mxArray*)prhs[5];
		gpuid = (int)(mxGetScalar(pv));
		
	case 5:
		/* racian noise */
		pv = (mxArray*)prhs[4];
		r = (int)(mxGetScalar(pv));
		
	case 4:
		/* Input image */
		xData = (mxArray*)prhs[0];
		ima = (float *) mxGetPr(xData);   // Get the realistic volume data
		ndim = mxGetNumberOfDimensions(prhs[0]);    // number of dimension of the input volume (should be 3 because of 3D image)
		dims = mxGetDimensions(prhs[0]);     // Pointer pointing to the first element of the dimension arrays (e.g. [L M N])
		
		/* Search raidus */
		pv = (mxArray*)prhs[1];
		v = (int)(mxGetScalar(pv));
		
		/* Smaller patch radius */
		pv = (mxArray*)prhs[2];
		f2 = (int)(mxGetScalar(pv));
		
		/* Larger patch radius */
		pv = (mxArray*)prhs[3];
		f1 = (int)(mxGetScalar(pv));
    	break;
    }

    request = blockwidth2+2*(v+f1);
    printf("########################## GPU-accelerated Adaptive Non-local Means Filter ##########################");
    printf("\nsearch size=%d\n1st filtering patch size=%d\n2nd filtering patch size=%d\nrician=%d\nGPU id=%d\n3D block size=%d\n",2*v+1,2*f2+1,2*f1+1,r,gpuid,blockwidth2);

	if(request<=18){
		flag = 0;								// full version B+Opt1+Opt2+Opt3+Opt4
		printf("You are using the full version (B+2(v+f)=%d)\n\n",request);
	}
	else if(request>18 && request<=23){			// reduced version by removing Opt1
		flag = 1;
		printf("You are using the reduced version due to the large shared memory request (B+2(v+f)=%d)\n\n",request);
	}
	else{										// exceeds the shared memory limit 48KB
		printf("This error may caused by the improper GPU block-width setting or the oversize search size and patch size.\nGiven block-width (B), search radius (v) and patch radisu (f), they should satisfy: B+2*(v+f)<=23\n");
		mexErrMsgTxt("Error: shared memory request exceeds the limit!\n");
	}

	if(f1==0){
		f1 = f2;
		flag = 2;		// Second patch size is set to 0, which implies single filtering
	}

    plhs[0] = mxCreateNumericArray(ndim,dims,mxSINGLE_CLASS, mxREAL);   // Create a real double matrix with the size of input volume
    plhs[1] = mxCreateNumericArray(ndim,dims,mxSINGLE_CLASS, mxREAL);   // Create a real double matrix with the size of input volume
    Estimate1 = (float*) mxGetData(plhs[0]);
    Estimate2 = (float*) mxGetData(plhs[1]);

    max=0;
    // Size of image for PRE-PROCESSING at each dimension
    dimfull[0] = dims[0]+2*(f1+v+s);
    dimfull[1] = dims[1]+2*(f1+v+s);
    dimfull[2] = dims[2]+2*(f1+v+s);

    Mxima_full = mxCreateNumericArray(ndim,dimfull,mxSINGLE_CLASS, mxREAL);
    float_ima = (float*) mxGetData(Mxima_full);	// Full input image for pre-processing (Adding apron)

    // Mirroring the boundary
    for(k=0; k<dimfull[2]; k++)
    {
		kt = k-(f1+v+s);
		if(kt<0) kt=-kt-1;
		if(kt>=dims[2]) kt=2*dims[2]-kt-1;
		for(j=0; j<dimfull[1]; j++)
		{
		    jt = j-(f1+v+s);
		    if(jt<0) jt=-jt-1;
		    if(jt>=dims[1]) jt=2*dims[1]-jt-1;
		    for(i=0; i<dimfull[0]; i++)
		    {
				it = i-(f1+v+s);
				if(it<0) it=-it-1;
				if(it>=dims[0]) it=2*dims[0]-it-1;
				float_ima[k*dimfull[0]*dimfull[1]+j*dimfull[0]+i] = ima[kt*(dims[0]*dims[1])+(jt*dims[0])+it];

				if(ima[kt*(dims[0]*dims[1])+(jt*dims[0])+it]>max) max=ima[kt*(dims[0]*dims[1])+(jt*dims[0])+it];
		    }
		}
    }


	filterParam.cols=dims[0];
	filterParam.rows=dims[1];
	filterParam.slices=dims[2];
	filterParam.in_image=float_ima;
	filterParam.estimate1=Estimate1;
	filterParam.estimate2=Estimate2;
	filterParam.rV=v;
	filterParam.rPl=f1;
	filterParam.rPs=f2;
	filterParam.max_image=max;
	filterParam.blockwidth=blockwidth;
	filterParam.blockwidth2=blockwidth2;
	filterParam.s=s;
	filterParam.gpuid=gpuid;
	filterParam.R=r;
	filterParam.flag=flag;
	
	passParam(&filterParam);

    return;

}