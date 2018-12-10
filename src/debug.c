#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "debug.h"
#include "filterGPU.h"
#include <stdbool.h>

int main(int argc, char **argv){
    int searchareasize=3, patchsize=2, patchsize2=1, s=1, count=0;
    bool isrician=0;
    char filename[255]={'\0'};
    int dims[3]={166,209,223};
    float *inputimg, *outputimg, *outputimg2;
    FILE *fpread, *fp;
    if(argc>1)
        memcpy(filename,argv[1],255);
    else if(argc>2)
        searchareasize=atoi(argv[2]);
    else if(argc>3)
        patchsize=atoi(argv[3]);
    else if(argc>4)
        isrician=atoi(argv[4]);

    /*Copy inputimg pointer x*/
    inputimg = (float *)calloc(dims[0]*dims[1]*dims[2],sizeof(float));  /* The pointer pointing to inputimg volume*/
    if((fpread=fopen(filename,"rb"))==NULL)
       exit(-1);
    fread((void*)inputimg,dims[0]*dims[1]*dims[2],sizeof(float),fpread);
    fclose(fpread);

 //    for(int i=0;i<dims[0]*dims[1]*dims[2];i++){
 //        if(i>3490300 && i<3490400)
 //    	   printf("input=%f\n",inputimg[i]);
 //        count++;
	// }
    printf("count=%d\n",count);

    filterdriver(searchareasize,patchsize,patchsize2,s,isrician,dims,inputimg, &outputimg, &outputimg2);

    fp=fopen("outputimg.dat","wr");
    fwrite(outputimg,sizeof(float),dims[0]*dims[1]*dims[2],fp);
    fclose(fp);
    
    fp=fopen("outputimg2.dat","wr");
    fwrite(outputimg2,sizeof(float),dims[0]*dims[1]*dims[2],fp);
    fclose(fp);

    return 0;
}

void filterdriver(int v,int f1, int f2, int s, bool r, int dims[3], float *ima, float **outputimg, float **outputimg2)
{
    /*Declarations*/
    float *ima_full, *Estimate, *Estimate2;
    float max;
    int i,j,k,width,it,jt,kt,gpuid;
    int dimfull[3];

    width = 8;
    gpuid = 1;
    dimfull[0] = dims[0]+2*(f1+v+s);
    dimfull[1] = dims[0]+2*(f1+v+s);
    dimfull[2] = dims[0]+2*(f1+v+s);

    /*Allocate memory and assign outputimg pointer*/
    outputimg[0] = (float*)calloc(dims[0]*dims[1]*dims[2],sizeof(float));   /* Create a real double matrix with the size of inputimg volume*/
    outputimg2[0] = (float*)calloc(dims[0]*dims[1]*dims[2],sizeof(float));   /* Create a real double matrix with the size of inputimg volume*/

    ima_full = (float*)malloc((dimfull[0]*dimfull[1]*dimfull[2])*sizeof(float));

    /*Get a pointer to the data space in our newly allocated memory*/
    Estimate = outputimg[0];
    Estimate2 = outputimg2[0];


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
                ima_full[k*dimfull[0]*dimfull[1]+j*dimfull[0]+i] = ima[kt*(dims[0]*dims[1])+(jt*dims[0])+it];

                if(ima[kt*(dims[0]*dims[1])+(jt*dims[0])+it]>max) max=ima[kt*(dims[0]*dims[1])+(jt*dims[0])+it];

                    // if(i>90 && i<100 && j==100 && k==100)
                    //     printf("x=%d \t y=%d \t z=%d\t value=%f\n", i, j, k, ima_full[k*dimfull[0]*dimfull[1]+j*dimfull[0]+i]);
            }
        }
    }


    // runFilter_s(ima_full, Estimate, f1, v, dims[0], dims[1], dims[2], max, width, width, s, gpuid, r);
    runFilter(ima_full, Estimate, f1, Estimate2, f2, v, dims[0], dims[1], dims[2], max, width, width, s, gpuid, r);
    //runFilter(float * ima_input, float * Estimate1, int f1, float * Estimate2, int f2, int v, int dimx, int dimy, int dimz, float MAX, int width2, int width, int s, int gpuid, bool rician)

    return;

}