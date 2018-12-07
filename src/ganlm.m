% ---------------------------------------------------------------------
% 
%           GPU-accelerated adaptive non-local means filter
% 
% ---------------------------------------------------------------------
% Copyright (c) 2018 Yaoshen Yuan, Qianqian Fang
% ---------------------------------------------------------------------
% 
% Author:       Yaoshen Yuan and Qianqian Fang
% Webpage:      http://mcx.space
% Contact:      yuan.yaos at husky.neu.edu
%               q.fang at neu.edu
% 
% Format:
%     [ima1,ima2]=ganlm(rima,v,f1,f2); (default)
%     [ima1,ima2]=ganlm(rima,v,f1,f2,rician,gpuid,bw); (other options)
%     [ima1,~]=ganlm(rima,v,f1,0,rician,gpuid,bw); (single filtering)
% 
% Input:
% == Required ==
%     *rima: 3-D raw image data
%     *v:    search radius. The search region size is (2v+1)x(2v+1)x(2v+1)
%     *f1:   patch radius for 1st filter (f1<f2). The patch region size is (2f1+1)x(2f1+1)x(2f1+1)
%     *f2:   patch radius for 2nd filter (f1<f2). The patch region size is (2f2+1)x(2f2+1)x(2f2+1)
% 
% == Optional ==
%     rician: [0]-no rician noise (default), [1]-rician noise
%     gpuid:  GPU id for filtering
%     bw:     GPU block width for filter (default: 8). The GPU block size is bw x bw x bw.
%             The block width should satisfy bw+2*(v+f2)<=23 due to the limited GPU memory
% 
% Output:
%     ima1: filtered image using patch radius f1
%     ima2: filtered image using patch radius f2 (empty when using single filtering)