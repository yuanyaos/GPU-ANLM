%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           GPU-accelerated adaptive non-local means filter
%
% This example demonstrates the basic usage of our GPU-accelerated adaptive
% non-local means filter described in paper "A GPU-accelerated Adaptive 
% Non-Local Means Filter for Denoising 3D Monte Carlo Photon Transport 
% Simulations."
%
%
% ******************************** NOTICE ********************************
%             Please make sure you have a proper parameter setting:
%
%             if     blockwidth+2*(v+f2)<=18
%                         Execute the full version
%                         (Baseline+Opt1+Opt2+Opt3+Opt4)
%
%             elseif 18<blockwidth+2*(v+f2)<=23
%                         Execute the reduced version due to the limited 
%                         shared memory (Baseline+Opt2+Opt3+Opt4)
%
%             else
%
%                         Error due to oversize shared memory request
%
%             end
% ************************************************************************

%%%%%%%%%% Read Monte-Carlo images %%%%%%%%%%
clear
close all
clc
addpath(genpath('../'))
load data

%% GPU-accelerated ANLM
%%%%%%%%%% Parameter setting (recommended for high speed) %%%%%%%%%%
% Baseline+Opt1+Opt2+Opt3+Opt4
v                    =           3;     % search radius
f1                   =           1;     % 1st filtering patch radius
f2                   =           2;     % 2nd filtering patch radius (f2>f1)
rician               =           0;     % rician=0: no rician noise. rician=1: rician noise
gpuid                =           1;     % GPU id in your computer
blockwidth           =           8;     % the 3D block width in GPU

% 1e7 photons
tic;
% The output has the same order of f1 and f2
[imaS1,imaL1]=ganlm(ima_1e7_refractive,v,f1,f2,rician,gpuid,blockwidth);
t = toc;
% Sub-band mixing process
tic
image1=mixingsubband(imaS1,imaL1); % originally fimau1,fimao1
t_mix=toc;

figure,
subplot(131),imagesc(squeeze(log10(ima_1e7_refractive(:,50,:))),[-16 8]),colormap jet, axis off
title('Refractive 1e7 photons')
subplot(132),imagesc(squeeze(log10(image1(:,50,:))),[-16 8]),colormap jet, axis off
title('Filtered image (v=3, full version)')
fprintf('Filter time=%fs mixing time=%fs  total time=%fs\n\n',t, t_mix, t+t_mix);



%% Larger search or patch size
%%%%%%%%%% Parameter setting 18<blockwidth+2*(v+f2)<=23 %%%%%%%%%%
% Baseline+Opt2+Opt3+Opt4
v                    =           5;     % search radius
f1                   =           1;     % 1st filtering patch radius
f2                   =           2;     % 2nd filtering patch radius (f2>f1)
rician               =           0;     % rician=0: no rician noise. rician=1: rician noise
gpuid                =           1;     % GPU id in your computer
blockwidth           =           8;     % the 3D block width in GPU

% 1e7 photons
tic;
% The output has the same order of f1 and f2
[imaS2,imaL2]=ganlm(ima_1e7_refractive,v,f1,f2,rician,gpuid,blockwidth);
t = toc;

% Sub-band mixing process
tic
image2=mixingsubband(imaS2,imaL2); % originally fimau1,fimao1
t_mix=toc;
subplot(133),imagesc(squeeze(log10(image2(:,50,:))),[-16 8]),colormap jet, axis off
title('Filtered image (v=5, reduced version)')
fprintf('Filter time=%fs mixing time=%fs  total time=%fs\n\n',t, t_mix, t+t_mix);



%% Single filtering
% It is not indispensable to have 2 filterings and sub-band mixing process
% if filtering time is your concern. The single filtering is also provided
% in our function. To use single filtering, simply set f2=0.

%%%%%%%%%% Parameter setting %%%%%%%%%%
% Baseline+Opt2+Opt3
v                    =           3;     % search radius
f1                   =           2;     % 1st filtering patch radius
f2                   =           0;     % 2nd filtering patch radius (f2=0)
rician               =           0;     % rician=0: no rician noise. rician=1: rician noise
gpuid                =           1;     % GPU id in your computer
blockwidth           =           8;     % the 3D block width in GPU

% 1e7 photons
tic;
% The output has the same order of f1 and f2
[imaS3,imaL3]=ganlm(ima_1e7_refractive,v,f1,0,rician,gpuid,blockwidth);
t = toc

figure,
subplot(121),imagesc(squeeze(log10(ima_1e7_refractive(:,50,:))),[-16 8]),colormap jet, axis off
title('Refractive 1e7 photons')
subplot(122),imagesc(squeeze(log10(imaS3(:,50,:))),[-16 8]),colormap jet, axis off
title('Filtered image (single filtering)')
