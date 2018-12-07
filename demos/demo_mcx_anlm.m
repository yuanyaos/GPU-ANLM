%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           GPU-accelerated adaptive non-local means filter
%
% This example incorperates GPU-ANLM with Monte Carlo eXtreme (MCX). To run
% this demo, please download MCX first.
% MCX software is a GPU-accelerated photon transport simulator. It
% can be downloaded from: https://github.com/fangq/mcx

%% Configure MCXLAB

addpath(genpath('...')) % add path of MCX
addpath(genpath('../')) % add path of ANLM filter

clear cfg
cfg.nphoton=1e7;
cfg.vol=uint8(ones(100,100,100));
cfg.srcpos=[50 50 1];
cfg.srcdir=[0 0 1];
cfg.gpuid=1;
cfg.seed = randi([1 2^31-1], 1);
% cfg.gpuid='11'; % use two GPUs together
cfg.autopilot=1;
cfg.prop=[0 0 1 1;0.005 1 0 1.37];
cfg.tstart=0;
cfg.tend=5e-9;
cfg.tstep=5e-9;

%% Configure ANLM
v                    =           3;     % search radius
f1                   =           1;     % 1st filtering patch radius
f2                   =           2;     % 2nd filtering patch radius (f2>f1)
rician               =           0;     % rician=0: no rician noise. rician=1: rician noise
gpuid                =           1;     % GPU id in your computer
blockwidth           =           8;     % the 3D block width in GPU

%% Streamline: MC+filter

% run MCXLAB
tic
flux=mcxlab(cfg);
t_mc1 = toc
rima = sum(flux.data,4);

% run filter
tic;
[imaS1,imaL1]=ganlm(rima,v,f1,f2,rician,gpuid,blockwidth);
% Sub-band mixing process
image1=mixingsubband(imaS1,imaL1); % originally fimau1,fimao1
t_filter=toc

%% Equivalent photon number

cfg.seed = randi([1 2^31-1], 1);
cfg.nphoton = 3.5*1e7;  % photon number equivalent to MC+filter (3.5x)
% run MCXLAB
tic
flux2=mcxlab(cfg);
t_mc2 = toc;
rima2 = sum(flux2.data,4);

figure,
subplot(131),imagesc(squeeze(log10(rima(:,50,:))),[-4 8]),colormap jet, axis off
title(['Before filter (t_{mc}=' num2str(t_mc1) 's)'])
subplot(132),imagesc(squeeze(log10(image1(:,50,:))),[-4 8]),colormap jet, axis off
title(['After filter (t_{mc}+t_{filter}=' num2str(t_mc1+t_filter) 's)'])
subplot(133),imagesc(squeeze(log10(rima2(:,50,:))),[-4 8]),colormap jet, axis off
title(['Equivalent photon # (t_{filter}=' num2str(t_mc2) 's)'])