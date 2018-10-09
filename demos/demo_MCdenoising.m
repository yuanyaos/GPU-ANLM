%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           GPU-accelerated adaptive non-local means filter
%
% This example denoises the Monte Carlo images with refractive inclusion


%%%%%%%%%% Read Monte-Carlo images %%%%%%%%%%
clear
close all
clc
addpath(genpath('../'))
load data

%%%%%%%%%% Parameter setting (recommended for high speed) %%%%%%%%%%
v                    =           3;     % search radius
f1                   =           1;     % 1st filtering patch radius
f2                   =           2;     % 2nd filtering patch radius (f2>f1)
rician               =           0;     % rician=0: no rician noise. rician=1: rician noise
gpuid                =           1;     % GPU id in your computer
blockwidth           =           8;     % the 3D block width in GPU


%% Refractive volume

% 1e6 photons/1e7 photons
tic;
% The output has the same order of f1 and f2
[imaS4,imaL4]=ganlm(ima_1e7_refractive,v,f1,f2,rician,gpuid,blockwidth);
t = toc;
% Sub-band mixing process
tic
image4=mixingsubband(imaS4,imaL4); % originally fimau1,fimao1
t_mix=toc;
figure,
subplot(121),imagesc(squeeze(log10(ima_1e7_refractive(:,50,:))),[-16 8]),colormap jet, axis off
title('Refractive 1e7 photons')
subplot(122),imagesc(squeeze(log10(image4(:,50,:))),[-16 8]),colormap jet, axis off
title('Filtered refractive 1e7 photons')
fprintf('Filter time=%fs mixing time=%fs  total time=%fs\n\n',t, t_mix, t+t_mix);