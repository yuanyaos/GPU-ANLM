%% Generate new volume
clear all
close all
clc

volume = uint8(ones(100,100,100));

N = 10;
for i=1:N
    volume(30+4*(i-1):30+4*(i-1)+1,30:70,10:50) = 2;
end
figure,imagesc(squeeze(volume(:,50,:)))

P = 1e6;
pho_cnt = [P, 10*P, 100*P];    % photon count shot from the bottom
data = zeros(100,100,100,3);
for k=1:3
    
    clear cfg
    cfg.nphoton=pho_cnt(k);
    cfg.vol=volume;
    cfg.srcpos=[50 50 1];
    cfg.srcdir=[0 0 1];
    cfg.gpuid=1;
    % cfg.gpuid='11'; % use two GPUs together
    cfg.autopilot=1;
%     cfg.isnormalized=0;

%     Version 2 (journal):
%     no absorber: 0.02 10 0.9 1.37
%     absorber3: 0.1 10 0.9 1.37      volume(20:40,20:40,10:30) = 2; 5x mu_a
%     refractive: 0.02 10 0.9 6.85    volume(30:70,30:70,10:50) = 2;
%     blood (633nm): 0.21 77.3 0.994 1.37
    cfg.prop=[0 0 1 1; 0.02 10 0.9 1.37; 0.21 77.3 0.994 1.37];    % [mua,mus,g,n]
    cfg.tstart=0;
    cfg.tend=5e-8; % OSA: 5e-8
    cfg.tstep=5e-8;
    tic
    [flux,detpos]=mcxlab(cfg);
    t = toc
    data(:,:,:,k) = flux.data(:,:,:,1);
    figure,imagesc(squeeze(log10(data(:,50,:,k)))),colormap jet, axis equal
end
% save review_blood_data

%% Filter volume
clear
close all
% clc
load review_data.mat

v                    =           3;     % search radius
f1                   =           1;     % 1st filtering patch radius
f2                   =           2;     % 2nd filtering patch radius (f2>f1)
rician               =           0;     % rician=0: no rician noise. rician=1: rician noise
gpuid                =           1;     % GPU id in your computer
blockwidth           =           8;     % the 3D block width in GPU

filter_data = zeros(100,100,100,3);
for k=1:3
    ima = single(squeeze(data(:,:,:,k)));

    tic;
    % The output has the same order of f1 and f2
    [imaS1,imaL1]=ganlm(ima,v,f1,f2,rician,gpuid,blockwidth);
    t = toc;
    
    % Sub-band mixing process
    tic
    image1=mixingsubband(imaS1,imaL1); % originally fimau1,fimao1
    t_mix=toc;
    
    filter_data(:,:,:,k) = image1;


    
    figure,
    subplot(121),imagesc(squeeze(log10(ima(:,50,:))),[-16 8]),colormap jet, axis off
    title('Refractive 1e7 photons')
    subplot(122),imagesc(squeeze(log10(image1(:,50,:))),[-16 8]),colormap jet, axis off
    title('Filtered image (v=3, full version)')
    fprintf('Filter time=%fs mixing time=%fs  total time=%fs\n\n',t, t_mix, t+t_mix);

end
% save review_filtered_blood_data filter_data

%% Filter brain
addpath(genpath('.'))
% close all
clear
clc

addpath(genpath('/drives/neza1/users/shijie/Projects/Iso2Mesh/brain2mesh'))
addpath('/drives/neza2/users/yaoshen/NEU/Research//Redbird/tensorlab/')
addpath(genpath('.'))

% Reading in data from c1-c5 SPM segmentations using a 4D array
for i = 1:5
    A = load_nii(sprintf('/drives/neza1/users/shijie/Projects/Iso2Mesh/brain2mesh/examples/SPM/c%iANTS19-5Years_head.nii.gz',i));
    dim = size(A.img);
    data(:,:,:,i) = A.img;
end

% Translate the probablity map to voxel domain
[vol]=brain2grid(data);

% vol1 = imgaussfilt3(vol);
% vol2 = imgaussfilt3(vol,1);
% vol_filtered=mixingsubband(vol1,vol2);

% Start filtering
load brain_1e8.mat

v                    =           3;     % search radius
f1                   =           1;     % 1st filtering patch radius
f2                   =           2;     % 2nd filtering patch radius (f2>f1)
rician               =           0;     % rician=0: no rician noise. rician=1: rician noise
gpuid                =           1;     % GPU id in your computer
blockwidth           =           8;     % the 3D block width in GPU

% Adding offset
fluence_brain(vol==0) = -1;
% s = size(fluence_brain);
% offset = 10;
% t = -1*ones(2*offset+s,'single');
% t(offset:offset+s(1)-1,offset:offset+s(2)-1,offset:offset+s(3)-1) = fluence_brain;
% fluence_brain = t;
% fluence_brain = fluence_brain(1:100,1:100,1:100);

tic;
% The output has the same order of f1 and f2
[imaS1,imaL1]=ganlm(fluence_brain,v,f1,f2,rician,gpuid,blockwidth);
t = toc;

% Sub-band mixing process
tic
filtered_fluence_brain=mixingsubband(imaS1,imaL1); % originally fimau1,fimao1
t_mix=toc;

% Remove offset
% fluence_brain = fluence_brain(offset:offset+s(1)-1,offset:offset+s(2)-1,offset:offset+s(3)-1);
% filtered_fluence_brain = filtered_fluence_brain(offset:offset+s(1)-1,offset:offset+s(2)-1,offset:offset+s(3)-1);

% Plot brain

cut = 80;
imaS1(imaS1<0) = 0;
imaL1(imaL1<0) = 0;
imaS1(imaS1>0) = 1;
imaL1(imaL1>0) = 1;
fluence_brain(fluence_brain>=0) = 1;
fluence_brain(fluence_brain<0) = 0;
figure,
subplot(221),imagesc(rot90(squeeze(log10(imaS1(:,cut,:)))),[-16 8]),colormap jet,axis equal
title(['Small ' num2str(cut)])
subplot(222),imagesc(rot90(squeeze(log10(imaL1(:,cut,:)))),[-16 8]),colormap jet,axis equal
title('Large')

fluence_brain(fluence_brain==-1) = 0;
subplot(223),imagesc(rot90(squeeze(log10(fluence_brain(:,cut,:)))),[-16 8]),colormap jet,axis equal
title('Brain 1e8 photons')
subplot(224),imagesc(rot90(squeeze(log10(filtered_fluence_brain(:,cut,:)))),[-16 8]),colormap jet,axis equal
title('Filtered brain image')
fprintf('Filter time=%fs mixing time=%fs  total time=%fs\n\n',t, t_mix, t+t_mix);

%%

% cut = 90;
binS = logical(imaS1);
binL = logical(imaL1);
binFluence = logical(fluence_brain);
binFiltered = logical(filtered_fluence_brain);

figure,
subplot(231),imagesc(rot90(squeeze((binFluence(:,cut,:))))),colormap jet
title('Original')
subplot(232),imagesc(rot90(squeeze((binS(:,cut,:))))),colormap jet
title(['Small ' num2str(cut)])
subplot(233),imagesc(rot90(squeeze((binL(:,cut,:))))),colormap jet
title('Large')

subplot(234),imagesc(rot90(squeeze((binS(:,cut,:)-binFluence(:,cut,:)))),[-1 1]),colormap jet
title('Small-raw')
subplot(235),imagesc(rot90(squeeze((binL(:,cut,:)-binFluence(:,cut,:)))),[-1 1]),colormap jet
title('Large-raw')
subplot(236),imagesc(rot90(squeeze((binL(:,cut,:)-binS(:,cut,:)))),[-1 1]),colormap jet
title('Large-Small')

% figure,slice3(log10(fluence_brain)),colormap jet
% figure,slice3(log10(filtered_fluence_brain)),colormap jet