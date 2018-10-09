%% Generate new volume
clear all
close all
clc

volume = uint8(ones(100,100,100));
% [Y,X] = meshgrid(1:100,1:100);
% Z1 = ones(size(X),'uint8');
% Z1((X-40).^2+(Y-50).^2<25 | (X-60).^2+(Y-50).^2<25) = 2;

% N = 80;
% for i=1:N
%     volume(:,:,10+i) = Z1;
% end

for i=30:2:70
    for j=30:2:70
        volume(i,j,10:50) = 2;
    end
end
figure,imagesc(squeeze(volume(:,:,50))), axis equal, axis off

P = 1e7;
pho_cnt = [P, 10*P, 100*P];    % photon count shot from the bottom
for k=1:1
    
    clear cfg
    cfg.nphoton=pho_cnt(k);
    cfg.vol=uint8(volume);
    cfg.srcpos=[50 50 1];
    cfg.srcdir=[0 0 1];
    cfg.gpuid=1;
    % cfg.gpuid='11'; % use two GPUs together
    cfg.autopilot=1;
    cfg.unitinmm = 1;
%     cfg.isnormalized=0;

%     refractive: 0.02 10 0.9 2*1.37    volume(30:70,30:70,10:50) = 2;
%     blood (633nm): 0.21 77.3 0.994 1.37
    cfg.prop=[0 0 1 1; 0.02 10 0.9 1.37; 0.21 77.3 0.994 1.37];    % [mua,mus,g,n]
    cfg.tstart=0;
    cfg.tend=5e-8; % OSA: 5e-8
    cfg.tstep=5e-8;
    tic
    [flux,detpos]=mcxlab(cfg);
    t = toc
    data(:,:,:,k) = flux.data(:,:,:,1);
    figure,imagesc(squeeze(log10(data(:,:,30,k))),[-16 8]),colormap jet, axis equal
    axis tight
    axis off
    c = colorbar;
    c.FontSize = 20;
    c.Location='northoutside';
end
% save review_blood_data

%% Filter volume
clear
close all
% clc
load blood_data.mat

v                    =           3;     % search radius
f1                   =           1;     % 1st filtering patch radius
f2                   =           2;     % 2nd filtering patch radius (f2>f1)
rician               =           0;     % rician=0: no rician noise. rician=1: rician noise
gpuid                =           1;     % GPU id in your computer
blockwidth           =           8;     % the 3D block width in GPU

filter_data = zeros(100,100,100,3);
for k=1:1
    ima = single(squeeze(data(:,:,:,k)));

    tic;
    % The output has the same order of f1 and f2
    [imaS1,imaL1]=ganlm(ima,v,f1,f2,rician,gpuid,blockwidth);
    t = toc;
    
    % Sub-band mixing process
    tic
    image1=mixingsubband(imaS1,imaL1); % originally fimau1,fimao1
    t_mix=toc;
    
    figure,imagesc(squeeze(log10(image1(:,:,30,k))),[-16 8]),colormap jet, axis equal
    axis tight
    axis off
    c = colorbar;
    c.FontSize = 20;
    c.Location='northoutside';
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

tic;
% The output has the same order of f1 and f2
[imaS1,imaL1]=ganlm(fluence_brain,v,f1,f2,rician,gpuid,blockwidth);
t = toc;

% Sub-band mixing process
tic
filtered_fluence_brain=mixingsubband(imaS1,imaL1); % originally fimau1,fimao1
t_mix=toc;


%% Plot brain
fluence_brain(fluence_brain==-1) = 0;
imaS1(imaS1==-1) = 0;
imaL1(imaL1==-1) = 0;

cut = 80;
figure,
subplot(221),imagesc(rot90(squeeze(log10((imaS1(:,cut,:))))),[-16 8]),colormap jet,axis equal
title(['Small ' num2str(cut)])
subplot(222),imagesc(rot90(squeeze(log10(abs(imaL1(:,cut,:))))),[-16 8]),colormap jet,axis equal
title('Large')

fluence_brain(fluence_brain==-1) = 0;
subplot(223),imagesc(rot90(squeeze(log10(fluence_brain(:,cut,:)))),[-16 8]),colormap jet,axis equal
title('Brain 1e8 photons')
subplot(224),imagesc(rot90(squeeze(log10(filtered_fluence_brain(:,cut,:)))),[-16 8]),colormap jet,axis equal
title('Filtered brain image')
% fprintf('Filter time=%fs mixing time=%fs  total time=%fs\n\n',t, t_mix, t+t_mix);

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