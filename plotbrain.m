clear
load('/drives/neza1/users/shijie/Projects/Iso2Mesh/brain2mesh/new_whole_head/19-5_fullhead_cfgsam1_2_maxvol_0.2.mat');
load('/drives/neza1/users/shijie/Projects/Iso2Mesh/brain2mesh/new_whole_head/src.mat'); % source postion and direction based on 10-20 system is pre-computed 
load('/drives/neza1/users/shijie/Projects/Iso2Mesh/brain2mesh/new_whole_head/det.mat');

%% simulation parameters
clear cfg_mmc

% prepare node and elem for mmc simulation
cfg_mmc.node=node;
cfg_mmc.elem=elem;
cfg_mmc.elem(:,5)=cfg_mmc.elem(:,5)+1;
cfg_mmc.nphoton=1e8;
cfg_mmc.seed=1648335518;
cfg_mmc.srcpos=srcpos+0.01*srcdir;  %[147.037002563477,89.4886016845703,187.197006225586]
cfg_mmc.srcdir=srcdir;
cfg_mmc.detpos=[detpos 1.5*ones(size(detpos,1),1)];
cfg_mmc.tstart=0;
cfg_mmc.tend=5e-9;
cfg_mmc.tstep=5e-10;
% cfg_mmc.prop=[0,0,1,1;0.02 1.0 0.89 1.37;0.02 1.0 0.89 1.37;0.02 1.0 0.89 1.37;0.02 1.0 0.89 1.37;0.02 1.0 0.89 1.37;0.02 1.0 0.89 1.37;];
% labeling,1-air,2-scalp,3-skull,4-csf,5-gray matter,6-white matter
cfg_mmc.prop=[0,0,1,1;0,0,1,1;0.019 7.0 0.89 1.37;0.019 7.0 0.89 1.37;0.004 0.009 0.89 1.37;0.02 9.0 0.89 1.37;0.08 40.9 0.84 1.37];
cfg_mmc.debuglevel='TP';
cfg_mmc.isreflect=1;
% cfg_mmc.outputtype='energy';
% cfg_mmc.basisorder=0;
cfg_mmc.method='elem';

%% run mmc simulation
% [phimmc,detmmc]=mmclab(cfg_mmc);
% save wholehead_phimmc_detmmc
%% prepare voxel domain for mcx simulation
clear cfg_mcx
% cfg_mcx.seed=hex2dec('623F9A9E');
cfg_mcx.seed = randi([1 2^31-1], 1, 1);
cfg_mcx.nphoton=1e8;

% set the interior air to 1 while the exterior air to 0
vol_logic=logical(vol);
vol_logic = imfill(vol_logic,[1 1 1],6);
new_vol=(vol+1).*vol_logic;
idx_0=find(new_vol==0);
idx_1=find(new_vol==1);
new_vol(idx_0)=1;
new_vol(idx_1)=0;
cfg_mcx.vol=new_vol;
clear vol_logic new_vol

cfg_mcx.srcpos=srcpos+0.01*srcdir;  %[147.037002563477,89.4886016845703,187.197006225586]
cfg_mcx.srcdir=srcdir;
cfg_mcx.issrcfrom0=1;
cfg_mcx.detpos=[detpos 1.5*ones(size(detpos,1),1)];

cfg_mcx.tstart=0;
cfg_mcx.tend=5e-9;
cfg_mcx.tstep=5e-10;
cfg_mcx.isreflect=1;

cfg_mcx.autopilot=1;
cfg_mcx.gpuid=2;

% labeling,1-air,2-gray matter,3-white matter,4-csf,5-skull,6-scalp
% cfg_mcx.prop=[0,0,1,1;0.02 1.0 0.89 1.37;0.02 1.0 0.89 1.37;0.02 1.0 0.89 1.37;0.02 1.0 0.89 1.37;0.02 1.0 0.89 1.37;0.02 1.0 0.89 1.37;];
cfg_mcx.prop=[0,0,1,1;0,0,1,1;0.02 9.0 0.89 1.37;0.08 40.9 0.84 1.37;0.004 0.009 0.89 1.37;0.019 7.0 0.89 1.37;0.019 7.0 0.89 1.37];

% save energy deposit
% cfg_mcx.outputtype='energy';

%% run mcx simulation
% tic
% [phimcx,detmcx]=mcxlab(cfg_mcx);
% t=toc

%% visualization and comparison
% clines=[-10:0.75:10];
% % [xx,yy]=meshgrid(1:size(cfg_mcx.vol,1),1:size(cfg_mcx.vol,3));
% % [xx,yy]=meshgrid(0:(size(cfg_mcx.vol,1)-1),0:(size(cfg_mcx.vol,3)-1));
% [xx,yy]=meshgrid(0.5:(size(cfg_mcx.vol,1)-0.5),0.5:(size(cfg_mcx.vol,3)-0.5));
% value=sum(phimmc.data,2);
% y_cut=89.5;   %very close to source plane
% [cutpos,cutvalue,facedata]=qmeshcut(cfg_mmc.elem(:,1:4),cfg_mmc.node,value,[0 y_cut 0; 0 y_cut 1; 1 y_cut 0]);
% phi_mmc=griddata(cutpos(:,1),cutpos(:,3),cutvalue,xx,yy);
% phi_mcx=sum(phimcx.data,4);
% 
% figure;
% contourf(log10(abs(squeeze(phi_mmc))),clines,'linestyle','--','color','w','linewidth',1);
% hold on;
% contour(log10(abs(squeeze(phi_mcx(:,y_cut+0.5,:))')),clines,'linestyle','--','color',[0.9100    0.4100    0.1700],'linewidth',1);
% colorbar('EastOutside');

% cross-cut view contour of tissue boundaries
% y_cut=89.5;
% figure;
% % imagesc(flipud(rot90(squeeze(cfg_mcx.vol(:,y_cut+0.5,:)))));
% axis equal;
% hold on;

%%
load brain_1e8_bg0.mat

y_cut=87.5;

figure;
imagesc(flipud(rot90(squeeze(log10(filtered_fluence_brain(:,y_cut+0.5,:))))),[-8 8]);
colormap jet
axis equal;
hold on;

yy_mesh=y_cut;      %choose the plane you want to plot
node_plot=cfg_mmc.node+0.5;  %for plot, use node+0.5 while for simulation, we should use node
% plane=[0 yy_mesh 0; 0 yy_mesh 1; 1 yy_mesh 0];
plane=[0 yy_mesh 0; 0 yy_mesh 1; 1 yy_mesh 0];
[cutpos,cutvalue,cutedges]=qmeshcut(face(:,1:3),node_plot,node_plot(:,1),plane);
[cutpos,cutedges]=removedupnodes(cutpos,cutedges);
cutloop=extractloops(cutedges);
[nanidx]=find(isnan(cutloop));
for i=1:size(nanidx,2)
    if(i==1)
        plot(cutpos(cutloop(1:(nanidx(i)-1)),1),cutpos(cutloop(1:(nanidx(i)-1)),3),'color','k','LineWidth',1);
        hold on
    else
        plot(cutpos(cutloop((nanidx(i-1)+1):(nanidx(i)-1)),1),cutpos(cutloop((nanidx(i-1)+1):(nanidx(i)-1)),3),'color','k','LineWidth',1);
        hold on
    end
end

axis equal;
colormap;
% set(gca,'ylim', [160 225]);
set(gca,'fontsize',20);
axis off