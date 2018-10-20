% Plot SNR and mean trend
% clear all

%% SNR trend 2 for 1000 trials
% Plot the SNR trend along z axis using 1000 trials
addpath gridLegend/
type = 'refractive';

PN = 100;    % sample number in one pack
NN = 1000;   % total sample number
z = 1:100;
variance_unfilter = zeros(4,100);
mean_unfilter = zeros(4,100);
pho = {'1e5','1e6','1e7','1e8'};
% pho = {'1e5', '1e6','1e7','1e8'};
for n=1:4
    for num=1:10
    %     str = ['/drives/neza2/users/yaoshen/NEU/Research/MRI filtering/mcxlab_nightlybuild/data/journal_50ns_vol100_' type '_1e' num2str(phocnt(n))];
        str = strcat('/drives/neza2/users/yaoshen/NEU/Research/MRI filtering/mcxlab_nightlybuild/data/journal2_50ns_vol100_', type, '_', pho(n),'_pack', num2str(num));
        load(str{1});
        for i=1:length(z)
            variance_unfilter(n,i) = variance_unfilter(n,i)+var(data(50,50,i,:))*(PN-1);
            mean_unfilter(n,i) = mean_unfilter(n,i)+sum(data(50,50,i,:));
        end
    end
end
variance_unfilter = variance_unfilter./(NN-1);
mean_unfilter = mean_unfilter./NN;
sigma_unfilter = sqrt(variance_unfilter);
SNR_unfilter = 20*log10(mean_unfilter./sigma_unfilter);

% variance_CPU = zeros(4,100);
% mean_CPU = zeros(4,100);
% for n=1:4
%     for num=1:10
%     %     str = ['/drives/neza2/users/yaoshen/NEU/Research/MRI filtering/GPU/GPU_ANLM/src/data/journal_beforelog_CPU_NOfilterregion_50ns_',type,'100_1e', num2str(phocnt(n)), '_V3F2F1'];   % CPU
%     %     str = strcat('/drives/neza2/users/yaoshen/NEU/Research/MRI filtering/GPU/GPU_ANLM/src/data/journal_beforelog_CPU_NOfilterregion_50ns_',type,'100_', pho(n), '_V3F2F1');   % CPU
%         str = strcat('/drives/neza2/users/yaoshen/NEU/Research/MRI filtering/GPU/GPU_ANLM/src/data/BM4D2_beforelog_NOfilterregion_50ns_',type,'100_', pho(n), '_pack',num2str(num));       % BM4D
%     %     str = strcat('/drives/neza2/users/yaoshen/NEU/Research/MRI filtering/GPU/GPU_ANLM/src/data/Gaussian_beforelog_NOfilterregion_50ns_',type,'100_', pho(n));
%         load(str{1});
%         for i=1:length(z)
%             variance_CPU(n,i) = variance_CPU(n,i)+var(abs(results(50,50,i,:)))*(PN-1);
%             mean_CPU(n,i) = mean_CPU(n,i)+sum(abs(results(50,50,i,:)));
%         end
%     end
% end
% variance_CPU = variance_CPU./(NN-1);
% mean_CPU = mean_CPU./NN;
% sigma_CPU = sqrt(variance_CPU);
% SNR_CPU = 20*log10(mean_CPU./sigma_CPU);

variance_GPU = zeros(4,100);
mean_GPU = zeros(4,100);
for n=1:4
    for num=1:10
    %     str = ['/drives/neza2/users/yaoshen/NEU/Research/MRI filtering/GPU/GPU_ANLM/src/data/journal_beforelog_GPU_NOfilterregion_50ns_',type,'100_1e', num2str(phocnt(n)), '_V3F2F1'];   % GPU
        str = strcat('/drives/neza2/users/yaoshen/NEU/Research/MRI filtering/GPU/GPU_ANLM/src/data/journal2__beforelog_GPU_NOfilterregion_50ns_',type,'100_', pho(n), '_V3F2F1','_pack', num2str(num));   % GPU
        load(str{1});
        for i=1:length(z)
            variance_GPU(n,i) = variance_GPU(n,i)+var(results(50,50,i,:))*(PN-1);
            mean_GPU(n,i) = mean_GPU(n,i)+sum(results(50,50,i,:));
        end
    end
end
variance_GPU = variance_GPU./(NN-1);
mean_GPU = mean_GPU./NN;
sigma_GPU = sqrt(variance_GPU);
SNR_GPU = 20*log10(mean_GPU./sigma_GPU);

%% Plot
% load SNR_data.mat
% load SNR_data_sphere.mat
% load SNR_data_zplane.mat

addpath gridLegend/
z = 1:100;
c_unfilter = ['r','b','k','m'];
c_CPU = ['-.r','-.b','-.k','-.m'];
c_GPU = ['r','b','k','m'];
figure,
for n=1:4
   p1(n)=plot(z,SNR_unfilter(n,:),c_unfilter(n),'LineWidth',1.5);
   hold on
%    t = SNR_CPU(n,:);
%    p2(n)=plot(1:2:100,t(1:2:100),[c_unfilter(n) '--'],'LineWidth',1.5);
%    hold on
   p3(n)=plot(z,SNR_GPU(n,:),[c_GPU(n) '.'],'LineWidth',1.5,'MarkerSize',7);
   hold on
end

axis([1,100,-40,100]);
xticks([1 20 40 60 80 100])
xticklabels({'0','20','40','60','80','100'})
lgd = legend([p1(1) p2(1) p3(1) p1(2) p1(3) p1(4)],'10^5 No filter', '10^5 BM4D filtered', '10^5 ANLM filtered', '10^6 No filter','10^7 No filter','10^8 No filter');
p4 = [p1(1) p3(1) p2(1) p1(2) p1(3) p1(4)];
gKey = {'10^5 original','10^5 with ANLM','10^5 with BM4D','10^6','10^7','10^8'};
gridLegend(p4,2,gKey,'Fontsize',16,'location','north');
xlabel('z axis (mm)'),ylabel('SNR (dB)')
text(0.55,0.82,'Effective region','FontSize',16,'Color','k')
set(gca,'FontSize',16,'FontName', 'Arial');
% boldify
grid on
grid minor
legend boxoff
set(gca,'Position',[0.16, 0.16, 0.7750, 0.810])
set(gcf,'paperpositionmode','auto')

%% SNR difference for 1e8 photons and 3 benchmarks before and after filtering using 1000 samples (for review)

clear all

% Plot the SNR trend along z axis using 1000 trials
addpath gridLegend/
type = {'homo', 'absorber3', 'refractive'};

PN = 100;    % sample number in one pack
NN = 1000;   % total sample number
z = 1:100;

for nt=1:3

    variance_unfilter = zeros(4,100);
    mean_unfilter = zeros(4,100);
    pho = {'1e5','1e6','1e7','1e8'};
    % pho = {'1e5', '1e6','1e7','1e8'};
    for n=4:4
        for num=1:10
        %     str = ['/drives/neza2/users/yaoshen/NEU/Research/MRI filtering/mcxlab_nightlybuild/data/journal_50ns_vol100_' type '_1e' num2str(phocnt(n))];
            str = strcat('/drives/neza2/users/yaoshen/NEU/Research/MRI filtering/mcxlab_nightlybuild/data/journal2_50ns_vol100_', type{nt}, '_', pho(n),'_pack', num2str(num));
            load(str{1});
            for i=1:length(z)
                variance_unfilter(n,i) = variance_unfilter(n,i)+var(data(50,50,i,:))*(PN-1);
                mean_unfilter(n,i) = mean_unfilter(n,i)+sum(data(50,50,i,:));
            end
        end
    end
    variance_unfilter = variance_unfilter./(NN-1);
    mean_unfilter = mean_unfilter./NN;
    sigma_unfilter = sqrt(variance_unfilter);
    SNR_unfilter(nt,:) = 20*log10(mean_unfilter(4,:)./sigma_unfilter(4,:));

    variance_GPU = zeros(4,100);
    mean_GPU = zeros(4,100);
    for n=4:4
        for num=1:10
        %     str = ['/drives/neza2/users/yaoshen/NEU/Research/MRI filtering/GPU/GPU_ANLM/src/data/journal_beforelog_GPU_NOfilterregion_50ns_',type,'100_1e', num2str(phocnt(n)), '_V3F2F1'];   % GPU
            str = strcat('/drives/neza2/users/yaoshen/NEU/Research/MRI filtering/GPU/GPU_ANLM/src/data/journal2__beforelog_GPU_NOfilterregion_50ns_',type{nt},'100_', pho(n), '_V3F2F1','_pack', num2str(num));   % GPU
            load(str{1});
            for i=1:length(z)
                variance_GPU(n,i) = variance_GPU(n,i)+var(results(50,50,i,:))*(PN-1);
                mean_GPU(n,i) = mean_GPU(n,i)+sum(results(50,50,i,:));
            end
        end
    end
    variance_GPU = variance_GPU./(NN-1);
    mean_GPU = mean_GPU./NN;
    sigma_GPU = sqrt(variance_GPU);
    SNR_GPU(nt,:) = 20*log10(mean_GPU(4,:)./sigma_GPU(4,:));

end
%% Plot
% load SNR_data.mat
% load SNR_data_sphere.mat
% load SNR_data_zplane.mat

SNR_diff = SNR_GPU-SNR_unfilter;

addpath gridLegend/
z = 1:100;
c_unfilter = ['g','k','r'];
figure,
for n=1:2:3
   p1(n)=plot(z,SNR_diff(n,:),c_unfilter(n),'LineWidth',1.5);
   hold on
end
legend('B1','B3')

axis([1,100,-0.5,8]);
xticks([1 20 40 60 80 100])
xticklabels({'0','20','40','60','80','100'})
% lgd = legend([p1(1) p2(1) p3(1) p1(2) p1(3) p1(4)],'10^5 No filter', '10^5 BM4D filtered', '10^5 ANLM filtered', '10^6 No filter','10^7 No filter','10^8 No filter');
% p4 = [p1(1) p3(1) p2(1) p1(2) p1(3) p1(4)];
% gKey = {'10^5 original','10^5 with ANLM','10^5 with BM4D','10^6','10^7','10^8'};
% gridLegend(p4,2,gKey,'Fontsize',16,'location','north');
xlabel('z axis (mm)'),ylabel('\Delta SNR (dB)')
% text(0.55,0.82,'Effective region','FontSize',16,'Color','k')
set(gca,'FontSize',16,'FontName', 'Arial');
% boldify
grid on
grid minor
legend boxoff
set(gca,'Position',[0.16, 0.16, 0.7750, 0.810])
set(gcf,'paperpositionmode','auto')

%% SNR improvement 2 before and after filtering using 1000 samples
PN = 100;    % sample number in one pack
NN = 1000;   % total sample number
type = 'homo';
pho = {'1e5','1e6','1e7','1e8'};

mean_unfilter = zeros(100,100,100,4);
variance_unfilter = zeros(100,100,100,4);
mean_GPU = zeros(100,100,100,4);
variance_GPU = zeros(100,100,100,4);
mean_BM4D = zeros(100,100,100,4);
variance_BM4D = zeros(100,100,100,4);

variance_GPU2 = zeros(4,100);
mean_GPU2 = zeros(4,100);
SNR_GPU2 = zeros(4,100);
z = 1:100;
for n=1:4
    for num=1:10
        str1 = strcat('/drives/neza2/users/yaoshen/NEU/Research/MRI filtering/mcxlab_nightlybuild/data/journal2_50ns_vol100_', type, '_', pho(n),'_pack',num2str(num));
        load(str1{1});
        mean_unfilter(:,:,:,n) = mean_unfilter(:,:,:,n)+sum(data,4);
        variance_unfilter(:,:,:,n) = variance_unfilter(:,:,:,n)+var(data,0,4)*(PN-1);
                
        str2 = strcat('/drives/neza2/users/yaoshen/NEU/Research/MRI filtering/GPU/GPU_ANLM/src/data/journal2__beforelog_GPU_NOfilterregion_50ns_',type,'100_', pho(n), '_V3F2F1','_pack',num2str(num));   % GPU
        load(str2{1});
        mean_GPU(:,:,:,n) = mean_GPU(:,:,:,n)+sum(results,4);
        variance_GPU(:,:,:,n) = variance_GPU(:,:,:,n)+var(results,0,4)*(PN-1);
        
        for i=1:length(z)
            variance_GPU2(n,i) = variance_GPU2(n,i)+var(results(50,50,i,:))*(PN-1);
            mean_GPU2(n,i) = mean_GPU2(n,i)+sum(results(50,50,i,:));
        end
        
        
        str3 = strcat('/drives/neza2/users/yaoshen/NEU/Research/MRI filtering/GPU/GPU_ANLM/src/data/BM4D2_beforelog_NOfilterregion_50ns_',type,'100_', pho(n), '_pack',num2str(num));   % BM4D
        load(str3{1});
        results = abs(results);
        mean_BM4D(:,:,:,n) = mean_BM4D(:,:,:,n)+sum(results,4);
        variance_BM4D(:,:,:,n) = variance_BM4D(:,:,:,n)+var(results,0,4)*(PN-1);
        
    end
end

mean_unfilter = mean_unfilter./NN;
variance_unfilter = variance_unfilter./(NN-1);
sigma_unfilter = sqrt(variance_unfilter);

mean_GPU = mean_GPU./NN;
variance_GPU = variance_GPU./(NN-1);
sigma_GPU = sqrt(variance_GPU);

mean_BM4D = mean_BM4D./NN;
variance_BM4D = variance_BM4D./(NN-1);
sigma_BM4D = sqrt(variance_BM4D);

mean_GPU2 = mean_GPU2./NN;
variance_GPU2 = variance_GPU2./(NN-1);
sigma_GPU2 = sqrt(variance_GPU2);

SNR_unfilter = 20*log10(mean_unfilter./sigma_unfilter);
SNR_GPU = 20*log10(mean_GPU./sigma_GPU);
SNR_BM4D = 20*log10(mean_BM4D./sigma_BM4D);
SNR_GPU2 = 20*log10(mean_GPU2./sigma_GPU2);

save SNR_homo SNR_unfilter SNR_GPU SNR_BM4D

%% z-plane SNR
load SNR_homo.mat
[Y,X,Z] = meshgrid(1:100,1:100,1:100);
src = [50,50,0];
SNR_unfilter = zeros(4,100);
SNR_GPU = zeros(4,100);


for d=1:100
    for n=1:4
        t1 = SNR1(:,:,d,n);
        SNR_unfilter(n,d) = mean(t1(~isnan(t1)));
        t2 = SNR2(:,:,d,n);
        SNR_GPU(n,d) = mean(t2(~isnan(t2)));
    end
end

save SNR_data_zplane SNR_unfilter SNR_GPU
%% SNR filtering improvement 2
clc
tau = 4;
% load SNR_homo
for n=1:4
%     SNRdiff = SNR_GPU(:,:,:,n)-SNR_unfilter(:,:,:,n);
% %     SNRimp_mean = mean(SNRdiff(~isnan(SNRdiff) & SNRdiff>tau))
% %     SNRimp_med = median(SNRdiff(~isnan(SNRdiff) & SNRdiff>tau))
%     SNRimp_mean = mean(SNRdiff(~isnan(SNRdiff)))
%     SNRimp_med = median(SNRdiff(~isnan(SNRdiff)))

    SNRdiff_BM4D = SNR_BM4D(:,:,:,n)-SNR_unfilter(:,:,:,n);
    SNRimp_mean = mean(SNRdiff_BM4D(~isnan(SNRdiff_BM4D) & SNRdiff_BM4D>tau))
    SNRimp_med = median(SNRdiff_BM4D(~isnan(SNRdiff_BM4D) & SNRdiff_BM4D>tau))
%     SNRimp_mean_BM4D = mean(SNRdiff_BM4D(~isnan(SNRdiff_BM4D)))
%     SNRimp_med_BM4D = median(SNRdiff_BM4D(~isnan(SNRdiff_BM4D)))
end


%% SNR photon improvement 2 using 1000 samples
PN = 100;    % sample number in one pack
NN = 1000;   % total sample number
type = 'homo';
pho = {'1e5','1e6','1e7','1e8'};
mean_1 = zeros(100,100,100);
variance_1 = zeros(100,100,100);
mean_2 = zeros(100,100,100);
variance_2 = zeros(100,100,100);
for n=1:3
    for num=1:10
        str1 = strcat('/drives/neza2/users/yaoshen/NEU/Research/MRI filtering/mcxlab_nightlybuild/data/journal2_50ns_vol100_', type, '_', pho(n),'_pack',num2str(num));
        load(str1{1});
        mean_1 = mean_1+sum(data,4);
        variance_1 = variance_1+var(data,0,4)*(PN-1);
                
        str2 = strcat('/drives/neza2/users/yaoshen/NEU/Research/MRI filtering/mcxlab_nightlybuild/data/journal2_50ns_vol100_', type, '_', pho(n+1),'_pack',num2str(num));
        load(str2{1});
        mean_2 = mean_2+sum(data,4);
        variance_2 = variance_2+var(data,0,4)*(PN-1);

    end
    mean_1 = mean_1./NN;
    variance_1 = variance_1./(NN-1);
    sigma_1 = sqrt(variance_1);
    
    mean_2 = mean_2./NN;
    variance_2 = variance_2./(NN-1);
    sigma_2 = sqrt(variance_2);
    
    SNR1(:,:,:,n) = 20*log10(mean_1./sigma_1);
    SNR2(:,:,:,n) = 20*log10(mean_2./sigma_2);
end

save SNR_photon SNR1 SNR2

%% SNR photon improvement 2
% tau = 0;
load SNR_photon
for n=1:3
    SNRdiff = SNR2(:,:,1:60,n)-SNR1(:,:,1:60,n);
    
%     SNRimp_mean = mean(SNRdiff(~isnan(SNRdiff) & SNRdiff>tau))
%     SNRimp_med = median(SNRdiff(~isnan(SNRdiff) & SNRdiff>tau))
%     SNRdiff = SNRdiff(1:50,1:50,1:50);
    SNRimp_mean = mean(SNRdiff(~isnan(SNRdiff)))
    SNRimp_med = median(SNRdiff(~isnan(SNRdiff)))
end


%% Mean trend 2 for 1000 samples
PN = 100;    % sample number in one pack
NN = 1000;   % total sample number

figure(10),

c_unfilter = ['g','b','r','m'];
c_GPU = [':g',':b',':r',':m'];
c1 = [0.8 1 0.8; 0.8 0.8 1; 1 0.8 0.8];
c2 = [0.6 1 0.6; 0.6 0.6 1; 1 0.6 0.6];
z = 1:100;
pho = '1e6';

type = {'homo','absorber3','refractive'};
interval = 0:10:100;
interval(1) = [];

mean_unfilter = zeros(3,100);
variance_unfilter = zeros(3,100);
mean_GPU = zeros(3,100);
variance_GPU = zeros(3,100);
mean_Gau = zeros(3,100);
variance_Gau = zeros(3,100);

for n=1:3
    for num=1:10
        str = strcat('/drives/neza2/users/yaoshen/NEU/Research/MRI filtering/mcxlab_nightlybuild/data/journal2_50ns_vol100_', type(n), '_', pho,'_pack',num2str(num));
        load(str{1});
        data2 = data;
        data2(data2<=0) = eps;
        for i=1:length(z)
            mean_unfilter(n,i) = mean_unfilter(n,i)+sum(squeeze(data(50,50,i,:)));
            variance_unfilter(n,i) = variance_unfilter(n,i)+var(squeeze(log10(data2(50,50,i,:))))*(PN-1);
        end
        
        str = strcat('/drives/neza2/users/yaoshen/NEU/Research/MRI filtering/GPU/GPU_ANLM/src/data/journal2__beforelog_GPU_NOfilterregion_50ns_',type(n),'100_', pho, '_V3F2F1','_pack',num2str(num));   % GPU
        load(str{1});
        results2 = results;
        results2(results2<=0) = eps;
        for i=1:length(z)
            mean_GPU(n,i) = mean_GPU(n,i)+sum(squeeze(results(50,50,i,:)));
            variance_GPU(n,i) = variance_GPU(n,i)+var(squeeze(log10(results2(50,50,i,:))))*(PN-1);
        end

    end
end
mean_unfilter = mean_unfilter./NN;
variance_unfilter = variance_unfilter./(NN-1);
mean_GPU = mean_GPU./NN;
variance_GPU = variance_GPU./(NN-1);

sigma_unfilter = sqrt(variance_unfilter);
sigma_GPU = sqrt(variance_GPU);

for num=1:10
    str = strcat('/drives/neza2/users/yaoshen/NEU/Research/MRI filtering/GPU/GPU_ANLM/src/data/Gaussian_beforelog_NOfilterregion_50ns_refractive100_', pho, '_pack',num2str(num));   % Gaussian
    load(str)
    results2 = results;
    results2(results2<=0) = eps;
    for i=1:length(z)
        mean_Gau(n,i) = mean_Gau(n,i)+sum(squeeze(results(50,50,i,:)));
        variance_Gau(n,i) = variance_Gau(n,i)+var(squeeze(log10(results2(50,50,i,:))))*(PN-1);
    end
end
mean_Gau = mean_Gau./NN;
variance_Gau = variance_Gau./(NN-1);
sigma_Gau = sqrt(variance_Gau);

save mean_data mean_unfilter mean_GPU mean_Gau sigma_unfilter sigma_GPU sigma_Gau
%%
load mean_data.mat

y_unfilter1 = log10(mean_unfilter)+sigma_unfilter;
y_unfilter2 = log10(mean_unfilter)-sigma_unfilter;
y_GPU1 = log10(mean_GPU)+sigma_GPU;
y_GPU2 = log10(mean_GPU)-sigma_GPU;

type = {'homo','absorber3','refractive'};
pho = '1e6';
c_unfilter = ['g','b','r','m'];
c_GPU = [':g',':b',':r',':m'];
c1 = [0.8 1 0.8; 0.8 0.8 1; 1 0.8 0.8];
c2 = [0.6 1 0.6; 0.6 0.6 1; 1 0.6 0.6];
z = 1:100;

for n=1:3
    figure(10),fill([1:100, 100:-1:1],[(y_unfilter1(n,:)), fliplr((y_unfilter2(n,:)))],c1(n,:),'LineStyle','none')
    hold on
    figure(10),fill([1:100, 100:-1:1],[y_GPU1(n,:), fliplr(y_GPU2(n,:))],c2(n,:),'LineStyle','none')
    hold on
    figure(10),
    p1(n)=plot(z,log10(mean_unfilter(n,:)),c_unfilter(n),'LineWidth',1);
    alpha(0.4)
    hold on
    figure(10),
    p3(n)=plot(z,log10(mean_GPU(n,:)),c_GPU((n-1)*2+1:n*2),'LineWidth',2);
    alpha(0.4)
    hold on
end

figure(10),
p4 =plot(z,log10(mean_Gau(3,:)),'--k','LineWidth',2);
hold on

axis([1 100 -14 8])
% axis([40 60 -3 3])
yticks([-12 -8 -4 0 4 8])
yticklabels({'10^{-12}','10^{-8}','10^{-4}','10^0','10^4','10^8'})
xticks([1 20 40 60 80 100])
xticklabels({'0','20','40','60','80','100'})
lgd=legend([p1(1) p3(1) p1(2) p3(2) p1(3) p3(3) p4],'B1', 'B1 ANLM', 'B2','B2 ANLM','B3','B3 ANLM','B3 Gaussian','Location','best');
% title('Mean trends');
xlabel('z axis (mm)'),ylabel('Fluence rate (W\cdot mm^{-2})')
% boldify
grid on
grid minor
set(gca,'FontSize',15,'FontName', 'Arial')
lgd.FontSize = 15;
legend boxoff
set(gca,'Position',[0.17, 0.155, 0.768, 0.785])
set(gcf,'paperpositionmode','auto')

rectangle('Position',[45 -2.3 10 3])
hold on

% inset
figure(10),axes('Position',[0.67,0.63, 0.24, 0.28])
c_unfilter = ['g','b','r','m'];
c_GPU = [':g',':b',':r',':m'];
c1 = [0.8 1 0.8; 0.8 0.8 1; 1 0.8 0.8];
c2 = [0.6 1 0.6; 0.6 0.6 1; 1 0.6 0.6];
for n=1:3
    figure(10),fill([1:100, 100:-1:1],[(y_unfilter1(n,:)), fliplr((y_unfilter2(n,:)))],c1(n,:),'LineStyle','none')
    hold on
    figure(10),fill([1:100, 100:-1:1],[y_GPU1(n,:), fliplr(y_GPU2(n,:))],c2(n,:),'LineStyle','none')
    hold on
    figure(10),
    p1(n)=plot(z,log10(mean_unfilter(n,:)),c_unfilter(n),'LineWidth',1);
    alpha(0.5)
    hold on
    figure(10),
    p3(n)=plot(z,log10(mean_GPU(n,:)),c_GPU((n-1)*2+1:n*2),'LineWidth',2);
    alpha(0.5)
    hold on
end
figure(10),
p4 =plot(z,log10(mean_Gau),'--k','LineWidth',2);
hold on

% axis([0 100 -14 8])
axis([45 55 -2.3 0.7])
set(gca, 'XTickLabel', [])
set(gca, 'YTickLabel', [])
% legend([p1(1) p3(1) p1(2) p3(2) p1(3) p3(3) p4],'B1', 'B1 ANLM', 'B2','B2 ANLM','B3','B3 ANLM','B3 Gaussian','Location','best');
% title('Mean trends');
% xlabel('z axis (mm)'),ylabel('Fluence rate (W\cdot mm^{-2})')
% boldify
grid on
grid minor

set(gcf,'paperpositionmode','auto')
%% 3D SNR plot
type = 'homo';
[X,Y] = meshgrid(1:100,1:100);
pho = {'1e6','4x1e6','16x1e6','64x1e6'};
% pho = {'1e5', '1e6','1e7','1e8'};

str = strcat('/drives/neza2/users/yaoshen/NEU/Research/MRI filtering/mcxlab_nightlybuild/data/journal_50ns_vol100_', type, '_', pho(2));
load(str{1});
variance_unfilter = squeeze(var(data(:,50,:,:),0,4));
mean_unfilter = squeeze(mean(data(:,50,:,:),4));
sigma_unfilter = sqrt(variance_unfilter);


str = strcat('/drives/neza2/users/yaoshen/NEU/Research/MRI filtering/GPU/GPU_ANLM/src/data/journal_beforelog_GPU_NOfilterregion_50ns_',type,'100_', pho(2), '_V3F2F1');   % GPU
load(str{1});
variance_GPU = squeeze(var(results(:,50,:,:),0,4));
mean_GPU = squeeze(mean(results(:,50,:,:),4));
sigma_GPU = sqrt(variance_GPU);


figure,
surf(X,Y,20*log10(mean_unfilter./sigma_unfilter))
hold on
s = surf(X,Y,20*log10(mean_GPU./sigma_GPU),'FaceAlpha',0.9)
s.EdgeColor = 'none';
% 
% legend([p1(1) p3(1) p1(2) p1(3) p1(4)],'10^6 No filter', '10^6 ANLM filtered', '4\times10^6 No filter','16\times10^6 No filter','64\times10^6 No filter');
% % title('SNR trend');
% xlabel('z axis (mm)'),ylabel('SNR (dB)')
% text(0.55,0.82,'Effective region','FontSize',13,'FontWeight','bold','Color','k')
% boldify
% grid on
% grid minor


%% SNR for multi-filtering
type = 'homo';

z = 1:100;
variance_GPU = zeros(10,100);
mean_GPU = zeros(10,100);
for n=1:10
    str = strcat('/drives/neza2/users/yaoshen/NEU/Research/MRI filtering/GPU/GPU_ANLM/src/data/journal_beforelog_GPU_NOfilterregion_50ns_',type,'_multifilter_4x1e6_V3F2F1','m',num2str(n));
    load(str);
    for i=1:length(z)
        variance_GPU(n,i) = var(results(50,50,i,:));
        mean_GPU(n,i) = mean(results(50,50,i,:));
    end
end
sigma_GPU = sqrt(variance_GPU);

figure,
for n=1:10
   p1(n)=plot(z,20*log10(mean_GPU(n,:)./sigma_GPU(n,:)));
   hold on
end

axis([1,100,-20,100]);
legend([p1(1) p1(2) p1(3) p1(4) p1(5) p1(6) p1(7) p1(8) p1(9) p1(10)],'1','2','3','4','5','6','7','8','9','10');
title('SNR trend');
xlabel('z axis (mm)'),ylabel('SNR (dB)')
% text(0.55,0.82,'Effective region','FontSize',13,'FontWeight','bold','Color','k')
boldify
grid on
grid minor

% Draw SNR improvement
figure,
diffSNR = zeros(9,100);
for n=1:9
   diffSNR(n,:) = 20*log10(mean_GPU(n+1,:)./sigma_GPU(n+1,:))-20*log10(mean_GPU(1,:)./sigma_GPU(1,:));
   p2(n)=plot(z,diffSNR(n,:));
   hold on
end
axis([1,100,0,10]);
legend([p2(1) p2(2) p2(3) p2(4) p2(5) p2(6) p2(7) p2(8) p2(9)],'2-1','3-1','4-1','5-1','6-1','7-1','8-1','9-1','10-1');
title('SNR improvement (n-1)');
xlabel('z axis (mm)'),ylabel('SNR (dB)')
boldify
grid on

% Average SNR improvement
cutoff = 20;
meanSNR = diffSNR(:,cutoff:end);
mean(meanSNR,2)'

% mean trend
figure,
for n=1:10
   p3(n)=semilogy(z,mean_GPU(n,:));
   hold on
end

legend([p3(1) p3(2) p3(3) p3(4) p3(5) p3(6) p3(7) p3(8) p3(9) p3(10)],'1','2','3','4','5','6','7','8','9','10');
title('Mean trend');
xlabel('z axis (mm)'),ylabel('Fluence rate (W\cdot m^{-2})')
boldify
grid on
grid minor
