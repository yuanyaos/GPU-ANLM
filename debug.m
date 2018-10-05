clear

%%
addpath(genpath('.'))
addpath('/drives/neza2/users/yaoshen/NEU/Research/Redbird/tensorlab/')

%% write to dat
load brain_1e8.mat

fid = fopen('input.dat', 'w');
fwrite(fid, fluence_brain, 'single');
fclose(fid);


fid = fopen('input.dat', 'r');
At = fread(fid,'single');
fclose(fid);
At = reshape(At,166,209,223);
%% read from dat
fid=fopen('outputimg.dat','r');
A = fread(fid,'single');
fclose(fid);

output = reshape(A,166,209,223);
% figure,imagesc(squeeze(log10(output(:,80,:))),[-16 8]),colormap jet
figure,slice3(output)

fid=fopen('outputimg2.dat','r');
A = fread(fid,'single');
fclose(fid);

output2 = reshape(A,166,209,223);
figure,slice3(output2)