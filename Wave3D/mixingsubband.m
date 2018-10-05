function [fima]=mixingsubband(fimau,fimao)                                                    
% Developed by Jose V. Manjon and Pierrick Coupe
% Modified by Yaoshen Yuan

% Adding mask
fimau(fimau<=0) = nan;
fimao(fimao<=0) = nan;

s = size(fimau);

p(1) = 2^(ceil(log2(s(1))));
p(2) = 2^(ceil(log2(s(2))));
p(3) = 2^(ceil(log2(s(3))));

pad1 = zeros(p(1),p(2),p(3));
pad2 = pad1;
pad1(1:s(1),1:s(2),1:s(3)) = fimau(:,:,:);
pad2(1:s(1),1:s(2),1:s(3)) = fimao(:,:,:); 

[af, sf] = farras;
w1 = dwt3D(pad1,1,af);
w2 = dwt3D(pad2,1,af);
  
w1{1}{3} = w2{1}{3};
w1{1}{5} = w2{1}{5};
w1{1}{6} = w2{1}{6};
w1{1}{7} = w2{1}{7};

fima = idwt3D(w1,1,sf);
fima = fima(1:s(1),1:s(2),1:s(3));

% negative checking (only for rician noise mixing)
ind=find(fima<0);
fima(ind)=fimau(ind);

% NAN checking
ind=find(isnan(fima(:)));
fima(ind)=fimau(ind);
fima(isnan(fima)) = 0;



% s = size(fimau);
% 
% p(1) = 2^(ceil(log2(s(1))));
% p(2) = 2^(ceil(log2(s(2))));
% p(3) = 2^(ceil(log2(s(3))));
% 
% pad1 = zeros(p(1),p(2),p(3));
% pad2=pad1;
% pad1(1:s(1),1:s(2),1:s(3)) = fimau(:,:,:);
% pad2(1:s(1),1:s(2),1:s(3)) = fimao(:,:,:); 
% 
% [af, sf] = farras;
% w1 = dwt3D(pad1,1,af);
% w2 = dwt3D(pad2,1,af);
% 
%   w1{1}{1} = (w1{1}{1} + w2{1}{1})/2;
%   w1{1}{2} = (w1{1}{2} + w2{1}{2})/2;
%   w1{1}{3} = w2{1}{3};
%   w1{1}{4} = (w1{1}{4} + w2{1}{4})/2;
%   w1{1}{5} = w2{1}{5};
%   w1{1}{6} = w2{1}{6};
%   w1{1}{7} = w2{1}{7};
% 
% fima = idwt3D(w1,1,sf);
% fima = fima(1:s(1),1:s(2),1:s(3));
