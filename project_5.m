% Image de-noising using Markov Random field
%%
close all;clear all;clc;
%% Converting image to binary scale
I = imread('lenna.png');
I = rgb2gray(I);
I = double(I)./255; % Standard grayscale image
I((I <0.5))=-1;
I((I >0.5))=1;
I_original = I;

%% Adding noise to image
noise_power = 0.1; %std deviation
n = noise_power*randn(size(I));
I = I + n; % noisy image
I_noisy = I;

%% Image de-noising
% I_temp = [];
% for ii = 1:size(I,2)
%     if rem(ii,2)==0
%         col = I(:,ii);
%         col = flipud(col);
%     else 
%         col = I(:,ii);
%     end
%     I_temp = [I_temp,col]; 
% end

%y = I_temp(:);
y=I(:);
N = length(y);
x = y;
lambda = 0.01;
iter = 1000;
for j = 1:iter
    X = zeros(N,1);
    for i=1:N
        if i==1
            X(i) = (1+lambda)^(-1)*(y(i+1) + lambda*x(2));
        end
        if (i>1 && i<N)
            X(i) = (2*lambda+1)^(-1)*(y(i) + lambda*x(i-1) + lambda*x(i+1));
        end
        if i==N
            X(i) =  (1+lambda)^(-1)*(y(i) + lambda*x(i-1));
        end
    end
    x = X;
    y = x;
end
I_denoise = reshape(x,512,512);
% I_temp = [];
% for ii = 1:size(I,2)
%     if rem(ii,2)==0
%         col = I_denoise(:,ii);
%         col = flipud(col);
%     else 
%         col = I(:,ii);
%     end
%     I_temp = [I_temp,col]; 
% end
% I_denoise = I_temp;
mse = (1/N)*sum(I_original(:)-I_denoise(:)).^2;
%%
figure
imshow(I_original)
title('Original Image');

figure
imshow(I_noisy)
title('Noisy Image');

figure
imshow(I_denoise)
title('De-Noised Image');



