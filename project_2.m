% Object recognition using kernel method
%%
close all;clc;clear all;
tic;
load 'Vowel.mat'
X = featureMat;
[D, N] = size(X);
[t,~] = find(labelMat>0); %targets
N_train = 600;
X_train = X(:,1:N_train);
t_train = t(1:N_train);
X_test = X(:,N_train+1:end);
t_test = t(N_train+1:end);
%% Kernel formulation
K = zeros(size(X_train,2),size(X_train,2));
sigma2 = 0.9;
lambda = 0.003;
for n=1:size(X_train,2)
    for m=1:size(X_train,2)
        x_n = X_train(:,n);
        x_m = X_train(:,m);
        K(n,m) = exp(-((norm(x_n-x_m))^2)/(2*sigma2));
    end
end

a = (K+lambda*eye(size(K,2)))\t_train; % (eq. 6.8)
kx = zeros(size(X_train,2),1);
t_hat_test = zeros(size(X_test,2),1);
for j=1:size(X_test,2)
    x = X_test(:,j);
    for n=1:size(X_train,2)
        x_n = X_train(:,n);
        kx(n) = exp(-((norm(x_n-x))^2)/(2*sigma2));
    end
    t_hat_test(j) = kx'*a;
end
t_hat_test = round(t_hat_test);
temp = find(t_test==t_hat_test);
test_accuracy = (size(temp)/size(t_test))*100


% %% Use multi-fold cross validation to find optimal lambda
% k = 5;
% foldsize = size(X_train,2)/k;
% lambda = 0.01:0.01:1;
% 
% average_cost = zeros(length(lambda),2);
% average_accuracy = zeros(length(lambda),2);
% 
% for l=1:length(lambda)
%     di = length(lambda)-l
%     cost = zeros(k,1);
%     test_accuracy = zeros(k,1);
%     
%     for i=1:k
%         start = (i-1)*foldsize + 1;
%         end_ = start + foldsize-1;
%         
%         test_Xi =  X_train(:,start:end_);
%         test_ti = t_train(start:end_);
%         
%         Xi = X_train;
%         Xi(:,start:end_) = [];
%         ti = t_train;
%         ti(start:end_) = [];
%         
%         Ki = zeros(size(Xi,2),size(Xi,2));
%         for n=1:size(Xi,2)
%             for m=1:size(Xi,2)
%                 x_n = Xi(:,n);
%                 x_m = Xi(:,m);
%                 Ki(n,m) = exp(-((norm(x_n-x_m))^2)/(2*sigma2));
%             end
%         end
%         
%         a = (Ki+lambda(l)*eye(size(Ki,2)))\ti; % (eq. 6.8)
%         cost(i) = 0.5*a'*Ki*Ki*a - a'*Ki*ti + 0.5*(ti'*ti) + (lambda(l)/2)*a'*Ki*a; %(eq. 6.7)
%         %cost(i) = 0.5*a'*Ki*Ki*a - a'*Ki*ti + 0.5*(ti'*ti);
%         
%         kx = zeros(size(Xi,2),1);
%         t_hat_test = zeros(size(test_Xi,2),1);        
%         for j=1:size(test_Xi,2)
%             x = test_Xi(:,j);
%             for n=1:size(Xi,2)
%                 x_n = Xi(:,n);
%                 kx(n) = exp(-((norm(x_n-x))^2)/(2*sigma2));
%             end
%             t_hat_test(j) = kx'*a;
%         end
%         
%         t_hat_test = round(t_hat_test);
%         temp = find(test_ti==t_hat_test);
%         test_accuracy(i) = (size(temp)/size(test_Xi))*100;
%         
%     end
%     average_cost(l,:) = [lambda(l), sum(cost)/k];
%     average_accuracy(l,:) = [lambda(l), sum(test_accuracy)/k];
%     
% end
% 
% [~,index1] = min(average_cost(:,2));
% optimal_lambda1 = lambda(index1);
% [~,index2] = max(average_accuracy(:,2));
% optimal_lambda2 = lambda(index2);
% 
% a1 = (K + optimal_lambda1*eye(size(K,2)))\ti; 
% a2 = (K + optimal_lambda2*eye(size(K,2)))\ti; 
% 
% 
% kx = zeros(size(X_train,2),1);
% t_hat_test2 = zeros(size(X_test,2),1);
% for j=1:size(X_test,2)
%     x = X_test(:,j);
%     for n=1:size(X_train,2)
%         x_n = X_train(:,n);
%         kx(n) = exp(-((norm(x_n-x))^2)/(2*sigma2));
%     end
%     t_hat_test2(j) = kx'*a2;
% end
% 
% 
% 
% t_hat_test = X_test*w_opt1;
% t_hat_test = round(t_hat_test);
% temp = find(t_test==t_hat_test);
% test_accuracy1 = (size(temp)/size(t_test))*100;
% 
% 
% t_hat_test = round(t_hat_test2);
% temp = find(t_test==t_hat_test);
% test_accuracy2 = (size(temp)/size(t_test))*100;
% D1 = ['lambda_1 = ',num2str(optimal_lambda1),'; Accuracy = ',num2str(test_accuracy1)];
% D2 = ['lambda_2 = ',num2str(optimal_lambda2),'; Accuracy = ',num2str(test_accuracy2)];
% disp(D1);
% disp(D2);
% 
% 
% toc;
