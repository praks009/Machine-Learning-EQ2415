% Object recognition using linear inference setup
%%
close all;clc;clear all;

load 'Vowel.mat'
X = featureMat';
[t,~] = find(labelMat>0); %targets
N_train = 600;
X_train = X(1:N_train,:);
t_train = t(1:N_train);
X_test = X(N_train+1:end,:);
t_test = t(N_train+1:end);

%% Use multi-fold cross validation to find optimal lambda
k = 5;
foldsize = size(X_train,1)/k;
lambda = 0:1:1000;

average_cost = zeros(length(lambda),2);
average_accuracy = zeros(length(lambda),2);

for l=1:length(lambda)
    cost = zeros(k,1);
    test_accuracy = zeros(k,1);
    for i=1:k
        start = (i-1)*foldsize + 1;
        end_ = start + foldsize-1;
        test_input =  X_train(start:end_,:);
        test_target = t_train(start:end_);
        train_input = X_train;
        train_target = t_train;
        
        train_input(start:end_,:) = [];
        train_target(start:end_) = [];
        
        w = (lambda(l)*eye(size(train_input,2)) + train_input'*train_input)\(train_input'*train_target);
        
        t_hat_test = test_input*w;
        t_hat_test = round(t_hat_test);
        temp = find(test_target==t_hat_test);
        test_accuracy(i) = (size(temp)/size(test_input))*100;
                
        cost(i) = (1/2)*sum((test_target-t_hat_test).^2);   
    end
    average_cost(l,:) = [lambda(l), sum(cost)/k];
    average_accuracy(l,:) = [lambda(l), sum(test_accuracy)/k];
    
end
[~,index1] = min(average_cost(:,2));
optimal_lambda1 = lambda(index1);
[~,index2] = max(average_accuracy(:,2));
optimal_lambda2 = lambda(index2);

w_opt1 = (optimal_lambda1*eye(size(X_train,2)) + ...
    X_train'*X_train)\(X_train'*t_train);

w_opt2 = (optimal_lambda2*eye(size(X_train,2)) + ...
    X_train'*X_train)\(X_train'*t_train);

%%
t_hat_test = X_test*w_opt1;
t_hat_test = round(t_hat_test);
temp = find(t_test==t_hat_test);
test_accuracy1 = (size(temp)/size(t_test))*100;

t_hat_test = X_test*w_opt2;
t_hat_test = round(t_hat_test);
temp = find(t_test==t_hat_test);
test_accuracy2 = (size(temp)/size(t_test))*100;
D1 = ['lambda_1 = ',num2str(optimal_lambda1),'; Accuracy = ',num2str(test_accuracy1)];
D2 = ['lambda_2 = ',num2str(optimal_lambda2),'; Accuracy = ',num2str(test_accuracy2)];
disp(D1);
disp(D2);



