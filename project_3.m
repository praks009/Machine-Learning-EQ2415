% Project 3: Extreme Learning Machine
%%
close all;clear all;clc;
load 'Vowel.mat'
X = featureMat;
T = labelMat; %targets
N_train = 600;
X_train = X(:,1:N_train);
T_train = T(:,1:N_train);
X_test = X(:,N_train+1:end);
T_test = T(:,N_train+1:end);
[t_test,~] = find(T_test>0);
n_node = 50:50:1000;
lambda = 30;
iter = 100;
test_accuracy = zeros(length(n_node),2);
for i=1:length(n_node)
    test_acc = zeros(iter,1);
    for j=1:iter
    W1 = randn(n_node(i),size(X_train,1));
    bias = randn(n_node(i),1);
    bias = bias-mean(bias);
    bias_matrix = repmat(bias,1,size(X_train,2));
    A = W1*X_train + bias_matrix; 
    %A = W1*X_train;
    %Y =  1 ./ (1 + exp(-A)); % Sigmoid function  
    Y = max(0,A); % ReLu function
    Pinv =  (lambda*eye(size(Y,2)) + Y'*Y)\(Y');
    O = Pinv'*T_train';
    %% testing
    bias_matrix = repmat(bias,1,size(X_test,2));
    A2 = W1*X_test + bias_matrix;
    %Y2 =  1 ./ (1 + exp(-A2));
    Y2 = max(0,A2);
    T_output = O'*(Y2);
    [~,t_hat_test]=max(T_output);
    t_hat_test = t_hat_test';
    temp = find(t_test==t_hat_test);
    test_acc(j) = (size(temp)/size(t_test));
    end
    test_accuracy(i,:) = [n_node(i),sum(test_acc)/iter];
end
%%
close all;
figure
set(0,'defaultlinelinewidth',3)
plot(n_node,test_accuracy(:,2))
xh = xlabel('number of nodes');
yh = ylabel('Accuracy');
th = title('ELM test accuracy plot');
set([xh,yh,th],'fontsize',20)
set(gca,'fontsize',20)
grid on 
grid minor

