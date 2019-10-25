% Project 4: Sparse representation
%%
close all;clear all;clc;
m = 500; % number of regressors
k = round(0.05*m); % Sparsity (5% of m)
n = k+5:20:300; % number of measurements
x_axis = n./m;
x1 = ones(k,1);
x2 = zeros(m-k,1);
x = [x1;x2];
index_true = sort(find(x>0));
Monte_Carlo = 3e2;
tf = zeros(6,Monte_Carlo);
NMSE = zeros(length(n),1);
nmse = zeros(length(n),1);
Monte_carlo = 10;
for i = 1:length(n)      
    for iter = 1:Monte_carlo
    x_hat = zeros(m,1);
    A = randn(n(i),m);
    A = normc(A); % Normalising the vectors
    b = A*x; %Observation without noise    
    index = sort(f_OMP(A,b));  %Support estimate via OMP til K_max
    A_OMP = A(:,index);
    x_coeff = ((A_OMP'*A_OMP))\(A_OMP'*b); 
    x_hat(index) = x_coeff;
    N = (1/m)*sum((x-x_hat).^2);
    D = (1/m)*sum(x.^2);
    nmse(iter) = N/D;
    end %loop over Monte carlo
    NMSE(i) = sum(nmse)/Monte_carlo;
end %loop over n

%% Plot
close all;
figure
ms = 20;
set(gca,'fontsize',30)
set(0,'defaultlinelinewidth',3)
hold on;grid on;box on;
plot(x_axis,NMSE,'-s','markers',ms,'Color',[0.85 0.325 0.1],'MarkerFaceColor',[0.85 0.325 0.1])
xh = xlabel(' $n/m$','Interpreter','Latex');
yh = ylabel('NMSE');
lh = legend('NMSE');
th = title([' $m$ = ',num2str(m),'; $k$ = ',num2str(k)],'Interpreter','Latex');
set([xh,yh,lh],'fontsize',30)

