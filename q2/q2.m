%% Q2 - Multivariate Linear Regression 
clear; clc; close all;

data = load('q2data.txt');  
X = data(:,1:3);
y = data(:,4);
m = length(y);

% Feature normalization
mu = mean(X);
sigma = std(X);            
X_norm = (X - mu) ./ sigma;

Xn = [ones(m,1), X_norm];

% Gradient Descent settings
theta0 = zeros(size(Xn,2),1);
num_iters = 400;
alphas = [0.001 0.003 0.01 0.03 0.1];

thetas = cell(length(alphas),1);
J_last = zeros(length(alphas),1);

figure; hold on;
for k = 1:length(alphas)
    alpha = alphas(k);
    [theta_tmp, J_hist] = gradientDescentMulti(Xn, y, theta0, alpha, num_iters);
    thetas{k} = theta_tmp;
    J_last(k) = J_hist(end);

    plot(1:num_iters, J_hist, 'LineWidth', 1.5);
end
xlabel('Iteration');
ylabel('J(\theta)');
title('Q2: Cost vs Iteration for different learning rates \alpha');
legend('0.001','0.003','0.01','0.03','0.1','Location','northeast');
grid on; hold off;

exportgraphics(gcf, 'q2_cost_vs_iter.png', 'Resolution', 300);

% Choose best alpha 
[~, bestIdx] = min(J_last);
alpha_best = alphas(bestIdx);
theta_gd = thetas{bestIdx};

fprintf('Best alpha: %.4f\n', alpha_best);
fprintf('Theta (GD, normalized):\n');
disp(theta_gd);

% Prediction with GD model 
age = 20;
dist = 2500;
stores = 5;

x_raw = [age, dist, stores];
x_norm = (x_raw - mu) ./ sigma  ;
x_in = [1, x_norm];

pred_gd = x_in * theta_gd;
fprintf('Predicted price (GD, normalized features): %.4f\n', pred_gd);

% Normal Equation (raw features, no normalization)
Xne = [ones(m,1), X];
theta_ne = (Xne' * Xne) \ (Xne' * y);

fprintf('Theta (normal equation, raw features):\n');
disp(theta_ne);

x_in_ne = [1, age, dist, stores];
pred_ne = x_in_ne * theta_ne;
fprintf('Predicted price (Normal Eq., raw features): %.4f\n', pred_ne);
