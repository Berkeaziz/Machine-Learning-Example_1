% q1_main.m

clear; close all; clc;


data = load("q1data.txt");
x = data(:,1);
y = data(:,2);
m = length(y);

% 1.1 
figure;
scatter(x, y, "filled");
xlabel("City Population");
ylabel("Monthly sales revenue");
title("q1 data : Population vs Revenue");
grid on;

% Task 1–3

X = [ones(m,1), x];       
theta0 = zeros(2,1);      
alphas = [0.0001, 0.001, 0.01, 0.1];
num_iters = 2000;

thetas = cell(length(alphas),1);        
J_last = zeros(length(alphas),1);       

figure; hold on; grid on;
for k = 1:length(alphas)
    theta_init = theta0;                
    alpha = alphas(k);

    [theta_tmp, J_hist] = gradientDescentL1(X, y, theta_init, alpha, num_iters);

    thetas{k} = theta_tmp;
    J_last(k) = J_hist(end);

   
    plot(1:num_iters, J_hist, 'DisplayName', sprintf('\\alpha = %.4f', alpha));
end

xlabel("Iteration");
ylabel("L1 cost J(\\theta)");
title("Convergence Behaviour for different learning rates");
legend show;
hold off;


%Task 3-4

[~, bestIdx] = min(J_last);
alpha_best = alphas(bestIdx);
theta = thetas{bestIdx};  

fprintf('Best alpha: %.4f\n', alpha_best);
fprintf('Theta found: theta0 = %.4f, theta1 = %.4f\n', theta(1), theta(2));

% Plot linear fit with data
figure;
scatter(x, y, 'filled'); hold on;
xlabel('City population');
ylabel('Monthly sales revenue');
title('Linear regression with L1 loss');
grid on;

x_line = linspace(min(x), max(x), 100)';
X_line = [ones(size(x_line)), x_line];
y_line = X_line * theta;

plot(x_line, y_line, 'LineWidth', 2);
legend('Training data', 'L1 linear fit');
hold off;

%Task 5
pop_real = [20000; 60000];      
x_pred   = pop_real / 1000;    

X_pred = [ones(2,1), x_pred];
y_pred = X_pred * theta;

fprintf('Predicted revenue for 20,000: %.4f\n', y_pred(1));
fprintf('Predicted revenue for 60,000: %.4f\n', y_pred(2));

%Task 6

theta0_vals = linspace(theta(1)-10, theta(1)+10, 100);
theta1_vals = linspace(theta(2)-1,  theta(2)+1,  100);

J_vals = zeros(length(theta0_vals), length(theta1_vals));

for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
        t = [theta0_vals(i); theta1_vals(j)];
        J_vals(i,j) = costL1(X, y, t);
    end
end

J_vals = J_vals';   

% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals);
xlabel('\theta_0'); ylabel('\theta_1'); zlabel('J(\theta)');
title('L1 cost function surface');

% Contour plot
figure;
contour(theta0_vals, theta1_vals, J_vals, 30); hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('\theta_0'); ylabel('\theta_1');
title('L1 cost function contour with optimum');
grid on;
hold off;


outDir = pwd;  

exportgraphics(figure(1), fullfile(outDir,"q1_scatter.png"),      "Resolution",300);
exportgraphics(figure(2), fullfile(outDir,"q1_convergence.png"),  "Resolution",300);
exportgraphics(figure(3), fullfile(outDir,"q1_l1_fit.png"),       "Resolution",300);
exportgraphics(figure(4), fullfile(outDir,"q1_cost_surface.png"), "Resolution",300);
exportgraphics(figure(5), fullfile(outDir,"q1_cost_contour.png"), "Resolution",300);

