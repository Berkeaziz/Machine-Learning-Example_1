%% Q4 - Regularized Logistic Regression

clear; clc; close all;

dataFile = 'q4data.txt'; 
data = load(dataFile);

X = data(:,1:2);
y = data(:,3);
m = length(y);

% 1) Plot raw data

figure; hold on;
plotData(X, y);
xlabel('Moisture (x_1)');
ylabel('Nutrient (x_2)');
title('Q4 Data');
grid on; hold off;
exportgraphics(gcf, 'q4_data.png', 'Resolution', 300);

% 2) Map to polynomial features up to degree 5 

degree = 5;
X_mapped = mapFeature(X(:,1), X(:,2), degree);  


% 3) Train and plot for different lambdas

lambdas = [0 1 10 100];

for k = 1:length(lambdas)
    lambda = lambdas(k);

    initial_theta = zeros(size(X_mapped,2), 1);

    try
        options = optimoptions('fminunc', ...
            'Algorithm', 'quasi-newton', ...
            'SpecifyObjectiveGradient', true, ...
            'MaxIterations', 400, ...
            'Display', 'iter');
    catch
        options = optimset('GradObj','on','MaxIter',400,'Display','iter');
    end

 
    [theta, Jval] = fminunc(@(t) costFunctionReg(t, X_mapped, y, lambda), initial_theta, options);


    p = predict(theta, X_mapped);
    acc = mean(double(p == y)) * 100;

    fprintf('\n============================\n');
    fprintf('lambda = %g\n', lambda);
    fprintf('Final cost J(theta) = %.6f\n', Jval);
    fprintf('Training accuracy (0.5 threshold) = %.2f%%\n', acc);
    fprintf('Theta:\n');
    disp(theta);

    % 4) Plot decision boundary

    figure; hold on;
    plotData(X, y);
    plotDecisionBoundary(theta, X, y, degree);
    xlabel('Moisture (x_1)');
    ylabel('Nutrient (x_2)');
    title(sprintf('Q4 Decision Boundary (degree=%d, \\lambda=%g)', degree, lambda));
    grid on; hold off;

    exportgraphics(gcf, sprintf('q4_boundary_lambda_%g.png', lambda), 'Resolution', 300);
end

