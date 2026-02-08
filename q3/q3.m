clear; clc; close all;

data = load('q3data.txt');
X = data(:,1:5);
y = data(:,6);
m = length(y);


% Part A

X_all = [ones(m,1) X];                
initial_theta = zeros(size(X_all,2),1);

options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'SpecifyObjectiveGradient', true, 'MaxIterations', 400, 'Display', 'iter');


[theta_all, J_all] = fminunc(@(t) costFunctionLogReg(t, X_all, y), initial_theta, options);

fprintf('Theta (ALL features):\n');
disp(theta_all);

% Prediction probability for:
% 50-year-old male, speed=120, helmet=1, seatbelt=1
x_new = [1 50 1 120 1 1];             
p_survive = sigmoid(x_new * theta_all);

fprintf('Predicted survival probability (50, male, speed=120, helmet=1, seatbelt=1): %.4f\n', p_survive);

% training accuracy with threshold 0.5
pred_train = sigmoid(X_all * theta_all) >= 0.5;
acc = mean(double(pred_train == y)) * 100;
fprintf('Training accuracy (threshold=0.5): %.2f%%\n', acc);

% Part B: Decision boundary using ONLY Age & Speed 

age = data(:,1);
speed = data(:,3);

X_2 = [ones(m,1) age speed];
initial_theta2 = zeros(size(X_2,2),1);

[theta_2, J_2] = fminunc(@(t) costFunctionLogReg(t, X_2, y), initial_theta2, options);

fprintf('\nTheta (Age & Speed only):\n');
disp(theta_2);

pos = (y == 1);
neg = (y == 0);

figure; hold on;
plot(age(pos), speed(pos), 'k+', 'LineWidth', 1.5);
plot(age(neg), speed(neg), 'ko', 'MarkerFaceColor', 'y');

% Decision boundary: theta0 + theta1*age + theta2*speed = 0
age_vals = linspace(min(age), max(age), 200);
speed_vals = -(theta_2(1) + theta_2(2)*age_vals) / theta_2(3);
plot(age_vals, speed_vals, 'b-', 'LineWidth', 2);

xlabel('Age');
ylabel('Speed of impact');
title('Q3: Decision boundary using Age and Speed');
legend('Survived (y=1)','Not survived (y=0)','Decision boundary', 'Location','best');
grid on; hold off;

