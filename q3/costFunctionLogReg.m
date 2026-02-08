function [J, grad] = costFunctionLogReg(theta, X, y)
% Logistic regression cost and gradient (no regularization)

m = length(y);
h = sigmoid(X * theta);

eps_val = 1e-9;

J = (1/m) * sum( -y .* log(h + eps_val) - (1 - y) .* log(1 - h + eps_val) );
grad = (1/m) * (X' * (h - y));
end
