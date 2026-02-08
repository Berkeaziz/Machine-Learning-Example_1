function [J, grad] = costFunctionReg(theta, X, y, lambda)
    m = length(y);
    h = sigmoid(X * theta);

    eps_val = 1e-9; 
    J = (1/m) * sum( -y .* log(h + eps_val) - (1-y) .* log(1 - h + eps_val) ) ...
        + (lambda/(2*m)) * sum(theta(2:end).^2);

    grad = (1/m) * (X' * (h - y));
    grad(2:end) = grad(2:end) + (lambda/m) * theta(2:end);
end