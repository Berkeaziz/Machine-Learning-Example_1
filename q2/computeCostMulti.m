function J = computeCostMulti(X, y, theta)
    m = length(y);
    errors = X * theta - y;
    J = (1/(2*m)) * (errors' * errors);
end