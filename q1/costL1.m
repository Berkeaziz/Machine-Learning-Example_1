function J = costL1(X, y, theta)
    m = length(y);
    h = X * theta;
    J = (1/m) * sum(abs(h - y));
end
