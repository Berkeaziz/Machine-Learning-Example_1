function [theta, J_hist] = gradientDescentMulti(X, y, theta, alpha, num_iters)
    m = length(y);
    J_hist = zeros(num_iters, 1);

    for iter = 1:num_iters
        errors = X * theta - y;              
        grad = (1/m) * (X' * errors);        
        theta = theta - alpha * grad;

        J_hist(iter) = computeCostMulti(X, y, theta);
    end
end
