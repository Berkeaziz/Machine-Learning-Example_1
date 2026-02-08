
%Task 1
function [theta, J_history] = gradientDescentL1(X, y, theta, alpha, num_iters)
   

    m = length(y);
    J_history = zeros(num_iters, 1);

    for iter = 1:num_iters
   
        h   = X * theta;        
        err = h - y;            
        grad = (1/m) * (X' * sign(err));  
        theta = theta - alpha * grad;
        J_history(iter) = (1/m) * sum(abs(err));
    end
end
