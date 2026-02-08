function model = fitLDA_custom(X, y)
    classes = categories(y);
    K = numel(classes);
    [m,p] = size(X);

    mu = zeros(K,p);
    pi_k = zeros(K,1);

    for k = 1:K
        idx = (y == classes{k});
        Xk = X(idx,:);
        mu(k,:) = mean(Xk,1);
        pi_k(k) = sum(idx)/m;
    end


    Sigma = zeros(p,p);
    for k = 1:K
        idx = (y == classes{k});
        Xk = X(idx,:);
        Xc = Xk - mu(k,:);
        Sigma = Sigma + (Xc' * Xc);
    end
    Sigma = Sigma / (m - K);
    Sigma = Sigma + 1e-6*eye(p);

    invS = inv(Sigma);


    theta  = (invS * mu')';   

    theta0 = zeros(K,1);
    for k = 1:K
        mk = mu(k,:)';
        theta0(k) = log(pi_k(k) + eps) - 0.5*(mk' * invS * mk);
    end

    model.type    = 'LDA';
    model.classes = classes;
    model.mu      = mu;
    model.pi      = pi_k;
    model.Sigma   = Sigma;
    model.invS    = invS;
    model.theta   = theta;    
    model.theta0  = theta0;   
end
