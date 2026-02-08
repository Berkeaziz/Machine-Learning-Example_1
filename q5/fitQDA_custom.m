function model = fitQDA_custom(X, y)
    classes = categories(y);
    K = numel(classes);
    [m,p] = size(X);

    mu = zeros(K,p);
    pi_k = zeros(K,1);

    Sigma_k = zeros(p,p,K);
    invSigma_k = zeros(p,p,K);
    logdet_k = zeros(K,1);
    theta = zeros(K,p);
    theta0 = zeros(K,1);

    for k=1:K
        idx = (y == classes{k});
        Xk = X(idx,:);
        mu(k,:) = mean(Xk,1);
        pi_k(k) = sum(idx)/m;

        Xc = Xk - mu(k,:);
        Sk = (Xc' * Xc) / (sum(idx)-1);
        Sk = Sk + 1e-6*eye(p);

        Sigma_k(:,:,k) = Sk;
        invSk = inv(Sk);
        invSigma_k(:,:,k) = invSk;

        logdet_k(k) = log(det(Sk));

        mk = mu(k,:)';
        theta(k,:) = (invSk * mk)';   

        theta0(k) = log(pi_k(k)) - 0.5*logdet_k(k) - 0.5*(mk' * invSk * mk);
    end

    model.type = 'QDA';
    model.classes = classes;
    model.mu = mu;
    model.pi = pi_k;
    model.Sigma_k = Sigma_k;
    model.invSigma_k = invSigma_k;
    model.logdet_k = logdet_k;
    model.theta = theta;   
    model.theta0 = theta0;   
end
