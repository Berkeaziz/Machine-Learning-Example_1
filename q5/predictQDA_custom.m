function yhat = predictQDA_custom(model, X)
    [m,p] = size(X);
    K = numel(model.classes);

    scores = zeros(m,K);


    for k=1:K
        invSk = model.invSigma_k(:,:,k);
        theta_k = model.theta(k,:)';     
        quad = 0.5 * sum((X*invSk) .* X, 2); 
        scores(:,k) = model.theta0(k) + (X * theta_k) - quad;
    end

    [~, idx] = max(scores, [], 2);
    yhat = categorical(model.classes(idx), model.classes);
end

