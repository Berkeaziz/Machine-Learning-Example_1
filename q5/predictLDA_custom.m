function yhat = predictLDA_custom(model, X)
    if size(X,2) ~= size(model.theta,2)
        if size(X,2) == size(model.theta,2) + 1 && all(abs(X(:,1) - 1) < 1e-12)
            X = X(:,2:end);
        else
            error('Dim mismatch: X is %dx%d, theta is %dx%d (KxP expected).', ...
                size(X,1), size(X,2), size(model.theta,1), size(model.theta,2));
        end
    end

    m = size(X,1);
    scores = X * model.theta' + ones(m,1) * (model.theta0');  
    [~, idx] = max(scores, [], 2);

    yhat = categorical(model.classes(idx), model.classes);
end