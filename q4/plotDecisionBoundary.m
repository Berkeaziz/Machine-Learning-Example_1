function plotDecisionBoundary(theta, X_raw, y, degree)
    x1 = X_raw(:,1);
    x2 = X_raw(:,2);

    u = linspace(min(x1)-0.1, max(x1)+0.1, 200);
    v = linspace(min(x2)-0.1, max(x2)+0.1, 200);
    Z = zeros(length(v), length(u));

    for i = 1:length(u)
        for j = 1:length(v)
            feats = mapFeature(u(i), v(j), degree); % 1 x 21
            Z(j,i) = feats * theta;
        end
    end

    contour(u, v, Z, [0, 0], 'LineWidth', 2);
end