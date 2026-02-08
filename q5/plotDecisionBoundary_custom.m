function plotDecisionBoundary_custom(model, X, y, methodName, outFile)
    x1 = X(:,1); x2 = X(:,2);

    u = linspace(min(x1)-0.1, max(x1)+0.1, 300);
    v = linspace(min(x2)-0.1, max(x2)+0.1, 300);
    [U,V] = meshgrid(u,v);
    gridX = [U(:), V(:)];

    if strcmpi(model.type,'LDA')
        ygrid = predictLDA_custom(model, gridX);
    else
        ygrid = predictQDA_custom(model, gridX);
    end

    classes = categories(y);
    K = numel(classes);

    ynum = zeros(numel(ygrid),1);
    for k=1:K
        ynum(ygrid == classes{k}) = k;
    end
    Z = reshape(ynum, size(U));

    figure; hold on;
    contour(U, V, Z, (1.5:1:(K-0.5)), 'k', 'LineWidth', 1.5);

    gscatter(X(:,1), X(:,2), y);
    xlabel('Bill Length');
    ylabel('Body Mass');
    title(sprintf('Q5 %s Decision Boundary (Bill Length vs Body Mass)', methodName));
    grid on;
    hold off;

    exportgraphics(gcf, outFile, 'Resolution', 300);
end