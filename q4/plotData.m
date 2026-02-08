function plotData(X, y)
    pos = (y == 1);
    neg = (y == 0);
    plot(X(pos,1), X(pos,2), 'k+', 'LineWidth', 1.5, 'MarkerSize', 8);
    plot(X(neg,1), X(neg,2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 6);
    legend('y=1 (grew)', 'y=0 (failed)', 'Location', 'best');
end