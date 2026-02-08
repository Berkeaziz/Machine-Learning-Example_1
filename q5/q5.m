%% Q5 - LDA / QDA


clear; clc; close all;

dataFile = 'Penguin.csv';   
T = readtable(dataFile);

vars = T.Properties.VariableNames;

% find columns robustly 
specIdx = find(strcmpi(vars,'species'), 1);
if isempty(specIdx)
    error('Cannot find "species" column in Penguin.csv');
end

billIdx = find(contains(lower(vars),'bill') & contains(lower(vars),'length'), 1);
massIdx = find(contains(lower(vars),'body') & contains(lower(vars),'mass'), 1);

if isempty(billIdx) || isempty(massIdx)
    error('Cannot find Bill Length and/or Body Mass columns. Check column names in Penguin.csv');
end

y = T{:,specIdx};
X = [T{:,billIdx}, T{:,massIdx}];   

% remove missing rows
mask = all(~isnan(X),2) & ~ismissing(y);
X = X(mask,:);
y = categorical(y(mask));

classes = categories(y);
K = numel(classes);
fprintf('Samples after cleaning: %d\n', size(X,1));
fprintf('Classes: %s\n', strjoin(classes', ', '));

% 5.1 - Plot raw data

figure; hold on;
gscatter(X(:,1), X(:,2), y);
xlabel('Bill Length');
ylabel('Body Mass');
title('Q5 Data (Bill Length vs Body Mass)');
grid on; hold off;
exportgraphics(gcf,'q5_data.png','Resolution',300);

% 5.2 - LDA (custom)

lda = fitLDA_custom(X, y);
yhat_lda = predictLDA_custom(lda, X);

[Clda, order] = confusionmat(y, yhat_lda);
acc_lda = mean(yhat_lda == y) * 100;

fprintf('\n===== CUSTOM LDA =====\n');
disp('Confusion matrix (rows=true, cols=pred):');
disp(array2table(Clda,'VariableNames',cellstr(order),'RowNames',cellstr(order)));
fprintf('Training accuracy: %.2f%%\n', acc_lda);

% decision boundary plot (LDA)
plotDecisionBoundary_custom(lda, X, y, 'LDA', 'q5_lda_boundary.png');


% 5.3 - QDA (custom)

qda = fitQDA_custom(X, y);
yhat_qda = predictQDA_custom(qda, X);

[Cqda, order] = confusionmat(y, yhat_qda);
acc_qda = mean(yhat_qda == y) * 100;

fprintf('\n===== CUSTOM QDA =====\n');
disp('Confusion matrix (rows=true, cols=pred):');
disp(array2table(Cqda,'VariableNames',cellstr(order),'RowNames',cellstr(order)));
fprintf('Training accuracy: %.2f%%\n', acc_qda);

plotDecisionBoundary_custom(qda, X, y, 'QDA', 'q5_qda_boundary.png');


% 5.4 - Compare with MATLAB fitcdiscr 

fprintf('\n===== MATLAB fitcdiscr compare =====\n');
try
    mdlL = fitcdiscr(X, y, 'DiscrimType','linear');
    yhatL = predict(mdlL, X);
    C = confusionmat(y, yhatL);
    acc = mean(yhatL == y) * 100;
    fprintf('fitcdiscr (linear) accuracy: %.2f%%\n', acc);

    mdlQ = fitcdiscr(X, y, 'DiscrimType','quadratic');
    yhatQ = predict(mdlQ, X);
    C = confusionmat(y, yhatQ);
    acc = mean(yhatQ == y) * 100;
    fprintf('fitcdiscr (quadratic) accuracy: %.2f%%\n', acc);
catch ME
    fprintf('fitcdiscr not available or failed: %s\n', ME.message);
end








