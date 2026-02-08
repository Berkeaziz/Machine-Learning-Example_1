% File: q6.m
clear; clc; close all;


csvFile = "Penguin.csv";  
k = 3;
K = 5;                    
doStandardize = true;     
rng(1);                   

if ~isfile(csvFile)
    error("CSV file not found: %s (update csvFile in q6.m)", csvFile);
end

T = readtable(csvFile);
T = rmmissing(T);         

varNames = string(T.Properties.VariableNames);

if all(ismember(["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g","species"], varNames))
    X = [T.bill_length_mm, T.bill_depth_mm, T.flipper_length_mm, T.body_mass_g];
    y = categorical(T.species);
else
    numMask = varfun(@isnumeric, T, "OutputFormat", "uniform");
    numCols = find(numMask);
    if numel(numCols) < 4
        error("Could not find 4 numeric feature columns in the CSV.");
    end
    X = table2array(T(:, numCols(1:4)));

    if any(varNames == "species")
        y = categorical(T.species);
    else
        catMask = ~numMask;
        catCols = find(catMask);
        if isempty(catCols)
            error("Could not find a categorical label column (e.g., 'species').");
        end
        y = categorical(T{:, catCols(end)});
    end
end

N = size(X,1);
classNames = categories(y);

% Cross-validation partition
cv = cvpartition(N, "KFold", K);

customScores = zeros(K,4);
matlabScores = zeros(K,4);

customCM = cell(K,1);
matlabCM = cell(K,1);

% CV loop
for fold = 1:K
    trIdx = training(cv, fold);
    teIdx = test(cv, fold);

    Xtr = X(trIdx,:);
    ytr = y(trIdx);

    Xte = X(teIdx,:);
    yte = y(teIdx);


    if doStandardize
        mu = mean(Xtr,1);
        sg = std(Xtr,0,1);
        sg(sg==0) = 1;

        XtrN = (Xtr - mu) ./ sg;
        XteN = (Xte - mu) ./ sg;
    else
        XtrN = Xtr;
        XteN = Xte;
    end

    yhatCustom = knnPredictCustom(XtrN, ytr, XteN, k, classNames);
    C1 = confusionmat(yte, yhatCustom, "Order", classNames);
    customCM{fold} = C1;

    m1 = classificationMetrics(C1);
    customScores(fold,:) = [m1.accuracy, m1.precision_macro, m1.recall_macro, m1.F1_macro];

    mdl = fitcknn(XtrN, ytr, "NumNeighbors", k, "Standardize", false);
    yhatMatlab = predict(mdl, XteN);

    C2 = confusionmat(yte, yhatMatlab, "Order", classNames);
    matlabCM{fold} = C2;

    m2 = classificationMetrics(C2);
    matlabScores(fold,:) = [m2.accuracy, m2.precision_macro, m2.recall_macro, m2.F1_macro];

    fprintf("\n=== FOLD %d / %d ===\n", fold, K);

    fprintf("CUSTOM k-NN (k=%d): Accuracy=%.4f, Precision=%.4f, Recall=%.4f, F1=%.4f\n", ...
        k, m1.accuracy, m1.precision_macro, m1.recall_macro, m1.F1_macro);
    disp("Confusion matrix (CUSTOM) [rows=true, cols=pred]:");
    disp(array2table(C1, "VariableNames", classNames, "RowNames", classNames));

    fprintf("fitcknn (k=%d):      Accuracy=%.4f, Precision=%.4f, Recall=%.4f, F1=%.4f\n", ...
        k, m2.accuracy, m2.precision_macro, m2.recall_macro, m2.F1_macro);
    disp("Confusion matrix (fitcknn) [rows=true, cols=pred]:");
    disp(array2table(C2, "VariableNames", classNames, "RowNames", classNames));
end

customAvg = mean(customScores, 1);
matlabAvg = mean(matlabScores, 1);

fprintf("\n\n===== 5-FOLD AVERAGES =====\n");
fprintf("CUSTOM k-NN: Accuracy=%.4f, Precision=%.4f, Recall=%.4f, F1=%.4f\n", ...
    customAvg(1), customAvg(2), customAvg(3), customAvg(4));
fprintf("fitcknn:     Accuracy=%.4f, Precision=%.4f, Recall=%.4f, F1=%.4f\n", ...
    matlabAvg(1), matlabAvg(2), matlabAvg(3), matlabAvg(4));

foldIdx = (1:K).';
customTable = array2table([foldIdx, customScores], ...
    "VariableNames", ["Fold","Accuracy","Precision_macro","Recall_macro","F1_macro"]);
matlabTable = array2table([foldIdx, matlabScores], ...
    "VariableNames", ["Fold","Accuracy","Precision_macro","Recall_macro","F1_macro"]);

disp(" "); disp("CUSTOM k-NN per-fold metrics:"); disp(customTable);
disp(" "); disp("fitcknn per-fold metrics:");   disp(matlabTable);
