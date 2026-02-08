
function m = classificationMetrics(C)

    C = double(C);

    TP = diag(C);
    FP = sum(C,1)' - TP;
    FN = sum(C,2)  - TP;

    total = sum(C(:));
    m.accuracy = sum(TP) / max(total, 1);

    precision = TP ./ max(TP + FP, 1);
    recall    = TP ./ max(TP + FN, 1);

    denom = precision + recall;
    F1 = 2 .* precision .* recall ./ max(denom, eps);

    m.precision_per_class = precision;
    m.recall_per_class    = recall;
    m.F1_per_class        = F1;

    m.precision_macro = mean(precision);
    m.recall_macro    = mean(recall);
    m.F1_macro        = mean(F1);
end
