
function yhat = knnPredictCustom(Xtr, ytr, Xte, k, classNames)

    ytr = categorical(ytr, classNames);   
    ytrInt = double(ytr);               
    G = numel(classNames);

    nTest = size(Xte,1);
    yhatInt = zeros(nTest,1);

    for i = 1:nTest
        diff = Xtr - Xte(i,:);
        d2 = sum(diff.^2, 2);            
        [d2s, idx] = sort(d2, "ascend");

        idxK = idx(1:k);
        d2K  = d2s(1:k);
        neighLabels = ytrInt(idxK);

        votes = accumarray(neighLabels, 1, [G 1], @sum, 0);
        maxVote = max(votes);
        tied = find(votes == maxVote);

        if numel(tied) == 1
            yhatInt(i) = tied;
        else
            meanD2 = inf(G,1);
            for c = tied(:).'
                meanD2(c) = mean(d2K(neighLabels == c));
            end
            [~, pick] = min(meanD2);
            yhatInt(i) = pick;
        end
    end

    yhat = categorical(yhatInt, 1:G, classNames);
end
