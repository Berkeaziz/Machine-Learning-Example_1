
function out = mapFeature(x1, x2, degree)
    x1 = x1(:); x2 = x2(:);
    m = length(x1);
    out = ones(m, 1);
    for i = 1:degree
        for j = 0:i
            out = [out, (x1.^(i-j)).*(x2.^j)]; 
        end
    end
end