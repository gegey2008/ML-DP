%%
% implement disjoint set by union-by-rank

% https://www.geeksforgeeks.org/union-find-algorithm-set-2-union-by-rank/
%%
function [y, elts] = findSet(x, elts)
    y = x;
    while(y ~= elts(y).p)
        y = elts(y).p;
    end
    elts(x).p = y;
end