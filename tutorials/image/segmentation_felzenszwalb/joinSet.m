%%
% implement disjoint set by union-by-rank

% https://www.geeksforgeeks.org/union-find-algorithm-set-2-union-by-rank/
%%
function [elts, num] = joinSet(x, y, elts, num)
    if (elts(x).rank > elts(y).rank)
        elts(y).p = x;
        elts(x).size = elts(x).size + elts(y).size;
    else
        elts(x).p = y;
        elts(y).size = elts(y).size + elts(x).size;
        if(elts(x).rank == elts(y).rank)
            elts(y).rank = elts(y).rank + 1;
        end
    end
    num = num - 1;
end