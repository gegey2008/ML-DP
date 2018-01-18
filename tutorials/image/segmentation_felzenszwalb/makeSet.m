%%
% implement disjoint set by union-by-rank

%%
function [elts, num] = makeSet(elements)
    num = elements;
    for i = 1 : elements;
        elts(i).rank = 0;
        elts(i).size = 1;
        elts(i).p = i;        %parent node, arrange all node as parent node by weight
    end
end