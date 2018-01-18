%%
% Segment a graph
% Returns a disjoint-set forest representing the segmentation.

function [elts, num_ccs] = segment_graph(width, height, num_edges, edges, c)

    num_vertices = width * height;
    
    % sort edges-struck by weight
    [new, index] = sort([edges.w], 'ascend');  %ÉıĞòÅÅÁĞ
    for i = 1 : size(index,2)
        temp(i).w = edges(index(i)).w;
        temp(i).a = edges(index(i)).a;
        temp(i).b = edges(index(i)).b;
    end
    edges = temp;
    
    % make a disjoint-set forest
    [elts, num] = makeSet(num_vertices);
    
    % init thresholds
    for i = 1 : num_vertices
        threshold(i) = c;
    end
    
    % for each edge, in non-decreasing weight order...
    for i =1 : num_edges
        % components conected by this edge
        [a, elts]  = findSet(edges(i).a, elts);
        [b, elts]  = findSet(edges(i).b, elts);
        if((a ~= b))
            if ((edges(i).w <= threshold(a)) && (edges(i).w <= threshold(b)))
                elts = joinSet(a, b, elts, num);
                a = findSet(a, elts);
                threshold(a) = edges(i).w + c/elts(a).size;
            end
        end
    end
    
    %num_ccs = num;  % number of connected components in the segmentation
end