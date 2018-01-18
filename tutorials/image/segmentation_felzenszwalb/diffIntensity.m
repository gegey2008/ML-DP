%%
% compute diff

function weight = diffIntensity(r, g, b, x1, y1, x2, y2)
    weight = sqrt((r(x1,y1)-r(x2,y2)).^2 + (g(x1,y1)-g(x2,y2)).^2 ...
                  + (b(x1,y1)-b(x2,y2)).^2);
end