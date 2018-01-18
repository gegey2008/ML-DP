%%
% random color
function c = random_rgb()
    c.r = uint8(round(255*rand(1,1)));
    c.g = uint8(round(255*rand(1,1)));
    c.b = uint8(round(255*rand(1,1)));
end