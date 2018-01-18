%%
% processing image
function [image, imRef, num_ccs] = segment_image(image, sigma, c, min_size)

    width = image.width;
    height = image.height;
    
    %smooth each color channel
    im = double(image.im);
    len = ceil(sigma * 4) + 1;  %滤波器模板大小
    gausFilter = fspecial('gaussian', [len len], sigma); %matlab自带高斯滤波
    im_gaussian = imfilter(im, gausFilter, 'conv');
    im_gaussian_r = im_gaussian(:,:,1);
    im_gaussian_g = im_gaussian(:,:,2);
    im_gaussian_b = im_gaussian(:,:,3);
    
%% start build graph
    num = 0;
    for y = 1 : height
        for x = 1 : width
            if(x < width)
                num = num + 1;
                edges(num).a = (y-1) * width + x;
                edges(num).b = (y-1) * width + (x+1);
                edges(num).w = diffIntensity(im_gaussian_r, im_gaussian_g,...
                                            im_gaussian_b, x, y, x+1, y);
            end
            
            if(y < height)
                num = num + 1;
                edges(num).a = (y-1) * width + x;
                edges(num).b = y * width + x;
                edges(num).w = diffIntensity(im_gaussian_r, im_gaussian_g,...
                                            im_gaussian_b, x, y, x, y+1);
            end
            
            if((x < width)&&(y<height))
                num = num + 1;
                edges(num).a = (y-1) * width + x;
                edges(num).b = y * width + (x+1);
                edges(num).w = diffIntensity(im_gaussian_r, im_gaussian_g,...
                                            im_gaussian_b, x, y, x+1, y+1);
            end
            
            if((x < width)&&(y>1))
                num = num + 1;
                edges(num).a = (y-1) * width + x;
                edges(num).b = (y-2) * width + (x+1);
                edges(num).w = diffIntensity(im_gaussian_r, im_gaussian_g,...
                                            im_gaussian_b, x, y, x+1, y-1);
            end
        end
    end
    
    image.edges = edges;
    image.edges_num = num;
    
%% segment
    elts = segment_graph(width, height, num, edges, c);
    
    % post process small components
    for i =1 : num
        % components conected by this edge
        [a, elts]  = findSet(edges(i).a, elts);
        [b, elts]  = findSet(edges(i).b, elts);
        if((a ~= b) && ((elts(a).size < min_size) || (elts(b).size < min_size)))
            [elts, num] = joinSet(a, b, elts, num);
        end
    end
    
    num_ccs = num;  % number of connected components in the segmentation
    
    % pick random colors for each component
    for i = 1 : width*height
        colors(i) = random_rgb();
    end
    
    for y = 1:height
        for x = 1:width
            comp = findSet((y-1)*width+x, elts);
            imRef(x,y,1) = colors(comp).r;
            imRef(x,y,2) = colors(comp).g;
            imRef(x,y,3) = colors(comp).b;
        end
    end

end