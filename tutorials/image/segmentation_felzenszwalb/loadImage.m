%%
% Load Image

function image = loadImage(filePath,Im_size)
    if nargin == 0
        fprintf('Error: No argv !');
    elseif nargin == 1
        Im_size = 224;
    elseif nargin > 2
        fprintf('Error: argv number more than 2 !');
    end
    im = imread(filePath);
    im = imresize(im, [Im_size, Im_size], 'bilinear');
    image.im = im;
    image.width = size(im, 1);
    image.height = size(im, 2);
    image.r = im(:,:,1);
    image.g = im(:,:,2);
    image.b = im(:,:,3);
end