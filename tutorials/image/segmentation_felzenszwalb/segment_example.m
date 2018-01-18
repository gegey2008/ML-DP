%%

% Graph-Based Image Segmentation

%%
clc;
clear all;

% input Parameter
sigma = 0.5;
k = 500;
min_size = 20;
Im_size = 224;

% loading input image
image = loadImage('MGG.jpg',Im_size);

% processing
[image, imRef, num_ccs] = segment_image(image, sigma, k, min_size);

%show
figure(1)
imshow(image.im);
figure(2)
imshow(imRef);
%saveas(figure(2), 'hcm.fig')
