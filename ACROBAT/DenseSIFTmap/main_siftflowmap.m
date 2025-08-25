clear;
close all
clc
datapath='ACROBAT_images_path';
files=dir(datapath);
for i=3:size(files,1)
    filename=files(i).name;
    im1=imread(fullfile(datapath,filename));
    im1=imfilter(im1,fspecial('gaussian',7,1.),'same','replicate');
    im1=im2double(im1);
    cellsize=3;
    gridspacing=1;
    sift = mexDenseSIFT(im1,cellsize,gridspacing);
    save(['savepath\',filename(1:end-4),'.mat'],'sift') ;
end
