clear;
close all
clc
datapath='ACROBAT_images_path';#######image size:1024*1024
files=dir(datapath);
for i=3:size(files,1)
    filename=files(i).name;
    im1=imread(fullfile(datapath,filename));
    im1=imfilter(im1,fspecial('gaussian',7,1.),'same','replicate');
    cellsize=3;
    gridspacing=1;
    
    im=im2double(im1);
    sift = mexDenseSIFT(im,cellsize,gridspacing);
    save(['savepath\',filename(1:end-4),'.mat'],'sift') ;
    %%%%%%%%resize to 512*512
    im1_2=imresize(im1,1/2,'bicubic');
    %%%%%%%%resize to different scale
    im1_2=im2double(im1_2);
    sift = mexDenseSIFT(im1_2,cellsize,gridspacing);
    save(['savepath_512\',filename(1:end-4),'.mat'],'sift') ;

    %%%%%%%%resize to 256*256
    im1_4=imresize(im1,1/4,'bicubic');
    %%%%%%%%resize to different scale
    im1_4=im2double(im1_4);
    sift = mexDenseSIFT(im1_4,cellsize,gridspacing);
    save(['savepath_256\',filename(1:end-4),'.mat'],'sift') ;
end







