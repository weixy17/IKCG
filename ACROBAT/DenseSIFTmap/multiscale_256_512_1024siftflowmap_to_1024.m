clear;
close all
clc
datapath='ACROBAT_images_path';%%%%%%%%image size:1024*1024
files=dir(datapath);
for i=3:size(files,1)
    filename=files(i).name;
        insize = 512;
        outsize=1024;
        dim=128;
        sift1 = load(['savepath_512\',filename(1:end-4),'.mat']);
        sift1 = sift1.sift;
        sift1_512to1024 = gl_multi_dimention_bilinear_interpolation(sift1, insize, outsize, dim);

        insize = 256;
        outsize=1024;
        dim=128;
        sift1 = load(['savepath_256\',filename(1:end-4),'.mat']);
        sift1 = sift1.sift;
        sift1_256to1024 = gl_multi_dimention_bilinear_interpolation(sift1, insize, outsize, dim);

        insize = 1024;
        outsize=1024;
        dim=128;
        sift1_1024 = load(['savepath\',filename(1:end-4),'.mat']);
        sift1_1024=sift1_1024.sift;
        
        sift1new = cat(1,sift1_256to1024, sift1_512to1024, sift1_1024);
        
        save(['savepath'],'sift1new');
    end
end
