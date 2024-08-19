clear all;
close all;
clc
pairflag=importdata('G:\for_submit\matrix_sequence_manual_validation.csv');
outsize = 1024;
dim = 128;
for image_num=0:480
    if ~isempty(strfind(pairflag{image_num+2},'training'))
       %% 256 to 1024 
        insize = 512;
        sift1 = load(['G:\re_do_from_ori\data\after_affine_siftflow_multi_scale\siftflowmap_', num2str(insize), '\', num2str(image_num),'sift1.mat']);
        sift1 = sift1.sift1new;
        sift2 = load(['G:\re_do_from_ori\data\after_affine_siftflow_multi_scale\siftflowmap_', num2str(insize), '\', num2str(image_num),'sift2.mat']);
        sift2 = sift2.sift2new;
        sift1_256to1024 = gl_multi_dimention_bilinear_interpolation(sift1, insize, outsize, dim);
        sift2_256to1024 = gl_multi_dimention_bilinear_interpolation(sift2, insize, outsize, dim);
        %% 2048 to 1024
        insize = 2048;
        sift1 = load(['G:\re_do_from_ori\data\after_affine_siftflow_multi_scale\siftflowmap_', num2str(insize), '\', num2str(image_num),'sift1.mat']);
        sift1 = sift1.sift1new;
        sift2 = load(['G:\re_do_from_ori\data\after_affine_siftflow_multi_scale\siftflowmap_', num2str(insize), '\', num2str(image_num),'sift2.mat']);
        sift2 = sift2.sift2new;
        sift2 = permute(sift2,[3,1,2]);
        sift1 = permute(sift1,[3,1,2]);
        sift1_512to1024 = gl_multi_dimention_bilinear_interpolation(sift1, insize, outsize, dim);
        sift2_512to1024 = gl_multi_dimention_bilinear_interpolation(sift2, insize, outsize, dim);
        %% 1024 load
        insize = 1024;
        sift1 = load(['G:\re_do_from_ori\data\after_affine_siftflow_multi_scale\siftflowmap_', num2str(insize), '\', num2str(image_num),'sift1.mat']);
        sift1_1024 = sift1.sift1new;
        sift2 = load(['G:\re_do_from_ori\data\after_affine_siftflow_multi_scale\siftflowmap_', num2str(insize), '\', num2str(image_num),'sift2.mat']);
        sift2_1024 = sift2.sift2new;
        %% combine and save
        sift1new = cat(1,sift1_256to1024, sift1_512to1024, sift1_1024);
        sift2new = cat(1,sift2_256to1024, sift2_512to1024, sift2_1024);
        
        save(['G:\re_do_from_ori\data\after_affine_siftflow_multi_scale\multi_siftflowmap_',num2str(outsize),'new\',num2str(image_num),'sift1.mat'],'sift1new') 
        save(['G:\re_do_from_ori\data\after_affine_siftflow_multi_scale\multi_siftflowmap_',num2str(outsize),'new\',num2str(image_num),'sift2.mat'],'sift2new') 
        
      
    end
end