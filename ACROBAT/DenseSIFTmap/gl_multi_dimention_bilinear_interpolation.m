function [sift2new] = gl_multi_dimention_bilinear_interpolation(sift2, insize, outsize, dim)
% by gelin, used to do bilinear interpolation on multi-dimention data
% input (D, H, W), D is dimention
%     pltsift1 = permute(sift2,[2,3,1]);
    %figure;imshow(pltsift1(:,:,1:3),[]);
    sift2new = zeros(dim, outsize, outsize);
    temp = zeros(128, insize+2, insize+2);
    temp(:, 2:insize+1, 2:insize+1) = sift2;
    temp(:, 1, 2:insize+1) = sift2(:,1,:);
    temp(:, insize+2, 2:insize+1) = sift2(:,insize,:);
    temp(:, 2:insize+1, 1) = sift2(:, :, 1);
    temp(:, 2:insize+1, insize+2) = sift2(:, :, insize);

    for zj = 1:outsize
        for zi = 1:outsize
            ii = (zi)/(outsize/insize);
            jj = (zj)/(outsize/insize);
            i = floor(ii); j = floor(jj);
            u = ii-i; v = jj-j;
            i = i+1; j = j+1;
            sift2new(:, zi, zj) = (1-u)*(1-v)*temp(:,i,j) +(1-u)*v*temp(:,i,j+1) + u*(1-v)*temp(:,i+1,j) +u*v*temp(:,i+1,j+1);
        end
    end
    sift2new = uint8(sift2new);
    %pltsift1 = permute(sift2new,[2,3,1]);
    %figure;imshow(pltsift1(:,:,1:3),[]);

end

