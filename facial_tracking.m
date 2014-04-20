clear all; close all;

faceimage = imread('face.jpg');
pic = rgb2ycbcr(faceimage); %RGB -> YCbCr conversion

%Remove noise from the pic by blurring with a gaussian filter
G = fspecial('gaussian',[5 5],2);
pic = imfilter(pic,G,'same');

%Split into Y, Cr, Cb components
y = size(pic,2);
x = size(pic,1);
Y=zeros(x,y); Cb=zeros(x,y); Cr=zeros(x,y);   % initialization;
Y=pic(:,:,1); Cb=pic(:,:,2); Cr=pic(:,:,3);

%Convert from uint8->double to work with the pixels
Y = double(Y);
Cb = double(Cb);
Cr = double(Cr);

new_Cb = Cb;
new_Cr = Cr;
new_Y = zeros(x,y);

%Loop through the pixels for thresholding
%Turn skincolored things white, background black
for i=1:size(Y,1)
    
   for k=1:size(Y,2)
       
       Y_val = Y(i,k);
       Cb_val = Cb(i,k);
       Cr_val = Cr(i,k);
       
       if Cb_val>76 && Cb_val<165 
           new_Cb_val = 256;
       else
           new_Cb_val = 0;
       end
       
       if Cr_val>130 && Cr_val<195
           new_Cr_val = 256;
       else
           new_Cr_val = 0;
       end
       
       new_Cb(i,k) = new_Cb_val;
       new_Cr(i,k) = new_Cr_val;
       
   end
end

image = cat(3,uint8(new_Y),uint8(new_Cb),uint8(new_Cr));
imshow(im2bw(image,.6))
%imshow(image);




% dx1=edge(pic(:,:,1),'canny');
% dx1=(dx1*255);
% img2(:,:,1)=dx1;
% img2(:,:,2)=pic(:,:,2);
% img2(:,:,3)=pic(:,:,3);
% rslt=ycbcr2rgb(uint8(img2));
% imshow(rslt);
% 
% 
% rslt = rgb2gray(rslt);
% K = filter2(fspecial('average',3),rslt)/255;
% %mshow(K)
% %figure,imshow(rslt);
% 
% %K is a greyscale sparse matrix
% 
% %Loop through K row by row and find where there are large rows of zeros
% 
% split_image = (size(K,2))/2;
% count = 0;
% 
% for i=1:size(K,1)
%     
%     %Find where there are zeros
%     row_zeros = find(K(i,:));
%     
%     consectutive=0;
%    
%     if size(row_zeros,2)>=split_image
%     
%        count = count+1;
%     end
% 
%     %What i think i want to do now is try to find the biggest consecutive
%     %row of indicies
% end