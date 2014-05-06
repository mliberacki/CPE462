%Facial Tracking and Suposition

clear all; close all;

raw_image = imread('face3.jpg'); %Read image
raw_image_adj = rgb2ycbcr(raw_image); %RGB -> YCbCr conversion

%Remove noise from the pic by blurring with a gaussian filter
G = fspecial('gaussian',[5 5],2);
raw_image_adj = imfilter(raw_image_adj,G,'same');

y = size(raw_image_adj,2);
x = size(raw_image_adj,1);

%Initialize YCbCr values
Y=zeros(x,y); Cb=zeros(x,y); Cr=zeros(x,y);
Y=raw_image_adj(:,:,1); Cb=raw_image_adj(:,:,2); Cr=raw_image_adj(:,:,3);

%Initialize RGB values
R=zeros(x,y); G=zeros(x,y); B=zeros(x,y);
R=raw_image(:,:,1); G=raw_image(:,:,2); B=raw_image(:,:,3);
R = double(R); G = double(G); B=double(B);

%Convert from uint8->double to work with the pixels
Y = double(Y);
Cb = double(Cb);
Cr = double(Cr);

%Define new variables
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
        
        if Cb_val>=77 && Cb_val<=135
            new_Cb_val = 256;
        else
            new_Cb_val = 0;
            G(i,k) = 0;
            R(i,k) = 0;
            B(i,k) = 0;
        end
        
        if Cr_val>=130 && Cr_val<=190
            new_Cr_val = 256;
        else
            new_Cr_val = 0;
            B(i,k)=0;
            R(i,k) = 0;
            G(i,k) = 0;
        end
        
        new_Cb(i,k) = new_Cb_val;
        new_Cr(i,k) = new_Cr_val;
        
    end
end

%New segmented image
image = cat(3,uint8(new_Y),uint8(new_Cb),uint8(new_Cr));

%Convert it to black and white
bl_image = im2bw(image,.6);
imshow(im2bw(image,.6))

%Make an image that blacks out all non-skin colored things
seg_image_rgb = cat(3,uint8(R),uint8(G),uint8(B));


%Current algorithm also picks up some hair
%Also the algorithm picks up some unnecessary things

%Crop out picture where ratio of black to white is too high

%Get number of zeros and ones in the image
zeros_bl = bl_image==0;
ones_bl = bl_image~=0;
zero = sum(zeros_bl,2);
one = sum(ones_bl,2);

ratio_vect = zeros(size(zero,1),1);

%Get the black to white ratio for each row
for i=1:length(zero)
    
    numzeros = zero(i);
    numones = one(i);
    
    ratio = numzeros / numones;
    ratio_vect(i) = ratio;
    
end

% %My way to adjust for picking up so much hair
% %Just cutoff the parts where there isn't as much white
% %Usually towards the top of the head
% mean_ratio = mean(~isinf(ratio_vect(:,1)));
% std_ratio = std(~isinf(ratio_vect(:,1)));
% highest_ratio = mean_ratio + std_ratio;
% 
% lowest_ratio = mean_ratio -std_ratio;
% 
% %Initialize RGB values
% y = size(seg_image_rgb,2);
% x = size(seg_image_rgb,1);
% R=zeros(x,y); G=zeros(x,y); B=zeros(x,y);
% R=seg_image_rgb(:,:,1); G=seg_image_rgb(:,:,2); B=seg_image_rgb(:,:,3);
% R = double(R); G = double(G); B=double(B);
% 
% %Loop through the image and make parts black for small black:white
% for i=1:size(R,1)
%     
%     if (ratio_vect(i)>highest_ratio && ratio_vect(i)<lowest_ratio)  || isinf(ratio_vect(i))
%         for k=1:size(R,2)
%             
%             G(i,k) = 0;
%             R(i,k) = 0;
%             B(i,k) = 0;
%             
%         end
%     end
% end


seg_image_rgb2 = cat(3,uint8(R),uint8(G),uint8(B));
figure; imshow(seg_image_rgb2);
bl_image2 = im2bw(seg_image_rgb2,.6);

%Use canny edge detection on the image to pic up facial features
%This threshold of .3 worked the best for us
edge_image = edge(rgb2gray(seg_image_rgb2),'canny',.3);
figure; imshow(edge_image);

%Facial feature detection

count = 0;
index = 0;

%Loop through image and get the first row of white pixel
%This is the top of the head
for i=1:size(edge_image,1)
    
    for j=1:size(edge_image,2)
        
        if edge_image(i,j)==1
            white_data = edge_image(i,:);
            
            if count==0
                index = i;
                face_top_array = white_data;
            end
            count = count +1;
        end
    end
end


count_ones = 0;
start_col = 0;

%Get column index where the ones start
for i=1:length(face_top_array)
    
    if face_top_array(1,i)==1
        
        if start_col==0
            start_col = i;
        end
        
        count_ones = count_ones + 1;
        
    end
    
end

%The middle of the top of the face
mid_face_index = round(start_col + count_ones/2);
mid_top_face = [index,mid_face_index];

%Get image centroids - get the centroid of the face
s  = regionprops(bl_image, 'centroid');
centroids = cat(1, s.Centroid);
imshow(bl_image);hold on;
plot(centroids(:,1), centroids(:,2), 'b*');

% centroid_range1 = round(start_col + count_ones/4);
% centroid_range2 = round(mid_face_index + 2.5*count_ones/2);
% 
% face_cen_vect = [];
% 
% for i=1:size(centroids,1)
%     
%     if centroids(i,1) > centroid_range1 && centroids(i,1) < centroid_range2
%         
%         if centroids(i,2) > mid_face_index+10
%             face_cen_vect = vertcat(centroids(i,:),face_cen_vect);
%             
%         end
%         
%         
%         
%     end
%     
% end
% 
% if size(face_cen_vect,1) > 1
%     %Pick centroid with point closest to col index of mid face top
%     dist = face_cen_vect(:,2) - mid_top_face(:,2);
%     
%     smallest_point = min(dist);
%     
%     for i=1:length(dist)
%         if dist(i)==smallest_point
%             face_centroid =  face_cen_vect(i,:);
%         end
%     end
% elseif size(face_cen_vect,1)==1
%     face_centroid = face_cen_vect(1,:);
% end


%What I'd like to do is find the nearest edge from canny to the centroid
%Hopefully, this will be the nose
%Let's try it!
one_indexes = [];


for i=1:size(edge_image,1)
    for j=1:size(edge_image,2)
        
        if edge_image(i,j)==1
            one_indexes  = vertcat( [ i , j] , one_indexes);
        end
        
    end
    
end
% 
% face_centroid2 = [face_centroid(2) , face_centroid(1)];
% 
% IDX = knnsearch(one_indexes,face_centroid2);
% 
% nearest_neigh = one_indexes(IDX,:);

%Find the centroid closest to the center of the image
center_image = [round(size(edge_image,2)/2), round(size(edge_image,1)/2) ];
IDX = knnsearch(centroids,center_image);
nearest_neigh = centroids(IDX,:);figure;
imshow(edge_image);hold on;
plot(nearest_neigh(:,1), nearest_neigh(:,2), 'b*');

hold on;
plot(centroids(:,1), centroids(:,2), 'r*');


face_centroid = nearest_neigh;
dist_btwn_top_center = face_centroid(:,2) - mid_top_face(:,1);

mid_bottom_face = [ face_centroid(1,2)+2*dist_btwn_top_center/3 , mid_face_index];
%face_centroid(1,2)+index , mid_face_index];

%Find the edge to mid bottom face
IDX = knnsearch(one_indexes,mid_bottom_face);
nearest_neigh2 = one_indexes(IDX,:);
%hold on; plot(nearest_neigh2(:,2), nearest_neigh2(:,1), 'b*');

mid_bottom_face = nearest_neigh2;


%The center of the face is probably above the face. I can track the nose if
%i search below that point.


imshow(seg_image_rgb2);
hold on;
plot(mid_bottom_face(:,2), mid_bottom_face(:,1), 'b*');
hold on;
plot(mid_top_face(:,2), mid_top_face(:,1), 'b*');
hold on;
plot(face_centroid(:,1), face_centroid(:,2), 'b*');
hold off;


