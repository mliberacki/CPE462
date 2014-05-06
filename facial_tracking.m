%Facial Tracking and Superposition
%Marissa Liberacki, Anthony Don, Tom Cruz
%CPE 462 - Image Processing and Coding

clear all; close all;

raw_image = imread('tom.jpg'); %Read image
raw_image_adj = rgb2ycbcr(raw_image); %RGB -> YCbCr conversion

%Remove noise from the pic by blurring with a gaussian filter
G = fspecial('gaussian',[5 5],2);
raw_image_adj = imfilter(raw_image_adj,G,'same');
figure; imshow(raw_image_adj);

%Define size of image
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
figure; imshow(im2bw(image,.6))

%Make an image that blacks out all non-skin colored things
seg_image_rgb = cat(3,uint8(R),uint8(G),uint8(B));
figure; imshow(seg_image_rgb);
bl_image2 = im2bw(seg_image_rgb,.6);

%Use canny edge detection on the image to pic up facial features
%This threshold of .3 worked the best for us
edge_image = edge(rgb2gray(seg_image_rgb),'canny',.3);
figure; imshow(edge_image);

count = 0;
index = 0;
count_ones = 0;
start_col = 0;

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
figure; imshow(bl_image);hold on;
plot(centroids(:,1), centroids(:,2), 'b*');

one_indexes = [];

for i=1:size(edge_image,1)
    for j=1:size(edge_image,2)
        
        if edge_image(i,j)==1
            one_indexes  = vertcat( [ i , j] , one_indexes);
        end
        
    end
    
end

%Find the centroid closest to the center of the image
center_image = [round(size(edge_image,2)/2), round(size(edge_image,1)/2) ];
IDX = knnsearch(centroids,center_image);
nearest_neigh = centroids(IDX,:);figure;

face_centroid = nearest_neigh;
dist_btwn_top_center = face_centroid(:,2) - mid_top_face(:,1);
mid_bottom_face = [ face_centroid(1,2)+2*dist_btwn_top_center/3 , mid_face_index];

%Find the edge to mid bottom face
IDX = knnsearch(one_indexes,mid_bottom_face);
nearest_neigh2 = one_indexes(IDX,:);
mid_bottom_face = nearest_neigh2;


figure; imshow(raw_image);
hold on;
plot(mid_bottom_face(:,2), mid_bottom_face(:,1), 'b*');
hold on;
plot(mid_top_face(:,2), mid_top_face(:,1), 'b*');
hold on;
plot(face_centroid(:,1), face_centroid(:,2), 'b*');

 %Puts the middle of the face in the same column as the mid top of face
 middleofface = [face_centroid(:,2), mid_top_face(:,2) ] ;
 for i= round(middleofface(1)):size(edge_image,1)
    if edge_image(i,round(middleofface(2)))==1
        nose = [i,middleofface(2)];
        disp(i);
        break;
    end
 end
 
 hold on;
 plot(nose(:,2), nose(:,1), 'b*');
 for i= round(nose(1)+5):size(edge_image,1)
    if edge_image(i,round(nose(2)))==1
        toplip = [i,middleofface(2)];
        disp(i);
        break;
    end
 end

imshow(raw_image);
hold on;
plot(middleofface(:,2), middleofface(:,1),'b*');
hold on;
plot(nose(:,2), nose(:,1), 'b*');
hold on;
plot(toplip(:,2), toplip(:,1), 'b*');

%Find place between nose and top of lip
btwn_nose_lip = ( toplip(1) - nose(1) ) /2;
btwn_nose_lip = nose(1) + btwn_nose_lip;

mustache_shift = [btwn_nose_lip, nose(2)];

hold on;
plot(mustache_shift(:,2), mustache_shift(:,1),'b*');

