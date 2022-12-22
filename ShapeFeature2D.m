
function [ShapeFeature_vector]=ShapeFeature2D (ROIonly)
ROIonly=round(ROIonly);

%% 1 Mesh Surface
ROIonly=double(ROIonly);    

ROIonly_sifirlar=int16(ROIonly);
[row,col] = find(ROIonly_sifirlar~=0);
NaNolmayanpikseller=ROIonly(~isnan(ROIonly));

maske=zeros(size(ROIonly_sifirlar,1)+2,size(ROIonly_sifirlar,2)+2);
maske(2:end-1,2:end-1)=ROIonly_sifirlar;
% BWedge = edge(logical(maske));
BWedge = edge((maske),'canny');

im_in=BWedge;
        if sum(sum(im_in))>2
        [x,y]=find(im_in);


        x  = int16(x);          
        y  = int16(y);          
        len_data = length(x);   
        data_ordered = zeros(len_data,2,'int16');   
        inf_int16 = intmax('int16');        
        temp = ones(1,len_data,'int16');     
        d = zeros(len_data,'int16');         
        a=1;

        xd =  x(:,temp);  xd = xd - xd';  
        yd =  y(:,temp);  yd = yd - yd'; 
        ix = 1:len_data + 1:len_data*len_data;  
        d(ix) = inf_int16;                       
        d = d + xd.*xd + yd.*yd;                 

            for index = 1:len_data                     
                [~,I] = min(d(:,a));                   
                data_ordered(index,:) = [x(I), y(I)];  
                d(a,:) = inf_int16;                    
                a = I;                                 
            end
             
    else
    end
        

data_ordered(length(data_ordered)+1,:)=data_ordered(1,:);
data_ordered=double(data_ordered);
x=data_ordered(:,1);
y=data_ordered(:,2);
p = (1:length(x))' ; 
C = [p(1:end-1) p(2:end)] ; 
dt = delaunayTriangulation(x,y,C) ; 
% figure
% triplot(dt)
xnew = dt.Points(:,1) ; ynew = dt.Points(:,2) ;
isInside = isInterior(dt) ;         % Find triangles inside the constrained edges
tri = dt(isInside, :);              % Get end point indices of the inner triangles
% figure
% triplot(tri,ynew,xnew);
     
vertices=[xnew(tri(:,1)) ynew(tri(:,1)) xnew(tri(:,2)) ynew(tri(:,2)) xnew(tri(:,3)) ynew(tri(:,3))];

area = 1/2*abs((vertices(:,4)-vertices(:,2)).*(vertices(:,5)-vertices(:,1))-(vertices(:,6)-vertices(:,2)).*(vertices(:,3)-vertices(:,1)));
ucgenalan=sum(area);

%% 2 Pixel Surface

pikselalan=length(NaNolmayanpikseller);

%% 3 Perimeter
boundary_dist = 0;
BW = ROIonly_sifirlar;
B = bwboundaries(BW);
% imshow(BW, [min(min(BW)) max(max(BW))]); hold on;
for k=1:length(B)
boundary = B{k};
% cidx = mod(k,length(colors))+1;
% plot(boundary(:,2), boundary(:,1),'r','LineWidth',2);
end
% hold off;
for i = 1:length(boundary)
   if i ~= length(boundary)
       pixel1 = boundary(i,:);
       pixel2 = boundary(i+1,:);
   else
       pixel1 = boundary(i,:);
       pixel2 = boundary(1,:); % when it reaches the last boundary pixel
   end
   pixel_dist = ((pixel1(1,1) - pixel2(1,1)).^2 + (pixel1(1,2) - pixel2(1,2)).^2).^0.5;
   boundary_dist = boundary_dist + pixel_dist;
end
% display(boundary_dist);


%% 4 perimeter to surface ratio

AlanCevreOran=boundary_dist/ucgenalan;

%% 5 Sphericity

dairesellik=(2*sqrt(pi*ucgenalan))/boundary_dist;

%% 6 Spherical Disproportion

InvDairesellik=inv(dairesellik);

%% 7 Maximum 2D diameter

T = regionprops('table',logical(ROIonly_sifirlar),'PixelList');
T = feretProperties(T);
 maks_daimet=T. MaxFeretDiameter;


%% 8 major axis

stats = regionprops(logical(ROIonly_sifirlar),'MajorAxisLength','MinorAxisLength');

LongestAxis=stats.MajorAxisLength;

%% 9 Minor Axis
ShortestAxis=stats.MinorAxisLength;

%% 10 Elongation

Elong=ShortestAxis/LongestAxis;

%%
ShapeFeature_vector=[ucgenalan,pikselalan, boundary_dist, AlanCevreOran, dairesellik,InvDairesellik,maks_daimet,LongestAxis,ShortestAxis, Elong];


