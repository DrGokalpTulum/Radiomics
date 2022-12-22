%% veri yükleme
clear all;clc;imtool close all;
folder=uigetdir('');
%%
[ I, BT, VS ] = dicomreadfolder(folder);
BT=double(BT);
volume_mm= [VS.PixelSpacing(:) ; VS.SliceThickness];

%%
info = dicominfo('F:\FERHAT\radiomics\DATABASE\TEXT SEG HASTALAR\ADENOKANSER\ADEM AYGÜNEÞ\adem aygüneþ.dcm');
brain_mri=dicomread(['F:\FERHAT\radiomics\DATABASE\TEXT SEG HASTALAR\ADENOKANSER\ADEM AYGÜNEÞ\adem aygüneþ']);

brain_mri_new=(zeros(256,256,size(brain_mri,4)));
brain_mri_new(:,:,1:size(brain_mri,4))=brain_mri(:,:,1,1:size(brain_mri,4));

BT=double(brain_mri_new);

slc1=info.PerFrameFunctionalGroupsSequence.Item_1.PixelMeasuresSequence.Item_1.PixelSpacing; 

slc2=info.NumberOfFrames; 

volume_mm= [slc1(:) ; slc2];




%%

imtool(BT(:,:,30),[])
if min(min(min(BT)))<0
    BT=BT+1024;
end


%%
folder2=uigetdir('');
filePattern = fullfile(folder2, '*.mat');
matFiles = dir(filePattern);

% i=input('deger gir=');
baseFileName = fullfile(folder2, matFiles.name);
storedStructure = load(baseFileName);

% Matdosyasi=storedStructure.segmented;
Matdosyasi=storedStructure.segmented;
%% 2D Wavelet
slice_yer=unique(ceil(find(Matdosyasi==1)/(size(Matdosyasi,1)*size(Matdosyasi,2))));

% wavelet2_BT=zeros(size(BT,1),size(BT,2),size(BT,3));
LL_wavelet2_BT=zeros(size(BT,1)/2,size(BT,2)/2,round(size(BT,3)));
LH_wavelet2_BT=zeros(size(BT,1)/2,size(BT,2)/2,round(size(BT,3)));
HL_wavelet2_BT=zeros(size(BT,1)/2,size(BT,2)/2,round(size(BT,3)));
HH_wavelet2_BT=zeros(size(BT,1)/2,size(BT,2)/2,round(size(BT,3)));

downsample_Matdosyasi=zeros(size(BT,1)/2,size(BT,2)/2,round(size(BT,3)));


for k=slice_yer(1):slice_yer(end)
    
    [LL,LH,HL,HH]=dwt2(BT(:,:,k),'DB1');
    
    downsample_Matdosyasi(:,:,k)=double((imresize(Matdosyasi(:,:,k),[size(LL,1) size(LL,2)])>.99));
    
    LL_wavelet2_BT(:,:,k)=LL;
    LH_wavelet2_BT(:,:,k)=LH;
    HL_wavelet2_BT(:,:,k)=HL;
    HH_wavelet2_BT(:,:,k)=HH;
    
    
end
    
ROI_LL=LL_wavelet2_BT.*downsample_Matdosyasi;
ROI_LH=LH_wavelet2_BT.*downsample_Matdosyasi;
ROI_HL=HL_wavelet2_BT.*downsample_Matdosyasi;
ROI_HH=HH_wavelet2_BT.*downsample_Matdosyasi;

CC_wlet = bwconncomp(downsample_Matdosyasi,8);
L_yeni_wlet = double(labelmatrix(CC_wlet));
% imtool(L_yeni_wlet(:,:,131),[])
stats_wlet = regionprops(CC_wlet,'BoundingBox');
kutu_wlet = cat(1, stats_wlet.BoundingBox);

%% LoG Filter
LOG_BT_2mm=zeros(size(BT,1),size(BT,2),round(size(BT,3)));
LOG_BT_4mm=zeros(size(BT,1),size(BT,2),round(size(BT,3)));
LOG_BT_6mm=zeros(size(BT,1),size(BT,2),round(size(BT,3)));


filterLOG_2mm = fspecial3('log',[3 3 1], .5);
LOG_BT_2mm(:,:,slice_yer(1)-4:slice_yer(end)+4)=imfilter(BT(:,:,slice_yer(1)-4:slice_yer(end)+4),filterLOG_2mm,'same');

filterLOG_4mm = fspecial3('log',[5 5 2], 1);
LOG_BT_4mm(:,:,slice_yer(1)-4:slice_yer(end)+4)=imfilter(BT(:,:,slice_yer(1)-4:slice_yer(end)+4),filterLOG_4mm,'same');

filterLOG_6mm = fspecial3('log',[7 7 3], 1.5);
LOG_BT_6mm(:,:,slice_yer(1)-4:slice_yer(end)+4)=imfilter(BT(:,:,slice_yer(1)-4:slice_yer(end)+4),filterLOG_6mm,'same');

LOG_ROI_2mm=Matdosyasi.*LOG_BT_2mm;
LOG_ROI_4mm=Matdosyasi.*LOG_BT_4mm;
LOG_ROI_6mm=Matdosyasi.*LOG_BT_6mm;

%%



%%
% ROI=Matdosyasi.*BT;
% 
% CC = bwconncomp(Matdosyasi,4);
% L_yeni = double(labelmatrix(CC));
% imtool(L_yeni(:,:,131),[])
% stats = regionprops(CC,'BoundingBox');
% kutu = cat(1, stats.BoundingBox);

% %% 3d wavelet
% ratio=1;
% wavelet='db1';
% wDEC=wavedec3(BT,1,wavelet);
% 
% 
% wave_matFiles=imresize3(Matdosyasi,[size(wDEC.dec{1},1)  size(wDEC.dec{1},2)  size(wDEC.dec{1},3)]);


%% Spatial
CC = bwconncomp(Matdosyasi,8);
L_yeni = double(labelmatrix(CC));
% imtool(L_yeni(:,:,131),[])
stats = regionprops(CC,'BoundingBox');
kutu = cat(1, stats.BoundingBox);

ROI=Matdosyasi.*BT;

CC = bwconncomp(Matdosyasi,8);
L_yeni = double(labelmatrix(CC));
% imtool(L_yeni(:,:,131),[])
stats = regionprops(CC,'BoundingBox');
kutu = cat(1, stats.BoundingBox);
%%
for k=1:max(max(max(L_yeni)))
%%
ornek=ROI(round(kutu(k,2)):round(kutu(k,2))+round(kutu(k,5)),round(kutu(k,1)):round(kutu(k,1))+round(kutu(k,4)),...
        round(kutu(k,3)));
ornek=double(ornek);
ornek(find(ornek==0))=NaN;
% Nbins=max(max(ornek))-min(min(ornek));
ROIonly=ornek;

[f_order_vector]=FirstOrderFeatures (ROIonly,volume_mm);
[ShapeFeature_vector]=ShapeFeature2D (ROIonly);
[GLCMFeature_vector]=GLCMfeatures(ROIonly);
[GLSZMFeature_vector]=GLSZMfeatures(ROIonly);
[GLRLMFeature_vector]=GLRLMMfeatures(ROIonly);
[NGTDMFeature_vector]=NGTDMfeatures(ROIonly);
[GLDMFeature_vector]=GLDMfeatures(ROIonly);
spatial_features(k,:)=[f_order_vector  ShapeFeature_vector GLCMFeature_vector GLSZMFeature_vector GLRLMFeature_vector NGTDMFeature_vector GLDMFeature_vector];

%%
ornek_LOG_2mm=LOG_ROI_2mm(round(kutu(k,2)):round(kutu(k,2))+round(kutu(k,5)),round(kutu(k,1)):round(kutu(k,1))+round(kutu(k,4)),...
        round(kutu(k,3)));
ornek_LOG_2mm=double(ornek_LOG_2mm);
ornek_LOG_2mm(find(ornek_LOG_2mm==0))=NaN;
% Nbins=max(max(ornek))-min(min(ornek));
ROIonly_LOG_2mm=ornek_LOG_2mm;

[f_order_vector]=FirstOrderFeatures (ROIonly_LOG_2mm,volume_mm);
% [ShapeFeature_vector]=ShapeFeature2D (ROIonly_LOG_2mm);
[GLCMFeature_vector]=GLCMfeatures(ROIonly_LOG_2mm);
[GLSZMFeature_vector]=GLSZMfeatures(ROIonly_LOG_2mm);
[GLRLMFeature_vector]=GLRLMMfeatures(ROIonly_LOG_2mm);
[NGTDMFeature_vector]=NGTDMfeatures(ROIonly_LOG_2mm);
[GLDMFeature_vector]=GLDMfeatures(ROIonly_LOG_2mm);
spatial_features_LOG_2mm(k,:)=[f_order_vector   GLCMFeature_vector GLSZMFeature_vector GLRLMFeature_vector NGTDMFeature_vector GLDMFeature_vector];

%%
ornek_LOG_4mm=LOG_ROI_4mm(round(kutu(k,2)):round(kutu(k,2))+round(kutu(k,5)),round(kutu(k,1)):round(kutu(k,1))+round(kutu(k,4)),...
        round(kutu(k,3)));
ornek_LOG_4mm=double(ornek_LOG_4mm);
ornek_LOG_4mm(find(ornek_LOG_4mm==0))=NaN;
% Nbins=max(max(ornek))-min(min(ornek));
ROIonly_LOG_4mm=ornek_LOG_4mm;

[f_order_vector]=FirstOrderFeatures (ROIonly_LOG_4mm,volume_mm);
% [ShapeFeature_vector]=ShapeFeature2D (ROIonly_LOG_4mm);
[GLCMFeature_vector]=GLCMfeatures(ROIonly_LOG_4mm);
[GLSZMFeature_vector]=GLSZMfeatures(ROIonly_LOG_4mm);
[GLRLMFeature_vector]=GLRLMMfeatures(ROIonly_LOG_4mm);
[NGTDMFeature_vector]=NGTDMfeatures(ROIonly_LOG_4mm);
[GLDMFeature_vector]=GLDMfeatures(ROIonly_LOG_4mm);
spatial_features_LOG_4mm(k,:)=[f_order_vector   GLCMFeature_vector GLSZMFeature_vector GLRLMFeature_vector NGTDMFeature_vector GLDMFeature_vector];

%%
ornek_LOG_6mm=LOG_ROI_6mm(round(kutu(k,2)):round(kutu(k,2))+round(kutu(k,5)),round(kutu(k,1)):round(kutu(k,1))+round(kutu(k,4)),...
        round(kutu(k,3)));
ornek_LOG_6mm=double(ornek_LOG_6mm);
ornek_LOG_6mm(find(ornek_LOG_6mm==0))=NaN;
% Nbins=max(max(ornek))-min(min(ornek));
ROIonly_LOG_6mm=ornek_LOG_6mm;

[f_order_vector]=FirstOrderFeatures (ROIonly_LOG_6mm,volume_mm);
% [ShapeFeature_vector]=ShapeFeature2D (ROIonly_LOG_6mm);
[GLCMFeature_vector]=GLCMfeatures(ROIonly_LOG_6mm);
[GLSZMFeature_vector]=GLSZMfeatures(ROIonly_LOG_6mm);
[GLRLMFeature_vector]=GLRLMMfeatures(ROIonly_LOG_6mm);
[NGTDMFeature_vector]=NGTDMfeatures(ROIonly_LOG_6mm);
[GLDMFeature_vector]=GLDMfeatures(ROIonly_LOG_6mm);
spatial_features_LOG_6mm(k,:)=[f_order_vector   GLCMFeature_vector GLSZMFeature_vector GLRLMFeature_vector NGTDMFeature_vector GLDMFeature_vector];

%%
ornek_wletLL=ROI_LL(round(kutu_wlet(k,2)):round(kutu_wlet(k,2))+round(kutu_wlet(k,5)),round(kutu_wlet(k,1)):round(kutu_wlet(k,1))+round(kutu_wlet(k,4)),...
        round(kutu_wlet(k,3)));
ornek_wletLL=double(ornek_wletLL);
ornek_wletLL(find(ornek_wletLL==0))=NaN;
% Nbins=max(max(ornek))-min(min(ornek));
ROIonly_wlet_LL=ornek_wletLL;

[f_order_vector_wlet_LL]=FirstOrderFeatures (ROIonly_wlet_LL,volume_mm);
% [ShapeFeature_vector_wlet_LL]=ShapeFeature2D (ROIonly_wlet_LL);
[GLCMFeature_vector_wlet_LL]=GLCMfeatures(ROIonly_wlet_LL);
[GLSZMFeature_vector_wlet_LL]=GLSZMfeatures(ROIonly_wlet_LL);
[GLRLMFeature_vector_wlet_LL]=GLRLMMfeatures(ROIonly_wlet_LL);
[NGTDMFeature_vector_wlet_LL]=NGTDMfeatures(ROIonly_wlet_LL);
[GLDMFeature_vector_wlet_LL]=GLDMfeatures(ROIonly_wlet_LL);
spatial_features_wlet_LL(k,:)=[f_order_vector_wlet_LL  GLCMFeature_vector_wlet_LL GLSZMFeature_vector_wlet_LL GLRLMFeature_vector_wlet_LL NGTDMFeature_vector_wlet_LL GLDMFeature_vector_wlet_LL];

%%
ornek_wletLL_LH=ROI_LH(round(kutu_wlet(k,2)):round(kutu_wlet(k,2))+round(kutu_wlet(k,5)),round(kutu_wlet(k,1)):round(kutu_wlet(k,1))+round(kutu_wlet(k,4)),...
        round(kutu_wlet(k,3)));
ornek_wletLL_LH=double(ornek_wletLL_LH);
ornek_wletLL_LH(find(ornek_wletLL_LH==0))=NaN;
% Nbins=max(max(ornek))-min(min(ornek));
ROIonly_wlet_LH=ornek_wletLL_LH;

[f_order_vector_LH]=FirstOrderFeatures (ROIonly_wlet_LH,volume_mm);
% [ShapeFeature_vector_wlet_LH]=ShapeFeature2D (ROIonly_wlet_LH);
[GLCMFeature_vector_wlet_LH]=GLCMfeatures(ROIonly_wlet_LH);
[GLSZMFeature_vector_wlet_LH]=GLSZMfeatures(ROIonly_wlet_LH);
[GLRLMFeature_vector_wlet_LH]=GLRLMMfeatures(ROIonly_wlet_LH);
[NGTDMFeature_vector_wlet_LH]=NGTDMfeatures(ROIonly_wlet_LH);
[GLDMFeature_vector_wlet__LH]=GLDMfeatures(ROIonly_wlet_LH);
spatial_features_wlet_LH(k,:)=[f_order_vector_LH GLCMFeature_vector_wlet_LH GLSZMFeature_vector_wlet_LH GLRLMFeature_vector_wlet_LH NGTDMFeature_vector_wlet_LH GLDMFeature_vector_wlet__LH];

%%

ornek_wletHL=ROI_HL(round(kutu_wlet(k,2)):round(kutu_wlet(k,2))+round(kutu_wlet(k,5)),round(kutu_wlet(k,1)):round(kutu_wlet(k,1))+round(kutu_wlet(k,4)),...
        round(kutu_wlet(k,3)));
ornek_wletHL=double(ornek_wletHL);
ornek_wletHL(find(ornek_wletHL==0))=NaN;
% Nbins=max(max(ornek))-min(min(ornek));
ROIonly_wlet_HL=ornek_wletHL;

[f_order_vector_wlet_HL]=FirstOrderFeatures (ROIonly_wlet_HL,volume_mm);
% [ShapeFeature_vector_wlet_HL]=ShapeFeature2D (ROIonly_wlet_HL);
[GLCMFeature_vector_wlet_HL]=GLCMfeatures(ROIonly_wlet_HL);
[GLSZMFeature_vector_wlet_HL]=GLSZMfeatures(ROIonly_wlet_HL);
[GLRLMFeature_vector_wlet_HL]=GLRLMMfeatures(ROIonly_wlet_HL);
[NGTDMFeature_vector_wlet_HL]=NGTDMfeatures(ROIonly_wlet_HL);
[GLDMFeature_vector_wlet_HL]=GLDMfeatures(ROIonly_wlet_HL);
spatial_features_wlet_HL(k,:)=[f_order_vector_wlet_HL  GLCMFeature_vector_wlet_HL GLSZMFeature_vector_wlet_HL GLRLMFeature_vector_wlet_HL NGTDMFeature_vector_wlet_HL GLDMFeature_vector_wlet_HL];

%%
ornek_wletHH=ROI_HH(round(kutu_wlet(k,2)):round(kutu_wlet(k,2))+round(kutu_wlet(k,5)),round(kutu_wlet(k,1)):round(kutu_wlet(k,1))+round(kutu_wlet(k,4)),...
        round(kutu_wlet(k,3)));
ornek_wletHH=double(ornek_wletHH);
ornek_wletHH(find(ornek_wletHH==0))=NaN;
% Nbins=max(max(ornek))-min(min(ornek));
ROIonly_wlet_HH=ornek_wletHH;

[f_order_vector_wlet_HH]=FirstOrderFeatures (ROIonly_wlet_HH,volume_mm);
% [ShapeFeature_vector_wlet_HH]=ShapeFeature2D (ROIonly_wlet_HH);
[GLCMFeature_vector_wlet_HH]=GLCMfeatures(ROIonly_wlet_HH);
[GLSZMFeature_vector_wlet_HH]=GLSZMfeatures(ROIonly_wlet_HH);
[GLRLMFeature_vector_wlet_HH]=GLRLMMfeatures(ROIonly_wlet_HH);
[NGTDMFeature_vector_wlet_HH]=NGTDMfeatures(ROIonly_wlet_HH);
[GLDMFeature_vector_wlet_HH]=GLDMfeatures(ROIonly_wlet_HH);
spatial_features_wlet_HH(k,:)=[f_order_vector_wlet_HH  GLCMFeature_vector_wlet_HH GLSZMFeature_vector_wlet_HH GLRLMFeature_vector_wlet_HH NGTDMFeature_vector_wlet_HH GLDMFeature_vector_wlet_HH];


end
% imtool(ornek,[])


% dicomwrite(ornek, 'ct_file.dcm', I);

FEATURES=[spatial_features spatial_features_LOG_2mm spatial_features_LOG_4mm spatial_features_LOG_6mm spatial_features_wlet_LL spatial_features_wlet_LH spatial_features_wlet_HL spatial_features_wlet_HH];
