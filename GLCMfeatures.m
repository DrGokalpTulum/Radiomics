%GLCM Features
function [GLCMFeature_vector]=GLCMfeatures(ROIonly)
warning off

ROIonly=round(ROIonly);

norm_ROIonly=ROIonly-min(min(ROIonly))+1;

[GLCM,SI] = graycomatrix(norm_ROIonly,'NumLevels',max(max(norm_ROIonly)),'GrayLimits',[],'Symmetric',true);


normalised_GLCM= GLCM/sum(sum(sum(GLCM))); % Normalize each glcm

%% 1 Autocorrelation

autokorelasyon_GLCM=0;
 for i = 1:size(normalised_GLCM,1)
        for j = 1:size(normalised_GLCM,2)
            
            autokorelasyon_GLCM = autokorelasyon_GLCM + ((i)*(j)*normalised_GLCM(i,j));
           
        end
 end

 %% 2 joint Average
  
 u_x=0;
 u_y=0;
 for i = 1:size(normalised_GLCM,1)
     for j = 1:size(normalised_GLCM,2)
         
         u_x= u_x + (i)*normalised_GLCM(i,j); 
         u_y= u_y + (j)*normalised_GLCM(i,j); 
         
     end
 end
 
 %% 3  Cluster Prominence and 4. Cluster Shade 5. Cluster Tendency

 cluster_prom=0;
 cluster_shade=0;
 cluster_tendency=0;
 
  for i = 1:size(normalised_GLCM,1)
        for j = 1:size(normalised_GLCM,2)
            
            cluster_prom = cluster_prom + (((i + j - u_x - u_y)^4)*normalised_GLCM(i,j));
            cluster_shade = cluster_shade+ (((i + j - u_x - u_y)^3)*normalised_GLCM(i,j));
            cluster_tendency = cluster_tendency+ (((i + j - u_x - u_y)^2)*normalised_GLCM(i,j));
        end
  end
 
    %% 6. Contrast
    kontrast=0;
 for i = 1:size(normalised_GLCM,1)
        for j = 1:size(normalised_GLCM,2)
            kontrast = kontrast + ((i - j))^2.*normalised_GLCM(i,j);
        end
 end   
 
  %% 7 COrrelation
  
  s_x=0;
  s_y=0;
  corp=0;
  for i = 1:size(normalised_GLCM,1)
        for j = 1:size(normalised_GLCM,2)
            corp= corp + ((i)*(j)*normalised_GLCM(i,j));
            s_x= s_x  + (((i) - u_x)^2)*normalised_GLCM(i,j);
            s_y= s_y  + (((j) - u_y)^2)*normalised_GLCM(i,j);
        end
  end
 s_x = s_x ^ 0.5;
 s_y = s_y ^ 0.5;
 
 korelasyon= (corp - u_x*u_y)/(s_x*s_y);

 
 %% 8. Difference Average
p_xplusy = zeros((size(normalised_GLCM,1)*2 - 1),1); 
p_xminusy = zeros((size(normalised_GLCM,1)),1); 
p_x = zeros(size(normalised_GLCM,1),1); % Ng x #glcms[1]  
p_y = zeros(size(normalised_GLCM,1),1); % Ng x #glcms[1]
 
 for i = 1:size(normalised_GLCM,1)
     
     for j = 1:size(normalised_GLCM,2)
            p_x(i) = p_x(i) + normalised_GLCM(i,j); 
            p_y(i) = p_y(i) + normalised_GLCM(j,i); 
         if (ismember((i + j),[2:2*size(normalised_GLCM,1)]))
             p_xplusy((i+j)-1) = p_xplusy((i+j)-1) + normalised_GLCM(i,j);
         end
         if (ismember(abs(i-j),[0:(size(normalised_GLCM,1)-1)]))
             p_xminusy((abs(i-j))+1) = p_xminusy((abs(i-j))+1) +normalised_GLCM(i,j);
         end
     end
 end
 
 
 diff_average=0;
 
 for i = 1:size(normalised_GLCM,1)
     diff_average=diff_average+i*p_xminusy(i);
 end
 
 %% 9. Difference Entropy 10. Difference Variance
 diff_entropy=0;
 diff_Variance=0;
 
 for i = 0:(size(normalised_GLCM,1)-1)
        diff_entropy = diff_entropy - (p_xminusy(i+1)*log(p_xminusy(i+1) + eps));
        diff_Variance = diff_Variance + (i^2)*p_xminusy(i+1);
 end
    
 
  %% 11 Dissimilarity
%  dissimilarity=0;
%  for i = 1:size(normalised_GLCM,1)
%      
%      for j = 1:size(normalised_GLCM,2)
%          
%          dissimilarity= dissimilarity + (abs(i - j)*normalised_GLCM(i,j));
%          
%      end
%  end
 
 
 %% 11. Joint Energy (Angular Second Moment)
 
 enerji=0;
 for i = 1:size(normalised_GLCM,1)
        for j = 1:size(normalised_GLCM,2)
            
            enerji = enerji + (normalised_GLCM(i,j).^2);
        end
 end
 
 %% 12. Joint Entropy
 
 joint_entropy=0;
 for i = 1:size(normalised_GLCM,1)
        for j = 1:size(normalised_GLCM,2)
           joint_entropy= joint_entropy - (normalised_GLCM(i,j)*log(normalised_GLCM(i,j) + eps));
        end
 end
 
 %% Homogenity 1 and Homogenity 2
 
%  homogen1=0;
%  homogen2=0;
%  
%  for i = 1:size(normalised_GLCM,1)
%      for j = 1:size(normalised_GLCM,2)
%          homogen1 = homogen1+ ( normalised_GLCM(i,j) / (1 + abs(i-j)) );
%          homogen2 = homogen2+ ( normalised_GLCM(i,j) / (1 + (abs(i-j))^2) );
%      end
%  end

%% 13. Informational Measure of Correlation (IMC) 1 14. Informational Measure of Correlation (IMC)2

hxy  = joint_entropy;
hxy1 = 0;
hx   = 0;
hy   = 0;
hxy2 = 0;

    for i = 1:size(normalised_GLCM,1)
        
        for j = 1:size(normalised_GLCM,2)
            hxy1 = hxy1 - (normalised_GLCM(i,j)*log(p_x(i)*p_y(j) + eps));
            hxy2 = hxy2 - (p_x(i)*p_y(j)*log(p_x(i)*p_y(j) + eps));
%            
        end
        hx = hx - (p_x(i)*log(p_x(i) + eps));
        hy = hy - (p_y(i)*log(p_y(i) + eps));
    end
    Inf_meas_of_corr1 = ( hxy - hxy1 ) / ( max([hx,hy]) );
    Inf_meas_of_corr2 = ( 1 - exp( -2*( hxy2 - hxy ) ) )^0.5;

%% 15. Inverse Difference Moment (IDM)(a.k.a Homogeneity 2)

%  homogen1=0;
%  homogen2=0;
%  
%  for i = 1:size(normalised_GLCM,1)
%      for j = 1:size(normalised_GLCM,2)
%          homogen1 = homogen1+ ( normalised_GLCM(i,j) / (1 + abs(i-j)) );
%          homogen2 = homogen2+ ( normalised_GLCM(i,j) / (1 + (abs(i-j))^2) );
%      end
%  end
 
 IDM=0;
 
 for i = 0:(size(normalised_GLCM,1)-1)
        IDM = IDM + (p_xminusy(i+1)/(1+i^2));
 end
 
 %% 16. Maximal Correlation Coefficient (MCC)
 
 Q = zeros(size(normalised_GLCM));
 
 rowCoOcMat = sum(normalised_GLCM,2);
 colCoOcMat = sum(normalised_GLCM);
 for i = 1 :size(normalised_GLCM,2)
     Q(i,:) = sum( ...
         (repmat(normalised_GLCM(i,:),size(normalised_GLCM,1),1) .* normalised_GLCM ) ./ ...
         repmat( rowCoOcMat(i) .* colCoOcMat, size(normalised_GLCM,1),1),...
         2,'omitnan');
 end
 
 Q(isnan(Q)) = 0;
 
 eigenvec = eig(Q);
 
 
 sort_eig= sort(eigenvec,'descend');
 MCC= sort_eig(2)^0.5;
 
%% 17. Inverse Difference Moment Normalized (IDMN)

% IDMN=0;
% for i = 1:size(normalised_GLCM,1)
% 
%         for j = 1:size(normalised_GLCM,2)
%            
%             IDMN = IDMN + (normalised_GLCM(i,j,k)/( 1 + ((i - j)/size(normalised_GLCM,1))^2));
%            
%         end
%         
% end

 IDMN1=0;
 
 for i = 0:(size(normalised_GLCM,1)-1)
        IDMN1 = IDMN1 + (p_xminusy(i+1)/(1+(i^2/(size(normalised_GLCM,1))^2)));
 end

    %% 18. Inverse Difference (ID) (a.k.a. Homogeneity 1)
    
%      homogen1=0;
%  homogen2=0;
%  
%  for i = 1:size(normalised_GLCM,1)
%      for j = 1:size(normalised_GLCM,2)
%          homogen1 = homogen1+ ( normalised_GLCM(i,j) / (1 + abs(i-j)) );
%          homogen2 = homogen2+ ( normalised_GLCM(i,j) / (1 + (abs(i-j))^2) );
%      end
%  end
 
 ID=0;
 
 for i = 0:(size(normalised_GLCM,1)-1)
        ID = ID + (p_xminusy(i+1)/(1+(i)));
 end
 
 %% 19. Inverse Difference Normalized (IDN)
 
 IDN=0;
 
 for i = 0:(size(normalised_GLCM,1)-1)
        IDN = IDN + (p_xminusy(i+1)/(1+(i/size(normalised_GLCM,1))));
 end
 
 %% 20. Inverse Variance
 
  INVAR=0;
 
 for i = 1:(size(normalised_GLCM,1)-1)
        INVAR = INVAR + (p_xminusy(i+1)/(1+(i^2)));
 end
 
 %% 21. Maximum Probability
 
 Max_prob=max(max(normalised_GLCM));
 
 
 %% 22 Sum Average 

 s_average=0;
 for i = 1:(2*(size(normalised_GLCM,1))-1)
     s_average = s_average + (i+1)*p_xplusy(i);
     
 end
 
 %% 23 Sum Entropy
 
 sum_entropy=0;
 for i =1:(2*(size(normalised_GLCM,1))-1)
         sum_entropy = sum_entropy - (p_xplusy(i)*log(p_xplusy(i) + eps));
 end
 
 %% Sum Variance 
%  sum_var=0;
%  for i = 1:(2*(size(normalised_GLCM,1))-1)
%      sum_var = sum_var + (((i+1) - sum_entropy)^2)*p_xplusy(i);
%  end

%% 24 Sum of Squares

glcm_mean = mean2(normalised_GLCM); % compute mean after norm
sum_of_sq=0;

 for i = 1:size(normalised_GLCM,1)
        for j = 1:size(normalised_GLCM,2)
            
%             sum_of_sq = sum_of_sq + normalised_GLCM(i,j)*((i - glcm_mean)^2);
            sum_of_sq = sum_of_sq + normalised_GLCM(i,j)*((i - u_x)^2);

        end
 end
 
 %%
 GLCMFeature_vector=[autokorelasyon_GLCM, u_x, cluster_prom,cluster_shade, cluster_tendency,kontrast, korelasyon,diff_average,...
     diff_entropy, diff_Variance,enerji,joint_entropy,Inf_meas_of_corr1, Inf_meas_of_corr2, IDM,MCC,IDMN1,ID,IDN,...
     INVAR,Max_prob, s_average, sum_entropy,sum_of_sq ];
 