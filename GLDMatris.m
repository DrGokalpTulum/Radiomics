function [GLDMFeature_vector]=GLDMfeatures(ROIonly)


% Gray Level Dependence Matrix (GLDM) Features

norm_ROIonly=ROIonly-min(min(ROIonly))+1;

norm_ROIonly = padarray(norm_ROIonly,[1 1],0,'both');

GLDMatris=zeros(max(max(norm_ROIonly)),9);

nan_olmayan_voksel_sayisi=size(norm_ROIonly,1)*size(norm_ROIonly,2)-length(norm_ROIonly(isnan(ROIonly)));


for k=1:max(max(norm_ROIonly))
    lojik_norm_ROIonly=norm_ROIonly==k;
    [x,y]=find(lojik_norm_ROIonly==1);
    
    if length(x)==0
        GLDMatris(k,:)=0;
    else
        
        for i=1:length(x)
            komsu=norm_ROIonly(x(i)-1:x(i)+1,y(i)-1:y(i)+1);
            komsu(5)=0;
            
            benzersayisi=length(find(komsu==k));
            
            if benzersayisi==0
                
                GLDMatris(k,benzersayisi+1)=GLDMatris(k,benzersayisi+1)+1;
            else
                GLDMatris(k,benzersayisi+1)=GLDMatris(k,benzersayisi+1)+1;
            end
            
            
            
        end
    end
end


stop = find(sum(GLDMatris),1,'last');
GLDMatris(:,(stop+1):end) = [];



Ng=size(GLDMatris,1);
Nd=size(GLDMatris,2);
Np=size(norm_ROIonly,1)*size(norm_ROIonly,2)-length(ROIonly(isnan(ROIonly)));
Nz=sum(sum(GLDMatris));

norm_GLDMatris=GLDMatris/Nz;


sz = size(norm_GLDMatris);
cVect = 1:sz(2); rVect = 1:sz(1);
[cMat,rMat] = meshgrid(cVect,rVect);
pg = sum(norm_GLDMatris,2)'; 
pr = sum(norm_GLDMatris); 
ug = (pg*rVect');
ur = (pr*cVect');

%% 1. Small Dependence Emphasis (SDE)

% SAE=0;
% 
% for i=1:Ng
%     for j=1:Nd
%         SAE=SAE+(GLDMatris(i,j)/j^2);
%     end
% end
% SAE=SAE/Nz;

SDE = pr*(cVect.^(-2))';

%% 2. Large Dependence Emphasis (LAE)

% LAE=0;
% 
% for i=1:Ng
%     for j=1:Nd
%         LAE=LAE+(GLDMatris(i,j)*j^2);
%     end
% end
% LAE=LAE/Nz;

LDE = pr*(cVect.^2)';


%% 3. Gray Level Non-Uniformity (GLN)

 GLND = sum(pg.^2);

% %%  Gray Level Non-Uniformity Normalized (GLNND)
% 
% GLNND=GLND/Nz;

%% 4. Dependence Non-Uniformity (DN)

DN = sum(pr.^2);

%% 5. Dependence Non-Uniformity Normalized (DNN)

DNN=DN/Nz;

 %% 7. Zone Percentage (ZP)
% 
% ZP = sum(pg)/(pr*cVect');
% % zp=Nz/Np;

%% 6. Gray Level Variance (GLV)

GLVD = 0;
for g = 1:sz(1)
    for r = 1:sz(2)
        GLVD = GLVD + norm_GLDMatris(g,r)*(g-ug)^2;
    end
end

% u_x=0
% 
% for i=1:Ng
%     for j=1:Nd
%         u_x=u_x+norm_GLDMatris(i,j)*i;
%     end
% end
% 
% glv=0;
% for i=1:Ng
%     for j=1:Nd
%         glv=glv+norm_GLDMatris(i,j)*(i-u_x)^2;
%     end
% end

%% 7. Dependence Variance (DV)

DV = 0;
for g = 1:sz(1)
    for r = 1:sz(2)
        DV = DV + norm_GLDMatris(g,r)*(r-ur)^2;
    end
end

% u_y=0
% 
% for i=1:Ng
%     for j=1:Nd
%         u_y=u_y+norm_GLSDMatrix(i,j)*j;
%     end
% end
% 
% zv=0;
% for i=1:Ng
%     for j=1:Nd
%         zv=zv+norm_GLSDMatrix(i,j)*(j-u_y)^2;
%     end
% end

%% 8. Dependence Entropy (ZE)

DE=0;

for i=1:Ng
    for j=1:Nd

        DE=DE-(norm_GLDMatris(i,j)*log(norm_GLDMatris(i,j) + eps));
    end
end

%% 9. Low Gray Level  Emphasis (LGLDE)

LGLDE= pg*(rVect.^(-2))';

% lglze=0;
% 
% for i=1:Ng
%     for j=1:Nd
%         lglze=lglze+(GLDMatris(i,j)/i^2);
%     end
% end
% lglze=lglze/Nz;

%% 10. High Gray Level  Emphasis (HGLDE)

HGLDE= pg*(rVect.^2)';

% hglze=0;
% 
% for i=1:Ng
%     for j=1:Nd
%         hglze=hglze+(GLDMatris(i,j)*i^2);
%     end
% end
% hglze=hglze/Nz;

%% 11. Small Dependence Low Gray Level Emphasis (SDLGLE)


% SALGLE=0;
% 
% for i=1:Ng
%     for j=1:Nd
%         SALGLE=SALGLE+(GLDMatris(i,j)/(i^2*j^2));
%     end
% end
% SALGLE=SALGLE/Nz;

SDLGLE = sum(sum(norm_GLDMatris.*(rMat.^(-2)).*(cMat.^(-2))));

%% 12. Small Dependence High Gray Level Emphasis (SDHGLE)

% SAHGLE=0;
% 
% for i=1:Ng
%     for j=1:Nd
%         SAHGLE=SAHGLE+((GLDMatris(i,j)*(i^2))/(j^2));
%     end
% end
% SAHGLE=SAHGLE/Nz;

SDHGLE = sum(sum(norm_GLDMatris.*(rMat.^2).*(cMat.^(-2))));

%% 13. Large Dependence Low Gray Level Emphasis (LDLGLE)

% LALGLE=0;
% 
% for i=1:Ng
%     for j=1:Nd
%         LALGLE=LALGLE+((GLDMatris(i,j)*(j^2))/(i^2));
%     end
% end
% LALGLE=LALGLE/Nz;

LDLGLE = sum(sum(norm_GLDMatris.*(rMat.^(-2)).*(cMat.^2)));

%% 14. Large Dependence High Gray Level Emphasis (LDHGLE)

% LAHGLE=0;
% 
% for i=1:Ng
%     for j=1:Nd
%         LAHGLE=LAHGLE+(GLDMatris(i,j)*(i^2*j^2));
%     end
% end
% LAHGLE=LAHGLE/Nz;

LDHGLE = sum(sum(norm_GLDMatris.*(rMat.^2).*(cMat.^2)));


%%
GLDMFeature_vector=[SDE,LDE, GLND, DN, DNN, GLVD,DV,DE,LGLDE, HGLDE,SDLGLE,SDHGLE,LDLGLE,LDHGLE]