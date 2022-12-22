%GLSZM Features

norm_ROIonly=ROIonly-min(min(ROIonly))+1;

GLSZMatrix=zeros(max(max(norm_ROIonly)),size(norm_ROIonly,1)*size(norm_ROIonly,2)-length(ROIonly(isnan(ROIonly))));


for i=1:max(max(norm_ROIonly))
    lojik_GLSZMatrix=norm_ROIonly==i;

     connObjects = bwconncomp(lojik_GLSZMatrix,8);
     baglantili_voksel=zeros(1,length(connObjects.PixelIdxList));

     for j=1: length(connObjects.PixelIdxList)
              baglantili_voksel(j)=length(connObjects.PixelIdxList{j});
     end
     
        edges = unique(baglantili_voksel);
        counts = histc(baglantili_voksel(:), edges);
        if sum(edges)==0
            GLSZMatrix(i,:)=GLSZMatrix(i,:);
        else
        for k=1:length(edges)
            GLSZMatrix(i,edges(k))=counts(k);
        end
        end
        clear edges;clear counts;
         
end

stop = find(sum(GLSZMatrix),1,'last');
GLSZMatrix(:,(stop+1):end) = [];

Ng=size(GLSZMatrix,1);
Ns=size(GLSZMatrix,2);
Np=size(norm_ROIonly,1)*size(norm_ROIonly,2)-length(ROIonly(isnan(ROIonly)));
Nz=sum(sum(GLSZMatrix));

norm_GLSZMatrix=GLSZMatrix/Nz;


sz = size(norm_GLSZMatrix);
cVect = 1:sz(2); rVect = 1:sz(1);
[cMat,rMat] = meshgrid(cVect,rVect);
pg = sum(norm_GLSZMatrix,2)'; 
pr = sum(norm_GLSZMatrix); 
ug = (pg*rVect');
ur = (pr*cVect');


%% 1. Small Area/Zone Emphasis (SAE)

% SAE=0;
% 
% for i=1:Ng
%     for j=1:Ns
%         SAE=SAE+(GLSZMatrix(i,j)/j^2);
%     end
% end
% SAE=SAE/Nz;

SZE = pr*(cVect.^(-2))';

%% 2. Large Area/Zone Emphasis (LAE)

% LAE=0;
% 
% for i=1:Ng
%     for j=1:Ns
%         LAE=LAE+(GLSZMatrix(i,j)*j^2);
%     end
% end
% LAE=LAE/Nz;

LZE = pr*(cVect.^2)';


%% 3. Gray Level Non-Uniformity (GLN)

GLN = sum(pg.^2);

%% 4. Gray Level Non-Uniformity Normalized (GLNN)

GLNN=GLN/Nz;

%% 5. Size-Zone Non-Uniformity (SZN)

SZN = sum(pr.^2);

%% 6. Size-Zone Non-Uniformity Normalized (SZNN)

SZNN=SZN/Nz;

%% 7. Zone Percentage (ZP)

ZP = sum(pg)/(pr*cVect');
% zp=Nz/Np;

%% 8. Gray Level Variance (GLV)

GLV = 0;
for g = 1:sz(1)
    for r = 1:sz(2)
        GLV = GLV + norm_GLSZMatrix(g,r)*(g-ug)^2;
    end
end

% u_x=0
% 
% for i=1:Ng
%     for j=1:Ns
%         u_x=u_x+norm_GLSZMatrix(i,j)*i;
%     end
% end
% 
% glv=0;
% for i=1:Ng
%     for j=1:Ns
%         glv=glv+norm_GLSZMatrix(i,j)*(i-u_x)^2;
%     end
% end

%% 9. Zone Variance (ZV)

ZV = 0;
for g = 1:sz(1)
    for r = 1:sz(2)
        ZV = ZV + norm_GLSZMatrix(g,r)*(r-ur)^2;
    end
end

% u_y=0
% 
% for i=1:Ng
%     for j=1:Ns
%         u_y=u_y+norm_GLSZMatrix(i,j)*j;
%     end
% end
% 
% zv=0;
% for i=1:Ng
%     for j=1:Ns
%         zv=zv+norm_GLSZMatrix(i,j)*(j-u_y)^2;
%     end
% end

%% 10. Zone Entropy (ZE)

ZE=0;

for i=1:Ng
    for j=1:Ns

        ZE=ZE-(norm_GLSZMatrix(i,j)*log(norm_GLSZMatrix(i,j) + eps));;
    end
end

%% 11. Low Gray Level Zone Emphasis (LGLZE)

LGLZE= pg*(rVect.^(-2))';

% lglze=0;
% 
% for i=1:Ng
%     for j=1:Ns
%         lglze=lglze+(GLSZMatrix(i,j)/i^2);
%     end
% end
% lglze=lglze/Nz;

%% 12. High Gray Level Zone Emphasis (HGLZE)

HGLZE= pg*(rVect.^2)';

% hglze=0;
% 
% for i=1:Ng
%     for j=1:Ns
%         hglze=hglze+(GLSZMatrix(i,j)*i^2);
%     end
% end
% hglze=hglze/Nz;

%% 13. Small Area Low Gray Level Emphasis (SALGLE)


% SALGLE=0;
% 
% for i=1:Ng
%     for j=1:Ns
%         SALGLE=SALGLE+(GLSZMatrix(i,j)/(i^2*j^2));
%     end
% end
% SALGLE=SALGLE/Nz;

SZLGE = sum(sum(norm_GLSZMatrix.*(rMat.^(-2)).*(cMat.^(-2))));

%% 14. Small Area High Gray Level Emphasis (SAHGLE)

% SAHGLE=0;
% 
% for i=1:Ng
%     for j=1:Ns
%         SAHGLE=SAHGLE+((GLSZMatrix(i,j)*(i^2))/(j^2));
%     end
% end
% SAHGLE=SAHGLE/Nz;

SZHGE = sum(sum(norm_GLSZMatrix.*(rMat.^2).*(cMat.^(-2))));

%% 15. Large Area Low Gray Level Emphasis (LALGLE)

% LALGLE=0;
% 
% for i=1:Ng
%     for j=1:Ns
%         LALGLE=LALGLE+((GLSZMatrix(i,j)*(j^2))/(i^2));
%     end
% end
% LALGLE=LALGLE/Nz;

LZLGE = sum(sum(norm_GLSZMatrix.*(rMat.^(-2)).*(cMat.^2)));

%% 16. Large Area High Gray Level Emphasis (LAHGLE)

% LAHGLE=0;
% 
% for i=1:Ng
%     for j=1:Ns
%         LAHGLE=LAHGLE+(GLSZMatrix(i,j)*(i^2*j^2));
%     end
% end
% LAHGLE=LAHGLE/Nz;

LZHGE = sum(sum(norm_GLSZMatrix.*(rMat.^2).*(cMat.^2)));


