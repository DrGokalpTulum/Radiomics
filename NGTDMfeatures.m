function [NGTDMFeature_vector]=NGTDMfeatures(ROIonly)


%% Neighbouring Gray Tone Difference Matrix (NGTDM)
ROIonly=round(ROIonly);


norm_ROIonly=ROIonly-min(min(ROIonly))+1;
nan_olmayan_voksel_sayisi=size(norm_ROIonly,1)*size(norm_ROIonly,2)-length(norm_ROIonly(isnan(ROIonly)));

for k=min(min(norm_ROIonly)):max(max(norm_ROIonly))
    lojik_norm_ROIonly=norm_ROIonly==k;
    n_i(k)=length(find(lojik_norm_ROIonly==1));
    p_i(k)=n_i(k)/nan_olmayan_voksel_sayisi;
    
    [x,y]=find(lojik_norm_ROIonly==1);
    
    if n_i(k)==0
        s_i(k)=0;
    else
        s_ara=0;
        for i=1:length(x)
            
            if x(i)==1 && y(i)==1
                komsu=[norm_ROIonly(x(i),y(i)+1)  norm_ROIonly(x(i)+1,y(i)) norm_ROIonly(x(i)+1,y(i)+1)];
                
                uzun=length(komsu)-length(komsu(isnan(komsu)));
                
            elseif x(i)==1 && y(i)==size(norm_ROIonly,2)
                komsu=[norm_ROIonly(x(i),y(i)-1) norm_ROIonly(x(i)+1,y(i)-1) norm_ROIonly(x(i)+1,y(i)) ];
                
                uzun=length(komsu)-length(komsu(isnan(komsu)));
                
            elseif x(i)==size(norm_ROIonly,1) && y(i)==size(norm_ROIonly,2)
                komsu=[norm_ROIonly(x(i)-1,y(i)-1) norm_ROIonly(x(i)-1,y(i))  norm_ROIonly(x(i),y(i)-1)];
                
                uzun=length(komsu)-length(komsu(isnan(komsu)));
                
            elseif x(i)==size(norm_ROIonly,1) && y(i)==size(norm_ROIonly,2)
                komsu=[norm_ROIonly(x(i)-1,y(i)-1) norm_ROIonly(x(i)-1,y(i))  norm_ROIonly(x(i),y(i)-1)];
                
                uzun=length(komsu)-length(komsu(isnan(komsu)));
                
            elseif x(i)==size(norm_ROIonly,1) && y(i)==size(norm_ROIonly,2)
                komsu=[norm_ROIonly(x(i)-1,y(i)-1) norm_ROIonly(x(i)-1,y(i))  norm_ROIonly(x(i),y(i)-1)];
                
                uzun=length(komsu)-length(komsu(isnan(komsu)));
                
            elseif x(i)==size(norm_ROIonly,1)  && y(i)==1
                komsu=[norm_ROIonly(x(i)-1,y(i)) norm_ROIonly(x(i)-1,y(i)+1) norm_ROIonly(x(i),y(i)+1)];
                
                uzun=length(komsu)-length(komsu(isnan(komsu)));
                
            elseif x(i)==1 && y(i)~=size(norm_ROIonly,2)
                komsu=[norm_ROIonly(x(i),y(i)-1) norm_ROIonly(x(i),y(i)+1) norm_ROIonly(x(i)+1,y(i)-1) norm_ROIonly(x(i)+1,y(i)) norm_ROIonly(x(i)+1,y(i)+1)];
                
                uzun=length(komsu)-length(komsu(isnan(komsu)));
                
                    
            elseif x(i)~=size(norm_ROIonly,1)  && y(i)==1
                komsu=[ norm_ROIonly(x(i)-1,y(i)) norm_ROIonly(x(i)-1,y(i)+1) ...
                    norm_ROIonly(x(i),y(i)+1)  norm_ROIonly(x(i)+1,y(i)) norm_ROIonly(x(i)+1,y(i)+1)];
                
                uzun=length(komsu)-length(komsu(isnan(komsu)));
                
                    
            elseif x(i)==size(norm_ROIonly,1) && y(i)~=size(norm_ROIonly,2)
                
                komsu=[norm_ROIonly(x(i)-1,y(i)-1) norm_ROIonly(x(i)-1,y(i)) norm_ROIonly(x(i)-1,y(i)+1) norm_ROIonly(x(i),y(i)-1)...
                    norm_ROIonly(x(i),y(i)+1) ];
                
                uzun=length(komsu)-length(komsu(isnan(komsu)));
                
                   
            elseif x(i)~=1  && y(i)==size(norm_ROIonly,2)
                komsu=[norm_ROIonly(x(i)-1,y(i)-1) norm_ROIonly(x(i)-1,y(i))  norm_ROIonly(x(i),y(i)-1)...
                    norm_ROIonly(x(i)+1,y(i)-1) norm_ROIonly(x(i)+1,y(i)) ];
                
                uzun=length(komsu)-length(komsu(isnan(komsu)));
                
          
            else
                
                
                komsu=[norm_ROIonly(x(i)-1,y(i)-1) norm_ROIonly(x(i)-1,y(i)) norm_ROIonly(x(i)-1,y(i)+1) norm_ROIonly(x(i),y(i)-1)...
                    norm_ROIonly(x(i),y(i)+1) norm_ROIonly(x(i)+1,y(i)-1) norm_ROIonly(x(i)+1,y(i)) norm_ROIonly(x(i)+1,y(i)+1)];
                
                uzun=length(komsu)-length(komsu(isnan(komsu)));
            end
            
                komsu(isnan(komsu))=0;
                s_ara=s_ara+abs(k-sum(komsu)/uzun);
            end
            s_i(k)=s_ara;
        end
        
        
    end
NGTDMmatr=[min(min(norm_ROIonly)):max(max(norm_ROIonly));n_i;p_i;s_i]';

countValid=NGTDMmatr(:,2);
NGTDM=NGTDMmatr(:,4);

nTot = sum(countValid); 
countValid = countValid./nTot; % Now representing the probability of gray-level occurences
NL = length(NGTDM);
Ng = sum(countValid~=0);
pValid = find(countValid>0);
nValid = length(pValid);


N_vp=sum(countValid);
N_gp = sum(countValid~=0);
N_g=length(countValid);


%% 1. Coarseness 
NGTD_Coarseness = (((countValid')*NGTDM) + eps)^(-1);

% Coarseness=(sum(countValid.*NGTDM))^(-1)

%% 2. Contrast 
val = 0;
for i = 1:NL
    for j = 1:NL
        val = val + countValid(i)*countValid(j)*(i-j)^2;
    end
end
NGTD_Contrast = val*sum(NGTDM)/(Ng*(Ng-1)*nTot);

%% 3. Busyness

denom = 0;
for i = 1:nValid
    for j = 1:nValid
        denom = denom + abs(pValid(i)*countValid(pValid(i))-pValid(j)*countValid(pValid(j)));
    end
end
NGTD_Busyness = ((countValid')*NGTDM)/denom;


%% 4. Complexity
val = 0;
for i = 1:nValid
    for j = 1:nValid
        val = val + (abs(pValid(i)-pValid(j))/(nTot*(countValid(pValid(i)) + countValid(pValid(j)))))*(countValid(pValid(i))*NGTDM(pValid(i)) + countValid(pValid(j))*NGTDM(pValid(j)));
    end
end
NGTD_Complexity = val;

%% 5. Strength
val = 0;
for i = 1:nValid
    for j = 1:nValid
        val = val + (countValid(pValid(i))+countValid(pValid(j)))*(pValid(i)-pValid(j))^2;
    end
end
NGTD_Strength = val/(eps+sum(NGTDM));

%%
NGTDMFeature_vector=[NGTD_Coarseness, NGTD_Contrast, NGTD_Busyness,NGTD_Complexity,NGTD_Strength];