function [GLRLMFeature_vector]=GLRLMMfeatures(ROIonly)


%% Gray Level Run Length Matrix
ROIonly=round(ROIonly);

norm_ROIonly=ROIonly-min(min(ROIonly))+1;

norm_ROIonly(isnan(ROIonly)) = max(max(norm_ROIonly))+1;

[GLRLMS,SI] = grayrlmatrix(norm_ROIonly,'NumLevels',max(max(norm_ROIonly)),'G',[]);

for k=1:length(GLRLMS)
    glrlm= GLRLMS{1,k};
    glrlm(end,:) = [];
    stop = find(sum(glrlm),1,'last');
    glrlm(:,(stop+1):end) = [];
    
    GLRLM{1,k}=glrlm;
end


numStats = 11;
numGLRLM=length(GLRLM);

for p = 1 : numGLRLM
    %N-D indexing not allowed for sparse.
    tGLRLM = GLRLM{1,p};

    s = size(tGLRLM);
    c_vector =1:s(1);
    r_vector =1:s(2);
    
    [c_matrix,r_matrix] = meshgrid(c_vector,r_vector);

    N_runs = sum(sum(tGLRLM));

    N_tGLRLM = s(1)*s(2);

   
    p_g = sum(tGLRLM);

    p_r = sum(tGLRLM,2)';

   
    %------------------------Statistics-------------------------------
    %% 1. Short Run Emphasis (SRE)
    SRE(p) = sum(p_r./(c_vector.^2))/N_runs;
    
    %% 2. Long Run Emphasis (LRE)
    LRE(p)=sum(p_r.*(c_vector.^2))/N_runs;
    
    %% 3. Gray-Level Nonuniformity (GLN)
    GLN(p) = sum(p_g.^2)/N_runs;
    
    %% 4 Gray Level Non-Uniformity Normalized (GLNN)
    
    GLNN(p)=GLN(p)/N_runs;
    
    %% 5. Run Length Nonuniformity (RLN)
    RLN(p) = sum(p_r.^2)/N_runs;
    
    %% 6. Run Length Non-Uniformity Normalized (RLNN)
        RLNN(p) = RLN(p)/N_runs;

    
    %% 7. Run Percentage (RP)
    RP(p) = N_runs/N_tGLRLM;
    
    %% 8. Gray Level Variance (GLV)

    u_x=0;
    
    for i=1:s(1)
        for j=1:s(2)
            u_x=u_x+(tGLRLM(i,j)/N_runs)*i;
        end
    end
    
    glv=0;
    for i=1:s(1)
        for j=1:s(2)
            glv=glv+(tGLRLM(i,j)/N_runs)*(i-u_x)^2;
        end
    end
    
    GLV(p)=glv;
    
    %% 9. Run Variance (RV)
    
    u_y=0;
    
    for i=1:s(1)
        for j=1:s(2)
            u_y=u_y+(tGLRLM(i,j)/N_runs)*j;
        end
    end
    
    rv=0;
    for i=1:s(1)
        for j=1:s(2)
            rv=rv+(tGLRLM(i,j)/N_runs)*(j-u_y)^2;
        end
    end
    
    RV(p)=rv;
    
    %% 10. Rone Entropy (RE)

    re=0;
    
    for i=1:s(1)
        for j=1:s(2)
            
            re=re-((tGLRLM(i,j)/N_runs)*log((tGLRLM(i,j)/N_runs) + eps));
        end
    end
    RE(p)=re;
    
    %% 11. Low Gray-Level Run Emphasis (LGRE)
    LGRE(p) = sum(p_g./(r_vector.^2))/N_runs;
    
    %% 12. High Gray-Level Run Emphasis (HGRE)
    HGRE(p) = sum(p_g.*r_vector.^2)/N_runs;
    
    %% 13. Short Run Low Gray-Level Emphasis (SRLGE)
    %     SGLGE =calculate_SGLGE(tGLRLM,r_matrix',c_matrix',N_runs);
    term = tGLRLM./((r_matrix'.*c_matrix').^2);
    SGLGE(p)= sum(sum(term))./N_runs;
    clear term
        
    %% 14. Short Run High Gray-Level Emphasis (SRHGE)
    %     SRHGE =calculate_SRHGE(tGLRLM,r_matrix',c_matrix',N_runs);
    
    term1  = tGLRLM.*(r_matrix'.^2)./(c_matrix'.^2);
    SRHGE(p) = sum(sum(term1))/N_runs;
    clear term1
    
    %% 15. Long Run Low Gray-Level Emphasis (LRLGE)
    %     LRLGE =calculate_LRLGE(tGLRLM,r_matrix',c_matrix',N_runs);
    
    term2  = tGLRLM.*(c_matrix'.^2)./(r_matrix'.^2);
    LRLGE(p) = sum(sum(term2))/N_runs;
    clear term2
    
    %% 16.Long Run High Gray-Level Emphasis (LRHGE
    %     LRHGE =calculate_LRHGE(tGLRLM,r_matrix',c_matrix',N_runs);
    
    term3  = tGLRLM.*(c_matrix'.^2).*(r_matrix'.^2);
    LRHGE(p) = sum(sum(term3))/N_runs;
    clear term3
end  

    GLRLMFeature_vector=[mean(SRE) mean(LRE) mean(GLN) mean(GLNN) mean(RLN) mean(RLNN) mean(RP) mean(GLV) mean(RV) mean(RE) mean(LGRE) mean(HGRE) mean(SGLGE) mean(SRHGE) mean(LRLGE)  mean(LRHGE) ];
