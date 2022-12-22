clear all;close all

KHUCRE=xlsread('E:\FERHAT\radiomics\DATABASE\FEATURE MATRIX1.xlsx',2);
KHUCRE(1:4,:)=[];
SQUMAZ=xlsread('E:\FERHAT\radiomics\DATABASE\FEATURE MATRIX1.xlsx',3);
SQUMAZ(1:4,:)=[];
ADENO = xlsread('E:\FERHAT\radiomics\DATABASE\FEATURE MATRIX1.xlsx',1);
ADENO(1:4,:)=[];
ADENO(89,:)=[];


p = randperm(size(ADENO,1),(size(ADENO,1)-size(KHUCRE,1)));
ADENO(p,:)=[];



M=[ADENO;SQUMAZ;KHUCRE];

X=M(:,1:end-1);
y=M(:,end);


% % P-value .005 altýndakilerin belirlenmesi
% dataTrainG1 = X((y==0),:);
% dataTrainG2 = X((y==1),:);
% dataTrainG3 = X((y==2),:);
% 
% % [h,p,ci,stat] = ttest2(dataTrainG1,dataTrainG2,dataTrainG3,'Vartype','unequal');
% % 
% % [h,p] = ttest(dataTrainG1,dataTrainG2,dataTrainG3,'Vartype','unequal');
% for k=1:size(X,2)
%     
% [p,~,~] = anova1(X(:,k),y,'off');
% pval(k)=p;
% end
% 
% 
% ecdf(pval);
% xlabel('P value');
% ylabel('CDF value')
% [deger,featureIdxSortbyP] = sort(pval,2); % sort the features
% deger_logic=deger<.05;
% % % 
% % 
% %% Corrcoef -0.7<featur<0.7 olanlarýn belirlenmesi
% 
% featureIafterP=featureIdxSortbyP(1:sum(deger_logic))';
% psonrasimatrix=X(:,[featureIafterP]);
% 
% korelasyon_katsayi=corrcoef(X(:,[featureIafterP]));
% 
% korelasyon_katsayi_um=korelasyon_katsayi;
% for k=1:size(korelasyon_katsayi,2)
%     for m=1:k
%         korelasyon_katsayi_um(k,m)=NaN;
%     end
% end
% 
% kucuk=find(abs(korelasyon_katsayi_um(1,:))<.7);
% 
% buyuk=find(abs(korelasyon_katsayi_um(1,:))>.7);
% korelasyon_katsayi_um(:,buyuk)=NaN;
% NotANumber_olmayan_dizi=find(abs(isnan(korelasyon_katsayi_um(1,:))-1)==1);
% 
% k=1;
% art=1;
% while k==1
% 
% buyuk=find(abs(korelasyon_katsayi_um(NotANumber_olmayan_dizi(art),:))>.7);
% korelasyon_katsayi_um(:,buyuk)=NaN;
% NotANumber_olmayan_dizi=find(abs(isnan(korelasyon_katsayi_um(1,:))-1)==1);
% art=art+1;
% 
% if art>length(NotANumber_olmayan_dizi)
%     k=2;
% end
% end
% 
% 
% KoralsyonKatsayisiSifirYediAlti=[1 NotANumber_olmayan_dizi];
% featureIafterSifirYediAlti=featureIafterP(KoralsyonKatsayisiSifirYediAlti);
% 
% yeni_matr=X(:,[featureIafterSifirYediAlti]);
% 
% korelasyon_katsayi_SifirYediAlti=corrcoef(yeni_matr);

% figure
% n=22
% L=['f01'; 'f02'; 'f03'; 'f04' ; 'f05'; 'f06'; 'f07'; 'f08'; 'f09'; 'f10'; 'f11'; 'f12'; 'f13'; 'f14'; 'f15'; 'f16'; 'f17'; 'f18'; 'f19'; 'f20'; 'f21'; 'f22']
% imagesc(korelasyon_katsayi_SifirYediAlti'); % plot the matrix
% set(gca, 'XTick', 1:n); % center x-axis ticks on bins 
% set(gca, 'YTick', 1:n); % center y-axis ticks on bins
% set(gca, 'XTickLabel', L); % set x-axis labels
% set(gca, 'YTickLabel', L); % set y-axis labels
% colormap('jet'); % set the colorscheme
% % colorbar on; % enable colorbar



% % svm ile feature seçim
% % y=abs(y-1);
% c = cvpartition(y,'k',10);
% opts = statset('Display','iter');
% fun = @(XT,yT,Xt,yt)loss(fitcecoc(XT,yT),Xt,yt);
% [fs,history] = sequentialfs(fun,yeni_matr,y,'cv',c,'options',opts)
% 
% FeatureSelection_son=yeni_matr(:,[fs]);
% 
% FeatureSelection_yer=featureIafterSifirYediAlti(fs)
% 
% % [22    16   382   361   486    17   697    86];
% % 22   118   394   578   408   319 gerçek
% 
% % [22    16   565   270   579   310] 7mart 21
%  My=X(:,[16   22   270   310   565   579]);
% figure
% [R,P]=corrcoef(My);
% n=size(My,2);
% L=['feat1';'feat2'; 'feat3'; 'feat4' ;'feat5';'feat6']
% imagesc(R); % plot the matrix
% set(gca, 'XTick', 1:n); % center x-axis ticks on bins
% set(gca, 'YTick', 1:n); % center y-axis ticks on bins
% set(gca, 'XTickLabel', L); % set x-axis labels
% set(gca, 'YTickLabel', L); % set y-axis labels
% title('Your Title Here', 'FontSize', 14); % set title
% colormap('jet'); % set the colorscheme
% colorbar on; % enable colorbar




% % % % % % % % % % % % % % % % % % % % MLP sýnýflandýrma
% % % % % % % % % % % % % % % % % % % My=X(:,[16   22   270   310   565   579]);
% % % % % % % % % % % % % % % % % % % % My=FeatureSelection_son;
% % % % % % % % % % % % % % % % % % % boyut_bilgisi=zeros(1,size(My,1));
% % % % % % % % % % % % % % % % % % % X=My;
% % % % % % % % % % % % % % % % % % % yy=zeros(3,length(y));
% % % % % % % % % % % % % % % % % % % yy(:,find(y==0))=repmat([1; 0; 0],1,length(find(y==0)));
% % % % % % % % % % % % % % % % % % % yy(:,find(y==1))=repmat([0; 1; 0],1,length(find(y==1)));
% % % % % % % % % % % % % % % % % % % yy(:,find(y==2))=repmat([0;0; 1],1,length(find(y==2)));
% % % % % % % % % % % % % % % % % % % yy=yy';
% % % % % % % % % % % % % % % % % % % yy=y';
% % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % cikisy=[];
% % % % % % % % % % % % % % % % % % % % % sonuc=[];
% % % % % % % % % % % % % % % % % % % % % boyut_bilgisi_kfold=[];
% % % % % % % % % % % % % % % % % % % % % indices = crossvalind('Kfold',y,10);
% % % % % % % % % % % % % % % % % % % % %     net = patternnet([8 6]);
% % % % % % % % % % % % % % % % % % % % %     net.trainFcn = 'trainlm'
% % % % % % % % % % % % % % % % % % % % %     net.trainParam.lr=0.3;
% % % % % % % % % % % % % % % % % % % % %     net.trainParam.mc=0.6;
% % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % %     
% % % % % % % % % % % % % % % % % % % % % mdeg=5;
% % % % % % % % % % % % % % % % % % % % % for m=1:mdeg
% % % % % % % % % % % % % % % % % % % % %     indices = crossvalind('Kfold',y,10);
% % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % cikisy=[];
% % % % % % % % % % % % % % % % % % % % % sonuc=[];
% % % % % % % % % % % % % % % % % % % % %     for i = 1:10
% % % % % % % % % % % % % % % % % % % % %         test = (indices == i); traini = ~test;
% % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % %         [trainInd,valInd,testInd] = dividetrain(size(X(traini),1));
% % % % % % % % % % % % % % % % % % % % %         Xx=X(traini,:)';
% % % % % % % % % % % % % % % % % % % % %         Yy=yy(:,traini);
% % % % % % % % % % % % % % % % % % % % %         p1=X(test,:)';
% % % % % % % % % % % % % % % % % % % % %         [net,tr] = train(net,Xx,Yy);
% % % % % % % % % % % % % % % % % % % % %         [Y,Pf,Af,E,perf] =sim(net,p1);
% % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % %     cikisy(:,size(cikisy,2)+1:(sum(test,1)+size(cikisy,2)))=yy(:,test); 
% % % % % % % % % % % % % % % % % % % % %     sonuc(:,size(sonuc,2)+1:(size(Y,2)+size(sonuc,2)))=Y; 
% % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % %     boyut_bilgisi_kfold(:,length(boyut_bilgisi_kfold)+1:(sum(test)+length(boyut_bilgisi_kfold)))=boyut_bilgisi(test);
% % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % %     end
% % % % % % % % % % % % % % % % % % % % %     
% % % % % % % % % % % % % % % % % % % % %  testIndices = vec2ind(sonuc);
% % % % % % % % % % % % % % % % % % % % %  
% % % % % % % % % % % % % % % % % % % % % [c,cm]= confusion(cikisy,sonuc)
% % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));
% % % % % % % % % % % % % % % % % % % % % fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);
% % % % % % % % % % % % % % % % % % % % % plotroc(cikisy,sonuc)
% % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % %    
% % % % % % % % % % % % % % % % % % % % % end

%% MLP sýnýflandýrma 
My=X(:,[16   22   270   310   565   579]);
% My=FeatureSelection_son;
boyut_bilgisi=zeros(1,size(My,1));
X=My;
yy=zeros(3,length(y));
yy(:,find(y==0))=repmat([1; 0; 0],1,length(find(y==0)));
yy(:,find(y==2))=repmat([0; 1; 0],1,length(find(y==2)));
yy(:,find(y==1))=repmat([0;0; 1],1,length(find(y==1)));
% yy=yy';
% yy=y';
cikisy=[];
sonuc=[];
boyut_bilgisi_kfold=[];
indices = crossvalind('Kfold',y,10);
net = patternnet([12 9]);
net.trainFcn = 'trainlm'
net.trainParam.lr=0.4;
net.trainParam.mc=0.7;
% mdeg=5;
% for m=1:mdeg
%     indices = crossvalind('Kfold',y,10);
cikisy=[];
sonuc=[];
for i = 1:10
test = (indices == i); traini = ~test;
%         [trainInd,valInd,testInd] = dividetrain(size(X(traini),1));
Xx=X(traini,:)';
Yy=yy(:,traini);
p1=X(test,:)';
[net,tr] = train(net,Xx,Yy);
[Y,Pf,Af,E,perf] =sim(net,p1);
cikisy(:,size(cikisy,2)+1:(sum(test,1)+size(cikisy,2)))=yy(:,test);
sonuc(:,size(sonuc,2)+1:(size(Y,2)+size(sonuc,2)))=Y;
%     boyut_bilgisi_kfold(:,length(boyut_bilgisi_kfold)+1:(sum(test)+length(boyut_bilgisi_kfold)))=boyut_bilgisi(test);
end
testIndices = vec2ind(sonuc);
[c,cm]= confusion(cikisy,sonuc)
fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));
fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);
plotroc(cikisy,sonuc)
[X1,Y1,T1,AUC1,OPTROCPT1] = perfcurve(cikisy(1,:),sonuc(1,:),'1');
[X2,Y2,T2,AUC2,OPTROCPT2] = perfcurve(cikisy(2,:),sonuc(2,:),'1');
[X3,Y3,T3,AUC3,OPTROCPT3] = perfcurve(cikisy(3,:),sonuc(3,:),'1');
[c,cm]= confusion(cikisy(3,:),sonuc(3,:)) %ova
grid on


%% SVM sýnýflandýrma (ova ile yaptigim)
% 
My=X(:,[16   22   270   310   565   579]);



classNames = [0 1 2];
numClasses = size(classNames,2);
inds = cell(3,10); % Preallocation
inds_test=cell(3,10);
SVMModel = cell(3,10);
posteriortest = cell(3,10); 
posteriortrain = cell(3,10); 

label=cell(3,10);
scores=cell(3,10);
predstest=cell(3,10);
predstrain=cell(3,10);

indices = crossvalind('Kfold',y,10);


for i=1:10
    test = (indices == i); 
    train = ~test;
    y_egitim=y(train);
    y_test=y(test);
for j = 1:numClasses

    dizi=zeros(sum(train),1);
    dizi(find(y_egitim==(j-1)))=1;
    inds{j,i} = dizi;  % OVA classification
    
    dizitest=zeros(sum(test),1);
    dizitest(find(y_test==(j-1)))=1;
    inds_test{j,i} = dizitest;  % OVA classification
    
%     SVMModel{j,i} = fitcsvm(My(train,:),inds{j,i},'ClassNames',[false true],...
%         'Standardize',true,'KernelFunction','rbf','KernelScale','auto');
    SVMModel{j,i} = fitcsvm(My(train,:),inds{j,i},'ClassNames',[false true],...
        'Standardize',true,'KernelFunction','rbf');


    SVMModel{j,i} = fitPosterior(SVMModel{j,i});

    [predstest{j,i},posteriortest{j,i}] = predict(SVMModel{j,i},My(test,:));
    [predstrain{j,i},posteriortrain{j,i}] = predict(SVMModel{j,i},My(train,:));

    
    [label{j,i},scores{j,i}] = resubPredict(SVMModel{j,i});
    
end
% clear dizi
end

POSTERIO_test1=[double(posteriortest{1,1});double(posteriortest{1,2});double(posteriortest{1,3})...
    ;double(posteriortest{1,4});double(posteriortest{1,5});double(posteriortest{1,6})...
    ;double(posteriortest{1,7});double(posteriortest{1,8});double(posteriortest{1,9});double(posteriortest{1,10})];

POSTERIO_test2=[double(posteriortest{2,1});double(posteriortest{2,2});double(posteriortest{2,3})...
    ;double(posteriortest{2,4});double(posteriortest{2,5});double(posteriortest{2,6})...
    ;double(posteriortest{2,7});double(posteriortest{2,8});double(posteriortest{2,9});double(posteriortest{2,10})];

POSTERIO_test3=[double(posteriortest{3,1});double(posteriortest{3,2});double(posteriortest{3,3})...
    ;double(posteriortest{3,4});double(posteriortest{3,5});double(posteriortest{3,6})...
    ;double(posteriortest{3,7});double(posteriortest{3,8});double(posteriortest{3,9});double(posteriortest{3,10})];

INDS_test1=[double(inds_test {1,1});double(inds_test {1,2});double(inds_test {1,3})...
    ;double(inds_test {1,4});double(inds_test {1,5});double(inds_test {1,6})...
    ;double(inds_test {1,7});double(inds_test {1,8});double(inds_test {1,9});double(inds_test {1,10})];

INDS_test2=[double(inds_test {2,1});double(inds_test {2,2});double(inds_test {2,3})...
    ;double(inds_test {2,4});double(inds_test {2,5});double(inds_test {2,6})...
    ;double(inds_test {2,7});double(inds_test {2,8});double(inds_test {2,9});double(inds_test {2,10})];

INDS_test3=[double(inds_test {3,1});double(inds_test {3,2});double(inds_test {3,3})...
    ;double(inds_test {3,4});double(inds_test {3,5});double(inds_test {3,6})...
    ;double(inds_test {3,7});double(inds_test {3,8});double(inds_test {3,9});double(inds_test {3,10})];

PRED_test1=[double(predstest {1,1});double(predstest {1,2});double(predstest {1,3})...
    ;double(predstest {1,4});double(predstest {1,5});double(predstest {1,6})...
    ;double(predstest {1,7});double(predstest {1,8});double(predstest {1,9});double(predstest {1,10})];

PRED_test2=[double(predstest {2,1});double(predstest {2,2});double(predstest {2,3})...
    ;double(predstest {2,4});double(predstest {2,5});double(predstest {2,6})...
    ;double(predstest {2,7});double(predstest {2,8});double(predstest {2,9});double(predstest {2,10})];

PRED_test3=[double(predstest {3,1});double(predstest {3,2});double(predstest {3,3})...
    ;double(predstest {3,4});double(predstest {3,5});double(predstest {3,6})...
    ;double(predstest {3,7});double(predstest {3,8});double(predstest {3,9});double(predstest {3,10})];

% degisken=1
% SKORD1=double(scores{1,degisken});
% POSTE1=double(posteriortest{1,degisken});

 [xLinSVM1,yLinSVM1,~,aucLinSVM1] = perfcurve(INDS_test1,POSTERIO_test1(:,2),1);
% [xLinSVM1,yLinSVM1,~,aucLinSVM1] = perfcurve(double(inds_test{1,degisken}),POSTE1(:,2),1);

% [c,cm]= confusion(double(inds{1,degisken}),SKORD1(:,2))

 
%  SKORD2=double(scores{2,degisken});
 [xLinSVM2,yLinSVM2,~,aucLinSVM2] = perfcurve(INDS_test2,POSTERIO_test2(:,2),1);
 
%   SKORD3=double(scores{3,degisken});
%  [xLinSVM3,yLinSVM3,~,aucLinSVM3] = perfcurve(double(inds{3,degisken}),SKORD3(:,2),1);
   [xLinSVM3,yLinSVM3,~,aucLinSVM3] = perfcurve(INDS_test3,POSTERIO_test3(:,2),1);

 plot(xLinSVM1,yLinSVM1,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, RBF SVM')

hold on

 plot(xLinSVM2,yLinSVM2,'LineWidth',2)
 plot(xLinSVM3,yLinSVM3,'LineWidth',2)
  [csvm3,cmsvm3]= confusion(INDS_test3',PRED_test3') %ova
   [csvm1,cmsvm1]= confusion(INDS_test1',PRED_test1') %ova
  [csvm2,cmsvm2]= confusion(INDS_test2',PRED_test2') %ova

%% svm (tek tek baktigim 3 sinif oldugu için y yi 3 kez deðistirip 3 kez çalýstirdim [1 0 0] [0 1 0] [0 0 1])

% KHUCRE=xlsread('D:\AREL PC\F SURUCUSU\FERHAT\radiomics\DATABASE\FEATURE MATRIX1.xlsx',2);
% KHUCRE(1:4,:)=[];
% SQUMAZ=xlsread('D:\AREL PC\F SURUCUSU\FERHAT\radiomics\DATABASE\FEATURE MATRIX1.xlsx',3);
% SQUMAZ(1:4,:)=[];
% ADENO = xlsread('D:\AREL PC\F SURUCUSU\FERHAT\radiomics\DATABASE\FEATURE MATRIX1.xlsx',1);
% ADENO(1:4,:)=[];
% ADENO(89,:)=[];
% 
% 
% p = randperm(size(ADENO,1),(size(ADENO,1)-size(KHUCRE,1)));
% ADENO(p,:)=[];
% 
% 
% 
% M=[ADENO;KHUCRE;SQUMAZ];
% 
% X=M(:,1:end-1);
% 
% 
% trainingData=X(:,[16   22   270   310   565   579]);
% y=[ones(75,1); zeros(75,1); zeros(72,1)];
% indices = crossvalind('Kfold',y,10);
% 
% for i = 1:10
%     test = (indices == i); 
%     train = ~test;
%     
%     % Linear SVM
%     classificationLinearSVM128 = fitcsvm(trainingData(train,1:end),...
%     y(train), ...
%     'KernelFunction', 'RBF', ...
%     'PolynomialOrder', [], ...
%     'KernelScale', 'auto', ...
%     'BoxConstraint', 1, ...
%     'Standardize', true, ...
%     'ClassNames', [0; 1]);
% % classificationLinearSVM128 = fitcsvm(trainingData(train,1:end),...
% %     y(train), ...
% %     'KernelFunction', 'RBF','Standardize', true, ...
% % 'ClassNames', [0; 1]);
% 
% 
%     % Training
%     [predsLinSVM128train,~] = predict(classificationLinearSVM128,trainingData(train,1:end));
%     targetsLinSVM128train = y(train,end);
%     [~,scoresLinSVM128] = resubPredict(fitPosterior(classificationLinearSVM128));
%     [xLinSVM128,yLinSVM128,~,aucLinSVM128] = perfcurve(y(train,end),scoresLinSVM128(:,2),1);
% %     [xLinSVM128test,yLinSVM128test,~,aucLinSVM128test] = perfcurve(y(test,end),scoresLinSVM128(:,2),1);
% 
%     % Validation
%     [predsLinSVM128test,~] = predict(classificationLinearSVM128,trainingData(test,1:end));
%     targetsLinSVM128test = y(test,end);
%        
%     
% 
%     [~,scoresLinSVM128test] = resubPredict(fitPosterior(classificationLinearSVM128));
% %     [xLinSVM128test,yLinSVM128test,~,aucLinSVM128test] = perfcurve(trainingData(test,end),scoresLinSVM128test(:,2),1);
% 
% end
% % 
% figure()
% subplot(121)
% confusionchart(targetsLinSVM128train,predsLinSVM128train)
% title('Linear SVM, training')
% subplot(122)
% confusionchart(targetsLinSVM128test,predsLinSVM128test)
% title('Linear SVM, validation')
% 
% figure()
% plot(xLinSVM128,yLinSVM128,'LineWidth',2)
% xlabel('False Positive Rate')
% ylabel('True Positive Rate')
% title('ROC, RBF SVM')

 
%% 
%  My=X(:,[16   22   270   310   565   579]);
% y=y-1;
trainingData=My;
t = templateSVM('KernelFunction','rbf');
t = templateSVM('KernelFunction', 'RBF', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [-1;0; 1]);
mdl = fitcecoc(trainingData, y,'Learners',t,'Coding','onevsall','CrossVal','on');
% CVMdl = crossval(Mdl);
label = kfoldPredict(mdl);
C = confusionchart(y,label)











%% kullanmadýgým





% % % % % indices = crossvalind('Kfold',y,10);
% % % % % cikisy=[];
% % % % % pred_label=[];
% % % % % skor=[];
% % % % % kost=[];
% % % % % cikis_dizi=[];
% % % % % yy=zeros(3,length(y));
% % % % % yy(:,find(y==0))=repmat([1; 0; 0],1,length(find(y==0)));
% % % % % yy(:,find(y==1))=repmat([0; 1; 0],1,length(find(y==1)));
% % % % % yy(:,find(y==2))=repmat([0;0; 1],1,length(find(y==2)));
% % % % % % % 
% % % % % for i = 1:10
% % % % %     test = (indices == i); 
% % % % %     train = ~test;
% % % % %     
% % % % % t = templateSVM('KernelFunction','gaussian');
% % % % % mdl = fitcecoc(trainingData(train,1:end), y(train),'Learners',t,'Coding','onevsone');
% % % % % % classificationLinearSVM128 = fitcecoc(trainingData(train,1:end),...
% % % % % %     y(train));
% % % % % [predictedLabels,score,cost] = predict(mdl,trainingData(test,1:end));
% % % % % % targetsLinSVM128train = y(test,end);
% % % % % 
% % % % % cikisy(:,size(cikisy,2)+1:(sum(test,1)+size(cikisy,2)))=yy(:,test);
% % % % % cikis_dizi(:,size(cikis_dizi,2)+1:(sum(test,1)+size(cikis_dizi,2)))=y(test)';
% % % % % 
% % % % % pred_label(:,size(pred_label,2)+1:(size(predictedLabels,1)+size(pred_label,2)))=predictedLabels';
% % % % % skor(:,size(skor,2)+1:(size(score,1)+size(skor,2)))=score';
% % % % % kost(:,size(kost,2)+1:(size(cost,1)+size(kost,2)))=cost';
% % % % % 
% % % % % 
% % % % % % skord = double(skor(:,classificationLinearSVM128.ClassNames + 1))'; % Compute the posterior probabilities (scores)
% % % % % % [X,Y,T,AUC] = perfcurve(targetsLinSVM128train,scores);
% % % % % 
% % % % % end
% % % % % [c,cm]= confusion(cikis_dizi,pred_label)
% % % % skord = double(skor(:,classificationLinearSVM128.ClassNames + 1))'; % Compute the posterior probabilities (scores)
% % % % 
% % % % [X,Y,T,AUC] = perfcurve(cikisy,skor,1);
% % % % % %  %% SVM sýnýflandýrma
% % % % % % % % 
% % % % % % trainingData=My;
% % % % % % indices = crossvalind('Kfold',y,10);
% % % % % % 
% % % % % % for i = 1:10
% % % % % %     test = (indices == i); 
% % % % % %     train = ~test;
% % % % % %     
% % % % % %     % Linear SVM
% % % % % % %     classificationLinearSVM128 = fitcecoc(trainingData(train,1:end),...
% % % % % % %     y(train), ...
% % % % % % %     'KernelFunction', 'RBF', ...
% % % % % % %     'PolynomialOrder', [], ...
% % % % % % %     'KernelScale', 'auto', ...
% % % % % % %     'BoxConstraint', 1, ...
% % % % % % %     'Standardize', true, ...
% % % % % % %     'ClassNames', [0; 1;2]);
% % % % % % 
% % % % % % t = templateSVM('Standardize',true,'SaveSupportVectors',true);
% % % % % % % predictorNames = {'petalLength','petalWidth'};
% % % % % % % responseName = 'irisSpecies';
% % % % % % classNames = {'0','1','2'}; % Specify class order
% % % % % % classificationLinearSVM128 = fitcecoc(trainingData(train,1:end),y(train),'Learners',t,'FitPosterior',true);
% % % % % % 
% % % % % % 
% % % % % % 
% % % % % % 
% % % % % % 
% % % % % % 
% % % % % % 
% % % % % %     % Training
% % % % % %     
% % % % % %   [label,NegLoss,PBScore,Posterior] = predict(classificationLinearSVM128,trainingData(train,1:end));
% % % % % % 
% % % % % %     [predsLinSVM128train,~] = predict(classificationLinearSVM128,trainingData(train,1:end));
% % % % % %     targetsLinSVM128train = y(train,end);
% % % % % %     [~,scoresLinSVM128] = resubPredict(fitPosterior(classificationLinearSVM128));
% % % % % %     [xLinSVM128,yLinSVM128,~,aucLinSVM128] = perfcurve(y(train,end),scoresLinSVM128(:,2),1);
% % % % % %     
% % % % % % 
% % % % % %     % Validation
% % % % % %     [predsLinSVM128test,~] = predict(classificationLinearSVM128,trainingData(test,1:end));
% % % % % %     targetsLinSVM128test = y(test,end);
% % % % % %        
% % % % % %     
% % % % % % 
% % % % % %     [~,scoresLinSVM128test] = resubPredict(fitPosterior(classificationLinearSVM128));
% % % % % %     [xLinSVM128test,yLinSVM128test,~,aucLinSVM128test] = perfcurve(trainingData(test,end),scoresLinSVM128test(:,2),1);
% % % % % % 
% % % % % % end
% % % % % % % 
% % % % % % figure()
% % % % % % subplot(121)
% % % % % % confusionchart(targetsLinSVM128train,predsLinSVM128train)
% % % % % % title('Linear SVM, training')
% % % % % % subplot(122)
% % % % % % confusionchart(targetsLinSVM128test,predsLinSVM128test)
% % % % % % title('Linear SVM, validation')
% % % % % % 
% % % % % % figure()
% % % % % % plot(xLinSVM128,yLinSVM128,'LineWidth',2)
% % % % % % xlabel('False Positive Rate')
% % % % % % ylabel('True Positive Rate')
% % % % % % title('ROC, RBF SVM')
% % % % % % % % 
% % % % % % % % 
