clear all;close all

ADENO = xlsread('F:\FERHAT\radiomics\DATABASE\FEATURE MATRIX2.xlsx',1);
ADENO(1:4,:)=[];
ADENO(89,:)=[];
SQUMAZ=xlsread('F:\FERHAT\radiomics\DATABASE\FEATURE MATRIX2.xlsx',2);
SQUMAZ(1:4,:)=[];
KHUCRE=xlsread('F:\FERHAT\radiomics\DATABASE\FEATURE MATRIX2.xlsx',3);
KHUCRE(1:4,:)=[];





p = randperm(size(ADENO,1),(size(ADENO,1)-size(KHUCRE,1)));
ADENO(p,:)=[];



M=[ADENO;SQUMAZ;KHUCRE];

X=M(:,1:end-1);
y=M(:,end);
cp = classperf(y);                      %# init performance tracker


%% MLP sınıflandırma 
My=X(:,[16   22   270   310   565   579]);
% My=FeatureSelection_son;
boyut_bilgisi=zeros(1,size(My,1));
X=My;

yy=zeros(3,length(y));
yy(:,find(y==0))=repmat([1; 0; 0],1,length(find(y==0)));
yy(:,find(y==1))=repmat([0; 1; 0],1,length(find(y==1)));
yy(:,find(y==2))=repmat([0;0; 1],1,length(find(y==2)));
% yy=yy';
% yy=y';

for zz=1:10

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

%%
classNames = [0 1 2];
numClasses = size(classNames,2);
inds = cell(3,10); % Preallocation
inds_test=cell(3,10);
SVMModel = cell(3,10);
posteriortest = cell(3,10); 
posteriortraini = cell(3,10); 

label=cell(3,10);
scores=cell(3,10);
predstest=cell(3,10);
predstraini=cell(3,10);
%%
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
[c,cm_mlp]= confusion(yy(:,test),Y);
accur_mlp(i,zz)=100*(1-c);

[c_ac,cm]= confusion(yy(1,test),Y(1,:));
accur_mlp_ac(i)=100*(1-c_ac);

[c_scc,cm]= confusion(yy(2,test),Y(2,:));
accur_mlp_scc(i)=100*(1-c_scc);

[c_sclc,cm]= confusion(yy(3,test),Y(3,:));
accur_mlp_sclc(i,zz)=100*(1-c_sclc);
%%


    y_egitim=y(traini);
    y_test=y(test);
for j = 1:numClasses

    dizi=zeros(sum(traini),1);
    dizi(find(y_egitim==(j-1)))=1;
    inds{j,i} = dizi;  % OVA classification
    
    dizitest=zeros(sum(test),1);
    dizitest(find(y_test==(j-1)))=1;
    inds_test{j,i} = dizitest;  % OVA classification
    
%     SVMModel{j,i} = fitcsvm(My(train,:),inds{j,i},'ClassNames',[false true],...
%         'Standardize',true,'KernelFunction','rbf','KernelScale','auto');
    SVMModel{j,i} = fitcsvm(My(traini,:),inds{j,i},'ClassNames',[false true],...
        'Standardize',true,'KernelFunction','rbf');


    SVMModel{j,i} = fitPosterior(SVMModel{j,i});

    [predstest{j,i},posteriortest{j,i}] = predict(SVMModel{j,i},My(test,:));
    [predstraini{j,i},posteriortraini{j,i}] = predict(SVMModel{j,i},My(traini,:));

    
    [label{j,i},scores{j,i}] = resubPredict(SVMModel{j,i});

    t = templateSVM('KernelFunction', 'RBF', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [-1;0; 1]);
mdl1 = fitcecoc(My(traini,:), y(traini),'Learners',t);

    
    %# train an SVM model over training instances
%     svmModel = svmtrain(meas(trainIdx,:), groups(trainIdx), ...
%                  'Autoscale',true, 'Showplot',false, 'Method','QP', ...
%                  'BoxConstraint',2e-1, 'Kernel_Function','rbf', 'RBF_Sigma',1);

    %# test using test instances
    pred = predict(mdl1, My(test,:));

    %# evaluate and update performance object
    cp = classperf(cp, pred, test);
    a=cp.CorrectRate;
     accur_svm(i,zz)=a;
%     [c_ac,cm]= confusion(pred', y(test)')
%     accur_svm_ac(i)=100*(1-c_ac)

    
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

for m=1:10

[csvm1_ac,cmsvm1]= confusion(double(inds_test {1,m})',double(predstest {1,m})') %ova
accur_svm_ac(m)=100*(1-csvm1_ac);

[csvm2_scc,cmsvm2]= confusion(double(inds_test {2,m})',double(predstest {2,m})') %ova
accur_svm_scc(m)=100*(1-csvm2_scc);

[csvm2_sclc,cmsvm3]= confusion(double(inds_test {3,m})',double(predstest {3,m})') %ova
accur_svm_sclc(m,zz)=100*(1-csvm2_sclc);

end
end
%%









% My=X(:,[16   22   270   310   565   579]);
% y=M(:,end);
% 
% y=y-1;
% trainingData=My;
% % t = templateSVM('KernelFunction','rbf');
% t = templateSVM('KernelFunction', 'rbf', ...
%     'PolynomialOrder', [], ...
%     'KernelScale', 'auto', ...
%     'BoxConstraint', 1, ...
%     'Standardize', true, ...
%     'ClassNames', [-1;0; 1]);
% mdl = fitcecoc(trainingData, y,'Learners',t,'Coding','onevsall','CrossVal','on');
% [elabel,escore,cost] = kfoldPredict(mdl);
% C = confusionchart(y,elabel)
% 
% 
% % mdl = fitcecoc(trainingData, y,'Learners',t);
% % CVMdl = crossval(mdl);
% % [elabel,escore,cost] = kfoldPredict(CVMdl);
% 
% dizi=[1 mdl.ModelParameters.Generator.Partition.TestSize]
% 
% Y_matr=zeros(222,3);
% 
% Y_matr(find(y==-1),1)=1;
% Y_matr(find(y==0),2)=1;
% Y_matr(find(y==1),3)=1;
% 
% elabel_matr=zeros(222,3);
% 
% elabel_matr(find(elabel==-1),1)=1;
% elabel_matr(find(elabel==0),2)=1;
% elabel_matr(find(elabel==1),3)=1;
% 
% 
% basla=1;
% art=1;
% bit=dizi(2)
% for k=2:10
% 
%      [csvm,cmsvm]= confusion(Y_matr(basla:bit,:)',elabel_matr(basla:bit,:)')
%      accur_svm(k)=100*(1-csvm);
% 
%      
%     art=art+1;
% 
%     basla=basla+dizi(art)
%     bit=bit+dizi(art+1)
% end

[h,p,ci,stats] = ttest(reshape(accur_svm,100,1),reshape(accur_mlp,100,1))

% [h_ac,p_ac,ci_ac,stats_ac] = ttest(accur_mlp_ac,accur_svm_ac)
% [h_scc,p_scc,ci_scc,stats_scc] = ttest(accur_mlp_scc,accur_svm_scc)
[h_sclc,p_sclc,ci_sclc,stats_sclc] = ttest(reshape(accur_svm_sclc,100,1),reshape(accur_mlp_sclc,100,1))
   

