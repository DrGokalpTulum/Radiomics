clear all;close all

KHUCRE=xlsread('D:\AREL PC\F SURUCUSU\FERHAT\radiomics\DATABASE\FEATURE MATRIX1.xlsx',2);
KHUCRE(1:4,:)=[];
SQUMAZ=xlsread('D:\AREL PC\F SURUCUSU\FERHAT\radiomics\DATABASE\FEATURE MATRIX1.xlsx',3);
SQUMAZ(1:4,:)=[];
ADENO = xlsread('D:\AREL PC\F SURUCUSU\FERHAT\radiomics\DATABASE\FEATURE MATRIX1.xlsx',1);
ADENO(1:4,:)=[];
ADENO(89,:)=[];


p = randperm(size(ADENO,1),(size(ADENO,1)-size(KHUCRE,1)));
ADENO(p,:)=[];



M=[ADENO;KHUCRE;SQUMAZ];

X=M(:,1:end-1);
 My=X(:,[16   22   270   310   565   579]);

figure
[R,P]=corrcoef(My);
n=6
L=['feat1';'feat2'; 'feat3'; 'feat4' ;'feat5' ;'feat6']
imagesc(R); % plot the matrix
set(gca, 'XTick', 1:n); % center x-axis ticks on bins
set(gca, 'YTick', 1:n); % center y-axis ticks on bins
set(gca, 'XTickLabel', L); % set x-axis labels
set(gca, 'YTickLabel', L); % set y-axis labels
title('Your Title Here', 'FontSize', 14); % set title
colormap('jet'); % set the colorscheme
% colorbar on; % enable colorbar

%%
norm_data = (My - repmat(min(My),222,1)) ./ ( repmat(max(My),222,1) - repmat(min(My),222,1)) ;

m1=mean(norm_data(1:75,:));
m2=mean(norm_data(76:150,:));
m3=mean(norm_data(151:end,:));


std1=std(norm_data(1:75,:));
std2=std(norm_data(76:150,:));
std3=std(norm_data(151:end,:));


m1plusstd1=m1+std1;
m1minusstd1=m1-std1;

m2plusstd2=m2+std2;
m2minusstd2=m2-std2;

m3plusstd3=m3+std3;
m3minusstd3=m3-std3;

figure;plot(m1);hold on; plot(m2,'r');plot(m3,'g');plot(m1plusstd1,':');plot(m1minusstd1,':');
plot(m2plusstd2,':r');plot(m2minusstd2,':r');plot(m3plusstd3,':g');plot(m3minusstd3,':g');



x1 = [[1 2 3 4 5 6], fliplr([1 2 3 4 5 6])];
inBetween = [m1plusstd1, fliplr(m1minusstd1)];
% fill(x2, inBetween,'r', 'EdgeColor','none');
    patch(x1,inBetween,1,'FaceColor','b','EdgeColor','none');
    
 x2 = [[1 2 3 4 5 6], fliplr([1 2 3 4 5 6])];
inBetween = [m2plusstd2, fliplr(m2minusstd2)];
% fill(x2, inBetween,'r', 'EdgeColor','none');
    patch(x2,inBetween,1,'FaceColor','r','EdgeColor','none');   

 x3 = [[1 2 3 4 5 6], fliplr([1 2 3 4 5 6])];
inBetween = [m3plusstd3, fliplr(m3minusstd3)];
% fill(x2, inBetween,'r', 'EdgeColor','none');
    patch(x3,inBetween,1,'FaceColor','g','EdgeColor','none'); 


alpha(.05);
grid on
    n=6
    L=['f01';'f02'; 'f03'; 'f04' ;'f05' ;'f06']
    set(gca, 'XTick', 1:n); % center x-axis ticks on bins
    set(gca, 'XTickLabel', L); % set x-axis labels

%% boxplot
norm_data = (My - repmat(min(My),222,1)) ./ ( repmat(max(My),222,1) - repmat(min(My),222,1)) ;


subplot(1,6,1)
boxplot((norm_data(:,1)))

subplot(1,6,2)
boxplot((norm_data(:,2)))

subplot(1,6,3)
boxplot((norm_data(:,3)))

subplot(1,6,4)
boxplot((norm_data(:,4)))

subplot(1,6,5)
boxplot((norm_data(:,5)))

subplot(1,6,6)
boxplot((norm_data(:,6)))




