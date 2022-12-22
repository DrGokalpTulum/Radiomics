% PRELIMINARY

function [enerji toplam_enerji entroppy minimum Tenthpercentile Ninethpercentile maksimum ortalama medyan Interquartile_Range menzil mad rmad rms stdev skewn kurt]=FirstOrderFeatures (ROIonly);
    
Nbins=max(max(ROIonly))-min(min(ROIonly));
vectorValid = ROIonly(~isnan(ROIonly));
histo = hist(vectorValid,Nbins);
histo = histo./(sum(histo(:)));
vectNg = 1:Nbins;
u = histo*vectNg';



% COMPUTATION OF TEXTURES

%%
NaNolmayanpikseller=ROIonly(~isnan(ROIonly));

voksel_deger=unique(NaNolmayanpikseller);
for k=1:length(voksel_deger)

    histogra(k)=length(find(NaNolmayanpikseller==voksel_deger(k)));
    vokseldeger(k)=voksel_deger(k); 
end
histogra=histogra/sum(histogra);

%1 Energy
enerji=sum(NaNolmayanpikseller.^2);

%2 Total Energy
toplam_enerji=volume_mm(1)*volume_mm(2)*volume_mm(3)*enerji;

%3 Entropy
entroppy = -sum(histogra.*log2(histogra));

%4 Minimum
minimum=min(voksel_deger);

% 5 Tenthpercentile
sirali=sort(NaNolmayanpikseller);

tenth=round(length(sirali)*.1);
Tenthpercentile=sirali(tenth);

% 6 Ninethpercentile
nineth=round(length(sirali)*.9);
Ninethpercentile=sirali(nineth);

%7 Maximum
maksimum=max(voksel_deger);

%8 Mean
ortalama=mean(NaNolmayanpikseller);

%9 Median
medyan=median(NaNolmayanpikseller);

%10 Interquartile Range
thirdquarter=round(length(sirali)*.75);
firstquarter=round(length(sirali)*.25);
Interquartile_Range=sirali(thirdquarter)-sirali(firstquarter);

%11 Range
menzil=maksimum-minimum;

%12 MAD
ortalamavektor=repmat(ortalama,length(NaNolmayanpikseller),1);
mad= (1/length(NaNolmayanpikseller))*sum(abs(NaNolmayanpikseller-ortalamavektor));

%13 rMAD
rmad=(1/length(sirali(tenth:nineth)))*sum(abs(sirali(tenth:nineth)-repmat(mean(sirali(tenth:nineth)),length(sirali(tenth:nineth)),1)));

%14 RMS
rms=sqrt((1/length(NaNolmayanpikseller))*sum(NaNolmayanpikseller.^2));

%15 STD

% stdev=sqrt((1/length(NaNolmayanpikseller))*sum((NaNolmayanpikseller-ortalamavektor).^2));
stdev=std(NaNolmayanpikseller);

%16 Skewness

skewn=((1/length(NaNolmayanpikseller))*sum((NaNolmayanpikseller-ortalamavektor).^3))/...
(sqrt((1/length(NaNolmayanpikseller))*sum((NaNolmayanpikseller-ortalamavektor).^2))).^3;

%17 Kurtosis

kurt=((1/length(NaNolmayanpikseller))*sum((NaNolmayanpikseller-ortalamavektor).^4))/...
(sqrt((1/length(NaNolmayanpikseller))*sum((NaNolmayanpikseller-ortalamavektor).^2))).^4;

% k = kurtosis(NaNolmayanpikseller)
end