clear 
close all
figure(1);

threshold = 30:0.1:40;

for jj=1:100

load('peak_ADL.mat'); %% FallsH
load('peaks_real.mat');
load('peaks_sim.mat');


l=randperm(size(maxacc1,2));
maxacc1=maxacc1(l);

l=randperm(size(maxacc,2));
maxacc=maxacc(l);

maxacc1val=maxacc1(1:8000);
maxacc1real=maxacc1(8001:16000);

withsim= [maxacc1val';maxacc(1:length(maxacc2))'];
labels= [zeros(length(maxacc1val),1);  ones(length(maxacc2),1)];
TabSim=[withsim labels];

withreal= [maxacc1real';maxacc2'];
labelsreal= [zeros(length(maxacc1real),1);  ones(length(maxacc2),1)];
TabReal=[withreal labelsreal];

for i=1:length(threshold)
fp = sum( TabSim(:,1)>=threshold(i) & (TabSim(:,2) == 0));
tp = sum( TabSim(:,1)>=threshold(i) & (TabSim(:,2) == 1));
fn = sum( TabSim(:,1)<=threshold(i) & (TabSim(:,2)== 1));
tn = sum( TabSim(:,1)<=threshold(i) & (TabSim(:,2)== 0));

SE(jj,i)=tp/(tp+fn);
SP(jj,i)=tn/(tn+fp);
end

for i=1:length(threshold)
fp = sum( TabReal(:,1)>=threshold(i) & (TabReal(:,2) == 0));
tp = sum( TabReal(:,1)>=threshold(i) & (TabReal(:,2) == 1));
fn = sum( TabReal(:,1)<=threshold(i) & (TabReal(:,2)== 1));
tn = sum( TabReal(:,1)<=threshold(i) & (TabReal(:,2)== 0));

myse1(jj,i)=tp/(tp+fn);
mysp1(jj,i)=tn/(tn+fp);
end

% REAL falls AUC

b1=1-mysp1(jj,:);
maximal=b1(end)-b1(1);
aucrel(jj)=trapz(1-mysp1(jj,:),myse1(jj,:));
aucperc(jj)=aucrel(jj)/maximal;

% Simulated falls AUC


bv1=1-SP(jj,:);
maximalv=bv1(end)-bv1(1);
aucrel_v(jj)=trapz(1-SP(jj,:),SE(jj,:));
aucperc_v(jj)=aucrel_v(jj)/maximalv;


end

%test data

AUCmean=mean(aucperc)
SEM = std(aucperc)/sqrt(length(aucperc));               % Standard Error
ts = tinv([0.025  0.975],length(aucperc)-1);      % T-Score
CI = mean(aucperc) + ts*SEM;                      % Confidence Intervals
plusmin=AUCmean-CI(1)

%calculate CI Simulated
AUCmean_v=mean(aucperc_v)
SEMv = std(aucperc_v)/sqrt(length(aucperc_v));               % Standard Error
tsv = tinv([0.025  0.975],length(aucperc_v)-1);      % T-Score
CIv = mean(aucperc_v) + tsv*SEMv;                      % Confidence Intervals
plusmin_v=AUCmean_v-CIv(1)

for i=1:length(threshold)
    SD_SEVal1(:,i)=std(SE(:,i));
    SD_SPVal(:,i)=std(SP(:,i));
    Mean_SEVal(:,i)=mean(SE(:,i));
    Mean_SPVal(:,i)=mean(SP(:,i));
    SD_SETest1(:,i)=std(myse1(:,i));
    SD_SPTest(:,i)=std(mysp1(:,i));
    Mean_SETest(:,i)=mean(myse1(:,i));
    Mean_SPTest(:,i)=mean(mysp1(:,i));
end

Fallsx=fliplr(Mean_SETest);
ebx=fliplr(Mean_SPTest);
Falls1x=fliplr(Mean_SEVal);
eb1x=fliplr(Mean_SPVal);



%%
EB=(1./(1-ebx))/115;
EB1=(1./(1-eb1x))/115;

load('Worksapce1000.mat')

figure(6);
title('Interpolated ROC curve with confidence intervals', 'Fontsize', 24 );
ax1=subplot(1,2,1)
errorbar(eb, Falls, SD_SETest);
hold on
errorbar(eb1, Falls1, SD_SEVal);
title('Random Forest', 'Fontsize', 12);
xlabel('Specificity','Fontsize', 20);
ylabel('Sensitivity','Fontsize', 20);
legend({'Real falls','Simulated falls'},'Fontsize', 16);
ax2=subplot(1,2,2)
errorbar(ebx, Fallsx, SD_SETest1);
hold on
errorbar(eb1x, Falls1x, SD_SEVal1);
title('Threshold based technique', 'Fontsize', 12 );
xlabel('Specificity','Fontsize', 20);
ylabel('Sensitivity','Fontsize', 20);
legend({'Real falls','Simulated falls'},'Fontsize', 16);
linkaxes([ax1,ax2],'y')
linkaxes([ax1,ax2],'x')


