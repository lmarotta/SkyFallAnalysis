clear 
close all
figure(1);

threshold = 30:0.1:40;

for jj=1:1000

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
    SD_SEVal(:,i)=std(SE(:,i));
    SD_SPVal(:,i)=std(SP(:,i));
    Mean_SEVal(:,i)=mean(SE(:,i));
    Mean_SPVal(:,i)=mean(SP(:,i));
    SD_SETest(:,i)=std(myse1(:,i));
    SD_SPTest(:,i)=std(mysp1(:,i));
    Mean_SETest(:,i)=mean(myse1(:,i));
    Mean_SPTest(:,i)=mean(mysp1(:,i));
end

Falls=fliplr(Mean_SETest);
eb=fliplr(Mean_SPTest);
Falls1=fliplr(Mean_SEVal);
eb1=fliplr(Mean_SPVal);

% Create a y-axis semilog plot using the semilogy function
% Plot SER data in blue and BER data in red


figure(2);
semilogx(eb, Falls, 'o-','MarkerSize',8, 'MarkerEdgeColor','red', 'MarkerFaceColor',[1 .6 .6]);
hold on
semilogx(eb1, Falls1, 'o-','MarkerSize',8, 'MarkerEdgeColor','blue', 'MarkerFaceColor',[1 .6 .6]);

% Turn on the grid
grid on

% Add title and axis labels
title('Fall detection rate as a function of specificity')
ylabel('Sensitivity')
xlabel('Specificity')



figure(4);
EB=(1./(1-eb))/115;
EB1=(1./(1-eb1))/115;

plot(EB, Falls, 's-','MarkerSize',8, 'MarkerEdgeColor','red', 'MarkerFaceColor',[1 .6 .6])
hold on
plot(EB1, Falls1, 's-','MarkerSize',8, 'MarkerEdgeColor','red', 'MarkerFaceColor',[1 .6 .6])

% Turn on the grid
grid on
title('Fall detection rate as a function of false alarms per day')
ylabel('Sensitivity')
xlabel('Number of days without a false alarm')


figure(5);
errorbar(eb, Falls, SD_SETest, 'o-','MarkerSize',8, 'MarkerEdgeColor','blue', 'MarkerFaceColor',[  0.46  0.99   0.66]);
hold on
errorbar(eb1, Falls1, SD_SEVal,'o-','MarkerSize',8, 'MarkerEdgeColor','red', 'MarkerFaceColor',[1 .6 .6]);
title('Interpolated ROC curve with confidence intervals', 'Fontsize', 24 );
xlabel('Specificity','Fontsize', 20);
ylabel('Sensitivity','Fontsize', 20);
legend({'Real falls','Simulated falls'},'Fontsize', 16);

figure(6);
errorbar(eb, Falls, SD_SETest);
hold on
errorbar(eb1, Falls1, SD_SEVal);
title('Interpolated ROC curve with confidence intervals', 'Fontsize', 24 );
xlabel('Specificity','Fontsize', 20);
ylabel('Sensitivity','Fontsize', 20);
legend({'Real falls','Simulated falls'},'Fontsize', 16);

% close all
figure(7);
x2=[eb, fliplr(eb)];
inBetween=[Falls+(SD_SETest),fliplr(Falls-(SD_SETest))];
x22=[eb1, fliplr(eb1)];
inBetween2=[Falls1+(SD_SEVal),fliplr(Falls1-(SD_SEVal))];
patch(x22, inBetween2,[1 0.7 1]);
patch(x2, inBetween,[0.7 1 0.8]);
hold on
plot(eb, Falls, 'o-','MarkerSize',8, 'MarkerEdgeColor','blue', 'MarkerFaceColor','blue');
hold on
plot(eb1, Falls1, 'o-','MarkerSize',8, 'MarkerEdgeColor','red', 'MarkerFaceColor','red');
hold on
plot(eb1,Falls1+(SD_SEVal),'Color','black');
hold on
plot(x22,inBetween2,'Color','black');
title('Interpolated ROC curve with confidence intervals', 'Fontsize', 24 );
xlabel('Specificity','Fontsize', 20);
ylabel('Sensitivity','Fontsize', 20);
legend({'Simulated falls','Real falls'},'Fontsize', 16);



