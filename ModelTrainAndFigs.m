clear 
close all
figure(1);

CostValues1=[zeros(100,1)';linspace(0.04,0.99,100)];
CostValues2=[linspace(0.99,0.1,100);zeros(100,1)'];

load('FallsL.mat'); %% FallsH
load('DeeptableNoFallsFinal.mat');
load('DeepTableNoSide5Sec.mat');

for nindex=1:2



l=randperm(size(DeepTableNoFalls,1),size(DeepTableNoFalls,1));
DeepTableNoFalls=DeepTableNoFalls(l,1:end);

l=randperm(size(DeepTableNoSide5Sec,1),size(DeepTableNoSide5Sec,1));
DeepTableNoSide5Sec=DeepTableNoSide5Sec(l,1:end);

%training the AD

% load('in_1.mat');
% index=[2 4:20 48];
index=[2 4 5 9 10 11 12 13 15 16 17 18 20 22 23 24 26 27 29 30 35 36 38 39 48 72];
%  index=[2 4:26 29 35 38 41 48 58 59 61 64];
%  index=[2 4:48 58 59 61 64];



% train treeBagger

TestTableNoFalls=DeepTableNoFalls(1:8000,:); %(1:size(DeepTableNoSide5Sec,1),:);

% create feature evaluation set

Ta1=table2array(TestTableNoFalls(:,2:end));
Sub1=table2array(TestTableNoFalls(:,1));

% randomly select as many falls as no falls we have
DeepTableNoSide5Sec2=DeepTableNoSide5Sec(1:end-80,:);
Test1=table2array(DeepTableNoSide5Sec2(:,5:end));
SubjectARRAY1=table2array(DeepTableNoSide5Sec2(:,2));
Sub_Arr=[Sub1;SubjectARRAY1];
F_0=zeros(size(Ta1,1),1);
F_1=ones(size(Test1,1),1);
F=[F_0;F_1];
TestData= [Ta1;Test1];
for i=1:size(TestData,2)
    media(:,i)=mean(TestData(:,i));
    sd(:,i)=std(TestData(:,i));
end


TableTest=array2table(TestData);
a1={'Variance','Angle','Kurtosis_a','Skewness_a','SD_a','Mean_a','Median_a','IQR_a','Kurtosis_g','Skewness_g','SD_g','Mean_g','Median_g','IQR_g','RMS','EnergyX','EnergyY','EnergyZ','EnergyG','Acc_steepness_afterpeak','Gyro_steepness_afterpeak','Acc_steepness_afterpeak_X','Acc_steepness_afterpeak_Y','Acc_steepness_afterpeak_Z','Kurtosis_x_a','Skewness_x_a','IQR_x_a','Kurtosis_y_a','Skewness_y_a','IQR_y_a','Kurtosis_z_a','Skewness_z_a','IQR_z_a','Kurtosis_x_g','Skewness_x_g','IQR_x_g','Kurtosis_y_g','Skewness_y_g','IQR_y_g','Kurtosis_z_g','Skewness_z_g','IQR_z_g','Max_f','Periodogram_maxf','Skewness_fft','Kurtosis_fft','S_Entropy','NF1','NF2','NF3','NF4','NF5','NF6','NF7','NF8','NF9','NF10','NF11','NF12','NF13','NF14','NF15','NF16','NF17','NF18','maxOrienx','varOrienx','maxOrieny','varOrieny','maxOrienz','varOrienz'};
TableTest.Properties.VariableNames = a1; 

a2={'Fall_Outcome','Subject'};

Table21=table(F);
a2={'Fall_Outcome'};
Table21.Properties.VariableNames = a2; 

EvalTable1=[Table21 TableTest];

% load('in_1.mat');
index1=[1 index];
EvalTable1=EvalTable1(:,index1);


RandomForrest = TreeBagger(200 ,EvalTable1,'Fall_Outcome','Cost',[0 0.80 ;0.20 0]);

%validation table
Variables=DeepTableNoFalls.Properties.VariableNames(index);
TestADLTable=DeepTableNoFalls(8001:11001,index);
TT=table2array(TestADLTable);
Sub1=table2array(DeepTableNoFalls(8001:11001,1));
% index=index+3 % only for laboratory falls!!
DeepTableNoSide5Sec2=DeepTableNoSide5Sec(end-79:end,index+3);
SubjectARRAY1=table2array(DeepTableNoSide5Sec2(:,1));
Sub_Arr=[Sub1;SubjectARRAY1];
F_0=zeros(size(TT,1),1);
F_1=ones(size(DeepTableNoSide5Sec2,1),1);
F=[F_0;F_1];
DeepTableNoSide5Sec2=table2array(DeepTableNoSide5Sec2);
TestData= [TT;DeepTableNoSide5Sec2];

X2=TestData;
yval2=F;

[prediction2, classifScore2]=RandomForrest.predict(X2);


for j=1:length(CostValues1)

[XA,YA,TA,AUC,OPTROCPT,suby] = perfcurve(yval2,classifScore2(:,2),1,'Cost',[CostValues1(:,j)';CostValues2(:,j)']);
SP(nindex,j)=1-OPTROCPT(1);
SE(nindex,j)=OPTROCPT(2);
AUCT(nindex,j)=AUC;
T1=TA((XA==OPTROCPT(1))&(YA==OPTROCPT(2)));
Tr1(nindex,j)=T1;
hold on
plot(XA,YA,'m')
hold on;
plot(OPTROCPT(1),OPTROCPT(2),'ro')

end

%test data

Variables=DeepTableNoFalls.Properties.VariableNames(index);
TestADLTable=DeepTableNoFalls(end-4830:end,index);
TT=table2array(TestADLTable);
Sub1=table2array(DeepTableNoFalls(end-4830:end,1));
Test1=table2array(RealFalls(:,index));
SubjectARRAY1=table2array(RealFalls(:,1));
Sub_Arr=[Sub1;SubjectARRAY1];
F_0=zeros(size(TT,1),1);
F_1=ones(size(Test1,1),1);
F=[F_0;F_1];
TestData= [TT;Test1];

TableTest=array2table(TestData);
TableTest.Properties.VariableNames = Variables;

Table2=table(F,Sub_Arr);
a2={'Fall_Outcome','Subject'};
Table2.Properties.VariableNames = a2; 

TestingTable=[Table2 TableTest];
X2=table2array(TestingTable(:,3:end));
yval2=table2array(TestingTable(:,1));

[prediction2, classifScore2]=RandomForrest.predict(X2);

for j=1:length(CostValues1)
fp = sum((classifScore2(:,2) >= Tr1(nindex,j)) & (yval2 == 0));
tp = sum((classifScore2(:,2) >= Tr1(nindex,j)) & (yval2 == 1));
fn = sum((classifScore2(:,2) <= Tr1(nindex,j)) & (yval2 == 1));
tn = sum((classifScore2(:,2) <= Tr1(nindex,j)) & (yval2 == 0));
% precision1=tp/(tp+fp);
% recall1=tp/(tp+fn);
% myF1Tree(nindex)=2*precision1*recall1/(precision1+recall1);
myse1(nindex,j)=tp/(tp+fn);
mysp1(nindex,j)=tn/(tn+fp);
end

% REAL falls AUC
b=[0 1-mysp1(nindex,:) 1];
a=[0 myse1(nindex,:) 1];
auctot(nindex)=trapz(b,a);

b1=1-mysp1(nindex,:);
maximal=b1(end)-b1(1);
aucrel(nindex)=trapz(1-mysp1(nindex,:),myse1(nindex,:));
aucperc(nindex)=aucrel(nindex)/maximal;

% Simulated falls AUC
bv=[0 1-SP(nindex,:) 1];
av=[0 SE(nindex,:) 1];
auctotv(nindex)=trapz(bv,av);

bv1=1-SP(nindex,:);
maximalv=bv1(end)-bv1(1);
aucrel_v(nindex)=trapz(1-SP(nindex,:),SE(nindex,:));
aucperc_v(nindex)=aucrel_v(nindex)/maximalv;

end

%calculate CI Real
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


%create figures
title('ROC curves for each classifier');
xlabel('True positive rate');
ylabel('True negative rate');
legend('Tree','optimalpointTree');


for i=1:length(CostValues1)
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

