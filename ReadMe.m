%This is an examplar file on how the PLOD algorithm could be used
%Type 'help PLOD_train' and 'help PLOD_test' under Matlab prompt for more detailed information
clear;clc;close all;fclose('all');
%more data sets are publicly available at: 
%https://palm.seu.edu.cn/zhangml/Resources.htm#partial_data
load('MSRCv2.mat');%load the partial label data set
X_load = zscore(data);%help zscore
y_load_p = transpose(full(partial_target));%partial target
y_load_r = transpose(full(target));%real target

numFolds = 10;%ten-fold cross validation
numInstances = size(X_load,1);
rand('state', 1);
idx_rand = randperm(numInstances);

ACC = zeros(numFolds,1);
for numFold=1:numFolds
    temp_str = ['Fold-', num2str(numFold), ' begins...'];
    disp(temp_str);
    %split dataset into training set and testing set
    [idx_train,idx_test] = CV_data_partition(numInstances,numFolds,numFold);
    X_train = X_load(idx_rand(idx_train),:);
    y_train = y_load_p(idx_rand(idx_train),:);
    X_test = X_load(idx_rand(idx_test),:);
    y_test = y_load_r(idx_rand(idx_test),:);
    %train & test
    PLOD_model = PLOD_train(X_train,y_train);
    ACC(numFold) = PLOD_test(PLOD_model,X_test,y_test);
end
temp_str = [ 'ACC = ', num2str(mean(ACC),'%4.3f'),'¡À', num2str(std(ACC),'%4.3f')];
disp(temp_str);