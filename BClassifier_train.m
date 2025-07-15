function [ model ] = BClassifier_train( X_train, y_train )
%BClassifier_train implements the training process of binary classification
%Type 'help BClassifier_train' under Matlab prompt for more detailed information
%
%	Syntax
%
%       [ model ] = BClassifier_train( X_train, y_train )
%
%	Description
%
%   BClassifier_train takes,
%       X_train     - An mxd array, the ith instance of training instance is stored in X_train(i,:)
%       y_train     - An mxq array, the ith class vector of training instance is stored in y_train(i,:)
%   and returns,
%       model	    - The predictive model (dependent on specific binary classification algorithm)
%See also BClassifier_test
%
% More details on libsvm: https://www.csie.ntu.edu.tw/~cjlin/libsvm/
% More details on liblinear: https://www.csie.ntu.edu.tw/~cjlin/liblinear/

    %libsvm with default parameters (RBF kernel)
    model = svmtrain(y_train, X_train, '-q');
    
%     %libsvm with linear kernel
%     model = svmtrain(y_train, X_train, '-t 0 -q');

%     %liblinear with the setting "L2-regularized logistic regression (primal)" 
%     model = train(y_train,sparse(X_train),'-s 0 -B 1 -R -q');
%     %liblinear with the setting "L2-regularized L2-loss support vector classification (dual)" 
%     model = train(y_train,sparse(X_train),'-s 1 -B 1 -q');
%     %liblinear with the setting "L2-regularized L2-loss support vector classification (primal)" 
%     model = train(y_train,sparse(X_train),'-s 2 -B 1 -R -q');
%     %liblinear with the setting "L2-regularized L1-loss support vector classification (dual)" 
%     model = train(y_train,sparse(X_train),'-s 3 -B 1 -q');
%     %liblinear with the setting "L1-regularized L2-loss support vector classification" 
%     model = train(y_train,sparse(X_train),'-s 5 -B 1 -R -q');
%     %liblinear with the setting "L1-regularized logistic regression" 
%     model = train(y_train,sparse(X_train),'-s 6 -B 1 -R -q');
%     %liblinear with the setting "L2-regularized logistic regression (dual)" 
%     model = train(y_train,sparse(X_train),'-s 7 -B 1 -q');

    %liblinear options (version 2.42):
    %-s type : set type of solver (default 1)
	%	0 -- L2-regularized logistic regression (primal)
	%	1 -- L2-regularized L2-loss support vector classification (dual)
	%	2 -- L2-regularized L2-loss support vector classification (primal)
	%	3 -- L2-regularized L1-loss support vector classification (dual)
	%	4 -- support vector classification by Crammer and Singer
	%	5 -- L1-regularized L2-loss support vector classification
	%	6 -- L1-regularized logistic regression
	%	7 -- L2-regularized logistic regression (dual)
    %   11 -- L2-regularized L2-loss support vector regression (primal)
	%   12 -- L2-regularized L2-loss support vector regression (dual)
	%   13 -- L2-regularized L1-loss support vector regression (dual)
    %   21 -- one-class support vector machine (dual)
    %-B bias 
    %-R : not regularize the bias (for -s 0, 2, 5, 6, 11)
    %-q : quiet mode (no outputs)
end