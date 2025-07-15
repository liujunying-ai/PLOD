function [ yp_b, yp_r ] = BClassifier_test( model, X_test )
%BClassifier_test implements the testing process of binary classification
%Type 'help BClassifier_test' under Matlab prompt for more detailed information
%
%	Syntax
%
%       [ yp_b, yp_r ] = BClassifier_test( model, X_test )
%
%	Description
%
%   BClassifier_train takes,
%       model       - The predictive model returned by BClassifier_train
%       X_test      - An pxd array, the ith instance of testing instance is stored in X_test(i,:)
%   and returns,
%       yp_b	    - An px1 array, the binary-valued prediction for test instance matrix X_test
%       yp_r	    - An px1 array, the real-valued prediction for test instance matrix X_test
%See also BClassifier_train
%
% More details on libsvm: https://www.csie.ntu.edu.tw/~cjlin/libsvm/
% More details on liblinear: https://www.csie.ntu.edu.tw/~cjlin/liblinear/

    %libsvm
    [yp_b,~,yp_r] = svmpredict(ones(size(X_test,1),1),X_test,model,'-q');
%     %liblinear
%     [yp_b, ~, yp_r] = predict(ones(size(X_test,1),1),sparse(X_test),model,'-q');

end