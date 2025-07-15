function [ Accuracy, y_predict ] = PLOD_test( PLOD_model,X_test,y_test )
%PLOD deals with paritial label learning problem with constrained k-means disambiguation [1]
%
%    Syntax
%
%       [ Accuracy, y_predict ] = PLOD_test( PLOD_model,X_test,y_test )
%
%    Description
%
%       PLOD_train takes,
%           PLOD_model  - A structure returned by PLOD_train
%           X_test      - An PxD array, the ith instance of testing instance is stored in X_test(i,:)
%           y_test      - A PxQ array, if the ith test instance belongs to the jth class, then y_test(i,j)=+1, otherwise y_test(i,j)=0
%      and returns,
%           Accuracy    - A scalar, the accuracy
%           y_predict   - A PxQ array, if the ith test instance belongs to the jth class, then y_test(i,j)=+1, otherwise y_test(i,j)=0
%
%  [1] J.-Y. Liu, J.-P. Sun, Y.-H. Zhao, B.-B. Jia, M.-L. Zhang. Partial Label Learning with Semi-Supervised Clustering Disambiguation. Pattern Recognition, 2025.
%

    %%%Parameters%%%
    num_testing = size(X_test,1);%number of testing examples
    num_label = size(y_test,2);%number of class labels
    ovo_mtx = PLOD_model.ovo_mtx;%In case that empty clusters for some class labels exist
    num_ovo = length(PLOD_model.model_ovo);%In case that empty clusters for some class labels exist
    num_stack = length(PLOD_model.model_stack);%In case that empty clusters for some class labels exist
    C = PLOD_model.C;%In case that empty clusters for some class labels exist
    
    %%%main%%%
    y_predict_ovo_b = zeros(num_testing,num_ovo);
    y_predict_ovo_r = zeros(num_testing,num_ovo);
    for iovo=1:num_ovo
        model_ovo = PLOD_model.model_ovo{iovo};
        [y_predict_ovo_b(:,iovo),y_predict_ovo_r(:,iovo)] = BClassifier_test(model_ovo, X_test);
    end
%     y_test_pred_ovo_m = zeros(num_testing,num_label);
%     for itest=1:num_testing%
%         y_test_ovo_bi = y_predict_ovo_b(itest,:);
%         Hamming_dist1 = bsxfun(@times, ovo_mtx, y_test_ovo_bi); 
%         Hamming_dist2 = (Hamming_dist1==-1);
%         Hamming_dist3 = sum(Hamming_dist2,2);
%         [min_val, min_idx] = min(Hamming_dist3);
%         y_test_pred_ovo_m(itest,C(min_idx)) = 1;%In case that empty clusters for some class labels exist
%     end
    y_predict_stack_b = zeros(num_testing,num_stack);
    y_predict_stack_r = zeros(num_testing,num_stack);
    for istack=1:num_stack
        model_stack = PLOD_model.model_stack{istack};
        select_ovo = (ovo_mtx(istack,:)~=0);%seclect relevant classifiers according to one-vs-one
        X_test_augment = y_predict_ovo_r(:,select_ovo);
        X_test_stack = [X_test, X_test_augment];
        [y_predict_stack_b(:,istack),y_predict_stack_r(:,istack)] = BClassifier_test(model_stack, X_test_stack);
    end
    
    [~, max_idx] = max(y_predict_stack_r,[],2);
    y_predict = zeros(size(y_test));
    for itest=1:num_testing
        y_predict(itest,C(max_idx(itest))) = 1;%In case that empty clusters for some class labels exist
    end     
    Accuracy = sum(sum(y_predict==y_test,2)==num_label)/num_testing;
end
%The end!