function PLOD_model = PLOD_train( X_train,y_train )
%PLOD deals with paritial label learning problem with constrained k-means disambiguation [1]
%
%    Syntax
%
%       PLOD_model = PLOD_train( X_train,y_train )
%
%    Description
%
%       PLOD_train takes,
%           X_train	- An MxD array, the ith instance of training instance is stored in train_data(i,:)
%           y_train - A MxQ array, if the jth class label is one of the partial labels for the ith training instance, then y_train(i,j)=+1, otherwise y_train(i,j)=0
%      and returns,
%           PLOD_model is a structure containing the following elements
%           PLOD_model.model_ovo	- A cell containing the one-vs-one models
%           PLOD_model.model_stack	- A cell containing the stacking models
%           PLOD_model.ovo_mtx      - An array containing the one-vs-one coding matrix (In case that empty clusters for some class labels exist)
%           PLOD_model.C            - A vector containing the index of nonempty clusters (In case that empty clusters for some class labels exist)
%           PLOD_model.loop         - The number of iterations for disambiguation (In case that empty clusters for some class labels exist)
%
%  [1] J.-Y. Liu, J.-P. Sun, Y.-H. Zhao, B.-B. Jia, M.-L. Zhang. Partial Label Learning with Semi-Supervised Clustering Disambiguation. Pattern Recognition, 2025.
%
%NOTE: This function needs to invoke built-in functions such as knnsearch, designecoc, unique.

    %%%Parameters%%%
    num_training = size(X_train,1);%number of training examples
    num_features = size(X_train,2);%number of input features
    num_label = size(y_train,2);%number of class labels
    
    %%%Label Disambiguation%%%
    % cluster center initialization
    MU= zeros(num_label,num_features);
    for iLabel=1:num_label
        X_train_iLabel = X_train(y_train(:,iLabel)==1,:);
        if size(X_train_iLabel,1)>1
            MU(iLabel,:) = mean(X_train_iLabel);
        elseif size(X_train_iLabel,1)==1
            MU(iLabel,:) = X_train_iLabel;
        else%no samples belong to the class, just keep MU(iLabel,:) as zero vector
            temp_str = ['[initialization] no samples belong to the ',num2str(iLabel),'-th label.'];
            disp(temp_str);
        end
    end

    % semi-supervised clustering
    MAX_ITER = 1000;
    for loop=1:MAX_ITER
        MU_p = MU;%previous MU
        %cluster assignment
        LAMBDA = zeros(num_training,1);
        [idx_knn, ~] = knnsearch(MU,X_train,'k',num_label);
        idx_all = 1:num_label;
        for itrain=1:num_training
            idx_itrain = idx_all(y_train(itrain,:)==1);
            for iLabel=1:num_label
                tmp_idx_label = idx_knn(itrain,iLabel);
                if ismember(tmp_idx_label,idx_itrain)
                    LAMBDA(itrain) = tmp_idx_label;
                    break;
                end
            end
        end
        %cluster center updating
        for iLabel=1:num_label
            X_train_iLabel = X_train(LAMBDA==iLabel,:);
            if size(X_train_iLabel,1)>1
                MU(iLabel,:) = mean(X_train_iLabel);
            elseif size(X_train_iLabel,1)==1
                MU(iLabel,:) = X_train_iLabel;
            else%no samples belong to the cluster, just keep MU(iLabel,:) unchanged
                temp_str = ['[loop=',num2str(loop),'] no samples belong to the cluster (iLabel=',num2str(iLabel),')'];
                disp(temp_str);
            end
        end
        %difference between two adjacent K_centroids
        diff_norm = norm(MU-MU_p,'fro');
        if diff_norm<1e-3
            break;
        end
    end
    y_train_cluster = zeros(num_training, num_label);
    for itrain=1:num_training
        y_train_cluster(itrain, LAMBDA(itrain)) = 1;
    end
    PLOD_model.loop = loop;%for debugging purposes
    
    %%%predictive model induction%%%
    %OvO-based learning
    [C,IA,IC] = unique(LAMBDA);%In case that empty clusters for some class labels exist
    PLOD_model.C = C;%In case that empty clusters for some class labels exist
    %num_cluster does not necessarily equal to num_label, because some clusters may not contain any samples
    num_cluster = length(C);
    ovo_mtx = designecoc(num_cluster,'onevsone');
    num_ovo = size(ovo_mtx,2);
    PLOD_model.ovo_mtx = ovo_mtx;
    y_train_ovo_b = zeros(num_training,num_ovo);%binary prediction (+1/-1)
    y_train_ovo_r = zeros(num_training,num_ovo);%real-valued prediction
    for iovo=1:num_ovo
        class_p = C(ovo_mtx(:,iovo)==+1);%positive class
        idx_p = (LAMBDA==class_p);%index of positive samples
        class_n = C(ovo_mtx(:,iovo)==-1);%negative class
        idx_n = (LAMBDA==class_n);%index of negative samples

        y_ovo_initial = zeros(num_training,1);
        y_ovo_initial(idx_p) = +1;
        y_ovo_initial(idx_n) = -1;
        y_ovo = y_ovo_initial(idx_p|idx_n);
        X_train_ovo = X_train(idx_p|idx_n,:);
        model_ovo = BClassifier_train(X_train_ovo, y_ovo);
        [y_train_ovo_b(:,iovo),y_train_ovo_r(:,iovo)] = BClassifier_test(model_ovo, X_train);
        PLOD_model.model_ovo{iovo} = model_ovo;
    end
    y_train_pred_ovo_m = zeros(num_training,num_label);
    for itrain=1:num_training%Hamming decoding/majority voting
        y_train_ovo_bi = y_train_ovo_b(itrain,:);
        Hamming_dist1 = bsxfun(@times, ovo_mtx, y_train_ovo_bi); 
        Hamming_dist2 = (Hamming_dist1==-1);
        Hamming_dist3 = sum(Hamming_dist2,2);
        [~, min_idx] = min(Hamming_dist3);
        y_train_pred_ovo_m(itrain,C(min_idx)) = 1;
    end      
    y_train_combine = (y_train_pred_ovo_m+y_train_cluster).*y_train;
    y_train_stack = (y_train_combine>0)+0;
    
    % OvO predictions stacking
    stack_mtx = designecoc(num_cluster,'onevsall');
    num_stack = size(stack_mtx,2);
    for istack=1:num_stack
        class_p = C(stack_mtx(:,istack)==+1);
        y_stack = y_train_stack(:,class_p);
        y_stack(y_stack==0) = -1;%negative class (-1)
        select_ovo = (ovo_mtx(istack,:)~=0);%seclect relevant classifiers according to one-vs-one
        X_train_augment = y_train_ovo_r(:,select_ovo);
        X_train_stack = [X_train, X_train_augment];
        model_stack = BClassifier_train(X_train_stack, y_stack);
        PLOD_model.model_stack{istack} = model_stack;
    end
end
%The end!