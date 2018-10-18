function [selectedFeatures, history, bootstrap] = selectFeatures(X, classes, discrimType, maxFeatures, verbose, maxDepth, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUTS:
% X,                an nObservations x nFeatures matrix of features from
%                   which to select
% 
% classes,          a numeric vector of class labels
% 
% discrimType,      'linear' or 'quadratic'
%                   Default: 'linear'
% 
% maxFeatures,      numeric specifying maximum number of features to select
%                   Default:  10
% 
% verbose,          boolean (true/false or 0/1) specifying whether or not
%                   to provide progress updates
%                   Default: true
% 
% maxDepth,         numeric <= 3; If multiple next features provide 
%                   equivalent improvement, the algorithm will select 
%                   between them by comparing improvements from the 
%                   addition of features at the next "depth". The higher 
%                   this number, the longer the algorithm can take. 
%                  	In practice, features can usually be selected 
%                   unambiguously at a depth of 3 or less. A depth of 1 may 
%                   provide suboptimal selection and should be considered a 
%                   fast or "greedy" method, offering more comparable 
%                   results to Matlab's sequentialfs.
%                   Default: 3
% 
% Optional Name-Value Pair Arguments:
% 
% 'bootstrap',      positive integer specifying the # of bootstrap samples
%                   for checking the significance of the final selected 
%                   features; if not specified, no bootstrapping will be
%                   performed (e.g., 'bootstrap',1000)
% 
% Optional cross-validation type (one of the following name-value pairs):
%                   -includes identical options to Matlab's fitdiscr.m 
%                   (w/ one addition):
% 
% Default: leave-one-out (i.e., include no input)
% 
% 'sets',           [] double vector of length == # observations specifying
%                   which hold-out set a given data-point belongs to; e.g.,
%                   [1,1,1,2,2,2,3,3] would denote that the algorithm
%                   should hold out (1) data-points while training on (2)
%                   and (3), then hold out (2) data-points while training
%                   on (1) and (3), etc.; this allows some data-points to
%                   be consistently kept together if desired
% 
% 'holdout',        scalar value in the range (0,1) denoting the fraction
%                   of data for hold-out validation
% 
% 'kfold',          positive integer value > 1 denoting the number of folds
%                   to use in a cross-validated classifier 
%                   e.g., 'KFold',10 (see fitdiscr.m documentation for
%                   further details)
% 
%
% OUTPUTS:
% 
% 'selectedFeatures',   a vector containing the indices of the selected
%                       features
% 
% 'history',            a structure containing fields, 'featureNum' &
%                       'accuracy'. Each is a cell array,
%                       wherein each cell denotes feature selection # / 
%                       depth of the tree. Numbers within a given cell of
%                       'featureNum' indicate the index of a feature under
%                       consideration at that depth; 'accuracy' entries
%                       within a cell indicate the cross-validated accuracy
%                       of the corresponding candidate feature in
%                       'featureNum'
% 
% 'bootstrap',          a structure containing the following fields as 
%                       outputs from bootstrapping:
%       'medianCoeffs', a vector containing the medians of coefficients
%                       across bootstrap iterations, one for each of
%                       'selectedFeatures'
%       'meanCoeffs',   a vector containing the means of coefficients
%                       across bootstrap iterations, one for each of
%                       'selectedFeatures' (these may be biased if the
%                       bootstrap distribution is skewed)
%       'ci',           a 2 x nFeatures matrix wherein the first row
%                       denotes lower bounds (2.5 percentile) of a 95% 
%                       confidence interval for each feature from 
%                       'selectedFeatures'; the second row denotes the 
%                       upper bounds (97.5 percentile)
%       'pvals',        a vector containing p-values corresponding to the
%                       2-tailed test that each feature's bootstrap
%                       distribution is significantly different from zero
%       'distribution', a nIterations x nFeatures matrix of bootstrapped
%                       coefficient values for each feature from
%                       'selectedFeatures'; note that the rows of this
%                       matrix have been sorted from least to greatest
% 
% Author: Elliot Layden, 2018, The University of Chicago
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Check Inputs:
    if (nargin < 2) || isempty(X) || isempty(classes)
        error('ERROR:  You must specify a features matrix ''X'' and a class labels vector ''classes''.')
    end
    if nargin < 3 || isempty(discrimType)
        discrimType = 'linear';
    end
    if nargin < 4 || isempty(maxFeatures)
        maxFeatures = 10;
    end
    if nargin < 5 || isempty(verbose)
        verbose = true;
    end
    if nargin < 6 || isempty(maxDepth)
        maxDepth = 3;
    end
    
    if (size(X,1)==size(classes,2))
        classes = classes';
    end
    
    % Search for name-value pairs for cross-validation:
    inputs = varargin;
    parsed_inputs = struct('sets',[],'holdout',[],'kfold',[],'bootstrap',[]); % 'parallel',[]
    poss_input = {'sets','holdout','kfold','bootstrap'};
  
    input_ind = zeros(1,length(poss_input));
    for ix = 1:length(poss_input)
        jx = find(strcmpi(poss_input{ix},inputs));
        if ~isempty(jx)
            input_ind(ix) = jx;
            input1 = inputs{input_ind(ix)+1};
            parsed_inputs.(poss_input{ix}) = input1;
        end
    end
    
    if ~isempty(parsed_inputs.holdout) || ~isempty(parsed_inputs.kfold)
        warning('off','all');
    end
    
    % Outputs:
    selectedFeatures = [];
    bootstrap = [];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    lastAcc = -1;
    currAcc = 0;
    featureNum = {};
    parentNode = {};
    accuracy = {};
    nFeature = 0; % actual tree depth
    depthImproved = true;
                
    % Begin Optimizing:
    while (currAcc > lastAcc) && (currAcc < 1) && (nFeature < maxFeatures) && depthImproved

        lastAcc = currAcc;
        trackDepth = 0; % make sure depth doesn't go below user spec for a given feature
        multipleOptions = true;
        depthImproved = true;
        
        % Build tree:
        while (trackDepth < maxDepth) && multipleOptions && depthImproved

            % Remember whether this depth offered any improvement
            if verbose; disp(['Accuracy: ',num2str(round(currAcc*100,2)),'%']); end
            depthImproved = false;
            accDepth = currAcc; % accuracy at this depth, use this to prune 
            nFeature = nFeature + 1;
            trackDepth = trackDepth + 1;
            if verbose; disp(['Feature ',num2str(nFeature),' - Depth ',num2str(trackDepth)]); end
            
            if isempty(featureNum)

                % Get root nodes of tree:
                [bestFeatures, acc] = selectBest(X, classes, selectedFeatures, discrimType, parsed_inputs); 
                if acc > currAcc
                    depthImproved = true;
                    currAcc = acc;
                    featureNum{nFeature} = bestFeatures; %#ok
                    parentNode{nFeature} = zeros(1,length(featureNum{nFeature})); %#ok % (Root node(s))
                    accuracy{nFeature} = repmat(acc,[1,length(bestFeatures)]); %#ok
                end
    
            else
                
                for ix = 1:length(featureNum{nFeature-1})
                    useFeatures = getBranch(featureNum, parentNode, 1, nFeature-1, ix); % get the branch that leads to current test
                    [bestFeatures, acc] = selectBest(X, classes, useFeatures, discrimType, parsed_inputs);
                    
                    if (acc > accDepth) && (acc > currAcc) % remove old options
                        featureNum{nFeature} = []; %#ok
                        parentNode{nFeature} = []; %#ok
                        accuracy{nFeature} = []; %#ok
                        accDepth = acc; 
                        currAcc = acc;
                        depthImproved = true;
                        featureNum{nFeature} = [featureNum{nFeature}, bestFeatures]; %#ok
                        parentNode{nFeature} = [parentNode{nFeature}, repmat(ix,[1,length(bestFeatures)])]; %#ok
                        accuracy{nFeature} = [accuracy{nFeature}, repmat(acc,[1,length(bestFeatures)])]; %#ok
                    elseif (acc == accDepth) && (acc > currAcc) % add this, but don't remove old options
                        if length(featureNum) < nFeature
                            featureNum{nFeature} = []; %#ok
                            parentNode{nFeature} = []; %#ok
                            accuracy{nFeature} = []; %#ok
                        end
                        depthImproved = true;
                        featureNum{nFeature} = [featureNum{nFeature}, bestFeatures]; %#ok
                        parentNode{nFeature} = [parentNode{nFeature}, repmat(ix,[1,length(bestFeatures)])]; %#ok
                        accuracy{nFeature} = [accuracy{nFeature}, repmat(acc,[1,length(bestFeatures)])]; %#ok
                    end

                end
            
            end
                        
            if (length(accuracy) < nFeature) || ~depthImproved % Nothing added/to add, stop
                nFeature = nFeature - 1;
                if verbose
                    disp('Stopping: additional features offer no improvement.');
                end
            elseif (trackDepth == maxDepth)
                if length(featureNum{nFeature})>1
                    featureNum{nFeature}(2:end) = []; %#ok
                    parentNode{nFeature}(2:end) = []; %#ok
                    accuracy{nFeature}(2:end) = []; %#ok
                    if verbose
                        warning(['Greedily selected feature ',...
                            num2str(featureNum{nFeature}(1)),...
                            ' at max depth ',num2str(trackDepth),'.']); 
                    end
                end
            elseif length(accuracy{nFeature})==1
                multipleOptions = false;
            end

        end 
            
    end
    
    % Get the final branch:
    selectedFeatures = fliplr(getBranch(featureNum, parentNode, 1, nFeature, 1));
    if length(selectedFeatures) > maxFeatures
        selectedFeatures = selectedFeatures(1:maxFeatures);
    end
    
    % Perform bootstrapping of coefficient estimates, if requested:
    if ~isempty(parsed_inputs.bootstrap)
        if length(unique(classes))==2
            if verbose; disp('Bootstrapping...'); end
            bootstrap = struct('medianCoeffs',[],'meanCoeffs',[],'ci',[],'pvals',[],'distribution',[]);
            hWait = waitbar(0,'Bootstrapping coefficients...');
            if (parsed_inputs.bootstrap>0)
                nIter = parsed_inputs.bootstrap;
            else nIter = 1000;
                warning('Invalid inputs for ''bootstrap''; must be a positive integer. Using 1,000 iterations by default.')
            end
            nFeatures = length(selectedFeatures);
            Xuse = X(:,selectedFeatures);
            bootDist = zeros(nIter,nFeatures);
            count = 0;
            while (count < nIter)
                waitbar(count/nIter,hWait);
                if ~isempty(parsed_inputs.sets)
                    nSub = length(unique(parsed_inputs.sets));
                    % Get Bootstrap Indices:;
                    boot_subj = randi(nSub,[nSub,1]);
                    Xdata = []; Ydata = [];
                    for ix = 1:nSub
                        Xdata = [Xdata; Xuse(parsed_inputs.sets==boot_subj(ix),:)]; %#ok
                        Ydata = [Ydata; classes(parsed_inputs.sets==boot_subj(ix))]; %#ok
                    end
                else
                    nRow = size(X,1);
                    % Get Bootstrap Indices:;
                    bootInd = randi(nRow,[nRow,1]);
                    Xdata = Xuse(bootInd,:); Ydata = classes(bootInd);
                end
                % Assure that at least one subject from each condition:
                if isempty(setdiff(classes, Ydata))
                    count = count + 1;
                    % Fit model (Z-score X for more comparable coefficients):
                    mdl1 = fitcdiscr(zscore(Xdata),Ydata,'DiscrimType',discrimType);
                    if strcmpi(discrimType,'linear')
                        bootDist(count,:) = mdl1.Coeffs(2,1).Linear; 
                    elseif strcmpi(discrimType,'quadratic')
                        bootDist(count,:) = mdl1.Coeffs(2,1).Quadratic; 
                    end
                end
            end; close(hWait);

            % Sort in ascending order:
            bootstrap.distribution = sort(bootDist,1);

            % Median and CI effect sizes:
            bootstrap.medianCoeffs = median(bootstrap.distribution);  % median should be used to avoid major skewed outliers
            bootstrap.meanCoeffs = mean(bootstrap.distribution);  
            bootstrap.ci = [bootstrap.distribution(round(.025*nIter),:); bootstrap.distribution(round(.975*nIter),:);];

            % Calculate P-vals:
            bootstrap.pvals = zeros(1,nFeatures);
            for ix = 1:nFeatures % feature
                if bootstrap.medianCoeffs(ix) > 0 % positive beta
                    bootstrap.pvals(ix) = (sum(bootstrap.distribution(:,ix)<0)+1)/(nIter+1)*2;
                else % negative beta
                    bootstrap.pvals(ix) = (sum(bootstrap.distribution(:,ix)>0)+1)/(nIter+1)*2;
                end
            end
            if verbose; disp('Done.'); end
        else
           warning('Bootstrapped coefficients are currently only available for two category classificiation problems.') 
        end
    end
    
    % Save history of tree:
    if nargout > 1
        history = struct('featureNum',[],'accuracy',[]);
        history.featureNum = featureNum;
        history.accuracy = accuracy;
    end
    
    % Turn warnings back on:
    if ~isempty(parsed_inputs.holdout) || ~isempty(parsed_inputs.kfold)
        warning('on','all');
    end
    
    % Function for retreiving the hierarchy of nodes on a branch:
    function [path] = getBranch(featureNum, parentNode, startDepth, endDepth, ind)
        path = [];
        
        % Error checking
        if isempty(startDepth) || startDepth<1
           startDepth = 1; 
        end
        if length(featureNum)<endDepth
            error('getBranch: endDepth specified is greater than depth of tree');
        end
        if length(featureNum)~=length(parentNode)
            error('getBranch: featureNum and parentNode must be of equal length');
        end
        if length(featureNum{endDepth})<ind
           error(['getBranch: there are not ',num2str(ind),' child nodes at this depth']); 
        end
        
        % Trace branch
        nextInd = ind; % starting index
        for ixx = endDepth:-1:startDepth
            path = [path, featureNum{ixx}(nextInd)]; %#ok
            nextInd = parentNode{ixx}(nextInd);
        end
        
    end

    % Sub-function to select best features:
    function [selectedFeatures, accuracyRate] = selectBest(featureMat, condition, initialFeatures, discrimType, inputs)
        
        % Initialize
        mAccuracy = zeros(1,size(featureMat,2)); available = true(1,size(featureMat,2));
        if (nargin < 4); initialFeatures = []; end
        
        % Remove any initial features from further consideration:
        available(initialFeatures) = false;
        if (nargin < 5) || isempty(discrimType)
           discrimType = 'linear'; 
        end

        for i = 1:size(featureMat,2)
            if (available(i))
                if ~isempty(inputs.sets)
                    pred = zeros(size(featureMat,1),1);
                    for j = 1:max(inputs.sets) % n = 15
                        % Form Folds:
                        leaveOut = find(inputs.sets==j);
                        keepIn = setdiff(1:size(featureMat,1),leaveOut);
                        % Model and Prediction:
                        try
                            mdl = fitcdiscr(featureMat(keepIn,[initialFeatures, i]),...
                                condition(keepIn),'DiscrimType',discrimType);
                            pred(leaveOut) = predict(mdl,featureMat(leaveOut,[initialFeatures, i]));
                        catch
                            pred(leaveOut) = 0; % error with covariance, just make these ~= to condition
                        end
                    end
                    mAccuracy(i) = sum(pred==condition)/length(condition);
                elseif ~isempty(inputs.holdout) % HOLDOUT
                    mdl = fitcdiscr(featureMat(:,[initialFeatures, i]),...
                        condition,'DiscrimType',discrimType, 'Holdout',...
                        inputs.holdout);
                    mAccuracy(i) = 1 - kfoldLoss(mdl);
                elseif ~isempty(inputs.kfold) % KFOLD
                    mdl = fitcdiscr(featureMat(:,[initialFeatures, i]),...
                        condition,'DiscrimType',discrimType, 'KFold',...
                        inputs.kfold);
                    mAccuracy(i) = 1 - kfoldLoss(mdl);
                else % Leave-one-out default
                    mdl = fitcdiscr(featureMat(:,[initialFeatures, i]),...
                        condition,'DiscrimType',discrimType,'Leaveout','on');
                    mAccuracy(i) = 1 - kfoldLoss(mdl);
                end
            else
                mAccuracy(i) = 0;
            end
        end
        accuracyRate = max(mAccuracy);
        selectedFeatures = find(mAccuracy==accuracyRate);
        
    end % end selectBest()

end