function out = PYTHIA(Z, Y, Ybin, W, Ybest, algolabels, opts)
% -------------------------------------------------------------------------
% PYTHIA.m
% -------------------------------------------------------------------------
%
% By: Mario Andres Munoz Acosta
%     School of Mathematics and Statistics
%     The University of Melbourne
%     Australia
%     2020
%
% -------------------------------------------------------------------------

disp('  -> Initializing PYTHIA. She may take a while to complete...');
[ninst,nalgos] = size(Ybin);
disp('-------------------------------------------------------------------------');
if opts.useweights
    disp('  -> PYTHIA will use different weights for each observation.');
    Waux = W;
else
    disp('  -> PYTHIA will use equal weights for each observation.');
    Waux = ones(ninst,nalgos);
end
disp('-------------------------------------------------------------------------');
[Znorm,out.mu,out.sigma] = zscore(Z);
if size(Znorm,1)>1e3
    KernelFcn = 'polynomial';
else
    KernelFcn = 'gaussian';
end
t = tic;

% Initialise temporary variables that MATLAB can handle in a parfor loop.
out_cp = cell(1,nalgos);
out_svm = cell(1,nalgos);
out_cvcmat = zeros(nalgos,4);
out_Ysub = false & Ybin;
out_Yhat = false & Ybin;
out_Pr0sub = 0.*Ybin;
out_Pr0hat = 0.*Ybin;
out_boxcosnt = zeros(1,nalgos);
out_kscale = out_boxcosnt;
cvfolds = opts.cvfolds;

% Need to use parfeval, otherwise parallel task distribution is wildly suboptimal
parfor i=1:nalgos
    % Need to check details or random number generation and tic-toc in parallel loops
    state = rng;
    rng('default');
    out_cp{i} = cvpartition(Ybin(:,i),'Kfold',cvfolds,'Stratify',true);
    rng('default');
    out_svm{i} = fitcsvm(Znorm,Ybin(:,i),'Standardize',false,...
                                         'Weights',Waux(:,i),...
                                         'CacheSize','maximal',...
                                         'RemoveDuplicates',true,...
                                         'KernelFunction',KernelFcn,...
                                         'OptimizeHyperparameters','auto',...
                                         'HyperparameterOptimizationOptions',...
                                         struct('CVPartition',out_cp{i},...
                                                'Verbose',0,...
                                                'AcquisitionFunctionName','probability-of-improvement',...
                                                'ShowPlots',false));
    out_svm{i} = fitSVMPosterior(out_svm{i});
    rng(state);
    [out_Ysub(:,i),aux] = out_svm{i}.resubPredict;
    out_Pr0sub(:,i) = aux(:,1);
    [out_Yhat(:,i),aux] = out_svm{i}.predict(Znorm);
    out_Pr0hat(:,i) = aux(:,1);
    aux = confusionmat(Ybin(:,i),out_Ysub(:,i));
    out_cvcmat(i,:) = aux(:);
    out_boxcosnt(i) = out_svm{i}.HyperparameterOptimizationResults.bestPoint{1,1};
    out_kscale(i) = out_svm{i}.HyperparameterOptimizationResults.bestPoint{1,2};

    disp(['    -> PYTHIA has trained a model for ''' algolabels{i}, ...
            ''', there may be some left to train but I refuse to speculate.']);
end

% Shift temporary variables back into the output structure.
out.cp = out_cp;
out.svm = out_svm;
out.cvcmat = out_cvcmat;
out.Ysub = out_Ysub;
out.Yhat = out_Yhat;
out.Pr0sub = out_Pr0sub;
out.Pr0hat = out_Pr0hat;
out.boxcosnt = out_boxcosnt;
out.kscale = out_kscale;

tn = out.cvcmat(:,1);
fp = out.cvcmat(:,3);
fn = out.cvcmat(:,2);
tp = out.cvcmat(:,4);
out.precision = tp./(tp+fp);
out.recall = tp./(tp+fn);
out.accuracy = (tp+tn)./ninst;
disp('-------------------------------------------------------------------------');
disp('  -> PYTHIA has completed training the models.');
disp(['  -> The average cross validated precision is: ' ...
      num2str(round(100.*mean(out.precision),1)) '%']);
disp(['  -> The average cross validated accuracy is: ' ...
      num2str(round(100.*mean(out.accuracy),1)) '%']);
  disp(['      -> Elapsed time: ' num2str(toc(t),'%.2f\n') 's']);
disp('-------------------------------------------------------------------------');
% We assume that the most precise SVM (as per CV-Precision) is the most
% reliable.
[best,out.selection0] = max(bsxfun(@times,out.Yhat,out.precision'),[],2);
[~,default] = max(mean(Ybin));
out.selection1 = out.selection0;
out.selection0(best<=0) = 0;
out.selection1(best<=0) = default;

sel0 = bsxfun(@eq,out.selection0,1:nalgos);
sel1 = bsxfun(@eq,out.selection1,1:nalgos);
avgperf = nanmean(Y);
stdperf = nanstd(Y);
Yfull = Y;
Ysvms = Y;
Y(~sel0) = NaN;
Yfull(~sel1) = NaN;
Ysvms(~out.Yhat) = NaN;

pgood = mean(any( Ybin & sel1,2));
fb = sum(any( Ybin & ~sel0,2));
fg = sum(any(~Ybin &  sel0,2));
tg = sum(any( Ybin &  sel0,2));
precisionsel = tg./(tg+fg);
recallsel = tg./(tg+fb);

disp('  -> PYTHIA is preparing the summary table.');
out.summary = cell(nalgos+3, 11);
out.summary{1,1} = 'Algorithms ';
out.summary(2:end-2, 1) = algolabels;
out.summary(end-1:end, 1) = {'Oracle','Selector'};
out.summary(1, 2:11) = {'Avg_Perf_all_instances';
                        'Std_Perf_all_instances';
                        'Probability_of_good';
                        'Avg_Perf_selected_instances';
                        'Std_Perf_selected_instances';
                        'CV_model_accuracy';
                        'CV_model_precision';
                        'CV_model_recall';
                        'BoxConstraint';
                        'KernelScale'};
out.summary(2:end, 2) = num2cell(round([avgperf nanmean(Ybest) nanmean(Yfull(:))],3));
out.summary(2:end, 3) = num2cell(round([stdperf nanstd(Ybest) nanstd(Yfull(:))],3));
out.summary(2:end, 4) = num2cell(round([mean(Ybin) 1 pgood],3));
out.summary(2:end, 5) = num2cell(round([nanmean(Ysvms) NaN nanmean(Y(:))],3));
out.summary(2:end, 6) = num2cell(round([nanstd(Ysvms) NaN nanstd(Y(:))],3));
out.summary(2:end, 7) = num2cell(round(100.*[out.accuracy' NaN NaN],1));
out.summary(2:end, 8) = num2cell(round(100.*[out.precision' NaN precisionsel],1));
out.summary(2:end, 9) = num2cell(round(100.*[out.recall' NaN recallsel],1));
out.summary(2:end-2, 10) = num2cell(round(out.boxcosnt,3));
out.summary(2:end-2, 11) = num2cell(round(out.kscale,3));
out.summary(cellfun(@(x) all(isnan(x)),out.summary)) = {[]}; % Clean up. Not really needed
disp('  -> PYTHIA has completed! Performance of the models:');
disp(' ');
disp(out.summary);

end