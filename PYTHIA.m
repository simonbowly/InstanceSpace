function out = PYTHIA(Z, Y, Ybin, W, Ybest, algolabels, opts, run_parallel)
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
out.cp = cell(1,nalgos);
out.svm = cell(1,nalgos);
out.post = cell(1,nalgos);
out.cvcmat = zeros(nalgos,4);
out.Ysub = false & Ybin;
out.Yhat = false & Ybin;
out.Pr0sub = 0.*Ybin;
out.Pr0hat = 0.*Ybin;
out.boxcosnt = zeros(1,nalgos);
out.kscale = out.boxcosnt;
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

if run_parallel
    % Need to use parfeval here; parfor does not do the kind of load balancing
    % required for a small number of variable-length tasks.
    futures(1:nalgos) = parallel.FevalFuture;
    for i = 1:nalgos
        futures(i) = parfeval(@PYTHIASingleModel, 9, ...
            Ybin(:,i), Waux(:,i), Znorm, opts.cvfolds, KernelFcn);
    end

    for idx = 1:nalgos
        [i, cp, svm, Ysub, Pr0sub, Yhat, Pr0hat, cvcmat, boxcosnt, kscale] = fetchNext(futures);
        out.cp{i} = cp;
        out.svm{i} = svm;
        out.Ysub(:,i) = Ysub;
        out.Pr0sub(:,i) = Pr0sub;
        out.Yhat(:,i) = Yhat;
        out.Pr0hat = Pr0hat;
        out.cvcmat(i,:) = cvcmat;
        out.boxcosnt(i) = boxcosnt;
        out.kscale(i) = kscale;
        disp(['    -> PYTHIA has trained a model for ''' algolabels{i} '''']);
    end
else
    % Pass results direct into the output structure when running in serial.
    for i=1:nalgos
        [out.cp{i}, out.svm{i}, ...
            out.Ysub(:,i), out.Pr0sub(:,i), out.Yhat(:,i), out.Pr0hat(:,i), ...
            out.cvcmat(i,:), out.boxcosnt(i), out.kscale(i)] = ...
            PYTHIASingleModel(Ybin(:,i), Waux(:,i), Znorm, opts.cvfolds, KernelFcn);
        disp(['    -> PYTHIA has trained a model for ''' algolabels{i} '''']);
    end
end

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


function [out_cp, out_svm, out_Ysub, out_Pr0sub, ...
        out_Yhat, out_Pr0hat, out_cvcmat, out_boxcosnt, out_kscale] = ...
        PYTHIASingleModel(Ybin, Waux, Znorm, cvfolds, KernelFcn)
    % TODO check details of random number generation and tic-toc in parallel loops.
    state = rng;
    rng('default');
    out_cp = cvpartition(Ybin,'Kfold',cvfolds,'Stratify',true);
    rng('default');
    out_svm = fitcsvm(Znorm,Ybin,'Standardize',false,...
                                         'Weights',Waux,...
                                         'CacheSize','maximal',...
                                         'RemoveDuplicates',true,...
                                         'KernelFunction',KernelFcn,...
                                         'OptimizeHyperparameters','auto',...
                                         'HyperparameterOptimizationOptions',...
                                         struct('CVPartition',out_cp,...
                                                'Verbose',0,...
                                                'AcquisitionFunctionName','probability-of-improvement',...
                                                'ShowPlots',false));
    out_svm = fitSVMPosterior(out_svm);
    rng(state);
    [out_Ysub,aux] = out_svm.resubPredict;
    out_Pr0sub = aux(:,1);
    [out_Yhat,aux] = out_svm.predict(Znorm);
    out_Pr0hat = aux(:,1);
    aux = confusionmat(Ybin,out_Ysub);
    out_cvcmat = aux(:);
    out_boxcosnt = out_svm.HyperparameterOptimizationResults.bestPoint{1,1};
    out_kscale = out_svm.HyperparameterOptimizationResults.bestPoint{1,2};
end
