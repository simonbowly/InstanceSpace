
function [elapsed_time, out_cp, out_svm, out_Ysub, out_Pr0sub, ...
    out_Yhat, out_Pr0hat, out_cvcmat, out_boxcosnt, out_kscale] = ...
    PYTHIASingleModel(Ybin, Waux, Znorm, cvfolds, KernelFcn)
% -------------------------------------------------------------------------
% PYTHIASingleModel.m
% -------------------------------------------------------------------------
%
% Fits a PYTHIA SVM for a single algorithm.
%
% TODO verify random number generation and timing works as expected
% when called using parfeval.
%
% -------------------------------------------------------------------------
start_time = tic;
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
elapsed_time = toc(start_time);
end
