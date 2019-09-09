function out = fitoracle(Z,Ybin,weight,opts)

% global params
Ybin = double(Ybin)+1;
nalgos = size(Ybin,2);
out.paramgrid = sortrows(2.^(opts.maxcvgrid.*lhsdesign(opts.cvgrid,2) + ...
                             opts.mincvgrid));  % Cross-validation grid
out.cvmcr = NaN.*ones(opts.cvgrid,nalgos);
out.paramidx = NaN.*ones(1,nalgos);
out.modelerr = NaN.*ones(1,nalgos);

for i=1:nalgos
    for j=1:opts.cvgrid
        out.cvmcr(j,i) = crossval('mcr', Z, Ybin(:,i),...
                                  'Kfold', opts.cvfolds,...
                                  'Options', statset('UseParallel',true),...
                                  'Predfun',@(xtrain,ytrain,xtest) svmwrap(xtrain,...
                                                                           ytrain,...
                                                                           xtest,...
                                                                           out.paramgrid(j,:),...
                                                                           weight(i),...
                                                                           false));
    end
    [out.modelerr(i),out.paramidx(i)] = min(out.cvmcr(:,i));
    disp(['    -> ' num2str(i) ' out of ' num2str(nalgos) ' models have been fitted.']);
end
disp(['    -> Completed - Average cross validation error is: ' ...
      num2str(round(100.*mean(out.modelerr),1)) '%']);

out.Yhat = 0.*Ybin;
out.probs = 0.*Ybin;
out.svm = cell(1,nalgos);
out.svmparams = zeros(nalgos,2);
for i=1:nalgos
    out.svmparams(i,:) = out.paramgrid(out.paramidx(i),:);
    [aux, out.svm{i}] = svmwrap(Z, Ybin(:,i), Z, out.svmparams(i,:), weight(i), true);
    out.Yhat(:,i)  = aux(:,1);
    out.probs(:,i) = aux(:,2);
end
out.Yhat = out.Yhat==2; % Make it binary
% We assume that the most accurate SVM (as per CV-Error) is the most
% reliable.
[mostaccurate,out.psel] = max(bsxfun(@times,out.Yhat,1-out.modelerr),[],2);
out.psel(mostaccurate<=0) = 0;

end