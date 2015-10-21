clear all
close all hidden
clc

[values, sample] = read_data ("dati.csv");

[X, avg, dev] = scale (sample);
idx = find (dev != 0);
X = X(:, idx);
avg = avg(idx);
dev = dev(idx);

[ytr, Xtr, ytst, Xtst, ycv, Xcv] = split_sample (values, X, 0.6, 0.2);

C = Inf;
eps = Inf;
MSE = Inf;
for (cc = logspace (-5, 5))
  for (ee = logspace (-5, 5))
    options = ["-s 3 -t 0 -q -p ", num2str(ee), " -c ", num2str(cc)];
    model = svmtrain (ytr, Xtr, options);
    [~, accuracy, ~] = svmpredict (ytst, Xtst, model, "-q");
    mse = accuracy(2);
    if (mse < MSE)
      C = cc;
      eps = ee;
      MSE = mse;
    endif
  endfor
endfor

options = ["-s 3 -t 0 -p ", num2str(eps), " -c ", num2str(C)];
model = svmtrain (ytr, Xtr, options);
[predictions, accuracy, ~] = svmpredict (ycv, Xcv, model);

avgR2 = kfold (values, X, C, eps, 10)
