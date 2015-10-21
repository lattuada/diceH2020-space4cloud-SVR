clear all
close all hidden
clc

[values, sample] = read_data ("dati.csv");

sample_nCores = sample;
sample_nCores(:, end) = 1 ./ sample_nCores(:, end);

[X, ~, dev] = scale (sample);
idx = find (dev != 0);
X = X(:, idx);
dev = dev(idx);

[X_nCores, ~, dev_nCores] = scale (sample);
idx = find (dev_nCores != 0);
X_nCores = X_nCores(:, idx);
dev_nCores = dev_nCores(idx);

[ytr, Xtr, ytst, Xtst, ycv, Xcv] = split_sample (values, X_nCores, 0.6, 0.2);
[ytr_nCores, Xtr_nCores, ytst_nCores, Xtst_nCores, ycv_nCores, Xcv_nCores] = ...
  split_sample (values, X_nCores, 0.6, 0.2);

MSEs = zeros (4, 1);

%% White box model, nCores
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
[~, accuracy, ~] = svmpredict (ycv, Xcv, model);
MSEs(1) = accuracy(2);

%% White box model, nCores^(-1)
C = Inf;
eps = Inf;
MSE = Inf;
for (cc = logspace (-5, 5))
  for (ee = logspace (-5, 5))
    options = ["-s 3 -t 0 -q -p ", num2str(ee), " -c ", num2str(cc)];
    model = svmtrain (ytr_nCores, Xtr_nCores, options);
    [~, accuracy, ~] = svmpredict (ytst_nCores, Xtst_nCores, model, "-q");
    mse = accuracy(2);
    if (mse < MSE)
      C = cc;
      eps = ee;
      MSE = mse;
    endif
  endfor
endfor

options = ["-s 3 -t 0 -p ", num2str(eps), " -c ", num2str(C)];
model = svmtrain (ytr_nCores, Xtr_nCores, options);
[~, accuracy, ~] = svmpredict (ycv_nCores, Xcv_nCores, model);
MSEs(2) = accuracy(2);

%% Black box model, Polynomial
C = Inf;
eps = Inf;
MSE = Inf;
for (cc = logspace (-5, 5))
  for (ee = logspace (-5, 5))
    options = ["-s 3 -t 1 -q -p ", num2str(ee), " -c ", num2str(cc)];
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

options = ["-s 3 -t 1 -p ", num2str(eps), " -c ", num2str(C)];
model = svmtrain (ytr, Xtr, options);
[~, accuracy, ~] = svmpredict (ycv, Xcv, model);
MSEs(3) = accuracy(2);

%% Black box model, RBF
C = Inf;
eps = Inf;
MSE = Inf;
for (cc = logspace (-5, 5))
  for (ee = logspace (-5, 5))
    options = ["-s 3 -t 2 -q -p ", num2str(ee), " -c ", num2str(cc)];
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

options = ["-s 3 -t 2 -p ", num2str(eps), " -c ", num2str(C)];
model = svmtrain (ytr, Xtr, options);
[~, accuracy, ~] = svmpredict (ycv, Xcv, model);
MSEs(4) = accuracy(2);

MSEs
