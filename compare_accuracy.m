clear all
close all hidden
clc

[values, sample] = read_from_directory ("/home/eugenio/Desktop/csv");

sample_nCores = sample;
sample_nCores(:, end) = 1 ./ sample_nCores(:, end);

[X, ~, ~] = scale (sample);
[X_nCores, ~, ~] = scale (sample);

[ytr, Xtr, ytst, Xtst, ycv, Xcv] = split_sample (values, X, 0.6, 0.2);
[ytr_nCores, Xtr_nCores, ytst_nCores, Xtst_nCores, ycv_nCores, Xcv_nCores] = ...
  split_sample (values, X_nCores, 0.6, 0.2);

RMSEs = zeros (4, 1);
range = [0.1 0.5];

%% White box model, nCores
[C, eps] = model_selection (ytr, Xtr, ytst, Xtst, "-s 3 -t 0 -q -h 0", range, range);
options = ["-s 3 -t 0 -h 0 -p ", num2str(eps), " -c ", num2str(C)];
model = svmtrain (ytr, Xtr, options);
[~, accuracy, ~] = svmpredict (ycv, Xcv, model);
RMSEs(1) = sqrt (accuracy(2));

%% White box model, nCores^(-1)
[C, eps] = model_selection (ytr_nCores, Xtr_nCores, ytst_nCores, Xtst_nCores, "-s 3 -t 0 -q -h 0", range, range);
options = ["-s 3 -t 0 -h 0 -p ", num2str(eps), " -c ", num2str(C)];
model = svmtrain (ytr_nCores, Xtr_nCores, options);
[~, accuracy, ~] = svmpredict (ycv_nCores, Xcv_nCores, model);
RMSEs(2) = sqrt (accuracy(2));

%% Black box model, Polynomial
[C, eps] = model_selection (ytr, Xtr, ytst, Xtst, "-s 3 -t 1 -q -h 0", range, range);
options = ["-s 3 -t 1 -h 0 -p ", num2str(eps), " -c ", num2str(C)];
model = svmtrain (ytr, Xtr, options);
[~, accuracy, ~] = svmpredict (ycv, Xcv, model);
RMSEs(3) = sqrt (accuracy(2));

%% Black box model, RBF
[C, eps] = model_selection (ytr, Xtr, ytst, Xtst, "-s 3 -t 2 -q -h 0", range, range);
options = ["-s 3 -t 2 -h 0 -p ", num2str(eps), " -c ", num2str(C)];
model = svmtrain (ytr, Xtr, options);
[~, accuracy, ~] = svmpredict (ycv, Xcv, model);
RMSEs(4) = sqrt (accuracy(2));

rel_RMSEs = RMSEs / max (RMSEs);

RMSEs
rel_RMSEs
