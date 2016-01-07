clear all
close all hidden
clc

[values, sample] = read_from_directory ("/home/gianniti/cineca-runs/R1/small");
[big_values, big_sample] = read_from_directory ("/home/gianniti/cineca-runs/R1/big");

sample_nCores = sample;
sample_nCores(:, end) = 1 ./ sample_nCores(:, end);

big_sample_nCores = big_sample;
big_sample_nCores(:, end) = 1 ./ big_sample_nCores(:, end);

[everything, ~, ~] = scale ([values, sample]);
y = everything(:, 1);
X = everything(:, 2:end);

[everything, ~, ~] = scale ([values, sample_nCores]);
y_nCores = everything(:, 1);
X_nCores = everything(:, 2:end);

[everything, ~, ~] = scale ([big_values, big_sample]);
big_y = everything(:, 1);
big_X = everything(:, 2:end);

[everything, ~, ~] = scale ([big_values, big_sample_nCores]);
big_y_nCores = everything(:, 1);
big_X_nCores = everything(:, 2:end);

test_frac = length (big_y) / length (y);
train_frac = 1 - test_frac;

[ytr, Xtr, ytst, Xtst, ~, ~] = split_sample (y, X, train_frac, test_frac);
[ytr_nCores, Xtr_nCores, ytst_nCores, Xtst_nCores, ~, ~] = ...
  split_sample (y_nCores, X_nCores, train_frac, test_frac);
ycv = big_y;
Xcv = big_X;
ycv_nCores = big_y_nCores;
Xcv_nCores = big_X_nCores;

small_dimensional = false;
C_range = linspace (0.1, 5, 20);
E_range = linspace (0.1, 5, 20);

RMSEs = zeros (4, 1);
Cs = zeros (4, 1);
Es = zeros (4, 1);
w = cell (2, 1);
b = cell (2, 1);

%% White box model, nCores
[C, eps] = model_selection (ytr, Xtr, ytst, Xtst, "-s 3 -t 0 -q -h 0", C_range, E_range);
options = ["-s 3 -t 0 -h 0 -p ", num2str(eps), " -c ", num2str(C)];
model = svmtrain (ytr, Xtr, options);
[~, accuracy, ~] = svmpredict (ycv, Xcv, model);
Cs(1) = C;
Es(1) = eps;
RMSEs(1) = sqrt (accuracy(2));
w{1} = model.SVs' * model.sv_coef;
b{1} = - model.rho;

%% White box model, nCores^(-1)
[C, eps] = model_selection (ytr_nCores, Xtr_nCores, ytst_nCores, Xtst_nCores, "-s 3 -t 0 -q -h 0", C_range, E_range);
options = ["-s 3 -t 0 -h 0 -p ", num2str(eps), " -c ", num2str(C)];
model = svmtrain (ytr_nCores, Xtr_nCores, options);
[~, accuracy, ~] = svmpredict (ycv_nCores, Xcv_nCores, model);
Cs(2) = C;
Es(2) = eps;
RMSEs(2) = sqrt (accuracy(2));
w{2} = model.SVs' * model.sv_coef;
b{2} = - model.rho;

%% Black box model, Polynomial
[C, eps] = model_selection (ytr, Xtr, ytst, Xtst, "-s 3 -t 1 -q -h 0", C_range, E_range);
options = ["-s 3 -t 1 -h 0 -p ", num2str(eps), " -c ", num2str(C)];
model = svmtrain (ytr, Xtr, options);
[~, accuracy, ~] = svmpredict (ycv, Xcv, model);
Cs(3) = C;
Es(3) = eps;
RMSEs(3) = sqrt (accuracy(2));

%% Black box model, RBF
[C, eps] = model_selection (ytr, Xtr, ytst, Xtst, "-s 3 -t 2 -q -h 0", C_range, E_range);
options = ["-s 3 -t 2 -h 0 -p ", num2str(eps), " -c ", num2str(C)];
model = svmtrain (ytr, Xtr, options);
[~, accuracy, ~] = svmpredict (ycv, Xcv, model);
Cs(4) = C;
Es(4) = eps;
RMSEs(4) = sqrt (accuracy(2));

percent_RMSEs = RMSEs / max (RMSEs);
rel_RMSEs = RMSEs / median (values);

RMSEs
percent_RMSEs
rel_RMSEs

if (small_dimensional)
  figure;
  h = plot (X, y, "g+");
  hold on;
  func = @(x) w{1}' * x + b{1};
  ezplot (func, get (h, "xlim"));
  axis auto;
  title ("Linear kernels");
  grid on;

  figure;
  h = plot (X_nCores, y_nCores, "g+");
  hold on;
  func = @(x) w{2}' * x + b{2};
  ezplot (func, get (h, "xlim"));
  axis auto;
  title ("Polynomial kernels");
  grid on;
endif
