clear all
close all hidden
clc

%% Parameters
query = "R1_one_col";
dataSize = "1000";
base_dir = "/home/eugenio/Desktop/cineca-runs-20150111/";

C_range = linspace (0.1, 5, 20);
E_range = linspace (0.1, 5, 20);

plot_subdivisions = 20;

%% Real stuff
[values, sample] = read_from_directory ([base_dir, query, "/", dataSize]);

sample_nCores = sample;
sample_nCores(:, end) = 1 ./ sample_nCores(:, end);

everything = [values, sample];
everything = clear_outliers (everything);
[everything, ~, ~] = scale (everything);
y = everything(:, 1);
X = everything(:, 2:end);

everything = [values, sample_nCores];
everything = clear_outliers (everything);
[everything, ~, ~] = scale (everything);
y_nCores = everything(:, 1);
X_nCores = everything(:, 2:end);

test_frac = 0.6;
train_frac = 0.2;

[ytr, Xtr, ytst, Xtst, ycv, Xcv] = split_sample (y, X, train_frac, test_frac);
[ytr_nCores, Xtr_nCores, ytst_nCores, Xtst_nCores, ycv_nCores, Xcv_nCores] = ...
  split_sample (y_nCores, X_nCores, train_frac, test_frac);

RMSEs = zeros (1, 4);
Cs = zeros (1, 4);
Es = zeros (1, 4);
predictions = zeros (numel (ycv), 4);
coefficients = cell (1, 4);
SVs = cell (1, 4);
b = cell (1, 4);

%% White box model, nCores
[C, eps] = model_selection (ytr, Xtr, ytst, Xtst, "-s 3 -t 0 -q -h 0", C_range, E_range);
options = ["-s 3 -t 0 -h 0 -p ", num2str(eps), " -c ", num2str(C)];
model = svmtrain (ytr, Xtr, options);
[predictions(:, 1), accuracy, ~] = svmpredict (ycv, Xcv, model);
Cs(1) = C;
Es(1) = eps;
RMSEs(1) = sqrt (accuracy(2));
coefficients{1} = model.sv_coef;
SVs{1} = model.SVs;
b{1} = - model.rho;

%% White box model, nCores^(-1)
[C, eps] = model_selection (ytr_nCores, Xtr_nCores, ytst_nCores, Xtst_nCores, "-s 3 -t 0 -q -h 0", C_range, E_range);
options = ["-s 3 -t 0 -h 0 -p ", num2str(eps), " -c ", num2str(C)];
model = svmtrain (ytr_nCores, Xtr_nCores, options);
[predictions(:, 2), accuracy, ~] = svmpredict (ycv_nCores, Xcv_nCores, model);
Cs(2) = C;
Es(2) = eps;
RMSEs(2) = sqrt (accuracy(2));
coefficients{2} = model.sv_coef;
SVs{2} = model.SVs;
b{2} = - model.rho;

%% Black box model, Polynomial
[C, eps] = model_selection (ytr, Xtr, ytst, Xtst, "-s 3 -t 1 -q -h 0", C_range, E_range);
options = ["-s 3 -t 1 -h 0 -p ", num2str(eps), " -c ", num2str(C)];
model = svmtrain (ytr, Xtr, options);
[predictions(:, 3), accuracy, ~] = svmpredict (ycv, Xcv, model);
Cs(3) = C;
Es(3) = eps;
RMSEs(3) = sqrt (accuracy(2));
coefficients{3} = model.sv_coef;
SVs{3} = model.SVs;
b{3} = - model.rho;

%% Black box model, RBF
[C, eps] = model_selection (ytr, Xtr, ytst, Xtst, "-s 3 -t 2 -q -h 0", C_range, E_range);
options = ["-s 3 -t 2 -h 0 -p ", num2str(eps), " -c ", num2str(C)];
model = svmtrain (ytr, Xtr, options);
[predictions(:, 4), accuracy, ~] = svmpredict (ycv, Xcv, model);
Cs(4) = C;
Es(4) = eps;
RMSEs(4) = sqrt (accuracy(2));
coefficients{4} = model.sv_coef;
SVs{4} = model.SVs;
b{4} = - model.rho;

robust_avg_value = median (ycv);

percent_RMSEs = 100 * RMSEs / max (RMSEs);
rel_RMSEs = RMSEs / abs (robust_avg_value);

abs_err = abs (predictions - ycv);
rel_err = abs_err ./ abs (ycv);

max_rel_err = max (rel_err);
min_rel_err = min (rel_err);
mean_rel_err = mean (rel_err);

max_abs_err = max (abs_err);
mean_abs_err = mean (abs_err);
min_abs_err = min (abs_err);

mean_y = mean (ycv);
mean_predictions = mean (predictions);
err_mean = mean_predictions - mean_y;
rel_err_mean = abs (err_mean / mean_y);

%% Plots
figure;
plot (X, y, "g+");
hold on;
w = SVs{1}' * coefficients{1};
func = @(x) w .* x + b{1};
M = max (X);
m = min (X);
x = linspace (m, M, plot_subdivisions);
plot (x, func (x), "r-", "linewidth", 2);
axis auto;
title ("Linear kernels");
grid on;

figure;
plot (X_nCores, y_nCores, "g+");
hold on;
w = SVs{2}' * coefficients{2};
func = @(x) w .* x + b{2};
M = max (X_nCores);
m = min (X_nCores);
x = linspace (m, M, plot_subdivisions);
plot (x, func (x), "r-", "linewidth", 2);
axis auto;
title ('Linear kernels, nCores^{- 1}');
grid on;

figure;
plot (X, y, "g+");
hold on;
M = max (X);
m = min (X);
x = linspace (m, M, plot_subdivisions);
z = zeros (size (x));
for (ii = 1:numel (z))
  point = x(ii);
  z(ii) = coefficients{4}' * exp (bsxfun (@minus, SVs{4}, point) .^ 2);
endfor
plot (x, z, "r-", "linewidth", 2);
axis auto;
title ("RBF kernels");
grid on;

%% Print metrics
display ("Root Mean Square Errors");
RMSEs
percent_RMSEs
rel_RMSEs

display ("Relative errors (absolute values)");
max_rel_err
mean_rel_err
min_rel_err

display ("Absolute errors (absolute values)");
max_abs_err
mean_abs_err
min_abs_err

display ("Relative error between mean measure and mean prediction (absolute value)");
rel_err_mean
