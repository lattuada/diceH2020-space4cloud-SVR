clear all
close all hidden
clc

%% Parameters
query = "R1_two_cols";
base_dir = "/home/eugenio/Desktop/cineca-runs-20150111/";

dimensions = 2;
C_range = linspace (0.1, 5, 20);
E_range = linspace (0.1, 5, 20);

%% Real stuff
[values, sample] = read_from_directory ([base_dir, query, "/small"]);
[big_values, big_sample] = read_from_directory ([base_dir, query, "/big"]);

sample_nCores = sample;
sample_nCores(:, end) = 1 ./ sample_nCores(:, end);

big_sample_nCores = big_sample;
big_sample_nCores(:, end) = 1 ./ big_sample_nCores(:, end);

big_size = max (big_sample(:, end - 1));
everything = [values, sample; big_values, big_sample];
before = size (everything);
everything = clear_outliers (everything);
idx_small = (everything(:, end - 1) < big_size);
idx_big = (everything(:, end - 1) == big_size);
[everything, ~, ~] = scale (everything);
y = everything(idx_small, 1);
X = everything(idx_small, 2:end);
big_y = everything(idx_big, 1);
big_X = everything(idx_big, 2:end);

big_size = max (big_sample_nCores(:, end - 1));
everything = [values, sample_nCores; big_values, big_sample_nCores];
everything = clear_outliers (everything);
idx_small = (everything(:, end - 1) < big_size);
idx_big = (everything(:, end - 1) == big_size);
[everything, ~, ~] = scale (everything);
y_nCores = everything(idx_small, 1);
X_nCores = everything(idx_small, 2:end);
big_y_nCores = everything(idx_big, 1);
big_X_nCores = everything(idx_big, 2:end);

test_frac = length (big_y) / length (y);
train_frac = 1 - test_frac;

[ytr, Xtr, ytst, Xtst, ~, ~] = split_sample (y, X, train_frac, test_frac);
[ytr_nCores, Xtr_nCores, ytst_nCores, Xtst_nCores, ~, ~] = ...
  split_sample (y_nCores, X_nCores, train_frac, test_frac);
ycv = big_y;
Xcv = big_X;
ycv_nCores = big_y_nCores;
Xcv_nCores = big_X_nCores;

RMSEs = zeros (1, 4);
Cs = zeros (1, 4);
Es = zeros (1, 4);
predictions = zeros (numel (ycv), 4);
w = cell (1, 2);
b = cell (1, 2);

%% White box model, nCores
[C, eps] = model_selection (ytr, Xtr, ytst, Xtst, "-s 3 -t 0 -q -h 0", C_range, E_range);
options = ["-s 3 -t 0 -h 0 -p ", num2str(eps), " -c ", num2str(C)];
model = svmtrain (ytr, Xtr, options);
[predictions(:, 1), accuracy, ~] = svmpredict (ycv, Xcv, model);
Cs(1) = C;
Es(1) = eps;
RMSEs(1) = sqrt (accuracy(2));
w{1} = model.SVs' * model.sv_coef;
b{1} = - model.rho;

%% White box model, nCores^(-1)
[C, eps] = model_selection (ytr_nCores, Xtr_nCores, ytst_nCores, Xtst_nCores, "-s 3 -t 0 -q -h 0", C_range, E_range);
options = ["-s 3 -t 0 -h 0 -p ", num2str(eps), " -c ", num2str(C)];
model = svmtrain (ytr_nCores, Xtr_nCores, options);
[predictions(:, 2), accuracy, ~] = svmpredict (ycv_nCores, Xcv_nCores, model);
Cs(2) = C;
Es(2) = eps;
RMSEs(2) = sqrt (accuracy(2));
w{2} = model.SVs' * model.sv_coef;
b{2} = - model.rho;

%% Black box model, Polynomial
[C, eps] = model_selection (ytr, Xtr, ytst, Xtst, "-s 3 -t 1 -q -h 0", C_range, E_range);
options = ["-s 3 -t 1 -h 0 -p ", num2str(eps), " -c ", num2str(C)];
model = svmtrain (ytr, Xtr, options);
[predictions(:, 3), accuracy, ~] = svmpredict (ycv, Xcv, model);
Cs(3) = C;
Es(3) = eps;
RMSEs(3) = sqrt (accuracy(2));

%% Black box model, RBF
[C, eps] = model_selection (ytr, Xtr, ytst, Xtst, "-s 3 -t 2 -q -h 0", C_range, E_range);
options = ["-s 3 -t 2 -h 0 -p ", num2str(eps), " -c ", num2str(C)];
model = svmtrain (ytr, Xtr, options);
[predictions(:, 4), accuracy, ~] = svmpredict (ycv, Xcv, model);
Cs(4) = C;
Es(4) = eps;
RMSEs(4) = sqrt (accuracy(2));

robust_avg_value = median (ycv);

percent_RMSEs = 100 * RMSEs / max (RMSEs);
rel_RMSEs = abs (RMSEs / robust_avg_value);

err = predictions - ycv;
rel_err = err ./ ycv;
max_rel_err = max (rel_err);
min_rel_err = min (rel_err);
mean_abs_err = mean (abs (err));
rel_mean_abs_err = abs (mean_abs_err / robust_avg_value);

mean_y = mean (ycv);
mean_predictions = mean (predictions);
err_mean = mean_predictions - mean_y;
rel_err_mean = err_mean / mean_y;

%% Plots
switch (dimensions)
  case {1}
    figure;
    plot (X, y, "g+");
    hold on;
    func = @(x) w{1} .* x + b{1};
    extremes = xlim ();
    x = linspace (extremes(1), extremes(2), 10);
    plot (x, func (x), "r-");
    axis auto;
    title ("Linear kernels");
    grid on;
    
    figure;
    plot (X_nCores, y_nCores, "g+");
    hold on;
    func = @(x) w{2} .* x + b{2};
    extremes = xlim ();
    x = linspace (extremes(1), extremes(2), 10);
    plot (x, func (x), "r-");
    axis auto;
    title ('Linear kernels, nCores^{- 1}');
    grid on;
  case {2}
    figure;
    plot3 (X(:, 1), X(:, 2), y, "g+");
    hold on;
    func = @(x, y) w{1}(1) .* x + w{1}(2) .* y + b{1};
    extremes = xlim ();
    x = linspace (extremes(1), extremes(2), 10);
    extremes = xlim ();
    x = linspace (extremes(1), extremes(2), 10);
    extremes = ylim ();
    yy = linspace (extremes(1), extremes(2), 10);
    [XX, YY] = meshgrid (x, yy);
    surf (XX, YY, func (XX, YY));
    axis auto;
    title ("Linear kernels");
    grid on;
    
    figure;
    plot3 (X_nCores(:, 1), X_nCores(:, 2), y_nCores, "g+");
    hold on;
    func = @(x, y) w{2}(1) .* x + w{2}(2) .* y + b{2};
    extremes = xlim ();
    x = linspace (extremes(1), extremes(2), 10);
    extremes = ylim ();
    yy = linspace (extremes(1), extremes(2), 10);
    [XX, YY] = meshgrid (x, yy);
    surf (XX, YY, func (XX, YY));
    axis auto;
    title ('Linear kernels, nCores^{- 1}');
    grid on;
endswitch

%% Print metrics
RMSEs
percent_RMSEs
rel_RMSEs
max_rel_err
min_rel_err
mean_abs_err
rel_mean_abs_err
rel_err_mean
