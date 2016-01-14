clear all
close all hidden
clc

%% Parameters
query = "R5_two_cols";
base_dir = "/home/eugenio/Desktop/cineca-runs-20150111/";

C_range = linspace (0.1, 5, 20);
E_range = linspace (0.1, 5, 20);

plot_subdivisions = 20;

%% Real stuff
[values, sample] = read_from_directory ([base_dir, query, "/small"]);
[big_values, big_sample] = read_from_directory ([base_dir, query, "/big"]);

dimensions = size (sample, 2);

sample_nCores = sample;
sample_nCores(:, end) = 1 ./ sample_nCores(:, end);

big_sample_nCores = big_sample;
big_sample_nCores(:, end) = 1 ./ big_sample_nCores(:, end);

big_size = max (big_sample(:, end - 1));
everything = [values, sample; big_values, big_sample];
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
switch (dimensions)
  case {2}
    figure;
    plot3 (X(:, 1), X(:, 2), y, "g+");
    hold on;
    plot3 (big_X(:, 1), big_X(:, 2), big_y, "bd");
    w = SVs{1}' * coefficients{1};
    func = @(x, y) w(1) .* x + w(2) .* y + b{1};
    Ms = max ([X; big_X]);
    ms = min ([X; big_X]);
    x = linspace (ms(1), Ms(1), plot_subdivisions);
    yy = linspace (ms(2), Ms(2), plot_subdivisions);
    [XX, YY] = meshgrid (x, yy);
    surf (XX, YY, func (XX, YY));
    axis auto;
    title ("Linear kernels");
    grid on;
    
    figure;
    plot3 (X_nCores(:, 1), X_nCores(:, 2), y_nCores, "g+");
    hold on;
    plot3 (big_X_nCores(:, 1), big_X_nCores(:, 2), big_y_nCores, "bd");
    w = SVs{2}' * coefficients{2};
    func = @(x, y) w(1) .* x + w(2) .* y + b{2};
    Ms = max ([X_nCores; big_X_nCores]);
    ms = min ([X_nCores; big_X_nCores]);
    x = linspace (ms(1), Ms(1), plot_subdivisions);
    yy = linspace (ms(2), Ms(2), plot_subdivisions);
    [XX, YY] = meshgrid (x, yy);
    surf (XX, YY, func (XX, YY));
    axis auto;
    title ('Linear kernels, nCores^{- 1}');
    grid on;
    
    figure;
    plot3 (X(:, 1), X(:, 2), y, "g+");
    hold on;
    plot3 (big_X(:, 1), big_X(:, 2), big_y, "bd");
    Ms = max ([X; big_X]);
    ms = min ([X; big_X]);
    x = linspace (ms(1), Ms(1), plot_subdivisions);
    yy = linspace (ms(2), Ms(2), plot_subdivisions);
    [XX, YY] = meshgrid (x, yy);
    [nr, nc] = size (XX);
    ZZ = zeros (nr, nc);
    for (r = 1:nr)
      for (c = 1:nc)
        point = [XX(r, c), YY(r, c)];
        ZZ(r, c) = coefficients{4}' * exp (sumsq (bsxfun (@minus, SVs{4}, point), 2) / 2);
      endfor
    endfor
    surf (XX, YY, ZZ);
    axis auto;
    title ("RBF kernels");
    grid on;
endswitch

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
