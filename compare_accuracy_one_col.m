clear all
close all hidden
clc

%% Parameters
query = "R1_one_col";
dataSize = "250";
base_dir = "/home/eugenio/Desktop/cineca-runs-20150111/";

train_frac = 0.6;
test_frac = 0.2;

C_range = linspace (0.1, 5, 20);
E_range = linspace (0.1, 5, 20);

printPlots = false;
plot_subdivisions = 20;

%% Real stuff
sample = read_from_directory ([base_dir, query, "/", dataSize]);
sample = sample(randperm (size (sample, 1)), :);
sample_nCores = sample;
sample_nCores(:, end) = 1 ./ sample_nCores(:, end);

sample = clear_outliers (sample);
[scaled, ~, ~] = zscore (sample);
y = scaled(:, 1);
X = scaled(:, 2:end);
[ytr, ytst, ycv] = split_sample (y, train_frac, test_frac);
[Xtr, Xtst, Xcv] = split_sample (X, train_frac, test_frac);

sample_nCores = clear_outliers (sample_nCores);
[scaled_nCores, ~, ~] = zscore (sample_nCores);
y_nCores = scaled_nCores(:, 1);
X_nCores = scaled_nCores(:, 2:end);
[ytr_nCores, ytst_nCores, ycv_nCores] = split_sample (y_nCores, train_frac, test_frac);
[Xtr_nCores, Xtst_nCores, Xcv_nCores] = split_sample (X_nCores, train_frac, test_frac);

RMSEs = zeros (1, 4);
Cs = zeros (1, 4);
Es = zeros (1, 4);
predictions = zeros (numel (ycv), 4);
coefficients = cell (1, 4);
SVs = cell (1, 4);
b = cell (1, 4);
models = cell (1, 4);

%% White box model, nCores
[C, eps] = model_selection (ytr, Xtr, ytst, Xtst, "-s 3 -t 0 -q -h 0", C_range, E_range);
options = ["-s 3 -t 0 -h 0 -p ", num2str(eps), " -c ", num2str(C)];
models{1} = svmtrain (ytr, Xtr, options);
[predictions(:, 1), accuracy, ~] = svmpredict (ycv, Xcv, models{1});
Cs(1) = C;
Es(1) = eps;
RMSEs(1) = sqrt (accuracy(2));
coefficients{1} = models{1}.sv_coef;
SVs{1} = models{1}.SVs;
b{1} = - models{1}.rho;

%% White box model, nCores^(-1)
[C, eps] = model_selection (ytr_nCores, Xtr_nCores, ytst_nCores, Xtst_nCores, "-s 3 -t 0 -q -h 0", C_range, E_range);
options = ["-s 3 -t 0 -h 0 -p ", num2str(eps), " -c ", num2str(C)];
models{2} = svmtrain (ytr_nCores, Xtr_nCores, options);
[predictions(:, 2), accuracy, ~] = svmpredict (ycv_nCores, Xcv_nCores, models{2});
Cs(2) = C;
Es(2) = eps;
RMSEs(2) = sqrt (accuracy(2));
coefficients{2} = models{2}.sv_coef;
SVs{2} = models{2}.SVs;
b{2} = - models{2}.rho;

%% Black box model, Polynomial
[C, eps] = model_selection (ytr, Xtr, ytst, Xtst, "-s 3 -t 1 -q -h 0", C_range, E_range);
options = ["-s 3 -t 1 -h 0 -p ", num2str(eps), " -c ", num2str(C)];
models{3} = svmtrain (ytr, Xtr, options);
[predictions(:, 3), accuracy, ~] = svmpredict (ycv, Xcv, models{3});
Cs(3) = C;
Es(3) = eps;
RMSEs(3) = sqrt (accuracy(2));
coefficients{3} = models{3}.sv_coef;
SVs{3} = models{3}.SVs;
b{3} = - models{3}.rho;

%% Black box model, RBF
[C, eps] = model_selection (ytr, Xtr, ytst, Xtst, "-s 3 -t 2 -q -h 0", C_range, E_range);
options = ["-s 3 -t 2 -h 0 -p ", num2str(eps), " -c ", num2str(C)];
models{4} = svmtrain (ytr, Xtr, options);
[predictions(:, 4), accuracy, ~] = svmpredict (ycv, Xcv, models{4});
Cs(4) = C;
Es(4) = eps;
RMSEs(4) = sqrt (accuracy(2));
coefficients{4} = models{4}.sv_coef;
SVs{4} = models{4}.SVs;
b{4} = - models{4}.rho;

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

%% Avg
avgs = zeros (length (unique (sort (X))), 4);
err_on_avg = zeros (1, 4);
for (jj = 1:length (err_on_avg))
  dataset = X;
  if (jj == 2)
    dataset = X_nCores;
  endif
  cores = unique (sort (dataset));
  avg = zeros (size (cores));
  for (ii = 1:numel (cores))
    avg(ii) = mean (y(dataset == cores(ii)));
  endfor
  avgs(:, jj) = avg;
  [pred, ~, ~] = svmpredict (avg, cores, models{jj});
  err_on_avg(jj) = mean (abs ((pred - avg) ./ avg));
endfor

%% Plots
if (printPlots)
  figure;
  plot (X, y, "g+");
  hold on;
  cores = unique (sort (X));
  plot (cores, avgs(:, 1), "kd");
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
  cores = unique (sort (X_nCores));
  plot (cores, avgs(:, 2), "kd");
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
  cores = unique (sort (X));
  plot (cores, avgs(:, 4), "kd");
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
endif

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

display ("Relative error between mean measure and mean prediction, grouped by number of cores (absolute value)");
err_on_avg
