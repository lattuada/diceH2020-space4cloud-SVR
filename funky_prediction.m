clear all
close all hidden
clc

%% Parameters
query = "everything/max";
base_dir = "/home/eugenio/Desktop/cineca-runs-20160116/";

target = 6 * 60 * 1000;

C_range = linspace (0.1, 5, 20);
E_range = linspace (0.1, 5, 20);

train_frac = 0.8;

%% Real work
sample = read_from_directory ([base_dir, query, "/big"]);

features = read_from_directory ([base_dir, query, "/predict"]);
labels = features(:, 1);
features = features(:, 2:end);
features_nCores = features;
features_nCores(:, end) = 1 ./ features_nCores(:, end);

[clean_sample, ~] = clear_outliers (sample);
clean_sample_nCores = clean_sample;
clean_sample_nCores(:, end) = 1 ./ clean_sample_nCores(:, end);

rand ("seed", 17);
idx = randperm (size (clean_sample, 1));

shuffled = clean_sample(idx, :);
[scaled, mu, sigma] = zscore (shuffled);
y = scaled(:, 1);
X = scaled(:, 2:end);
mu_y = mu(1);
sigma_y = sigma(1);
mu_X = mu(2:end);
sigma_X = sigma(2:end);

shuffled_nCores = clean_sample_nCores(idx, :);
[scaled_nCores, mu, sigma] = zscore (shuffled_nCores);
y_nCores = scaled_nCores(:, 1);
X_nCores = scaled_nCores(:, 2:end);
mu_X_nCores = mu(2:end);
sigma_X_nCores = sigma(2:end);

test_frac = 1 - train_frac;
[ytr, ytst, ~] = split_sample (y, train_frac, test_frac);
[Xtr, Xtst, ~] = split_sample (X, train_frac, test_frac);
[ytr_nCores, ytst_nCores, ~] = split_sample (y_nCores, train_frac, test_frac);
[Xtr_nCores, Xtst_nCores, ~] = split_sample (X_nCores, train_frac, test_frac);

%% White box model, nCores
[C, eps] = model_selection (ytr, Xtr, ytst, Xtst, "-s 3 -t 0 -q -h 0", C_range, E_range);
options = ["-s 3 -t 0 -h 0 -q -p ", num2str(eps), " -c ", num2str(C)];
lm = svmtrain (ytr, Xtr, options);

%% White box model, nCores^(-1)
[C, eps] = model_selection (ytr_nCores, Xtr_nCores, ytst_nCores, Xtst_nCores, "-s 3 -t 0 -q -h 0", C_range, E_range);
options = ["-s 3 -t 0 -h 0 -q -p ", num2str(eps), " -c ", num2str(C)];
nlm = svmtrain (ytr_nCores, Xtr_nCores, options);

%% Predictions
safe_sigma_X = sigma_X + (sigma_X == 0);
safe_sigma_X_nCores = sigma_X_nCores + (sigma_X_nCores == 0);
scaled_features = bsxfun (@rdivide, bsxfun (@minus, features, mu_X), safe_sigma_X);
scaled_features_nCores = bsxfun (@rdivide, bsxfun (@minus, features_nCores, mu_X_nCores), safe_sigma_X_nCores);

scaled_predictions = svmpredict (labels, scaled_features, lm, "-q");
scaled_predictions_nCores = svmpredict (labels, scaled_features_nCores, nlm, "-q");

safe_sigma_y = sigma_y + (sigma_y == 0);
predictions = mu_y + scaled_predictions * sigma_y;
predictions_nCores = mu_y + scaled_predictions_nCores * sigma_y;
