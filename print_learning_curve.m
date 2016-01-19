clear all
close all hidden
clc

%% Parameters
query = "R1_two_cols";

train_frac = 0.6;
test_frac = 0.2;

%% Work
base_dir = "/home/eugenio/Desktop/cineca-runs-20150111/";
sample = read_from_directory ([base_dir, query, "/small"]);
sample_big = read_from_directory ([base_dir, query, "/big"]);

rand ("seed", 17);
complete_sample = [sample; sample_big];
scaled = zscore (complete_sample);
scaled = scaled(randperm (size (scaled, 1)), :);
[train, test, cv] = split_sample (scaled, train_frac, test_frac);
ytr = train(:, 1);
Xtr = train(:, 2:end);
ytst = test(:, 1);
Xtst = test(:, 2:end);
ycv = cv(:, 1);
Xcv = cv(:, 2:end);

C_range = linspace (0.1, 5, 20);
E_range = linspace (0.1, 5, 20);
initial_options = "-s 3 -t 0 -q";
[C, eps] = model_selection (ytr, Xtr, ytst, Xtst, initial_options, C_range, E_range);
options = [initial_options, " -p ", num2str(eps), " -c ", num2str(C)];
[m, MSEtrain, MSEcv] = learning_curves (ytr, Xtr, ycv, Xcv, options);
plot_learning_curves (m, MSEtrain, MSEcv);
