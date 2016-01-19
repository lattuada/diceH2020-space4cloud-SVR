clear all
close all hidden
clc

%% Parameters
query = "R1_two_cols";
base_dir = "/home/eugenio/Desktop/cineca-runs-20160116/";

seeds = 1:17;

train_frac = 0.6;
test_frac = 0.2;

%% Model stuff
C_range = linspace (0.1, 5, 20);
E_range = linspace (0.1, 5, 20);
initial_options = "-s 3 -t 0 -q";

%% Work
sample = read_from_directory ([base_dir, query, "/small"]);
sample_big = read_from_directory ([base_dir, query, "/big"]);
complete_sample = [sample; sample_big];
scaled = zscore (complete_sample);

rand ("seed", seeds(1));
shuffled = scaled(randperm (size (scaled, 1)), :);
[train, test, ~] = split_sample (shuffled, train_frac, test_frac);
ytr = train(:, 1);
Xtr = train(:, 2:end);
ytst = test(:, 1);
Xtst = test(:, 2:end);

[C, eps] = model_selection (ytr, Xtr, ytst, Xtst, initial_options, C_range, E_range);
options = [initial_options, " -p ", num2str(eps), " -c ", num2str(C)];

seeds = seeds(:)';
m = MSEtrain = MSEcv = [];
for (seed = seeds)
  rand ("seed", seed);
  shuffled = scaled(randperm (size (scaled, 1)), :);
  [train, ~, cv] = split_sample (shuffled, train_frac, test_frac);
  ytr = train(:, 1);
  Xtr = train(:, 2:end);
  ycv = cv(:, 1);
  Xcv = cv(:, 2:end);
  
  [current_m, current_MSEtrain, current_MSEcv] = learning_curves (ytr, Xtr, ycv, Xcv, options);
  m = [m; current_m];
  MSEtrain = [MSEtrain; current_MSEtrain];
  MSEcv = [MSEcv; current_MSEcv];
endfor

old_m = m;
m = mean (old_m);
if (any (m != old_m))
  error ("Something went wrong with the sample size steps");
endif
MSEtrain = mean (MSEtrain);
MSEcv = mean (MSEcv);

plot_learning_curves (m, MSEtrain, MSEcv);
