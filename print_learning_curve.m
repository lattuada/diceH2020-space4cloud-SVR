clear all
close all hidden
clc

train_frac = 0.8;
test_frac = 0.2;

[values, sample] = read_from_directory ("/home/eugenio/Desktop/cineca-runs-20150111/R1/small");
[values_big, sample_big] = read_from_directory ("/home/eugenio/Desktop/cineca-runs-20150111/R1/big");
[everything, ~, ~] = zscore ([values, sample]);
y = everything(:, 1);
X = everything(:, 2:end);
[everything, ~, ~] = zscore ([values_big, sample_big]);
ycv = everything(:, 1);
Xcv = everything(:, 2:end);
[ytr, Xtr, ytst, Xtst, ~, ~] = split_sample (y, X, train_frac, test_frac);

C_range = linspace (0.1, 5, 20);
E_range = linspace (0.1, 5, 20);
initial_options = "-s 3 -t 0 -q";
[C, eps] = model_selection (ytr, Xtr, ytst, Xtst, initial_options, C_range, E_range);
options = [initial_options, " -p ", num2str(eps), " -c ", num2str(C)];
learning_curves (ytr, Xtr, ycv, Xcv, options);
