clear all
close all hidden
clc

train_frac = 0.8;
test_frac = 0.2;

sample = read_from_directory ("/home/eugenio/Desktop/cineca-runs-20150111/R1/small");
sample_big = read_from_directory ("/home/eugenio/Desktop/cineca-runs-20150111/R1/big");

scaled = zscore (sample_big);
ycv = scaled(:, 1);
Xcv = scaled(:, 2:end);
scaled = zscore (sample);

scaled = scaled(randperm (size (scaled, 1)), :);
[train, test, ~] = split_sample (scaled, train_frac, test_frac);
ytr = train(:, 1);
Xtr = train(:, 2:end);
ytst = test(:, 1);
Xtst = test(:, 2:end);

C_range = linspace (0.1, 5, 20);
E_range = linspace (0.1, 5, 20);
initial_options = "-s 3 -t 0 -q";
[C, eps] = model_selection (ytr, Xtr, ytst, Xtst, initial_options, C_range, E_range);
options = [initial_options, " -p ", num2str(eps), " -c ", num2str(C)];
learning_curves (ytr, Xtr, ycv, Xcv, options);
