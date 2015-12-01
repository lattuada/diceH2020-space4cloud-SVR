clear all
close all hidden
clc

train_frac = 0.6;
test_frac = 0.2;
cv_frac = 1 - train_frac - test_frac;

[values, sample] = read_from_directory ("/home/eugenio/Desktop/csv");
[everything, ~, ~] = scale ([values, sample]);
y = everything(:, 1);
X = everything(:, 2:end);
[ytr, Xtr, ytst, Xtst, ycv, Xcv] = split_sample (y, X, train_frac, test_frac);

C_range = [0.1 0.3 0.5 1];
E_range = [0.1 0.3 0.5 1];
initial_options = "-s 3 -t 0 -q";
[C, eps] = model_selection (ytr, Xtr, ytst, Xtst, initial_options, C_range, E_range);
options = [initial_options, " -p ", num2str(eps), " -c ", num2str(C)];
learning_curves (ytr, Xtr, ycv, Xcv, options, cv_frac);
