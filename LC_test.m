clear all
close all hidden
clc

[values, sample] = read_from_directory ("/home/eugenio/Desktop/csv");
[X, ~, ~] = scale (sample);
[ytr, Xtr, ytst, Xtst, ycv, Xcv] = split_sample (values, X, 0.6, 0.2);

learning_curves (ytr, Xtr, ytst, Xtst, ycv, Xcv, "-s 3 -t 0 -q");
