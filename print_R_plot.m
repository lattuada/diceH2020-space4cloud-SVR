clear all
close all hidden
clc

query = "R5";
directory = ["/home/gianniti/policloud-runs/", query];

data = read_from_directory (directory);

values = data(:, 1);
sample = data(:, 2:end);

plot_R (values, sample);

figure_name = [directory, "/", query, "_3D.eps"];
print ("-depsc2", figure_name);
