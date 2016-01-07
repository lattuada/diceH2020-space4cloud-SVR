clear all
close all hidden
clc

query = "R5";
directory = ["/home/gianniti/policloud-runs/", query];

[values, sample] = read_from_directory (directory);

plot_R (values, sample);

figure_name = [query, "_3D.eps"];
print ("-depsc2", figure_name);
