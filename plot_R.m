## -*- texinfo -*- 
## @deftypefn {Function File} {@var{h} =} plot_R (@var{y}, @var{X})
##
## Plot response time, @var{y}, against number of cores and dataset size found
## in @var{X}.
## Return a handle @var{h} to the plot.
##
## @end deftypefn

function [h] = plot_R (y, X)

nCores = X(:, end);
dataSize = X(:, end-1);
h = figure;
plot3 (nCores, dataSize, y, "bx");
xlabel ('Number of cores');
ylabel ('Dataset size');
zlabel ('Response time');
title ('Job response time against cores and dataset size');

endfunction
