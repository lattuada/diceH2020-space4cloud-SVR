## -*- texinfo -*- 
## @deftypefn {Function File} {@var{h} =} plot_learning_curves (@var{m}, @var{MSE_train}, @var{MSE_cv})
##
## Plot learning curves given sample sizes @var{m} and mean squared errors,
## both on the training set @var{MSE_train} and on the cross validation
## set @var{MSE_cv}.
## Return the handle @var{h} to the plot.
##
## @seealso {learning_curves}
## @end deftypefn

function h = plot_learning_curves (m, MSE_train, MSE_cv)

h = figure;
plot (m, MSE_train, "b-", "linewidth", 2);
hold on;
plot (m, MSE_cv, "r-", "linewidth", 2);
legend ("Training set", "Cross validation set");
xlabel ('m');
ylabel ('MSE');
title ('Learning curve at varying training set size');
grid on;
hold off;

endfunction
