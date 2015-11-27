## -*- texinfo -*- 
## @deftypefn {Function File} {@var{h} =} plot_RMSE (@var{ytrain}, @var{Xtrain}, @var{ytest}, @var{Xtest}, @var{options}, @var{C}, @var{epsilon})
##
## Train an SVR model specified by @var{options} on the training set
## @var{ytrain}, @var{Xtrain} and plot the root mean squared error obtained on the
## test set @var{ytest}, @var{Xtest}.
## All the combinations of values in @var{C} and @var{epsilon} are considered.
## Return the handle @var{h} to the plot.
##
## @end deftypefn

function h = plot_RMSE (ytrain, Xtrain, ytest, Xtest, options, C, epsilon)

[cc, ee] = meshgrid (C, epsilon);
RMSE = zeros (size (cc));
raw_options = options;

for (ii = 1:length (cc))
  options = [raw_options, " -c ", num2str(cc(ii)), " -p ", num2str(ee(ii))];
  model = svmtrain (ytrain, Xtrain, options);
  [~, accuracy, ~] = svmpredict (ytest, Xtest, model, "-q");
  RMSE(ii) = sqrt (accuracy(2));
endfor

rel_RMSE = RMSE / mean ([ytrain; ytest]);

h = surf (cc, ee, rel_RMSE);
xlabel ('C');
ylabel ('\epsilon');
zlabel ('Relative RMSE');
title ('Relative RMSE at varying model parameters');

endfunction
