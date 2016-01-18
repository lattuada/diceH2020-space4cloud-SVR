## -*- texinfo -*- 
## @deftypefn {Function File} {@var{h} =} plot_MSE (@var{ytrain}, @var{Xtrain}, @var{ytest}, @var{Xtest}, @var{options}, @var{C}, @var{epsilon})
##
## Train an SVR model specified by @var{options} on the training set
## @var{ytrain}, @var{Xtrain} and plot the mean squared error obtained on the
## test set @var{ytest}, @var{Xtest}.
## All the combinations of values in @var{C} and @var{epsilon} are considered.
## Return the handle @var{h} to the plot.
##
## @end deftypefn

function h = plot_MSE (ytrain, Xtrain, ytest, Xtest, options, C, epsilon)

[cc, ee] = meshgrid (C, epsilon);
MSE = zeros (size (cc));
raw_options = options;

for (ii = 1:length (cc))
  options = [raw_options, " -q -c ", num2str(cc(ii)), " -p ", num2str(ee(ii))];
  model = svmtrain (ytrain, Xtrain, options);
  [~, accuracy, ~] = svmpredict (ytest, Xtest, model, "-q");
  MSE(ii) = accuracy(2);
endfor

h = surf (cc, ee, MSE);
xlabel ('C');
ylabel ('\epsilon');
zlabel ('MSE');
title ('MSE at varying model parameters');

endfunction
