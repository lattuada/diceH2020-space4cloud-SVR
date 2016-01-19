## -*- texinfo -*- 
## @deftypefn {Function File} {[@var{m}, @var{MSE_train}, @var{MSE_cv}] =} learning_curves (@var{ytrain}, @var{Xtrain}, @var{ycv}, @var{Xcv}, @var{options})
##
## Train an SVR model specified by @var{options} on the training set
## @var{ytrain}, @var{Xtrain} at varying dataset size.
## Return the arrays of sample sizes @var{m} and mean squared errors,
## both on the training set @var{MSE_train} and on the cross validation
## set @var{MSE_cv}.
##
## @seealso {plot_learning_curves}
## @end deftypefn

function [m, MSE_train, MSE_cv] = learning_curves (ytrain, Xtrain, ycv, Xcv, options)

m_train = length (ytrain);
m_cv = length (ycv);

m = round (linspace (m_cv, m_train, 20));
MSE_train = zeros (size (m));
MSE_cv = zeros (size (m));

for (ii = 1:length (m))
  m_part = m(ii);
  ytr = ytrain(1:m_part);
  Xtr = Xtrain(1:m_part, :);
  model = svmtrain (ytr, Xtr, options);
  [~, accuracy, ~] = svmpredict (ytr, Xtr, model, "-q");
  MSE_train(ii) = accuracy(2);
  [~, accuracy, ~] = svmpredict (ycv, Xcv, model, "-q");
  MSE_cv(ii) = accuracy(2);
endfor

endfunction
