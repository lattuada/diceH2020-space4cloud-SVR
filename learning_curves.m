## -*- texinfo -*- 
## @deftypefn {Function File} {@var{h} =} learning_curves (@var{ytrain}, @var{Xtrain}, @var{ycv}, @var{Xcv}, @var{options}, @var{alpha0})
##
## Train an SVR model specified by @var{options} on the training set
## @var{ytrain}, @var{Xtrain} at varying dataset size, starting from
## a fraction @var{alpha0}, and plot the learning curves considering the
## cross validation set @var{ycv}, @var{Xcv}.
## Return the handle @var{h} to the plot.
##
## @end deftypefn

function h = learning_curves (ytrain, Xtrain, ycv, Xcv, options, alpha0)

if (alpha0 > 1 || alpha0 < 0)
  error ("learning_curves: ALPHA0 should be between 0 and 1");
endif

alphas = alpha0:0.01:1;
RMSE_train = zeros (size (alphas));
RMSE_cv = zeros (size (alphas));
m = length (ytrain);

for (ii = 1:length (alphas))
  alpha = alphas(ii);
  m_part = round (alpha * m);
  ytr = ytrain(1:m_part);
  Xtr = Xtrain(1:m_part, :);
  model = svmtrain (ytr, Xtr, options);
  [~, accuracy, ~] = svmpredict (ytr, Xtr, model, "-q");
  RMSE_train(ii) = sqrt (accuracy(2));
  [~, accuracy, ~] = svmpredict (ycv, Xcv, model, "-q");
  RMSE_cv(ii) = sqrt (accuracy(2));
endfor

h = figure;
plot (alphas, RMSE_train, "b-", "linewidth", 2);
hold on;
plot (alphas, RMSE_cv, "r-", "linewidth", 2);
legend ("Training set", "Cross validation set");
xlabel ('\alpha');
ylabel ('RMSE');
title ('Learning curve at varying training set size');
grid on;
hold off;

endfunction
