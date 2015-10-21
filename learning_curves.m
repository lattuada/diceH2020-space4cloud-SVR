## -*- texinfo -*- 
## @deftypefn {Function File} {@var{h} =} learning_curves (@var{ytrain}, @var{Xtrain}, @var{ytest}, @var{Xtest}, @var{ycv}, @var{Xcv}, @var{options})
##
## Train an SVR model specified by @var{options} on the training set
## @var{ytrain}, @var{Xtrain} performing model selection on the
## test set @var{ytest}, @var{Xtest}.
## Then, plot the learning curves at varying training set size considering the
## cross validation set @var{ycv}, @var{Xcv}.
## Return the handle @var{h} to the plot.
##
## @end deftypefn

function h = learning_curves (ytrain, Xtrain, ytest, Xtest, ycv, Xcv, options)

raw_options = options;
alphas = 0.05:0.05:1;
MSE_train = zeros (size (alphas));
MSE_cv = zeros (size (alphas));
m = length (ytrain);

for (ii = 1:length (alphas))
  alpha = alphas(ii);
  ytr = ytrain(1:alpha*m);
  Xtr = Xtrain(1:alpha*m);
  ytst = ytest(1:alpha*m);
  Xtst = Xtest(1:alpha*m);
  
  C = Inf;
  eps = Inf;
  MSE = Inf;
  for (cc = logspace (-5, 5))
    for (ee = logspace (-5, 5))
      options = [raw_options, " -p ", num2str(ee), " -c ", num2str(cc)];
      model = svmtrain (ytr, Xtr, options);
      [~, accuracy, ~] = svmpredict (ytst, Xtst, model, "-q");
      mse = accuracy(2);
      if (mse < MSE)
        C = cc;
        eps = ee;
        MSE = mse;
      endif
    endfor
  endfor
  
  options = [raw_options, " -p ", num2str(eps), " -c ", num2str(C)];
  model = svmtrain (ytr, Xtr, options);
  [~, accuracy, ~] = svmpredict (ytrain, Xtrain, model, "-q");
  MSE_train(ii) = accuracy(2);
  [~, accuracy, ~] = svmpredict (ycv, Xcv, model, "-q");
  MSE_cv(ii) = accuracy(2);
endfor

h = figure;
plot (alphas, MSE_train, "b-", "linewidth", 2);
hold on;
plot (alphas, MSE_cv, "r-", "linewidth", 2);
legend ("Training set", "Cross validation set");
xlabel ('\alpha');
ylabel ('MSE');
title ('Learning curve at varying training set size');
grid on;
hold off;

endfunction
