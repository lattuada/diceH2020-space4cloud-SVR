## -*- texinfo -*- 
## @deftypefn {Function File} {@var{h} =} learning_curves (@var{ytrain}, @var{Xtrain}, @var{ycv}, @var{Xcv}, @var{options})
##
## Train an SVR model specified by @var{options} on the training set
## @var{ytrain}, @var{Xtrain} at varying dataset size
## and plot the learning curves considering the
## cross validation set @var{ycv}, @var{Xcv}.
## Return the handle @var{h} to the plot.
##
## @end deftypefn

function h = learning_curves (ytrain, Xtrain, ycv, Xcv, options)

m_train = length (ytrain);
m_cv = length (ycv);

m = round (linspace (m_cv, m_train, 20));
MSE_train = zeros (size (m));
MSE_cv = zeros (size (m));
steps = zeros (size (m));

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
