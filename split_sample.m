## -*- texinfo -*- 
## @deftypefn {Function File} {[@var{ytrain}, @var{Xtrain}, @var{ytest}, @var{Xtest}, @var{ycv}, @var{Xcv}] =} split_sample (@var{y}, @var{X}, @var{train}, @var{test})
##
## Split sample @var{X} and values @var{y} so that a fraction @var{train}, @var{test}
## of the examples are, respectively, either in the training set @var{ytrain}, @var{Xtrain},
## or in the test set @var{ytest}, @var{Xtest}.
## The remaining examples are in the cross validation set @var{ycv}, @var{Xcv}.
##
## @end deftypefn

function [ytrain, Xtrain, ytest, Xtest, ycv, Xcv] = split_sample (y, X, train, test)

if (train + test > 1)
  error ("split_sample: wrong fractions");
endif

m = size (X, 1);
[my, cy] = size (y);
if (cy != 1)
  error ("split_sample: Y should be a column vector");
endif
if (m != my)
  error ("split_sample: Y and X don't have the same sample size");
endif

idx = randperm (m);
m_train = round (m * train);
m_test = round (m * (test + train));
idx_train = idx(1:m_train);
idx_test = idx(m_train+1:m_test);
idx_cv = idx(m_test+1:end);

Xtrain = X(idx_train, :);
Xtest = X(idx_test, :);
Xcv = X(idx_cv, :);
ytrain = y(idx_train);
ytest = y(idx_test);
ycv = y(idx_cv);

endfunction
