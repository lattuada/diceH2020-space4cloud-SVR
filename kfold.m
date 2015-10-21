## -*- texinfo -*- 
## @deftypefn {Function File} {@var{avgR2} =} kfold (@var{y}, @var{X}, @var{C}, @var{eps}, @var{k})
##
## Validate parameters @var{C} and @var{eps} with k-fold using @var{k} splits.
## @var{y} and @var{X} are, respectively, the actual values and the features.
## Return @var{avgR2}, geometric mean of the obtained R^2.
##
## @end deftypefn

function avgR2 = kfold (y, X, C, eps, k)

m = size (X, 1);
[my, cy] = size (y);
if (cy != 1)
  error ("kfold: Y should be a column vector");
endif
if (m != my)
  error ("kfold: Y and X don't have the same sample size");
endif
if (k > m)
  error (["kfold: can't split m = ", num2str(m), " examples in K = ", num2str(k), " splits"]);
endif

m_split = round (m / k);
row_perm = randperm (m);

splits = {};
for (first = 1:m_split:m)
  last = first + m_split - 1;
  split = first:last;
  split = split(split <= m);
  splits{end+1} = split;
endfor

R2 = zeros (k, 1);
for (ii = 1:k)
  idx = 1:k;
  idx = idx(idx != ii);
  row_train = row_perm([splits{idx}]);
  row_cv = row_perm(splits{ii});
  
  ytr = y(row_train);
  ycv = y(row_cv);
  Xtr = X(row_train, :);
  Xcv = X(row_cv, :);
  
  options = ["-s 3 -t 0 -q -p ", num2str(eps), " -c ", num2str(C)];
  model = svmtrain (ytr, Xtr, options);
  [~, accuracy, ~] = svmpredict (ycv, Xcv, model, "-q");
  R2(ii) = accuracy(3);
endfor

avgR2 = mean (R2, "g");

endfunction
