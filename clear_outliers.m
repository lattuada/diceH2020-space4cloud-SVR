## -*- texinfo -*- 
## @deftypefn {Function File} {[@var{clean}, @var{indices}] =} clear_outliers (@var{dirty})
##
## Clear outliers from @var{dirty} by excluding rows where the value on a
## column is more than 3 standard deviations away from the mean.
## Return the @var{clean} dataset and the original @var{indices}
## kept after the procedure.
##
## @end deftypefn

function [clean, indices] = clear_outliers (dirty)

avg = mean (dirty);
dev = std (dirty);

cols = size (dirty, 2);
clean = dirty;
indices = 1:size (dirty, 1)';
for (jj = 1:cols)
  if (dev(jj) > 0)
    idx = (abs (clean(:, jj) - avg(jj)) < 3 * dev(jj));
    clean = clean(idx, :);
    indices = indices(idx);
  endif
endfor

endfunction
