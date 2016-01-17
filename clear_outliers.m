## -*- texinfo -*- 
## @deftypefn {Function File} {@var{clean} =} clear_outliers (@var{dirty})
##
## Clear outliers from @var{dirty} by excluding rows where the value on a
## column is more than 3 standard deviations away from the mean.
##
## @end deftypefn

function clean = clear_outliers (dirty)

avg = mean (dirty);
dev = std (dirty);

cols = size (dirty, 2);
clean = dirty;
for (jj = 1:cols)
  if (dev(jj) > 0)
    idx = (abs (clean(:, jj) - avg(jj)) < 3 * dev(jj));
    clean = clean(idx, :);
  endif
endfor

endfunction
