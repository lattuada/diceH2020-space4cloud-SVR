## -*- texinfo -*- 
## @deftypefn {Function File} {@var{result} =} unscale (@var{scaled}, @var{average}, @var{deviation})
##
## Return the unscaled features in @var{scaled} given the
## original @var{average} and standard @var{deviation}.
##
## @seealso{scale}
## @end deftypefn

function result = unscale (features, average, deviation)

result = bsxfun (@times, features, deviation);
result = bsxfun (@plus, result, average);

endfunction
