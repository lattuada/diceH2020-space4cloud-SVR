## -*- texinfo -*- 
## @deftypefn {Function File} {[@var{scaled}, @var{average}, @var{deviation}] =} scale (@var{sample})
##
## Scale the features in @var{sample} and return the @var{scaled} sample and
## the original @var{average} and standard @var{deviation}.
##
## @seealso{unscale, zscore}
## @end deftypefn

function [scaled, average, deviation] = scale (sample)

[scaled, average, deviation] = zscore (sample);

endfunction
