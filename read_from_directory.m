## -*- texinfo -*- 
## @deftypefn {Function File} {[@var{y}, @var{X}] =} read_data (@var{directory})
##
## Read data from the input CSV files contained in @var{directory} and return
## their first columns as @var{y} and the remaining ones as matrix @var{X}.
##
## @seealso{read_data}
## @end deftypefn

function [y, X] = read_from_directory (directory)

if (! ischar (directory))
  error ("read_from_directory: DIRECTORY should be a string");
endif

files = glob ([directory, "/*.csv"]);

y = X = [];

for ii = 1:numel (files)
  file = files{ii};
  [lasty, lastX] = read_data (file);
  if (isempty (y))
    y = lasty;
    X = lastX;
  else
    y = [y; lasty];
    X = [X; lastX];
  endif
endfor

endfunction
