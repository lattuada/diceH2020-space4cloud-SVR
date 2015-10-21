## -*- texinfo -*- 
## @deftypefn {Function File} {[@var{y}, @var{X}] =} read_data (@var{filename})
##
## Read data from the input CSV file named @var{filename} and return its
## first column as @var{y} and the remaining ones as matrix @var{X}.
##
## @seealso{read_from_directory}
## @end deftypefn

function [y, X] = read_data (filename)

if (! ischar (filename))
  error ("read_data: FILENAME should be a string");
endif

input = csvread (filename, 1, 0);
y = input(:, 1);
X = input(:, 2:end);

endfunction
