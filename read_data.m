## -*- texinfo -*- 
## @deftypefn {Function File} {@var{sample} =} read_data (@var{filename})
##
## Read data from the input CSV file named @var{filename} and return its
## content in @var{sample}.
##
## @seealso{read_from_directory}
## @end deftypefn

function sample = read_data (filename)

if (! ischar (filename))
  error ("read_data: FILENAME should be a string");
endif

sample = csvread (filename, 1, 0);

endfunction
