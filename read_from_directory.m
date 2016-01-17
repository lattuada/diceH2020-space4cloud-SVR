## -*- texinfo -*- 
## @deftypefn {Function File} {@var{sample} =} read_data (@var{directory})
##
## Read data from the input CSV files contained in @var{directory} and return
## their content in @var{sample}.
##
## @seealso{read_data}
## @end deftypefn

function sample = read_from_directory (directory)

if (! ischar (directory))
  error ("read_from_directory: DIRECTORY should be a string");
endif

files = glob ([directory, "/*.csv"]);

sample = [];

for ii = 1:numel (files)
  file = files{ii};
  last_sample = read_data (file);
  sample = [sample; last_sample];
endfor

endfunction
