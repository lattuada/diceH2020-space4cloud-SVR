## Copyright 2017 Eugenio Gianniti
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

## -*- texinfo -*-
## @deftypefn {Function File} {@var{errors} =} relative_error (@var{measures}, @var{results})
##
## Compute the relative error of @var{results} with respect to the real @var{measures}.
## The arguments should have a common size or be scalar.
##
## @end deftypefn

function errors = relative_error (measures, results)

errors = [];

[err, measures, results] = common_size (measures, results);
if (err == 0)
  errors = abs ((measures - results) ./ measures);
else
  error ("relative_error: MEASURES and RESULTS cannot be brought to a common size");
endif

endfunction
