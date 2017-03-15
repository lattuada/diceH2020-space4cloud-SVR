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
## @deftypefn {Function File} {[@var{available}, @var{missing}] =} find_configurations (@var{runs}, @var{missing_runs})
##
## Find the @var{available} and @var{missing} indices among @var{runs},
## considering the @var{missing_runs}.
##
## @end deftypefn

function [available, missing] = find_configurations (runs, missing_runs)

available = missing = [];

for (ii = 1:numel(runs))
  if (any (missing_runs == runs(ii)))
    missing(end + 1) = ii;
  else
    available(end + 1) = ii;
  endif
endfor

endfunction
