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
## @deftypefn {Function File} {@var{R} =} compute_waves_prediction (@var{cores}, @var{tasks}, @var{demand})
##
## Compute the average response time @var{R} given the number of @var{cores},
## @var{tasks}, and the @var{demand} per task.
##
## @end deftypefn

function R = compute_waves_prediction (cores, tasks, demand)

R = mean (sum (ceil (bsxfun (@rdivide, tasks, cores)) .* demand));

endfunction
