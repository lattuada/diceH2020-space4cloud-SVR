## Copyright 2016 Eugenio Gianniti
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
## @deftypefn {Function File} {[@var{w}, @var{b}, @var{mu}, @var{sigma}]} parse_framework_results (@var{dir})
##
## Parse the output found in @var{dir} and return the weights @var{w},
## the constant term @var{b}, the means @var{mu} and the standard deviations
## @var{sigma}.
## Beware that @var{mu} and @var{sigma} are one element longer than @var{w}
## because they also store the values associated to the prediction in the
## first position.
##
## @end deftypefn

function [w, b, mu, sigma] = parse_framework_results (dir)

if (! isdir (dir))
  error ("parse_framework_results: DIR is not a directory");
else
  modelfile = [dir, "/models.mat"];
  if (exist (modelfile, "file") != 2)
    error ("parse_framework_results: DIR does not contain 'models.mat'");
  else
    load (modelfile);
    b = b{1};
    w = SVs{1}' * coefficients{1};
    mu = mu(:);
    sigma = sigma(:);
  endif
endif

endfunction
