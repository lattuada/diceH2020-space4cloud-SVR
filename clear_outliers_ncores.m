## Copyright 2016 Pietro Ferretti, Andrea Battistello, Eugenio Gianniti
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

function [clean, indices] = clear_outliers_ncores (dirty)

CORES_TO_SEARCH = [20,40,60,72,80,90,100,120];

cols = size (dirty, 2);

clean = dirty;
indices = 1:size (dirty, 1);

for (l = 1:length (CORES_TO_SEARCH))
  idx_cores = (clean(:, end) == CORES_TO_SEARCH(l));
  idx_other = (clean(:, end) != CORES_TO_SEARCH(l));

  if (sum (idx_cores) == 0)
    continue;
  endif

  good_idx = idx_cores;
  for (jj = 1:cols)
    avg = mean (clean(idx_cores, jj));
    dev = std (clean(idx_cores, jj));
    if (dev > 0)
      good_idx &= (abs (clean(:, jj) - avg) < 2 * dev);
    endif
  endfor
  good_idx |= idx_other;

  clean = clean(good_idx, :);
  indices = indices(good_idx);
endfor

endfunction
