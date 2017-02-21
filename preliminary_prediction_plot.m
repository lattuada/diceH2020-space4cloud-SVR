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

clear all
close all hidden
clc

# This assumes that the files are named <cores>.csv
configurations = [6 8];

# You can retrieve the indices with:
#   head -n 1 <cores>.csv | tr ',' '\n' | grep -n 'nTask' | cut -d : -f 1 | xargs echo
task_idx = [5 12 19 26];
avg_idx = task_idx + 2;

data = cell (size (configurations));
for (ii = 1:numel (configurations))
  filename = sprintf ("%d.csv", configurations(ii));
  data{ii} = csvread (filename, 1, 0);
endfor

clean = cell (size (data));
for (ii = 1:numel (data))
  [clean{ii}, ~] = clear_outliers (data{ii});
endfor

response = cellfun (@(A) mean (A(:, 1)), clean);

tasks = cellfun (@(A) mean (A(:, task_idx)), clean, "UniformOutput", false);
times = cellfun (@(A) mean (A(:, avg_idx)), clean, "UniformOutput", false);

prediction = zeros (size (configurations));
for (ii = 1:numel (configurations))
  prediction(ii) = sum (ceil (tasks{ii} / configurations(ii)) .* times{ii});
endfor

figure;
plot (configurations, prediction);
hold all;
plot (configurations, response);
xlabel Cores;
ylabel ("Time [ms]");
grid on;
legend Predictions Measurements;
