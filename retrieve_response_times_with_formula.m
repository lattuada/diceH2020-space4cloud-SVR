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

configurations = [6];
task_idx = [6 13 20 27];
base_path = "/home/osboxes/Desktop/test/%d.csv";

operational_data = cell (size (configurations));
for (idx = 1:numel (configurations))
  filename = sprintf (base_path, configurations(idx));
  operational_data{idx} = read_data (filename);
endfor

clean_data = cellfun (@(A) nthargout (1, @clear_outliers, A), operational_data,
                      "UniformOutput", false);

tasks = cellfun (@(A) A(:, task_idx), clean_data, "UniformOutput", false);
times = cellfun (@(A) A(:, task_idx + 2), clean_data, "UniformOutput", false);
containers = cellfun (@(A) A(:, end), clean_data, "UniformOutput", false);

predictions = zeros (size (containers));
for (idx = 1:numel (predictions))
  predictions(idx) = compute_waves_prediction (containers{idx},
                                               tasks{idx},
                                               times{idx});
endfor

plot (configurations, predictions, '-');

results = [configurations(:), predictions(:)];
filename = "/home/osboxes/Desktop/test/estimated_response_times.csv";
csvwrite (filename, results);
