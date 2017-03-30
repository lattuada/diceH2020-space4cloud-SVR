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

base_directory = "/Users/eugenio/Desktop/TPCDS500-D_processed_logs/ml/Q52/full";

configuration.runs = [12 16 20 24 28 32 36 40 44 48 52];
configuration.missing_runs = [];

configuration.train_fraction = 0.6;
configuration.test_fraction = 0.2;

configuration.options = "-s 3 -t 0 -q -h 0 ";
configuration.C_range = linspace (1e-4, 1, 20);
configuration.epsilon_range = linspace (1e-10, 1e-2, 20);

%% End of configuration

experimental_data = cell (size (configuration.runs));
for (ii = 1:numel (configuration.runs))
  experimental_data{ii} = ...
    read_data (sprintf ("%s/%d.csv", base_directory, configuration.runs(ii)));
endfor

clean_experimental_data = cellfun (@(A) nthargout (1, @clear_outliers, A),
                                   experimental_data, "UniformOutput", false);

avg_times = cellfun (@(A) mean (A(:, 1)), clean_experimental_data);

sample = vertcat (clean_experimental_data{:});
sample(:, end) = 1 ./ sample(:, end);

idx = randperm (rows (sample));
shuffled = sample(idx, :);

[~, mu, sigma] = zscore (shuffled);

constant_columns = find (sigma == 0);
cols = 1:columns (shuffled);
useful_columns = setdiff (cols, constant_columns);
working_sample = shuffled(:, useful_columns);
working_mu = mu(useful_columns);
working_sigma = sigma(useful_columns);

weights = ones (rows (working_sample), 1);

results = model_selection_with_thresholds (working_sample, weights, avg_times,
                                           configuration);

model = results.model;
b = - model.rho;
w = model.SVs' * model.sv_coef;
useful_columns = useful_columns(:);
working_mu = working_mu(:);
working_sigma = working_sigma(:);

C = results.C;
epsilon = results.epsilon;
train_error = results.train_error;
test_error = results.test_error;
cv_error = results.cv_error;

one_table = sprintf ("%s/%d.csv", base_directory, configuration.runs(1));
fid = fopen (one_table, "r");
% Twice to discard the first line with only the application name
line = fgetl (fid);
line = strtrim (fgetl (fid));
fclose (fid);

headers = strsplit (line, ",");
% +1 to discard the applicationId, 2:end to avoid the predicted time
useful_headers = { headers{useful_columns(2:end) + 1} }';

outfilename = [base_directory, "/model.txt"];
save (outfilename, "b", "w", "useful_headers", "useful_columns", "working_mu",
      "working_sigma", "C", "epsilon", "train_error", "test_error", "cv_error");
