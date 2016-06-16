%% Copyright 2016 Pietro Ferretti & Andrea Battistello
%%
%% Licensed under the Apache License, Version 2.0 (the "License");
%% you may not use this file except in compliance with the License.
%% You may obtain a copy of the License at
%%
%%     http://www.apache.org/licenses/LICENSE-2.0
%%
%% Unless required by applicable law or agreed to in writing, software
%% distributed under the License is distributed on an "AS IS" BASIS,
%% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
%% See the License for the specific language governing permissions and
%% limitations under the License.

clear all;
clc;
close all hidden;
warning('off', 'Octave:possible-matlab-short-circuit-operator');

addpath('./utility/');

TEST_ID = 'fixed_datasizes_shuffled';
OUTPUT_FOLDER = 'output/';
LATEX_OUTPUT_FOLDER = 'latex_output/';

SAVE_DATA = true;       % Plots and data
ALL_THE_PLOTS = false;      % Plot every feature
BEST_MODELS = true;       % Make additional plots with only the best 3 models

LEARNING_CURVES = false;

OUTPUT_LATEX = true;      % Save results to a latex file
LATEX_TABLE = true;       % Include tables with training and testing metrics
LATEX_PLOT = false;       % Include regular plots
LATEX_PLOT_BESTMODELS = true; % Include "best models" plots
SECTION_HEADER = 'Fixed datasizes, shuffle, $ncores^{-1}$';   % Header for the latex document

OUTPUT_FORMATS = {  {'-deps', '.eps'},          % generates only one .eps file black and white
          {'-depslatex', '.eps'},       % generates one .eps file containing only the plot and a .tex file that includes the plot and fill the legend with plain text
          {'-depsc', '.eps'},         % generates only one .eps file with colour
          {'-dpdflatex', '.pdf'}        % generates one .pdf file containing only the plot and a .tex file that includes the plot and fill the legend with plain text
          {'-dpdf', '.pdf'}         % generates one complete .pdf file A4
        };
PLOT_SAVE_FORMAT = 3;

BASE_DIR = './dati/Query R/';

QUERIES = {'R1', 'R2', 'R3', 'R4', 'R5'};   % Queries to analyze
% QUERIES = {'R1', 'R2'};
DATASIZES = {'250', '500', '750', '1000'};    % Datasizes to consider
% DATASIZES = {'250', '500'}

%% Choose which SVR models to use
% 1 -> Linear SVR
% 2 -> Polynomial SVR (2 degree)
% 3 -> Polynomial SVR (3 degree)
% 4 -> Polynomial SVR (4 degree)
% 5 -> Polynomial SVR (6 degree)
% 6 -> RBF SVR
MODELS_CHOSEN = [1, 2, 3, 4, 5, 6];
COLORS = {'g', [1, 0.5, 0.2], 'c', 'k', 'm', 'r'};    % Magenta, orange, cyan, black, green, red

LINEAR_REGRESSION = true; % Add linear regression to the models used

%% Choose which features to use
% 1 -> N map
% 2 -> N reduce
% 3 -> Map time avg
% 4 -> Map time max
% 5 -> Reduce time avg
% 6 -> Reduce time max
% 7 -> Shuffle time avg
% 8 -> Shuffle time max
% 9 -> Bandwidth avg
% 10 -> Bandwidth max
% 11 -> N Users
% 12 -> Datasize
% 13 -> N Cores
CHOOSE_FEATURES = true;
FEATURES = [3:8, 13];   % nMap, nReduce, Bandwidth avg and Bandwidth max are constant when the datasize is fixed

N_CORES_INVERSE = true;   % To use ncores^(-1) instead of ncores

ALL_FEATURES_DESCRIPTIONS = {   % These will be used for the plot labels
  'N map',
  'N reduce',
  'Map time avg',
  'Map time max',
  'Reduce time avg',
  'Reduce time max',
  'Shuffle time avg',
  'Shuffle time max',
  'Bandwidth avg',
  'Bandwidth max',
  'N Users',
  'Datasize',
  'N cores'
};

%% Fractions of data reserved for training and testing (the remaining amount is used for cross-validation)
TRAIN_FRAC_WO_TEST = 0.6;
TEST_FRAC_WO_TEST = 0.2;

%% Not used
TRAIN_FRAC_W_TEST = 0.7;

DIFF_MEANS = false;     % To add the 'difference between means' metric

NORMALIZE_FEATURE = true;       % Normalize all the features
CLEAR_OUTLIERS = true;          % Clear all outliers outside of 2 standard deviations
ENABLE_FEATURE_FILTERING = false;   % Filter every datapoint with completion time over the threshold
COMPLETION_TIME_THRESHOLD = 32000;

SHUFFLE_DATA = true;    % Used for cross-validation
rand('seed', 18);

% Ranges of the parameters values for model selection
C_range = linspace (0.1, 5, 20);
E_range = linspace (0.1, 5, 20);


%------------------------------------------------------------------------------------------------------------
%------------------------------------------------------------------------------------------------------------


%% Create a latex file with all the results, already formatted
if OUTPUT_LATEX
  % Create  latexoutput folder
  if ~ exist(LATEX_OUTPUT_FOLDER)   %% Checks if the folder exists
    if ~ mkdir(LATEX_OUTPUT_FOLDER)   %% Try with the mkdir function
      if system(cstrcat('mkdir -p ', LATEX_OUTPUT_FOLDER))
        fprintf('[ERROR] Could not create output folder\nCreate the output folder first and then restart this script\n');
        quit;
      end
    end
  end
  flatex = fopen(cstrcat(LATEX_OUTPUT_FOLDER, TEST_ID, '.tex'), 'w');


  fprintf(flatex, cstrcat('\\newpage\n', ...
              '\\section{', SECTION_HEADER, '}\n'));
end

for query_id = 1:length(QUERIES)

  QUERY = QUERIES{query_id};

  if OUTPUT_LATEX
    fprintf(flatex, cstrcat('\\subsection{Query ', QUERY, '}\n'));
  end

  TESTS_TO_DO = DATASIZES;

  while length(TESTS_TO_DO) > 0
    try
      DATASIZE = TESTS_TO_DO{1};

      printf(cstrcat('\n\nStarting test ', QUERY, ' - ', DATASIZE));

      if OUTPUT_LATEX
        fprintf(flatex, cstrcat('\\subsubsection{Query ', QUERY, ' --- Datasize ', DATASIZE, 'GB}\n'));
      end

      close all hidden;

      %% List of all directories with train data
      TRAIN_DATA_LOCATION = {strcat(QUERY, '/Datasize/', DATASIZE)};
      % TRAIN_DATA_LOCATION = {strcat('')};
      % TRAIN_DATA_LOCATION = {'Core/60', 'Core/80', 'Core/100', 'Core/120', 'Core/72'};
      % TRAIN_DATA_LOCATION = {'Query R/R1/Datasize/750'};

      %% List of all directories with test data (leave {} if test data equals train data)
      % TEST_DATA_LOCATION = {'Query R/R1/Core/120'};
      TEST_DATA_LOCATION = {};

      TABLE_CAPTION_TRAIN = cstrcat('Training results for ', QUERY, ' (', DATASIZE, ' GB)');
      TABLE_CAPTION_TEST = cstrcat('Testing results for ', QUERY, ' (', DATASIZE, ' GB)');
      PLOT_CAPTION = cstrcat('Completion time vs ncores for ', QUERY, ' (', DATASIZE, ' GB)');
      TABLE_LABEL_TRAIN = cstrcat('tab1:', TEST_ID, '_', QUERY, '_', DATASIZE);
      TABLE_LABEL_TEST = cstrcat('tab2:', TEST_ID, '_', QUERY, '_', DATASIZE);
      PLOT_LABEL = cstrcat('fig:', TEST_ID, '_', QUERY, '_', DATASIZE);

      % OUTPUT_SUBFOLDER = strcat('output/', QUERY, '_ALL_FEATURES/');
      OUTPUT_SUBFOLDER = strcat(OUTPUT_FOLDER, upper(TEST_ID), '_', QUERY, '_', DATASIZE, '/');
      printf('\nSaving in folder "%s"', OUTPUT_SUBFOLDER);
      fflush(stdout);

      % Create output folder
      if ~ exist(OUTPUT_SUBFOLDER)    %% Checks if the folder exists
        if ~ mkdir(OUTPUT_SUBFOLDER)    %% Try with the mkdir function
          if system(cstrcat('mkdir -p ', OUTPUT_SUBFOLDER))   %% This creates subfolders
            fprintf('[ERROR] Could not create output folder\nCreate the output folder first and then restart this script\n');
            quit;
          end
        end
      end



      %% Retrieve the data
      printf('\nLoading data...');

      train_data = get_all_data_from_dirs(BASE_DIR, TRAIN_DATA_LOCATION);

      if CHOOSE_FEATURES
        tmp = train_data(:, 2:end);
        train_data = [train_data(:, 1) , tmp(:, FEATURES)];
        FEATURES_DESCRIPTIONS = ALL_FEATURES_DESCRIPTIONS(FEATURES);
      end

      test_data = [];
      if not (isempty(TEST_DATA_LOCATION))
        test_data = get_all_data_from_dirs(BASE_DIR, TEST_DATA_LOCATION);
        if CHOOSE_FEATURES
          tmp = test_data(:, 2:end);
          test_data = [test_data(:, 1) , tmp(:, FEATURES)];
        end
      end


      if ENABLE_FEATURE_FILTERING
        rows_ok = train_data(:, 1) < COMPLETION_TIME_THRESHOLD;
        train_data = train_data(rows_ok, :);

        if not (isempty(TEST_DATA_LOCATION))
          rows_ok = test_data(:, 1) < COMPLETION_TIME_THRESHOLD;
          test_data = test_data(rows_ok, :);
        end
      end



      M = size(train_data, 2) - 1;      %% Number of features
      N_train = size(train_data, 1);    %% Number of train tuples
      N_test = size(test_data, 1);    %% Number of test tuples

      complete_data = [train_data ; test_data];


      if CLEAR_OUTLIERS
        % printf('\nClearing outliers...');

        % [clean, indices] = clear_outliers(complete_data);

        [clean,indices] = clear_outliers_ncores(complete_data);

        train_data = clean(indices <= N_train, :);
        test_data = clean(indices > N_train, :);

        N_train = size(train_data, 1);    %% Number of train tuples
        N_test = size(test_data, 1);    %% Number of test tuples

        complete_data = [train_data ; test_data];
      end


      if N_CORES_INVERSE

        complete_data(:, end) = 1./complete_data(:, end);  %% replace nCores with 1/nCores

      end


      mu = zeros(M+1, 1);
      sigma = ones(M+1, 1);

      if NORMALIZE_FEATURE
        % printf('\nNormalizing features...');

        [scaled, mu, sigma] = zscore(complete_data);

        train_data = scaled(1:N_train, :);
        test_data = scaled(N_train+1:end, :);

        if SAVE_DATA
          % Save data for - maybe - later uses
          save(strcat(OUTPUT_SUBFOLDER, 'mu_sigma.mat'), 'mu', 'sigma');
        end
      end


      if SHUFFLE_DATA
        % printf('\nShuffling data...');

        r = randperm(N_train);
        train_data = train_data(r, :);

        %% There is no need to shuffle test data
      end


      %% SPLIT THE DATA
      % printf('\nSplitting the sample...');

      cv_data = [];
      N_cv = 0;

      if isempty(TEST_DATA_LOCATION)
        [train_data, test_data, cv_data] = split_sample(train_data, TRAIN_FRAC_WO_TEST, TEST_FRAC_WO_TEST);
        N_train = size(train_data, 1);
        N_cv = size(cv_data, 1);
        N_test = size(test_data, 1);
      else
        [train_data, cv_data, ~] = split_sample(train_data, TRAIN_FRAC_W_TEST, 1-TRAIN_FRAC_W_TEST);
        N_train = size(train_data, 1);
        N_cv = size(cv_data, 1);
      end


      %% Organize data

      y_tr = train_data(:, 1);
      X_tr = train_data(:, 2:end);

      y_cv = cv_data(:, 1);
      X_cv = cv_data(:, 2:end);

      y_test = test_data(:, 1);
      X_test = test_data(:, 2:end);


      mu_y = mu(1);
      mu_X = mu(2:end);

      sigma_y = sigma(1);
      sigma_X = sigma(2:end);


      %% DECLARE USEFUL VARIABLES

      Cs = [];
      Es = [];
      predictions_tr = [];
      predictions = [];
      coefficients = {};
      SVs = {};
      b = {};
      SVR_DESCRIPTIONS = {};
      models = {};
      means = [];

      RMSEs_tr = [];
      R_2_tr = [];

      % Saving test metrics
      RMSEs = [];
      R_2 = [];
      MAE = []; % Mean absolute error
      MRE = []; % Mean relative error
      DM = [];  % Difference between means

      %% SVR

      % svmtrain parameters
      % -s --> SVM type (3 = epsilon-SVR)
      % -t --> kernel tyle (0 = linear, 1 = polynomial, 2 = gaussian, 3 = sigmoid)
      % -q --> No output
      % -h --> (0 = No shrink)
      % -p --> epsilon
      % -c --> cost

      printf('\nTraining models...');

      %% Linear SVR
      if ismember(1, MODELS_CHOSEN)
        %fprintf('\nTraining model with linear SVR');

        SVR_DESCRIPTIONS{end + 1} = 'Linear SVR';

        [C, eps] = model_selection (y_tr, X_tr, y_cv, X_cv, '-s 3 -t 0 -q -h 0', C_range, E_range);
        options = ['-s 3 -t 0 -h 0 -q -p ', num2str(eps), ' -c ', num2str(C)];
        model = svmtrain (y_tr, X_tr, options);

        [predictions_tr(:, end + 1), accuracy_tr, ~] = svmpredict(y_tr, X_tr, model, '-q');
        [predictions(:, end + 1), accuracy, ~] = svmpredict (y_test, X_test, model, '-q');  %% quiet

        if LEARNING_CURVES
          [m, mse_train, mse_test] = learning_curves(y_tr, X_tr, y_test, X_test, [options, ' -q']);
          h = plot_learning_curves(m, mse_train, mse_test);
          print('-depsc', cstrcat(OUTPUT_SUBFOLDER, 'learning_curve_', SVR_DESCRIPTIONS{end}, '.eps'));
          close(h);
        end

        models{end + 1} = model;
        Cs(end + 1) = C;
        Es(end + 1) = eps;
        RMSEs(end + 1) = sqrt (accuracy(2));
        coefficients{end + 1} = model.sv_coef;
        SVs{end + 1} = model.SVs;
        b{end + 1} = - model.rho;
        R_2(end + 1) = accuracy(3);
        RMSEs_tr(end + 1) = accuracy_tr(2);
        R_2_tr(end + 1) = accuracy_tr(3);
      end


      %% Polynomial SVR
      if ismember(2, MODELS_CHOSEN)
        % fprintf('\nTraining model with polynomial(2) SVR');
        %fflush(stdout);
        SVR_DESCRIPTIONS{end + 1} = 'Polynomial SVR (2)';

        [C, eps] = model_selection (y_tr, X_tr, y_cv, X_cv, '-s 3 -d 2 -t 1 -q -h 0', C_range, E_range);
        options = ['-s 3 -d 2 -t 1 -h 0 -q -p ', num2str(eps), ' -c ', num2str(C)];
        model = svmtrain (y_tr, X_tr, options);

        [predictions_tr(:, end + 1), accuracy_tr, ~] = svmpredict(y_tr, X_tr, model, '-q');
        [predictions(:, end + 1), accuracy, ~] = svmpredict (y_test, X_test, model, '-q');  %% quiet

        if LEARNING_CURVES
          [m, mse_train, mse_test] = learning_curves(y_tr, X_tr, y_test, X_test, [options, ' -q']);
          h = plot_learning_curves(m, mse_train, mse_test);
          print('-depsc', cstrcat(OUTPUT_SUBFOLDER, 'learning_curve_', SVR_DESCRIPTIONS{end}, '.eps'));
          close(h);
        end

        models{end + 1} = model;
        Cs(end + 1) = C;
        Es(end + 1) = eps;
        RMSEs(end + 1) = sqrt (accuracy(2));
        coefficients{end + 1} = model.sv_coef;
        SVs{end + 1} = model.SVs;
        b{end + 1} = - model.rho;
        R_2(end + 1) = accuracy(3);
        RMSEs_tr(end + 1) = accuracy_tr(2);
        R_2_tr(end + 1) = accuracy_tr(3);
      end

      %% Polynomial SVR
      if ismember(3, MODELS_CHOSEN)
        % fprintf('\nTraining model with polynomial(3) SVR');
        %fflush(stdout);
        SVR_DESCRIPTIONS{end + 1} = 'Polynomial SVR (3)';

        [C, eps] = model_selection (y_tr, X_tr, y_cv, X_cv, '-s 3 -d 3 -t 1 -q -h 0', C_range, E_range);
        options = ['-s 3 -d 3 -t 1 -h 0 -q -p ', num2str(eps), ' -c ', num2str(C)];
        model = svmtrain (y_tr, X_tr, options);

        [predictions_tr(:, end + 1), accuracy_tr, ~] = svmpredict(y_tr, X_tr, model, '-q');
        [predictions(:, end + 1), accuracy, ~] = svmpredict (y_test, X_test, model, '-q');  %% quiet

        if LEARNING_CURVES
          [m, mse_train, mse_test] = learning_curves(y_tr, X_tr, y_test, X_test, [options, ' -q']);
          h = plot_learning_curves(m, mse_train, mse_test);
          print('-depsc', cstrcat(OUTPUT_SUBFOLDER, 'learning_curve_', SVR_DESCRIPTIONS{end}, '.eps'));
          close(h);
        end

        models{end + 1} = model;
        Cs(end + 1) = C;
        Es(end + 1) = eps;
        RMSEs(end + 1) = sqrt (accuracy(2));
        coefficients{end + 1} = model.sv_coef;
        SVs{end + 1} = model.SVs;
        b{end + 1} = - model.rho;
        R_2(end + 1) = accuracy(3);
        RMSEs_tr(end + 1) = accuracy_tr(2);
        R_2_tr(end + 1) = accuracy_tr(3);
      end

      %% Polynomial SVR
      if ismember(4, MODELS_CHOSEN)
        % fprintf('\nTraining model with polynomial(4) SVR');
        %fflush(stdout);
        SVR_DESCRIPTIONS{end + 1} = 'Polynomial SVR (4)';

        [C, eps] = model_selection (y_tr, X_tr, y_cv, X_cv, '-s 3 -d 4 -t 1 -q -h 0', C_range, E_range);
        options = ['-s 3 -d 4 -t 1 -h 0 -q -p ', num2str(eps), ' -c ', num2str(C)];
        model = svmtrain (y_tr, X_tr, options);

        [predictions_tr(:, end + 1), accuracy_tr, ~] = svmpredict(y_tr, X_tr, model, '-q');
        [predictions(:, end + 1), accuracy, ~] = svmpredict (y_test, X_test, model, '-q');  %% quiet

        if LEARNING_CURVES
          [m, mse_train, mse_test] = learning_curves(y_tr, X_tr, y_test, X_test, [options, ' -q']);
          h = plot_learning_curves(m, mse_train, mse_test);
          print('-depsc', cstrcat(OUTPUT_SUBFOLDER, 'learning_curve_', SVR_DESCRIPTIONS{end}, '.eps'));
          close(h);
        end

        models{end + 1} = model;
        Cs(end + 1) = C;
        Es(end + 1) = eps;
        RMSEs(end + 1) = sqrt (accuracy(2));
        coefficients{end + 1} = model.sv_coef;
        SVs{end + 1} = model.SVs;
        b{end + 1} = - model.rho;
        R_2(end + 1) = accuracy(3);
        RMSEs_tr(end + 1) = accuracy_tr(2);
        R_2_tr(end + 1) = accuracy_tr(3);
      end

      %% Polynomial SVR
      if ismember(5, MODELS_CHOSEN)
        % fprintf('\nTraining model with polynomial(6) SVR');
        %fflush(stdout);
        SVR_DESCRIPTIONS{end + 1} = 'Polynomial SVR (6)';

        [C, eps] = model_selection (y_tr, X_tr, y_cv, X_cv, '-s 3 -d 6 -t 1 -q -h 0', C_range, E_range);
        options = ['-s 3 -d 6 -t 1 -h 0 -q -p ', num2str(eps), ' -c ', num2str(C)];
        model = svmtrain (y_tr, X_tr, options);

        [predictions_tr(:, end + 1), accuracy_tr, ~] = svmpredict(y_tr, X_tr, model, '-q');
        [predictions(:, end + 1), accuracy, ~] = svmpredict (y_test, X_test, model, '-q');  %% quiet

        if LEARNING_CURVES
          [m, mse_train, mse_test] = learning_curves(y_tr, X_tr, y_test, X_test, [options, ' -q']);
          h = plot_learning_curves(m, mse_train, mse_test);
          print('-depsc', cstrcat(OUTPUT_SUBFOLDER, 'learning_curve_', SVR_DESCRIPTIONS{end}, '.eps'));
          close(h);
        end

        models{end + 1} = model;
        Cs(end + 1) = C;
        Es(end + 1) = eps;
        RMSEs(end + 1) = sqrt (accuracy(2));
        coefficients{end + 1} = model.sv_coef;
        SVs{end + 1} = model.SVs;
        b{end + 1} = - model.rho;
        R_2(end + 1) = accuracy(3);
        RMSEs_tr(end + 1) = accuracy_tr(2);
        R_2_tr(end + 1) = accuracy_tr(3);
      end

      %% Gaussian SVR
      if ismember(6, MODELS_CHOSEN)
        % fprintf('\nTraining model with RBF SVR');
        %fflush(stdout);
        SVR_DESCRIPTIONS{end + 1} = 'Gaussian SVR';

        [C, eps] = model_selection (y_tr, X_tr, y_cv, X_cv, '-s 3 -t 2 -q -h 0', C_range, E_range);
        options = ['-s 3 -t 2 -h 0 -q -p ', num2str(eps), ' -c ', num2str(C)];
        model = svmtrain (y_tr, X_tr, options);

        [predictions(:, end + 1), accuracy, ~] = svmpredict (y_test, X_test, model, '-q');  %% quiet

        [predictions_tr(:, end + 1), accuracy_tr, ~] = svmpredict(y_tr, X_tr, model, '-q');
        [predictions(:, end + 1), accuracy, ~] = svmpredict (y_test, X_test, model, '-q');  %% quiet

        if LEARNING_CURVES
          [m, mse_train, mse_test] = learning_curves(y_tr, X_tr, y_test, X_test, [options, ' -q']);
          h = plot_learning_curves(m, mse_train, mse_test);
          print('-depsc', cstrcat(OUTPUT_SUBFOLDER, 'learning_curve_', SVR_DESCRIPTIONS{end}, '.eps'));
          close(h);
        end

        models{end + 1} = model;
        Cs(end + 1) = C;
        Es(end + 1) = eps;
        RMSEs(end + 1) = sqrt (accuracy(2));
        coefficients{end + 1} = model.sv_coef;
        SVs{end + 1} = model.SVs;
        b{end + 1} = - model.rho;
        R_2(end + 1) = accuracy(3);
        RMSEs_tr(end + 1) = accuracy_tr(2);
        R_2_tr(end + 1) = accuracy_tr(3);
      end


      %% Linear Regression
      if LINEAR_REGRESSION
        % fprintf('\nTraining Linear regression.');

        X_tr = [ones(N_train, 1) , X_tr]; %% Add the intercept

        [theta, ~, ~, ~, results] = regress(y_tr, X_tr);

        predictions_tr(:, end+1) = X_tr * theta;
        predictions(:, end+1) = [ones(N_test, 1) X_test] * theta;

        models{end + 1} = {};
        Cs(end + 1) = 0;
        Es(end + 1) = 0;
        % RMSEs(end + 1) = -1;      %% Will be computed later
        coefficients{end + 1} = 0;
        SVs{end + 1} = 0;
        b{end+1} = 0;
        % R_2(end + 1) = -1;        %% Will be computed later

        % Removes the intercept
        X_tr = X_tr(:, 2:end);
      end



      fd = -1;
      if SAVE_DATA

        results_filename = strcat(OUTPUT_SUBFOLDER, 'report.txt');
        fd = fopen(results_filename, 'w');

        %% Prints train and test data location

        fprintf(fd, 'TRAIN DATA:\n');
        for index = 1:length(TRAIN_DATA_LOCATION)
          fprintf(fd, '%s\n', TRAIN_DATA_LOCATION{index});
        end

        fprintf(fd, '\n\nTEST DATA:\n');
        for index = 1:length(TEST_DATA_LOCATION)
          fprintf(fd, '%s\n', TEST_DATA_LOCATION{index});
        end

        fprintf(fd, '\n\n\n');
      end

      if OUTPUT_LATEX
        if ~ exist(OUTPUT_SUBFOLDER)    %% Checks if the folder exists
          if ~ mkdir(OUTPUT_SUBFOLDER)    %% Try with the mkdir function
            if system(cstrcat('mkdir -p ', OUTPUT_SUBFOLDER))   %% This creates subfolders
              fprintf('[ERROR] Could not create output folder\nCreate the output folder first and then restart this script\n');
              quit;
            end
          end
        end
        latex_filename_table = strcat(OUTPUT_SUBFOLDER, 'outputlatex_table.tex');
        flatex_table = fopen(latex_filename_table, 'w');

        latex_filename_plot = strcat(OUTPUT_SUBFOLDER, 'outputlatex_plot.tex');
        flatex_plot = fopen(latex_filename_plot, 'w');

        if BEST_MODELS
          latex_filename_plot_bestmodels = strcat(OUTPUT_SUBFOLDER, 'outputlatex_plot_bestmodels.tex');
          flatex_plot_bestmodels = fopen(latex_filename_plot_bestmodels, 'w');
        end

      end

      %% Compute metrics for all models
      % printf('\nComputing metrics...');


      % Latex training table
      if OUTPUT_LATEX
        fprintf(flatex_table, cstrcat('\\begin{table}[H]\n', ...
        '\\centering\n', ...
        '\\begin{adjustbox}{center}\n'));

        fprintf(flatex_table, cstrcat('\\begin{tabular}{c | c M{1.4cm} M{2.5cm} M{2.3cm}}\n', ...
            'Model & RMSE & R\\textsuperscript{2} & Mean absolute error & Mean relative error \\tabularnewline\n'));

        fprintf(flatex_table, '\\hline\n');
      end

      if LINEAR_REGRESSION
        y_mean = mean(y_tr);

        sum_residual = sum((y_tr - predictions_tr(:, end)).^2);
        sum_total = sum((y_tr - y_mean).^2);

        real_tr_values = mu_y + sigma_y * y_tr;
        real_predictions_tr = mu_y + sigma_y * predictions_tr(:, end);

        abs_err = abs(real_tr_values - real_predictions_tr);
        rel_err = abs_err ./ real_tr_values;

        mean_abs = mean(abs_err);
        mean_rel = mean(rel_err);

        RMSE = sqrt(sum_residual / length(y_tr));   % Root Mean Squared Error
        R2 = 1 - (sum_residual / sum_total);      % R^2

        fprintf(flatex_table, 'Linear regression & %5.4f & %5.4f & %6.0f & %5.4f \\\\\n', RMSE, R2, mean_abs, mean_rel);

      end


      for index = 1:length(MODELS_CHOSEN)
        real_predictions_tr = mu_y + sigma_y * predictions_tr(:, index);
        real_tr_values = mu_y + sigma_y * y_tr;

        abs_err = abs(real_predictions_tr - real_tr_values);
        rel_err = abs_err ./ real_tr_values;

        mean_abs = mean(abs_err);
        mean_rel = mean(rel_err);

        fprintf(flatex_table, '%s & %5.4f & %5.4f & %6.0f & %5.4f \\\\\n', SVR_DESCRIPTIONS{index}, RMSEs_tr(index), R_2_tr(index), mean_abs, mean_rel);
      end

      if OUTPUT_LATEX
        fprintf(flatex_table, cstrcat('\\end{tabular}\n', ...
                      '\\end{adjustbox}\n', ...
                      '\\\\\n', ...
                      '\\caption{', TABLE_CAPTION_TRAIN, '}\n', ...
                      '\\label{', TABLE_LABEL_TRAIN, '}\n', ...
                      '\\end{table}\n'));
      end

      % Latex testing table
      if OUTPUT_LATEX
        fprintf(flatex_table, cstrcat('\\begin{table}[H]\n', ...
        '\\centering\n', ...
        '\\begin{adjustbox}{center}\n'));
        if DIFF_MEANS
          fprintf(flatex_table, cstrcat('\\begin{tabular}{c | c M{1.4cm} M{2.5cm} M{2.3cm} M{1.8cm}}\n', ...
            'Model & RMSE & R\\textsuperscript{2} & Mean absolute error & Mean relative error & Mean difference \\tabularnewline\n'));
        else
          fprintf(flatex_table, cstrcat('\\begin{tabular}{c | c M{1.4cm} M{2.5cm} M{2.3cm}}\n', ...
            'Model & RMSE & R\\textsuperscript{2} & Mean absolute error & Mean relative error \\tabularnewline\n'));
        end

        fprintf(flatex_table, '\\hline\n');
      end


      if LINEAR_REGRESSION
        y_mean = mean(y_test);

        sum_residual = sum((y_test - predictions(:, end)).^2);
        sum_total = sum((y_test - y_mean).^2);

        real_test_values = mu_y + sigma_y * y_test;
        real_predictions = mu_y + sigma_y * predictions(:, end);

        abs_err = abs(real_test_values - real_predictions);
        rel_err = abs_err ./ real_test_values;

        lin_mean_abs = mean(abs_err);
        lin_mean_rel = mean(rel_err);


        % sum_abs = sum(abs(y_test - predictions(:, end)));
        % sum_rel = sum(sigma_y * abs((y_test - predictions(:, end)) ./ (sigma_y * predictions(:, end)) + mu_y);

        lin_RMSE = sqrt(sum_residual / N_test);     % Root Mean Squared Error
        lin_R2 = 1 - (sum_residual / sum_total);    % R^2
        % lin_mean_abs = ((sum_abs / N_test));
        % lin_mean_rel = sum_rel / N_test;

        % fprintf('\n Testing results for linear regression:\n');
        % fprintf('   RMSE = %f\n', lin_RMSE);
        % fprintf('   R^2 = %f\n', lin_R2);
        % fprintf('   Mean abs error = %f\n', lin_mean_abs);
        % fprintf('   Mean rel error = %f\n', lin_mean_rel);

        if lin_RMSE > 1000
          lin_RMSE = Inf;
        end

        if lin_R2 < -1000
          lin_R2 = -Inf;
        end

        if lin_mean_abs > 10000000
          lin_mean_abs = Inf;
        end

        if lin_mean_rel > 1000
          lin_mean_rel = Inf;
        end


        if SAVE_DATA
          fprintf(fd, '\n Testing results for linear regression:\n');
          fprintf(fd, '   RMSE = %f\n', lin_RMSE);
          fprintf(fd, '   R^2 = %f\n', lin_R2);
          fprintf(fd, '   Mean abs error = %f\n', lin_mean_abs);
          fprintf(fd, '   Mean rel error = %f\n', lin_mean_rel);
        end

        RMSEs(end + 1) = lin_RMSE;
        R_2(end + 1) = lin_R2;

        pred_mean = mean(predictions(:, end));
        means(end + 1) = pred_mean;
        if DIFF_MEANS
          diff_means = pred_mean - y_mean;
          fprintf('   Difference between means = %f\n', diff_means);
          if SAVE_DATA
            fprintf(fd, '   Difference between means = %f\n', diff_means);
          end
        end


        if (OUTPUT_LATEX & ~DIFF_MEANS)
          fprintf(flatex_table, 'Linear regression & %5.4f & %5.4f & %6.0f & %5.4f \\\\\n', lin_RMSE, lin_R2, lin_mean_abs, lin_mean_rel);
        end

        if (OUTPUT_LATEX & DIFF_MEANS)
          fprintf(flatex_table, 'Linear regression & %5.4f & %5.4f & %6.0f & %5.4f & %5.4f \\\\\n', lin_RMSE, lin_R2, lin_mean_abs, lin_mean_rel, diff_means);
        end

      end


      for index = 1:length(MODELS_CHOSEN)
        real_predictions = mu_y + sigma_y * predictions(:, index);
        real_test_values = mu_y + sigma_y * y_test;

        abs_err = abs(real_predictions - real_test_values);
        rel_err = abs_err ./ real_test_values;

        mean_abs = mean(abs_err);
        mean_rel = mean(rel_err);

        % fprintf('\n Testing results for %s:\n', SVR_DESCRIPTIONS{index});
        % fprintf('   RMSE = %f\n', RMSEs(index));
        % fprintf('   R^2 = %f\n', R_2(index));
        % fprintf('   Mean abs error = %f\n', mean_abs);
        % fprintf('   Mean rel error = %f\n', mean_rel);

        if RMSEs(index) > 1000
          RMSes(index) = Inf;
        end

        if R_2(index) < -1000
          R_2(index) = -Inf;
        end

        if mean_abs > 10000000
          mean_abs = Inf;
        end

        if mean_rel > 1000
          mean_rel = Inf;
        end

        if SAVE_DATA
          fprintf(fd, '\n Testing results for %s:\n', SVR_DESCRIPTIONS{index});
          fprintf(fd, '   RMSE = %f\n', RMSEs(index));
          fprintf(fd, '   R^2 = %f\n', R_2(index));
          fprintf(fd, '   Mean abs error = %f\n', mean_abs);
          fprintf(fd, '   Mean rel error = %f\n', mean_rel);
        end


        y_mean = mean(y_test);
        pred_mean = mean(predictions(:, index));
        means(end + 1) = pred_mean;
        if DIFF_MEANS
          diff_means = pred_mean - y_mean;
          fprintf('   Difference between means = %f\n', diff_means);
          if SAVE_DATA
            fprintf(fd, '   Difference between means = %f\n', diff_means);
          end
        end

        if (OUTPUT_LATEX & ~DIFF_MEANS)
          fprintf(flatex_table, '%s & %5.4f & %5.4f & %6.0f & %5.4f \\\\\n', SVR_DESCRIPTIONS{index}, RMSEs(index), R_2(index), mean_abs, mean_rel);
        end

        if (OUTPUT_LATEX & DIFF_MEANS)
          fprintf(flatex_table, '%s & %5.4f & %5.4f & %6.0f & %5.4f & %5.4f \\\\\n', SVR_DESCRIPTIONS{index}, RMSEs(index), R_2(index), mean_abs, mean_rel, diff_means);
        end

      end

      if OUTPUT_LATEX
        fprintf(flatex_table, cstrcat('\\end{tabular}\n', ...
                      '\\end{adjustbox}\n', ...
                      '\\\\\n', ...
                      '\\caption{', TABLE_CAPTION_TEST, '}\n', ...
                      '\\label{', TABLE_LABEL_TEST, '}\n', ...
                      '\\end{table}\n'));
        fclose(flatex_table);
      end

      if OUTPUT_LATEX
        fprintf(flatex_plot, cstrcat('\n\\begin {figure}[hbtp]\n', ...
                      '\\centering\n', ...
                      '\\includegraphics[width=\\textwidth]{', OUTPUT_SUBFOLDER, 'plot_', QUERY, '_', DATASIZE, OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{2}, '}\n', ...
                      '\\caption{', PLOT_CAPTION, '}\n', ...
                      '\\label{', PLOT_LABEL, '}\n', ...
                      '\\end {figure}\n'));
        fclose(flatex_plot);

        if BEST_MODELS
          fprintf(flatex_plot_bestmodels, cstrcat('\n\\begin {figure}[hbtp]\n', ...
                              '\\centering\n', ...
                              '\\includegraphics[width=\\textwidth]{', OUTPUT_SUBFOLDER, 'plot_', QUERY, '_', DATASIZE, '_bestmodels', OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{2}, '}\n', ...
                              '\\caption{', PLOT_CAPTION, '}\n', ...
                              '\\label{', PLOT_LABEL, '}\n', ...
                              '\\end {figure}\n'));
          fclose(flatex_plot_bestmodels);
        end
      end

      %% Stores the context and closes the file descriptor
      if SAVE_DATA
        fprintf(fd, '\n\n\n========================\n\n\n');
        fprintf(fd, 'ENABLE_FEATURE_FILTERING: %d\n', ENABLE_FEATURE_FILTERING);
        fprintf(fd, 'COMPLETION_TIME_THRESHOLD: %d\n', COMPLETION_TIME_THRESHOLD);
        fprintf(fd, 'TRAIN_FRAC_WO_TEST: %f\n', TRAIN_FRAC_WO_TEST);
        fprintf(fd, 'TEST_FRAC_WO_TEST: %f\n', TEST_FRAC_WO_TEST);
        fprintf(fd, 'TRAIN_FRAC_W_TEST: %f\n', TRAIN_FRAC_W_TEST);
        fprintf(fd, 'NORMALIZE_FEATURE: %d\n', NORMALIZE_FEATURE);
        fprintf(fd, 'CLEAR_OUTLIERS: %d\n', CLEAR_OUTLIERS);
        fprintf(fd, 'CHOOSE_FEATURES: %d\n', CHOOSE_FEATURES);
        fprintf(fd, 'FEATURES: %s --> ', mat2str(FEATURES));
        for id = 1:length(FEATURES)
          fprintf(fd, '%s   ', FEATURES_DESCRIPTIONS{id});
        end
        fprintf(fd, '\n');
        fprintf(fd, 'DIFF_MEANS: %d\n', DIFF_MEANS);
        fprintf(fd, 'SHUFFLE_DATA: %d\n', SHUFFLE_DATA);
        save(strcat(OUTPUT_SUBFOLDER, 'models.mat'), 'SVs', 'coefficients', 'b', 'models', 'Cs', 'Es', 'theta', 'mu', 'sigma');


        fclose(fd);
      end





      % Denormalize means
      means = (means * sigma_y) + mu_y;




      %% Denormalize features

      if NORMALIZE_FEATURE
        % printf('\nDenormalizing features...');
        X_tr_denorm = X_tr .* (ones(N_train, 1) * sigma_X) .+ (ones(N_train, 1) * mu_X);
        y_tr_denorm = y_tr * sigma_y + mu_y;
        X_test_denorm = X_test .* (ones(N_test, 1) * sigma_X) .+ (ones(N_test, 1) * mu_X);
        y_test_denorm = y_test * sigma_y + mu_y;
      else
        X_tr_denorm = X_tr;
        y_tr_denorm = y_tr;
        X_test_denorm = X_test;
        y_test_denorm = y_test;
      end

      %% Determine the best 3 models
      if BEST_MODELS
        tempR_2 = R_2;
        best_models_idx = [];
        [~, best_models_idx(end+1)] = max(tempR_2);
        tempR_2(best_models_idx(end)) = -1;
        [~, best_models_idx(end+1)] = max(tempR_2);
        tempR_2(best_models_idx(end)) = -1;
        [~, best_models_idx(end+1)] = max(tempR_2);
      end


      %% PLOTTING SVR vs LR
      printf('\nDrawing and saving plots...');

      if ALL_THE_PLOTS
        for col = 1:M

          figure;
          hold on;

          % scatter(X_tr_denorm(:, col), y_tr_denorm, 'r', 'x');
          % scatter(X_test_denorm(:, col), y_test_denorm, 'b');
          X_tr_denorm_col = X_tr_denorm(:, col);
          X_test_denorm_col = X_test_denorm(:, col);

          if (N_CORES_INVERSE & ismember(13, FEATURES) & (col == M))
            X_tr_denorm_col = 1./X_tr_denorm_col;
            X_test_denorm_col = 1./X_test_denorm_col;
          end

          my_scatter(X_tr_denorm_col, y_tr_denorm, 'r', 'x');
          my_scatter(X_test_denorm_col, y_test_denorm, 'b');


          % x = linspace(min(X_test(:, col)), max(X_test(:, col)));   % Normalized, we need this for the predictions
          x = linspace(min(min(X_test(:, col)), min(X_tr(:, col))), max(max(X_test(:, col)), max(X_tr(:, col))));  %% fill all the plot
          x_denorm = (x * sigma_X(col)) + mu_X(col);

          xsvr = zeros(length(x), M);     % xsvr is a matrix of zeros, except for the column we're plotting currently
          xsvr(:, col) = x;         % It must be normalized to use svmpredict with the SVR models we found


          if LINEAR_REGRESSION
            ylin = x * theta(col+1);

            % Denormalize y
            if NORMALIZE_FEATURE
              ylin = (ylin * sigma_y) + mu_y;
            end

            x_plot = x_denorm;
            if (N_CORES_INVERSE & ismember(13, FEATURES) & (col == M))
              x_plot = 1./x_plot;
            end

            plot(x_plot, ylin, 'color', [0.5, 0, 1], 'linewidth', 1);

            x = x_denorm;
            y = ylin;
            if SAVE_DATA
              save(cstrcat(OUTPUT_SUBFOLDER, 'Linear Regression.mat'), 'x', 'y', 'QUERY', 'DATASIZE');
            end

          end

          for index = 1:length(MODELS_CHOSEN)
            [ysvr, ~, ~] = svmpredict(zeros(length(x), 1), xsvr, models{index}, '-q');  %% quiet

            % Denormalize
            if NORMALIZE_FEATURE
              ysvr = (ysvr * sigma_y) + mu_y;
            end

            x_plot = x_denorm;
            if (N_CORES_INVERSE & ismember(13, FEATURES) & (col == M))
              x_plot = 1./x_plot;
            end

            plot(x_plot, ysvr, 'color', COLORS{index}, 'linewidth', 1);

            x = x_denorm;
            y = ysvr;
            if SAVE_DATA
              save(cstrcat(OUTPUT_SUBFOLDER, SVR_DESCRIPTIONS{index}, '.mat'), 'x', 'y', 'QUERY', 'DATASIZE');
            end
          end

          % Plot the mean of the test values (for nCores)
          % if (DIFF_MEANS & ismember(13, FEATURES) & (col == M))
          %   scatter(X_test_denorm(1, col), mean(y_test_denorm), 10, 'k', '.');
          % end

          labels = {'Training set', 'Testing set'};
          if LINEAR_REGRESSION
            labels{end+1} = 'Linear Regression';
          end
          labels(end+1:end+length(SVR_DESCRIPTIONS)) = SVR_DESCRIPTIONS;
          legend(labels, 'location', 'northeastoutside');



          % Labels the axes
          xlabel(FEATURES_DESCRIPTIONS{col});
          ylabel('Completion Time');
          % title(cstrcat('Linear regression vs ', SVR_DESCRIPTIONS{svr_index}));
          if SAVE_DATA
            % NOTE: the file location shouldn't have any spaces
            file_location = strrep(strcat(OUTPUT_SUBFOLDER, 'plot_', FEATURES_DESCRIPTIONS{col}, OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{2}), ' ', '');
            % file_location = strrep(strcat(OUTPUT_SUBFOLDER, 'plot_', QUERY, '_', DATASIZE, FEATURES_DESCRIPTIONS{col}, OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{2}), ' ', '');
            print(OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{1}, file_location);
          end

          if SAVE_DATA & ismember(13, FEATURES) & (col == M)
            file_location = strrep(strcat(OUTPUT_SUBFOLDER, 'plot_', QUERY, '_', DATASIZE, OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{2}), ' ', '');
            print(OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{1}, file_location);
          end

          hold off;

          % pause;

        end

      else

        col = M;

        figure;
        hold on;

        % scatter(X_tr_denorm(:, col), y_tr_denorm, 'r', 'x');
        % scatter(X_test_denorm(:, col), y_test_denorm, 'b');
        X_tr_denorm_col = X_tr_denorm(:, col);
        X_test_denorm_col = X_test_denorm(:, col);

        if (N_CORES_INVERSE & ismember(13, FEATURES) & (col == M))
          X_tr_denorm_col = 1./X_tr_denorm_col;
          X_test_denorm_col = 1./X_test_denorm_col;
        end

        my_scatter(X_tr_denorm_col, y_tr_denorm, 'r', 'x');
        my_scatter(X_test_denorm_col, y_test_denorm, 'b');


        % x = linspace(min(X_test(:, col)), max(X_test(:, col)));   % Normalized, we need this for the predictions
        x = linspace(min(min(X_test(:, col)), min(X_tr(:, col))), max(max(X_test(:, col)), max(X_tr(:, col))));  %% fill all the plot
        x_denorm = (x * sigma_X(col)) + mu_X(col);

        xsvr = zeros(length(x), M);     % xsvr is a matrix of zeros, except for the column we're plotting currently
        xsvr(:, col) = x;         % It must be normalized to use svmpredict with the SVR models we found


        if LINEAR_REGRESSION
          ylin = x * theta(col+1);

          % Denormalize y
          if NORMALIZE_FEATURE
            ylin = (ylin * sigma_y) + mu_y;
          end

          x_plot = x_denorm;
          if (N_CORES_INVERSE & ismember(13, FEATURES) & (col == M))
            x_plot = 1./x_plot;
          end

          plot(x_plot, ylin, 'color', [0.5, 0, 1], 'linewidth', 1);

          x = x_denorm;
          y = ylin;
          if SAVE_DATA
            save(cstrcat(OUTPUT_SUBFOLDER, 'Linear Regression.mat'), 'x', 'y', 'QUERY', 'DATASIZE');
          end

        end

        for index = 1:length(MODELS_CHOSEN)
          [ysvr, ~, ~] = svmpredict(zeros(length(x), 1), xsvr, models{index}, '-q');  %% quiet

          % Denormalize
          if NORMALIZE_FEATURE
            ysvr = (ysvr * sigma_y) + mu_y;
          end

          x_plot = x_denorm;
          if (N_CORES_INVERSE & ismember(13, FEATURES) & (col == M))
            x_plot = 1./x_plot;
          end

          plot(x_plot, ysvr, 'color', COLORS{index}, 'linewidth', 1);

          x = x_denorm;
          y = ysvr;
          if SAVE_DATA
            save(cstrcat(OUTPUT_SUBFOLDER, SVR_DESCRIPTIONS{index}, '.mat'), 'x', 'y', 'QUERY', 'DATASIZE');
          end
        end

        % Plot the mean of the test values (for nCores)
        % if (DIFF_MEANS & ismember(13, FEATURES) & (col == M))
        %   scatter(X_test_denorm(1, col), mean(y_test_denorm), 10, 'k', '.');
        % end

        labels = {'Training set', 'Testing set'};
        if LINEAR_REGRESSION
          labels{end+1} = 'Linear Regression';
        end
        labels(end+1:end+length(SVR_DESCRIPTIONS)) = SVR_DESCRIPTIONS;
        legend(labels, 'location', 'northeastoutside');



        % Labels the axes
        xlabel(FEATURES_DESCRIPTIONS{col});
        ylabel('Completion Time');
        % title(cstrcat('Linear regression vs ', SVR_DESCRIPTIONS{svr_index}));
        if SAVE_DATA
          % NOTE: the file location shouldn't have any spaces
          file_location = strrep(strcat(OUTPUT_SUBFOLDER, 'plot_', FEATURES_DESCRIPTIONS{col}, OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{2}), ' ', '');
          % file_location = strrep(strcat(OUTPUT_SUBFOLDER, 'plot_', QUERY, '_', DATASIZE, FEATURES_DESCRIPTIONS{col}, OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{2}), ' ', '');
          print(OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{1}, file_location);
        end

        if SAVE_DATA & ismember(13, FEATURES) & (col == M)
          file_location = strrep(strcat(OUTPUT_SUBFOLDER, 'plot_', QUERY, '_', DATASIZE, OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{2}), ' ', '');
          print(OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{1}, file_location);
        end

        hold off;

        % pause;

      end


      %% Plot and save (only the best models)

      if BEST_MODELS

        if ALL_THE_PLOTS

          for col = 1:M

            figure;
            hold on;

            % scatter(X_tr_denorm(:, col), y_tr_denorm, 'r', 'x');
            % scatter(X_test_denorm(:, col), y_test_denorm, 'b');
            X_tr_denorm_col = X_tr_denorm(:, col);
            X_test_denorm_col = X_test_denorm(:, col);

            if (N_CORES_INVERSE & ismember(13, FEATURES) & (col == M))
              X_tr_denorm_col = 1./X_tr_denorm_col;
              X_test_denorm_col = 1./X_test_denorm_col;
            end

            my_scatter(X_tr_denorm_col, y_tr_denorm, 'r', 'x');
            my_scatter(X_test_denorm_col, y_test_denorm, 'b');


            % x = linspace(min(X_test(:, col)), max(X_test(:, col)));   % Normalized, we need this for the predictions
            x = linspace(min(min(X_test(:, col)), min(X_tr(:, col))), max(max(X_test(:, col)), max(X_tr(:, col))));  %% fill all the plot
            x_denorm = (x * sigma_X(col)) + mu_X(col);

            xsvr = zeros(length(x), M);     % xsvr is a matrix of zeros, except for the column we're plotting currently
            xsvr(:, col) = x;         % It must be normalized to use svmpredict with the SVR models we found


            if (LINEAR_REGRESSION && ismember(length(MODELS_CHOSEN)+1, best_models_idx))  % if linear regression is one of the best models
              ylin = x * theta(col+1);

              % Denormalize y
              if NORMALIZE_FEATURE
                ylin = (ylin * sigma_y) + mu_y;
              end

              x_plot = x_denorm;
              if (N_CORES_INVERSE & ismember(13, FEATURES) & (col == M))
                x_plot = 1./x_plot;
              end

              plot(x_plot, ylin, 'color', [0.5, 0, 1], 'linewidth', 1);

            end

            for index = 1:length(MODELS_CHOSEN)
              if ismember(index, best_models_idx)
                [ysvr, ~, ~] = svmpredict(zeros(length(x), 1), xsvr, models{index}, '-q');  %% quiet

                % Denormalize
                if NORMALIZE_FEATURE
                  ysvr = (ysvr * sigma_y) + mu_y;
                end

                x_plot = x_denorm;
                if (N_CORES_INVERSE & ismember(13, FEATURES) & (col == M))
                  x_plot = 1./x_plot;
                end

                plot(x_plot, ysvr, 'color', COLORS{index}, 'linewidth', 1);
              end
            end

            % Plot the mean of the test values (for nCores)
            % if (DIFF_MEANS & ismember(13, FEATURES) & (col == M))
            %   scatter(X_test_denorm(1, col), mean(y_test_denorm), 10, 'k', '.');
            % end

            labels = {'Training set', 'Testing set'};
            if (LINEAR_REGRESSION && ismember(length(MODELS_CHOSEN)+1, best_models_idx))
              labels{end+1} = 'Linear Regression';
            end
            for index = 1:length(SVR_DESCRIPTIONS)
              if ismember(index, best_models_idx)
                labels(end+1) = SVR_DESCRIPTIONS{index};
              end
            end
            legend(labels, 'location', 'northeastoutside');



            % Labels the axes
            xlabel(FEATURES_DESCRIPTIONS{col});
            ylabel('Completion Time');
            % title(cstrcat('Linear regression vs ', SVR_DESCRIPTIONS{svr_index}));
            if SAVE_DATA
              % NOTE: the file location shouldn't have any spaces
              file_location = strrep(strcat(OUTPUT_SUBFOLDER, 'plot_', FEATURES_DESCRIPTIONS{col}, '_bestmodels', OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{2}), ' ', '');
              % file_location = strrep(strcat(OUTPUT_SUBFOLDER, 'plot_', QUERY, '_', DATASIZE, FEATURES_DESCRIPTIONS{col}, OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{2}), ' ', '');
              print(OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{1}, file_location);
            end

            if SAVE_DATA & ismember(13, FEATURES) & (col == M)
              file_location = strrep(strcat(OUTPUT_SUBFOLDER, 'plot_', QUERY, '_', DATASIZE, '_bestmodels', OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{2}), ' ', '');
              print(OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{1}, file_location);
            end

            hold off;

            % pause;

          end

        else
          col = M;

          figure;
          hold on;

          % scatter(X_tr_denorm(:, col), y_tr_denorm, 'r', 'x');
          % scatter(X_test_denorm(:, col), y_test_denorm, 'b');
          X_tr_denorm_col = X_tr_denorm(:, col);
          X_test_denorm_col = X_test_denorm(:, col);

          if (N_CORES_INVERSE & ismember(13, FEATURES) & (col == M))
            X_tr_denorm_col = 1./X_tr_denorm_col;
            X_test_denorm_col = 1./X_test_denorm_col;
          end

          my_scatter(X_tr_denorm_col, y_tr_denorm, 'r', 'x');
          my_scatter(X_test_denorm_col, y_test_denorm, 'b');


          % x = linspace(min(X_test(:, col)), max(X_test(:, col)));   % Normalized, we need this for the predictions
          x = linspace(min(min(X_test(:, col)), min(X_tr(:, col))), max(max(X_test(:, col)), max(X_tr(:, col))));  %% fill all the plot
          x_denorm = (x * sigma_X(col)) + mu_X(col);

          xsvr = zeros(length(x), M);     % xsvr is a matrix of zeros, except for the column we're plotting currently
          xsvr(:, col) = x;         % It must be normalized to use svmpredict with the SVR models we found


          if (LINEAR_REGRESSION && ismember(length(MODELS_CHOSEN)+1, best_models_idx))  % if linear regression is one of the best models
            ylin = x * theta(col+1);

            % Denormalize y
            if NORMALIZE_FEATURE
              ylin = (ylin * sigma_y) + mu_y;
            end

            x_plot = x_denorm;
            if (N_CORES_INVERSE & ismember(13, FEATURES) & (col == M))
              x_plot = 1./x_plot;
            end

            plot(x_plot, ylin, 'color', [0.5, 0, 1], 'linewidth', 1);

          end

          for index = 1:length(MODELS_CHOSEN)
            if ismember(index, best_models_idx)
              [ysvr, ~, ~] = svmpredict(zeros(length(x), 1), xsvr, models{index}, '-q');  %% quiet

              % Denormalize
              if NORMALIZE_FEATURE
                ysvr = (ysvr * sigma_y) + mu_y;
              end

              x_plot = x_denorm;
              if (N_CORES_INVERSE & ismember(13, FEATURES) & (col == M))
                x_plot = 1./x_plot;
              end

              plot(x_plot, ysvr, 'color', COLORS{index}, 'linewidth', 1);
            end
          end

          % Plot the mean of the test values (for nCores)
          % if (DIFF_MEANS & ismember(13, FEATURES) & (col == M))
          %   scatter(X_test_denorm(1, col), mean(y_test_denorm), 10, 'k', '.');
          % end

          labels = {'Training set', 'Testing set'};
          if (LINEAR_REGRESSION && ismember(length(MODELS_CHOSEN)+1, best_models_idx))
            labels{end+1} = 'Linear Regression';
          end
          for index = 1:length(SVR_DESCRIPTIONS)
            if ismember(index, best_models_idx)
              labels(end+1) = SVR_DESCRIPTIONS{index};
            end
          end
          legend(labels, 'location', 'northeastoutside');



          % Labels the axes
          xlabel(FEATURES_DESCRIPTIONS{col});
          ylabel('Completion Time');
          % title(cstrcat('Linear regression vs ', SVR_DESCRIPTIONS{svr_index}));
          if SAVE_DATA
            % NOTE: the file location shouldn't have any spaces
            file_location = strrep(strcat(OUTPUT_SUBFOLDER, 'plot_', FEATURES_DESCRIPTIONS{col}, '_bestmodels', OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{2}), ' ', '');
            % file_location = strrep(strcat(OUTPUT_SUBFOLDER, 'plot_', QUERY, '_', DATASIZE, FEATURES_DESCRIPTIONS{col}, OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{2}), ' ', '');
            print(OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{1}, file_location);
          end

          if SAVE_DATA & ismember(13, FEATURES) & (col == M)
            file_location = strrep(strcat(OUTPUT_SUBFOLDER, 'plot_', QUERY, '_', DATASIZE, '_bestmodels', OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{2}), ' ', '');
            print(OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{1}, file_location);
          end

          hold off;

          % pause;

        end

      end

      if OUTPUT_LATEX
        if LATEX_TABLE
          fprintf(flatex, cstrcat('\\input{', latex_filename_table, '}\n'));
        end

        if LATEX_PLOT
          fprintf(flatex, cstrcat('\\input{', latex_filename_plot, '}\n'));
        end

        if LATEX_PLOT_BESTMODELS
          fprintf(flatex, cstrcat('\\input{', latex_filename_plot_bestmodels, '}\n'));
        end

        fprintf(flatex, '\n\\newpage\n');
      end

      TESTS_TO_DO(1) = [];

      printf('\nDone.\n');

    catch
      % do nothing, redo test
      warning('Test failed, retrying...\n');
      continue;
    end
  end

end

if OUTPUT_LATEX
  fclose(flatex);
end
