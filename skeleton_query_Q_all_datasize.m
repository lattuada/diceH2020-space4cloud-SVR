## Copyright 2016 Andrea Battistello and Pietro Ferretti
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


## Run a test for Q queries joining all the datasizes.

clear all;
clc;
close all hidden;

addpath('./utility/');

BASE_DIR = './dati query Q/Dati_aggiornati_2016_06_07/';        % Data root directory


QUERIES = {'Q2', 'Q3', 'Q4'};
ALL_DATASIZES = {{'40', '50'}, {'50'}, {'50'}};

% TRAIN_CORES = {[8, 12, 16, 20]};
% TEST_CORES = {[]};
TRAIN_CORES = {[24, 32, 40], [16, 32, 40], [16, 24, 40], [16, 24, 32]};
TEST_CORES = {[16], [24], [32], [40]};
% TRAIN_CORES = {[8, 20]};
% TEST_CORES = {[12,16]};

CORE_TEST = true;                 % Perform training on the cores specified and testing on the test core specified
                      % If false the TRAIN_CORES and TEST_CORES will be ignored


OUTPUT_LATEX = true;            % Generate a latex file
LATEX_TABLE = true;             % With the table of results
LATEX_PLOT = true;              % With the plot of all model trained
LATEX_PLOT_BESTMODELS = true;       % With the plot of the best models found
LATEX_FILE_LOCATION = "latex_output/queryQ/";
FILENAME = "core_comparison_nmap_nreduce_ncores_all_datasize.tex";    % Latex output file name (the ouput will be saved in latex_output/Query_Q/<filename>)
TEST_ID = "CORE_COMPARISON_NMAP_NREDUCE_NCORES";      % The output folder will be outputQ/<query>_<datasize>_<suffix>
OUTPUT_FOLDER = "outputQ/";
CAPTION = 'All Datasize, all features core comparison nmap/ncores and nreduce/ncores and 1/ncores';   % Title of LaTex document
TABLE_CAPTION_TEST = 'All Datasize, all features core comparison nmap/ncores and nreduce/ncores and 1/ncores';    % Title of LaTex document
TABLE_CAPTION_TRAIN = 'All Datasize, all features core comparison nmap/ncores and nreduce/ncores and 1/ncores';   % Title of LaTex document

SAVE_DATA = true;


OUTPUT_FORMATS = {  {'-deps', '.eps'},          % generates only one .eps file black and white
          {'-depslatex', '.eps'},       % generates one .eps file containing only the plot and a .tex file that includes the plot and fill the legend with plain text
          {'-depsc', '.eps'},         % generates only one .eps file with colour
          {'-dpdflatex', '.pdf'}        % generates one .pdf file containing only the plot and a .tex file that includes the plot and fill the legend with plain text
          {'-dpdf', '.pdf'}         % generates one complete .pdf file A4
        };
PLOT_SAVE_FORMAT = 3;



ENABLE_FEATURE_FILTERING = false;           % Remove all the data with completion time below COMPLETION_TIME_THRESHOLD
COMPLETION_TIME_THRESHOLD = 32000;


%% CHANGE THESE IF TEST == TRAIN
TRAIN_FRAC_WO_TEST = 0.6;       % Train data fraction if test cases are chosen among all data
TEST_FRAC_WO_TEST = 0.2;        % Test data fraction if test cases are chosen among all data

%% CHANGE THESE IF TEST != TRAIN
TRAIN_FRAC_W_TEST = 0.7;        % Train data fraction for actual training. (The rest will be used for cross validation)


N_CORES_INVERSE = true;       % Replace the ncores feature with its inverse ( ncores^(-1) )
ADD_NMAP_NREDUCE = true;        % Adds the features nmap/ncore and nreduce/ncores

NORMALIZE_FEATURE = true;
CLEAR_OUTLIERS = true;


LEARNING_CURVES = false;        % Print and save learning curves for each model trained
ALL_THE_PLOTS = false;          % If true prints all the plots for each feature. Otherwise prints only the plot
                    % relative to ncores.


CHOOSE_FEATURES = true;         % Performs feature selection, considering ONLY the selected columns

% These will be used to describe the plot axis
ALL_ALL_FEATURES_DESCRIPTIONS = {
  % Query Q2
  {'N map 1','M1avg','M1max','nR2','R2avg','R2max','S2avg','S2max','S2Bavg','S2Bmax','nR3','R3avg','R3max','S3avg','S3max','S3Bavg',...
  'S3Bmax','nM4','M4avg','M4max','nR5','R5avg','R5max','S5avg','S5max','S5Bavg','S5Bmax','users','dataSize','nCores'},
  % Query Q3
  {'nM','nR','Mavg','Mmax','Ravg','Rmax','SHavg','SHmax','Bavg','Bmax','users','dataSize','nCores'},
  % Query Q4
  {'N map 1','M1avg','M1max','nR2','R2avg','R2max','S2avg','S2max','S2Bavg','S2Bmax','nR3','R3avg','R3max','S3avg','S3max',...
  'S3Bavg','S3Bmax','users','dataSize','nCores'}
};

ALL_CORE_IDX = {30, 13, 20};      % Columns of the ncores feature for each query (used for determining if the plot is on ncores or not)
ALL_NMAP_IDX = {1, 1, 1};       % Columns of the nmap feature for each query (used to add nmap/ncores feature)
ALL_NREDUCE_IDX = {4, 2, 4};      % Columns of the nreduce feature for each query (used to add nreduce/ncores feature)
ALL_ALL_FEATURES = {          % Choose which feature will be considered for training and testing
  % Q2
  [1:8, 12:15, 19, 30],
  % Q3
  [1:5, 7, 9, 13],
  %Q4
  [1:8, 12, 14, 20]
};


%% Choose which SVR models to use
% 1 -> Linear SVR
% 2 -> Polynomial SVR (2 degree)
% 3 -> Polynomial SVR (3 degree)
% 4 -> Polynomial SVR (4 degree)
% 5 -> Polynomial SVR (6 degree)
% 6 -> RBF SVR
MODELS_CHOSEN = [1, 2, 3, 4, 5, 6];
COLORS = {'g', [1, 0.5, 0.2], 'c', 'k', 'm', 'r'};  % magenta, orange, cyan, black, green, red

LINEAR_REGRESSION = true;     % Adds linear regression to the models

BEST_MODELS = true;         % Prints the plots with the best models

DIFF_MEANS = false;       % To add the 'difference between means' metric

rand('seed', 18);
SHUFFLE_DATA = true;

% Range of values used for model selection with SVR
C_range = linspace (0.1, 5, 20);
E_range = linspace (0.1, 5, 20);





      % --------------------------------------------------------------------------------------------------
      % |                        AND NOW..                             |
      % |                      THE CODE BEGINS                           |
      % --------------------------------------------------------------------------------------------------







% Create output folder
if ~ exist(LATEX_FILE_LOCATION)   %% Checks if the folder exists
  if ~ mkdir(LATEX_FILE_LOCATION)   %% Try with the mkdir function
    if system(cstrcat('mkdir -p ', LATEX_FILE_LOCATION))    %% This creates subfolders
      fprintf('[ERROR] Could not create output folder\nCreate the output folder first and then restart this script\n');
      quit;
    end
  end
end

%% Create a latex file with all the results, already formatted
if OUTPUT_LATEX
  flatex = fopen(cstrcat(LATEX_FILE_LOCATION, FILENAME), 'w');
  fprintf(flatex, cstrcat('\\newpage\n', ...
              '\\section{', CAPTION,',}\n'));
end



for query_id = 1:length(QUERIES)
  % Get the data related to this specific query

  FEATURES = ALL_ALL_FEATURES{query_id};
  ALL_FEATURES_DESCRIPTIONS = ALL_ALL_FEATURES_DESCRIPTIONS{query_id};
  CORE_IDX = ALL_CORE_IDX{query_id};
  CORE_IDX_AFTER_FEAT_SEL = CORE_IDX;
  DATASIZES = ALL_DATASIZES{query_id};
  NMAP_INDEX = ALL_NMAP_IDX{query_id};
  NREDUCE_INDEX = ALL_NREDUCE_IDX{query_id};

  QUERY = QUERIES{query_id};
  fprintf(flatex, cstrcat('\\subsection{Query ', QUERY, '}\n'));

  for core_id = 1:length(TRAIN_CORES)

    TR_CORE = TRAIN_CORES{core_id};
    TS_CORE = TEST_CORES{core_id};

    if CORE_TEST
      fprintf(flatex, cstrcat('\\subsubsection{Query ', QUERY, ' --- All Datasize test on ', strjoin(strsplit(int2str(TS_CORE)), ', '), '}\n'));
    else
      fprintf(flatex, cstrcat('\\subsubsection{Query ', QUERY, ' --- All Datasize}\n'));
    end
    % fprintf(flatex, cstrcat('\\subsubsection{Query ', QUERY, ' --- All datasize, test on ', strjoin(strsplit(int2str(TS_CORE)), ', '), '}\n'));
    close all hidden;

    TRAIN_DATA_LOCATION = {};
    for datasize_id = 1:length(DATASIZES)
      TRAIN_DATA_LOCATION{end+1} = strcat(QUERY, '/Datasize/', DATASIZES{datasize_id});
    end

    TEST_DATA_LOCATION = {};


    TABLE_CAPTION_TRAIN = cstrcat('Training results for ', QUERY, ' (All Datasize)');
    TABLE_CAPTION_TEST = cstrcat('Testing results for ', QUERY, ' (All Datasize)');
    PLOT_CAPTION = cstrcat('Completion time vs ncores for ', QUERY, ' (All Datasize)');
    TABLE_LABEL_TRAIN = cstrcat('tab1:', TEST_ID, '_', QUERY, '_All_Datasize');
    TABLE_LABEL_TEST = cstrcat('tab2:', TEST_ID, '_', QUERY, '_All_Datasize');
    PLOT_LABEL = cstrcat('fig:', TEST_ID, '_', QUERY, '_All_Datasize');

    if CORE_TEST
      OUTPUT_SUBFOLDER = strcat(OUTPUT_FOLDER, upper(TEST_ID), '_', QUERY, '_ALL_DATASIZE_', strjoin(strsplit(int2str(TS_CORE)), '_'), '/');
    else
      OUTPUT_SUBFOLDER = strcat(OUTPUT_FOLDER, upper(TEST_ID), '_', QUERY, '_ALL_DATASIZE/');
    end



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
    if CORE_TEST
      all_data = get_all_data_from_dirs(BASE_DIR, TRAIN_DATA_LOCATION);

        if CHOOSE_FEATURES
          tmp = all_data(:, 2:end); % Isolates the completion time in the first column with the features on the other columns
          all_data = [all_data(:, 1) , tmp(:, FEATURES)];
          FEATURES_DESCRIPTIONS = ALL_FEATURES_DESCRIPTIONS(FEATURES);
        end

      idx_train = zeros(size(all_data, 1), 1);
      for idx = 1:length(TR_CORE)
        CORE = TR_CORE(idx);
        idx_train = idx_train | (all_data(:,end) == CORE);
      end

      train_data = all_data(idx_train, :);
      test_data = all_data(!idx_train, :);

    else
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

    CORE_IDX_AFTER_FEAT_SEL = M;    %% The #cores feature is always the last column


    % Clear outliers

    complete_data = [train_data ; test_data];


    if CLEAR_OUTLIERS
      [clean, indices] = clear_outliers(complete_data);

      train_data = clean(indices <= N_train, :);
      test_data = clean(indices > N_train, :);

      N_train = size(train_data, 1);    %% Number of train tuples
      N_test = size(test_data, 1);    %% Number of test tuples

      complete_data = [train_data ; test_data];
    end

    % Add non-linear features if needed

    if ADD_NMAP_NREDUCE
      ncores = complete_data(:,CORE_IDX_AFTER_FEAT_SEL+1);
      complete_data(:, end+1) = complete_data(:,NMAP_INDEX+1) ./ ncores;    % ADD NMAP/NCORES
      complete_data(:, end+1) = complete_data(:,NREDUCE_INDEX+1) ./ ncores; % ADD NREDUCE/NCORES
    end


    if N_CORES_INVERSE
      complete_data(:, CORE_IDX_AFTER_FEAT_SEL+1) = 1./complete_data(:, CORE_IDX_AFTER_FEAT_SEL+1);  %% replace nCores with 1/nCores
    end

    % Normalize feature

    mu = zeros(M+1, 1);
    sigma = ones(M+1, 1);

    if NORMALIZE_FEATURE
      [scaled, mu, sigma] = zscore(complete_data);

      train_data = scaled(1:N_train, :);
      test_data = scaled(N_train+1:end, :);

      % Save data for - maybe - later uses
      save(strcat(OUTPUT_SUBFOLDER, 'mu_sigma.mat'), 'mu', 'sigma');

    end


    if SHUFFLE_DATA
      r = randperm(N_train);
      train_data = train_data(r, :);

      %% There is no need to shuffle test data
    end


    %% SPLIT THE DATA IN TRAINING AND CROSS VALIDATION

    cv_data = [];
    N_cv = 0;

    if CORE_TEST
      [train_data, cv_data, ~] = split_sample(train_data, TRAIN_FRAC_W_TEST, 1-TRAIN_FRAC_W_TEST);
      N_train = size(train_data, 1);
      N_cv = size(cv_data, 1);
    else
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
    end


    %% Organize data

    y_tr = train_data(:, 1);
    X_tr = train_data(:, 2:end);

    y_cv = cv_data(:, 1);
    X_cv = cv_data(:, 2:end);

    y_test = test_data(:, 1);
    X_test = test_data(:, 2:end);
    test_col_means = mean(X_test);

    mu_y = mu(1);
    mu_X = mu(2:end);

    sigma_y = sigma(1);
    sigma_X = sigma(2:end);


    %% DECLARE USEFUL VARIABLES

    Cs = [];
    Es = [];
    predictions_tr = [];    % Predictions on training test
    predictions = [];
    coefficients = {};
    SVs = {};
    b = {};
    SVR_DESCRIPTIONS = {};
    models = {};
    means = [];

    % Saving test metrics
    RMSEs = [];
    R_2 = [];
    MAE = []; % Mean absolute error
    MRE = []; % Mean relative error
    DM = [];  % Difference between means
    RMSEs_tr = [];  % RMSE training
    R_2_tr = [];  % R^2 for training

    %% SVR

    % svmtrain parameters
    % -s --> SVM type (3 = epsilon-SVR)
    % -t --> kernel tyle (0 = linear, 1 = polynomial, 2 = gaussian, 3 = sigmoid)
    % -q --> No output
    % -h --> (0 = No shrink)
    % -p --> epsilon
    % -c --> cost


    %% White box model, nCores  LINEAR
    if ismember(1, MODELS_CHOSEN)
      fprintf('\nTraining model with linear SVR\n');
      fflush(stdout);
      SVR_DESCRIPTIONS{end + 1} = 'Linear SVR';

      [C, eps] = model_selection (y_tr, X_tr, y_cv, X_cv, '-s 3 -t 0 -q -h 0', C_range, E_range);
      options = ['-s 3 -t 0 -h 0 -p ', num2str(eps), ' -c ', num2str(C)];
      model = svmtrain (y_tr, X_tr, options);

      [predictions_tr(:, end + 1), accuracy_tr, ~] = svmpredict(y_tr, X_tr, model, '-q');
      [predictions(:, end + 1), accuracy, ~] = svmpredict (y_test, X_test, model, '-q');  %% quiet

      if LEARNING_CURVES
        [m, mse_train, mse_test] = learning_curves(y_tr, X_tr, y_test, X_test, [options, ' -q']);
        h = plot_learning_curves(m, mse_train, mse_test);
        print('-depsc', cstrcat(OUTPUT_SUBFOLDER, 'learning_curve_', SVR_DESCRIPTIONS{end}, '.eps'));
        close(h);
      end

      % Save data

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


    %% Black box model, Polynomial
    if ismember(2, MODELS_CHOSEN)
      fprintf('\nTraining model with polynomial(2) SVR\n');
      fflush(stdout);
      SVR_DESCRIPTIONS{end + 1} = 'Polynomial SVR (2)';

      [C, eps] = model_selection (y_tr, X_tr, y_cv, X_cv, '-s 3 -d 2 -t 1 -q -h 0', C_range, E_range);
      options = ['-s 3 -d 2 -t 1 -h 0 -p ', num2str(eps), ' -c ', num2str(C)];
      model = svmtrain (y_tr, X_tr, options);

      [predictions_tr(:, end + 1), accuracy_tr, ~] = svmpredict(y_tr, X_tr, model, '-q');
      [predictions(:, end + 1), accuracy, ~] = svmpredict (y_test, X_test, model, '-q');  %% quiet

      if LEARNING_CURVES
        [m, mse_train, mse_test] = learning_curves(y_tr, X_tr, y_test, X_test, [options, ' -q']);
        h = plot_learning_curves(m, mse_train, mse_test);
        print('-depsc', cstrcat(OUTPUT_SUBFOLDER, 'learning_curve_', SVR_DESCRIPTIONS{end}, '.eps'));
        close(h);
      end

      % Save data

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

    %% Black box model, Polynomial
    if ismember(3, MODELS_CHOSEN)
      fprintf('\nTraining model with polynomial(3) SVR\n');
      fflush(stdout);
      SVR_DESCRIPTIONS{end + 1} = 'Polynomial SVR (3)';

      [C, eps] = model_selection (y_tr, X_tr, y_cv, X_cv, '-s 3 -d 3 -t 1 -q -h 0', C_range, E_range);
      options = ['-s 3 -d 3 -t 1 -h 0 -p ', num2str(eps), ' -c ', num2str(C)];
      model = svmtrain (y_tr, X_tr, options);

      [predictions_tr(:, end + 1), accuracy_tr, ~] = svmpredict(y_tr, X_tr, model, '-q');
      [predictions(:, end + 1), accuracy, ~] = svmpredict (y_test, X_test, model, '-q');  %% quiet

      if LEARNING_CURVES
        [m, mse_train, mse_test] = learning_curves(y_tr, X_tr, y_test, X_test, [options, ' -q']);
        h = plot_learning_curves(m, mse_train, mse_test);
        print('-depsc', cstrcat(OUTPUT_SUBFOLDER, 'learning_curve_', SVR_DESCRIPTIONS{end}, '.eps'));
        close(h);
      end

      % Save data

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

    %% Black box model, Polynomial
    if ismember(4, MODELS_CHOSEN)
      fprintf('\nTraining model with polynomial(4) SVR\n');
      fflush(stdout);
      SVR_DESCRIPTIONS{end + 1} = 'Polynomial SVR (4)';

      [C, eps] = model_selection (y_tr, X_tr, y_cv, X_cv, '-s 3 -d 4 -t 1 -q -h 0', C_range, E_range);
      options = ['-s 3 -d 4 -t 1 -h 0 -p ', num2str(eps), ' -c ', num2str(C)];
      model = svmtrain (y_tr, X_tr, options);

      [predictions_tr(:, end + 1), accuracy_tr, ~] = svmpredict(y_tr, X_tr, model, '-q');
      [predictions(:, end + 1), accuracy, ~] = svmpredict (y_test, X_test, model, '-q');  %% quiet

      if LEARNING_CURVES
        [m, mse_train, mse_test] = learning_curves(y_tr, X_tr, y_test, X_test, [options, ' -q']);
        h = plot_learning_curves(m, mse_train, mse_test);
        print('-depsc', cstrcat(OUTPUT_SUBFOLDER, 'learning_curve_', SVR_DESCRIPTIONS{end}, '.eps'));
        close(h);
      end

      % Save data

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

    %% Black box model, Polynomial
    if ismember(5, MODELS_CHOSEN)
      fprintf('\nTraining model with polynomial(6) SVR\n');
      fflush(stdout);
      SVR_DESCRIPTIONS{end + 1} = 'Polynomial SVR (6)';

      [C, eps] = model_selection (y_tr, X_tr, y_cv, X_cv, '-s 3 -d 6 -t 1 -q -h 0', C_range, E_range);
      options = ['-s 3 -d 6 -t 1 -h 0 -p ', num2str(eps), ' -c ', num2str(C)];
      model = svmtrain (y_tr, X_tr, options);

      [predictions_tr(:, end + 1), accuracy_tr, ~] = svmpredict(y_tr, X_tr, model, '-q');
      [predictions(:, end + 1), accuracy, ~] = svmpredict (y_test, X_test, model, '-q');  %% quiet

      if LEARNING_CURVES
        [m, mse_train, mse_test] = learning_curves(y_tr, X_tr, y_test, X_test, [options, ' -q']);
        h = plot_learning_curves(m, mse_train, mse_test);
        print('-depsc', cstrcat(OUTPUT_SUBFOLDER, 'learning_curve_', SVR_DESCRIPTIONS{end}, '.eps'));
        close(h);
      end

      % Save data

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

    %% Black box model, RBF (Radial Basis Function)
    if ismember(6, MODELS_CHOSEN)
      fprintf('\nTraining model with RBF SVR\n');
      fflush(stdout);
      SVR_DESCRIPTIONS{end + 1} = 'Gaussian SVR';

      [C, eps] = model_selection (y_tr, X_tr, y_cv, X_cv, '-s 3 -t 2 -q -h 0', C_range, E_range);
      options = ['-s 3 -t 2 -h 0 -p ', num2str(eps), ' -c ', num2str(C)];
      model = svmtrain (y_tr, X_tr, options);

      [predictions_tr(:, end + 1), accuracy_tr, ~] = svmpredict(y_tr, X_tr, model, '-q');
      [predictions(:, end + 1), accuracy, ~] = svmpredict (y_test, X_test, model, '-q');  %% quiet

      if LEARNING_CURVES
        [m, mse_train, mse_test] = learning_curves(y_tr, X_tr, y_test, X_test, [options, ' -q']);
        h = plot_learning_curves(m, mse_train, mse_test);
        print('-depsc', cstrcat(OUTPUT_SUBFOLDER, 'learning_curve_', SVR_DESCRIPTIONS{end}, '.eps'));
        close(h);
      end

      % Save data

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
      fprintf('\nTraining Linear regression.\n');

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

      latex_filename_table = strcat(OUTPUT_SUBFOLDER, 'outputlatex_table.tex');
      flatex_table = fopen(latex_filename_table, 'w');

      latex_filename_plot = strcat(OUTPUT_SUBFOLDER, 'outputlatex_plot.tex');
      flatex_plot = fopen(latex_filename_plot, 'w');

      if BEST_MODELS
        latex_filename_plot_bestmodels = strcat(OUTPUT_SUBFOLDER, 'outputlatex_plot_bestmodels.tex');
        flatex_plot_bestmodels = fopen(latex_filename_plot_bestmodels, 'w');
      end

    end




    %% COMPUTE METRICS FOR ALL MODELS


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


      fprintf(flatex_table, cstrcat('\\begin{tabular}{c | c M{1.4cm} M{2.5cm} M{2.3cm}}\n', ...
          'Model & RMSE & R\\textsuperscript{2} & Mean absolute error & Mean relative error \\tabularnewline\n'));

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



      lin_RMSE = sqrt(sum_residual / N_test);     % Root Mean Squared Error
      lin_R2 = 1 - (sum_residual / sum_total);    % R^2


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
                    '\\includegraphics[width=\\textwidth]{', OUTPUT_SUBFOLDER, 'plot_', QUERY, '_all_datasize', OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{2}, '}\n', ...
                    '\\caption{', PLOT_CAPTION, '}\n', ...
                    '\\label{', PLOT_LABEL, '}\n', ...
                    '\\end {figure}\n'));
      fclose(flatex_plot);

      if BEST_MODELS
        fprintf(flatex_plot_bestmodels, cstrcat('\n\\begin {figure}[hbtp]\n', ...
                            '\\centering\n', ...
                            '\\includegraphics[width=\\textwidth]{', OUTPUT_SUBFOLDER, 'plot_', QUERY, '_all_datasize_bestmodels', OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{2}, '}\n', ...
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


    %% Determine the best 3 models based on RMSE
    if BEST_MODELS
      tempRMSEs = RMSEs;
      best_models_idx = [];
      [~, best_models_idx(end+1)] = min(tempRMSEs);
      tempRMSEs(best_models_idx(end)) = Inf;
      [~, best_models_idx(end+1)] = min(tempRMSEs);
      tempRMSEs(best_models_idx(end)) = Inf;
      [~, best_models_idx(end+1)] = min(tempRMSEs);
    end

    %% PLOTTING SVR vs LR


    feature_index_to_plot = [];
    if ALL_THE_PLOTS
      feature_index_to_plot = [1:M];
    else
      feature_index_to_plot = [M];
    end


    for col=feature_index_to_plot
      figure;
      hold on;

      % scatter(X_tr_denorm(:, col), y_tr_denorm, 'r', 'x');
      % scatter(X_test_denorm(:, col), y_test_denorm, 'b');
      X_tr_denorm_col = X_tr_denorm(:, col);
      X_test_denorm_col = X_test_denorm(:, col);

      if (N_CORES_INVERSE & ismember(CORE_IDX, FEATURES) & (col == CORE_IDX_AFTER_FEAT_SEL))
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
        if (N_CORES_INVERSE & ismember(CORE_IDX, FEATURES) & (col == CORE_IDX_AFTER_FEAT_SEL))
          x_plot = 1./x_plot;
        end

        plot(x_plot, ylin, 'color', [0.5, 0, 1], 'linewidth', 1);

        x = x_denorm;
        y = ylin;
        save(cstrcat(OUTPUT_SUBFOLDER, 'Linear Regression.mat'), 'x', 'y', 'QUERY');


      end

      for index = 1:length(MODELS_CHOSEN)
        [ysvr, ~, ~] = svmpredict(zeros(length(x), 1), xsvr, models{index}, '-q');  %% quiet

        % Denormalize
        if NORMALIZE_FEATURE
          ysvr = (ysvr * sigma_y) + mu_y;
        end

        x_plot = x_denorm;
        if (N_CORES_INVERSE & ismember(CORE_IDX, FEATURES) & (col == CORE_IDX_AFTER_FEAT_SEL))
          x_plot = 1./x_plot;
        end

        plot(x_plot, ysvr, 'color', COLORS{index}, 'linewidth', 1);

        x = x_denorm;
        y = ysvr;
        save(cstrcat(OUTPUT_SUBFOLDER, SVR_DESCRIPTIONS{index}, '.mat'), 'x', 'y', 'QUERY');

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

      if SAVE_DATA & ismember(CORE_IDX, FEATURES) & (col == CORE_IDX_AFTER_FEAT_SEL)
        file_location = strrep(strcat(OUTPUT_SUBFOLDER, 'plot_', QUERY, '_all_datasize', OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{2}), ' ', '');
        print(OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{1}, file_location);
      end

      hold off;

      % pause;
    end


    %% Plot and save (only the best models)


    if BEST_MODELS
      for col = feature_index_to_plot

        figure;
        hold on;

        X_tr_denorm_col = X_tr_denorm(:, col);
        X_test_denorm_col = X_test_denorm(:, col);

        if (N_CORES_INVERSE & ismember(CORE_IDX, FEATURES) & (col == CORE_IDX_AFTER_FEAT_SEL))
          X_tr_denorm_col = 1./X_tr_denorm_col;
          X_test_denorm_col = 1./X_test_denorm_col;
        end

        my_scatter(X_tr_denorm_col, y_tr_denorm, 'r', 'x');
        my_scatter(X_test_denorm_col, y_test_denorm, 'b');


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
          if (N_CORES_INVERSE & ismember(CORE_IDX, FEATURES) & (col == CORE_IDX_AFTER_FEAT_SEL))
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
            if (N_CORES_INVERSE & ismember(CORE_IDX, FEATURES) & (col == CORE_IDX_AFTER_FEAT_SEL))
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

        if SAVE_DATA & ismember(CORE_IDX, FEATURES) & (col == CORE_IDX_AFTER_FEAT_SEL)
          file_location = strrep(strcat(OUTPUT_SUBFOLDER, 'plot_', QUERY, '_all_datasize_bestmodels', OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{2}), ' ', '');
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

  end
end

if OUTPUT_LATEX
  fclose(flatex);
end
