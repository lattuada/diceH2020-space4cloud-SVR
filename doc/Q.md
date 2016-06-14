# Script description

_Andrea Battistello and Pietro Ferretti_

## Script overview

There are several different scripts, each of them performs a
slightly different operation.
The main reason why there are different scripts are: the Q queries have
different columns than the R queries, the query comparison task is slightly
different than the other and easier data manipulation.

The included files are:
- `skeleton_query_Q.m`: runs with Q2, Q3 and Q4.
                        It is fixed datasize, so every run will consider
                        only one dataset.
- `skeleton_query_Q_all_datasize.m`: runs with Q2, Q3 and Q4 and it is quite the
                                     same as `skeleton_query_Q.m`, but this one
                                     merges all the datasets toghether.
- `skeleton_query_comparison`: will run a query comparison task, so training
                               with one query and testing with another one.
                               This works only with R queries.
- `skeleton_fixed_datasize_shuffled.m`: Performs a fixed-datasize test with all
                                        the test data chosen among training
                                        (shuffled).
                                        Works only with R queries
- `skeleton_fixed_datasizes_test_on_cores.m`: Performs a fixed-datasize test
                                              with allthe test data chosen based
                                              on the #cores used.
                                              Works only with R queries.
- `skeleton_mixed_datasize_shuffled.m`: Joins all the datasizes of each query
                                        with all the test data chosen among
                                        training (shuffled).
                                        Works only with R queries.
- `skeleton_mixed_datasizes_test_on_cores.m`: Joins all the datasizes of each
                                              query and test with data chosen
                                              based on the #cores used.
                                              Works only with R queries.
- `utility/...`: All these files are nearly the same as Eugenio gave us, except
                 for some additions like `clear_outliers_ncores` that tries to
                 reduce variance removing the outliers from all the data with
                 fixed number of cores.

## Workflow

1. Retrieve data for each query and datasize.
2. Split training and testing data in two different matrices.
   If the test data were already different from training data, they are
   kept separated.
3. Clear outliers, add nonlinear features and normalize.
   The normalization is performed across the whole dataset, possibly
   merging and splitting it again afterwards.
4. Training data are shuffled and a subset of training is chosen as cross
   validation.
5. Train all models.
6. After training, test and training statistics are evaluated and saved in
   a LaTeX file.
7. Then, all the models are plotted, and all the plots are saved in EPS format.
