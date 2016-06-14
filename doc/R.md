# README

## Introduction

The scripts train and test various models on the data we were provided.
Every script is specialized on a specific type of test, on a certain
set of data.

For the R queries:
* `R_fixed_datasizes_shuffled.m`: Runs a test for each query, for each
                                  datasize.
                                  Testing is done on a randomly selected
                                  fraction of the data.
* `R_fixed_datasizes_test_on_cores.m`: Runs a test for each query, for
                                       each datasize.
                                       Testing is done on sets of cores.
* `R_mixed_datasizes_shuffled.m`: Runs a test for each query, considering
                                  all datasizes together.
                                  Testing is done on a randomly selected
                                  fraction of the data.
* `R_mixed_datasizes_test_on_cores.m`: Runs a test for each query,
                                       considering all datasizes together.
                                       Testing is done on sets of cores.
* `R_query_comparison.m`: Still a draft.
                          Runs a test between different sets of queries.


## Requirements

* GNU Octave.
* The [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) library.
* The `dati` folder must be in the same directory as the scripts.
* The `utility` folder must be in the same directory as the scripts.


## Usage

* open Octave
* cd in the scripts directory
* run the script you need

You will find all the plots and results in the `OUTPUT_FOLDER` you selected
(the default is `output/`) and the LaTeX snippet to include them all in the
`LATEX_OUTPUT_FOLDER` you selected (default `latex_output/`).

To use the LaTeX output files add them to the `template.tex` file inside an
`\input{}` tag.
You need all the packages listed in the template file to compile it.


## Configuration

* `TEST_ID`: this will be the name of the output LaTeX file, and the prefix of
             all the output subfolders.
* `OUTPUT_FOLDER`: all the results and plots will be saved here.
                   The default is `output`.
* `LATEX_OUTPUT_FOLDER`: the output LaTeX file will be saved here.
* `QUERIES`: which queries to consider.
* `DATASIZES`: which datasizes to consider.
               This is only present in fixed_datasize scripts.
* `MODELS_CHOSEN`: which Support Vector Machines to use as models.
* `LINEAR_REGRESSION`: `true` if you want to consider linear regression too.
* `FEATURES`: which features to consider.
* `SPECIAL_FEATURES`: `true` if you want to add nmap/ncores and
                      nreduce/ncores as features.
* `N_CORES_INVERSE`: `true` if you want to use ncores^-1 instead of ncores.
