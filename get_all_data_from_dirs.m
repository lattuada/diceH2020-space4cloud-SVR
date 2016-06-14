## Copyright 2016 Andrea Battistello
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


function train_data = get_all_data_from_dirs(base, queries)
	train_data = [];
	for q = queries
		fprintf("\nDEBUG: loading %s", char(q));
		train_data = [train_data; read_from_directory(strcat(char(base), char(q)))];
	endfor
endfunction
