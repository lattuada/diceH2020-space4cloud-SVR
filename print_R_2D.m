clear all
close all hidden
clc

query = "R5";
directory = ["/home/gianniti/policloud-runs/", query];

[y, X] = read_from_directory (directory);

dataSize = X(:, end-1);
sizes = unique (sort (dataSize))';

nCores = X(:, end);

for (sz = sizes)

  idx = (dataSize == sz);
  nCores_loc = nCores(idx);
  y_loc = y(idx);

  figure;
  plot (nCores_loc, y_loc, "bx", "linewidth", 2);
  grid on;
  xlabel ('Number of cores');
  ylabel ('Response time');
  size_string = num2str (sz);
  title_string = ['Job response time against cores at ', size_string, ' GB'];
  title (title_string);

  values = unique (nCores_loc);
  avg = zeros (size (values));
  for (ii = 1:length (values))
    avg(ii) = mean (y_loc(nCores_loc == values(ii)));
  endfor
  hold on;
  plot (values, avg, "r:", "linewidth", 2);
  hold off;

  figure_name = [query, "_s", size_string, ".eps"];
  print ("-depsc2", figure_name);

  close all hidden;

endfor
