addpath('usages/');
shape = size(Us);
iterations = shape(1);
rows = shape(2);
cols = shape(3);
accs = [];
for iter = 1:iterations
    U = Us(iter, :, :);
    U = reshape(U, rows, cols);
    [~, y_pre] = max(U, [], 2);
    acc = CalMetricOfCluster(y_pre, Y);
    accs = [accs, acc];
end
clear acc
clear shape
clear U
clear y_pre
clear rows
clear cols
clear iter*