function [acc_r, acc_std] = cal_acc(prefix, n)

addpath('./usages');

% n = 10;
% prefix = '/Users/hyzhang/MachineLearning/DFKM/results/jaffe/ae_';
accs = [];
for i = 0: n-1
    name = [prefix, mat2str(i), '.mat']
    load (name)
    acc = CalMetricOfCluster(y_predicted, y);
    accs = [accs, acc];
end
acc_r = mean(accs);
acc_std = std(accs);
