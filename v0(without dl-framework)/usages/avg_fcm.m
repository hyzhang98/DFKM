function [acc_r, nmi_r, acc_std, nmi_std, pur_r, pur_std] = avg_fcm(X, Y, n)
addpath('./usages/');

[count, ~] = size(X);
% options = [1.2;100;1e-5;0];
options = [1.15;100;1e-5;0];
accs = [];
nmis = [];
purs = [];
label_count = length(unique(Y));

for i = 1: n
    [~, U] = fcm(X, label_count, options);
    y_predicted = [];
    for j = 1: count
        [~, loc] = max(U(:, j));
        y_predicted = [y_predicted; loc];
    end
    [acc, nmi, ~] = CalMetricOfCluster(y_predicted, Y);
    accs = [accs, acc];
    nmis = [nmis, nmi];
    pur = CalPurity(y_predicted, Y);
    purs = [purs, pur];
end
acc_r = mean(accs);
acc_std = std(accs);
nmi_r = mean(nmis);
nmi_std = std(nmis);
pur_r = mean(purs);
pur_std = std(purs);