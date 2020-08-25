function [acc_r, nmi_r, ARI, Precision, Recall, Fscore] = avg_kmeans(X, Y, n)

addpath('./usages/');

acc_r = 0;
nmi_r = 0;
ARI = 0;
Precision = 0;
Recall = 0;
Fscore = 0;

for i=1:n
    [acc, nmi, a, pre, rec, f] = CalMetricOfCluster(kmeans(X, length(unique(Y))), Y);
    acc_r = acc_r + acc;
    nmi_r = nmi_r + nmi;
    ARI = ARI + a;
    Precision = Precision + pre;
    Recall = Recall + rec;
    Fscore = Fscore + f;
end
acc_r = acc_r / n;
nmi_r = nmi_r / n;
ARI = ARI / n;
Precision = Precision / n;
Recall = Recall / n;
Fscore = Fscore / n;