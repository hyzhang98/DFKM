L1 = [2, 1, 1, 4, 1, 3, 4, 3, 1, 4, 1, 1, 4, 2, 2, 4, 2, 2, 4, 1]
L2 = [1, 4, 4, 2, 3, 3, 2, 4, 4, 3, 2, 4, 1, 1, 3, 2, 4, 1, 1, 1]

[acc_r, nmi_r, ARI, Precision, Recall, Fscore] = CalMetricOfCluster(L2,L1)
Predict_label = bestMap(L1,L2);
sum(Predict_label'==L1)