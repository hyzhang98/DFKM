% Description:  This code is developed to calculate the Clustering Accuracy and NMI

function [acc_r, nmi_r, ARI, Precision, Recall, Fscore] = CalMetricOfCluster(Predict_label,ttls)
if size(Predict_label,1)<size(Predict_label,2)
    Predict_label=Predict_label';
end;
if size(ttls,1)<size(ttls,2)
    ttls=ttls';
end;

Predict_label = bestMap(ttls,Predict_label);
Predict_label = reshape(Predict_label,1,[]);
ttls = reshape(ttls,size(Predict_label,1),[]);
acc_r = length(find(ttls == Predict_label))/length(ttls);
nmi_r = MutualInfo(ttls,Predict_label);

[ARI] = RandIndex(ttls,Predict_label); % returns the adjusted Rand index, the unadjusted Rand index, "Mirkin's" index and "Hubert's" index.
[Fscore,Precision,Recall] = compute_f(ttls,Predict_label);
