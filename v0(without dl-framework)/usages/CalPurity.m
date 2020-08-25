function [purity] = CalPurity(y_predicted, y)
min_label = min(y_predicted);
y_predicted = y_predicted - min_label + 1;
n = length(y_predicted);
T = {};
count_pred = max(y_predicted);
for i = 1: count_pred
    T{1,i} = [];
end

for i = 1: n
    t = T{1, y_predicted(i,1)};
    t = [t; y(i,1)];
    T{1, y_predicted(i,1)} = t;
end
count = length(T);
pur_count = 0;
for i = 1: count
    t = T{i};
    max_label = mode(t);
    pur_count = pur_count + length(find(t == max_label));
end
purity = pur_count / n;