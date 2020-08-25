% =========================================================================
% An example code for the algorithm proposed in
%
%   [1] Xi Peng, Zhang Yi, and Huajin Tang.
%       Robust Subspace Clustering via Thresholding Ridge Regression.
%       The Twenty-Ninth AAAI Conference on Artificial Intelligence (AAAI), Austin, Texas, USA, January 25–29, 2015.

%   [2] Xi Peng, et al.
%       Constructing the L2-Graph for Robust Subspace Learning and Subspace Clustering.
%       Under review.

%
% Written by Xi Peng @ I2R A*STAR
% Nov., 2014.
% =========================================================================



function predict_label = SC(CKSym,nClass)

% parameters for k-means clustering
MAXiter = 2000;% change the values of MAXiter and REPlic may produce different clustering results
REPlic = 200;

N = size(CKSym, 1);
DKS = ( diag( sum(CKSym) ) + 1e-12)^(-1/2); % 1e-12 for removing nan or null entries
LapKS = speye(N) - DKS * CKSym * DKS;
clear DKS CKSym;
[uKS,sKS,vKS] = svd(LapKS);
f = size(vKS,2);
kerKS = vKS(:,f-nClass+1:f);
for j = 1:N
    kerKS(j,:) = kerKS(j,:) ./ norm(kerKS(j,:));
end
clear j;
predict_label = kmeans(kerKS,nClass,'start','sample','maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');

%predict_label = litekmeans(kerKS,nClass,'MaxIter',MAXiter,'Replicates',REPlic);

ind = isnan(predict_label);
if sum(isnan(predict_label)) > 0
    if sum(isnan(predict_label)) > 10
        fprintf('errors in sc.m');
    else
        for i =1:sum(isnan(predict_label))
            perm = randperm(nClass);
            predict_label(ind(i))= perm(1);
        end
    end
end;
