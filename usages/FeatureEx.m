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

function dat = FeatureEx(DATA, par)

% eigenface extracting
if par.nDim < size(DATA, 1)
    [disc_set,disc_value,Mean_Image]  =  Eigenface_f(DATA,par.nDim);
    dat  =  disc_set'*DATA;
else
    dat = DATA;
end;
dat  =  dat./( repmat(sqrt(sum(dat.*dat)), [size(dat, 1),1]) );
