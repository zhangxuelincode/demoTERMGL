function [y,X,theta,w,Xw ] = SynthesizeDataset( N, P, T, L, S,synth )
%SYNTHESIZEDATASET Summary of this function goes here
%   Detailed explanation goes here
%\\\\  Design matrices
if ~isfield(synth.design,'setting')
    synth.design.setting = 'general';
end

%\\\\ Features

if ~isfield(synth.features,'distrib')
    synth.features.distrib = 'normal';
end

if synth.features.param(2)==0
    synth.features.param(2) = eps;
end

%% GROUP STRUCTURE
nvar = P;
switch synth.groups.distrib
    case 'equal'
        assert(mod(nvar,L)==0,'nvar should be multiple of L for groups of equal size (synthesis convention)');
        tmp = repmat([1:L]',[1 nvar/L])';
        groupBelonging = tmp(1:end);
    case 'randn1'
        groupBelonging = sort(datasample([1:L],nvar,'Replace',true)');
    case 'randn2'
        r                   = randn(L,nvar);
        prob                = exp(r)./(sum(exp(r),1));
        [~,groupBelonging]  = max(prob);
        groupBelonging      = sort(groupBelonging);
    case 'rand'
        r                   = randn(L,1);
        groupSize           = round(P*exp(r)./(sum(exp(r))));
        groupSize(1)        = groupSize(1) + (nvar-sum(groupSize));
        groupBelonging      = 1+sum([1:nvar]>cumsum(groupSize));
end


theta = zeros(P,L);
for pp=1:P
    theta(pp,groupBelonging(pp)) = 1;
end



%% ORACLE FEATURES

w   = zeros(P,T);
w0  = ones(P, T);
for tt=1:T
    for ss=1:S
        indi = randi(L);
        w(groupBelonging==indi,tt) = w0(groupBelonging==indi,tt);
    end
end



%% DESIGN MATRICES

for tt=1:T
    X_trtmp = normrnd(synth.design.mean,synth.design.std,N,P);
    X_vltmp = normrnd(synth.design.mean,synth.design.std,N,P);
    X_tstmp = normrnd(synth.design.mean,synth.design.std,N,P);
    
    X.trn{tt}       = X_trtmp;
    X.val{tt}       = X_vltmp;
    X.tst{tt}       = X_tstmp;
      
    Xw.trn(:,tt)    = X.trn{tt}*w(:,tt);
    Xw.val(:,tt)    = X.val{tt}*w(:,tt);
    Xw.tst(:,tt)    = X.tst{tt}*w(:,tt);
end


%% NOISY VECTORS OF OUTPUTS
noiselevel = synth.noise.percentage;

switch synth.noise.distrib
    case 'normal'
        y.trn   = Xw.trn + noiselevel*randraw(synth.noise.distrib,synth.noise.param,[N T]);  %noiselevel*trnd(3,N,T);%
        y.val   = Xw.val + noiselevel*randraw(synth.noise.distrib,synth.noise.param,[N T]);
        y.tst   = Xw.tst + noiselevel*randraw(synth.noise.distrib,synth.noise.param,[N T]);    
        y.true  = Xw.tst;
    case 't'
        y.trn   = Xw.trn + noiselevel*randraw(synth.noise.distrib,synth.noise.degree,[N T]);  %noiselevel*trnd(3,N,T);%
        y.val   = Xw.val + noiselevel*randraw(synth.noise.distrib,synth.noise.degree,[N T]);
        y.tst   = Xw.tst + noiselevel*randraw(synth.noise.distrib,synth.noise.degree,[N T]);    
        y.true  = Xw.tst;
    case 'chisq'
        y.trn   = Xw.trn + noiselevel*randraw(synth.noise.distrib,synth.noise.degree,[N T]);  %noiselevel*trnd(3,N,T);%
        y.val   = Xw.val + noiselevel*randraw(synth.noise.distrib,synth.noise.degree,[N T]);
        y.tst   = Xw.tst + noiselevel*randraw(synth.noise.distrib,synth.noise.degree,[N T]);    
        y.true  = Xw.tst;
    case 'exp'
        y.trn   = Xw.trn + noiselevel*exprnd(2, N, T) ;  %noiselevel*trnd(3,N,T);%
        y.val   = Xw.val + noiselevel*exprnd(2, N, T) ;
        y.tst   = Xw.tst + noiselevel*exprnd(2, N, T) ;    
        y.true  = Xw.tst;
end

ytmp = y;
clear y;
for tt=1:T
    y.trn{tt} = ytmp.trn(:,tt);
    mean_trn  = mean(ytmp.trn(:,tt));
    std_trn   = std2(ytmp.trn(:,tt));
    y.trn{tt} =  (y.trn{tt} - mean_trn)/std_trn;
    
    y.val{tt} = ytmp.val(:,tt);
    mean_val  = mean(ytmp.val(:,tt));
    std_val   = std2(ytmp.val(:,tt));
    y.val{tt} =  (y.val{tt} - mean_val)/std_val;
    
    y.tst{tt} = ytmp.tst(:,tt);
    mean_tst  = mean(ytmp.tst(:,tt));
    std_tst   = std2(ytmp.tst(:,tt));
    y.tst{tt} =  (y.tst{tt} - mean_tst)/std_tst;

    y.true{tt} = ytmp.true(:,tt);
    mean_true  = mean(ytmp.true(:,tt));
    std_true   = std2(ytmp.true(:,tt));
    y.true{tt} =  (y.true{tt} - mean_true)/std_true;
end

%Add Outliers
if synth.outlier.type == 1
    [X,y]  = Outliers1( X,y,synth.outlier.number );
else
    [X,y]  = Outliers2( X,y,synth.outlier.number );
end

end

function [X,y ] = Outliers1( X,y,Outliernum )
    nTasks = length(y.trn);
    for i = 1:nTasks 
        index  = randperm(length(y.trn{1}),Outliernum);
        y.trn{i}(index,:) = y.trn{i}(index,:)*10000;
        X.trn{i}(index,:) = X.trn{i}(index,:)*100;
    end
end

function [X,y ] = Outliers2( X,y,Outliernum )
    nTasks = length(y.trn);
    for i = 1:nTasks 
        index  = randperm(length(y.trn{1}),Outliernum);
        y.trn{i}(index,:) = y.trn{i}(index,:)+100;
    end
end

