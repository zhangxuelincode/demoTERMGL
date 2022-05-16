clearvars;
close all;
addpath(genpath('functions'))
%% Synthesis Parameters 
param.TERM                  =    -0.001;                                   %TERM parameter t<0 
param.N                     =        50;                                   %The number of samples
param.P                     =        50;                                   %The dimension of features
param.T                     =       500;                                   %The number of tasks
param.L                     =         5;                                   %The number of groups, 1 <= L <= P
param.G                     =         2;                                   %The number of non-zero groups per task
param.sigma                 =         1;                                   %The bandwidth of kernel      
param.lambda                =       0.5;                                   %The regularization parameter
param.mu                    =     0.001;                                   %The smooth parameter for \mu/2\|\beta\|_2^2
param.alpha                 =       0.9;                                   %The confidence level 
synth.noise.distrib         =  'normal';                                   %'normal', 'chisq', 't' or 'exp';   
synth.noise.degree          =         2;                                   %The freedom of Chi-square distribution or t distribution 
synth.noise.param           =     [0 1];                                   %Mean 'param(1)' and var 'param(2)'
synth.noise.percentage      =         1;                                   %Noise Percentage
synth.outlier.number        =         0;                                   %Outlier numbers
synth.outlier.type          =         1;                                   %Outlier type: 1 for multiplying case,2 for adding case
synth.features.param        =     [0,1];                                   %mean 'param(1)' and var 'param(2)'
synth.groups.distrib        =   'equal';                                   %Group distribution
synth.design.setting        = 'general';
synth.design.mean           =         0;                            
synth.design.std            =         1;
%------------------------------------------------------------------------------------------------------------
%Data Generation
[y,X,thetastar,wstar,Xwstar ] = SynthesizeDataset( param.N, param.P, param.T, param.L, param.G,synth);  
%------------------------------------------------------------------------------------------------------------
%Parameters for Lower Level Problem
param.inner.modalIter       =                  2;                          %M:  Max-Iter of HQ 
param.inner.sigma           =        param.sigma;                          %The bandwidth of modal kernel
param.inner.nTasks          =            param.T;                          %The number of tasks
param.inner.nObs            =            param.N;                           
param.inner.nFeatures       =            param.P;
param.inner.nGroups         =            param.L;       
param.inner.tol             =               1e-6;
param.inner.itermax         =                400;                          %Q: Max-Iter of DFBB in lower level problem
param.inner.compObjective   =               true;
param.inner.saveIterates    =               true;
%------------------------------------------------------------------------------------------------------------
%Parameter For Upper Level Problem
param.outer.nTasks          =           param.T;
param.outer.nFeatures       =           param.P;
param.outer.fixedPointHG    =              true;                           %Fixed-point approach for computing the hypergradient
param.outer.projection      =         'simplex';                           %Space Theta (see also, 'posSphere' and 'box')
param.outer.itermax         =              2000;                           %Z: Max-Iter in upper level problem
param.outer.nGroups         =           param.L;                           %Number of groups
param.outer.batchSize       =                 1;                           %Batch size for stochastic optimization
param.outer.displayOnline   =              true;                           %Online display of variable structure
inner_function              = Inner_setting (param);                       %The functions in HQ-DFBB

%% ANALYSIS
%------------------------------------------------------------------------------------------------------------
% Training
[opt]                               =      MAIN(y,X,param,thetastar,inner_function);          
[error_val,error_tst,width,coverage]=      Evaluation(X,y,opt.theta,param,inner_function); 
%------------------------------------------------------------------------------------------------------------
% Evaluation
disp(['ASE:'  ,num2str(error_val)])
disp(['TD :',num2str(error_tst)])
disp(['WPI:',num2str(width)])
disp(['SCP:',num2str(coverage)])
%------------------------------------------------------------------------------------------------------------
% Structural Matrix Transformation 
Structureprocess(opt.theta)