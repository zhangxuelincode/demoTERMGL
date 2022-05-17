code for "Robust Variable Structure Discovery Based on Tilted Empirical Risk Minimization"

*********************************************************
* Recommenedation :                                     *
* This code is designed to work with Matlab 2019a       *
*********************************************************

------------------------------------------------------------------------------------------------------------------------------------------------
DESCRIPTION :

This code provides an efficient and robust approach to learn the variable structure automarically without prior information. 
Our framework is based on a continuous bilevel formulation of the problem of learning the variable structure.
------------------------------------------------------------------------------------------------------------------------------------------------
Experimental setting :

TERM hyperparameter "t":	param.TERM		           =        -0.001;
Noise Percentage:		      synth.noise.level		     =           0.5; 
Outlier Numbers:		      synth.outlier.number		 =             0; 
Outlier Deviation:		    synth.outlier.deviation	 =             5;
HQ Loops:			            param.inner.modalIter    =             3;
Inner Loops:		          param.inner.itermax      =           200;
Outer Loops:		          param.outer.itermax      =          2000;
------------------------------------------------------------------------------------------------------------------------------------------------
Training :

To train the model in the paper, run Main.m;
Besides, we provide a demo to help to get start with our code.
-----------------------------------------------------------------------------------------------------------------------------------------------
Evaluation :

To evaluate the model in the paper, run evaluation.m;
We provide four criteria mentioned in our paper, inlcuding average square error (ASE), true deviation (TD), width of prediction intervals (WPI) and sample coverage probability (SCP).
