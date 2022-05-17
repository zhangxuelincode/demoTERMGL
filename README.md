# Demo for "Robust Variable Structure Discovery Based on Tilted Empirical Risk Minimization"



## Requirements

* Recommenedation :                                                         

* This code is designed to work with Matlab 2019a       

  

## DESCRIPTION 

This code provides an efficient and robust approach to learn the variable structure automatically without prior information. 

Our framework is based on a continuous bilevel formulation of the problem of learning the variable structure.



## Experimental setting

All parameters are  set in the "demo.m" file, here we show the key parameters:

TERM Hyperparameter "t":	param.TERM		              =           -0.001;
Samples For Each Task:		  param.N                             =           50;
Dimension:		                       param.P                              =           50;
Total Tasks:		                       param.T                              =           500;
Total Groups:		                    param.L                             =           50;
Regularization Parameter:	 param.lambda                  =         0.01;
Noise Types:	                         synth.noise.distrib           =          'normal';
Noise Percentage:		          synth.noise.level		       =           0.5; 
Outlier Numbers:		           synth.outlier.number	   =             0; 
Outlier Deviation:		           synth.outlier.deviation	 =             5;
HQ Loops:			                    param.inner.modalIter    =             3;
Inner Loops:		                    param.inner.itermax        =           200;
Outer Loops:		                   param.outer.itermax        =          2000;



## Training

To train the model in the paper, run Main.m;
Besides, we provide a demo to help to get start with our code.



## Evaluation

To evaluate the model in the paper, run evaluation.m;
We provide four criteria mentioned in our paper, inlcuding average square error (ASE), true deviation (TD), width of prediction intervals (WPI) and sample coverage probability (SCP).



## Variable structure

For example, if the variables satisfy:

![image-20220517151659130](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220517151659130.png)

Then the corresponding structure is:

![image-20220517151722729](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220517151722729.png)



# Simulation results

 We compare the proposal method (named TERMGL) with the baseline BiGL (see [1]). Besides, we also compare with some sparse learning methods (Lasso & Group Lasso) and some robust learning methods (MCC [] & Huber regression & TERM).

We randomly add four types of noises (Gaussian & Student & Exponential & Chi-square noises) and five levels of percentages of outliers (0% & 10% & 20% & 30% & 40% outliers) to compare and highlight the robustness of our proposal.

TERMGL has better performamce on prediction (see tables 2-6 in the paper) and variable recovery (see figures 1-5 in the paper).

For more details, please refer to the paper (submitting).

























