# Demo for "Robust Variable Structure Discovery Based on Tilted Empirical Risk Minimization"


## Requirements

* Recommenedation :                                                         

* This code is designed to work with Matlab 2019a       

  

## DESCRIPTION 

This code provides an efficient and robust approach to learn the variable structure automatically without prior information. 

Our framework is based on a continuous bilevel formulation of the problem of learning the variable structure.

The proposal is called "TERMGL", which aims to solve Group Lasso regression problems based on TERM robustly without limitations of prior structure information.


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

![structure1.png](https://s2.loli.net/2022/05/17/RkliBFmyfPpSwoN.png)

Then the corresponding structure is:

![structure2.png](https://s2.loli.net/2022/05/17/5KbjlsgekFXn2xd.png)



# Simulation results

 We compare the proposal method (named TERMGL) with the baseline BiGL (see[1]). Besides, we also compare with some sparse learning methods (Lasso & Group Lasso) and some robust learning methods (MCC[2] & Huber regression & TERM[3]).

We randomly add four types of noises (Gaussian & Student & Exponential & Chi-square noises) and five levels of percentages of outliers (0% & 10% & 20% & 30% & 40% outliers) to compare and highlight the robustness of our proposal.

TERMGL has better performamce on prediction (see tables 2-6 in the paper) and variable recovery (see figures 1-5 in the paper).

For more details, please refer to the paper (submitting).

# Reference

[1] Frecon J, Salzo S, Pontil M. Bilevel learning of the group lasso structure[J]. Advances in neural information processing systems, 2018, 31.

[2] Feng Y, Huang X, Shi L, et al. Learning with the maximum correntropy criterion induced losses for regression[J]. J. Mach. Learn. Res., 2015, 16(30): 993-1034.

[3] Li T, Beirami A, Sanjabi M, et al. Tilted empirical risk minimization[J]. arXiv preprint arXiv:2007.01162, 2020.

# Citation
If you find the proposed method and codes helpful, please add the citation:
@article{zhang2023robust,
  title={Robust variable structure discovery based on tilted empirical risk minimization},
  author={Zhang, Xuelin and Wang, Yingjie and Zhu, Liangxuan and Chen, Hong and Li, Han and Wu, Lingjuan},
  journal={Applied Intelligence},
  pages={1--22},
  year={2023},
  publisher={Springer}
}
