﻿/*
Code in R:
B <- matrix(c(5,10.753,3.8275,-5.1651,-0.45723,-65.361,87.567,
10,36.678,-6.8827,-19.267,15.106,21.841,373.17,
15,-45.177,4.0883,23.064,133.31,-40.738,973.73,
20,17.243,9.2923,-40.117,-49.066,-16.075,-118.86,
25,6.3753,2.7867,-2.6924,-93.59,26.534,-464.71,
30,-26.154,5.8978,-34.765,-20.478,49.876,-2.7761,
35,-8.6718,4.1432,-50.289,98.854,-53.647,1048.2,
40,6.8525,-1.7296,-102.31,31.401,60.512,646.22,
45,71.568,1.6751,18.648,24.04,31.687,926.5,
50,55.389,-4.4875,-9.8814,51.891,-3.2575,1110.7,
55,-26.998,5.0639,-40.726,-47.105,-9.3706,14.341,
60,60.698,-6.5383,-40.349,3.173,-10.445,839.86,
65,14.508,-6.0926,27.714,3.4263,-14.549,678.47,
70,-1.2611,-4.6141,19.585,-18.101,1.1062,493.09,
75,14.295,-16.782,42.063,27.109,2.4619,942.08,
80,-4.0993,8.1988,109.88,-4.0917,39.651,674.38,
85,-2.4829,1.8536,-462.52,6.3303,73.295,806.71,
90,29.794,-4.3031,2.7143,16.88,22.412,1079.5,
95,28.181,7.8107,-100.44,7.2777,-10.066,1045.3,
100,28.344,-9.7556,-6.0129,-34.527,30.009,769.99
),nrow = 20, ncol = 7, byrow=TRUE);

Y <- B[, 7];
X1 <- B[, 1];
X2 <- B[, 2];
X3 <- B[, 3];
X4 <- B[, 4];
X5 <- B[, 5];
X6 <- B[, 6];

model <- lm(Y ~ X1 + X2 + X3 + X4 + X5 + X6);
stepmod <- step(lm(Y ~ 1), direction="forward", scope=(~ X1+X2+X3+X4+X5+X6));
summary(stepmod);

Output :
       Df Sum of Sq     RSS    AIC
+ X5    1   2174039 1904029 233.28
+ X1    1   1094784 2983285 242.26
+ X3    1    515942 3562126 245.80
+ X2    1    407424 3670645 246.40
<none>              4078069 246.51
+ X4    1     17137 4060932 248.42
+ X6    1      7146 4070922 248.47

Step:  AIC=233.28
Y ~ X5

       Df Sum of Sq     RSS    AIC
+ X1    1   1579324  324706 199.90
+ X2    1    621696 1282333 227.37
+ X3    1    358683 1545347 231.10
+ X6    1    190611 1713418 233.16
<none>              1904029 233.28
+ X4    1     22470 1881559 235.04

Step:  AIC=199.9
Y ~ X5 + X1

       Df Sum of Sq    RSS     AIC
+ X2    1    324270    436  69.631
+ X3    1     42690 282015 199.080
<none>              324706 199.899
+ X4    1      4584 320121 201.614
+ X6    1       121 324584 201.891

Step:  AIC=69.63
Y ~ X5 + X1 + X2

       Df Sum of Sq    RSS    AIC
+ X3    1    72.025 363.81 68.018
<none>              435.84 69.631
+ X6    1    39.848 395.99 69.713
+ X4    1     0.640 435.20 71.601

Step:  AIC=68.02
Y ~ X5 + X1 + X2 + X3

       Df Sum of Sq    RSS    AIC
+ X6    1    44.010 319.80 67.439
<none>              363.81 68.018
+ X4    1     0.632 363.18 69.983

Step:  AIC=67.44
Y ~ X5 + X1 + X2 + X3 + X6

       Df Sum of Sq    RSS    AIC
<none>              319.80 67.439
+ X4    1    12.382 307.42 68.650

Call:
lm(formula = Y ~ X5 + X1 + X2 + X3 + X6)

Residuals:
    Min      1Q  Median      3Q     Max 
-7.7587 -3.0863 -0.9634  3.2965  6.8206 

Coefficients:
            Estimate Std. Error t value Pr(>|t|)    
(Intercept)  3.54572    2.41905   1.466   0.1648    
X5           7.77811    0.02374 327.671   <2e-16 ***
X1           9.01350    0.04166 216.373   <2e-16 ***
X2           4.53922    0.04088 111.047   <2e-16 ***
X3           0.33253    0.18208   1.826   0.0892 .  
X6           0.04706    0.03390   1.388   0.1868    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 4.779 on 14 degrees of freedom
Multiple R-squared:  0.9999,    Adjusted R-squared:  0.9999 
F-statistic: 3.57e+04 on 5 and 14 DF,  p-value: < 2.2e-16

*/   
	 
	 
	 IMPORT ML;
	 
   value_record := RECORD
   UNSIGNED rid;
   UNSIGNED X_1;
   REAL X_2;
	 REAL X_3;
	 REAL X_4;
	 REAL X_5;
	 REAL X_6;
   REAL Y;
   END;
   d := DATASET([{1,5,10.753,3.8275,-5.1651,-0.45723,-65.361,87.567},
		{2,10,36.678,-6.8827,-19.267,15.106,21.841,373.17},
		{3,15,-45.177,4.0883,23.064,133.31,-40.738,973.73},
		{4,20,17.243,9.2923,-40.117,-49.066,-16.075,-118.86},
		{5,25,6.3753,2.7867,-2.6924,-93.59,26.534,-464.71},
		{6,30,-26.154,5.8978,-34.765,-20.478,49.876,-2.7761},
		{7,35,-8.6718,4.1432,-50.289,98.854,-53.647,1048.2},
		{8,40,6.8525,-1.7296,-102.31,31.401,60.512,646.22},
		{9,45,71.568,1.6751,18.648,24.04,31.687,926.5},
		{10,50,55.389,-4.4875,-9.8814,51.891,-3.2575,1110.7},
		{11,55,-26.998,5.0639,-40.726,-47.105,-9.3706,14.341},
		{12,60,60.698,-6.5383,-40.349,3.173,-10.445,839.86},
		{13,65,14.508,-6.0926,27.714,3.4263,-14.549,678.47},
		{14,70,-1.2611,-4.6141,19.585,-18.101,1.1062,493.09},
		{15,75,14.295,-16.782,42.063,27.109,2.4619,942.08},
		{16,80,-4.0993,8.1988,109.88,-4.0917,39.651,674.38},
		{17,85,-2.4829,1.8536,-462.52,6.3303,73.295,806.71},
		{18,90,29.794,-4.3031,2.7143,16.88,22.412,1079.5},
		{19,95,28.181,7.8107,-100.44,7.2777,-10.066,1045.3},
		{20,100,28.344,-9.7556,-6.0129,-34.527,30.009,769.99}],value_record);
   	ML.ToField(d,o);
	X := O(Number IN [1, 2, 3, 4, 5, 6]); // Pull out the X
  Y := O(Number = 7); // Pull out the Y
	vars := DATASET([{1},{2},{4}], {UNSIGNED4 number});
	modelf := ML.StepRegression.ForwardRegression(X, Y);
	modelb := ML.StepRegression.BackwardRegression(X, Y);
	modelbi := ML.StepRegression.BidirecRegression(X, Y, vars);
	OUTPUT(modelf.Steps, NAMED('ForwardSteps'));
	OUTPUT(modelf.BestModel.betas, NAMED('ForwardBestBetas'));
	OUTPUT(modelb.Steps, NAMED('BackwardSteps'));
	OUTPUT(modelb.BestModel.betas, NAMED('BackwardBestBetas'));
	OUTPUT(modelbi.Steps, NAMED('BidirecSteps'));
	OUTPUT(modelbi.BestModel.betas, NAMED('BidirecBestBetas'));
	