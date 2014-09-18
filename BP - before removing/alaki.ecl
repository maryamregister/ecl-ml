
IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;
a:= Ml.Mat.Each.ScalarMul ($.RandMat (10,800),0.005);

 OUTPUT  (a, NAMED ('a'));