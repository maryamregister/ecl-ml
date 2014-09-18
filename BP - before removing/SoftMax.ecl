IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;

//d is input data
//y is labels : c(i,j)=1 IFF the lable of the jth sample is i
//theta is the softmax parameters which it's size is num_classes * num_features(input size)
//LAMBDA is wight decay parameter
EXPORT SoftMax(DATASET(ML.Types.NumericField) input_data, DATASET(ML.Types.NumericField) y, REAL8 LAMBDA, REAL8 ALPHA,DATASET($.M_Types.MatRecord) IntTHETA , UNSIGNED LoopNum ) := Function

dTmp := ML.Types.ToMatrix (input_data);
d := Ml.Mat.Trans(dTmp);
groundTruth:= $.ToGroundTruth (y); 

Update_param := $.SoftMax_matrixinput (d, groundTruth, LAMBDA, ALPHA, IntTHETA, LoopNum ).SoftMaxGradIterations;

RETURN Update_param;

END;
