
IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;

//This function applies back propagation algorithm
//input data is all samples with independent attributes in NumericField format
//y is the label of samples in NumericField format
// IntParam is thw initialize weight and bias values which actually show the astructure of the network as well
//Intparam is in CellMatRec format which is actually {id, matrix}. as this dataset includes both bias and wights values
//id field is used to seprate bias matrices from weight matrices. If the network has n+1 layaer, id=1 to id =n shows the id shows
//the bias matrix for layer number id+1 and for id=n+1 to 2n the id shows weight between layer number id-n and id-n+1
EXPORT BP ( DATASET(ML.Types.NumericField) input_data, DATASET(ML.Types.NumericField) y, DATASET ($.M_Types.CellMatRec) IntParam, REAL8 LAMBDA, REAL8 ALPHA, UNSIGNED LoopNum):= FUNCTION

dTmp := ML.Types.ToMatrix (input_data);
d := Ml.Mat.Trans(dTmp);
groundTruth:= $.ToGroundTruth (y); 

Updated_Param := $.GradDesLoop (  d, y,param,  LAMBDA,  ALPHA,  1).GDIterations;

RETURN Updated_Param;

END;