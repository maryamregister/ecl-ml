IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $; 

//This can be used to update any matrix (oldM) based on it's gradient values (gradM)
//the only different between this module and the UpdateWB module is that this one applies on 
// just one matrix, but UpdateWB applies on more than juts one matrix by providing cellMat structure
//so this fucntion is useful in SOFTMax calssifier which we need to just update one matrix (THETA) and the UpdateWB
//is useful for updating wight and bias matrices in back propagation algorithm which we want to update more than one
//matrices

EXPORT UpdateWB_Mat (DATASET($.M_Types.MatRecord) oldM, DATASET($.M_Types.MatRecord) gradM, REAL8 ALPHA) := MODULE

EXPORT Regular := FUNCTION // regular update which is M_new = M_old - (ALPHA*Mgrad)

AlphaGradM := Ml.Mat.each.ScalarMul (gradM, ALPHA); // ALPHA multiplyis by M grad matrix 
UpdatedM := ML.Mat.Sub (oldM,AlphaGradM); 

RETURN UpdatedM;

END;

END;