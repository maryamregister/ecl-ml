﻿IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $; 

//This can be used to update weight and Bias Values 
// the input WB is actually the parameter you want to be updated (W or B)
//WB is ofcourse in the cellmat format [{id matrix}]
//APLHA is the learning rate to update W it should be called like UpdateWB(W,Wgrad,alpha).Regular
EXPORT UpdateWB (DATASET($.M_Types.CellMatRec) WB, DATASET($.M_Types.CellMatRec) WBgrad, REAL8 ALPHA) := MODULE




EXPORT Regular := FUNCTION // regular update which is w_new = w_old - (ALPHA*Wgrad)


$.M_Types.CellMatRec Jupd (WB l, WBgrad r ) := TRANSFORM
//either use the next two lines or the third line
		// AlphaWg := Ml.Mat.each.ScalarMul (r.cellmat, ALPHA); // ALPHA multiplyis by weight grad matrix (it can bias grad matrix too)
		// SELF.cellmat := ML.Mat.Sub (l.cellmat,AlphaWg); 
		SELF.cellmat := $.UpdateWB_Mat(l.cellmat, r.cellmat, ALPHA).Regular;
		SELF := l;
END;

UpdatedWB := JOIN (WB, WBgrad, LEFT.id = RIGHT.id , Jupd (LEFT, RIGHT));


RETURN UpdatedWB;

END;

END;