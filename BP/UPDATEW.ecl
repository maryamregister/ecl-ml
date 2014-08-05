IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $; 
 
EXPORT UpdateWB (DATASET($.M_Types.CellMatRec) W, DATASET($.M_Types.CellMatRec) Wgrad, REAL8 ALPHA) := MODULE

EXPORT Regular := FUNCTION // regular update which is w_new = w - (ALPHA*Wgrad)


$.M_Types.CellMatRec Pupd (W l) := TRANSFORM

		AlphaW := Ml.Mat.each.ScalarMul (l.cellmat, ALPHA); // ALPHA multiplyis by weight matrix
		SELF.cellmat := ML.Mat.Sub (l.cellmat,AlphaW); 
		SELF := l;
END;

UpdatedW := PROJECT (w, Pupd (LEFT));


RETURN UpdatedW;

END;

END;