IMPORT * FROM ML;
IMPORT ML.Mat;
// This function divide all the elements of the matrix by the value "v".


EXPORT Div_Mat_Num(DATASET(ML.Mat.Types.Element) l, REAL8 v) := FUNCTION

ML.Mat.Types.Element Di(l le) := TRANSFORM
		SELF.value := le.value / v;
		SELF := le;
	END;
	
R := PROJECT(l, Di(LEFT)); //multiply each matrix element by v

RETURN R;

END;