IMPORT * FROM ML;
IMPORT ML.Mat;
// This function multiply all the elements of the matrix to the value "v".


EXPORT Mul_Mat_Num(DATASET(ML.Mat.Types.Element) l, REAL8 v) := FUNCTION

ML.Mat.Types.Element Mu(l le) := TRANSFORM
		SELF.value := le.value * v;
		SELF := le;
	END;
	
R := PROJECT(l, Mu(LEFT)); //multiply each matrix element by v

RETURN R;

END;