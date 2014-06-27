IMPORT * FROM ML;
IMPORT ML.Mat;
// This function multiply all the elements of the matrix to the value "v".

MatRecord := ML.Mat.Types.Element;

VecRecord := ML.Mat.Types.VecElement;
MatIDRec := RECORD
UNSIGNED8  id;
MatRecord;
END; 


EXPORT Mul_Mat_Num_2(DATASET(MatIDRec) l, REAL8 v) := FUNCTION

MatIDRec Mu(l le) := TRANSFORM
		SELF.value := le.value * v;
		SELF := le;
	END;
	
R := PROJECT(l, Mu(LEFT)); //multiply each matrix element by v

RETURN R;

END;