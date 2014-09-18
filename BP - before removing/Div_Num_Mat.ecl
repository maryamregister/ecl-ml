IMPORT * FROM ML;
IMPORT ML.Mat;
// This function divide a number by each elemnet of the matrix 


EXPORT Div_Num_Mat(DATASET(ML.Mat.Types.Element) l, REAL8 v) := FUNCTION

ML.Mat.Types.Element Di(l le) := TRANSFORM
		SELF.value := v/ le.value ;
		SELF := le;
	END;
	
R := PROJECT(l, Di(LEFT)); //multiply each matrix element by v

RETURN R;

END;