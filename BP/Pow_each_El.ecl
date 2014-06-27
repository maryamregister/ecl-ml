IMPORT * FROM ML;
IMPORT ML.Mat;
// This function power all elements of the matrix to the value "v" (this is an elementwise operation)


EXPORT Pow_Each_El(DATASET(ML.Mat.Types.Element) l,REAL8 v) := FUNCTION

ML.Mat.Types.Element Po(l le) := TRANSFORM
		SELF.value := POWER (le.value,v);
		SELF := le;
	END;
	
R := PROJECT(l, Po(LEFT)); //perfomr sigmoid function on each record

RETURN R;

END;