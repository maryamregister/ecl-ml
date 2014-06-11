IMPORT * FROM ML;
IMPORT ML.Mat;
// This function adds all the elements of the matrix to a specific Value.


EXPORT Add_Mat_Num(DATASET(ML.Mat.Types.Element) l, REAL8 v) := FUNCTION

ML.Mat.Types.Element Ad(l le) := TRANSFORM
		SELF.value := le.value + v;
		SELF := le;
	END;
	
RRR := PROJECT(l, Ad(LEFT)); //perfomr sigmoid function on each record

RETURN RRR;

END;