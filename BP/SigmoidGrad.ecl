IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;
// This function calculates sigmoid function for each element of the matrix


EXPORT sigmoid(DATASET(ML.Mat.Types.Element) l) := FUNCTION

ML.Mat.Types.Element sig(l le) := TRANSFORM
		SELF.value := 1 / (1 + EXP(le.value * -1));
		SELF := le;
	END;
	
P := PROJECT(l, sig(LEFT)); //perfomr sigmoid function on each record

RETURN P;

END;