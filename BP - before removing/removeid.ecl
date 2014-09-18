IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;


MatRecord := ML.Mat.Types.Element;

VecRecord := ML.Mat.Types.VecElement;
MatIDRec := RECORD
UNSIGNED8  id;
MatRecord;
END;



CellMatRecord := RECORD 
UNSIGNED8  id;
INTEGER1 NumRows;
DATASET(MatIDRec) cellMat {MAXCOUNT(100)};
END;

CellMatRecord2 := RECORD 
UNSIGNED8  id;
INTEGER1 NumRows;
DATASET(MatRecord) cellMat {MAXCOUNT(100)};
END;


MatRecord RMVID (DATASET(MatIDRec) l) := FUNCTION

myrec := RECORD
l.x;
l.y;
l.value;
END;
RETURN TABLE (l,myrec);
END;


EXPORT CellMatRecord2 removeid(DATASET(CellMatRecord) le):= FUNCTION


CellMatRecord2 sig(le lef) := TRANSFORM
		SELF.cellmat := 
		SELF := lef;
	END;
	
P := PROJECT(l, sig(LEFT)); //perfomr sigmoid function on each record

RETURN P;



END;

