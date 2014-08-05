IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;

EXPORT M_Types := MODULE

EXPORT MatRecord := ML.Mat.Types.Element;

EXPORT VecRecord := ML.Mat.Types.VecElement;

EXPORT IDMatRec := RECORD
UNSIGNED8  id;
MatRecord;
END;


EXPORT CellIDMatRec := RECORD 
UNSIGNED8  id;
INTEGER1 NumRows;
DATASET(IDMatRec) cellMat {MAXCOUNT(100)};
END;

EXPORT CellMatRec := RECORD 
UNSIGNED8  id;
DATASET(MatRecord) cellMat {MAXCOUNT(100)};
END;

EXPORT CellMatRec2 := RECORD 
UNSIGNED8  id;
DATASET(MatRecord) cellMat1 {MAXCOUNT(100)};
DATASET(MatRecord) cellMat2 {MAXCOUNT(100)};
END;

EXPORT IDRec := RECORD
    UNSIGNED8  id;
END;

EXPORT IDNUMRec := RECORD
    UNSIGNED8  id;
		UNSIGNED8  Num;
END;


END;