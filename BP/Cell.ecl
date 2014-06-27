IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;

EXPORT Cell(DATASET($.M_Types.IDRec) P,DATASET($.M_types.IDMatRec) M) := FUNCTION

$.M_Types.CellIDMatRec ParentLoad($.M_Types.IDRec L) := TRANSFORM
    SELF.NumRows := 0;
		SELF.cellMat := [];
    SELF := L;
END;
//Ptbl := TABLE(NamesTable,DenormedRec);
Ptbl := PROJECT(P,ParentLoad(LEFT));



$.M_Types.CellIDMatRec DeNormThem($.M_Types.CellIDMatRec L, $.M_types.IDMatRec R, INTEGER C) := TRANSFORM
    SELF.NumRows := C;
    SELF.cellMat := L.cellMat + R;
    SELF := L;
END;

Result := DENORMALIZE(Ptbl, M,
				    LEFT.id = RIGHT.id,
				    DeNormThem(LEFT,RIGHT,COUNTER));
						
						
Final_Result := PROJECT(Result,$.M_Types.CellMatRec) ;


RETURN Final_Result;
END;