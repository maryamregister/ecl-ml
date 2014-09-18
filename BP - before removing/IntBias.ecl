IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;

EXPORT IntBias ( DATASET($.M_Types.IDNUMRec) NodeNums) := FUNCTION



//remove the first record from NodeNums and decrease  id in other records by 1. (B1 means the bias which goes to
//secod layer, thats why we reduce id valeus by one)

one := DATASET ([{1,1,0}],$.M_Types.MatRecord);

 $.M_Types.CellMatRec IntB ($.M_Types.IDNUMRec l) := TRANSFORM 

SELF.id := l.id-1;
SELF. cellMat := ML.Mat.Repmat (one, l.Num, 1);


END;

B_Rnd := NORMALIZE (NodeNums, 1, IntB(LEFT));

RETURN B_Rnd;

END;


