IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat;
IMPORT * FROM ML.Types;
IMPORT PBblas;
Layout_Cell := PBblas.Types.Layout_Cell;
Layout_Part := PBblas.Types.Layout_Part;
emptyC := DATASET([], Types.NumericField);
emptyL := DATASET([], Layout_Part);

EXPORT myfunc44 ( DATASET(PBblas.Types.Layout_Part) xalaki=emptyL, DATASET(Types.NumericField) CostFunc_pms=emptyC, DATASET(Layout_Part) trnjjd=emptyL, DATASET(Layout_Part) khkg=emptyL) := FUNCTION
      return PBblas.MU.TO(xalaki,1);
     END;