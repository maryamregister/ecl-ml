
IMPORT * FROM ML;
IMPORT ML.Mat;
// This function multiplies elements of a matrix with elements of a vector. If R_C=1 then the multiplication is done per 
//row, it means that the elements of row 1 are multiplied by the elemnet 1 of the vector, the elements of row 2 are multiplied
//by the element 2 of the vector and so on. If R_C=2 then the multiplication is done column-wise
// test file : 
EXPORT Mul_Mat_Vec(DATASET(ML.Mat.Types.Element) l,DATASET(ML.Mat.Types.VecElement) r, R_C) := FUNCTION

StatsL := ML.Mat.Has(l).Stats;
StatsR := ML.Mat.Has(r).Stats;
SizeMatch :=  (R_C=1 AND StatsL.XMax=StatsR.XMax) OR (R_C=2 AND StatsL.YMax=StatsR.XMax);


checkAssert := ASSERT(SizeMatch, 'Add FAILED - Size mismatch', FAIL);	

ML.Mat.Types.Element M(l le,r ri) := TRANSFORM
    SELF.x := le.x;
    SELF.y := le.y;
	  SELF.value := le.value * ri.value; 
  END;

Result :=  IF(R_C=1,JOIN(l,r,LEFT.x=RIGHT.x,M(LEFT,RIGHT)),JOIN(l,r,LEFT.y=RIGHT.x,M(LEFT,RIGHT))); 

RETURN WHEN(Result, checkAssert);


END;