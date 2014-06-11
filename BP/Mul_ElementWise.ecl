IMPORT * FROM ML;
IMPORT ML.Mat;

// this functions multiplies two matrixs elementwise, similar to ".*" in MATLAB

EXPORT Mul_ElementWise(DATASET(ML.Mat.Types.Element) l,DATASET(ML.Mat.Types.Element) r) := FUNCTION
StatsL := ML.Mat.Has(l).Stats;
StatsR := ML.Mat.Has(r).Stats;
SizeMatch := StatsL.XMax=StatsR.XMax AND StatsL.YMax=StatsR.YMax;


ML.Mat.Types.Element Mu(l le,r ri) := TRANSFORM
    SELF.x := IF ( le.x = 0, ri.x, le.x );
    SELF.y := IF ( le.y = 0, ri.y, le.y );
	  SELF.value := le.value * ri.value; 
  END;
	
	
checkAssert := ASSERT(SizeMatch, 'Add FAILED - Size mismatch', FAIL);		
result := IF(SizeMatch, JOIN(l,r,LEFT.x=RIGHT.x AND LEFT.y=RIGHT.y,Mu(LEFT,RIGHT),FULL OUTER), DATASET([], ML.Mat.Types.Element));
	RETURN WHEN(result, checkAssert);
END;
