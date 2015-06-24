IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat;
IMPORT * FROM ML.Types;

EXPORT roots (REAL8 a, REAL8 b, REAL8 c) := FUNCTION
  DELTA := (b*b)-(4*a*c);
  root_DELTA_Zero := (-1*b)/(2*a);
  Result_DELTA_Zero := DATASET([
      {1,1,1},
      {2,1,root_DELTA_Zero}],
      Types.NumericField);
  SQRTDELTA := SQRT (DELTA);
  root1 := (-1*b + SQRTDELTA) / (2*a);
  root2 := (-1*b - SQRTDELTA) / (2*a);
  Result_DELTA_Positive := DATASET([
      {1,1,1},
      {2,1,root1},
      {3,1,root2}],
      Types.NumericField);
  Result_DELTA_Negative := DATASET([
      {1,1,0}],
      Types.NumericField);
  finalresult := IF (DELTA=0, Result_DELTA_Zero, IF(DELTA>0, Result_DELTA_Positive, Result_DELTA_Negative) );

RETURN finalresult;
END;