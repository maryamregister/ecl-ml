IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat;
IMPORT * FROM ML.Types;
//return the roots of the polynomiala a*x^2+b*x+c
//the retunr structure is a numericfield dataset
// id =1 : whether the roots are imaginary or real values (-1/1)
//id = {1,2} : the actual root values 
EXPORT roots (REAL8 a, REAL8 b, REAL8 c) := FUNCTION
  DELTA := (b*b)-(4*a*c);
  root_DELTA_Zero := (-1*b)/(2*a);
  Result_DELTA_Zero := DATASET([
      {1,1,1},
      {2,1,root_DELTA_Zero},
      {3,1,root_DELTA_Zero}],
      Types.NumericField);
  SQRTDELTA := SQRT (DELTA);
  SQRTDELTA_ := SQRT (-1*DELTA);
  root1 := (-1*b + SQRTDELTA) / (2*a);
  root2 := (-1*b - SQRTDELTA) / (2*a);
  root1_ := (-1*b + SQRTDELTA_) / (2*a);
  root2_ := (-1*b - SQRTDELTA_) / (2*a);
  Result_DELTA_Positive := DATASET([
      {1,1,1},
      {2,1,root1},
      {3,1,root2}],
      Types.NumericField);
  Result_DELTA_Negative := DATASET([
      {1,1,-1},
      {1,1,root1_},
      {1,1,root2_}],
      Types.NumericField);
  finalresult := IF (DELTA=0, Result_DELTA_Zero, IF(DELTA>0, Result_DELTA_Positive, Result_DELTA_Negative) );

RETURN finalresult;
END;