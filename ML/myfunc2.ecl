// function [f,g] = myfun(x)
// f = sum(sin(x) + 3);
// g = cos(x); 
// end

IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat;
IMPORT * FROM ML.Types;
emptyC := DATASET([], Types.NumericField);
// x: parameters
EXPORT myfunc2 ( DATASET(Types.NumericField) x, DATASET(Types.NumericField) CostFunc_params=emptyC, DATASET(Types.NumericField) TrainData=emptyC , DATASET(Types.NumericField) TrainLabel=emptyC) := FUNCTION
Types.NumericField SinTran (x l):= TRANSFORM
  SELF.value := SIN (l.value)+3;
  SELF := l;
END;
Types.NumericField CosTran (x l):= TRANSFORM
  SELF.value := COS (l.value);
  SELF := l;
END;
sin_x := PROJECT (x,SinTran(LEFT));
cos_x := PROJECT (x,CosTran(LEFT));
fvalue :=  SUM(sin_x,sin_x.value);
result := cos_x + DATASET([{MAX(cos_x,id)+1,1,fvalue}], Types.NumericField);
RETURN result;
END;
