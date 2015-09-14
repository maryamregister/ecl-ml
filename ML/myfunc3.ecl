// function [f,g] = myfun(x)
// f = sum(x^2);
// g = 2*x; 
// end

IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat;
IMPORT * FROM ML.Types;
emptyC := DATASET([], Types.NumericField);
// x: parameters
EXPORT myfunc3 ( DATASET(Types.NumericField) x, DATASET(Types.NumericField) CostFunc_params=emptyC, DATASET(Types.NumericField) TrainData=emptyC , DATASET(Types.NumericField) TrainLabel=emptyC) := FUNCTION
Types.NumericField x2 (x l):= TRANSFORM
  SELF.value := l.value*l.value;
  SELF := l;
END;
Types.NumericField grad (x l):= TRANSFORM
  SELF.value := 2*l.value;
  SELF := l;
END;
x_2 := PROJECT (x,x2(LEFT));
grad_x_2 := PROJECT (x,grad(LEFT));
fvalue :=  SUM(x_2,x_2.value);
result := grad_x_2 + DATASET([{MAX(grad_x_2,id)+1,1,fvalue}], Types.NumericField);
RETURN x_2;
END;
