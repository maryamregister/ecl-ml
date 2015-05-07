IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat;
IMPORT * FROM ML.Types;
emptyC := DATASET([], Types.NumericField);

EXPORT myfunc ( DATASET(Types.NumericField) x, DATASET(Types.NumericField) CostFunc_params=emptyC, DATASET(Types.NumericField) TrainData=emptyC , DATASET(Types.NumericField) TrainLabel=emptyC) := FUNCTION
Types.NumericField tr (Types.NumericField l) := TRANSFORM
SELF.value := l.value +1;
SELF := l;
END;
xx := PROJECT (x, tr(LEFT));
mid := Max (x,id);
Cost :=  DATASET([{mid+1, 1, 509}],Types.NumericField);
RETURN  xx+cost;
END;