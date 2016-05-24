IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat;
IMPORT * FROM ML.Types;
IMPORT PBblas;
Layout_Cell := PBblas.Types.Layout_Cell;
Layout_Part := PBblas.Types.Layout_Part;
emptyC := DATASET([], Types.NumericField);
EXPORT myfunc_field (DATASET(Types.NumericField) x, DATASET(Types.NumericField) CostFunc_params, DATASET(Types.NumericField) TrainData=emptyC , DATASET(Types.NumericField) TrainLabel=emptyC) := FUNCTION
  P_num := CostFunc_params[1].value;
  param_map := PBblas.Matrix_Map(P_num,1,3,1);
  xdist := ML.DMat.Converted.FromNumericFieldDS(x, param_map);
  res := myfunc (  xdist, param_map, P_num);
  RETURN 1;
END;
// myfunc ( DATASET(Layout_Part) x, PBblas.IMatrix_Map xmap,UNSIGNED p) := FUNCTION
