IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat;
IMPORT * FROM ML.Types;
IMPORT PBblas;
Layout_Cell := PBblas.Types.Layout_Cell;
Layout_Part := PBblas.Types.Layout_Part;
emptyC := DATASET([], Types.NumericField);
emptyL := DATASET([], Layout_Part);
 EXPORT myfunc_new ( DATASET(Layout_Part) x, DATASET(Types.NumericField) CostFunc_params=emptyC, DATASET(Layout_Part) TrainData=emptyL , DATASET(Layout_Part) TrainLabel=emptyL) := FUNCTION
p := CostFunc_params(id=1)[1].value;// number of parameters
p_part := CostFunc_params(id=2)[1].value;
xmap := PBblas.Matrix_Map(1,p,1,p_part);
PBblas.Types.value_t c_cos(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := cos(v);
PBblas.Types.value_t sin_3(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := sin(v)+3;
g := PBblas.Apply2Elements (xmap, x, c_cos);
x_3 := PBblas.Apply2Elements (xmap, x, sin_3);
  Elem := {PBblas.Types.value_t v};  //short-cut record def
  
  Elem dosum(Layout_Part lr) := TRANSFORM
    s:= SUM(lr.mat_part);
    SELF.v := s;
  END;
  sumdataset := PROJECT(x_3, dosum(LEFT),LOCAL);
  cost :=  SUM(sumdataset,sumdataset.v);
	costField := DATASET ([{1,1,cost}],ML.Types.NumericField);
  one_map := PBblas.Matrix_Map(1,1,1,1);
  Cost_part_no := PBblas.MU.TO(ML.DMat.Converted.FromNumericFieldDS(costField,one_map),2);
RETURN  PBblas.MU.TO(g,1) + Cost_part_no;
END;

