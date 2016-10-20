IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat;
IMPORT * FROM ML.Types;
IMPORT PBblas;
Layout_Cell := PBblas.Types.Layout_Cell;
Layout_Part := PBblas.Types.Layout_Part;
emptyC := DATASET([], Types.NumericField);
emptyL := DATASET([], Layout_Part);
 EXPORT myfunc5 ( DATASET(Layout_Part) x, DATASET(Types.NumericField) CostFunc_params=emptyC, DATASET(Layout_Part) TrainData=emptyL , DATASET(Layout_Part) TrainLabel=emptyL) := FUNCTION
Elem := {REAL8 v};  //short-cut record def

Elem applyF(Elem e) := TRANSFORM
    SELF.v := SIN(e.v)+3;
  END;

Elem applyF2(Elem e) := TRANSFORM
    SELF.v := COS(e.v);
  END;


Elem cost_cal(Layout_Part lr) := TRANSFORM
    elems := DATASET(lr.mat_part, Elem);
    new_elems := PROJECT(elems, applyF(LEFT));
		s:= SUM(new_elems, new_elems.v);
    SELF.v := s;
  END;
sumdataset := PROJECT(x, cost_cal(LEFT),LOCAL);
cost := SUM(sumdataset,sumdataset.v);




Layout_Part grad_cal(Layout_Part lr) := TRANSFORM
    elems := DATASET(lr.mat_part, Elem);
    new_elems := PROJECT(elems, applyF2(LEFT));
    SELF.mat_part := SET(new_elems, v);
    SELF := lr;
  END;
grad := PROJECT(x,grad_cal(LEFT),LOCAL);


costField := DATASET ([{1,1,cost}],ML.Types.NumericField);
      one_map := PBblas.Matrix_Map(1,1,1,1);
      Cost_part_no := PBblas.MU.TO(ML.DMat.Converted.FromNumericFieldDS(costField,one_map),2);


RETURN  Cost_part_no + PBblas.MU.TO(grad,1);
END;


// a := DATASET ([{1,1,1},{2,1,3},{3,1,4}], Layout_Cell);
// OUTPUT(a);
// amap := PBblas.Matrix_Map(3,1,3,1);
// adist := DMAT.Converted.FromCells(amap, a);
// output(myfunc(adist,amap,3));