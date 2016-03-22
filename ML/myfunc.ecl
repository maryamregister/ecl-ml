IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat;
IMPORT * FROM ML.Types;
IMPORT PBblas;
Layout_Cell := PBblas.Types.Layout_Cell;
Layout_Part := PBblas.Types.Layout_Part;

 EXPORT myfunc ( DATASET(Layout_Part) x, PBblas.IMatrix_Map xmap,UNSIGNED p) := FUNCTION
PBblas.Types.value_t c_cos(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := cos(v);
PBblas.Types.value_t sin_3(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := sin(v)+3;
g := PBblas.Apply2Elements (xmap, x, c_cos);
x_3 := PBblas.Apply2Elements (xmap, x, sin_3);




    Layout_Cell gen(UNSIGNED4 c, UNSIGNED4 NumRows, REAL8 v) := TRANSFORM
      SELF.x := c;
      SELF.y := 1;
      SELF.v := v;
     END;
     onesmap := xmap;
     ones := DATASET(p, gen(COUNTER, p, 1.0));
     onesdist := DMAT.Converted.FromCells(onesmap, ones);
     f_map := PBblas.Matrix_Map(1,1,1,1);
     f := PBblas.PB_dgemm(True, FALSE, 1,xmap, x_3, onesmap, onesdist, f_map);
RETURN  PBblas.MU.TO(g,1)+PBblas.MU.TO(f,2);
END;

// a := DATASET ([{1,1,1},{2,1,3},{3,1,4}], Layout_Cell);
// OUTPUT(a);
// amap := PBblas.Matrix_Map(3,1,3,1);
// adist := DMAT.Converted.FromCells(amap, a);
// output(myfunc(adist,amap,3));