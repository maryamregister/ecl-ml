
IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat;
IMPORT * FROM ML.Types;
IMPORT PBblas;
Layout_Cell := PBblas.Types.Layout_Cell;
Layout_Part := PBblas.Types.Layout_Part;

emptyC := DATASET([], Types.NumericField);
//x=[15, 80, 40, 39];
// x := DATASET([
// {1, 1, 0.01},
// {2,1,0.01},
// {3,1,0.01},
// {4,1,0.01}],
// Types.NumericField);


//New Vector Generator
      Types.NumericField gen(UNSIGNED4 c, UNSIGNED4 NumRows) := TRANSFORM
        SELF.id := c;
        SELF.number := 1;
        SELF.value := 100;
      END;
      
      Types.NumericField gen10(UNSIGNED4 c, UNSIGNED4 NumRows) := TRANSFORM
        SELF.id := c;
        SELF.number := 1;
        SELF.value := 0.1;
      END;
      //Create Ones Vector for the calculations in the step fucntion
      Ones_Vec := DATASET(5, gen(COUNTER, 1));
      ten_vec  := DATASET(5, gen10(COUNTER, 1));

//OUTPUT(ones_vec);
//OUTPUT(ten_vec);

//x:= Ones_Vec;
x := DATASET(
[{1,1,100},{2,1,20},{3,1,90},{4,1,80},{5,1,90}],Types.NumericField);

x2 := DATASET(
[{1,1,10},{2,1,20},{3,1,2},{4,1,87},{5,1,40}],Types.NumericField);

x3 := DATASET(
[{1,1,1},{2,1,2},{3,1,3},{4,1,4},{5,1,5}],Types.NumericField);

x4 := DATASET(
[{1,1,78},{2,1,200},{3,1,30},{4,1,56},{5,1,90}],Types.NumericField);

d:= ten_vec;

t:=1;

//d=[0.1, 0.4, -0.2, 0.8];
// d := DATASET([
// {1, 1, 10},
// {2,1,10},
// {3,1,10},
// {4,1,10}],
// Types.NumericField);

P_num := MAX (x,id);
ExtractGrad (DATASET(PBblas.Types.MUElement) inp) := FUNCTION
      RETURN PBblas.MU.FROM(inp,1); 
    END;
    
    ExtractCost (DATASET(PBblas.Types.MUElement) inp) := FUNCTION
      inp2 := inp (no=2);
      RETURN inp2[1].mat_part[1]; 
    END;
  
param_map := PBblas.Matrix_Map(P_num,1,3,1);

xdist := ML.DMat.Converted.FromNumericFieldDS(x, param_map);
x2dist := ML.DMat.Converted.FromNumericFieldDS(x2, param_map);
x3dist := ML.DMat.Converted.FromNumericFieldDS(x3, param_map);
x4dist := ML.DMat.Converted.FromNumericFieldDS(x4, param_map);


ddist := ML.DMat.Converted.FromNumericFieldDS(d, param_map);


CostGrad_new := myfunc (  xdist, param_map, P_num);

 g := ExtractGrad (CostGrad_new);

 f := ExtractCost (CostGrad_new);

//gtd = g'*d;
 one_map := PBblas.Matrix_Map(1,1,1,1);

gtddist := PBblas.PB_dgemm(TRUE, FALSE, 1.0,param_map, g,param_map, ddist,one_map );
gtd := gtddist[1].mat_part[1];


//WolfeLineSearch4( cccc, x,  param_map, param_num,  t, d,  f,  g,  gtd,  c1=0.0001,  c2=0.9,  maxLS=25,  tolX=0.000000001)
WResult := Optimization2 (0, 0, 0, 0).WolfeLineSearch4( 1, xdist,  param_map, P_num,  t, ddist,  f,  g,  gtd, 0.0001,  0.9,  25,  0.000000001);
//lbfgs_4 (DATASET(Layout_Part) g, DATASET(PBblas.Types.MUElement) s, DATASET(PBblas.Types.MUElement) y, REAL8 Hdiag, PBblas.IMatrix_Map param_map)
PBblas.Types.value_t h_mul(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := v * v;
PBblas.Types.value_t h_sin(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := SIN(v);
x_2 := PBblas.Apply2Elements(param_map, xdist, h_mul);
g_2 := PBblas.Apply2Elements(param_map, g, h_mul);
g_3 := PBblas.Apply2Elements(param_map, g, h_sin);
s := PBblas.MU.TO(xdist,1)+ PBblas.MU.TO(x2dist,2)+PBblas.MU.TO(x3dist,3)+ PBblas.MU.TO(x4dist,4);
y := PBblas.MU.TO(g,5)+ PBblas.MU.TO(g_2,6)+PBblas.MU.TO(g_3,7)+PBblas.MU.TO(xdist,8);
LB := Optimization2 (0, 0, 0, 0).lbfgs_4(g,s,y,1,param_map);
output(y,named('yyyy'));
output(s,named('ssss'));
//OUTPUT(LB);