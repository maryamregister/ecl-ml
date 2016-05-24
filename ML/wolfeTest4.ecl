
IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat;
IMPORT * FROM ML.Types;
IMPORT PBblas;
IMPORT std.system.Thorlib;
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

   
   
   
   
   
x_test := DATASET(
[{1,1,-0.5610},{2,1,-1.6323},{3,1,-1.3247},{4,1,-1.6170},{5,1,-1.5600}],Types.NumericField);


t_test := 1;
 
   
   
   
   
   d_test := DATASET(
[{1,1,-1.9738},{2,1,-0.2270},{3,1,-0.8289},{4,1,-0.2577},{5,1,-0.3712 }],Types.NumericField);

  f_test :=   10.5011;
  gtd_test :=   -1.8513;
     
  
    
   
    
  g_test := DATASET(
[{1,1,0.8467},{2,1, -0.0615},{3,1,0.2437},{4,1,-0.0462},{5,1,0.0107}],Types.NumericField);

  
   
   
   
   
    
   
x := DATASET(
[{1,1,1},{2,1,20},{3,1,45},{4,1,90},{5,1,2}],Types.NumericField);

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
//LP_up := Optimization2 (0, 0, 0, 0).lbfgsUpdate ( s+y, x_2,x_2 , 4, param_map, 1);
// output(x_2,named('yyyy'));
// output(g,named('ssss'));
//OUTPUT(LP_up);
MF := Optimization2 (0, 0, 0, 0).MinFUNC_4(xdist,param_map, 5, 10, 0.00001, 0.000000001,  1000, 3, 0, 0, 0,0) ;
//OUTPUT(MF);

// iteration number 6 does not produce the right value for t, check it out


x_test_dist := ML.DMat.Converted.FromNumericFieldDS(x_test, param_map);
d_test_dist := ML.DMat.Converted.FromNumericFieldDS(d_test, param_map);
g_test_dist :=  ML.DMat.Converted.FromNumericFieldDS(g_test, param_map);
//WolfeLineSearch4( cccc, x,  param_map, param_num,  t, d,  f,  g,  gtd,  c1=0.0001,  c2=0.9,  maxLS=25,  tolX=0.000000001)
WResult_test := Optimization2 (0, 0, 0, 0).WolfeLineSearch4( 1, x_test_dist,  param_map, P_num,  t_test, d_test_dist,  f_test,  g_test_dist,  gtd_test, 0.0001,  0.9,  25,  0.000000001);
//OUTPUT(WResult_test);
//output (param_map)
//EXPORT MinFUNC_4( x0,  param_map , param_num,  MaxIter = 100,  tolFun = 0.00001,  TolX = 0.000000001,  maxFunEvals = 1000,  corrections = 100, prows=0, pcols=0, Maxrows=0, Maxcols=0) := FUNCTION
emptyL := DATASET([], Layout_Part);
MF2 := Optimization4 (0, 0, 0, 0).MinFUNC_4(xdist,param_map,emptyC,emptyL,emptyL,myfunc4, 5,10, 0.00001, 0.000000001,  1000, 3, 0, 0, 0,0) ;


 SumProduct (DATASET(Layout_Part) inp1, DATASET(Layout_Part) inp2) := FUNCTION

      Product(REAL8 val1, REAL8 val2) := val1 * val2;
      Elem := {REAL8 v};  //short-cut record def
      Elem hadPart(Layout_Part xrec, Layout_Part yrec) := TRANSFORM //hadamard product
        elemsX := DATASET(xrec.mat_part, Elem);
        elemsY := DATASET(yrec.mat_part, Elem);
        new_elems := COMBINE(elemsX, elemsY, TRANSFORM(Elem, SELF.v := Product(LEFT.v,RIGHT.v)));
        SELF.v :=  SUM(new_elems,new_elems.v);
      END;

      prod := JOIN(inp1, inp2, LEFT.partition_id=RIGHT.partition_id, hadPart(LEFT,RIGHT), FULL OUTER, LOCAL);
      RETURN prod;
    END;//END SumProduct
    
    //LB2 := Optimization4 (0, 0, 0, 0).lbfgs_4(g,s,y,1);
    WResult2 := Optimization4 (0, 0, 0, 0).WolfeLineSearch4( 1, xdist,emptyC,emptyL,emptyL,myfunc4, P_num,  t, ddist,  f,  g,  gtd, 0.0001,  0.9,  25,  0.000000001);
    
    //WResult_test2 := Optimization4 (0, 0, 0, 0).WolfeLineSearch4( 1, x_test_dist,  emptyC,emptyL,emptyL,myfunc4, P_num,  t_test, d_test_dist,  f_test,  g_test_dist,  gtd_test, 0.0001,  0.9,  25,  0.000000001);
OUTPUT(MF2);
hrec := RECORD 
      REAL8 h ;//hdiag value
    END;
 // hproj := PROJECT(MF2, TRANSFORM(hrec, SELF.h:=LEFT.h));
     // hdup := dedup(hproj, hproj.h,LOCAL);
//OUTPUT(hproj);
//OUTPUT(hdup);