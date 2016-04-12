
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

x:= Ones_Vec;
// x := DATASET(
// [{1,1,100},{2,1,20},{3,1,90},{4,1,80},{5,1,90}],Types.NumericField);

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


// output(x,named('xxxx'));
// output(d,named('dddd'));

// output(xdist,named('xdist'));
// output(ddist,named('ddist'));
// output(CostGrad_new,named('CostGrad_new'));

 // output(g, named('ggg'));
 // output(f,named('fff'));
// output(gtd,named('gtd'));

output(WResult,named('WResult'));



    