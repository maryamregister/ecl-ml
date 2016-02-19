
IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat;
IMPORT * FROM ML.Types;
IMPORT PBblas;

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
        SELF.value := 0.09;
      END;
      
      Types.NumericField gen10(UNSIGNED4 c, UNSIGNED4 NumRows) := TRANSFORM
        SELF.id := c;
        SELF.number := 1;
        SELF.value := 0.03;
      END;
      //Create Ones Vector for the calculations in the step fucntion
      Ones_Vec := DATASET(5, gen(COUNTER, 1));
      ten_vec  := DATASET(5, gen10(COUNTER, 1));

//OUTPUT(ones_vec);
//OUTPUT(ten_vec);

x:= Ones_Vec;
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
ExtractGrad (DATASET(Types.NumericField) inp) := FUNCTION
      RETURN inp (id <= P_num);
    END;
    ExtractCost (DATASET(Types.NumericField) inp) := FUNCTION
      RETURN inp (id = (P_num+1))[1].value;
    END;
//[f g]= myfunc2(x);
/*
g := DATASET([
   { 1.0000 ,   1.0000,   0.9950},
   { 2.0000  ,  1.0000 ,  0.9950},
   { 3.0000   , 1.0000  ,  0.9950},
    {4.0000 ,   1.0000 ,  0.9950},
   { 5.0000 ,   1.0000 , 0.9950}

],Types.NumericField);

f :=  15.4992;





*/
// OUTPUT (x, NAMED ('xx'));
// OUTPUT (g, NAMED ('gg'));
// OUTPUT (f,NAMED('ff'));
// OUTPUT (d,NAMED('dd'));

CostGrad_new := myfunc2 (  x, emptyC, emptyC , emptyC);
g := ExtractGrad (CostGrad_new);
f := ExtractCost (CostGrad_new);
//gtd = d'*d;
gtdT := ML.Mat.Mul (ML.Mat.Trans(ML.Types.ToMatrix(g)),ML.Types.ToMatrix(d));
gtd := gtdT[1].value;
OUTPUT(g, named('gggg'));
OUTPUT (d, named('dddd'));
OUTPUT (gtd, named('gtd'));
//OUTPUT (gtd,NAMED('gtd'));

//WolfeLineSearch(wolfeout, x,t,d,f,g,gtd,0.0001,0.9,25,0.000000001,myfunc2,emptyC, emptyC, emptyC,0,0,0,0);
 //wolfeout := WolfeLineSearch(x,t,d,f,g,gtd,0.0001,0.9,3,0.000000001,emptyC, emptyC, emptyC,myfunc2,0,0,0,0);
//OUTPUT(wolfeout);

WResult := Optimization2 (0, 0, 0, 0).WolfeLineSearch3(1,x,t,d,f,g, gtd,0.0001,0.9,25,0.000000001,emptyC, emptyC, emptyC,myfunc2,0,0,0,0);

OUTPUT(WResult,NAMED('WResult'));
// OUTPUT (Optimization2 (0, 0, 0, 0).wolfe_gnew_ext (WResult));
// WWWresult := Optimization (0, 0, 0, 0).WolfeOut_FromField(WResult);
// OUTPUT (WWWresult , named ('wwwresult'));
// a :=  Mat.MU.FROM (WWWresult, 1);
 // aa := p_um;
 
 
  
//OUTPUT(WWWresult,NAMED('WWWresult'));
//OUTPUT (WWWresult,, 'maryam::output1');

// funresult := myfunc2 ( x, emptyC, emptyC , emptyC);
// OUTPUT(funresult, NAMED('funresults'));

// pol := Optimization (0, 0, 0, 0).polyinterp_both (  10.0000 ,  12.8652 ,  -1.3527,100.0000  , 12.9898 ,   1.0860,10.0900, 100);
// pol2 :=  Optimization (0, 0, 0, 0).polyinterp_noboundry (10.0000  , 12.8652 ,  -1.3527,100.0000 ,  12.9898 ,   1.0860);
// pol3 := Optimization (0, 0, 0, 0).polyinterp_img ( 0 , 1.0000 , 19.0000 ,8.0000,2.0000,8 );

   // Mr :=  MinFUNCALAKI(x, myfunc2, emptyC, emptyC , emptyC, 3, 0.00001, 5, 3,0, 0, 0,0);  
   // OUTPUT(Mr);
   //MinFUNC( x0, CostFunc ,  CostFunc_params, TrainData , TrainLabel,  MaxIter = 100,  tolFun = 0.00001,  TolX = 0.000000001,  maxFunEvals = 1000,  corrections = 100, prows=0, pcols=0, Maxrows=0, Maxcols=0) := FUNCTION
   man:= Optimization2 (0, 0, 0, 0).MinFUNC3 (x, myfunc2, emptyC, emptyC , emptyC, 15, 0.00001, 0.000000001,1000, 3,0, 0, 0,0);  
   //OUTPUT(man);
   
   //MinFUNCkk(x0, CostFunc ,  CostFunc_params,  TrainData ,  TrainLabel, MaxIter = 500,  tolFun = 0.00001,  TolX = 0.000000001,  maxFunEvals = 1000,  corrections = 100, =0, =0, =0, =0) := FUNCTION
//OUTPUT (man);


//how much cost function takes time?
// costfun := myfunc2 (x ,emptyC,emptyC, emptyC);
// OUTPUT(costfun);
