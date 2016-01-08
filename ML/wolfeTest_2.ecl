
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
        SELF.value := 0.01;
      END;
      
      Types.NumericField gen10(UNSIGNED4 c, UNSIGNED4 NumRows) := TRANSFORM
        SELF.id := c;
        SELF.number := 1;
        SELF.value := 10;
      END;
      //Create Ones Vector for the calculations in the step fucntion
      Ones_Vec := DATASET(5, gen(COUNTER, 1));
      ten_vec  := DATASET(5, gen10(COUNTER, 1));

//OUTPUT(ones_vec);
//OUTPUT(ten_vec);

x:= DATASET([
    {1.0000  ,  1.0000    ,0.0060},
    {2.0000  ,  1.0000   , 0.0093},
    {3.0000  ,  1.0000  ,  0.0102},
    {4.0000  ,  1.0000   , 0.0062},
    {5.0000   , 1.0000  ,  0.0061},
    {6.0000  ,  1.0000   , 0.0070},
    {7.0000  ,  1.0000  , -0.0045},
    {8.0000   , 1.0000  , -0.0065},
    {9.0000  ,  1.0000  , -0.0063},
   {10.0000   , 1.0000  , -0.0049},
  { 11.0000   , 1.0000  , -0.0040},
   {12.0000   , 1.0000  , -0.0061},
   {13.0000   , 1.0000  , -2.2153},
   {14.0000   , 1.0000  , -2.2221},
   {15.0000   , 1.0000   , 0.1399},
   {16.0000    ,1.0000  ,  0.0732},
   {17.0000   , 1.0000  , -0.1951}
   ],Types.NumericField);
d:= DATASET ([
    {1.0000,    1.0000  ,  0.0079},
    {2.0000   , 1.0000 ,  -0.0014},
    {3.0000  ,  1.0000  , -0.0014},
   { 4.0000  ,  1.0000  ,  0.0041},
    {5.0000  ,  1.0000  ,  0.0046},
    {6.0000   , 1.0000    ,0.0004},
    {7.0000  ,  1.0000   , 0.0089},
    {8.0000  ,  1.0000   , 0.0137},
    {9.0000  ,  1.0000   , 0.0131},
   {10.0000  ,  1.0000   , 0.0103},
   {11.0000  ,  1.0000  ,  0.0084},
   {12.0000 ,   1.0000   , 0.0126},
   {13.0000  ,  1.0000  ,  0.0277},
   {14.0000   , 1.0000  ,  0.0280},
   {15.0000  ,  1.0000  , -0.0036},
   {16.0000  ,  1.0000  , -0.0034},
   {17.0000  ,  1.0000 ,  -0.0019}
   ],Types.NumericField);
t:=1;

//d=[0.1, 0.4, -0.2, 0.8];
// d := DATASET([
// {1, 1, 10},
// {2,1,10},
// {3,1,10},
// {4,1,10}],
// Types.NumericField);

p_um := MAX (x,id);

//[f g]= myfunc2(x);

g := DATASET([
   { 1.0000 ,   1.0000,   -0.0004},
   { 2.0000  ,  1.0000 ,  -0.0010},
   { 3.0000   , 1.0000  ,  0.0001},
    {4.0000 ,   1.0000 ,  -0.0012},
   { 5.0000 ,   1.0000 ,  -0.0002},
    {6.0000  ,  1.0000 ,  -0.0009},
   { 7.0000 ,   1.0000 ,  -0.0004},
   { 8.0000  ,  1.0000 ,  -0.0006},
   { 9.0000 ,   1.0000 ,  -0.0006},
   {10.0000  ,  1.0000  , -0.0005},
  { 11.0000  ,  1.0000  , -0.0004},
   {12.0000   , 1.0000   ,-0.0006},
   {13.0000   , 1.0000  , -0.0018},
   {14.0000   , 1.0000  , -0.0036},
   {15.0000    ,1.0000  ,  0.0003},
   {16.0000    ,1.0000   , 0.0003},
   {17.0000   , 1.0000   , 0.0003}
],Types.NumericField);

f := 0.1398;


// OUTPUT (x, NAMED ('xx'));
// OUTPUT (g, NAMED ('gg'));
// OUTPUT (f,NAMED('ff'));
// OUTPUT (d,NAMED('dd'));


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

WResult := Optimization2 (0, 0, 0, 0).WolfeLineSearch2(1,x,1,d,f,g,gtd,0.0001,0.9,25,0.000000001,emptyC, emptyC, emptyC,myfunc2,0,0,0,0);

OUTPUT(WResult,NAMED('WResult'));
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
   
   man:= Optimization (0, 0, 0, 0).MinFUNC (x, myfunc2, emptyC, emptyC , emptyC, 2, 0.00001, 0.000000001,1000, 3,0, 0, 0,0);  
   
   //MinFUNCkk(x0, CostFunc ,  CostFunc_params,  TrainData ,  TrainLabel, MaxIter = 500,  tolFun = 0.00001,  TolX = 0.000000001,  maxFunEvals = 1000,  corrections = 100, =0, =0, =0, =0) := FUNCTION
//OUTPUT (man);


//how much cost function takes time?
// costfun := myfunc2 (x ,emptyC,emptyC, emptyC);
// OUTPUT(costfun);
