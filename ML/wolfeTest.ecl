
IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat;
IMPORT * FROM ML.Types;
IMPORT PBblas;
emptyC := DATASET([], Types.NumericField);
//x=[15, 80, 40, 39];
x := DATASET([
{1, 1, 15},
{2,1,80},
{3,1,40},
{4,1,39}],
Types.NumericField);

t:=1;

//d=[0.1, 0.4, -0.2, 0.8];
d := DATASET([
{1, 1, 0.1},
{2,1,0.4},
{3,1,-0.2},
{4,1,0.8}],
Types.NumericField);

p_um := MAX (x,id);

//[f g]= myfunc2(x);
fg := myfunc2(x, emptyC, emptyC, emptyC);
g := fg (id <= p_um);
f := fg (id = p_um+1)[1].value;


// OUTPUT (x, NAMED ('xx'));
// OUTPUT (g, NAMED ('gg'));
// OUTPUT (f,NAMED('ff'));
// OUTPUT (d,NAMED('dd'));


//gtd = d'*d;
gtdT := ML.Mat.Mul (ML.Mat.Trans(ML.Types.ToMatrix(g)),ML.Types.ToMatrix(d));
gtd := gtdT[1].value;
//OUTPUT (gtd,NAMED('gtd'));

//WolfeLineSearch(wolfeout, x,t,d,f,g,gtd,0.0001,0.9,25,0.000000001,myfunc2,emptyC, emptyC, emptyC,0,0,0,0);
 //wolfeout := WolfeLineSearch(x,t,d,f,g,gtd,0.0001,0.9,3,0.000000001,emptyC, emptyC, emptyC,myfunc2,0,0,0,0);
//OUTPUT(wolfeout);

WResult := Optimization (0, 0, 0, 0).WolfeLineSearch(x,1,d,f,g,gtd,0.0001,0.9,2,0.000000001,emptyC, emptyC, emptyC,myfunc2,0,0,0,0);
//OUTPUT(gtd,named('gtd'));
//OUTPUT(WResult,NAMED('WResult'));
WWWresult := Optimization (0, 0, 0, 0).WolfeOut_FromField(WResult);
//OUTPUT (WWWresult , named ('wwwresult'));
// funresult := myfunc2 ( x, emptyC, emptyC , emptyC);
// OUTPUT(funresult, NAMED('funresults'));

// pol := Optimization (0, 0, 0, 0).polyinterp_both (  10.0000 ,  12.8652 ,  -1.3527,100.0000  , 12.9898 ,   1.0860,10.0900, 100);
// pol2 :=  Optimization (0, 0, 0, 0).polyinterp_noboundry (10.0000  , 12.8652 ,  -1.3527,100.0000 ,  12.9898 ,   1.0860);
// pol3 := Optimization (0, 0, 0, 0).polyinterp_img ( 0 , 1.0000 , 19.0000 ,8.0000,2.0000,8 );

   // Mr :=  MinFUNCALAKI(x, myfunc2, emptyC, emptyC , emptyC, 3, 0.00001, 5, 3,0, 0, 0,0);  
   // OUTPUT(Mr);
   
   man:= Optimization (0, 0, 0, 0).MinFUNCkk (x, myfunc2, emptyC, emptyC , emptyC, 3, 0.00001, 0.000000001,5, 3,0, 0, 0,0);  
OUTPUT (man);

   
   
   
   
   // lbg := DATASET([
// {1, 1,  -0.3604},
// {2,1,-0.2707},
// {3,1, 0.0872},
// {4,1,-0.9317}],
// Mat.Types.Element);

 
    
   
   
   
   // old_dir0 := DATASET([
// {1,1,0},
// {2,1,0},
// {3,1,0},
// {4,1,0},
// {1,2, 2.6474},
// {2,2,0.3847},
// {3,2, 2.3242},
// {4,2,-0.9292}],
// Mat.Types.Element);

 
    
    
    
    
     // old_step0 := DATASET([
// {1,1,0},
// {2,1,0},
// {3,1,0},
// {4,1,0},
// {1,2, 1.1200},
// {2,2,0.3810},
// {3,2, 0.5797},
// {4,2,0.6651}],
// Mat.Types.Element);



// HDIG := DATASET([
// {1,1, 1.7636}],
// Mat.Types.Element);
// lb := Optimization (0, 0, 0, 0).Limited_Memory_BFGS (4, 2).lbfgs ( lbg, old_dir0, old_step0,  HDIG);  
  
// OUTPUT (lb);



 // A := DATASET([
    // {1,1,0},
    // {1,2,0},
    // {1,3,0},
    // {1,4,1},
    // {2,1,0},
    // {2,2,0},
    // {2,3,0},
    // {2,4,1},
    // {3,1,1},
    // {3,2,1},
    // {3,3,1},
    // {3,4,1},
    // {4,1,0},
    // {4,2,0},
    // {4,3,1},
    // {4,4,0}],
    // Types.NumericField);
    

   
   
      
    
    // b := DATASET([
    // {1,1,13.3653},
    // {2,1,13.3655},
    // {3,1,14.0000},
    // {4,1,0.2266}],
    // Types.NumericField);
  
    // A_map := PBblas.Matrix_Map(4, 4, 4, 4);
    // b_map := PBblas.Matrix_Map(4, 1, 4, 1);
    // A_part := ML.DMat.Converted.FromNumericFieldDS (A, A_map);
    // b_part := ML.DMat.Converted.FromNumericFieldDS (b, b_map);
  
    // params_part := DMAT.solvelinear (A_map,  A_part, FALSE, b_map, b_part) ; // for now
    // OUTPUT(A);