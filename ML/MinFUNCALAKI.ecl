IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat;
IMPORT $.DMat;
IMPORT * FROM ML.Types;
IMPORT PBblas;
Layout_Cell := PBblas.Types.Layout_Cell;
Layout_Part := PBblas.Types.Layout_Part;
//expecting numeric field and giving
//LBFGS algorithm
//xout, final updated parameter: numericField foramt
//x0: input parameter vector (column) to be updated : numericField foramt
//CostFunc : function handler , it should return a Cost value and the gradient values which is a vector with the same size of x0
//The output of the CostFunc function should be in numericField format where the last id's value (the id with maximum value) represents the cost value and the other
//values represent the gradient vector
//So basically CostFunc should recive all its parameters in one single vector and return a vector of the gradients+costvalue
//Cost function should have a universal interface, soit recives all parameters in numericfield format and returns in numericfield format
//CostFunc_params : parameters that need to be passed to the CostFunc
//TrainData : Train data in numericField format
//TrainLabel : labels asigned to the train data ( if it is an un-supervised task this parameter would be '')
//x0: starting vector to be updated
//MaxIter: Maximum number of iteration allowed in the optimization algorithm
//MethodOptions: The options of the LBFGS Macro : id=1: number of corrections to store in memory
//this Macro recives all the parameters in numeric field fromat and returns in numeric field format
//prows and maxrows related to "numer of parameters (P)" which is actually the length of the x0 vector
//pcols and Maxcols are relaetd to "number of correstions to store in the memory (corrections)" which is in the MethodOptions
//In all operation I want to f or g get nan value if it is devided by zero (do I need to include #option on top of the CostFunc)????????

EXPORT MinFUNCALAKI(DATASET(Types.NumericField) x0, DATASET(Types.NumericField) CostFunc (DATASET(Types.NumericField) x, DATASET(Types.NumericField) CostFunc_params, DATASET(Types.NumericField) TrainData , DATASET(Types.NumericField) TrainLabel), DATASET(Types.NumericField) CostFunc_params, DATASET(Types.NumericField) TrainData , DATASET(Types.NumericField) TrainLabel, INTEGER MaxIter, REAL8 tolFun, INTEGER maxFunEvals, INTEGER corrections, prows=0, pcols=0, Maxrows=0, Maxcols=0) := FUNCTION
//#option ('divideByZero', 'nan'); //In all operation I want to f or g get nan value if it is devided by zero
//initial parameters
tolX := 0.000000001;

P := Max (x0, id); //the length of the parameters vector
//Optimization Module
O := Optimization (prows, pcols, Maxrows, Maxcols).Limited_Memory_BFGS (P, corrections);
//initialize Hdiag,old_dir,old_steps, gradient and cost
//The size of the old_dir and old_steps matrices are "number of parameters" * "number of corrections to store in memory (corrections)"
// old_dir0 := Mat.Zeros(P, corrections);
// old_steps0 := old_dir0;

 //New Matrix Generator
Mat.Types.Element gen(UNSIGNED4 c, UNSIGNED4 NumRows, REAL8 v=0) := TRANSFORM
  SELF.x := ((c-1) % NumRows) + 1;
  SELF.y := ((c-1) DIV NumRows) + 1;
  SELF.value := v;
END;
// MyVec := DATASET(P * corrections, gen(COUNTER, P));
old_dir0 := DATASET(P * corrections, gen(COUNTER, P));
old_steps0 := DATASET(P * corrections, gen(COUNTER, P));
// old_dir0 := DATASET([
// {1,1,1},
// {2,1,2},
// {3,1,3},
// {4,1,4},
// {1,2,5},
// {2,2,6},
// {3,2,7},
// {4,2,8}],
// Mat.Types.Element);
// old_steps0 := DATASET([
// {1, 1, 11},
// {2,1,12},
// {3,1,13},
// {4,1,14},
// {1,2,15},
// {2,2,16},
// {3,2,17},
// {4,2,18}],
// Mat.Types.Element);
Hdiag0no := DATASET([{1,1,1,5}], Mat.Types.MUElement); //initialize hessian diag as 1
//Evaluate Initial Point
CostGrad0 := CostFunc (x0,CostFunc_params,TrainData, TrainLabel);
g0 := CostGrad0 (id<=P);
Cost0 := CostGrad0 (id = P+1)[1].value;

FunEval := 1; //number of time the function is evaluated

//Check the optimality of the initial point (if sum(abs(g)) <= tolFun)
IsInitialPointOptimal := FALSE; //to be calculated ??????

//Perform up to a maximum of 'maxIter' descent steps:


//put all the parameters that need to be sent to the step fucntion in a Mat.Types.MUElement format
x0no := Mat.MU.To (ML.Types.ToMatrix(x0),1);
g0n0 := Mat.MU.To (ML.Types.ToMatrix(g0),2);
old_steps0no := Mat.MU.To (old_steps0,3);
old_dir0no := Mat.MU.To (old_dir0,4);
C0n0 := DATASET([{1,1,Cost0,6}], Mat.Types.MUElement);
FunEvalno := DATASET([{1,1,FunEval,7}], Mat.Types.MUElement);
dno := DATASET([{1,1,-1,8}], Mat.Types.MUElement);
f_fno := DATASET([{1,1,100 +tolFun ,9}], Mat.Types.MUElement); //f_new-f_old in that iteration
tno := DATASET([{1,1,1,10}], Mat.Types.MUElement);//initial step value
Topass := x0no + g0n0 + old_steps0no + old_dir0no + Hdiag0no + C0n0 + FunEvalno + dno +f_fno + tno;
//updating step function
step (DATASET (Mat.Types.MUElement) inputp, INTEGER coun) := FUNCTION
  x_pre := Mat.MU.From (inputp,1);// x_pre : x previouse : parameter vector from the last iteration (to be updated)
  g_pre := Mat.MU.From (inputp,2);
  g_pre_ := ML.Mat.Scale (g_pre , -1);
  Step_pre := Mat.MU.From (inputp,3);
  Dir_pre := Mat.MU.From (inputp,4);
  H_pre := Mat.MU.From (inputp,5);
  f_pre := Mat.MU.From (inputp,6)[1].value;
  FunEval_pre := Mat.MU.From (inputp,7)[1].value;
  HG_ :=  IF (coun = 1, O.Steepest_Descent (g_pre), O.lbfgs(g_pre_,Dir_pre, Step_pre,H_pre));//the result is the approximate inverse Hessian, multiplied by the gradient and it is in PBblas.layout format
  d := ML.DMat.Converted.FromPart2DS(HG_);
  d_Nextno := Mat.MU.To (ML.Types.ToMatrix(d),8);
  //check if d is legal 940 //??????
  //HG_ is actually search direction in fromual 3.1
  // ************Compute Step Length **************
  //Directional Derivative : gtd = g'*d;
  gtd := ML.Mat.Mul(ML.Mat.Trans((g_pre)),ML.Types.ToMatrix(d));
  // Select Initial Guess
  t := IF (coun = 1,0.5544,1); //to be calculated ????
  //calculate gtd to be passed to the wolfe algortihm
  //find point satisfiying wolfe
  //[t,f,g,LSfunEvals] = WolfeLineSearch(x,t,d,f,g,gtd,c1,c2,LS,25,tolX,debug,doPlot,1,funObj,varargin{:});
  t_neworig := Optimization (prows, pcols, Maxrows, Maxcols).WolfeLineSearch(ML.Types.FromMatrix(x_pre), t, d, f_pre, ML.Types.FromMatrix(g_pre), gtd[1].value, 0.0001, 0.9, 10, 0.000000001, CostFunc_params, TrainData , TrainLabel, CostFunc , prows, pcols, Maxrows, Maxcols);
  no_t_t_ := Optimization (prows, pcols, Maxrows, Maxcols).WolfeOut_FromField(t_neworig);
  //Check that progress can be made along direction : if gtd > -tolX
  //update the parameter vector:x_new = xold+alpha*HG_
  t_new := Mat.MU.FROM (no_t_t_,1)[1].value;
  
  t_newno := DATASET([{1,1,t_new,10}], Mat.Types.MUElement);
  FunEval_Wolfe := Mat.MU.FROM (no_t_t_,4)[1].value;
  
  x_pre_updated :=  ML.Mat.Add((x_pre),ML.Mat.Scale(ML.Types.ToMatrix(d),t_new));
  x_Next := ML.Types.FromMatrix(x_pre_updated);
  FunEval_new := FunEval_pre + FunEval_Wolfe;
  // CostGrad_Next := CostFunc (x_Next,CostFunc_params,TrainData, TrainLabel);
  // g_Next := CostGrad_Next (id<=P);
  // Cost_Next := CostGrad_Next (id = P+1)[1].value;
  g_Next := Mat.MU.FROM (no_t_t_,3);
  Cost_Next := Mat.MU.FROM (no_t_t_,2)[1].value;
  fpre_fnext := Cost_Next - f_pre;
  fpre_fnextno := DATASET([{1,1,fpre_fnext,9}], Mat.Types.MUElement);
  x_Next_no := Mat.MU.To (ML.Types.ToMatrix(x_Next),1);
  g_Nextno := Mat.MU.To (g_Next,2);
  C_Nextno := DATASET([{1,1,Cost_Next,6}], Mat.Types.MUElement);
  //calculate new Hessian diag, dir and steps
  Step_Next :=  O.lbfgsUpdate_corr (g_pre, g_Next, Step_pre);
  Step_Nextno := Mat.MU.To (Step_Next, 3);
  Dir_Next := O.lbfgsUpdate_corr(x_pre, x_pre_updated, Dir_pre);
  Dir_Nextno := Mat.MU.To (Dir_Next, 4);
  //H_Next := O.lbfgsUpdate_Hdiag (x_pre, g_pre);
  H_Next := O.lbfgsUpdate_Hdiag (x_pre, x_pre_updated, g_pre, g_Next);
  H_Nextno := DATASET([{1,1,H_Next,5}], Mat.Types.MUElement);
  //update FunEval (FunEval = funEval + FunEvalLS
  FunEval_Next := FunEval_New; //felan ???
  FunEval_Nextno := DATASET([{1,1,FunEval_Next,7}], Mat.Types.MUElement);
  //creat the return value which is appending all the values that need to be passed
  ToReturn := x_Next_no + g_Nextno + Step_Nextno + Dir_Nextno + H_Nextno+  C_Nextno + FunEval_Nextno + d_Nextno + fpre_fnextno + t_newno;
  //RETURN  Mat.MU.To (ML.Types.ToMatrix(HG_field),1);
  //RETURN Mat.MU.To ((x_pre),1) + DATASET([{1,1,t,2}], Mat.Types.MUElement) + Mat.MU.To (ML.Types.ToMatrix(d),3) + DATASET([{1,1,f_pre,4}], Mat.Types.MUElement) + Mat.MU.To ((g_pre),5) + DATASET([{1,1,gtd[1].value,6}], Mat.Types.MUElement) ;
  //RETURN DATASET([{1,1,FunEval_new,2}], Mat.Types.MUElement);
  RETURN ToReturn;
//ML.Types.FromMatrix(x_pre), t, d, f_pre, ML.Types.FromMatrix(g_pre), gtd[1].value, 0.0001, 0.9, 10, 0.000000001
//RETURN IF (coun =1 , ToReturn,  Mat.MU.To(g_pre,1) + Mat.MU.To(Dir_pre,2) + Mat.MU.To(Step_pre,3) + Mat.MU.To(H_pre,4));  
END; //END step
myout := LOOP(topass, COUNTER <= 8, step(ROWS(LEFT),COUNTER));
//Check Wethere the calculated d is legal
IsLegald (DATASET (Mat.Types.MUElement) q) := FUNCTION //??????????
  d_temp := Mat.MU.From (q,8);
  //check wethere d_temp is legal
  RETURN TRUE;
END;

// Check Optimality Condition: if sum(abs(g)) <= tolFun ????????????
OptimalityCond (DATASET (Mat.Types.MUElement) q) := FUNCTION
  g_temp := Mat.MU.From (q,2);
  r := RECORD
    Mat.Types.t_Index x := 1 ;
    Mat.Types.t_Index y := g_temp.y;
    Mat. Types.t_Value value := SUM(GROUP,ABS(g_temp.value));
  END;
  SumABSCol := TABLE(g_temp,r,g_temp.y);
  RETURN IF( SumABSCol[1].value < tolFun , TRUE, FALSE);
END;

// Check Lack of Progress 
//1-if sum(abs(t*d)) <= tolX ??????
LackofProgress1 (DATASET (Mat.Types.MUElement) q) := FUNCTION
  d_temp := Mat.MU.From (q,8);
  t_temp := Mat.MU.From (q,10)[1].value;
  r := RECORD
    Mat.Types.t_Index x := 1 ;
    Mat.Types.t_Index y := d_temp.y;
    Mat. Types.t_Value value := SUM(GROUP,ABS(d_temp.value * t_temp));
  END;
  SumABSCol := TABLE(d_temp,r,d_temp.y);
  RETURN IF( SumABSCol[1].value < tolX , TRUE, FALSE);
END;


//2- if abs(f-f_old) < tolX
LackofProgress2 (DATASET (Mat.Types.MUElement) q) := FUNCTION
  f_f_temp := Mat.MU.From (q,9)[1].value;
  RETURN IF (ABS (f_f_temp) < tolX, TRUE, FALSE);
END;

//Check for going over evaluation limit
//if funEvals*funEvalMultiplier > maxFunEvals   (funEvalMultiplier=1)
EvaluationLimit (DATASET (Mat.Types.MUElement) q) := FUNCTION
  fun_temp := Mat.MU.From (q,7)[1].value;
  RETURN IF (fun_temp > maxFunEvals, TRUE, FALSE);
END;
//xout := IF (IsInitialPointOptimal, x0, step(Topass,1));
//loopcond := COUNTER <MaxItr & ~IsLegald () & ~OptimalityCond & ~LackofProgress1 & ~LackofProgress2 & ~EvaluationLimit
xout := Topass;
RETURN myout;
END;