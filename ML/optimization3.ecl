

  //no = 1 : t
  //no = 2 : f
  //no = 3 : g
  //no = 4 : FuncEval : number of times the cost fucntion has been evaluated in the Wolfe fucntion
  EXPORT WolfeOut_FromField(DATASET(Types.NumericField) mod) := FUNCTION
    modelD_Map :=	DATASET([{'id','ID'},{'x','1'},{'y','2'},{'value','3'},{'no','4'}], {STRING orig_name; STRING assigned_name;});
    FromField(mod,Mat.Types.MUElement,dOut,modelD_Map);
    RETURN dOut;
  END;
  //x,t,d,f,fr,g,gtd,c1,LS,tolX,debug,doPlot,saveHessianComp,funObj,varargin)
  // Backtracking linesearch to satisfy Armijo condition
  //
  // Inputs:
  //   x: starting location
  //   t: initial step size
  //   d: descent direction
  //   f: function value at starting location
  //   fr: reference function value (usually funObj(x))
  //   gtd: directional derivative at starting location
  //   c1: sufficient decrease parameter
  //   debug: display debugging information
  //   LS: type of interpolation
  //   tolX: minimum allowable step length
  //   doPlot: do a graphical display of interpolation
  //   funObj: objective function
  //   varargin: parameters of objective function
  EXPORT ArmijoBacktrack(DATASET(Types.NumericField)x, REAL8 t, DATASET(Types.NumericField)d, REAL8 f, REAL8 fr, DATASET(Types.NumericField) g, REAL8 gtd, REAL8 c1=0.0001, REAL8 tolX=0.000000001,DATASET(Types.NumericField) CostFunc_params, DATASET(Types.NumericField) TrainData , DATASET(Types.NumericField) TrainLabel, DATASET(Types.NumericField) CostFunc (DATASET(Types.NumericField) x, DATASET(Types.NumericField) CostFunc_params, DATASET(Types.NumericField) TrainData , DATASET(Types.NumericField) TrainLabel), prows=0, pcols=0, Maxrows=0, Maxcols=0):=FUNCTION
      P_num := Max (x, id); //the length of the parameters vector (number of parameters)
      IsNotLegal (DATASET (Mat.Types.Element) Mat) := FUNCTION //???to be defined
        RETURN FALSE;
      END;
      ExtractGrad (DATASET(Types.NumericField) inp) := FUNCTION
        RETURN inp (id <= P_num);
      END;
      ExtractCost (DATASET(Types.NumericField) inp) := FUNCTION
        RETURN inp (id = (P_num+1))[1].value;
      END;
      // Evaluate the Objective and Gradient at the Initial Step
      xNew := ML.Types.FromMatrix (ML.Mat.Add(ML.Types.ToMatrix(x),ML.Mat.Scale(ML.Types.ToMatrix(d),t)));
      CostGradNew := CostFunc (xNew ,CostFunc_params,TrainData, TrainLabel);
      gNew := ExtractGrad (CostGradNew);
      fNew := ExtractCost (CostGradNew);
      //loop term to be used in the loop condition
      LoopTerm := fr + c1*t*gtd;
      //build what should be passed to the bracketing loop
      fNewno := DATASET([{1,1,fNew,1}], Mat.Types.MUElement);
      gNewno := Mat.MU.To (ML.Types.ToMatrix(gNew),2);
      funevalno := DATASET([{1,1,1,3}], Mat.Types.MUElement);
      tno := DATASET([{1,1,t,4}], Mat.Types.MUElement);
      fPrevno := DATASET([{1,1,-1,5}], Mat.Types.MUElement);
      tPrevno := DATASET([{1,1,0,6}], Mat.Types.MUElement);
      breakno := DATASET([{1,1,0,7}], Mat.Types.MUElement);
      toPass := fNewno + gNewno + funevalno + tno + fPrevno + tPrevno + breakno;
      Armijo (DATASET (Mat.Types.MUElement) Inp) := FUNCTION
        //calculate new t
        fNewit := Mat.MU.From (Inp,1);
        gNewit := Mat.MU.FROM (Inp,2);
        tit := Mat.MU.From (Inp,4)[1].value;
        tPrevit := Mat.MU.From (Inp,6)[1].value;
        gtdNewit := ML.Mat.Mul (ML.Mat.Trans((gNewit)),ML.Types.ToMatrix(d));
        FEval := Mat.MU.FROM (Inp,3)[1].value;
        fPrevit := Mat.MU.FROM (Inp,5);
        cond := IsNotLegal (fPrevit) | (FEval <2);
        // t = polyinterp([0 f gtd; t f_new sqrt(-1)],doPlot);
        tTemp := polyinterp_img (0, f, gtd, tit, fNewit[1].value, gtdNewit[1].value);
        tTemp2 := tTemp;//t = polyinterp([0 f gtd; t f_new sqrt(-1); t_prev f_prev sqrt(-1)],doPlot);
        tNew := IF (IsNotLegal(fNewit),tit*0.5,IF (cond,tTemp,tTemp2));
        //Adjust tNew if change in t is too small/large
        T1 := tit*0.001;
        T2 := tit*0.6;
        AdCond1 := tNew < T1 ; //t < temp*1e-3
        AdCond2 := tNew > T2; //t > temp*0.6
        AdtNew := IF (AdCond1, T1, IF (AdCond2, T2, tNew));
        //Calculate new f and g values
        fPrvno := DATASET([{1,1,fNewit[1].value,5}], Mat.Types.MUElement);
        tPrvno := DATASET([{1,1,tit,6}], Mat.Types.MUElement);
        //calculate new f and g
        xN := ML.Types.FromMatrix (ML.Mat.Add(ML.Types.ToMatrix(x),ML.Mat.Scale(ML.Types.ToMatrix(d),AdtNew)));
        CGN := CostFunc (xN ,CostFunc_params,TrainData, TrainLabel);
        gN := ExtractGrad (CGN);
        fN := ExtractCost (CGN);
        gNno := Mat.MU.To (ML.Types.ToMatrix(gN),2);
        fNno := DATASET([{1,1,fN,1}], Mat.Types.MUElement);
        FEalno := DATASET([{1,1,FEval+1,3}], Mat.Types.MUElement);
        tNno := DATASET([{1,1,AdtNew,4}], Mat.Types.MUElement);
        //sum(abs(t*d))???? to be calculated
        sumA := 3;
        brk := IF(sumA<=tolX,1,0);
        brkno := DATASET([{1,1,brk,7}], Mat.Types.MUElement);
        RETURN fNno + gNno + FEalno + tNno + fPrvno +  tPrvno + brkno;
      END;
      //f_new > fr + c1*t*gtd || ~isLegal(f_new)
      LoopResult := LOOP (toPass, (Mat.MU.From (ROWS(LEFT),7)[1].value = 0) & (IsNotLegal (Mat.MU.From (ROWS(LEFT),1)) & (Mat.MU.From (ROWS(LEFT),1)[1].value < LoopTerm ) ), Armijo(ROWS(LEFT)));
      BCond := Mat.MU.From (LoopResult,7)[1].value = 1;
      //[t,f_new,g_new,funEvals] return x_new too????
      RegularResult := LoopResult (no=4) + LoopResult (no=1) + LoopResult (no=2) + LoopResult (no=3);
      Breakresult := DATASET([{1,1,0,4}], Mat.Types.MUElement) + DATASET([{1,1,f,1}], Mat.Types.MUElement) + Mat.MU.To (ML.Types.ToMatrix(g),2) + LoopResult (no=3);
      FinalResult := IF (BCond, Breakresult, RegularResult);
    RETURN FinalResult;
  END; // END ArmijoBacktrack

//LBFGS algorithm
//Returns final updated parameter: numericField foramt
//x0: input parameter vector (column)
//CostFunc : function handler , it should return a Cost value and the gradient values which is a vector with the same size of x0
//The output of the CostFunc function should be in numericField format where the last id's value (the maximum id) represents cost value and the rest
//represent the gradient vector
//So basically CostFunc should recive all its parameters in one single numericField structure + training data + training labels and return a vector of the gradients+costvalue
//Cost function should have a universal interface, so it recives all parameters in numericfield format and returns in numericfield format
//CostFunc_params : parameters that need to be passed to the CostFunc
//TrainData : Train data in numericField format
//TrainLabel : labels asigned to the train data ( if it is an un-supervised task this parameter would be empty dataset)
//MaxIter: Maximum number of iteration allowed in the optimization algorithm
//tolFun : Termination tolerance on the first-order optimality (1e-5)
//TolX : Termination tolerance on progress in terms of function/parameter changes (1e-9)
//maxFunEvals : Maximum number of function evaluations allowed (1000)
//corrections : number of corrections to store in memory (default: 100) higher numbers converge faster but use more memory)
//this Macro recives all the parameters in numeric field fromat and returns in numeric field format
//prows and maxrows related to "numer of parameters (P)" which is actually the length of the x0 vector
//pcols and Maxcols are relaetd to "number of correstions to store in the memory (corrections)" which is in the MethodOptions
//In all operation I want to f or g get nan value if it is devided by zero (do I need to include #option on top of the CostFunc)????????
EXPORT MinFUNC(DATASET(Types.NumericField) x0, DATASET(Types.NumericField) CostFunc (DATASET(Types.NumericField) x, DATASET(Types.NumericField) CostFunc_params, DATASET(Types.NumericField) TrainData , DATASET(Types.NumericField) TrainLabel), DATASET(Types.NumericField) CostFunc_params, DATASET(Types.NumericField) TrainData , DATASET(Types.NumericField) TrainLabel, INTEGER MaxIter = 100, REAL8 tolFun = 0.00001, REAL8 TolX = 0.000000001, INTEGER maxFunEvals = 1000, INTEGER corrections = 100, prows=0, pcols=0, Maxrows=0, Maxcols=0) := FUNCTION
//#option ('divideByZero', 'nan'); //In all operation I want to f or g get nan value if it is devided by zero
//Functions used
SumABSg (DATASET (Types.NumericField) g_temp) := FUNCTION
  r := RECORD
    t_RecordID id := 1 ;
    t_FieldNumber number := 1;
    t_FieldReal value := SUM(GROUP,ABS(g_temp.value));
  END;
  SumABSCol := TABLE(g_temp,r);
  RETURN SumABSCol[1].value;
END;
SumABSg_matrix (DATASET (Mat.Types.Element) g_temp) := FUNCTION
  r := RECORD
    Mat.Types.t_Index x := 1 ;
    Mat.Types.t_Index y := 1;
    Mat.Types.t_value value := SUM(GROUP,ABS(g_temp.value));
  END;
  SumABSCol := TABLE(g_temp,r);
  RETURN SumABSCol[1].value;
END;
// Check Optimality Condition: if sum(abs(g)) <= tolFun return True
OptimalityCond (DATASET (Types.NumericField) g_temp) := FUNCTION
  sag := SumABSg (g_temp);
  RETURN IF( sag <= tolFun , TRUE, FALSE);
END;
//Check Optimality Condition for the Loop : if sum(abs(g)) <= tolFun return TRUE
OptimalityCond_mat (DATASET (Mat.Types.Element) g_t) := FUNCTION
  sag := SumABSg_matrix (g_t);
  RETURN IF( sag <= tolFun , TRUE, FALSE);
END;//END OptimalityCond_mat
//Check for lack of progress 1: if sum(abs(t*d)) <= tolX return TRUE
LackofProgress1 (DATASET (Mat.Types.Element) d_temp, REAL8 t_temp) := FUNCTION
  r := RECORD
    Mat.Types.t_Index x := 1 ;
    Mat.Types.t_Index y := d_temp.y;
    Mat.Types.t_Value value := SUM(GROUP,ABS(d_temp.value * t_temp));
  END;
  SumABSCol := TABLE(d_temp,r,d_temp.y);
  RETURN IF( SumABSCol[1].value <= tolX , TRUE, FALSE);
END;
//Check for lack of progress 2: if abs(f-f_old) < tolX return TRUE
LackofProgress2 (REAL8 f_f_temp) := FUNCTION
  RETURN IF (ABS (f_f_temp) < tolX, TRUE, FALSE);
END;
//Check for going over evaluation limit
//if funEvals*funEvalMultiplier > maxFunEvals return TRUE  (funEvalMultiplier=1)
EvaluationLimit (INTEGER8 fun_temp) := FUNCTION
  RETURN IF (fun_temp > maxFunEvals, TRUE, FALSE);
END;
IsLegal (DATASET (Types.NumericField) inp) := FUNCTION //???to be defined
  RETURN TRUE;
END;
//the length of the parameters vector (for example in an neural network algorith, all the weight parameters are passed in ONE vector
//to the optimization algorithm to get updated

P := Max (x0, id); // the maximum matrix id in the parameters numericfield dataset
ExtractGrad (DATASET(Types.NumericField) inp) := FUNCTION
  RETURN inp (id <= P);
END;
ExtractCost (DATASET(Types.NumericField) inp) := FUNCTION
  RETURN inp (id = (P+1))[1].value;
END;
//Evaluate Initial Point
CostGrad0 := CostFunc (x0,CostFunc_params,TrainData, TrainLabel);

g0 := ExtractGrad (CostGrad0);
Cost0 := ExtractCost (CostGrad0);
//Check the optimality of the initial point (if sum(abs(g)) <= tolFun)
IsInitialPointOptimal := OptimalityCond (g0);

//LBFGS Module
O := Limited_Memory_BFGS (P, corrections);
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
// Initialize dirs and steps matrices
old_dir0   := DATASET(P * corrections, gen(COUNTER, P));
old_steps0 := DATASET(P * corrections, gen(COUNTER, P));
Hdiag0 := 1;
//initialize hessian diag as 1
Hdiag0no := DATASET([{1,1,1,5}], Mat.Types.MUElement);
//number of times the cost function is evaluated
FunEval := 1; 
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
dLegalno := DATASET([{1,1,1,11}], Mat.Types.MUElement);
ProgressAlongDirectionno := DATASET([{1,1,1,12}], Mat.Types.MUElement);//Check that progress can be made along direction ( if gtd > -tolX)
Topass := x0no + g0n0 + old_steps0no + old_dir0no + Hdiag0no + C0n0 + FunEvalno + dno +f_fno + tno + dLegalno + ProgressAlongDirectionno;
emptyE := DATASET([], Mat.Types.Element);
MinFRecord := RECORD
  DATASET(Mat.Types.Element) x;
  DATASET(Mat.Types.Element) g;
  DATASET(Mat.Types.Element) old_steps;
  DATASET(Mat.Types.Element) old_dirs;
  REAL8 Hdiag;
  REAL8 Cost;
  INTEGER8 funEvals_;
  DATASET(Mat.Types.Element) d;
  REAL8 fnew_fold; //f_new-f_old in that iteration
  REAL8 t_;
  BOOLEAN dLegal;
  BOOLEAN ProgAlongDir; //Progress Along Directionno //Check that progress can be made along direction ( if gtd > -tolX)
  BOOLEAN optcond; // Check Optimality Condition
  BOOLEAN lackprog1; //Check for lack of progress 1
  BOOLEAN lackprog2; //Check for lack of progress 2
  BOOLEAN exceedfuneval; //Check for going over evaluation limit
END;

ToPassMinF := DATASET ([{ML.Types.ToMatrix(x0),ML.Types.ToMatrix(g0),old_steps0,old_dir0,Hdiag0,Cost0,FunEval,emptyE,100 +tolFun,1,TRUE,TRUE,FALSE,FALSE,FALSE,FALSE}],MinFRecord);


MinFstep (DATASET (MinFRecord) inputp, INTEGER coun) := FUNCTION
  x_pre := inputp[1].x;
  g_pre := inputp[1].g;
  g_pre_ := ML.Mat.Scale (g_pre , -1);
  step_pre := inputp[1].old_steps;
  dir_pre := inputp[1].old_dirs;
  H_pre := inputp[1].Hdiag;
  f_pre := inputp[1].Cost;
  FunEval_pre := inputp[1].funEvals_;
  //HG_ is actually search direction in fromual 3.1
  HG_ :=  IF (coun = 1, O.Steepest_Descent (g_pre), O.lbfgs(g_pre_,Dir_pre, Step_pre,H_pre));//the result is the approximate inverse Hessian, multiplied by the gradient and it is in PBblas.layout format
  d := ML.DMat.Converted.FromPart2DS(HG_);
  dlegalstep := IsLegal (d);
  d_next := ML.Types.ToMatrix(d);
  //Directional Derivative : gtd = g'*d;
  //calculate gtd to be passed to the wolfe algortihm
  gtd := ML.Mat.Mul(ML.Mat.Trans((g_pre)),ML.Types.ToMatrix(d));
  //Check that progress can be made along direction : if gtd > -tolX then break!
  gtdprogress := IF (gtd[1].value > -1*tolX, FALSE, TRUE);
  // Select Initial Guess If coun =1 then t = min(1,1/sum(abs(g)));
  t := IF (coun = 1,MIN ([1, 1/SumABSg (ML.Types.FromMatrix (g_pre))]),1);
   //find point satisfiying wolfe
  //[t,f,g,LSfunEvals] = WolfeLineSearch(x,t,d,f,g,gtd,c1,c2,LS,25,tolX,debug,doPlot,1,funObj,varargin{:});
  t_neworig := WolfeLineSearch2(1, ML.Types.FromMatrix(x_pre), t, d, f_pre, ML.Types.FromMatrix(g_pre), gtd[1].value, 0.0001, 0.9, 25, 0.000000001, CostFunc_params, TrainData , TrainLabel, CostFunc , prows, pcols, Maxrows, Maxcols);
  t_new := t_neworig[1].t;
  g_Next := t_neworig[1].g_new;
  Cost_Next := t_neworig[1].f_new;
  FunEval_Wolfe := t_neworig[1].WolfeFunEval;
  FunEval_next := FunEval_pre + FunEval_Wolfe;
  x_pre_updated :=  ML.Mat.Add((x_pre),ML.Mat.Scale(ML.Types.ToMatrix(d),t_new));
  x_Next := ML.Types.FromMatrix(x_pre_updated);
  fpre_fnext := Cost_Next-f_pre;
  //calculate new Hessian diag, dir and steps
  //Step_Next :=  O.lbfgsUpdate_corr (g_pre, g_Next, Step_pre);
  Step_Next := O.lbfgsUpdate_Stps (x_pre, x_pre_updated, g_pre, g_Next, Step_pre);
  //Dir_Next := O.lbfgsUpdate_corr(x_pre, x_pre_updated, Dir_pre);
  Dir_Next := O.lbfgsUpdate_Dirs (x_pre, x_pre_updated, g_pre, g_Next, Dir_pre);
  H_Next := O.lbfgsUpdate_Hdiag (x_pre, x_pre_updated, g_pre, g_Next, H_pre);
  optcond := OptimalityCond_mat (g_Next);
  lack1 := LackofProgress1 (d_next, t_new);
  lack2 := LackofProgress2 (fpre_fnext);
  evalimit := EvaluationLimit (FunEval_next);
  //ToReturn := x_Next_no + g_Nextno + Step_Nextno + Dir_Nextno + H_Nextno+  C_Nextno + FunEval_Nextno + d_Nextno + fpre_fnextno + t_newno + dlegalstepno + gtdprogressno;
  MinFRecord MF (MinFRecord l) := TRANSFORM
    SELF.x := x_pre_updated;
    SELF.g := g_Next;
    SELF.old_steps := Step_Next;
    SELF.old_dirs := Dir_Next;
    SELF.Hdiag := H_Next;
    SELF.Cost := Cost_Next;
    SELF.funEvals_ := FunEval_next;
    SELF.d := d_next;
    SELF.fnew_fold := fpre_fnext;
    SELF.t_ := t_new;
    SELF.dLegal := dlegalstep;
    SELF.ProgAlongDir := gtdprogress;
    SELF.optcond := optcond; // Check Optimality Condition
    SELF.lackprog1 := lack1; //Check for lack of progress 1
    SELF.lackprog2 := lack2; //Check for lack of progress 2
    SELF.exceedfuneval := evalimit;
    SELF := l;
  END;
  MinFRecord MF_dnotleg (MinFRecord l) := TRANSFORM
    SELF.d := d_next;
    SELF.dLegal := FALSE;
    SELF := l;
  END;
  MFreturn := PROJECT (inputp,MF(LEFT));
  MFndreturn := PROJECT (inputp,MF_dnotleg(LEFT));
  RETURN MFreturn;
  //RETURN IF(dlegalstep,MFreturn,MFndreturn); orig
END;
//updating step function


MinFstepout := MinFstep(ToPassMinF,1);
// MinFstepout := LOOP(ToPassMinF, COUNTER <= MaxIter AND ROWS(LEFT)[1].dLegal AND ROWS(LEFT)[1].ProgAlongDir   
// AND ~ROWS(LEFT)[1].optcond AND ~ROWS(LEFT)[1].lackprog1 AND ~ROWS(LEFT)[1].lackprog2 AND ~ROWS(LEFT)[1].exceedfuneval  , MinFstep(ROWS(LEFT),COUNTER)); orig
//LOOP(ToPass_Zooming, COUNTER <= Zoom_Max_Itr AND ~ROWS(LEFT)[1].done AND ~ROWS(LEFT)[1].break, ZoomingStep(ROWS(LEFT),COUNTER));
//MinFstepout := LOOP(ToPassMinF, COUNTER <= 1, MinFstep(ROWS(LEFT),COUNTER));
/*
outrec := RECORD
  DATASET(Mat.Types.Element) x;
  REAL8 cost;
END;
output_xfinal_costfinal := PROJECT(MinFstepout, TRANSFORM(outrec, SELF := LEFT));
output_x0_cost0 := DATASET([{ML.Types.ToMatrix(x0),90}],outrec);
FinalResult := IF (IsInitialPointOptimal,output_x0_cost0,output_xfinal_costfinal ); orig */
//RETURN FinalResult; orig
RETURN MinFstepout;

END;//END MinFUNC
  
END;// END Optimization
