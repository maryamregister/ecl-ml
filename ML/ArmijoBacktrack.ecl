IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat;
IMPORT * FROM ML.Types;
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
    polyinterp (REAL8 t_1, REAL8 f_1, REAL8 gtd_1, REAL8 t_2, REAL8 f_2, REAL8 gtd_2) := FUNCTION
      d1 := gtd_1 + gtd_2 - (3*((f_1-f_2)/(t_1-t_2)));
      d2 := SQRT ((d1*d1)-(gtd_1*gtd_2));
      d2real := TRUE; //check it ???
      temp := IF (d2real,t_2 - ((t_2-t_1)*((gtd_2+d2-d1)/(gtd_2-gtd_1+(2*d2)))),-100);
      temp100 := temp =-100;
      polResult := IF (temp100,(t_1+t_2)/2,MIN([MAX([temp,t_1]),t_2]));
      RETURN polResult;
    END;
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
    //loop term to be used in the loop condition 42741332
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
      tTemp := polyinterp ( 0,  f,  gtd,  tit,  fNewit[1].value,  gtdNewit[1].value);
      tTemp2 := tTemp;// t = polyinterp([0 f gtd; t f_new sqrt(-1)]); for now instead of sqrt(-1) I put gtdNewit[1].value ???
      tTemp3 := tTemp;//t = polyinterp([0 f gtd; t f_new sqrt(-1); t_prev f_prev sqrt(-1)],doPlot);
      tNew := IF (IsNotLegal(fNewit),tit*0.5,IF(IsNotLegal(gNewit),IF (cond,tTemp2,tTemp3),tTemp));
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
END;