// Bracketing Line Search to Satisfy Wolfe Conditions
//Source "Numerical Optimization Book" and Matlab implementaion of minFunc :
// M. Schmidt. minFunc: unconstrained differentiable multivariate optimization in Matlab. http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html, 2005.
//  x:  Starting Location (numeric field format)
//  t: Initial step size (its a number, usually 1)
//  d:  descent direction  (Pk in formula 3.1 in the book) (numeric field format)
//  g: gradient at starting location (numeric field format)
//  gtd:  directional derivative at starting location :  gtd = g'*d (its a number), TRANS(Deltafk)*Pk in formula 3.6a
//  c1: sufficient decrease parameter (c1 in formula 3.6a, its a number)
//  c2: curvature parameter (c2 in formula 3.6b, its a number)
//  maxLS: maximum number of iterations in WOLFE algorithm
//  tolX: minimum allowable step length
//  CostFunc: objective function(it returns the gradient and cost value in numeric field format, cost value has the highest
//  id in the returned numeric field structure
//  TrainData and TrainLabel: train and label data for the objective fucntion (numeric field format)
//  The rest are PBblas parameters
//  Define a general FunVAL and add it in wolfebracketing and WolfeZoom ??????
//  WolfeOut is what the macro returns, it include t,f_new,g_new,funEvals (t the calculated step size
//  f_new the cost value in the new point, g_new is the gradient value in the new point and funevals is the number of
//  time the cost function has been calculated in WolfeLineSearch algorithm
IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat;
IMPORT * FROM ML.Types;

EXPORT WolfeLineSearch(DATASET(Types.NumericField)x, REAL8 t, DATASET(Types.NumericField)d, REAL8 f, DATASET(Types.NumericField) g, REAL8 gtd, REAL8 c1=0.0001, REAL8 c2=0.9, INTEGER maxLS=25, REAL8 tolX=0.000000001,DATASET(Types.NumericField) CostFunc_params, DATASET(Types.NumericField) TrainData , DATASET(Types.NumericField) TrainLabel, DATASET(Types.NumericField) CostFunc (DATASET(Types.NumericField) x, DATASET(Types.NumericField) CostFunc_params, DATASET(Types.NumericField) TrainData , DATASET(Types.NumericField) TrainLabel), prows=0, pcols=0, Maxrows=0, Maxcols=0):=FUNCTION

//initial parameters
P_num := Max (x, id); //the length of the parameters vector (number of parameters)
ExtractGrad (DATASET(Types.NumericField) inp) := FUNCTION
  RETURN inp (id <= P_num);
END;
ExtractCost (DATASET(Types.NumericField) inp) := FUNCTION
  RETURN inp (id = (P_num+1))[1].value;
END;
Bracket1no := DATASET([{1,1,-1,10}], Mat.Types.MUElement); //the result of the bracketing algorithm
Bracket2no := DATASET([{1,1,-1,11}], Mat.Types.MUElement); //the result of the bracketing algorithm

emptyE := DATASET([], Mat.Types.Element);
LSiter := 0;
IsNotLegal (DATASET (Mat.Types.Element) Mat) := FUNCTION //???to be defined
  
  RETURN FALSE;
END;

ArmijoBacktrack4 (DATASET (Mat.Types.MUElement) inputpp) := FUNCTION // to be defined with recieving real parameters (should be a macro similar to this one)
  
  RETURN inputpp;
END;

polyinterp (REAL8 t_1, REAL8 f_1, REAL8 gtd_1, REAL8 t_2, REAL8 f_2, REAL8 gtd_2) := FUNCTION
  d1 := gtd_1 + gtd_2 - (3*((f_1-f_2)/(t_1-t_2)));
  d2 := SQRT ((d1*d1)-(gtd_1*gtd_2)); 
  d2real := TRUE; //check it ???
  temp := IF (d2real,t_2 - ((t_2-t_1)*((gtd_2+d2-d1)/(gtd_2-gtd_1+(2*d2)))),-100);
  temp100 := temp =-100;
  polResult := IF (temp100,(t_1+t_2)/2,MIN([MAX([temp,t_1]),t_2]));
  RETURN polResult;
END;

//OK
WolfeBracketing ( Real8 fNew, Real8 fPrev, Real8 gtdNew, REAL8 gtdPrev, REAL8 tt, REAL8 tPrev, DATASET(Mat.Types.Element) gNew, DATASET(Mat.Types.Element) gPrev, UNSIGNED8 inputFunEval, UNSIGNED8 BrackLSiter) := FUNCTION
  SetBrackets (REAL8 t1, REAL8 t2, REAL8 fval1, REAL8 fval2, DATASET(Mat.Types.Element) gval1 , DATASET(Mat.Types.Element) gval2) := FUNCTION
    t1no := DATASET([{1,1,t1,10}], Mat.Types.MUElement); //the result of the bracketing algorithm
    t2no := DATASET([{1,1,t2,11}], Mat.Types.MUElement); //the result of the bracketing algorithm
    fval1no := DATASET([{1,1,fval1,12}], Mat.Types.MUElement); //the result of the bracketing algorithm
    fval2no := DATASET([{1,1,fval2,13}], Mat.Types.MUElement); //the result of the bracketing algorithm
    gval1no := Mat.MU.To (gval1,14); //the result of the bracketing algorithm
    gval2no := Mat.MU.To (gval2,15); //the result of the bracketing algorithm
    FEnochange := DATASET([{1,1,inputFunEval,7}], Mat.Types.MUElement);
    RETURN t1no + t2no + fval1no + fval2no + gval1no + gval2no +FEnochange;
  END;

  SetNewValues () := FUNCTION
    //t_prev = t;
    tPrevno := DATASET([{1,1,tt,6}], Mat.Types.MUElement);
    // minStep = t + 0.01*(t-temp);
    // maxStep = t*10;
    minstep := tt + 0.01* (tt-tPrev);
    maxstep := tt*10;
    //t = polyinterp([temp f_prev gtd_prev; t f_new gtd_new],doPlot,minStep,maxStep);
    newt := polyinterp (tPrev, fPrev,gtdPrev, tt, fNew, gtdNew);
    newtno := DATASET([{1,1,newt,5}], Mat.Types.MUElement);
    // f_prev = f_new;
    // g_prev = g_new;
    // gtd_prev = gtd_new;
    fPrevno := DATASET([{1,1,fNew,1}], Mat.Types.MUElement);
    gPrevno := Mat.MU.To (gNew,3);
    gtdPrevno:= DATASET([{1,1,gtdNew,8}], Mat.Types.MUElement);
    //calculate fnew gnew gtdnew
    xNew := ML.Types.FromMatrix (ML.Mat.Add(ML.Types.ToMatrix(x),ML.Mat.Scale(ML.Types.ToMatrix(d),newt)));
    CostGradNew := CostFunc (xNew ,CostFunc_params,TrainData, TrainLabel);
    gNewwolfe := ExtractGrad (CostGradNew);
    fNewWolfe := ExtractCost (CostGradNew);
    gtdNewWolfe := ML.Mat.Mul (ML.Mat.Trans(ML.Types.ToMatrix(gNewwolfe)),ML.Types.ToMatrix(d));
    gNewwolfeno := Mat.MU.To (ML.Types.ToMatrix(gNewwolfe),4);
    fNewWolfeno := DATASET([{1,1,fNewWolfe,2}], Mat.Types.MUElement);
    gtdNewWolfeno := Mat.MU.To (gtdNewWolfe,9);
    FEno := DATASET([{1,1,inputFunEval + 1,7}], Mat.Types.MUElement);
    Rno := fPrevno + fNewWolfeno + gPrevno + gNewwolfeno + newtno + tPrevno + FEno + gtdPrevno + gtdNewWolfeno + Bracket1no + Bracket2no;
   RETURN Rno;
  END;

  //If the strong wolfe conditions satisfies then retun the final t or the bracket, otherwise do the next iteration
  //f_new > f + c1*t*gtd || (LSiter > 1 && f_new >= f_prev)
  con1 := (fNew > f + c1 * tt* gtd)| ((BrackLSiter > 1) & (fNew >= fPrev)) ;
  //abs(gtd_new) <= -c2*gtd
  con2 := ABS(gtdNew) <= (-1*c2*gtd); 
  // gtd_new >= 0
  con3 := gtdNew >= 0;
  //
  BracketValues := IF (con1, SetBrackets (tPrev,tt,fPrev,fNew, gPrev, gNew), IF (con2,SetBrackets (tt,-1,fNew,-1, gNew, EmptyE),IF (con3, SetBrackets (tPrev,tt,fPrev,fNew, gPrev, gNew),SetNewValues ()) ));
  //If the conditions have been satisfied then only the final interval or final t is returned in the BracketValues, otherwise
  //a new t is evaluated and all the new values for f_new, f_prev, etc. are returned in the BracketValues
  RETURN BracketValues;
END;





//WI : Wolfe Interval 
WolfeZooming (DATASET (Mat.Types.MUElement) WI, INTEGER coun) := FUNCTION
  t_first  := Mat.MU.From (WI,10)[1].value;
  t_second := Mat.MU.From (WI,11)[1].value;
  f_first  := Mat.MU.From (WI,12)[1].value;
  f_second := Mat.MU.From (WI,13)[1].value;
  g_first  := Mat.MU.From (WI,14);
  g_second := Mat.MU.From (WI,15);
  gtd_first := ML.Mat.Mul (ML.Mat.Trans (g_first),ML.Types.ToMatrix(d));
  gtd_second := ML.Mat.Mul (ML.Mat.Trans (g_second),ML.Types.ToMatrix(d));
  //
  // Find High and Low Points in bracket
  LOt_i := IF (f_first < f_second, 10 , 11);
  LOf_i := LOt_i + 2;
  LOg_i := LOt_i + 4;
  HIt_i := -1 * LOt_i + 21;
  HIf_i := -1 * LOf_i + 25;
  HIg_i := -1 * LOg_i + 29;
  //
  LOt := Mat.MU.From (WI,LOt_i);
  HIt := Mat.MU.From (WI,HIt_i);
  LOf := Mat.MU.From (WI,LOf_i);
  HIf := Mat.MU.From (WI,HIf_i);
  LO_g := Mat.MU.From (WI,LOg_i);
  HIg := Mat.MU.From (WI,HIg_i);
  //
  // Compute new trial value
  //t = polyinterp([bracket(1) bracketFval(1) bracketGval(:,1)'*d bracket(2) bracketFval(2) bracketGval(:,2)'*d],doPlot);
  tTmp := polyinterp (t_first, f_first, gtd_first[1].value, t_second, f_second, gtd_second[1].value);
  //
  //Test that we are making sufficient progress
  insufProgress := (BOOLEAN)Mat.MU.From (WI,300)[1].value;
  BList := [t_first,t_second];
  MaxB := MAX (BList);
  MinB := MIN (BList);
  //if min(max(bracket)-t,t-min(bracket))/(max(bracket)-min(bracket)) < 0.1
  MainPCond := (MIN ((MAXB - tTmp) , (tTmp - MINB)) / (MAXB - MINB) ) < 0.1 ;
  //if insufProgress || t>=max(bracket) || t <= min(bracket)
  PCond2 := insufProgress | (tTmp >= MAXB) | (tTmp <= MINB);
  //abs(t-max(bracket)) < abs(t-min(bracket))
  PCond2_1 := ABS (tTMP - MAXB) < ABS (tTmp - MINB);
  // t = max(bracket)-0.1*(max(bracket)-min(bracket));
  MMTemp := 0.1 * (MAXB - MINB);
  tIF    := MAXB - MMTemp;
  // t = min(bracket)+0.1*(max(bracket)-min(bracket));
  tELSE := MINB + MMTemp;
  tZOOM := IF (MainPCond,IF (PCond2, IF (PCond2_1, tIF, tELSE) , tTmp),tTmp);
  insufProgress_new := IF (MainPCond, IF (PCond2, 0, 1) , 0);
  //
  // Evaluate new point with tZoom
  x_td := ML.Types.FromMatrix (ML.Mat.Add(ML.Types.ToMatrix(x),ML.Mat.Scale(ML.Types.ToMatrix(d),tZOOM)));
  CG_New := CostFunc (x_td ,CostFunc_params,TrainData, TrainLabel);
  gNew := ExtractGrad (CG_New);
  fNew := ExtractCost(CG_New);
  gtdNew := ML.Mat.Mul (ML.Mat.Trans(ML.Types.ToMatrix(gNew)),ML.Types.ToMatrix(d));
  inputZFunEval := Mat.MU.From (WI,7)[1].value;
  ZoomFunEvalno := DATASET([{1,1,inputZFunEval + 1,7}], Mat.Types.MUElement);
  
  SetIntervalIF1 := FUNCTION
    //bracket(HIpos) = t;
    bracket_HIt := DATASET([{1,1,tZOOM,HIt_i}], Mat.Types.MUElement);;
    //bracketFval(HIpos) = f_new;
    bracket_HIf := DATASET([{1,1,fNew,HIf_i}], Mat.Types.MUElement);
    //bracketGval(:,HIpos) = g_new;
    bracket_HIg := Mat.MU.To (ML.Types.ToMatrix(gNew),HIg_i);
    done := DATASET([{1,1,0,200}], Mat.Types.MUElement);
    RETURN bracket_HIt + bracket_HIf + bracket_HIg + WI(no = LOt_i) + WI (no = LOf_i) + WI (no = LOg_i) + done ;
  END;
  SetIntervalELSE1 := FUNCTION
   //bracket(LOpos) = t;
    bracket_LOt := DATASET([{1,1,tZOOM,LOt_i}], Mat.Types.MUElement);
    //bracketFval(LOpos) = f_new;
    bracket_LOf := DATASET([{1,1,fNew,LOf_i}], Mat.Types.MUElement);
    //bracketGval(:,LOpos) = g_new;
    bracket_LOg := Mat.MU.To (ML.Types.ToMatrix(gNew),LOg_i);
    done := DATASET([{1,1,0,200}], Mat.Types.MUElement);
    RETURN bracket_LOt + bracket_LOf + bracket_LOg + WI (no = HIt_i) + WI (no = HIf_i) + WI (no = HIg_i) + done;
  END;
  SETIntervalELSE1_1 := FUNCTION
    //bracket(LOpos) = t;
    bracket_LOt := DATASET([{1,1,tZOOM,LOt_i}], Mat.Types.MUElement);
    //bracketFval(LOpos) = f_new;
    bracket_LOf := DATASET([{1,1,fNew,LOf_i}], Mat.Types.MUElement);
    //bracketGval(:,LOpos) = g_new;
    bracket_LOg := Mat.MU.To (ML.Types.ToMatrix(gNew),LOg_i);
    done := DATASET([{1,1,1,200}], Mat.Types.MUElement);
    RETURN bracket_LOt + bracket_LOf + bracket_LOg + WI (no = HIt_i) + WI (no = HIf_i) + WI (no = HIg_i) + done;
  END;
  SETIntervalELSE1_2 := FUNCTION
    //bracket(LOpos) = t;
    bracket_LOt := DATASET([{1,1,tZOOM,LOt_i}], Mat.Types.MUElement);
    //bracketFval(LOpos) = f_new;
    bracket_LOf := DATASET([{1,1,fNew,LOf_i}], Mat.Types.MUElement);
    //bracketGval(:,LOpos) = g_new;
    bracket_LOg := Mat.MU.To (ML.Types.ToMatrix(gNew),LOg_i);
    // bracket(HIpos) = bracket(LOpos);
    bracket_HIt := WI(no = LOt_i);
    // bracketFval(HIpos) = bracketFval(LOpos);
    bracket_HIf := WI (no = LOf_i);
    // bracketGval(:,HIpos) = bracketGval(:,LOpos);
    bracket_HIg := WI (no = LOg_i);
    done := DATASET([{1,1,0,200}], Mat.Types.MUElement);
    RETURN bracket_LOt + bracket_LOf + bracket_LOg + bracket_HIt + bracket_HIf + bracket_HIg + done;
  END;
  //IF f_new > f + c1*t*gtd || f_new >= f_LO
  ZoomCon1 := (fNew > f + c1 * tZoom * gtd) | (fNew >LOf[1].value);
  //if abs(gtd_new) <= - c2*gtd
  ZOOMCon1_1 := ABS (gtdNew[1].value) <= (-1 * c2 * gtd);
  //gtd_new*(bracket(HIpos)-bracket(LOpos)) >= 0
  ZOOMCon1_2 := (gtdNew[1].value * (HIt[1].value - LOt[1].value)) >= 0;
  ZOOOMResult := IF (ZoomCon1, SetIntervalIF1, (IF(ZOOMCon1_1, SETIntervalELSE1_1, IF (ZOOMCon1_2,SETIntervalELSE1_2, SetIntervalELSE1 ))));
  //~done && abs((bracket(1)-bracket(2))*gtd_new) < tolX
  ZOOMTermination :=( (Mat.MU.FROM (ZOOOMResult,200)[1].value = 0) & (ABS((gtdNew[1].value * (t_first-t_second)))<tolX) ) | (Mat.MU.FROM (ZOOOMResult,200)[1].value = 1);
  ZOOMTermination_num := (INTEGER)ZOOMTermination;
  ZOOMFinalResult := ZOOOMResult (no<200) + DATASET([{1,1,ZOOMTermination_num,200}], Mat.Types.MUElement)+ DATASET([{1,1,insufProgress_new,300}], Mat.Types.MUElement) +ZoomFunEvalno ;
  RETURN ZOOMFinalResult;
END;


//x_new = x+t*d 
x_new := ML.Types.FromMatrix (ML.Mat.Add(ML.Types.ToMatrix(x),ML.Mat.Scale(ML.Types.ToMatrix(d),t)));
// Evaluate the cost and Gradient at the Initial Step
CostGrad_new := CostFunc (x_new ,CostFunc_params,TrainData, TrainLabel);
g_new := ExtractGrad (CostGrad_new);
f_new := ExtractCost (CostGrad_new);
funEvals := 1;
//gtd_new = g_new'*d;
gtd_new := ML.Mat.Mul (ML.Mat.Trans(ML.Types.ToMatrix(g_new)),ML.Types.ToMatrix(d));

// Bracket an Interval containing a point satisfying the Wolfe Criteria
t_prev := 0;
f_prev := f;
g_prev := g;
gtd_prev := gtd;

//Bracketing algorithm, either produces the final t value or a bracket that contains the final t value
//prepare the parameters to be passed to the bracketing algorithm
f_prevno := DATASET([{1,1,f_prev,1}], Mat.Types.MUElement);
f_newno := DATASET([{1,1,f_new,2}], Mat.Types.MUElement);
g_prevno := Mat.MU.To (ML.Types.ToMatrix(g_prev),3);
g_newno := Mat.MU.To (ML.Types.ToMatrix(g_new),4);
tno := DATASET([{1,1,t,5}], Mat.Types.MUElement);
t_prevno := DATASET([{1,1,t_prev,6}], Mat.Types.MUElement);
funEvalsno := DATASET([{1,1,funEvals,7}], Mat.Types.MUElement);
gtd_prevno := DATASET([{1,1,gtd_prev,8}], Mat.Types.MUElement);
gtd_newno := DATASET([{1,1,gtd_new[1].value,9}], Mat.Types.MUElement);
//at the begining no interval or final t value is found so the assigned values to Bracket1no and Bracket2no are -1
//In each iteration these two values are checked to see if the interval is found (both should be ~-1) or the final t is found (just the first one should be ~-1)
Topass := f_prevno + f_newno + g_prevno + g_newno + tno + t_prevno + funEvalsno + gtd_prevno + gtd_newno + Bracket1no + Bracket2no  ;
Bracketing (DATASET (Mat.Types.MUElement) inputp, INTEGER coun) := FUNCTION
  fi_prev :=  Mat.MU.From (inputp,1);
  fi_new := Mat.MU.From (inputp,2);
  gi_prev :=  Mat.MU.From (inputp,3);
  gi_new := Mat.MU.From (inputp,4);
  ti :=  Mat.MU.From (inputp,5);
  ti_prev :=  Mat.MU.From (inputp,6);
  FunEvalsi := Mat.MU.From (inputp,7);
  gtdi_prev := Mat.MU.From (inputp,8);
  gtdi_new := Mat.MU.From (inputp,9);
  AreTheyLegal := IsNotLegal(fi_new) | IsNotLegal(gi_new);
  //armijo only returns final t results and then the loop will stop becasue bracket1 would be ~-1
  WolfeH := WolfeBracketing ( fi_new[1].value, fi_prev[1].value, gtdi_new[1].value, gtdi_prev[1].value, ti[1].value, ti_prev[1].value, gi_new, gi_prev, FunEvalsi[1].value, (coun-1));
  Bracketing_Result := IF (AreTheyLegal, ArmijoBacktrack4(inputp), WolfeH );
  tobereturn := Bracketing_Result + DATASET([{1,1,coun,100}], Mat.Types.MUElement);
  RETURN tobereturn;
END;
Bracketing_Result := LOOP(Topass, COUNTER <= maxLS AND Mat.MU.From (ROWS(LEFT),10)[1].value = -1, Bracketing(ROWS(LEFT),COUNTER));
FoundInterval := Bracketing_Result (no = 10) + Bracketing_Result (no = 11) + Bracketing_Result (no = 12) + Bracketing_Result (no = 13) + Bracketing_Result (no = 14) + Bracketing_Result (no = 15);
Interval_Found := Mat.MU.From (Bracketing_Result,10)[1].value != -1 AND Mat.MU.From (Bracketing_Result,11)[1].value !=-1;
final_t_found := Mat.MU.From (Bracketing_Result,10)[1].value != -1 AND Mat.MU.From (Bracketing_Result,11)[1].value =-1;
ItrExceedInterval := DATASET([{1,1,0,10},
{1,1,Mat.MU.From (Bracketing_Result,5)[1].value ,11},
{1,1,f ,12},
{1,1,Mat.MU.From (Bracketing_Result,2)[1].value ,13}
], Mat.Types.MUElement) + Mat.MU.To (ML.Types.ToMatrix(g),14) + Mat.MU.To (Mat.MU.FROM(Bracketing_Result,4),15);
//
Zoom_Max_itr_tmp :=  maxLS - Mat.MU.From (Bracketing_Result,100)[1].value;
Zoom_Max_Itr := IF (Zoom_Max_itr_tmp >0, Zoom_Max_itr_tmp, 0);
TOpassZOOM := FoundInterval + DATASET([{1,1,0,200}], Mat.Types.MUElement) + DATASET([{1,1,0,300}], Mat.Types.MUElement) + Bracketing_Result (no = 7); // pass the found interval + {done=0} to Zoom LOOP +insufficientProgress+FunEval
ZOOMInterval := LOOP(TOpassZOOM, COUNTER <= Zoom_Max_Itr AND Mat.MU.From (ROWS(LEFT),200)[1].value = 0, WolfeZooming(ROWS(LEFT), COUNTER));
FinalBracket := IF (final_t_found, FoundInterval, IF (Interval_Found,ZOOMInterval,ItrExceedInterval));
WolfeOut :=FinalBracket;
RETURN WolfeOut;
END;