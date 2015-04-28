// Bracketing Line Search to Satisfy Wolfe Conditions
//Source "Numerical Optimization Book" and Matlab implementaion of minFunc :
// M. Schmidt. minFunc: unconstrained differentiable multivariate optimization in Matlab. http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html, 2005.
//  x:  Starting Location (numeric field format)
//  t: Initial step size
//  d:  descent direction  (Pk in formula 3.1 in the book) (numeric field format)
//  g: gradient at starting location
//  gtd:  directional derivative at starting location =  gtd = g'*d. Deltafk*Pk in formula 3.6a
//  c1: sufficient decrease parameter (c1 in formula 3.6a)
//  c2: curvature parameter (c2 in formula 3.6b)
//  maxLS: maximum number of iterations
//  tolX: minimum allowable step length
//  CostFunc: objective function
//  CostFunc_params:  parameters for the objective function
//  TrainData and TrainLabel: train and label data for the objective fucntion
// the rest are PBblas parameters
//QUESTION : gtd is positive or negative?? based on con2 is wolfebracketing algorithm it should be negative : IT is negative bcz d is negative
//Define a general FINEVAL and add it in wolfebracketing and WolfeZoom ??????
EXPORT WolfeLineSearch(WolfeOut, x,t,d,f,g,gtd,c1,c2,maxLS,tolX,CostFunc,CostFunc_params='', TrainData ='', TrainLabel='', prows=0, pcols=0, Maxrows=0, Maxcols=0):=MACRO
//initial parameters
P_num := Max (x, id); //the length of the parameters vector
Bracket1no := DATASET([{1,1,-1,10}], Mat.Types.MUElement); //the result of the bracketing algorithm
Bracket2no := DATASET([{1,1,-1,11}], Mat.Types.MUElement); //the result of the bracketing algorithm

emptyE := DATASET([], Mat.Types.Element);
LSiter := 1;
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

WolfeBracketing ( Real8 fNew, Real8 fPrev, Real8 gtdNew, REAL8 gtdPrev, REAL8 tt, REAL8 tPrev, DATASET(Mat.Types.Element) gNew, DATASET(Mat.Types.Element) gPrev, UNSIGNED8 inputFunEval) := FUNCTION
  SetBrackets (REAL8 t1, REAL8 t2, REAL8 fval1, REAL8 fval2, DATASET(Mat.Types.Element) gval1 , DATASET(Mat.Types.Element) gval2) := FUNCTION
    t1no := DATASET([{1,1,t1,10}], Mat.Types.MUElement); //the result of the bracketing algorithm
    t2no := DATASET([{1,1,t2,11}], Mat.Types.MUElement); //the result of the bracketing algorithm
    fval1no := DATASET([{1,1,fval1,12}], Mat.Types.MUElement); //the result of the bracketing algorithm
    fval2no := DATASET([{1,1,fval2,13}], Mat.Types.MUElement); //the result of the bracketing algorithm
    gval1no := Mat.MU.To (gval1,14); //the result of the bracketing algorithm
    gval2no := Mat.MU.To (gval2,15); //the result of the bracketing algorithm
    RETURN t1no + t2no + fval1no + fval2no + gval1no + gval2no;
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
    gNewwolfe := CostGradNew (id<=P_num);
    fNewWolfe := CostGradNew (id = P_num+1)[1].value;
    gtdNewWolfe := ML.Mat.Mul (ML.Mat.Trans(ML.Types.ToMatrix(gNewwolfe)),ML.Types.ToMatrix(d));
    gNewwolfeno := Mat.MU.To (ML.Types.ToMatrix(gNewwolfe),4);
    fNewWolfeno := DATASET([{1,1,gtdNew,2}], Mat.Types.MUElement);
    gtdNewWolfeno := Mat.MU.To (gtdNewWolfe,9);
    FEno := DATASET([{1,1,inputFunEval + 1,7}], Mat.Types.MUElement);
    Rno := fPrevno + fNewWolfeno + gPrevno + gNewwolfeno + newtno + tPrevno + FEno + gtdPrevno + gtdNewWolfeno + Bracket1no + Bracket2no;
   RETURN Rno;
  END;

  //If the strong wolfe conditions satisfies then retun the final t or the bracket, otherwise do the next iteration
  //f_new > f + c1*t*gtd || (LSiter > 1 && f_new >= f_prev)
  con1 := (fNew > f + c1 * tt* gtd)| ((LSiter > 1) & (fNew >= fPrev)) ;
  //abs(gtd_new) <= -c2*gtd
  con2 := ABS(gtdNew) <= (-1*c2*gtd); 
  // gtd_new >= 0
  con3 := gtdNew >= 0;
  //assigne bracket values
  // bracket = [t_prev t];
  // bracketFval = [f_prev f_new];
  // bracketGval = [g_prev g_new];
  BracketValues := IF (con1, SetBrackets (tPrev,tt,fPrev,fNew, gPrev, gNew), IF (con2,SetBrackets (tt,-1,fNew,-1, gNew, EmptyE),IF (con3, SetBrackets (tPrev,tt,fPrev,fNew, gPrev, gNew),SetNewValues ()) ));
  //WolfeBracketing ( Real8 fNew, Real8 fPrev, Real8 gtdNew, REAL8 gtdPrev, REAL8 tt, REAL8 tPrev, DATASET(Mat.Types.Element) gNew, DATASET(Mat.Types.Element) gPrev) := FUNCTION
  //if not bracket values is assigned then it means you need to calculate new f,other wise just retunr whatever it is
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
  LOf_i := IF (f_first < f_second, 12 , 13);
  LOg_i := IF (f_first < f_second, 14 , 15);
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
  //Test that we are making sufficient progress
  //if min(max(bracket)-t,t-min(bracket))/(max(bracket)-min(bracket)) < 0.1
  insufProgress := FALSE;
  BList := [t_first,t_second];
  MainPCond := (MIN (MAX (BList) - tTmp , tTmp - MIN (BList)) / (MAX (BList) - MIN (BList)) ) < 0.1 ;
  //if insufProgress || t>=max(bracket) || t <= min(bracket)
  PCond2 := insufProgress | (tTmp >= MAX (BList)) | (tTmp <= MIN (BList));
  //abs(t-max(bracket)) < abs(t-min(bracket))
  PCond2_1 := ABS (tTMP - MAX (BList)) < ABS (tTmp - MIN (BList));
  // max(bracket)-0.1*(max(bracket)-min(bracket));
  tIF := MAX (BList) - (0.1 * (MAX (BList) - MIN (BList)));
  // t = min(bracket)+0.1*(max(bracket)-min(bracket));
  tELSE := MIN (BList) + (0.1 * (MAX (BList) - MIN (BList)));
  tZOOM := IF (MainPCond,IF (PCond2, IF (PCond2_1, tIF, tELSE) , tTmp),tTmp);
  insufProgress_new := IF (MainPCond, IF (PCond2, 0, 1) , 0);
  //
  // Evaluate new point
  x_td := ML.Types.FromMatrix (ML.Mat.Add(ML.Types.ToMatrix(x),ML.Mat.Scale(ML.Types.ToMatrix(d),tZOOM)));
  CG_New := CostFunc (x_td ,CostFunc_params,TrainData, TrainLabel);
  gNew := CG_New (id<=P_num);
  fNew := CG_New (id = P_num+1)[1].value;
  gtdNew := ML.Mat.Mul (ML.Mat.Trans(ML.Types.ToMatrix(gNew)),ML.Types.ToMatrix(d));
  //update funeval ???
  
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
  ZOOOMResult := IF (ZoomCon1, SetIntervalIF1, (IF(ZOOMCon1_1, SETIntervalELSE1_1, IF (ZOOMCon1_2,SETIntervalELSE1_2, WI+DATASET([{1,1,0,200}], Mat.Types.MUElement) ))));
  //~done && abs((bracket(1)-bracket(2))*gtd_new) < tolX
  ZOOMTermination := (Mat.MU.FROM (ZOOOMResult,200)[1].value = 0) & ((gtdNew[1].value * (t_first-t_second))<tolX);
  ZOOMTermination_num := (INTEGER)ZOOMTermination;
  ZOOMFinalResult := ZOOOMResult (no<200) + DATASET([{1,1,ZOOMTermination_num,200}], Mat.Types.MUElement);
  RETURN ZOOMFinalResult;
END;

//x_new = x+t*d 
x_new := ML.Types.FromMatrix (ML.Mat.Add(ML.Types.ToMatrix(x),ML.Mat.Scale(ML.Types.ToMatrix(d),t)));
//[f_new,g_new] = feval(funObj, x_new, varargin{:});
CostGrad_new := CostFunc (x_new ,CostFunc_params,TrainData, TrainLabel);
g_new := CostGrad_new (id<=P_num);
f_new := CostGrad_new (id = P_num+1)[1].value;
funEvals := 1;
//gtd_new = g_new'*d;
gtd_new := ML.Mat.Mul (ML.Mat.Trans(ML.Types.ToMatrix(g_new)),ML.Types.ToMatrix(d));

// Bracket an Interval containing a point satisfying the
t_prev := 0;
f_prev := f;
g_prev := g;
gtd_prev := gtd;

//Bracketing algorithm, either produces the final t value or a bracket that contains the final t value
found_t := 0; //have final t being found?
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
  //Bracketing_Result := IF (AreTheyLegal, ArmijoBacktrack(inputp), WolfeBracketing(inputp) ); this is correct
  //armijo only returns bracketing results and then the loop will stop becasue bracket1 would be ~1
  WolfeH := WolfeBracketing ( fi_new[1].value, fi_prev[1].value, gtdi_new[1].value, gtdi_prev[1].value, ti[1].value, ti_prev[1].value, gi_new, gi_prev, FunEvalsi[1].value);
  //ArmijoBacktrack(ArmOut,inputp);
  Bracketing_Result := IF (AreTheyLegal, ArmijoBacktrack4(inputp), WolfeH );
  tobereturn := Bracketing_Result + DATASET([{1,1,coun,100}], Mat.Types.MUElement); ;
  RETURN tobereturn;
END;




Bracketing_Result := LOOP(Topass, COUNTER <= maxLS AND Mat.MU.From (ROWS(LEFT),10)[1].value = -1, Bracketing(ROWS(LEFT),COUNTER));
FoundInterval := Bracketing_Result (no = 10) + Bracketing_Result (no = 11) + Bracketing_Result (no = 12) + Bracketing_Result (no = 13) + Bracketing_Result (no = 14) + Bracketing_Result (no = 15)  ;
Interval_Found := Mat.MU.From (Bracketing_Result,10)[1].value != -1 AND Mat.MU.From (Bracketing_Result,11)[1].value !=-1;
final_t_found := Mat.MU.From (Bracketing_Result,10)[1].value != -1 AND Mat.MU.From (Bracketing_Result,11)[1].value =-1;
ItrExceedInterval := DATASET([{1,1,0,10},
{1,1,Mat.MU.From (Bracketing_Result,5)[1].value ,11},
{1,1,f ,12},
{1,1,Mat.MU.From (Bracketing_Result,2)[1].value ,13}
], Mat.Types.MUElement) + Mat.MU.To (ML.Types.ToMatrix(g),14) + Mat.MU.To (Mat.MU.FROM(Bracketing_Result,4),15);



Zoom_Max_itr_tmp :=  maxLS - Mat.MU.From (Bracketing_Result,100)[1].value;
Zoom_Max_Itr := IF (Zoom_Max_itr_tmp >0, Zoom_Max_itr_tmp, 0);
FinalBracket := IF (final_t_found, FoundInterval, IF (Interval_Found,ItrExceedInterval,ItrExceedInterval) );
TOpassZOOM := FoundInterval + DATASET([{1,1,0,200}], Mat.Types.MUElement);
ZOOMInterval := LOOP(TOpassZOOM, COUNTER <= Zoom_Max_Itr AND Mat.MU.From (ROWS(LEFT),200)[1].value = 0, WolfeZooming(ROWS(LEFT), COUNTER));
//IntervalParams := IF (final_t_found,Bracketing_Result );
//Pre_Final_Result := IF (exceed_max_itr, maxLS_felan_result, IF (final_t_found, Bracketing_Result, Zoom_Selection (Bracketing_Result)) );

//build the real final results from Pre_Final_Result
WolfeOut :=1;
ENDMACRO;