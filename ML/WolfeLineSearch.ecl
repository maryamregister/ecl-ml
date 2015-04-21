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
EXPORT WolfeLineSearch(WolfeOut, x,t,d,f,g,gtd,c1,c2,maxLS,tolX,CostFunc,CostFunc_params='', TrainData ='', TrainLabel='', prows=0, pcols=0, Maxrows=0, Maxcols=0):=MACRO
emptyE := DATASET([], Mat.Types.Element);
LSiter := 1;
IsNotLegal (DATASET (Mat.Types.Element) Mat) := FUNCTION //???to be defined
  
  RETURN FALSE;
END;

ArmijoBacktrack (DATASET (Mat.Types.MUElement) inputpp) := FUNCTION // to be defined with recieving real parameters (should be a macro similar to this one)
  
  RETURN inputpp;
END;

Zoom_Selection (DATASET (Mat.Types.MUElement) inputpp) := FUNCTION
  
  RETURN inputpp;
END;

WolfeBracketing ( Real8 fNew, Real8 fPrev, Real8 gtdNew, REAL8 gtdPrev, REAL8 tt, REAL8 tPrev, DATASET(Mat.Types.Element) gNew, DATASET(Mat.Types.Element) gPrev) := FUNCTION
  SetBrackets (REAL8 t1, REAL8 t2, REAL8 fval1, REAL8 fval2, DATASET(Mat.Types.Element) gval1 , DATASET(Mat.Types.Element) gval2) := FUNCTION
    t1no := DATASET([{1,1,t1,8}], Mat.Types.MUElement); //the result of the bracketing algorithm
    t2no := DATASET([{1,1,t2,9}], Mat.Types.MUElement); //the result of the bracketing algorithm
    fval1no := DATASET([{1,1,fval1,10}], Mat.Types.MUElement); //the result of the bracketing algorithm
    fval2no := DATASET([{1,1,fval2,11}], Mat.Types.MUElement); //the result of the bracketing algorithm
    gval1no := Mat.MU.To (gval1,12); //the result of the bracketing algorithm
    gval2no := Mat.MU.To (gval2,13); //the result of the bracketing algorithm
    RETURN t1no + t2no + fval1no + fval2no + gval1no + gval2no;
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
   RETURN tPrevno;
  END;

  //If the strong wolfe conditions satisfies then retunr the final t or the bracket, otherwise do the next iteration
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
  //if not bracket values is assigned then it means you need to calculate new f,other wise just retunr whatever it is 123
  RETURN 5;
END;



//initial parameters
P_num := Max (x, id); //the length of the parameters vector

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
Bracket1no := DATASET([{1,1,-1,8}], Mat.Types.MUElement); //the result of the bracketing algorithm
Bracket2no := DATASET([{1,1,-1,9}], Mat.Types.MUElement); //the result of the bracketing algorithm
Bracketfval1no := DATASET([{1,1,-1,10}], Mat.Types.MUElement); //the result of the bracketing algorithm
Bracketfval2no := DATASET([{1,1,-1,11}], Mat.Types.MUElement); //the result of the bracketing algorithm
Bracketgval1no := DATASET([{1,1,-1,12}], Mat.Types.MUElement); //the result of the bracketing algorithm
Bracketgval2no := DATASET([{1,1,-1,13}], Mat.Types.MUElement); //the result of the bracketing algorithm
gtd_prevno := DATASET([{1,1,gtd_prev,14}], Mat.Types.MUElement);
gtd_newno := DATASET([{1,1,gtd_new[1].value,15}], Mat.Types.MUElement);
Topass := f_prevno + f_newno + g_prevno + g_newno + tno + t_prevno + funEvalsno + Bracket1no + Bracket2no   ;
Bracketing (DATASET (Mat.Types.MUElement) inputp, INTEGER coun) := FUNCTION
  fi_prev :=  Mat.MU.From (inputp,1);
  fi_new := Mat.MU.From (inputp,2);
  gi_prev :=  Mat.MU.From (inputp,3);
  gi_new := Mat.MU.From (inputp,4);
  ti :=  Mat.MU.From (inputp,5);
  ti_prev :=  Mat.MU.From (inputp,6);

  AreTheyLegal := IsNotLegal(fi_new) | IsNotLegal(gi_new);
  //Bracketing_Result := IF (AreTheyLegal, ArmijoBacktrack(inputp), WolfeBracketing(inputp) ); this is correct
  Bracketing_Result := IF (AreTheyLegal, ArmijoBacktrack(inputp), ArmijoBacktrack(inputp) );
  Counno := DATASET([{1,1,coun,14}], Mat.Types.MUElement);
  //update funeval
  tobereturn := Bracketing_Result + Counno;
  RETURN inputp;
END;




Bracketing_Result := LOOP(Topass, COUNTER <= maxLS AND Mat.MU.From (ROWS(LEFT),8)[1].value = -1, Bracketing(ROWS(LEFT),COUNTER));
Bracketing_zoom_interval := Mat.MU.From (Bracketing_Result,8);
exceed_max_itr := Mat.MU.From (Bracketing_Result,14)[1].value = maxLS;
maxLS_felan_result := Bracketing_Result;
final_t_found := Mat.MU.From (Bracketing_Result,8)[1].value != -1 AND Mat.MU.From (Bracketing_Result,9)[1].value =-1;
Pre_Final_Result := IF (exceed_max_itr, maxLS_felan_result, IF (final_t_found, Bracketing_Result, Zoom_Selection (Bracketing_Result)) );

//build the real final results from Pre_Final_Result
WolfeOut :=final_t_found;
ENDMACRO;