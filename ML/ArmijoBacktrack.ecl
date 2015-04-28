//
// Backtracking linesearch to satisfy Armijo condition
//
// Inputs:
//   x: starting location
//   t: initial step size
//   d: descent direction
//   f: function value at starting location
//   gtd: directional derivative at starting location
//   c1: sufficient decrease parameter
//   tolX: minimum allowable step length
//   CostFunc: objective function
//   CostFunc_params: parameters of objective function
EXPORT ArmijoBacktrack(ArmOut,x,t,d,f,g,gtd,c1,tolX,CostFunc,CostFunc_params='', TrainData ='', TrainLabel='', prows=0, pcols=0, Maxrows=0, Maxcols=0):=MACRO
P_num := Max (x, id); //the length of the parameters vector
// Evaluate the Objective and Gradient at the Initial Step
//x_new = x+t*d 
x_new := ML.Types.FromMatrix (ML.Mat.Add(ML.Types.ToMatrix(x),ML.Mat.Scale(ML.Types.ToMatrix(d),t)));
//[f_new,g_new] = feval(funObj, x_new, varargin{:});
CostGrad_new := CostFunc (x_new ,CostFunc_params,TrainData, TrainLabel);
g_new := CostGrad_new (id<=P_num);
f_new := CostGrad_new (id = P_num+1)[1].value;
funEvals := 1;
//gtd_new = g_new'*d;
//gtd_new := ML.Mat.Mul (ML.Mat.Trans(ML.Types.ToMatrix(g_new)),ML.Types.ToMatrix(d));
Bracketing (DATASET (Mat.Types.MUElement) inputp, INTEGER coun) := FUNCTION
  RETURN 2;
END;
ArmOut := x;
ENDMACRO;