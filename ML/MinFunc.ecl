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

EXPORT MinFunc(xout, x0, CostFunc, CostFunc_params='', TrainData ='', TrainLabel='', MaxIter, MethodOptions, prows=0, pcols=0, Maxrows=0, Maxcols=0):=MACRO

//initial parameters
P := Max (x0, id); //the length of the parameters vector
corrections := MethodOptions(id=(1))[1].value;//the number of correction vector that need to be kept in the memory


//Optimization Module
O := Optimization (prows, pcols, Maxrows, Maxcols).Limited_Memory_BFGS (P, corrections);



//initialize Hdiag,old_dir,old_steps, gradient and cost
//The size of the old_dir and old_steps matrices are "number of parameters" * "number of corrections to store in memory (corrections)"
// old_dir0 := Mat.Zeros(P, corrections);
// old_steps0 := old_dir0;
old_dir0 := DATASET([
{1, 1, 1},
{2,1,2},
{3,1,3},
{4,1,4},
{1,2,5},
{2,2,6},
{3,2,7},
{4,2,8}],
Mat.Types.Element);
old_steps0 := DATASET([
{1, 1, 11},
{2,1,12},
{3,1,13},
{4,1,14},
{1,2,15},
{2,2,16},
{3,2,17},
{4,2,18}],
Mat.Types.Element);;
Hdiag0no := DATASET([{1,1,1,5}], Mat.Types.MUElement);
CostGrad0 := CostFunc (x0,CostFunc_params,TrainData, TrainLabel);
g0 := CostGrad0 (id<=P);
Cost0 := CostGrad0 (id = P+1)[1].value;
C0n0 := DATASET([{1,1,Cost0,6}], Mat.Types.MUElement);

//put all the parameters that need to be sent to the step fucntion in a Mat.Types.MUElement format
x0no := Mat.MU.To (ML.Types.ToMatrix(x0),1);
g0n0 := Mat.MU.To (ML.Types.ToMatrix(g0),2);
old_steps0no := Mat.MU.To (old_steps0,3);
old_dir0no := Mat.MU.To (old_dir0,4);
Topass := x0no + g0n0 + old_steps0no + old_dir0no + Hdiag0no + C0n0;
//updating step function
step (DATASET (Mat.Types.MUElement) inputp) := FUNCTION
  xs := Mat.MU.From (inputp,1);
  gs := Mat.MU.From (inputp,2);
  Steps := Mat.MU.From (inputp,3);
  Dirs := Mat.MU.From (inputp,4);
  Hs := Mat.MU.From (inputp,5);
  HG_ := O.lbfgs(gs,Steps,Dirs,Hs);//the result is the approximate inverse Hessian, multiplied by the gradient, multiplied by -1 and it is in PBblas.layout format
  //find alpha that satisfies Wolfe Condition???
  //update the parameter vector:x_new = xold+alpha*HG_
  //For now consider the newx as the oldx????
  xs_updated := xs;
  x_Next := ML.Types.FromMatrix(xs_updated);
  CostGrad_Next := CostFunc (x_Next,CostFunc_params,TrainData, TrainLabel);
  g_Next := CostGrad_Next (id<=P);
  Cost_Next := CostGrad_Next (id = P+1)[1].value;
  x_Next_no := Mat.MU.To (ML.Types.ToMatrix(x_Next),1);//??? should be modified when the true updated x is calculated above
  g_Nextno := Mat.MU.To (ML.Types.ToMatrix(g_Next),2);
  C_Nextno := DATASET([{1,1,Cost_Next,6}], Mat.Types.MUElement);
  //calculate new Hessian diag, dir and steps
  Steps_Next := O.lbfgsUpdate_corr(xs, Steps);
  Steps_Nextno := Mat.MU.To (Steps_Next, 3);
  Dirs_Next := O.lbfgsUpdate_corr (gs, Dirs);
  Dirs_Nextno := Mat.MU.To (Dirs_Next, 4);
  RETURN HG_;
END; //END step
xout := step(Topass);
//xout := Topass;
ENDMACRO;