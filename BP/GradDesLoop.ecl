IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;

//Apply Gradient Des  for LoopNum number of iterations
//d : data
//y : output

EXPORT GradDesLoop ( DATASET ($.M_Types.MatRecord) d, DATASET ($.M_Types.MatRecord) y, DATASET ($.M_Types.CellMatRec) IntParam, REAL8 LAMBDA, REAL8 ALPHA, UNSIGNED LoopNum):= MODULE  

SHARED GradDes ( DATASET ($.M_Types.CellMatRec) Param) := FUNCTION 


//EXtract W and B

num_p := (MAX (param, param.id) /2);

B := Param (Param.id <= num_p AND Param.id >0); // >0 bcz no bias for the input layer

Wtmp :=  Param (Param.id > num_p );

$.M_Types.CellMatRec MinusNum (Wtmp l) := TRANSFORM
SELF.id := l.id-num_p;
SELF := l;
END;

W := PROJECT (Wtmp,MinusNum(LEFT));




//apply feed forward pass

A := $.FF(d,W,B );

//call Wgrad, Bgrad and Cost cal from BPCost

//1- calculate DELTA

DELTA := $.BPcost( y, W, A, LAMBDA ).DELTA;

//2- calculate Weigh Gradeints 

WGrad := $.BPcost( y, W, A, LAMBDA ).Wgrad (DELTA);

//3- calculate Bias Gradients 
BGrad := $.BPcost( y, W, A, LAMBDA ).Bgrad (DELTA);

//Update W 
UpW := $.UpdateWB (W,  Wgrad,  ALPHA).Regular;

//UPdate B
UpB := $.UpdateWB (B,  Bgrad,  ALPHA).Regular;

//Make (wrap Upw and UpB ) Updated param and return it (add the id of updated Ws by add_num)
add_num := MAX (UpW, UpW.id);


$.M_Types.CellMatRec addone (UpW l) := TRANSFORM
SELF.id := l.id+add_num;
SELF := l;
END;

UpWadd := PROJECT (UpW,addone(LEFT));


UpdatedParam := UpWadd+UpB; 


RETURN UpdatedParam;
END;





EXPORT GDIterations := FUNCTION

//apply GradDes fucntion on Param for LoopNum number of iterations to update the param
//then return the updated param




loopBody(DATASET ($.M_Types.CellMatRec) ds) :=
 GradDes (ds);
		
		
Final_Updated_Param := LOOP(IntParam,  COUNTER <= LoopNum,  loopBody(ROWS(LEFT)));		





RETURN 	Final_Updated_Param;

END;










END;