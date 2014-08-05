IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;

//Apply Gradient Des
EXPORT GradDes ( DATASET ($.M_Types.MatRecord) d, DATASET ($.M_Types.MatRecord) y, DATASET ($.M_Types.CellMatRec) Param, REAL8 LAMBDA, REAL8 ALPHA):= FUNCTION //NodeNum is 
//the number of neurons (nodes) in each layer which is in the format {layer number, number of nodes}
//layer number starts at 1 , for example NodeNum can be like [{1,3},{2, 5},{3,4}] which means that first layer has 3 nodes
//second layer has 5 nodes a nd third layer has 4 nodes
//LAMBDA : weight decay parameter
//ALPHA learning rate parameter



//EXtract W and B

num_p := (MAX (param, param.id) /2);

B := Param (Param.id <= num_p AND Param.id >0);

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



//Maked Updated param and return it (add the id of updated Ws by add_num)
add_num := MAX (UpW, UpW.id);


$.M_Types.CellMatRec addone (UpW l) := TRANSFORM
SELF.id := l.id+add_num;
SELF := l;
END;

UpWadd := PROJECT (UpW,addone(LEFT));


UpdatedParam := UpWadd+UpB; 


RETURN UpdatedParam;
END;