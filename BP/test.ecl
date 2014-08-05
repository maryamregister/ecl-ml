IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;

// for the input data each colomn corresponds to one instance (sample)
d := DATASET([
{1,1,0.1},
{1,2,0.9},
{1,3,0.6},
{1,4,0.3},
{2,1,0.4},
{2,2,0.3},
{2,3,0.1},
{2,4,0.1},
{3,1,0.2},
{3,2,0.4},
{3,3,0.7},
{3,4,0.8},
{4,1,0.4},
{4,2,0.5},
{4,3,0.3},
{4,4,0.9},
{5,1,0.4},
{5,2,0.7},
{5,3,0.8},
{5,4,0.1}],
$.M_Types.MatRecord);
OUTPUT  (d, ALL, NAMED ('d'));

// for the desired output values (class labeld for the samples) each colomns corresponds to one sample's output
Y := DATASET([
{1,1,1},
{1,2,1},
{1,3,0},
{1,4,0},
{2,1,0},
{2,2,0},
{2,3,1},
{2,4,1}],
$.M_Types.MatRecord);

OUTPUT  (y, ALL, NAMED ('y'));
NodeNum := DATASET ([{1,5},{2,2},{3,3},{4,2}],$.M_Types.IDNUMRec);



LAMBDA := 0.1;
ALPHA := 0.1;

//initilize the weight and bias values (weights with randome samll number, bias with zeros)
W0 := $.IntWeights  (NodeNum);
OUTPUT  (W0, ALL, NAMED ('W0'));


B0 := $.IntBias (NodeNum);
OUTPUT  (B0, ALL, NAMED ('B0'));


//Maked PARAM and pass it to Gradietn Desent Function
add_num := MAX (W0, W0.id);
OUTPUT (add_num, NAMED('add_num'));

$.M_Types.CellMatRec addone (W0 l) := TRANSFORM
SELF.id := l.id+add_num;
SELF := l;
END;

Wadd := PROJECT (W0,addone(LEFT));

OUTPUT  (Wadd, ALL, NAMED ('Wadd'));
Parameters := Wadd+B0; //Now the ids related to B matrices are from 0 to n (number of layers)
// and ids for W matrices are from 1+n to n+n
// in the GradDes W and B matrix are going to be extracted from the "Parameters" again and it is
//done based on id values (the B matrix related to id=0 is not nessecary and do not need to be extracted);
OUTPUT  (Parameters, ALL, NAMED ('Parameters'));


param := Parameters;
LoopNum := 3;



num_p := (MAX (param, param.id) /2);

B:= Param (Param.id <= num_p AND Param.id >0); // >0 bcz no bias enters the input layer

Wtmp :=  Param (Param.id > num_p );

$.M_Types.CellMatRec MinusNum (Wtmp l) := TRANSFORM
SELF.id := l.id-num_p;
SELF := l;
END;

W := PROJECT (Wtmp,MinusNum(LEFT));


OUTPUT  (B, ALL, NAMED ('B'));
OUTPUT  (W, ALL, NAMED ('W'));
//Updated_Param:= $.GradDesLoop (  d, y,Parameters,  LAMBDA,  ALPHA,  3).GDIterations;
//now updated_parameters contain the updated weights and bias values. and you need to extract W and B matrices
//by considering weight ids as id+number(number od layers-1) of w matrices

//OUTPUT  (Updated_Param, ALL, NAMED ('Updated_Param'));









//apply feed forward pass

A := $.FF(d,W,B );
OUTPUT  (A, ALL, NAMED ('A'));
//call Wgrad, Bgrad and Cost cal from BPCost

//1- calculate DELTA

DELTA := $.BPcost( y, W, A, LAMBDA ).DELTA;
OUTPUT  (DELTA, ALL, NAMED ('DELTA'));
//2- calculate Weigh Gradeints 

WGrad := $.BPcost( y, W, A, LAMBDA ).Wgrad (DELTA);
OUTPUT  (WGrad, ALL, NAMED ('WGrad'));
//3- calculate Bias Gradients 
BGrad := $.BPcost( y, W, A, LAMBDA ).Bgrad (DELTA);
OUTPUT  (BGrad, ALL, NAMED ('BGrad'));

//Update W 
UpW := $.UpdateWB (W,  Wgrad,  ALPHA).Regular;
OUTPUT  (UpW, ALL, NAMED ('UpW'));

//UPdate B
UpB := $.UpdateWB (B,  Bgrad,  ALPHA).Regular;
OUTPUT  (UpB, ALL, NAMED ('UpB'));



WW := UpW;
BB := UpB;


//apply feed forward pass

AA := $.FF(d,WW,BB );
OUTPUT  (AA, ALL, NAMED ('AA'));
//call Wgrad, Bgrad and Cost cal from BPCost

//1- calculate DELTA

DELTAA := $.BPcost( y, WW, AA, LAMBDA ).DELTA;
OUTPUT  (DELTAA, ALL, NAMED ('DELTAA'));
//2- calculate Weigh Gradeints 

WGradd := $.BPcost( y, WW, AA, LAMBDA ).Wgrad (DELTAA);
OUTPUT  (WGradd, ALL, NAMED ('WGradd'));
//3- calculate Bias Gradients 
BGradd := $.BPcost( y, WW, AA, LAMBDA ).Bgrad (DELTAA);
OUTPUT  (BGradd, ALL, NAMED ('BGradd'));

//Update W 
UpWW := $.UpdateWB (WW,  Wgradd,  ALPHA).Regular;
OUTPUT  (UpWW, ALL, NAMED ('UpWW'));

//UPdate B
UpBB := $.UpdateWB (BB,  Bgradd,  ALPHA).Regular;
OUTPUT  (UpBB, ALL, NAMED ('UpBB'));


WWW := UpWW;
BBB := UpBB;


//apply feed forward pass

AAA := $.FF(d,WWW,BBB );
OUTPUT  (AAA, ALL, NAMED ('AAA'));
//call Wgrad, Bgrad and Cost cal from BPCost

//1- calculate DELTA

DELTAAA := $.BPcost( y, WWW, AAA, LAMBDA ).DELTA;
OUTPUT  (DELTAAA, ALL, NAMED ('DELTAAA'));
//2- calculate Weigh Gradeints 

WGraddd := $.BPcost( y, WWW, AAA, LAMBDA ).Wgrad (DELTAAA);
OUTPUT  (WGraddd, ALL, NAMED ('WGraddd'));
//3- calculate Bias Gradients 
BGraddd := $.BPcost( y, WWW, AAA, LAMBDA ).Bgrad (DELTAAA);
OUTPUT  (BGraddd, ALL, NAMED ('BGraddd'));

//Update W 
UpWWW := $.UpdateWB (WWW,  Wgraddd,  ALPHA).Regular;
OUTPUT  (UpWWW, ALL, NAMED ('UpWWW'));

//UPdate B
UpBBB := $.UpdateWB (BBB,  Bgraddd,  ALPHA).Regular;
OUTPUT  (UpBBB, ALL, NAMED ('UpBBB'));










Updated_Param:= $.GradDesLoop (  d, y,Parameters,  LAMBDA,  ALPHA,  3).GDIterations;
//now updated_parameters contain the updated weights and bias values. and you need to extract W and B matrices
//by considering weight ids as id+number(number od layers-1) of w matrices

OUTPUT  (Updated_Param, ALL, NAMED ('Updated_Param'));
