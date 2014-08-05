N := 4;
NodeNum := DATASET ([{1,4},{2,2},{3,5},{4,2}],$.M_Types.IDNUMRec);

 LAMBDA := 0.1;
 ALPHA := 0.1;





d := DATASET([
{1,1,0.1},
{1,2,0.9},
{1,3,0.6},
{2,1,0.4},
{2,2,0.3},
{2,3,0.1},
{3,1,0.2},
{3,2,0.4},
{3,3,0.7},
{4,1,0.4},
{4,2,0.5},
{4,3,0.3}],
$.M_Types.MatRecord);

Y := DATASET([
{1,1,1},
{1,2,0},
{1,3,0},
{2,1,0},
{2,2,1},
{2,3,1}],
$.M_Types.MatRecord);

YY := DATASET([
{1,1,1},
{1,2,0},
{1,3,0},
{2,1,0},
{2,2,1},
{2,3,1}],
$.M_Types.MatRecord);

MM := MAX (YY,YY.value);
OUTPUT  (MM, ALL, NAMED ('MM'));

OUTPUT (d, ALL, NAMED ('d'));
OUTPUT  (y, ALL, NAMED ('y'));

//Iitialize the network (initialize Weight and Bias values and retun W and B)



//first initialize the weights to random values

W := $.IntWeights  (NodeNum);
OUTPUT  (W, ALL, NAMED ('W'));
//second initilize the Bias values to zero

B := $.IntBias (NodeNum);

OUTPUT  (B, ALL, NAMED ('B'));

//apply feed forward pass

A := $.FF(d,W,B );

OUTPUT  (A, ALL, NAMED ('A'));

//call Wgrad, Bgrad and Cost cal from BPCost

//1- calculate DELTA

DELTA := $.BPcost( y, W, A, LAMBDA ).DELTA;




OUTPUT  (DELTA, ALL, NAMED ('DELTA'));
//2- calculate Weigh Gradeints 

WGrad := $.BPcost( y, W, A, LAMBDA ).Wgrad (DELTA);

OUTPUT  (Wgrad, ALL, NAMED ('Wgrad'));

//3- calculate Bias Gradients 
BGrad := $.BPcost( y, W, A, LAMBDA ).Bgrad (DELTA);

OUTPUT  (BGrad, ALL, NAMED ('BGrad'));



//Update W 
UpW := $.UpdateWB (W,  Wgrad,  ALPHA).Regular;
OUTPUT  (UpW, ALL, NAMED ('UpW'));

//UPdate B
UpB := $.UpdateWB (B,  Bgrad,  ALPHA).Regular;

OUTPUT  (UpB, ALL, NAMED ('UpB'));

add_num := MAX (W, W.id);
OUTPUT (add_num);

$.M_Types.CellMatRec addone (W l) := TRANSFORM
SELF.id := l.id+add_num;
SELF := l;
END;

w4 := PROJECT (W,addone(LEFT));


param := W4+B;
OUTPUT (param,ALL,NAMED('param'));


num_p := (COUNT (param) /2) -0.5;
Btmp := Param (Param.id <= num_p AND Param.id >0);
OUTPUT (Btmp,ALL,NAMED('Btmp'));
Wtmp :=  Param (Param.id > num_p );
OUTPUT (Wtmp,ALL,NAMED('Wtmp'));

$.M_Types.CellMatRec MinusNum (W l) := TRANSFORM
SELF.id := l.id-num_p;
SELF := l;
END;

w5 := PROJECT (Wtmp,MinusNum(LEFT));
OUTPUT (w5,ALL,NAMED('w5'));


new_w_g :=  $.GradDes ( d,  y, param,  LAMBDA,  ALPHA);

OUTPUT (new_w_g,ALL,NAMED('new_w_g'));

loopBody(DATASET ($.M_Types.CellMatRec) ds) :=
 $.GradDes ( d,  y, ds,  LAMBDA,  ALPHA);
		
		
Final_Updated_Param := LOOP(Param,  COUNTER <= 1,  loopBody(ROWS(LEFT)));	


OUTPUT (Final_Updated_Param,ALL,NAMED('Final_Updated_Param'));



final2 := $.GradDesLoop (  d, y,param,  LAMBDA,  ALPHA,  1).GDIterations;

OUTPUT (final2,ALL,NAMED('final2'));