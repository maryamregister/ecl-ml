IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;

//Apply Gradient Des
EXPORT GradDes ( DATASET ($.M_Types.IDNUMRec) NodeNum, REAL8 LAMBDA, REAL8 ALPHA):= MODULE //NodeNum is 
//the number of neurons (nodes) in each layer which is in the format {layer number, number of nodes}
//layer number starts at 1 , for example NodeNum can be like [{1,3},{2, 5},{3,4}] which means that first layer has 3 nodes
//second layer has 5 nodes a nd third layer has 4 nodes
//LAMBDA : weight decay parameter
//ALPHA learning rate parameter

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
{2,2,0},
{2,3,1},
{3,1,0},
{3,2,1},
{3,3,0}],
$.M_Types.MatRecord);

//Iitialize the network (initialize Weight and Bias values and retun W and B)



//first initialize the weights to random values

W := $.IntWeights  (NodeNum);

//second initilize the Bias values to zero

B := $.IntBias (NodeNum);

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

END;