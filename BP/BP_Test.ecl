IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;


// Number of iterations in back propagation algortihm
LoopNum := 3; 
// for the input data each colomn corresponds to one instance (sample)
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
// for the desired output values (class labeld for the samples) each colomns corresponds to one sample's output
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




//the NodeNum is the essential input which shows the structure of the network, each element of
//the nodeNum is like {layer number, number of nodes in that layer}
//obviously the first layer has the same number of nodes of the number of input data features
// and the output layer has the same number of nodes as the number of classes (if the problem is numeric prediction
//then the output layer has just one node)
//Please note that the number of nodes in each layer should be without considering biad node (bias nodes are seprately considered in the implementation)
NodeNum := DATASET ([{1,4},{2,2},{3,5},{4,3}],$.M_Types.IDNUMRec);


//LAMDA : weight decay parameter
//ALPHA : Learning rate parameter
LAMBDA := 0.1;
ALPHA := 0.1;

//initilize the weight and bias values (weights with randome samll number, bias with zeros)
W := $.IntWeights  (NodeNum);
OUTPUT  (W, ALL, NAMED ('W'));


B := $.IntBias (NodeNum);
OUTPUT  (B, ALL, NAMED ('B'));


//Maked PARAM and pass it to Gradietn Desent Function
add_num := MAX (W, W.id);
OUTPUT (add_num, NAMED('add_num'));

$.M_Types.CellMatRec addone (W l) := TRANSFORM
SELF.id := l.id+add_num;
SELF := l;
END;

Wadd := PROJECT (W,addone(LEFT));

OUTPUT  (Wadd, ALL, NAMED ('Wadd'));
Parameters := Wadd+B; //Now the ids related to B matrices are from 0 to n (number of layers)
// and ids for W matrices are from 1+n to n+n
// in the GradDes W and B matrix are going to be extracted from the "Parameters" again and it is
//done based on id values (the B matrix related to id=0 is not nessecary and do not need to be extracted);
OUTPUT  (Parameters, ALL, NAMED ('Parameters'));

Updated_Param:= $.GradDesLoop (  d, y,Parameters,  LAMBDA,  ALPHA,  LoopNum).GDIterations;
//now updated_parameters contain the updated weights and bias values. and you need to extract W and B matrices
//by considering weight ids as id+number(number od layers-1) of w matrices

OUTPUT  (Updated_Param, ALL, NAMED ('Updated_Param'));