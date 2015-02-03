IMPORT * FROM ML;
IMPORT * FROM $;
IMPORT PBblas;
Layout_Cell := PBblas.Types.Layout_Cell;
//net is the structure of the Back Propagation Network that shows number of neurons in each layer
//net is in NumericFiled format {id, number, value}, "value" is the number of nodes in the "id"th layer
//basically in the first layer number of neurons is : number of features
//Number of neurons in the last layer is number of output assigned to each sample
net := DATASET([
{1, 1, 3},
{2,1,2},
{3,1,3}],
Types.DiscreteField);

//input data

value_record := RECORD
  unsigned  id;
  real  f1;
  real  f2;
  real  f3;
END;
input_data := DATASET([
{1, 0.1, 0.2, 0.2},
{2, 0.8, 0.9,0.4},
{3, 0.5, 0.9,0.5},
{4, 0.8, 0.7, 0.8},
{5, 0.9,0.1,0.1},
{6, 0.1, 0.3,0.7}],
 value_record);

OUTPUT  (input_data, ALL, NAMED ('input_data'));

ML.ToField(input_data, indepDataC);
OUTPUT  (indepDataC, ALL, NAMED ('indepDataC'));

//define the parameters for the back propagation algorithm
//ALPHA is learning rate
//LAMBDA is weight decay rate
REAL8 sparsityParam  := 0.1;
REAL8 BETA := 0.1;
REAL8 ALPHA := 0.1;
REAL8 LAMBDA :=0.1;
UNSIGNED2 MaxIter :=1;
UNSIGNED4 prows:=0;
UNSIGNED4 pcols:=0;
UNSIGNED4 Maxrows:=0;
UNSIGNED4 Maxcols:=0;



//define the Neural Network Module to initialize the sparse autoencoder network
NN := NeuralNetworks(net);
//initialize weight and bias values for the Back Propagation algorithm
IntW := NN.IntWeights;
Intb := NN.IntBias;
output(IntW,ALL, named ('IntW'));
output(IntB,ALL, named ('IntB'));
//trainer module
trainer :=DeepLearning.Sparse_Autoencoder(IntW, Intb,BETA, sparsityParam, LAMBDA, ALPHA, MaxIter, prows, pcols, Maxrows,  Maxcols);

testsds := trainer.testit(indepDataC);
 // output(testsds, named ('testsds'));

// output(ML.DMat.Converted.FromPart2Elm(testsds), named ('testsds'));

t1 := ML.DMat.Converted.FromPart2Elm( PBblas.MU.From(testsds,1));
output(t1, named ('t1'));

t4 := ML.DMat.Converted.FromPart2Elm( PBblas.MU.From(testsds,4));
output(t4, named ('t4'));

t2 := ML.DMat.Converted.FromPart2Elm( PBblas.MU.From(testsds,2));
output(t2, named ('t2'));

t3 := ML.DMat.Converted.FromPart2Elm( PBblas.MU.From(testsds,3));
output(t3, named ('t3'));