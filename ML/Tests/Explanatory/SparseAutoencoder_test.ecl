IMPORT * FROM ML;
IMPORT * FROM $;
IMPORT PBblas;
Layout_Cell := PBblas.Types.Layout_Cell;
//Number of neurons in the last layer is number of output assigned to each sample
INTEGER4 hl := 2;//number of nodes in the hiddenlayer
INTEGER4 f := 3;//number of input features

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
UNSIGNED2 MaxIter :=10;
UNSIGNED4 prows:=0;
UNSIGNED4 pcols:=0;
UNSIGNED4 Maxrows:=0;
UNSIGNED4 Maxcols:=0;
//initialize weight and bias values for the Back Propagation algorithm
IntW := DeepLearning.Sparse_Autoencoder_IntWeights(f,hl);
Intb := DeepLearning.Sparse_Autoencoder_IntBias(f,hl);
output(IntW,ALL, named ('IntW'));
output(IntB,ALL, named ('IntB'));
//trainer module
SA :=DeepLearning.Sparse_Autoencoder(IntW, Intb,BETA, sparsityParam, LAMBDA, ALPHA, MaxIter, prows, pcols, Maxrows,  Maxcols);

LearnModel := SA.LearnC(indepDataC);
output(LearnModel, named ('LearnModel'));

LearnModel2 := SA.LearnC2(indepDataC);
output(LearnModel2, named ('LearnModel2'));
MatrixModel := SA.Model (LearnModel);
output(MatrixModel, named ('MatrixModel'));
