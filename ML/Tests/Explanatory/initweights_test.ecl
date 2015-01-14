IMPORT * FROM ML;
IMPORT * FROM $;
IMPORT PBblas;
Layout_Cell := PBblas.Types.Layout_Cell;
//The structure of the Back Propagation Network
//Number of neurons in each layer, {id, number, value}, value is number of nodes in the idth layer
//basically in the first layer number of neurons is : number of features + 1 (1 is added for the itersect term)
//Number of neurons in the last layer is number of output assigned to each sample
net := DATASET([
{1, 1, 4},
{2,1,3},
{3,1,4},
{4,1,2}],
Types.DiscreteField);

//input data
label_record := RECORD
unsigned  id;
  real  f1;
  real  f2;
END;
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
label := DATASET([
{1, 0.1, 0.2},
{2, 0.8,0.4},
{3, 0.5, 0.9},
{4,  0.7, 0.8},
{5, 0.9,0.1},
{6, 0.1, 0.3}],
label_record);
OUTPUT  (input_data, ALL, NAMED ('input_data'));
Sampledata_Format := RECORD
  input_data.id;
  input_data.f1;
  input_data.f2;
  input_data.f3;
END;
sample_table := TABLE(input_data,Sampledata_Format);
OUTPUT  (sample_table, ALL, NAMED ('sample_table'));
OUTPUT  (label, ALL, NAMED ('label'));

ML.ToField(sample_table, indepDataC);
OUTPUT  (indepDataC, ALL, NAMED ('indepDataC'));

ML.ToField(label, depDataC);
OUTPUT  (depDataC, ALL, NAMED ('depDataC'));
//define the parameters for the back propagation algorithm
//ALPHA is learning rate
//LAMBDA is weight decay rate
REAL8 ALPHA := 0.1;
REAL8 LAMBDA :=0.1;
UNSIGNED2 MaxIter :=3;
UNSIGNED4 prows:=0;
UNSIGNED4 pcols:=0;
UNSIGNED4 Maxrows:=0;
UNSIGNED4 Maxcols:=0;
NNtrainer := NeuralNetworks(net);
//initialize weight and bias values for the Back Propagation algorithm
IntW := NNtrainer.IntWeights;
Intb := NNtrainer.IntBias;
output(IntW,ALL, named ('IntW'));
output(IntB,ALL, named ('IntB'));
trainer :=NNtrainer.BackPropagation(IntW, Intb,  LAMBDA, ALPHA, MaxIter, prows, pcols, Maxrows,  Maxcols);
//trainer:= ML.Classify.BackPropagation(net,IntW, Intb,  LAMBDA, ALPHA, MaxIter, prows, pcols, Maxrows,  Maxcols);
Learntmodel := trainer.NNLearn(indepDataC, depDataC);

OUTPUT  (Learntmodel, ALL, NAMED ('Learntmodel'));
Mod := trainer.Model(Learntmodel);
OUTPUT  (mod, ALL, NAMED ('mod'));


FW := trainer.ExtractWeights(Learntmodel);
OUTPUT  (FW, ALL, NAMED ('FW'));

FB := trainer.ExtractBias(Learntmodel);
OUTPUT  (FB, ALL, NAMED ('FB'));

AEnd :=trainer.NNoutput(indepDataC,Learntmodel);
OUTPUT  (AEnd, ALL, NAMED ('AEnd'));

Class := trainer.NNClassify(indepDataC,Learntmodel);
OUTPUT  (Class, ALL, NAMED ('Class'));
//output(ML.DMat.Converted.FromPart2Elm(AEnd), ALL, named('AEnd_mat'));
// output(ML.DMat.Converted.FromPart2Elm(PBblas.MU.From(model, 2)), ALL, named('wg2'));
// output(ML.DMat.Converted.FromPart2Elm(PBblas.MU.From(model, 3)), ALL, named('wg3'));
// output(ML.DMat.Converted.FromPart2Elm(PBblas.MU.From(model, 1)), ALL, named('wg1'));
// output(ML.DMat.Converted.FromPart2Elm(PBblas.MU.From(model, 6)), ALL, named('wg6'));
// output(ML.DMat.Converted.FromPart2Elm(PBblas.MU.From(model, 7)), ALL, named('wg7'));
// output(ML.DMat.Converted.FromPart2Elm(PBblas.MU.From(model, 5)), ALL, named('wg5'));