IMPORT * FROM ML;
IMPORT * FROM $;
IMPORT PBblas;
Layout_Cell := PBblas.Types.Layout_Cell;
net := DATASET([
{1, 1, 4},
{2,1,3},
{3,1,4},
{4,1,2}],
Types.DiscreteField);
//pay attention intersect is added so first layer should have number of features+1 nodes
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
//convert input data to two datset: samples dataset and labels dataset
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
IntW := utils.IntWeights(net);
Intb := utils.IntBias(net);
output(IntW,ALL, named ('IntW'));
output(IntB,ALL, named ('IntB'));
REAL8 ALPHA := 0.1;
REAL8 LAMBDA :=0.001;
UNSIGNED2 MaxIter :=1;
UNSIGNED4 prows:=0;
UNSIGNED4 pcols:=0;
UNSIGNED4 Maxrows:=0;
UNSIGNED4 Maxcols:=0;
trainer:= ML.Classify.BackPropagation(net,IntW, Intb,  LAMBDA, ALPHA, MaxIter, prows, pcols, Maxrows,  Maxcols);
model := trainer.testit(indepDataC, depDataC);

OUTPUT  (Model, ALL, NAMED ('Model'));
output (MAX (model,no));

output(ML.DMat.Converted.FromPart2Elm(PBblas.MU.From(model, 2)), ALL, named('wg2'));
output(ML.DMat.Converted.FromPart2Elm(PBblas.MU.From(model, 3)), ALL, named('wg3'));
output(ML.DMat.Converted.FromPart2Elm(PBblas.MU.From(model, 1)), ALL, named('wg1'));
output(ML.DMat.Converted.FromPart2Elm(PBblas.MU.From(model, 6)), ALL, named('wg6'));
output(ML.DMat.Converted.FromPart2Elm(PBblas.MU.From(model, 7)), ALL, named('wg7'));
output(ML.DMat.Converted.FromPart2Elm(PBblas.MU.From(model, 5)), ALL, named('wg5'));