IMPORT * FROM ML;
IMPORT * FROM $;
IMPORT PBblas;
Layout_Cell := PBblas.Types.Layout_Cell;
net := DATASET([
{1, 1, 4},
{2,1,2},
{3,1,3},
{4,1,5}],
Types.DiscreteField);
//pay attention intersect is added so first layer should have number of features+1 nodes
//input data
value_record := RECORD
  unsigned  id;
  real  f1;
  real  f2;
  real  f3;
  integer1  label;
END;
input_data := DATASET([
{1, 0.1, 0.2, 0.2,1},
{2, 0.8, 0.9,0.4, 2},
{3, 0.5, 0.9,0.5, 3},
{4, 0.8, 0.7, 0.8, 3},
{5, 0.9,0.1,0.1, 2},
{6, 0.1, 0.3,0.7, 1}],
 value_record);
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
labeldata_Format := RECORD
  input_data.id;
  input_data.label;
END;
label_table := TABLE(input_data,labeldata_Format);
OUTPUT  (label_table, ALL, NAMED ('label_table'));
ML.ToField(sample_table, indepDataC);
OUTPUT  (indepDataC, ALL, NAMED ('indepDataC'));
ML.ToField(label_table, depDataC);
OUTPUT  (depDataC, ALL, NAMED ('depDataC'));
label := PROJECT(depDataC,Types.DiscreteField);
OUTPUT  (label, ALL, NAMED ('label'));
IntW := utils.IntWeights(net);
Intb := utils.IntBias(net);
output(IntW,ALL, named ('IntW'));
output(IntB,ALL, named ('IntB'));
REAL8 ALPHA := 0.1;
REAL8 LAMBDA :=0.001;
UNSIGNED2 MaxIter :=10;
UNSIGNED4 prows:=0;
UNSIGNED4 pcols:=0;
UNSIGNED4 Maxrows:=0;
UNSIGNED4 Maxcols:=0;
trainer:= ML.Classify.BackPropagation(net,IntW, Intb,  LAMBDA, ALPHA, MaxIter, prows, pcols, Maxrows,  Maxcols);
model := trainer.testit(indepDataC, label);
OUTPUT  (Model, ALL, NAMED ('Model'));