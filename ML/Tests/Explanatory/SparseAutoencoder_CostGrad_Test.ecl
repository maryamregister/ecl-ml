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
REAL8 BETA := 3;
REAL8 ALPHA := 0.1;
REAL8 LAMBDA :=0.1;
UNSIGNED2 MaxIter :=15;
UNSIGNED4 prows:=0;
UNSIGNED4 pcols:=0;
UNSIGNED4 Maxrows:=0;
UNSIGNED4 Maxcols:=0;
//initialize weight and bias values for the Back Propagation algorithm
//IntW := DeepLearning.Sparse_Autoencoder_IntWeights(f,hl); orig
IntW := DATASET([
{2	,3,	0.046472,	1},
{1	,1	,0.582412	,1},
{2	,1,	0.05823,	1},
{1	,2	,0.226398,	1},
{2,	2,	0.289924,	1},
{1	,3	,0.439007,	1},
{1	,1	,0.264311,	2},
{2	,2,	0.195966	,2},
{3,	2	,0.414649,	2},
{2	,1	,0.49252,	2},
{3,	1	,0.471306	,2},
{1	,2,	0.311734,	2}
],Mat.Types.MUElement);
Intb := DeepLearning.Sparse_Autoencoder_IntBias(f,hl);
OUTPUT(IntW,ALL, named ('IntW'));
OUTPUT(IntB,ALL, named ('IntB'));

//trainer module
SA :=DeepLearning.Sparse_Autoencoder (f, hl, 0, 0,0,0);

LearntModel := SA.LearnC(indepDataC,IntW, Intb,BETA, sparsityParam, LAMBDA, ALPHA, MaxIter);
//OUTPUT(LearntModel, named ('LearntModel'));

MatrixModel := SA.Model (LearntModel);
//OUTPUT(MatrixModel, named ('MatrixModel'));

lbfgs_model := SA.LearnC_lbfgs(indepDataC,IntW,  Intb, BETA,sparsityParam ,LAMBDA, MaxIter);
//OUTPUT(lbfgs_model,NAMED('lbfgs_model'));
//OUTPUT(SA.Model (lbfgs_model), named ('MatrixModel'));

// Out := SA.SAOutput (indepDataC, LearntModel);
// OUTPUT(Out, named ('Out'));

// Extractedweights := SA.ExtractWeights (LearntModel);
// OUTPUT(Extractedweights, named ('Extractedweights'));

// ExtractedBias := SA.ExtractBias (LearntModel);
// OUTPUT(ExtractedBias, named ('ExtractedBias'));

// dd := DATASET([{1,1,60,1}], ML.Mat.Types.MUElement);
// thisout := Mat.MU.myFrom (dd,98);
// OUTPUT(thisout, NAMED('thisout'));
// OUTPUT(COUNT (thisout));

// h := IF(5 <= 0, 8,ERROR('Recs not in order'));
// output(h);
SA_mine :=DeepLearning.Sparse_Autoencoder_mine (f, hl, 0, 0,0,0);

IntW1 := DATASET([
{2	,3,	0.046472},
{1	,1	,0.582412	},
{2	,1,	0.05823},
{1	,2	,0.226398},
{2,	2,	0.289924},
{1	,3	,0.439007}

],Mat.Types.Element);


IntW2 := DATASET([

{1	,1	,0.264311},
{2	,2,	0.195966	},
{3,	2	,0.414649},
{2	,1	,0.49252},
{3,	1	,0.471306	},
{1	,2,	0.311734}
],Mat.Types.Element);

Intb1 := DATASET([

{1	,1	,1},
{2	,1,	1	}
],Mat.Types.Element);

Intb2 := DATASET([

{1	,1	,1},
{2	,1,	1	},
{3,1,1}
],Mat.Types.Element);


OUTPUT(IntW,ALL, named ('IntW1'));
OUTPUT(IntW,ALL, named ('IntW2'));
OUTPUT(IntB,ALL, named ('IntB1'));
OUTPUT(IntB,ALL, named ('IntB2'));

lbfgs_model_mine := SA_mine.LearnC_lbfgs(indepDataC,IntW1, IntW2, Intb1, Intb2, BETA,sparsityParam ,LAMBDA, MaxIter);
OUTPUT(lbfgs_model_mine,NAMED('lbfgs_model_mine'));