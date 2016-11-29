IMPORT * FROM ML;
IMPORT STD;
IMPORT * FROM $;
IMPORT PBblas;
IMPORT STD;
Layout_Cell := PBblas.Types.Layout_Cell;
Layout_part := PBblas.Types.Layout_part;
// test Sparse_Autoencoder_lbfgs on an MNIST dataset which contains only five digits {0,1,2,3,4} : workunit W20160724-100930
INTEGER4 hl := 3;//number of nodes in the hiddenlayer
INTEGER4 f := 4;//number of input features

//input data
// check points
// data is distributed among all nodes and no node has extra parts of data with the same partition_id
// check mini batch number is consistent
//check correction numbers
value_record := RECORD
REAL8	f1	;
REAL8	f2	;
REAL8	f3	;
REAL8	f4	;
END;

input_data_tmp := DATASET([

{16,2,3,4},
{11,2,3,4},
{17,22,3,4},
{18,2,3,4},
{61,23,3,4},
{19,2,3,54},
{91,26,3,4},
{91,2,3,4},
{19,52,3,4},
{1,62,53,64},
{10,26,33,24},
{1,26,73,40},
{1,25,3,44},
{1,62,39,4},
{1,28,38,40},
{1,2,93,4},
{1,72,53,4},
{10,20,33,4},
{1,2,3,4},
{1,29,3,4},
{11,2,3,4},
{21,2,328,44},
{14,27,3,45},
{13,52,36,45},
{12,62,43,4},
{11,2,3,54},
{13,2,63,4},
{15,2,33,4},
{31,26,33,45},
{13,25,33,4},
{16,2,3,4},
{11,2,3,4},
{17,22,3,4},
{18,2,3,4},
{61,23,3,4},
{19,2,3,54},
{91,26,3,4},
{91,2,3,4},
{19,52,3,4},
{1,62,53,64},
{10,26,33,24},
{1,26,73,40},
{1,25,3,44},
{1,62,39,4},
{1,28,38,40},
{1,2,93,4},
{1,72,53,4},
{10,20,33,4},
{1,2,3,4},
{1,29,3,4},
{111,2,3,4},
{21,2,328,44},
{14,27,3,45},
{13,52,36,45},
{12,62,43,4},
{121,2,3,54},
{13,2,63,4},
{15,2,33,4},
{31,26,33,45},
{13,25,33,4},
{16,2,3,4},
{11,2,3,4},
{17,22,3,4},
{18,2,3,4},
{61,23,3,4},
{19,2,3,54},
{231,26,3,4},
{9,2,3,4},
{19,52,3,4},
{12,625,53,64},
{10,26,33,24},
{1,26,73,40},
{1,25,3,44},
{18,62,39,4},
{1,28,38,40},
{11,42,93,4},
{1,72,53,4},
{10,20,33,4},
{1,2,33,4},
{1,29,3,4},
{11,2,3,4},
{21,32,328,44},
{14,27,3,45},
{13,52,36,45},
{12,62,43,4},
{11,12,343,54},
{13,32,63,4},
{15,22,33,4},
{31,26,33,45},
{13,232,33,44},
{113,25,33,73},
{13,23,3,31},
{143,25,6,2},
{23,7,33,2},
{13,21,23,54},
{13,25,33,7},
{1,35,33,32},
{2,24,33,4},
{13,25,33,47},
{113,25,32,73},
{1,62,39,4},
{1,28,38,40},
{1,2,93,4},
{1,72,53,4},
{10,20,33,4},
{1,2,3,4},
{1,29,3,4},
{111,2,3,4},
{21,2,328,44},
{14,27,3,45},
{13,52,36,45},
{12,62,43,4},
{121,2,3,54},
{13,2,63,4},
{15,2,33,4},
{31,26,33,45},
{13,25,33,4},
{16,2,3,4},
{11,2,3,4},
{17,22,3,4},
{18,2,3,4},
{61,23,3,4},
{19,2,3,54},
{231,26,3,4},
{9,2,3,4},
{19,52,3,4},
{12,625,53,64},
{10,26,33,24},
{1,26,73,40},
{1,25,3,44},
{18,62,39,4},
{1,28,38,40},
{11,42,93,4},
{1,72,53,4},
{10,20,33,4},
{1,2,33,4},
{1,29,3,4},
{11,2,3,4},
{21,32,328,44},
{14,27,3,45},
{13,52,36,45},
{12,62,43,4},
{11,12,343,54},
{13,32,63,4},
{15,22,33,4},
{31,26,33,45},
{13,232,33,44},
{113,25,33,73},
{13,23,3,31},
{143,25,6,2},

{16,2,3,4},
{11,2,3,4},
{17,22,3,4},
{18,2,3,4},
{61,23,3,4},
{19,2,3,54},
{91,26,3,4},
{91,2,3,4},
{19,52,3,4},
{1,62,53,64},
{10,26,33,24},
{1,26,73,40},
{1,25,3,44},
{1,62,39,4},
{1,28,38,40},
{1,2,93,4},
{1,72,53,4},
{10,20,33,4},
{1,2,3,4},
{1,29,3,4},
{11,2,3,4},
{21,2,328,44},
{14,27,3,45},
{13,52,36,45},
{12,62,43,4},
{11,2,3,54},
{13,2,63,4},
{15,2,33,4},
{31,26,33,45},
{13,25,33,4},
{16,2,3,4},
{11,2,3,4},
{17,22,3,4},
{18,2,3,4},
{61,23,3,4},
{19,2,3,54},
{91,26,3,4},
{91,2,3,4},
{19,52,3,4},
{1,62,53,64},
{10,26,33,24},
{1,26,73,40},
{1,25,3,44},
{1,62,39,4},
{1,28,38,40},
{1,2,93,4},
{1,72,53,4},
{10,20,33,4},
{1,2,3,4},
{1,29,3,4},
{111,2,3,4},
{21,2,328,44},
{14,27,3,45},
{13,52,36,45},
{12,62,43,4},
{121,2,3,54},
{13,2,63,4},
{15,2,33,4},
{31,26,33,45},
{13,25,33,4},
{16,2,3,4},
{11,2,3,4},
{17,22,3,4},
{18,2,3,4},
{61,23,3,4},
{19,2,3,54},
{231,26,3,4},
{9,2,3,4},
{19,52,3,4},
{12,625,53,64},
{10,26,33,24},
{1,26,73,40},
{1,25,3,44},
{18,62,39,4},
{1,28,38,40},
{11,42,93,4},
{1,72,53,4},
{10,20,33,4},
{1,2,33,4},
{1,29,3,4},
{11,2,3,4},
{21,32,328,44},
{14,27,3,45},
{13,52,36,45},
{12,62,43,4},
{11,12,343,54},
{13,32,63,4},
{15,22,33,4},
{31,26,33,45},
{13,232,33,44},
{113,25,33,73},
{13,23,3,31},
{143,25,6,2},
{23,7,33,2},
{13,21,23,54},
{13,25,33,7},
{1,35,33,32},
{2,24,33,4},
{13,25,33,47},
{113,25,32,73},
{1,62,39,4},
{1,28,38,40},
{1,2,93,4},
{1,72,53,4},
{10,20,33,4},
{1,2,3,4},
{1,29,3,4},
{111,2,3,4},
{21,2,328,44},
{14,27,3,45},
{13,52,36,45},
{12,62,43,4},
{121,2,3,54},
{13,2,63,4},
{15,2,33,4},
{31,26,33,45},
{13,25,33,4},
{16,2,3,4},
{11,2,3,4},
{17,22,3,4},
{18,2,3,4},
{61,23,3,4},
{19,2,3,54},
{231,26,3,4},
{9,2,3,4},
{19,52,3,4},
{12,625,53,64},
{10,26,33,24},
{1,26,73,40},
{1,25,3,44},
{18,62,39,4},
{1,28,38,40},
{11,42,93,4},
{1,72,53,4},
{10,20,33,4},
{1,2,33,4},
{1,29,3,4},
{11,2,3,4},
{21,32,328,44},
{14,27,3,45},
{13,52,36,45},
{12,62,43,4},
{11,12,343,54},
{13,32,63,4},
{15,22,33,4},
{31,26,33,45},
{13,232,33,44},
{113,25,33,73},
{13,23,3,31},
{143,25,6,2}

], value_record);


/*
{23,7,33,2},
{13,21,23,54},
{13,25,33,7},
{1,35,33,32},
{2,24,33,4},
{13,25,33,47},
{113,25,32,73}
*/
//OUTPUT(input_data_tmp);
ML.AppendID(input_data_tmp, id, input_data);
sample_table := input_data;
ML.ToField(sample_table, indepDataC);
//OUTPUT(MAX(input_data,input_data.id));

//define the parameters for the Sparse Autoencoder
REAL8 sparsityParam  := 0.1;
REAL8 BETA := 3;
REAL8 ALPHA := 0.1;
REAL8 LAMBDA :=0.003;
UNSIGNED2 MaxIter :=10;
UNSIGNED4 prows:=0;
UNSIGNED4 pcols:=0;
UNSIGNED4 Maxrows:=0;
UNSIGNED4 Maxcols:=0;
//initialize weight and bias values for the Back Propagation algorithm
IntW := DeepLearning.Sparse_Autoencoder_IntWeights(f,hl);
Intb := DeepLearning.Sparse_Autoencoder_IntBias(f,hl);
SA_mine4_1 :=DeepLearning.Sparse_Autoencoder_lbfgs_part (f,hl,2,3,2);
//SA_mine4_1 :=DeepLearning.Sparse_Autoencoder_lbfgs (f,hl,62,73261,62,73261);
// SA_mine4_1 :=DeepLearning.Sparse_Autoencoder_lbfgs (f,hl,0,0,0,0);
// IntW1 := Mat.MU.From(IntW,1);
// IntW2 := Mat.MU.From(IntW,2);


IntW1 := DATASET ([{1,	1	,-0.5876242172298551},
{2	,1,	-0.5769582483568458},
{3	,1,	0.2302553702565914},
{1	,2,	0.8645289118866991},
{2,	2,	-0.507192973878779},
{3	,2,	-0.5404760621970226},
{1	,3,	0.04069453372383081},
{2	,3,	0.2883604786981045},
{3	,3,	-0.477077806487579},
{1	,4	,0.8586503314458102},
{2,	4	,0.3285648420433932},
{3	,4	,-0.851037968146545}
], Mat.Types.Element);

IntW2 := DATASET ([{1,	1,	-0.7101927926274666},
{2	,1,	-0.7924634739361807},
{3	,1,	0.4087051008587977},
{4	,1,	-0.05390142113154339},
{1	,2	,-0.6364489974444141},
{2,	2	,-0.0130787156479526},
{3	,2,	0.6428783700421098},
{4	,2,	-0.49690459208182},
{1	,3,	-0.4800344172160991},
{2	,3,	-0.7086651238151909},
{3,	3	,-0.1827642691638602},
{4	,3,	-0.6670075698423518}

], Mat.Types.Element);
//Intb1 := Mat.MU.From(Intb,1);
Intb1 := DATASET ([
{1	,1	,0.81},
{2	,1,	7},
{3	,1,	0.93}
],Mat.Types.Element);
//Intb2 := Mat.MU.From(Intb,2);

Intb2 := DATASET ([
{1	,1	,0.81},
{2	,1,	7},
{3	,1,	0.93},
{4	,1,	5}
],Mat.Types.Element);
// IntW1 := DeepLearning.Sparse_Autoencoder_IntWeights1(f,hl);
// IntW2 := DeepLearning.Sparse_Autoencoder_IntWeights2(f,hl);
// Intb1 := DeepLearning.Sparse_Autoencoder_IntBias1(f,hl);
// Intb2 := DeepLearning.Sparse_Autoencoder_IntBias2(f,hl);
OUTPUT(IntW1,ALL, named ('IntW1'));
OUTPUT(IntW2,ALL, named ('IntW2'));
OUTPUT(IntB1,ALL, named ('IntB1'));
OUTPUT(IntB2,ALL, named ('IntB2'));
OUTPUT(input_data);
// train the sparse autoencoer with train data
lbfgs_model_mine4_1 := SA_mine4_1.LearnC(indepDataC,IntW1, IntW2, Intb1, Intb2, BETA,sparsityParam ,LAMBDA, MaxIter,3);//the output includes the learnt parameters for the sparse autoencoder (W1,W2,b1,b2) in numericfield format
//OUTPUT(MAX(lbfgs_model_mine4_1,lbfgs_model_mine4_1.node_id));
//OUTPUT(MAX(lbfgs_model_mine4_1, lbfgs_model_mine4_1.node_id));
//OUTPUT(lbfgs_model_mine4_1, ALL);
output(input_data, ALL);
myrec := RECORD (Pbblas.Types.MuElement)
UNSIGNED real_node;

END;

thisrecjfdhgdhord := RECORD 
UNSIGNED real_node;
UNSIGNED pid;
UNSIGNED nid;
UNSIGNED no;
END;
// output(PROJECT(lbfgs_model_mine4_1, TRANSFORM(myrec, SELF.real_node :=STD.System.Thorlib.Node();SELF:= LEFT;) ), ALL);
// lb := PROJECT(lbfgs_model_mine4_1, TRANSFORM(thisrecjfdhgdhord, SELF.no := LEFT.no; SELF.pid := LEFT.partition_id; SELF.nid:=LEFT.node_id;SELF.real_node :=STD.System.Thorlib.Node()) );
// output(lb, ALL);

minfunrec := RECORD
UNSIGNED no;
UNSIGNED partition_id;
UNSIGNED real_node;
END;
OUTPUT (lbfgs_model_mine4_1, ALL);

layout_node := RECORD (Layout_Part)
	INTEGER real_node;
END;
// OUTPUT (PROJECT(lbfgs_model_mine4_1, TRANSFORM (layout_node, SELF.real_node := STD.System.Thorlib.Node(); SELF := LEFT), LOCAL), ALL);

//OUTPUT(repeatbias(3, [-0.1,0.2 ,-0.3], [0.78,0.223 ,10]));


 // MatrixModel := SA_mine4_1.Model (lbfgs_model_mine4_1);//convert the model to matrix format where no=1 is W1, no=2 is W2, no=3 is b1 and no=4 is b2
// OUTPUT(MatrixModel, named ('MatrixModel'));
 // Extractedweights := SA_mine4_1.ExtractWeights (lbfgs_model_mine4_1);
// OUTPUT(Extractedweights, named ('Extractedweights'));
 // ExtractedBias := SA_mine4_1.ExtractBias (lbfgs_model_mine4_1);
// OUTPUT(ExtractedBias, named ('ExtractedBias'));
// W1_matrix := ML.Mat.MU.FROM(Extractedweights,1);
// OUTPUT(W1_matrix, NAMED('W1_matrix'));
// W2_matrix := ML.Mat.MU.FROM(Extractedweights,2);
// OUTPUT(W2_matrix, NAMED('W2_matrix'));
// b1_matrix := ML.Mat.MU.FROM(ExtractedBias,1);
// OUTPUT(b1_matrix, NAMED('b1_matrix'));
// b2_matrix := ML.Mat.MU.FROM(ExtractedBias,2);
// OUTPUT(b2_matrix, NAMED('b2_matrix'));
