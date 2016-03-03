IMPORT * FROM ML;
IMPORT * FROM $;
IMPORT PBblas;
Layout_Cell := PBblas.Types.Layout_Cell;
//This is the test on MNIST patches
//Number of neurons in the last layer is number of output assigned to each sample
INTEGER4 hl := 25;//number of nodes in the hiddenlayer
INTEGER4 f := 64;//number of input features

//input data

value_record := RECORD
real	f1	;
real	f2	;
real	f3	;
real	f4	;
real	f5	;
real	f6	;
real	f7	;
real	f8	;
real	f9	;
real	f10	;
real	f11	;
real	f12	;
real	f13	;
real	f14	;
real	f15	;
real	f16	;
real	f17	;
real	f18	;
real	f19	;
real	f20	;
real	f21	;
real	f22	;
real	f23	;
real	f24	;
real	f25	;
real	f26	;
real	f27	;
real	f28	;
real	f29	;
real	f30	;
real	f31	;
real	f32	;
real	f33	;
real	f34	;
real	f35	;
real	f36	;
real	f37	;
real	f38	;
real	f39	;
real	f40	;
real	f41	;
real	f42	;
real	f43	;
real	f44	;
real	f45	;
real	f46	;
real	f47	;
real	f48	;
real	f49	;
real	f50	;
real	f51	;
real	f52	;
real	f53	;
real	f54	;
real	f55	;
real	f56	;
real	f57	;
real	f58	;
real	f59	;
real	f60	;
real	f61	;
real	f62	;
real	f63	;
real	f64	;
END;

input_data_tmp := DATASET('~maryam::mytest::mnist_patches', value_record, CSV);

OUTPUT(input_data_tmp, NAMED('input_data_tmp'));

ML.AppendID(input_data_tmp, id, input_data);
OUTPUT  (input_data, NAMED ('input_data'));



sample_table := input_data;
OUTPUT  (sample_table, NAMED ('sample_table'));

ML.ToField(sample_table, indepDataC);
OUTPUT  (indepDataC, NAMED ('indepDataC'));


//define the parameters for the Sparse Autoencoder
//ALPHA is learning rate
//LAMBDA is weight decay rate
REAL8 sparsityParam  := 0.01;
REAL8 BETA := 3;
REAL8 ALPHA := 0.1;
REAL8 LAMBDA :=0.0001;
UNSIGNED2 MaxIter :=100;
UNSIGNED4 prows:=0;
UNSIGNED4 pcols:=0;
UNSIGNED4 Maxrows:=0;
UNSIGNED4 Maxcols:=0;
//initialize weight and bias values for the Back Propagation algorithm
IntW := DeepLearning.Sparse_Autoencoder_IntWeights(f,hl);
Intb := DeepLearning.Sparse_Autoencoder_IntBias(f,hl);
output(IntW, named ('IntW'));
output(IntB, named ('IntB'));
//trainer module
// SA :=DeepLearning.Sparse_Autoencoder(f,hl,prows, pcols, Maxrows,  Maxcols);

// LearntModel := SA.LearnC(indepDataC,IntW, Intb,BETA, sparsityParam, LAMBDA, ALPHA, MaxIter);
// mout := max(LearntModel,id);
//output(LearntModel(id=1));

// MatrixModel := SA.Model (LearntModel);
// output(MatrixModel, named ('MatrixModel'));

// Out := SA.SAOutput (indepDataC, LearntModel);
// output(Out, named ('Out'));


//MINE
SA_mine :=DeepLearning.Sparse_Autoencoder_mine (f, hl, 0, 0,0,0);

IntW1 := Mat.MU.From(IntW,1);




IntW2 := Mat.MU.From(IntW,2);

Intb1 := Mat.MU.From(Intb,1);

Intb2 := Mat.MU.From(Intb,2);


OUTPUT(IntW1,ALL, named ('IntW1'));
OUTPUT(IntW2,ALL, named ('IntW2'));
OUTPUT(IntB1,ALL, named ('IntB1'));
OUTPUT(IntB2,ALL, named ('IntB2'));

lbfgs_model_mine := SA_mine.LearnC_lbfgs(indepDataC,IntW1, IntW2, Intb1, Intb2, BETA,sparsityParam ,LAMBDA, MaxIter);
OUTPUT(lbfgs_model_mine,NAMED('lbfgs_model_mine'));



