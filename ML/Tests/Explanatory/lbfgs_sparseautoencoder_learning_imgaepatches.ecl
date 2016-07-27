IMPORT * FROM ML;
IMPORT * FROM $;
IMPORT PBblas;
Layout_Cell := PBblas.Types.Layout_Cell;
// W20160722-150442
// test Sparse_Autoencoder_lbfgs on an image dataset, the dataset includes patches of randome imagae of size 8 by 8

INTEGER4 hl := 25;//number of nodes in the hiddenlayer
INTEGER4 f := 8*8;//number of input features

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

input_data_tmp := DATASET('~maryam::mytest::patches', value_record, CSV);
ML.AppendID(input_data_tmp, id, input_data);
sample_table := input_data;
ML.ToField(sample_table, indepDataC);
//define the parameters for the Sparse Autoencoder
REAL8 sparsityParam  := 0.01;
REAL8 BETA := 3;
REAL8 ALPHA := 0.1;
REAL8 LAMBDA :=0.0001;
UNSIGNED2 MaxIter :=400;
UNSIGNED4 prows:=0;
UNSIGNED4 pcols:=0;
UNSIGNED4 Maxrows:=0;
UNSIGNED4 Maxcols:=0;
//initialize weight and bias values for the Back Propagation algorithm
IntW := DeepLearning.Sparse_Autoencoder_IntWeights(f, hl);
Intb := DeepLearning.Sparse_Autoencoder_IntBias(f, hl);
SA_mine4_1 := DeepLearning.Sparse_Autoencoder_lbfgs(f, hl, 0, 0, 0, 0);
IntW1 := Mat.MU.From(IntW,1);
IntW2 := Mat.MU.From(IntW,2);
Intb1 := Mat.MU.From(Intb,1);
Intb2 := Mat.MU.From(Intb,2);
lbfgs_model_mine4_1 := SA_mine4_1.LearnC(indepDataC,IntW1, IntW2, Intb1, Intb2, BETA,sparsityParam ,LAMBDA, MaxIter);////the output includes the learnt parameters for the sparse autoencoder (W1,W2,b1,b2) in numericfield format
OUTPUT(lbfgs_model_mine4_1);
MatrixModel := SA_mine4_1.Model (lbfgs_model_mine4_1);//convert the model to matrix format where no=1 is W1, no=2 is W2, no=3 is b1 and no=4 is b2
OUTPUT(MatrixModel, named ('MatrixModel'));
Extractedweights := SA_mine4_1.ExtractWeights (lbfgs_model_mine4_1);
//OUTPUT(Extractedweights, named ('Extractedweights'));
ExtractedBias := SA_mine4_1.ExtractBias (lbfgs_model_mine4_1);
//OUTPUT(ExtractedBias, named ('ExtractedBias'));
W1_matrix := ML.Mat.MU.FROM(Extractedweights,1) ;
//OUTPUT(W1_matrix, NAMED('W1_matrix'));
W2_matrix := ML.Mat.MU.FROM(Extractedweights,2) ;
//OUTPUT(W2_matrix, NAMED('W2_matrix'));
b1_matrix := ML.Mat.MU.FROM(ExtractedBias,1) ;
OUTPUT(b1_matrix, NAMED('b1_matrix'));
b2_matrix := ML.Mat.MU.FROM(ExtractedBias,2) ;
OUTPUT(b2_matrix, NAMED('b2_matrix'));
OUTPUT(W1_matrix,,'~thor::maryam::mytest::W1_matrix_patche_2',CSV(HEADING(SINGLE)));
OUTPUT(W2_matrix,,'~thor::maryam::mytest::W2_matrix_patche_2',CSV(HEADING(SINGLE)));
OUTPUT(b1_matrix,,'~thor::maryam::mytest::b1_matrix_patche_2',CSV(HEADING(SINGLE)));
OUTPUT(b2_matrix,,'~thor::maryam::mytest::b2_matrix_patche_2',CSV(HEADING(SINGLE)));