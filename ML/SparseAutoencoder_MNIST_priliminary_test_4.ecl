IMPORT * FROM ML;
IMPORT * FROM $;
IMPORT PBblas;
Layout_Cell := PBblas.Types.Layout_Cell;
//This is not test on MNIST, this is test on image patches

INTEGER4 hl := 4;//number of nodes in the hiddenlayer
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

input_data_tmp := DATASET('~maryam::mytest::mnist_patches', value_record, CSV); // this is a dataset of size 60000 samples,  each sample is a random patch of MNIST data.
//input_data_tmp := DATASET('~maryam::mytest::mnist_patches_100sa.csv', value_record, CSV); // this is a dataset of size 100 samples,  each sample is a random patch of MNIST data.

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
//output(IntW, named ('IntW'));
//output(IntB, named ('IntB'));
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
SA_mine4 :=DeepLearning4.Sparse_Autoencoder_mine (f, hl, 2, 2,2,2);
SA_mine4_1 :=DeepLearning4_1.Sparse_Autoencoder_mine (f, hl, 0, 0,0,0);
IntW1 := Mat.MU.From(IntW,1);


// IntW1 := DATASET('~maryam::mytest::w1_4_64', ML.Mat.Types.Element, CSV);
// IntW2 := DATASET('~maryam::mytest::w2_64_4', ML.Mat.Types.Element, CSV);


IntW2 := Mat.MU.From(IntW,2);


Intb1 := Mat.MU.From(Intb,1);

Intb2 := Mat.MU.From(Intb,2);


// OUTPUT(IntW1,ALL, named ('IntW1'));
// OUTPUT(IntW2,ALL, named ('IntW2'));
// OUTPUT(IntB1,ALL, named ('IntB1'));
// OUTPUT(IntB2,ALL, named ('IntB2'));

lbfgs_model_mine := SA_mine.LearnC_lbfgs(indepDataC,IntW1, IntW2, Intb1, Intb2, BETA,sparsityParam ,LAMBDA, 200);
lbfgs_model_mine4 := SA_mine4.LearnC_lbfgs_4(indepDataC,IntW1, IntW2, Intb1, Intb2, BETA,sparsityParam ,LAMBDA, 200);
lbfgs_model_mine4_1 := SA_mine4_1.LearnC_lbfgs_4(indepDataC,IntW1, IntW2, Intb1, Intb2, BETA,sparsityParam ,LAMBDA, 200);

IdElementRec := RECORD
      INTEGER1 id;
      Mat.Types.Element;
   END;
MinFRecord := RECORD
    INTEGER1 id;
    DATASET(IdElementRec) x;
    DATASET(IdElementRec) g;
    DATASET(IdElementRec) old_steps;
    DATASET(IdElementRec) old_dirs;
    REAL8 Hdiag;
    REAL8 Cost;
    INTEGER8 funEvals_;
    DATASET(IdElementRec) d;
    REAL8 fnew_fold; //f_new-f_old in that iteration
    REAL8 t_;
    BOOLEAN dLegal;
    BOOLEAN ProgAlongDir; //Progress Along Directionno //Check that progress can be made along direction ( if gtd > -tolX)
    BOOLEAN optcond; // Check Optimality Condition
    BOOLEAN lackprog1; //Check for lack of progress 1
    BOOLEAN lackprog2; //Check for lack of progress 2
    BOOLEAN exceedfuneval; //Check for going over evaluation limit
    INTEGER itr;
  END;
  
  x_ext (DATASET (MinFRecord) br) := FUNCTION
    IdElementRec NewChildren(IdElementRec R) := TRANSFORM
    SELF := R;
    END;
    NewChilds := NORMALIZE(br,LEFT.x,NewChildren(RIGHT));

    RETURN PROJECT(NewChilds, TRANSFORM(Mat.Types.Element, SELF := LEFT));
  END; // END x_ext
  g_ext (DATASET (MinFRecord) br) := FUNCTION
    IdElementRec NewChildren(IdElementRec R) := TRANSFORM
    SELF := R;
    END;
    NewChilds := NORMALIZE(br,LEFT.g,NewChildren(RIGHT));

    RETURN PROJECT(NewChilds, TRANSFORM(Mat.Types.Element, SELF := LEFT));
  END; // END g_ext
  old_steps_ext (DATASET (MinFRecord) br) := FUNCTION
    IdElementRec NewChildren(IdElementRec R) := TRANSFORM
    SELF := R;
    END;
    NewChilds := NORMALIZE(br,LEFT.old_steps,NewChildren(RIGHT));

    RETURN PROJECT(NewChilds, TRANSFORM(Mat.Types.Element, SELF := LEFT));
  END; // END old_steps_ext
  old_dirs_ext (DATASET (MinFRecord) br) := FUNCTION
    IdElementRec NewChildren(IdElementRec R) := TRANSFORM
    SELF := R;
    END;
    NewChilds := NORMALIZE(br,LEFT.old_dirs,NewChildren(RIGHT));

    RETURN PROJECT(NewChilds, TRANSFORM(Mat.Types.Element, SELF := LEFT));
  END; // END old_dirs_ext
  d_ext (DATASET (MinFRecord) br) := FUNCTION
    IdElementRec NewChildren(IdElementRec R) := TRANSFORM
    SELF := R;
    END;
    NewChilds := NORMALIZE(br,LEFT.d,NewChildren(RIGHT));

    RETURN PROJECT(NewChilds, TRANSFORM(Mat.Types.Element, SELF := LEFT));
  END; // END d_ext
// OUTPUT(lbfgs_model_mine,NAMED('lbfgs_model'));
// OUTPUT(x_ext(lbfgs_model_mine),NAMED('lbfgs_x'));
// OUTPUT(g_ext(lbfgs_model_mine),NAMED('lbfgs_g'));
// OUTPUT(d_ext(lbfgs_model_mine),NAMED('lbfgs_d'));
// OUTPUT(old_dirs_ext(lbfgs_model_mine),NAMED('lbfgs_olddirs'));
// OUTPUT(old_steps_ext(lbfgs_model_mine),NAMED('lbfgs_oldsteps'));

// OUTPUT(lbfgs_model_mine[1].t_,NAMED('lbfgs_t'));
// OUTPUT(lbfgs_model_mine[1].Cost,NAMED('lbfgs_Cost'));
// OUTPUT(lbfgs_model_mine[1].hdiag,NAMED('lbfgs_Hdiag'));

OUTPUT(lbfgs_model_mine4_1);
CG := SA_mine.CostGrad_cal(indepDataC,IntW1, IntW2, Intb1, Intb2, BETA,sparsityParam ,LAMBDA, MaxIter);
//OUTPUT(CG,ALL, named ('cost_grad'));
