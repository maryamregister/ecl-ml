IMPORT * FROM ML;
IMPORT * FROM $;
IMPORT PBblas;
Layout_Cell := PBblas.Types.Layout_Cell;
//This is the test on MNIST patches

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

OUTPUT(input_data, NAMED('input_data'));

ML.ToField(input_data, indepDataC);
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

//IntW1 := Mat.MU.From(IntW,1);
IntW1 := DATASET ([{1	,1	,0.199817},
{2	,1	,0.184172},
{1,	2	,0.650816},
{2	,2,	0.771599},
{1,	3,	0.21031},
{2,	3,	0.151326}
], ML.Mat.Types.Element);

// IntW1 := DATASET('~maryam::mytest::w1_4_64', ML.Mat.Types.Element, CSV);
// IntW2 := DATASET('~maryam::mytest::w2_64_4', ML.Mat.Types.Element, CSV);


//IntW2 := Mat.MU.From(IntW,2);
IntW2 := DATASET ([{1,	1,	0.986138},
{2,	1,0.301454},
{3,	1	,0.263632},
{1,	2	,0.261552},
{2,	2,	0.5883930000000001},
{3,	2	,0.530342}
], ML.Mat.Types.Element);

Intb1 := Mat.MU.From(Intb,1);

Intb2 := Mat.MU.From(Intb,2);


OUTPUT(IntW1,ALL, named ('IntW1'));
OUTPUT(IntW2,ALL, named ('IntW2'));
OUTPUT(IntB1,ALL, named ('IntB1'));
OUTPUT(IntB2,ALL, named ('IntB2'));

lbfgs_model_mine := SA_mine.LearnC_lbfgs(indepDataC,IntW1, IntW2, Intb1, Intb2, BETA,sparsityParam ,LAMBDA, 200);

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
OUTPUT(lbfgs_model_mine,NAMED('lbfgs_model'));
OUTPUT(x_ext(lbfgs_model_mine),NAMED('lbfgs_x'));
OUTPUT(g_ext(lbfgs_model_mine),NAMED('lbfgs_g'));
OUTPUT(d_ext(lbfgs_model_mine),NAMED('lbfgs_d'));
OUTPUT(old_dirs_ext(lbfgs_model_mine),NAMED('lbfgs_olddirs'));
OUTPUT(old_steps_ext(lbfgs_model_mine),NAMED('lbfgs_oldsteps'));

OUTPUT(lbfgs_model_mine[1].t_,NAMED('lbfgs_t'));
OUTPUT(lbfgs_model_mine[1].Cost,NAMED('lbfgs_Cost'));
OUTPUT(lbfgs_model_mine[1].hdiag,NAMED('lbfgs_Hdiag'));

CG := SA_mine.CostGrad_cal(indepDataC,IntW1, IntW2, Intb1, Intb2, BETA,sparsityParam ,LAMBDA, MaxIter);
//OUTPUT(CG,ALL, named ('cost_grad'));
