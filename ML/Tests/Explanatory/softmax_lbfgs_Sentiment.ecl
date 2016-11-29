IMPORT * FROM ML;
IMPORT STD;
IMPORT * FROM $;
IMPORT PBblas;
IMPORT STD;
Layout_Cell := PBblas.Types.Layout_Cell;
Layout_part := PBblas.Types.Layout_part;



// fileName := '~vherrara::datasets::sparsearfffile.arff';
fileName := '~vherrara::datasets::sentiment_75pct.arff';
   InDS    := DATASET(fileName, {STRING Line}, CSV(SEPARATOR([])));
   ParseDS := PROJECT(InDS, TRANSFORM({UNSIGNED RecID, STRING Line}, SELF.RecID:= COUNTER, SELF := LEFT));
   //Parse the fields and values out
   PATTERN ws       := ' ';
   PATTERN RecStart := '{';
   PATTERN ValEnd   := '}' | ',';
   PATTERN FldNum   := PATTERN('[0-9]')+;
   PATTERN DataQ    := '"' PATTERN('[ a-zA-Z0-9]')+ '"';
   PATTERN DataNQ   := PATTERN('[a-zA-Z0-9]')+;
   PATTERN DataVal  := DataQ | DataNQ;
   PATTERN FldVal   := OPT(RecStart) FldNum ws DataVal ValEnd;
   OutRec := RECORD
     UNSIGNED RecID;
     STRING   FldName;
     STRING   FldVal;
   END;
   Types.DiscreteField XF(ParseDS L) := TRANSFORM
     SELF.id     := L.RecID;
     SELF.number := (TYPEOF(SELF.number))MATCHTEXT(FldNum) + 1;
     SELF.value  := (TYPEOF(SELF.value))MATCHTEXT(DataVal);
   END;
TrainDS :=  PARSE(ParseDS, Line, FldVal, XF(LEFT));
indepData := TrainDS(Number<109736);
depData   := TrainDS(Number=109736);
// input_data_tmp := DATASET('~maryam::mytest::mnist_5digits_traindata', value_record, CSV); // This dataset is a subset of MNIST dtaset that includes 5 digits (0 to 4), it is used for traibn
//// max(id) = 15298
indepDataC := PROJECT (indepData, TRANSFORM (ML.Types.NumericField,SELF:=LEFT));


labeldata_Format := RECORD
  UNSIGNED id;
  INTEGER label;
END;

label_table := DATASET('~maryam::mytest::news20_train_label', labeldata_Format, CSV); 
// OUTPUT  (label_table,  NAMED ('label_table'));


depDataC_ := PROJECT (depData, TRANSFORM (ML.Types.NumericField,SELF:=LEFT));
depDataC := PROJECT(depDataC_, TRANSFORM (ML.Types.NumericField,SELF.value := IF (LEFT.value=0, 1, 2); SELF:=LEFT));// labels are 0 or 4, we convert them to 1 and 2 with this project. The input to softmax should have its lables starting from 1



// ML.ToField(label_table, depDataC);
OUTPUT  (depDataC,  NAMED ('depDataC'));

//initialize THETA
Numclass := 2;
OUTPUT  (Numclass, NAMED ('Numclass'));
InputSize := MAX (indepDataC, indepDataC.number);// 109733
numsamples := MAX (indepDataC, indepDataC.id);
OUTPUT (InputSize, named ('InputSize'));
OUTPUT (numsamples, named('numsamples'));
OUTPUT (MAX (depDataC, depDataC.id), named ('numberlabeledamples'));
OUTPUT (MIN (depDataC, depDataC.value), named ('minlabels'));
OUTPUT (MAX (depDataC, depDataC.value), named ('maxabels'));


T1 := Mat.RandMat (Numclass,InputSize);

OUTPUT  (T1,  NAMED ('T1'));
IntTHETA := Mat.Scale (T1,0.005);
OUTPUT  (IntTHETA,  NAMED ('IntTHETA'));
//SoftMax_Sparse Classfier

//Set Parameters
LoopNum := 200; // Number of iterations in softmax algortihm
LAMBDA := 0.01; // weight decay parameter in  claculation of SoftMax Cost fucntion

UNSIGNED4 prows:=2195;
 UNSIGNED4 pcols:=24000;

 UNSIGNED corr := 10;

trainer := DeepLearning.softmax_lbfgs (InputSize, Numclass, prows,  pcols); 
//SM( X,  Y, Inttheta, LAMBDA,  MaxIter,  LBFGS_corrections ) 
softresult := trainer.LearnC (indepDataC, depDataC,IntTHETA, LAMBDA, LoopNum, corr);

// OUTPUT (softresult, named('softresult'));

thsirec := RECORD 
 Pbblas.types.node_t          node_id;
    Pbblas.types.partition_t     partition_id;
    Pbblas.types.dimension_t     block_row;
    Pbblas.types.dimension_t     block_col;
    Pbblas.types.dimension_t     first_row;
    Pbblas.types.dimension_t     part_rows;
    Pbblas.types.dimension_t     first_col;
    Pbblas.types.dimension_t     part_cols;
    
UNSIGNED real_node;
END;

OUTPUT (PROJECT(softresult, TRANSFORM (thsirec,  SELF.real_node := STD.System.Thorlib.Node(); SELF := LEFT), LOCAL),named ('realnodes'), ALL);
OUTPUT(softresult,,'~thor::maryam::mytest::news20',CSV(HEADING(SINGLE)), OVERWRITE);
/*

 lbfgs_rec := RECORD 
Pbblas.types.node_t          node_id;
    Pbblas.types.partition_t     partition_id;
    Pbblas.types.dimension_t     block_row;
    Pbblas.types.dimension_t     block_col;
    Pbblas.types.dimension_t     first_row;
    Pbblas.types.dimension_t     part_rows;
    Pbblas.types.dimension_t     first_col;
    Pbblas.types.dimension_t     part_cols;
			REAL8 cost_value;
      REAL8 h ;//hdiag value
      INTEGER8 min_funEval;
      INTEGER break_cond ;
			REAL8 sty  ;
			PBblas.Types.t_mu_no no;
			INTEGER8 update_itr ; //This value is increased whenever a update is done and s and y vectors are added to the corrections. If no update is done due to the condition ys > 1e-10 then this value is not increased
		
			INTEGER8 itr_counter;
			UNSIGNED real_node;
			// Pbblas.Types.matrix_t        mat_part;
    END;
		
		OUTPUT (PROJECT(softresult, TRANSFORM (lbfgs_rec,  SELF.real_node := STD.System.Thorlib.Node(); SELF := LEFT), LOCAL),named ('realnodes_lbfgs'),ALL);

// testdata := DATASET([{1,1,9},{1,3,8},{3,3,3.3},{4,6,10}], ML.Types.NumericField);  
testdata_ := DATASET('~maryam::mytest::news20_test_data_sparse', ML.Types.NumericField, CSV); //  11269 samples, 53975 features, sparse matrix in numeric field format
// testdata := testdata_ (number <= 53975);
testdata := indepDataC;
// testmap := PBblas.Matrix_Map(4,6,2,2);
// h := ML.DMat.Converted.FromNumericFieldDS (testdata, testmap);
// OUTPUT (h);
OUTPUT (MAX (testdata, id),named ('number_of_test_samples'));
OUTPUT (MAX (testdata, number),named ('number_of_test_features'));
 SAmod := trainer.model(softresult);

OUTPUT(SAmod, named('SAmod'));

prob := trainer.ClassProbDistribC(testdata,softresult);

OUTPUT(prob, named('prob'), ALL);

classprob := trainer.ClassifyC(testdata,softresult);

OUTPUT(classprob, named('classprob'), ALL);

//calculate accuracy
acc_rec := RECORD
  ML.Types.t_RecordID id;
  ML.Types.t_Discrete actual_class;
  ML.Types.t_Discrete predicted_class;
  INTEGER match; // if actual_class= predicted_class then match=1 else match=0
END;
label_test_ := DATASET('~maryam::mytest::news20_test_label',labeldata_Format, CSV);
// ML.ToField(label_test_, label_test);
// OUTPUT (label_test, named ('label_test'));
label_test := depDataC;
classified := classprob;
acc_rec build_acc (label_test l, classified r) := TRANSFORM
  SELF.id := l.id;
  SELF.actual_class := l.value;
  SELF.predicted_class := r.value;
  SELF.match := IF (l.value=r.value, 1, 0);
END;

acc_data := JOIN(label_test, classified, LEFT.id=RIGHT.id,build_acc(LEFT,RIGHT));
OUTPUT(acc_data, named ('acc_data'));
OUTPUT(sum(acc_data, acc_data.match));
OUTPUT(MAX(label_test, label_test.id));
OUTPUT(sum(acc_data, acc_data.match)/MAX(label_test, label_test.id), NAMED('accuracy'));
// lambda - 0.1 and corr=5 -. accuracy = 0.77 on train data W20161125-154537 
//lambda - 0.01 and corr=5 -. accuracy = 0.92 on tain data  W20161128-111749 

*/

