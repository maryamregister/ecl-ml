IMPORT * FROM ML;
IMPORT STD;
IMPORT * FROM $;
IMPORT PBblas;
IMPORT STD;
Layout_Cell := PBblas.Types.Layout_Cell;
Layout_part := PBblas.Types.Layout_part;





// input_data_tmp := DATASET('~maryam::mytest::mnist_5digits_traindata', value_record, CSV); // This dataset is a subset of MNIST dtaset that includes 5 digits (0 to 4), it is used for traibn
//// max(id) = 15298
indepDataC := DATASET('~maryam::mytest::news20_train_data_sparse', ML.Types.NumericField, CSV); //  11269 samples, 53975 features, sparse matrix in numeric field format

// indepDataC := DATASET([{1,1,9},{2,4,8},{4,5,3.3}], ML.Types.NumericField); 

//from notepad++

labeldata_Format := RECORD
  UNSIGNED id;
  INTEGER label;
END;

label_table := DATASET('~maryam::mytest::news20_train_label', labeldata_Format, CSV); 
OUTPUT  (label_table,  NAMED ('label_table'));



ML.ToField(label_table, depDataC);
OUTPUT  (depDataC,  NAMED ('depDataC'));
label := PROJECT(depDataC,Types.DiscreteField);
OUTPUT  (label,  NAMED ('label'));

//initialize THETA
Numclass := MAX (label, label.value);
OUTPUT  (Numclass, NAMED ('Numclass'));
InputSize := MAX (indepDataC, indepDataC.number);
numsamples := MAX (indepDataC, indepDataC.id);
OUTPUT (InputSize, named ('InputSize'));
OUTPUT (numsamples, named('numsamples'));

T1 := Mat.RandMat (Numclass,InputSize);

OUTPUT  (T1,  NAMED ('T1'));
IntTHETA := Mat.Scale (T1,0.005);
OUTPUT  (IntTHETA,  NAMED ('IntTHETA'));
//SoftMax_Sparse Classfier

//Set Parameters
LoopNum := 200; // Number of iterations in softmax algortihm
LAMBDA := 0.01; // weight decay parameter in  claculation of SoftMax Cost fucntion

UNSIGNED4 prows:=1080;
 UNSIGNED4 pcols:=1127;

 UNSIGNED corr := 5;

// trainer := DeepLearning.softmax_lbfgs (InputSize, Numclass, prows,  pcols); 
trainer := DeepLearning.softmax_lbfgs_partitions_datadist (InputSize, Numclass, prows,  pcols); 
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
testdata := testdata_ (number <= 53975);
// testdata := indepDataC;
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
ML.ToField(label_test_, label_test);
OUTPUT (label_test, named ('label_test'));
// label_test := depDataC;
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
// W20161208-165216 works for softmax_lbfgs_partitions_datadist
