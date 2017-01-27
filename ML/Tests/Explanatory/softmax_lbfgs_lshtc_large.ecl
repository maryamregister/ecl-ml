IMPORT * FROM ML;
IMPORT STD;
IMPORT * FROM $;
IMPORT PBblas;
IMPORT STD;
Layout_Cell := PBblas.Types.Layout_Cell;
Layout_part := PBblas.Types.Layout_part;
Layout_Cell4 := PBblas.Types.Layout_Cell4;
Layout_part4 := PBblas.Types.Layout_part4;

// filename2 := '~maryam::mytest::small_lshtc_train';
 // filename2 := '~maryam::mytest::small_lshtc_train_task3.txt';
filename2:= '~online::maryam::large_lshtc_train_task1.txt'; // 347256 number of features, 12294 labels, 93805 samples
InDS2    := DATASET(filename2, {STRING Line}, CSV(SEPARATOR([])));
ParseDS2 := PROJECT(InDS2, TRANSFORM({UNSIGNED4 RecID, STRING Line}, SELF.Line := LEFT.Line + ' '; SELF.RecID:= COUNTER, SELF := LEFT));
//Parse the fields and values out
PATTERN lb := PATTERN('[0-9]')+;
PATTERN ws       := ' ';
PATTERN cl := ':';
PATTERN RecStart := '{';
PATTERN ValEnd   := '}' | ',';
PATTERN FldNum   := PATTERN('[0-9]')+;
PATTERN DataQ    := '"' PATTERN('[ a-zA-Z0-9]')+ '"';
PATTERN DataNQ   := PATTERN('[a-zA-Z0-9]')+;
PATTERN DataVal  := DataQ | DataNQ;
PATTERN FldVal   := OPT(RecStart) FldNum ws DataVal ValEnd;
PATTERN ValEnd2 := ws | '';
PATTERN FldVal2   := FldNum cl DataVal ws;
OutRec := RECORD
 UNSIGNED RecID;
 STRING   FldName;
 STRING   FldVal;
END;
Types.NumericField4 XF(ParseDS2 L) := TRANSFORM
 SELF.id     := L.RecID;
 SELF.number := (TYPEOF(SELF.number))MATCHTEXT(FldNum) + 1;
 SELF.value  := (TYPEOF(SELF.value))MATCHTEXT(DataVal);
END;
TrainDS2 :=  PARSE(ParseDS2, Line, FldVal2, XF(LEFT));
// OUTPUT (InDS2, named ('InDS2'));
// OUTPUT (ParseDS2, named ('ParseDS2'));
L_n := PROJECT (ParseDS2, TRANSFORM (Types.NumericField4, SELF.id := LEFT.recid; SELF.number :=1; SELF.value := (TYPEOF(Types.t_FieldReal4)) STD.Str.GetNthWord (LEFT.Line,1)));
// OUTPUT (L_n, named ('L_n'));
// OUTPUT (MAX (L_n,value), named ('maxLN'));
indepDataC := utils.DistinctFeaturest4(TrainDS2);
// OUTPUT (indepDataC, named ('indepDataC'));
// OUTPUT (MIN (TrainDS2, value), named ('indepDataCMin'));
// OUTPUT (MAX (TrainDS2, value), named ('indepDataCMax'));
// OUTPUT (COUNT (indepDataC), named ('CountindepDataC'));

depDataC := L_n;

//initialize THETA
Numclass := COUNT (DEDUP (SORT (depDataC, value),value)); // number of distinct values represents the numebr of classes
// OUTPUT  (Numclass, NAMED ('Numclass'));
InputSize := COUNT (DEDUP (SORT (indepDataC, number),number));
numsamples := MAX (indepDataC, indepDataC.id);
// OUTPUT (InputSize, named ('InputSize'));
// OUTPUT (MAX (indepDataC,number),named('maxnumber'));
// OUTPUT (MIN (indepDataC,number),named('minnumber'));
// OUTPUT (numsamples, named('numsamples'));


//Set Parameters
LoopNum := 200; // Number of iterations in softmax algortihm
LAMBDA := 0.001; // weight decay parameter in  claculation of SoftMax Cost fucntion


 UNSIGNED corr := 5;
prows := 246;
pcols := 1;
T1 := ML.Utils.distrow_ranmap(12294, 347256, prows ) ;

 // OUTPUT  (T1,  NAMED ('T1'));

// IntTHETA := Mat.Scale (T1,0.005);
IntTHETA := PROJECT (T1, TRANSFORM (Mat.Types.Element4, SELF.value := LEFT.value * 0.005; SELF := LEFT), LOCAL);
 // OUTPUT  (IntTHETA,  NAMED ('IntTHETA'));
//SoftMax_Sparse Classfier

// OUTPUT (MAX (IntTHETA,IntTHETA.x));

// OUTPUT (MAX (IntTHETA, IntTHETA.y));

// OUTPUT (COUNT (IntTHETA));


// trainer := DeepLearning.softmax_lbfgs (InputSize, Numclass, prows,  pcols); 
trainer := DeepLearning.softmax_lbfgs_partitions_datadist (InputSize, Numclass, prows,  pcols , TRUE); 

depDataC_distinc := Utils.DistinctLabel4( depDataC);
// OUTPUT (depDataC_distinc);
// OUTPUT (COUNT (depDataC_distinc));
// OUTPUT (MIN (depDataC_distinc, value));
// OUTPUT (MAX (depDataC_distinc, value));
// OUTPUT (COUNT (depDataC_distinc (value >=12055)),named('labelsMore'));

//SM( X,  Y, Inttheta, LAMBDA,  MaxIter,  LBFGS_corrections ) 
softresult := trainer.LearnC (indepDataC, depDataC_distinc, IntTHETA, LAMBDA, LoopNum, corr);
OUTPUT (COUNT(softresult));

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
		REAL8 cost_value := 0;
     // REAL8 cost_value;
UNSIGNED real_node;
END;
// OUTPUT (count (indepDataC),named('indepDataCCount'));
// OUTPUT (count (softresult),named ('softresultCount'));
// COUNT (DEDUP (SORT (softresult, block_col),block_col));

// R1 := RECORD
  // softresult.block_col;
  // Number_count := COUNT(GROUP);

// END;
// T1kk := TABLE(softresult, R1,  block_col);
// OUTPUT (MAX (T1kk, Number_count), named ('T1kk'));

OUTPUT (PROJECT(softresult, TRANSFORM (thsirec,  SELF.real_node := STD.System.Thorlib.Node(); SELF := LEFT), LOCAL),named ('realnodes'),ALL);


OUTPUT(softresult,,'~thor::maryam::mytest::lshtclarge_theta_mat',CSV(HEADING(SINGLE)), OVERWRITE);
// outputformat := RECORD 
	// softresult.node_id;
    // softresult.partition_id;
    // softresult.block_row;
    // softresult.block_col;
    // softresult.first_row;
    // softresult.part_rows;
    // softresult.first_col;
    // softresult.part_cols;
		// softresult.mat_part;
// END;
// OUTPUT(softresult, outputformat,'~thor::maryam::mytest::lshtclarge_inttheta');

	
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
      REAL8 fprev;
		REAL8 tprev;// this is the actualy previous t calculated in the previous iteration
		REAL8 prev_t;// this should actually be tnew, however in order for this record format to be consistent with the ourput of wolfe line search , the new t has to be named prev_t
		UNSIGNED wolfe_funevals;
		INTEGER armCond;
		REAL8 glob_f;
		REAL8 gtdnew;
		BOOLEAN islegal_gnew := TRUE;
		REAL8 local_sumd;
			UNSIGNED real_node
    END;
		
OUTPUT (PROJECT(softresult, TRANSFORM (lbfgs_rec,  SELF.real_node := STD.System.Thorlib.Node(); SELF := LEFT), LOCAL),named ('realnodes_lbfgs'));

	
	
	
*/
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
			REAL4 cost_value;
      INTEGER id;
			REAL4 prev_t;
			REAL4 prev_gtd;
			UNSIGNED wolfe_funEvals;
			UNSIGNED8 c;
			INTEGER bracketing_cond;
			INTEGER zooming_cond := 0;
			REAL4 next_t;
			REAL4 high_t;
			REAL4 high_cost_value;
			REAL4 high_gtd;
			REAL4 glob_f; // this is the f value we recive through wolfelinesearch function call
			BOOLEAN insufProgress;
			BOOLEAN zoomtermination;
			UNSIGNED real_node
    END;
		
OUTPUT (PROJECT(softresult, TRANSFORM (lbfgs_rec,  SELF.real_node := STD.System.Thorlib.Node(); SELF := LEFT), LOCAL),named ('realnodes_lbfgs'));

// OUTPUT (COUNT (softresult));
// the lbfgs one 
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
		
OUTPUT (PROJECT(softresult, TRANSFORM (lbfgs_rec,  SELF.real_node := STD.System.Thorlib.Node(); SELF := LEFT), LOCAL),named ('realnodes_lbfgs'));
/*
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

*/


// line 6 in PBblas.types is changed from UNSIGNED2 to UNSIGNED4 because parition_used might be morr than UNSIGNED2