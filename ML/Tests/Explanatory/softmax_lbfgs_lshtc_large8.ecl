IMPORT * FROM ML;
IMPORT STD;
IMPORT * FROM $;
IMPORT PBblas;
IMPORT STD;
IMPORT std.system.Thorlib;
Layout_Cell := PBblas.Types.Layout_Cell;
Layout_part := PBblas.Types.Layout_part;


// filename2 := '~maryam::mytest::small_lshtc_train';
 // filename2 := '~maryam::mytest::small_lshtc_train_task3.txt';
// filename2:= '~online::maryam::large_lshtc_train_task1.txt'; // 347256 number of features, 12294 labels, 93805 samples
filename2:= '~online::maryam::wikipediamediumpreproclshtcv3-train.txt'; // 31521 classes, 346299 features
InDS2    := DATASET(filename2, {STRING Line}, CSV(SEPARATOR([])));
ParseDS2 := PROJECT(InDS2, TRANSFORM({UNSIGNED RecID, STRING Line}, SELF.Line := LEFT.Line + ' '; SELF.RecID:= COUNTER, SELF := LEFT));
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
Types.NumericField XF(ParseDS2 L) := TRANSFORM
 SELF.id     := L.RecID;
 SELF.number := (TYPEOF(SELF.number))MATCHTEXT(FldNum) + 1;
 SELF.value  := (TYPEOF(SELF.value))MATCHTEXT(DataVal);
END;
TrainDS2 :=  PARSE(ParseDS2, Line, FldVal2, XF(LEFT));
// OUTPUT (InDS2, named ('InDS2'));
// OUTPUT (ParseDS2, named ('ParseDS2'));
L_n := PROJECT (ParseDS2, TRANSFORM (Types.NumericField, SELF.id := LEFT.recid; SELF.number :=1; SELF.value := (TYPEOF(INTEGER8)) STD.Str.GetNthWord (LEFT.Line,1)));
// OUTPUT (L_n, named ('L_n'));
// OUTPUT (MAX (L_n,value), named ('maxLN'));
indepDataC := utils.DistinctFeaturest(TrainDS2);
// OUTPUT (indepDataC, named ('indepDataC'));
// OUTPUT (MIN (indepDataC, value), named ('indepDataCMin'));
// OUTPUT (MAX (indepDataC, value), named ('indepDataCMax'));
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
LAMBDA := 0.00001; // weight decay parameter in  claculation of SoftMax Cost fucntion


 UNSIGNED corr := 5;
// prows := 31;
// prows := 41;
// prows := 49;
// prows := 61;
// prows := 123;
prows := 82; //medium wikipedia
pcols := 1;
// T1 := ML.Utils.distrow_ranmap(12294, 347256, prows ) ;

 // OUTPUT  (T1,  NAMED ('T1'));
 
// IntTHETA := Mat.Scale (T1,0.005);
// IntTHETA := PROJECT (T1, TRANSFORM (Mat.Types.Element, SELF.value := LEFT.value * 0.005; SELF := LEFT), LOCAL);
IntTHETA := DATASET ([],Mat.Types.Element);
// OUTPUT  (IntTHETA,  NAMED ('IntTHETA'));
//SoftMax_Sparse Classfier

// OUTPUT (MAX (IntTHETA,IntTHETA.x));

// OUTPUT (MAX (IntTHETA, IntTHETA.y));

// OUTPUT (COUNT (IntTHETA));


depDataC_distinc := Utils.DistinctLabel( depDataC);




// OUTPUT (depDataC_distinc, named ('depDataC_distinc'));
// OUTPUT (MAX(depDataC_distinc, depDataC_distinc.value), named ('numFeatures'));



//start
/*

  Layout_Cell_nid := RECORD (Layout_Cell)
UNSIGNED4 node_id;
END;
 X := indepDataC;
 Y := depDataC_distinc;
 NumberofFeatures := InputSize;
 OUTPUT (NumberofFeatures, named ('NumberofFeatures'));
 NumberofClasses := Numclass;
  OUTPUT (NumberofClasses, named ('NumberofClasses'));
 tonorm := TRUE;
 

	
	m := MAX (X,X.id);// number of samples
	f := NumberofFeatures; // number of features
	labelmap := PBblas.Matrix_Map(m,1,m,1);
	//Create block matrix d
	Xtran := PROJECT (X,TRANSFORM ( ML.Types.NumericField, SELF.id := LEFT.number; SELF.number := LEFT.id; SELF:=LEFT),LOCAL);//through the analysis rows represent features and columns represent samples
	

	dmap := PBblas.Matrix_Map(f,m,prows,pcols);
	dmap_usednodes := MIN (dmap.row_blocks, Thorlib.nodes());
	insert_columns:=0;
	insert_value:=0.0d;
	Layout_Cell cvt_2_cell(ML.Types.NumericField lr) := TRANSFORM
		SELF.x := lr.id;     // 1 based
		SELF.y := lr.number; // 1 based
		SELF.v := lr.value;
  END;
  Xtran_cell := PROJECT(Xtran, cvt_2_cell(LEFT));
	Y_cell := PROJECT(Y, cvt_2_cell(LEFT));
	Work1 := RECORD(Layout_Cell)
    PBblas.Types.partition_t     partition_id;
    PBblas.Types.node_t          node_id;
    PBblas.Types.dimension_t     block_row;
    PBblas.Types.dimension_t     block_col;
  END;
  FromCells(PBblas.IMatrix_Map mat_map, DATASET(Layout_Cell) cells,
                   PBblas.Types.dimension_t insert_columns=0,
                   PBblas.Types.value_t insert_value=0.0d) := FUNCTION
    Work1 cvt_2_xcell(Layout_Cell lr) := TRANSFORM
      block_row           := mat_map.row_block(lr.x);
      block_col           := mat_map.col_block(lr.y + insert_columns);
      partition_id        := mat_map.assigned_part(block_row, block_col);
      SELF.partition_id   := partition_id;
      SELF.node_id        := ((block_row-1) % dmap_usednodes);// instead of using partition id in order to distribute the data, block row number is used to distribute the data
      SELF.block_row      := block_row;
      SELF.block_col      := block_col;
      SELF := lr;
    END;
    inMatrix := cells.x BETWEEN 1 AND mat_map.matrix_rows
            AND cells.y BETWEEN 1 AND mat_map.matrix_cols - insert_columns;
    d0 := PROJECT(cells(inMatrix), cvt_2_xcell(LEFT));
    d1 := DISTRIBUTE(d0, node_id);
    d2 := SORT(d1, partition_id, y, x, LOCAL);    // prep for column major
    d3 := GROUP(d2, partition_id, LOCAL);
    Layout_Part roll_cells(Work1 parent, DATASET(Work1) cells) := TRANSFORM
      first_row     := mat_map.first_row(parent.partition_id);
      first_col     := mat_map.first_col(parent.partition_id);
      part_rows     := mat_map.part_rows(parent.partition_id);
      part_cols     := mat_map.part_cols(parent.partition_id);
      SELF.mat_part := PBblas.MakeR8Set(part_rows, part_cols, first_row, first_col,
                                        PROJECT(cells, Layout_Cell),
                                        insert_columns, insert_value);
      SELF.partition_id:= parent.partition_id;
      SELF.node_id     := parent.node_id;
      SELF.block_row   := parent.block_row;
      SELF.block_col   := parent.block_col;
      SELF.first_row   := first_row;
      SELF.part_rows   := part_rows;
      SELF.first_col   := first_col;
      SELF.part_cols   := part_cols;
      SELF := [];
    END;
    rslt := ROLLUP(d3, GROUP, roll_cells(LEFT, ROWS(LEFT)));
    RETURN rslt;
  END;
	ddist_ := FromCells(dmap, Xtran_cell, insert_columns, insert_value);
	labeldist := FromCells(labelmap, Y_cell, insert_columns, insert_value);

	// groundTruth := Utils.LabelToGroundTruth (Y);
  //groundTruth is a Numclass*NumSamples matrix. groundTruth(i,j)=1 if label of the jth sample is i, otherwise groundTruth(i,j)=0
	//ymap := PBblas.Matrix_Map(NumberofClasses,m,NumberofClasses,pcols);
	ymap_label := PBblas.Matrix_Map(NumberofClasses,m,prows,m);
	// ydist := DMAT.Converted.FromElement(groundTruth,ymap); orig
	//ydist := DMAT.Converted.FromNumericFieldDS(groundTruth,ymap);// for groundtruth matrix partiion id equals the block)col so DMAT.Converted.FromNumericFieldDS distributes the data based on block_col which is what we want 
//	ydist_label := DMAT.Converted.FromNumericFieldDS(groundTruth,ymap_label);

	//NormalizeFeaturesOne for numerifcField dataset
	NormalizeFeaturesOne_nf (DATASET (Types.NumericField) d_in) := FUNCTION
		d_in_dist := DISTRIBUTE (d_in, d_in.number);
		d_in_dist_sorted := SORT (d_in_dist, d_in_dist.number, LOCAL);
		d_in_dist_sorted_grouped := GROUP (d_in_dist_sorted, d_in_dist_sorted.number);
		sumcol_red := RECORD
			d_in_dist_sorted_grouped.number;
			REAL8 sc := SUM (GROUP, d_in_dist_sorted_grouped.value);//sum col
		END;
		sumcol := TABLE (d_in_dist_sorted_grouped, sumcol_red, number, LOCAL);
		norm_d_in := JOIN (d_in_dist_sorted, sumcol, LEFT.number = RIGHT.number, TRANSFORM (Types.NumericField, SELF.value := LEFT.value/RIGHT.sc ;SELF:= LEFT),LOCAL);
		RETURN norm_d_in;
	END;
	Xtran_normone_ := NormalizeFeaturesOne_nf (Xtran);
	Xtran_normone := IF (tonorm,Xtran_normone_ , Xtran);
	thetamap_label := PBblas.Matrix_Map(NumberofClasses,f,prows,f);
	theta_node_used := thetamap_label.nodes_used;
	Layout_Cell_nid x_norm_tran (Types.NumericField le, UNSIGNED coun) := TRANSFORM
		SELF.x := le.id;
		SELF.y := le.number;
		SELF.v := le.value;
		SELF.node_id := coun % theta_node_used;
	END;
	Xtran_normone_norm := NORMALIZE(Xtran_normone, theta_node_used, x_norm_tran(LEFT, COUNTER) );
	Xtran_normone_norm_dist := DISTRIBUTE (Xtran_normone_norm, node_id);

	Layout_Part label_norm_tran (Layout_Part le, UNSIGNED coun) := TRANSFORM
		SELF.node_id := coun%theta_node_used;
		SELF := le;
	END;
	labeldist_norm := NORMALIZE(labeldist, theta_node_used, label_norm_tran(LEFT, COUNTER) );
	labeldist_norm_dist := DISTRIBUTE (labeldist_norm, node_id);
	
		part_theta := ML.Utils.distrow_ranmap_part(NumberofClasses,f,prows , 0.005) ;
 
		OUTPUT(part_theta,,'~maryam::mytest::parttheta', OVERWRITE);
		OUTPUT(Xtran_normone_norm_dist,,'~maryam::mytest::Xtrannormonenormdist', OVERWRITE);
		OUTPUT(labeldist_norm_dist,,'~maryam::mytest::labeldistnormdist', OVERWRITE);
 
 

*/
 
 
/*

*/







// trainer := DeepLearning.softmax_lbfgs (InputSize, Numclass, prows,  pcols); 
trainer := DeepLearning8.softmax_lbfgs_partitions_datadist (InputSize, Numclass, prows,  pcols , TRUE); 
//SM( X,  Y, Inttheta, LAMBDA,  MaxIter,  LBFGS_corrections ) 
softresult := trainer.LearnC (indepDataC, depDataC_distinc,IntTHETA, LAMBDA, LoopNum, corr);
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
		REAL8 cost_value :=12;
     // REAL8 cost_value;
UNSIGNED real_node ;
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

OUTPUT (PROJECT(softresult(no=2), TRANSFORM (thsirec,  SELF.real_node := STD.System.Thorlib.Node(); SELF := LEFT), LOCAL),named ('realnodes'),ALL);
	SET OF Pbblas.Types.value_t4 to4(PBblas.types.dimension_t N, PBblas.types.matrix_t M) := BEGINC++

    #body
    __lenResult = n * sizeof(float);
    __isAllResult = false;
    float * result = new float[n];
    __result = (void*) result;
		double *cellm = (double*) m;
    uint32_t i;
		for (i=0; i<n; i++) {
			result[i] = (float)cellm[i];
    }
		
  ENDC++;

softresult_4 := PROJECT (softresult(no=2), TRANSFORM (PBblas.Types.Layout_Part4, SELF.mat_part := to4(LEFT.part_rows*LEFT.part_cols,LEFT.mat_part); SELF:=LEFT), LOCAL);
OUTPUT(softresult_4,,'~thor::maryam::mytest::lshtc', OVERWRITE);


	
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
			REAL8 cost_value;
      INTEGER id;
			REAL8 prev_t;
			REAL8 prev_gtd;
			INTEGER wolfe_funEvals;
			UNSIGNED8 c;
			INTEGER bracketing_cond;
			INTEGER zooming_cond := 0;
			REAL8 next_t;
			REAL8 high_t;
			REAL8 high_cost_value;
			REAL8 high_gtd;
			REAL8 glob_f; // this is the f value we recive through wolfelinesearch function call
			BOOLEAN insufProgress;
			BOOLEAN zoomtermination;
			UNSIGNED real_node
    END;
		
OUTPUT (PROJECT(softresult, TRANSFORM (lbfgs_rec,  SELF.real_node := STD.System.Thorlib.Node(); SELF := LEFT), LOCAL),named ('realnodes_lbfgs'),ALL);

// OUTPUT (COUNT (softresult));
// the lbfgs one 
*/


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
			Pbblas.Types.matrix_t        cost_mat;
    END;
		lbfgs_result := PROJECT(softresult, TRANSFORM (lbfgs_rec,  SELF.real_node := STD.System.Thorlib.Node(); SELF := LEFT), LOCAL);
OUTPUT (lbfgs_result,named ('realnodes_lbfgs'));
// OUTPUT(lbfgs_result,,'~thor::maryam::mytest::lshtc_realnodes', OVERWRITE);

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

//microsoft malware classification challenge
//mscoco microsoft dataset
