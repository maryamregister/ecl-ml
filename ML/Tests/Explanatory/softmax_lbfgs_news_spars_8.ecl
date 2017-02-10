IMPORT * FROM ML;
IMPORT STD;
IMPORT * FROM $;
IMPORT PBblas;
IMPORT STD;
IMPORT std.system.Thorlib;
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

// OUTPUT  (T1,  NAMED ('T1'));
IntTHETA := Mat.Scale (T1,0.005);
// OUTPUT  (IntTHETA,  NAMED ('IntTHETA'));
//SoftMax_Sparse Classfier

//Set Parameters
LoopNum := 200; // Number of iterations in softmax algortihm
LAMBDA := 0.1; // weight decay parameter in  claculation of SoftMax Cost fucntion

UNSIGNED4 prows:=1;
 UNSIGNED4 pcols:=0;

 UNSIGNED corr := 50;
 
 //start
 /*
  Layout_Cell_nid := RECORD (Layout_Cell)
UNSIGNED4 node_id;
END;
 X := indepDataC;
 Y := depDataC;
 NumberofFeatures := InputSize;
 NumberofClasses := Numclass;
 tonorm := FALSE;
 

	
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
 
		// OUTPUT(part_theta,,'~maryam::mytest::parttheta', OVERWRITE);
		// OUTPUT(Xtran_normone_norm_dist,,'~maryam::mytest::Xtrannormonenormdist', OVERWRITE);
		// OUTPUT(labeldist_norm_dist,,'~maryam::mytest::labeldistnormdist', OVERWRITE);
 
 


 
 
*/
 
 
 
 
 
 
 

// trainer := DeepLearning.softmax_lbfgs (InputSize, Numclass, prows,  pcols); 
// trainer := DeepLearning.softmax_lbfgs_partitions_datadist (InputSize, Numclass, prows,  pcols); 
trainer := DeepLearning8.softmax_lbfgs_partitions_datadist (InputSize, Numclass, prows,  pcols , FALSE);
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
     REAL8 cost_value := 12;
UNSIGNED real_node;
END;
// OUTPUT (softresult);

OUTPUT (PROJECT(softresult, TRANSFORM (thsirec,  SELF.real_node := STD.System.Thorlib.Node(); SELF := LEFT), LOCAL),named ('realnodes'),ALL);

OUTPUT(softresult,,'~thor::maryam::mytest::lshtc',CSV(HEADING(SINGLE)), OVERWRITE);
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
			PBblas.Types.matrix_t cost_mat;
			// Pbblas.Types.matrix_t        mat_part;
    END;
		
OUTPUT (PROJECT(softresult, TRANSFORM (lbfgs_rec,  SELF.real_node := STD.System.Thorlib.Node(); SELF := LEFT), LOCAL),named ('realnodes_lbfgs'));

/*
*/