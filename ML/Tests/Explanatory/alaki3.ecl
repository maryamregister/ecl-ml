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
   Types.NumericField XF(ParseDS L) := TRANSFORM
     SELF.id     := L.RecID;
     SELF.number := (TYPEOF(SELF.number))MATCHTEXT(FldNum) + 1;
     SELF.value  := (TYPEOF(SELF.value))MATCHTEXT(DataVal);
   END;
TrainDS :=  PARSE(ParseDS, Line, FldVal, XF(LEFT));
indepData := TrainDS(Number<109736);
depData   := TrainDS(Number=109736);
// input_data_tmp := DATASET('~maryam::mytest::mnist_5digits_traindata', value_record, CSV); // This dataset is a subset of MNIST dtaset that includes 5 digits (0 to 4), it is used for traibn
//// max(id) = 15298
indepDatatran := PROJECT (indepData,TRANSFORM ( Types.NumericField, SELF.id := LEFT.number; SELF.number := LEFT.id; SELF:=LEFT),LOCAL);
m := 1200000;
prows := 2195;
pcols := 24000; 
mat_map := PBblas.Matrix_Map(109733,m,prows,pcols);
/*
OUTPUT (indepData);
OUTPUT (MAX(indepData,id));
OUTPUT (MAX(indepDatatran,id));
OUTPUT (MAX(indepDatatran,number));
// OUTPUT (MAX (indepData_t, id), named ('col'));
// OUTPUT (MAX (indepData_t, number), named ('row'));

insert_columns:=0;
insert_value:=0.0d;
 Layout_Cell cvt_2_cell(ML.Types.NumericField lr) := TRANSFORM
      SELF.x              := lr.id;     // 1 based
      SELF.y              := lr.number; // 1 based
      SELF.v              := lr.value;
    END;

    d00 := PROJECT(indepDatatran, cvt_2_cell(LEFT));
		OUTPUT (d00);
		
		Work1 := RECORD(Pbblas.Types.Layout_Cell)
    Pbblas.Types.partition_t     partition_id;
    Pbblas.Types.node_t          node_id;
    Pbblas.Types.dimension_t     block_row;
    Pbblas.Types.dimension_t     block_col;
  END;
		FromCells(PBblas.IMatrix_Map mat_map, DATASET(Layout_Cell) cells,
                   PBblas.Types.dimension_t insert_columns=0,
                   PBblas.Types.value_t insert_value=0.0d) := FUNCTION
    Work1 cvt_2_xcell(Layout_Cell lr) := TRANSFORM
      block_row           := mat_map.row_block(lr.x);
      block_col           := mat_map.col_block(lr.y + insert_columns);
      partition_id        := mat_map.assigned_part(block_row, block_col);
      SELF.partition_id   := partition_id;
      SELF.node_id        := mat_map.assigned_node(partition_id);
      SELF.block_row      := block_row;
      SELF.block_col      := block_col;
      SELF := lr;
    END;
    inMatrix := cells.x BETWEEN 1 AND mat_map.matrix_rows
            AND cells.y BETWEEN 1 AND mat_map.matrix_cols - insert_columns;
    d0 := PROJECT(cells(inMatrix), cvt_2_xcell(LEFT),local);
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
		
		result := FromCells(mat_map, d00, insert_columns, insert_value);
*/
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

// resultnode := PROJECT(result, TRANSFORM (thsirec,  SELF.real_node := STD.System.Thorlib.Node(); SELF := LEFT), LOCAL);
// OUTPUT (resultnode, named ('resultnode'),ALL);
// OUTPUT (MAX (d00,x));
// OUTPUT (MIN (d00,x));
// OUTPUT (MAX (d00,y));
// OUTPUT (MIN (d00,y));
OUTPUT (mat_map);

pb := DMAT.Converted.FromNumericFieldDS(indepDatatran,mat_map);
resultnodpke := PROJECT(pb, TRANSFORM (thsirec,  SELF.real_node := STD.System.Thorlib.Node(); SELF := LEFT), LOCAL);
OUTPUT(pb,,'~thor::maryam::mytest::pb',CSV(HEADING(SINGLE)), OVERWRITE);
OUTPUT (resultnodpke,ALL);

// ll := DATASET ([{1,1,1},{2,1,3},{3,1,2},{4,1,1},{5,1,3},{6,1,4}],ML.Types.NumericField);
// output (ll,named ('ll'));
// llgrnd := utils.LabelToGroundTruth (ll);


// Types.NumericField grnd (Types.NumericField le, UNSIGNED c) := TRANSFORM
		// SELF.number := le.id;
		// SELF.id := c;
		// SELF.value := IF (c=le.value,1,0);
		// SELF:= le;
	// END;
	// grndt_result := NORMALIZE (ll,3,grnd(LEFT, COUNTER));
// output (llgrnd, named('llgrnd'));