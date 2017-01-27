IMPORT * FROM ML;
IMPORT STD;
IMPORT * FROM $;
IMPORT PBblas;
IMPORT STD;
Layout_Cell := PBblas.Types.Layout_Cell;
Layout_part := PBblas.Types.Layout_part;

dimension_t := PBblas.Types.dimension_t;
value_t := PBblas.Types.value_t;





// OUTPUT (TrainDS);
// OUTPUT (count (TrainDS));

// input_data_tmp := DATASET('~maryam::mytest::mnist_5digits_traindata', value_record, CSV); // This dataset is a subset of MNIST dtaset that includes 5 digits (0 to 4), it is used for traibn
//// max(id) = 15298
indepDatatran := DATASET ([
{1,1,2},
{2,2,8},
{3,2,9},
{1,3,10},
{1,4,11},
{2,5,9},
{3,5,8},
{1,6,7},
{2,6,0.9},
{3,6,17}
],Types.NumericField);
// OUTPUT (indepDatatran);
// OUTPUT (indepData);

// OUTPUT (MAX (indepDatatran, id));
// OUTPUT (MAX (indepDatatran, number));

// OUTPUT (MAX (indepData_t, id), named ('col'));
// OUTPUT (MAX (indepData_t, number), named ('row'));
m := 6;
prows := 3;
pcols := 3; 
mat_map := PBblas.Matrix_Map(3,m,prows,pcols);
insert_columns:=0;
insert_value:=0.0d;
 Layout_Cell cvt_2_cell(ML.Types.NumericField lr) := TRANSFORM
      SELF.x              := lr.id;     // 1 based
      SELF.y              := lr.number; // 1 based
      SELF.v              := lr.value;
    END;

    d00 := PROJECT(indepDatatran, cvt_2_cell(LEFT));
		// OUTPUT (MAX(d00,y));
	
		Work1 := RECORD(Pbblas.Types.Layout_Cell)
    Pbblas.Types.partition_t     partition_id;
    Pbblas.Types.node_t          node_id;
    Pbblas.Types.dimension_t     block_row;
    Pbblas.Types.dimension_t     block_col;
  END;
	layout_mat := RECORD
		Pbblas.Types.dimension_t     x;    // 1 based index position
    Pbblas.Types.dimension_t     y;    // 1 based index position
    Pbblas.Types.matrix_t         mt;
	END;
Work2 := RECORD (layout_mat)
	
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
		Work2 cvt_2_xcell2(Layout_Cell lr) := TRANSFORM
      block_row           := mat_map.row_block(lr.x);
      block_col           := mat_map.col_block(lr.y + insert_columns);
      partition_id        := mat_map.assigned_part(block_row, block_col);
      SELF.partition_id   := partition_id;
      SELF.node_id        := mat_map.assigned_node(partition_id);
      SELF.block_row      := block_row;
      SELF.block_col      := block_col;
			SELF.mt := [lr.v];
      SELF := lr;
    END;
		
		
		
		
    inMatrix := cells.x BETWEEN 1 AND mat_map.matrix_rows
            AND cells.y BETWEEN 1 AND mat_map.matrix_cols - insert_columns;
    d0 := PROJECT(cells(inMatrix), cvt_2_xcell(LEFT),local);
		d02 := PROJECT(cells(inMatrix), cvt_2_xcell2(LEFT),local);
    d1 := DISTRIBUTE(d0, node_id);
		d12 := DISTRIBUTE(d02, node_id);
    d2 := SORT(d1, partition_id, y, x, LOCAL); 
		d22 := SORT(d12, partition_id, y, x, LOCAL);// prep for column major
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
		
		
 SET OF REAL8 appnd(dimension_t r, dimension_t s,
                              dimension_t first_row, dimension_t first_col,
                              DATASET(Layout_Cell) D,
                              dimension_t insert_columns,
                              value_t insert_value) := BEGINC++
    typedef struct work1 {      // copy of Layout_Cell translated to C
      uint32_t x;
      uint32_t y;
      double v;
    };
    #body
    __lenResult = r * s * sizeof(double);
    __isAllResult = false;
    double * result = new double[r*s];
    __result = (void*) result;
    work1 *cell = (work1*) d;
    uint32_t cells = lenD / sizeof(work1);
    uint32_t i;
    uint32_t pos;
    for (i=0; i<r*s; i++) {
      result[i] =  i/r < insert_columns  ? insert_value   : 0.0;
    }
    int x, y;
    for (i=0; i<cells; i++) {
      x = cell[i].x - first_row;                   // input co-ordinates are one based,
      y = cell[i].y + insert_columns - first_col;  //x and y are zero based.
      if(x < 0 || (uint32_t) x >= r) continue;   // cell does not belong
      if(y < 0 || (uint32_t) y >= s) continue;
      pos = (y*r) + x;
      result[pos] = cell[i].v;
    }
  ENDC++;
		
		
		
		
Layout_Part roll_cells2(Work2 parent, Work2 cells) := TRANSFORM
      first_row     := mat_map.first_row(parent.partition_id);
      first_col     := mat_map.first_col(parent.partition_id);
      part_rows     := mat_map.part_rows(parent.partition_id);
      part_cols     := mat_map.part_cols(parent.partition_id);
      SELF.mat_part := [1,2];
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
    RETURN d02;
  END; 
		
		result := FromCells(mat_map, d00, insert_columns, insert_value);
		
		OUTPUT ((result));
		
		// result_r := PROJECT (result, TRANSFORM ({LAYOUT_part, UNSIGNED rn}, SELF.rn := LEFT.mat_part[1]; SELF:= LEFT),LOCAL);
		// OUTPUT(result,,'~thor::maryam::mytest::kk',CSV(HEADING(SINGLE)), OVERWRITE);
// OUTPUT (result_r);
		/*
		
R1 := RECORD
  result_r.rn;
  
  Number := COUNT(GROUP);

END;
T1 := TABLE(result_r, R1,  rn);

R2 := RECORD
	result.node_id;
  result.partition_id;
  cnt := COUNT(GROUP);

END;
T2 := TABLE(result, R2, node_id, partition_id, LOCAL);
OUTPUT (T2,ALL);
		// OUTPUT (T2);
	*/	
		/*
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

resultnode := PROJECT(result, TRANSFORM (thsirec,  SELF.real_node := STD.System.Thorlib.Node(); SELF := LEFT), LOCAL);
// OUTPUT (resultnode, named ('resultnode'),ALL);
OUTPUT (MAX (d00,x));
OUTPUT (MIN (d00,x));
OUTPUT (MAX (d00,y));
OUTPUT (MIN (d00,y));
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
// output (llgrnd, named('llgrnd'));*/