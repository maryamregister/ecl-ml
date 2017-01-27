//Take a dataset of cells for a partition and pack into a dense matrix.  Specify Row or Column major
//First row and first column are one based.
//Insert is used insert columns with a spacific value.  Typical use is building a matrix for a solver
//where the first column is an inserted column of 1 values for the intercept.
IMPORT ML;
IMPORT * FROM $;
IMPORT PBblas.Types;
dimension_t := Types.dimension_t;
value_t := Types.value_t;
Layout_Cell := Types.Layout_Cell;
matrix_t := Types.matrix_t;
lnrec := RECORD (Layout_Cell)
UNSIGNED4 node_id;
END;






 SET OF REAL8 part_sparse_mul(dimension_t r, dimension_t s, dimension_t samp, dimension_t num, matrix_t M, DATASET(Layout_Cell) D) := BEGINC++
    typedef struct work1 {      // copy of numericfield translated to C
      uint32_t x;
      uint32_t y;
      double v;
    };
    #body
    __lenResult = r * samp * sizeof(double);
    __isAllResult = false;
    double *result = new double[r * samp];
    __result = (void*) result;
    work1 *celld = (work1*) d;
		double *cellm = (double*) m;
		uint32_t cells = num;
    uint32_t i, j;
    uint32_t pos;
    for (i=0; i< r * samp; i++) {
      result[i] =  0.0;
    }
		uint32_t x, y;
		for (i=0; i < cells; i++){
			x = celld[i].x - 1;   // input co-ordinates are one based,
      y = celld[i].y - 1;  //x and y are zero based.
			for (j=0; j<r; j++){
				pos = y * r + j;
				result[pos] = result[pos] + cellm[r * x + j] * celld[i].v;
			}		
		}
  ENDC++;
	
	
	 SET OF REAL8 part_sparse_mul2(dimension_t r, dimension_t s, dimension_t samp, dimension_t num, matrix_t M, DATASET(Layout_Cell) D) := BEGINC++
    typedef struct work1 {      // copy of numericfield translated to C
      uint32_t x;
      uint32_t y;
      double v;
    };
    #body
    __lenResult = r * samp * sizeof(double);
    __isAllResult = false;
    double *result = new double[r * samp];
    __result = (void*) result;
    work1 *celld = (work1*) d;
		double *cellm = (double*) m;
		uint32_t cells = num;
    uint32_t i, j;
    uint32_t pos;
    for (i=0; i< r * samp; i++) {
      result[i] =  0.0;
    }
		uint32_t x, y;
		for (i=0; i < cells; i++){
			result [i] = celld[i].v;
		}
  ENDC++;
	
REAL8 simplefunc() := BEGINC++
 struct work2 {      // copy of numericfield translated to C
      uint64_t id;
      uint32_t number;
			double v;
    };
    #body
    double result = 0;
		return(sizeof(work2));

  ENDC++;
OUTPUT (simplefunc());

thism := [1 ,2, 3 ,4, 5 ,6];
thisd := DATASET ([{1,1,2,0},
{2,2,10,0},
{3,2,11,0},
{1,3,3,0},
{2,4,4,0},
{1,5,7,0}
], lnrec);

thispart := DATASET ([{0,1,1,1,1,1,1,1,[1 ,2, 3 ,4, 5 ,6]},{0,1,1,1,1,1,1,1,[1 ,2, 3 ,4, 5 ,6]}],TYPES.Layout_part);
OUTPUT (thispart);
thisres := part_sparse_mul(2, 3, 5, 6, thism, PROJECT(thisd, TRANSFORM (layout_cell, SELF:= LEFT)));

OUTPUT (thisres);


TYPES.Layout_Part thistran (Types.Layout_Part le, DATASET(Layout_Cell) cells) := TRANSFORM 
SELF.mat_part := part_sparse_mul(2, 3, 5, 6, le.mat_part, PROJECT(cells, TRANSFORM (layout_cell, SELF:= LEFT)));
SELF := le;
END;

a := DENORMALIZE(thispart, thisd,
                            LEFT.node_id = RIGHT.node_id,
                            GROUP,
                            thistran(LEFT,ROWS(RIGHT)), LOCAL);




OUTPUT (a);
