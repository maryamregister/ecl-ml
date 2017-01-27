IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat; 
IMPORT * FROM ML.Types;
IMPORT PBblas;
Layout_Cell := PBblas.Types.Layout_Cell4;
b := DATASET ([{1,1,1},
{1,2,2},
{2,1,3},
{2,2,4},
{3,1,8},
{3,2,10}], Layout_Cell);

a := Pbblas.makeR4Set(3, 2,1, 1, b,0,0);
// output (b);
// output (a);

Y := DATASET ([{1,1,1},{2,1,10}], Types.NumericField4);
 d := Utils.DistinctLabeltoGroundTruth4 (Y);
 // output (d);
 
  h := utils.distrow_ranmap_part4(4, 3, 2,1 );
	output (h);
	
	SET OF Pbblas.Types.value_t4 part_sparse_mul(PBblas.types.dimension_t r, PBblas.types.dimension_t s, PBblas.types.dimension_t samp, PBblas.types.dimension_t num, PBblas.types.matrix_t4 M, DATASET(PBblas.types.Layout_Cell4) D) := BEGINC++
    typedef struct work3 {      // copy of numericfield4 translated to C
      uint32_t x;
      uint32_t y;
      float v;
    };
    #body
    __lenResult = r * samp * sizeof(float);
    __isAllResult = false;
    float *result = new float[r * samp];
    __result = (void*) result;
    work3 *celld = (work3*) d;
		float *cellm = (float*) m;
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
	
	coco := DATASET ([{1,1,10},
	{2,2,20},
	{3,2,50},
	{1,3,4},
	{2,4,3}],PBblas.types.Layout_Cell4);
	
	dd := part_sparse_mul (2, 3, 4, 5, [1,2,3,4,5,6], coco);
	output (dd);
	
	
	SET OF Pbblas.Types.value_t4 max_col (PBblas.Types.dimension_t r, PBblas.Types.dimension_t s, PBblas.Types.matrix_t4 D) := BEGINC++

    #body
    __lenResult = s * sizeof(float);
    __isAllResult = false;
    float * result = new float[s];
    __result = (void*) result;
    float *cell = (float*) d;
		float max_tmp;
    uint32_t i;
		uint32_t j;
    uint32_t pos;
		uint32_t posj;
		for (i=0; i<s; i++) {
			pos = i * r;
			max_tmp = cell[pos];
			for (j=1; j<r; j++)
			{
				posj = pos +j;
				if(cell[posj]>max_tmp)
					max_tmp = cell[posj];
			}
			result[i]=max_tmp;
    }

  ENDC++;
	
	SET OF Pbblas.Types.value_t4 arr_max (PBblas.Types.dimension_t s, PBblas.Types.matrix_t4 C, PBblas.Types.matrix_t4 D) := BEGINC++
	#body
    __lenResult = s * sizeof(float);
    __isAllResult = false;
    float * result = new float[s];
    __result = (void*) result;
    float *celld = (float*) d;
		float *cellc = (float*) c;
		float max_tmp;
    uint32_t i;
		uint32_t j;
    uint32_t pos;
		uint32_t posj;
		for (i=0; i<s; i++) {
			if (celld[i]>cellc[i])
			{
				result[i]= celld[i];
			}
			else
			{
				result[i]= cellc[i];
			}
    }

  ENDC++;
	
	OUTPUT (arr_max(6, [4,50,80, 90, 20, 20],[4,50,8, 90, 2, 2]));
	
	SET OF Pbblas.Types.value_t4 part_sparse_sum(PBblas.types.dimension_t r, PBblas.types.dimension_t s, PBblas.types.dimension_t f, PBblas.types.matrix_t4 M, PBblas.types.matrix_t4 D) := BEGINC++

    #body
    __lenResult = r * s * sizeof(float);
    __isAllResult = false;
    float * result = new float[r*s];
    __result = (void*) result;
		float *cellm = (float*) m;
    float *celld = (float*) d;
		float tmp;
    uint32_t i;
		uint32_t j;
    uint32_t pos;
		uint32_t f_ = f-1;
		uint32_t posj = r+f;
		for (i=0; i<s*r; i++) {
			result[i] = cellm[i];
    }
		for (i=0; i<s; i++) {
			tmp = celld[i];
			if (tmp >= f && tmp <posj) {
				pos = (i*r) + tmp - f;
				result [pos] = result [pos] - 1;
			}
		}

  ENDC++;
	
	// OUTPUT (part_sparse_sum(2, 2, 3, [30,40,70,80], [4,5]));
	
	SET OF Pbblas.Types.value_t4 summation(PBblas.types.dimension_t N, PBblas.types.matrix_t4 M, PBblas.types.matrix_t4 D) := BEGINC++

    #body
    __lenResult = n * sizeof(float);
    __isAllResult = false;
    float * result = new float[n];
    __result = (void*) result;
		float *cellm = (float*) m;
    float *celld = (float*) d;
    uint32_t i;
		for (i=0; i<n; i++) {
			result[i] = cellm[i]+celld[i];
    }
		
  ENDC++;
	
	// OUTPUT (summation(4, [1,2,3,4],[40,50,60,70]));
	
	Pbblas.Types.value_t4 log_cost_c(PBblas.types.dimension_t r, PBblas.types.dimension_t s, PBblas.types.dimension_t f, PBblas.types.matrix_t4 M, PBblas.types.matrix_t4 D) := BEGINC++

    #body
    float result = 0;
		float *cellm = (float*) m;
    float *celld = (float*) d;
		float tmp;
    uint32_t i;
		uint32_t j;
    uint32_t pos;
		uint32_t f_ = f-1;
		uint32_t posj = r+f;
		for (i=0; i<s; i++) {
			tmp = celld[i];
			if (tmp >= f && tmp <posj) {
				pos = (i*r) + tmp - f;
				result  = result + (cellm [pos]);
				
			}
		}
return (result);
  ENDC++;
	



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
// OUTPUT (to4(4,[12.234, 23.454,12.90,50]), named ('tihs'));

	SET OF Pbblas.Types.value_t x2_der(PBblas.types.dimension_t N, PBblas.types.matrix_t M) := BEGINC++

    #body
    __lenResult = n * sizeof(double);
    __isAllResult = false;
    double * result = new double[n];
    __result = (void*) result;
		double *cellm = (double*) m;
    uint32_t i;
		for (i=0; i<n; i++) {
			result[i] = 2 * cellm[i];
    }
		
  ENDC++;
	
	// OUTPUT (x2_der(4,[5,10,19.868,12.354]), named ('resuljhl'));
	
	OUTPUT (utils.distcol_ranmap_part(1,9,5,1.0), named ('matrix'));
	
	
	UNSIGNED4 funfun( PBblas.types.matrix_t M) := BEGINC++
		uint32_t result;
		double *cellm = (double*) m;
		result = (uint32_t)cellm[2];
	return (result);
  ENDC++;
	
	// OUTPUT (funfun([12186348,349698, 2147483650]),named('funfun'));
	// OUTPUT (log_cost_c(2,3,1,[1,12,18,19,20,13],[1,2,4]));
	
	// OUTPUT (ML.Utils.distrow_ranmap_part(4,5,2 , 0.005));
	
	kk := ML.Utils.distrow_ranmap_part4(4,2,2 , 1) ;
	// OUTPUT (kk);
	
	// SET OF Pbblas.Types.value_t4 rand_vec(PBblas.Types.dimension_t N, PBblas.Types.dimension_t s, REAL4 cc) := BEGINC++

    // #body
    // __lenResult = n * sizeof(float);
    // __isAllResult = false;
    // float * result = new float[n];
    // __result = (void*) result;
    // uint32_t i;
		// double G = 1000000.0;
		// srand (s);
    // for (i=0; i<n; i++) {
      // result[i] = (rand() % G) / G;
			// srand (s);
			// result[i] =  cc * ((float) rand() / (float)(RAND_MAX));
    // }
  // ENDC++;
	
	// OUTPUT ( utils.distrow_ranmap_part4(4, 3, 2, 1.0 ));
	
	
// SET OF Pbblas.Types.value_t4 testit(PBblas.types.dimension_t r, PBblas.types.dimension_t s, PBblas.types.dimension_t f, PBblas.types.matrix_t4 M, PBblas.types.matrix_t4 D) := BEGINC++

    // #body
    // __lenResult = s* sizeof(float);
    // __isAllResult = false;
    // float * result = new float[s];
    // __result = (void*) result;
		// float *cellm = (float*) m;
    // float *celld = (float*) d;
		// float tmp;
    // uint32_t i;
		// uint32_t j=0;
    // uint32_t pos;
		// uint32_t f_ = f-1;
		// uint32_t posj = r+f;
		// for (i=0; i<s; i++) {
		// result[i]=-100;
		// }
		// for (i=0; i<s; i++) {
			// tmp = celld[i];
			// if (tmp >= f && tmp <posj) {
				// pos = (i*r) + tmp - f;
				// result [j] = cellm[pos] ;
				// j = j+1;
			// }
		// }
// result [j]=j;
  // ENDC++;
	
	
		
// PBblas.types.dimension_t testit2(PBblas.types.dimension_t r, PBblas.types.dimension_t s, PBblas.types.dimension_t f, PBblas.types.matrix_t4 M, PBblas.types.matrix_t4 D) := BEGINC++

    // #body
    // uint32_t result = 0;
		// float *cellm = (float*) m;
    // float *celld = (float*) d;
		// uint32_t tmp=12294;
    // uint32_t i;
		// uint32_t j=0;
    // uint32_t pos;
		// uint32_t f_ = f-1;
		// uint32_t posj = r+f;

		// i=93804;
		// pos = (i*r) + 12294 - 12055;
		// result =(93804 * 240)+12294 - 12055;
  // ENDC++;
	// OUTPUT (testit2(240, 93805, 12055, [10,6,1,7,2,9,3,11], [8,2,7,4]));
	
	// SET OF Pbblas.Types.value_t4 scale_mat (PBblas.Types.dimension_t N, PBblas.Types.matrix_t4 M, Pbblas.Types.value_t4 c) := BEGINC++

    // #body
    // __lenResult = n * sizeof(float);
    // __isAllResult = false;
    // float * result = new float[n];
    // __result = (void*) result;
    // float *cellm = (float*) m;
    // uint32_t i;
		// for (i=0; i<n; i++){
			// result[i] = cellm[i]*c;
		// }

  // ENDC++;
	
	// OUTPUT (scale_mat (5, [1.0,2,3,4,5], 11.1));