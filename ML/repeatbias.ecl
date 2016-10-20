//Take a dataset of cells for a partition and pack into a dense matrix.  Specify Row or Column major
//First row and first column are one based.
//Insert is used insert columns with a spacific value.  Typical use is building a matrix for a solver
//where the first column is an inserted column of 1 values for the intercept.
IMPORT PBblas.Types;
IMPORT PBblas;
dimension_t := PBblas.Types.dimension_t;
value_t := PBblas.Types.value_t;
Layout_Cell := PBblas.Types.Layout_Cell;
matrix_t    := PBblas.Types.matrix_t;

// EXPORT SET OF REAL8 repeatbias(dimension_t r, dimension_t s, matrix_t D) := BEGINC++

    // #body
    // __lenResult = r * s * sizeof(double);
    // __isAllResult = false;
    // double * result = new double[r*s];
    // __result = (void*) result;
    // double *cell = (double*) d;
    // uint32_t cells =  r * s;
    // uint32_t i;
    // uint32_t pos;
    // for (i=0; i<cells; i++) {
      // pos = i % r;
      // result[i] = cell[pos];
    // }
  // ENDC++;
	
	// EXPORT SET OF REAL8 d3_calculate(PBblas.Types.dimension_t N, PBblas.Types.matrix_t A, PBblas.Types.matrix_t Y) := BEGINC++

    // #body
    // __lenResult = n * sizeof(double);
    // __isAllResult = false;
    // double * result = new double[n];
    // __result = (void*) result;
    // double *cella = (double*) a;
		// double *celly = (double*) y;
    // uint32_t cells =  n;
    // uint32_t i;
    // for (i=0; i<cells; i++) {
      // result[i] = (cella[i]-celly[i])*(cella[i]*(1-cella[i]));
    // }
  // ENDC++;
	
		// EXPORT SET OF REAL8 mat_vec_sum(PBblas.Types.dimension_t N, PBblas.Types.dimension_t r, PBblas.Types.matrix_t M, PBblas.Types.matrix_t V) := BEGINC++

    // #body
    // __lenResult = n * sizeof(double);
    // __isAllResult = false;
    // double * result = new double[n];
    // __result = (void*) result;
    // double *cellm = (double*) m;
		// double *cellv = (double*) v;
    // uint32_t cells =  n;
    // uint32_t i;
		// uint32_t pos;
    // for (i=0; i<cells; i++) {
		  // pos = i % r;
      // result[i] = cellm[i] + cellv[pos];
    // }
  // ENDC++;
	
	
	// EXPORT	SET OF REAL8 mat_vec_sum_sigmoid(PBblas.Types.dimension_t N, PBblas.Types.dimension_t r, PBblas.Types.matrix_t M, PBblas.Types.matrix_t V) := BEGINC++

    // #body
    // __lenResult = n * sizeof(double);
    // __isAllResult = false;
    // double * result = new double[n];
    // __result = (void*) result;
    // double *cellm = (double*) m;
		// double *cellv = (double*) v;
    // uint32_t cells =  n;
    // uint32_t i;
		// uint32_t pos;
    // for (i=0; i<cells; i++) {
		  // pos = i % r;
      // result[i] = 1/(1 + exp(-1*(cellm[i] + cellv[pos])));
    // }
  // ENDC++;


// EXPORT SET OF REAL8 repeatbias(dimension_t r, dimension_t s, matrix_t D) := BEGINC++

    // #body
    // __lenResult = r * sizeof(double);
    // __isAllResult = false;
    // double * result = new double[r];
    // __result = (void*) result;
    // double *cell = (double*) d;
    // uint32_t cells =  r * s;
    // uint32_t i;
    // uint32_t pos;
		// for (i=0; i<r; i++) {
      // result[i] = 0;
    // }
    // for (i=0; i<cells; i++) {
      // pos = i % r;
      // result[pos] = result[pos] + cell[i];
    // }
		// for (i=0; i<r; i++) {
      // result[i] = result[i]/s;
    // }
  // ENDC++;
	// sum(0.5*sum((x-a3).^2))
	// EXPORT REAL8 repeatbias(PBblas.Types.dimension_t N, PBblas.Types.matrix_t M, PBblas.Types.matrix_t V) := BEGINC++

    // #body
    // double result = 0;
		// double tmpp ;
    // double *cellm = (double*) m;
		// double *cellv = (double*) v;
    // uint32_t i;
		// for (i=0; i<n; i++) {
		  // tmpp =(cellm[i] - cellv [i]);
      // result = result + (tmpp*tmpp);
    // }
		// return(0.5*result);

  // ENDC++;
	
		// EXPORT REAL8 repeatbias(PBblas.Types.dimension_t N, PBblas.Types.matrix_t M, PBblas.Types.value_t rho) := BEGINC++

    // #body
    // double result = 0;
		// double tmpp ;
    // double *cellm = (double*) m;
    // uint32_t i;
		// for (i=0; i<n; i++) {
			// result = result + (rho*log(rho/cellm[i])) + ((1-rho)*log((1-rho)/(1-cellm[i])));
    // }
		// return(result);

  // ENDC++;
	
	EXPORT REAL8 repeatbias(PBblas.Types.dimension_t N, PBblas.Types.matrix_t M, PBblas.Types.matrix_t V) := BEGINC++

    #body
    double result = 0;
		double tmpp ;
    double *cellm = (double*) m;
		double *cellv = (double*) v;
    uint32_t i;
		for (i=0; i<n; i++) {
		  tmpp =(cellm[i] * cellv [i]);
      result = result + tmpp;
    }
		return(result);

  ENDC++;
	
