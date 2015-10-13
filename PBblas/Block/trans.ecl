//transpose matrix x os size m by n. the result is the transpose of matrix x with teh size n by m
IMPORT PBblas;
matrix_t := PBblas.Types.matrix_t;
dimension_t := PBblas.Types.dimension_t;
value_t := PBblas.Types.value_t;

EXPORT matrix_t trans(dimension_t m, dimension_t n, matrix_t x=[]) := BEGINC++
#body
  int cells = m * n;
  __isAllResult = false;
  __lenResult = cells * sizeof(double);
  double *tr = new double[cells];
  double *in_x = (double*)x;
  unsigned int r, c;    //row and column
  for (int i=0; i<cells; i++) {
    r = i % n;
    c = i / n;
    //c = (m*((i-1)%m))+((i-1) DIV m)+1 ; 
    //tr[i] = in_x[c*m+r+1];
    //tr[i] = in_x[r*3+c+1];
    tr[i]= in_x[r*m+c];
  }
  __result = (void*) tr;
ENDC++;

