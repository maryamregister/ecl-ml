//Apply a function to each element of the matrix
IMPORT PBblas;
IMPORT PBblas.IMatrix_Map;
IMPORT PBblas.Types;
IMPORT PBblas.Constants;
IMPORT PBblas.BLAS;
//Alias entries for convenience
Part := Types.Layout_Part;
value_t := Types.value_t;
IFunc := PBblas.IElementFunc;
dim_t := PBblas.Types.dimension_t;


EXPORT SumElements( DATASET(Part) X) := FUNCTION
  Elem := {value_t v};  //short-cut record def
  
  Elem dosum(Part lr) := TRANSFORM
    s:= SUM(lr.mat_part);
    SELF.v := s;
  END;
  sumdataset := PROJECT(X, dosum(LEFT));
  RETURN SUM(sumdataset,sumdataset.v);
END;
