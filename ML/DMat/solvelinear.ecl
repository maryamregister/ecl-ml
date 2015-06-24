// Invert the matrix.  This version multiplies input by the transpose to avoid
//need to permute the factorization.  A**t A  A**-1  = A**t is the equation.
//If A is not square, this finds the right inverse.  An optional parameter
//is used to find the left inverse.
IMPORT PBblas;
IMPORT PBblas.IMatrix_Map;
IMPORT PBblas.Types;
IMPORT ML.DMAT;
Part := Types.Layout_Part;
Side := Types.Side;
Triangle := Types.Triangle;
Diagonal := Types.Diagonal;
//Solve systems of linear equations Ax = B for x
EXPORT solvelinear(IMatrix_Map map_a, DATASET(Part) A, BOOLEAN findLeft=FALSE, IMatrix_Map map_b, DATASET(Part) B) := FUNCTION
  map_at := PBblas.Matrix_Map(map_a.matrix_cols, map_a.matrix_rows,
                             map_a.part_cols(1), map_a.part_rows(1));
  AT := PBblas.PB_dtran(map_a, map_at, 1.0, A);
  C_cols := map_at.matrix_cols;
  C_rows := map_at.matrix_rows;
  C_pcol := map_at.part_rows(1);
  C_prow := map_at.part_cols(1);
  map_c  := PBblas.Matrix_Map(C_rows, C_cols, C_prow, C_pcol);
  C := PBblas.PB_dgemm(TRUE, FALSE, 1.0, map_at, A, map_a, A, map_c);
  F := PBblas.PB_dpotrf(Triangle.Lower, map_c, C);
  sideSw := Side.Ax;
  map_ATB := PBblas.Matrix_Map(map_at.matrix_rows, map_b.matrix_cols, map_at.part_rows(1), map_b.part_cols(1));
  ATB := PBblas.PB_dgemm(TRUE, FALSE, 1.0, map_a, A, map_b, B, map_ATB);
  S := PBblas.PB_dtrsm(sideSw, Triangle.Lower, FALSE,
                       Diagonal.NotUnitTri, 1.0, map_c, F, map_ATB, ATB);
  T := PBblas.PB_dtrsm(sideSw, Triangle.Upper, TRUE,
                       Diagonal.NotUnitTri, 1.0, map_c, F, map_ATB, S);
  RETURN T;
END;