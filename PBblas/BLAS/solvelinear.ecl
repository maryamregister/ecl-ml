//Solve systems of linear equations Ax = B for x
//A is a lda by K matrix
//B is a M by N matrix

IMPORT PBblas;
IMPORT PBblas.Types;
dimension_t := Types.dimension_t;
value_t     := Types.value_t;
matrix_t    := Types.matrix_t;
Side := Types.Side;
Triangle := Types.Triangle;
Diagonal := Types.Diagonal;

//B is M*N
//A is lda * K
  EXPORT  solvelinear (matrix_t Aset, matrix_t Bset, dimension_t M, dimension_t N,  dimension_t lda, dimension_t K) := FUNCTION

    //cset = Aset'*Aset;
    Cset := PBblas.BLAS.dgemm(TRUE, FALSE,K, K, lda, 1.0, Aset, Aset, 0.0); 
    Fset := PBblas.LAPACK.dpotf2(Triangle.Lower, K, Cset);
    ATBset := PBblas.BLAS.dgemm(TRUE, FALSE, K, N, lda, 1.0, Aset, Bset, 0.0);
    sideSw := Side.Ax;
    Sset := PBblas.BLAS.dtrsm(sideSw, Triangle.Lower, FALSE, Diagonal.NotUnitTri, K,  N,  K, 1.0, Fset, ATBset);
    //transpose F
    Fsize := K*K;
    coll := K;
    myrec := RECORD
      value_t number;
    END; 
    
    myrec tran(UNSIGNED4 c) := TRANSFORM
      SELF.number := Fset[(coll*((c-1)%coll))+((c-1) DIV coll)+1];
    END;

    Ftr := DATASET(Fsize, tran(COUNTER));

    //FSet_TR := SET (Ftr, number);
    Fset_TR := PBblas.Block.trans (lda,K,Fset);
    Tset := PBblas.BLAS.dtrsm (sideSw, Triangle.Upper, FALSE, Diagonal.NotUnitTri, K,N,  K, 1.0, FSet_TR, Sset);
    RETURN Tset;
  END;//END solvelinear


