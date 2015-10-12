IMPORT PBblas;
IMPORT PBblas.IMatrix_Map;
IMPORT PBblas.Types;
IMPORT ML.DMAT;
Part := Types.Layout_Part;
Side := Types.Side;
Triangle := Types.Triangle;
Diagonal := Types.Diagonal;

IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat;
IMPORT * FROM ML.Types;
IMPORT PBblas;
Layout_Cell := PBblas.Types.Layout_Cell;
Layout_Part := PBblas.Types.Layout_Part;
emptyC := DATASET([], Types.NumericField);
bss := [ 42.5000,  137.7000  ,276.7000];
ass :=  [ 10 ,   13  ,  90,
    11 ,   12 ,   91,
    14,    43 ,   92,
    15,    56 ,   93];

Aset := ass;
bset := bss;
M:= 3;
N:= 1;
lda := 3;
K:= 4;
sideSw := Side.Ax;

  Cset := PBblas.BLAS.dgemm(TRUE, FALSE,K, K, lda, 1.0, Aset, Aset, 0.0); 
  Fset := PBblas.LAPACK.dpotf2(Triangle.Lower, 4, Cset);
  ATBset := PBblas.BLAS.dgemm(TRUE, FALSE, K, N, lda, 1.0, Aset, Bset, 0.0);
  Sset := PBblas.BLAS.dtrsm(sideSw, Triangle.Lower, FALSE, Diagonal.NotUnitTri, K,  N,  K, 1.0, Fset, ATBset);
  Fsize := K*K;
  coll := K;
    myrec := RECORD
    REAL number;
  END; 

  myrec tran(UNSIGNED4 c) := TRANSFORM
    SELF.number := Fset[(coll*((c-1)%coll))+((c-1) DIV coll)+1];
  END;

  Ftr := DATASET(Fsize, tran(COUNTER));

  FSet_TR := SET (Ftr, number);

  Tset := PBblas.BLAS.dtrsm (sideSw, Triangle.Upper, FALSE, Diagonal.NotUnitTri, K,N,  K, 1.0, FSet_TR, Sset);

OUTPUT(Fset);
//cset = Aset'*Aset;
  