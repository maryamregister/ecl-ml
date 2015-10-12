

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

// onemap := PBblas.Matrix_Map (1,1,1, 1);
// x := DATASET([
// {1, 1, 7675}],
// Mat.Types.Element);
//output(x);
// xdist := DMAT.Converted.FromElement(x,onemap);
//output(xdist);
// output(xdist[1].mat_part[1]);

  // no_t_t_ := DATASET([
  // { 1,	1,	1.52316841663803	,1},
  // {1,	1	,8.04810059244658,	2},
  // {1,	1	,2.0,	4},
  // {1,	1	,0.1546146566020936	,3},
  // {2,	1	,0.1546146566020936	,3},
  // {3	,1	,0.1546146566020936,	3},{
// 4	,1	,0.1546146566020936,	3}],Mat.Types.MUElement);

  // t_new := Mat.MU.FROM (no_t_t_,1)[1].value; 
  
  // output(no_t_t_);
  
  EXPORT  polyinterp_both (REAL8 t_1, REAL8 f_1, REAL8 gtd_1, REAL8 t_2, REAL8 f_2, REAL8 gtd_2, REAL8 xminBound, REAL8 xmaxBound) := FUNCTION
    poly1 := FUNCTION
      setp1 := FUNCTION
        points := DATASET([{1,1,t_1},{2,1,t_2},{3,1,f_1},{4,1,f_2},{5,1,gtd_1},{6,2,gtd_2}], Types.NumericField);
        RETURN points;
      END;
      setp2 := FUNCTION
        points := DATASET([{2,1,t_1},{1,1,t_2},{4,1,f_1},{3,1,f_2},{6,1,gtd_1},{5,2,gtd_2}], Types.NumericField);
        RETURN points;
      END;
      orderedp := IF (t_1<t_2,setp1 , setp2);
      //Find the min and max values
      tmin := orderedp (id=1)[1].value;
      tmax := orderedp (id=2)[1].value;
      fmin := orderedp (id=3)[1].value;
      fmax := orderedp (id=4)[1].value;
      gtdmin := orderedp (id=5)[1].value;
      gtdmax := orderedp (id=6)[1].value;
      //build A and B matrices
      // A= [t_1^3 t_1^2 t_1 1
      //    t_2^3 t_2^2 t_2 1
      //    3*t_1^2 2*t_1 t_1 0
      //    3*t_2^2 2*t_2 t_2 0]
      //b = [f_1 f_2 dtg_1 gtd_2]'
      AA := DATASET([
      {1,1,POWER(t_1,3)},
      {1,2,POWER(t_1,2)},
      {1,3,POWER(t_1,3)},
      {1,4,1},
      {2,1,POWER(t_2,3)},
      {2,2,POWER(t_2,2)},
      {2,3,POWER(t_2,1)},
      {2,4,1},
      {3,1,3*POWER(t_1,2)},
      {3,2,2*t_1},
      {3,3,1},
      {3,4,0},
      {4,1,3*POWER(t_2,2)},
      {4,2,2*t_2},
      {4,3,1},
      {4,4,0}],
      Types.NumericField);
      bb := DATASET([
      {1,1,f_1},
      {2,1,f_2},
      {3,1,gtd_1},
      {4,1,gtd_2}],
      Types.NumericField);
      // Find interpolating polynomial
      A_map := PBblas.Matrix_Map(4, 4, 4, 4);
      b_map := PBblas.Matrix_Map(4, 1, 4, 1);
      A_part := ML.DMat.Converted.FromNumericFieldDS (AA, A_map);
      b_part := ML.DMat.Converted.FromNumericFieldDS (bb, b_map);
      //params = A\b;

      params_part := DMAT.solvelinear (A_map,  A_part, FALSE, b_map, b_part) ; // for now
      
      
      //solvelinear(IMatrix_Map map_a, DATASET(Part) A, BOOLEAN findLeft=FALSE, IMatrix_Map map_b, DATASET(Part) B) 
      map_a := A_map;
      A := A_part;
      findLeft := FALSE;
      map_b := b_map;
      B := b_part;      
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
                           
     //set
     Aset := SET(AA,value);
     Bset := SET (bb, value);
     //cset = Aset'*Aset;
     Cset := PBblas.BLAS.dgemm(FALSE, TRUE,
                      4, 4, 4,
                      1.0, Aset, Aset,
                      0.0); //wierd, I wanted ASet (transpose) by Aset, I had to put the second transpose at True though
            //Bblas.PB_dpotrf(Triangle.Lower, map_c, C);          
     Fset := PBblas.LAPACK.dpotf2(Triangle.Lower, 4, Cset); // this is actually F(transpose)
     ATBset := PBblas.BLAS.dgemm(FALSE, TRUE,
                      4, 1, 4,
                      1.0, Aset, Bset,
                      0.0);
             
// S := PBblas.PB_dtrsm(sideSw, Triangle.Lower, FALSE,
                           // Diagonal.NotUnitTri, 1.0, map_c, F, map_ATB, ATB);
      // T := PBblas.PB_dtrsm(sideSw, Triangle.Upper, TRUE,
                           // Diagonal.NotUnitTri, 1.0, map_c, F, map_ATB, S);             
             
             
     Sset := PBblas.BLAS.dtrsm(sideSw, Triangle.Lower,
                      FALSE, Diagonal.NotUnitTri,
                       4,  1,  4,
                      1.0, Fset, ATBset);
    
    
    //transpose F
    Fsize := 16;
R := 4;
coll := Fsize/R;
myrec := RECORD
REAL number;
END; 
  
  
        myrec tran(UNSIGNED4 c) := TRANSFORM
        SELF.number := Fset[(coll*((c-1)%coll))+((c-1) DIV coll)+1];

      END;
      
      Ftr := DATASET(Fsize, tran(COUNTER));
      
      FSet_TR := SET (Ftr, number);
      
    Tset := PBblas.BLAS.dtrsm (sideSw, Triangle.Upper,
                      FALSE, Diagonal.NotUnitTri,
                      4,1,  4,
                      1.0, FSet, Sset);
 Aset2 := [1,216,3,108, 1, 36, 2, 12, 1, 6, 1,1,1,1,0,0 ];
 

  // Cset2 := PBblas.BLAS.dgemm(TRUE, FALSE,
                      // 4, 4, 4,
                      // 1.0, Aset2, Aset2,
                      // 0.0);
 // Fset2 := PBblas.LAPACK.dpotf2(Triangle.Lower, 4, Cset2);
 
  // ATBset2 := PBblas.BLAS.dgemm(TRUE, FALSE,
                      // 4, 1, 4,
                      // 1.0, Aset2, Bset,
                      // 0.0);
                      
     // Sset2 := PBblas.BLAS.dtrsm(sideSw, Triangle.Lower,
    // FALSE, Diagonal.NotUnitTri,
     // 4,  1,  4,
    // 1.0, Fset2, ATBset2);
    
      // Tset2 := PBblas.BLAS.dtrsm (sideSw, Triangle.Upper,
                      // FALSE, Diagonal.NotUnitTri,
                      // 4,1,  4,
                      // 1.0, Fset2, Sset2);
//RETURN DMat.Converted.FromPart2Elm(T);

//solvelinear (matrix_t Aset, matrix_t Bset, dimension_t M, dimension_t N,  dimension_t lda, dimension_t K)
bss := [ 42.5000,  137.7000  ,276.7000];
ass :=  [ 10 ,   13  ,  90,
    11 ,   12 ,   91,
    14,    43 ,   92,
    15,    56 ,   93];

tt := PBblas.BLAS.solvelinear (ass, bss, 3,1,3,4);
   RETURN tt;
    END;//END poly1
    polResult := poly1;
    RETURN polResult;
  END;//end polyinterp_both
  
  newt := polyinterp_both (1, 0,5, 6, 5, 4, 2, 1);
  output(newt, named('newt'));

