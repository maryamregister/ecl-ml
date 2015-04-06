IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat;
IMPORT * FROM ML.Types;
IMPORT PBblas;
Layout_Cell := PBblas.Types.Layout_Cell;
Layout_Part := PBblas.Types.Layout_Part;
//Func : handle to the function we want to minimize it, its output should be the error cost and the error gradient
EXPORT Optimization (UNSIGNED4 prows=0, UNSIGNED4 pcols=0, UNSIGNED4 Maxrows=0, UNSIGNED4 Maxcols=0) := MODULE
//Implementation of Limited_Memory BFGS algorithm
//The implementation is done based on "Numerical Optimization Authors: Nocedal, Jorge, Wright, Stephen"
//corrections : number of corrections to store in memory
//MaxIter : Maximum number of iterations allowed
//This function returns the approximate inverse Hessian, multiplied by the gradient multiplied by -1
//g : the gradient values
// s: old steps values ("s" in the book)
//y: old dir values ("y" in the book)
//Hdiag value to initialize Hessian0 as Hdiag*I

EXPORT lbfgs (DATASET(Mat.Types.Element) g, DATASET(Mat.Types.Element) s, DATASET(Mat.Types.Element) d, DATASET(Mat.Types.Element) Hdiag) := FUNCTION
//P (number of parameters) and K (number of corrections)
dstats := Mat.Has(s).Stats;
P := dstats.XMax;
K := dstats.YMax;

//drive map
sizeRec := RECORD
  PBblas.Types.dimension_t m_rows;
  PBblas.Types.dimension_t m_cols;
  PBblas.Types.dimension_t f_b_rows;
  PBblas.Types.dimension_t f_b_cols;
END;
havemaxrow := maxrows > 0;
havemaxcol := maxcols > 0;
havemaxrowcol := havemaxrow and havemaxcol;
derivemap := IF(havemaxrowcol, PBblas.AutoBVMap(P, K,prows,pcols,maxrows, maxcols),
                   IF(havemaxrow, PBblas.AutoBVMap(P, K,prows,pcols,maxrows),
                      IF(havemaxcol, PBblas.AutoBVMap(P, K,prows,pcols,,maxcols),
                      PBblas.AutoBVMap(P, K,prows,pcols))));
sizeTable := DATASET([{derivemap.matrix_rows,derivemap.matrix_cols,derivemap.part_rows(1),derivemap.part_cols(1)}], sizeRec);
//maps used
MainMap := PBblas.Matrix_Map (P,K,sizeTable[1].f_b_rows,sizeTable[1].f_b_cols);
ColMap := PBblas.Matrix_Map (1,K,1,sizeTable[1].f_b_cols);
RowMap := PBblas.Matrix_Map (P,1,sizeTable[1].f_b_rows, 1);
OnevalueMap := PBblas.Matrix_Map (1,1,1, 1);
ddist := DMAT.Converted.FromElement(d,MainMap);
sdist := DMAT.Converted.FromElement(s,MainMap);
//calculate rho values
stepdir := PBblas.HadamardProduct(MainMap, ddist, sdist);
//calculate column sums which are rho values
Ones_VecMap := PBblas.Matrix_Map(1, P, 1, sizeTable[1].f_b_rows);
//New Vector Generator
Layout_Cell gen(UNSIGNED4 c, UNSIGNED4 NumRows) := TRANSFORM
  SELF.x := ((c-1) % NumRows) + 1;
  SELF.y := ((c-1) DIV NumRows) + 1;
  SELF.v := 1;
END;
//Create Ones Vector for the calculations in the step fucntion
Ones_Vec := DATASET(P, gen(COUNTER, 1));
Ones_Vecdist := DMAT.Converted.FromCells(Ones_VecMap, Ones_Vec);
stepdir_sumcol := PBblas.PB_dgemm(FALSE, FALSE, 1.0, Ones_VecMap, Ones_Vecdist, MainMap, stepdir, ColMap);
rho := ML.DMat.Converted.FromPart2Elm (stepdir_sumcol); //rho is in Mat.element format so we can retrive the ith rho values as rho(y=i)[1].value
// // Algorithm 9.1 (L-BFGS two-loop recursion)
//first loop : q calculation
q0 := g;
q0dist := DMAT.Converted.FromElement(q0,RowMap);
q0distno := PBblas.MU.TO(q0dist,0);
loop1 (DATASET(PBblas.Types.MUElement) inputin, INTEGER coun) := FUNCTION
  q := PBblas.MU.FROM(inputin,0);
  i := K-coun+1;//assumption is that in the steps and dir matrices the highest index the recent the vector is
  si := PROJECT(s(y=i),TRANSFORM(Mat.Types.Element,SELF.y :=1;SELF:=LEFT));//retrive the ith column and the y value should be 1 (not sure if it is important or not)???
  sidist := DMAT.Converted.FromElement(si,RowMap);//any way to extract sidist directly from sdist?
  //ai=rhoi*siT*q
  ai := PBblas.PB_dgemm(TRUE, FALSE, rho(y=i)[1].value, RowMap, sidist, RowMap, q, OnevalueMap);
  aiM := ML.DMat.Converted.FromPart2Elm (ai);//any easier way to retrive ai value?
  //q=q-ai*yi
  yi := PROJECT(d(y=i),TRANSFORM(Mat.Types.Element,SELF.y :=1;SELF:=LEFT));//retrive the ith column and the y value should be 1 (not sure if it is important or not)???
  yidist := DMAT.Converted.FromElement(yi,RowMap);//any way to extract sidist directly from sdist?
  qout := PBblas.PB_daxpy(-1*aiM[1].value, yidist, q);
  qoutno := PBblas.MU.TO(qout,0);
  RETURN qoutno+inputin(no>0)+PBblas.MU.TO(ai,i);
END;
R1 := LOOP(q0distno, COUNTER <= 2, loop1(ROWS(LEFT),COUNTER));
finalq := PBblas.MU.from(R1(no=0),0);
Aivalues := R1(no>0);
//by now tested by Matlab
//r=Hdiag*q
r0 := PBblas.PB_dscal(Hdiag[1].value, finalq);
loop2 (DATASET(Layout_Part) r, INTEGER coun) := FUNCTION
  i := coun;
  yi := PROJECT(d(y=i),TRANSFORM(Mat.Types.Element,SELF.y :=1;SELF:=LEFT));
  yidist := DMAT.Converted.FromElement(yi,RowMap);
  bi := PBblas.PB_dgemm(TRUE, FALSE, rho(y=i)[1].value, RowMap, yidist, RowMap, r, OnevalueMap);
  biM := ML.DMat.Converted.FromPart2Elm (bi);
  ai := PBblas.MU.From(Aivalues,i);
  aiM := ML.DMat.Converted.FromPart2Elm (ai);
  si := PROJECT(s(y=i),TRANSFORM(Mat.Types.Element,SELF.y :=1;SELF:=LEFT));
  sidist := DMAT.Converted.FromElement(si,RowMap);
  rout := PBblas.PB_daxpy(aiM[1].value-biM[1].value, sidist, r);
RETURN rout;
END;
R2 := LOOP(r0, COUNTER <= 2, loop2(ROWS(LEFT),COUNTER));
R2_ := PBblas.PB_dscal(-1, R2) ;
RETURN R2_;
END;// END lbfgs
END;// END Optimization