//#option ('divideByZero', 'nan');
IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat;
IMPORT * FROM ML.Types;
IMPORT PBblas;
Layout_Cell := PBblas.Types.Layout_Cell;
Layout_Part := PBblas.Types.Layout_Part;
     OutputRecord := RECORD
      REAL8 t;
      REAL8 f_new;
      DATASET(Mat.Types.Element) g_new;
      INTEGER8 WolfeFunEval;
    END;
      

//Func : handle to the function we want to minimize it, its output should be the error cost and the error gradient
EXPORT Optimization2 (UNSIGNED4 prows=0, UNSIGNED4 pcols=0, UNSIGNED4 Maxrows=0, UNSIGNED4 Maxcols=0) := MODULE

  //polyinterp when the boundry values are provided
  EXPORT  polyinterp_both (REAL8 t_1, REAL8 f_1, REAL8 gtd_1, REAL8 t_2, REAL8 f_2, REAL8 gtd_2, REAL8 xminBound, REAL8 xmaxBound) := FUNCTION

      points_t1 :=[t_1,t_2,f_1,f_2,gtd_1,gtd_2];
      points_t2 := [t_2,t_1,f_2,f_1,gtd_2,gtd_1];
      orderedp := IF (t_1<t_2,points_t1 , points_t2);    
      tmin := orderedp [1];
      tmax := orderedp [2];
      fmin := orderedp [3];
      fmax := orderedp [4];
      gtdmin := orderedp [5];
      gtdmax := orderedp [6];
      // A= [t_1^3 t_1^2 t_1 1
      //    t_2^3 t_2^2 t_2 1
      //    3*t_1^2 2*t_1 t_1 0
      //    3*t_2^2 2*t_2 t_2 0]
      //b = [f_1 f_2 dtg_1 gtd_2]'
      Aset := [POWER(t_1,3),POWER(t_2,3),3*POWER(t_1,2),3*POWER(t_2,2),
      POWER(t_1,2),POWER(t_2,2), 2*t_1,2*t_2,
      POWER(t_1,3),POWER(t_2,1), 1, 1,
      1, 1, 0, 0]; // A 4*4 Matrix      
      Bset := [f_1, f_2, gtd_1, gtd_2]; // A 4*1 Matrix
      // Find interpolating polynomial
      //params = A\b;
      params_partset := PBblas.BLAS.solvelinear (Aset, Bset, 4,1,4,4);
      //params_partset := [1,2,3,4];
      params1 := params_partset[1];
      params2 := params_partset[2];
      params3 := params_partset[3];
      params4 := params_partset[4];
      dParams1 := 3*params_partset[1];
      dparams2 := 2*params_partset[2];
      dparams3 := params_partset[3];
      Rvalues := roots (dParams1, dparams2, dparams3);
      // Compute Critical Points
      INANYINF := FALSE; //????for now
      cp1 := xminBound;
      cp2 := xmaxBound;
      cp3 := t_1;
      cp4 := t_2;
      cp5 := Rvalues (id=2)[1].value;
      cp6 := Rvalues (id=3)[1].value;
      ISrootsreal := (BOOLEAN) Rvalues (id=1)[1].value;
      cp_real := DATASET([
      {1,1,cp1},
      {2,1,cp2},
      {3,1,cp3},
      {4,1,cp4},
      {5,1,cp5},
      {6,1,cp6}],
      Types.NumericField);
      cp_imag := DATASET([
      {1,1,cp1},
      {2,1,cp2},
      {3,1,cp3},
      {4,1,cp4}],
      Types.NumericField);
      cp := IF (ISrootsreal, cp_real, cp_imag);
      itr := IF (ISrootsreal, 6, 4);
      
      /*
for xCP = cp
    if imag(xCP)==0 && xCP >= xminBound && xCP <= xmaxBound
        fCP = polyval(params,xCP);
        if imag(fCP)==0 && fCP < fmin
            minPos = real(xCP);
            fmin = real(fCP);
        end
    end
end*/
      // I was using LOOP before to implement this part and it make some LOOP related errors in the wolfesearch function later, so I changed the code in a way that it does not use LOOP
      out := FUNCTION
        fmin1 := 1000000;
        minpos1 := (xminBound+xmaxBound)/2;
        xCP1 := cp1;
        cond1_1 := xCP1 >= xminBound AND xCP1 <= xmaxBound; //???
        // fCP = polyval(params,xCP);
        fCP1 := params1*POWER(xCP1,3)+params2*POWER(xCP1,2)+params3*xCP1+params4;
        //if imag(fCP)==0 && fCP < fmin
        cond1_2 := ISrootsreal; // If the roots are imaginary so is FCP
        minpos2 := IF (cond1_1,IF (cond1_2, xCP1, minpos1),minpos1);
        fmin2 := IF (cond1_1,IF (cond1_2, fCP1, fmin1),fmin1);
        
        xCP2 := cp2;
        cond2_1 := xCP2 >= xminBound AND xCP2 <= xmaxBound; //???
        // fCP = polyval(params,xCP);
        fCP2:= params1*POWER(xCP2,3)+params2*POWER(xCP2,2)+params3*xCP2+params4;
        //if imag(fCP)==0 && fCP < fmin
        cond2_2 := (fCP2<fmin2) AND ISrootsreal; // If the roots are imaginary so is FCP
        minpos3 := IF (cond2_1,IF (cond2_2, xCP2, minpos2),minpos2);
        fmin3 := IF (cond2_1,IF (cond2_2, fCP2, fmin2),fmin2);
        
        
        xCP3 := cp3;
        cond3_1 := xCP3 >= xminBound AND xCP3 <= xmaxBound; //???
        // fCP = polyval(params,xCP);
        fCP3:= params1*POWER(xCP3,3)+params2*POWER(xCP3,2)+params3*xCP3+params4;
        //if imag(fCP)==0 && fCP < fmin
        cond3_2 := (fCP3<fmin3) AND ISrootsreal; // If the roots are imaginary so is FCP
        minpos4 := IF (cond3_1,IF (cond3_2, xCP3, minpos3),minpos3);
        fmin4 := IF (cond3_1,IF (cond3_2, fCP3, fmin3),fmin3);
        
        
        xCP4 := cp4;
        cond4_1 := xCP4 >= xminBound AND xCP4 <= xmaxBound; //???
        // fCP = polyval(params,xCP);
        fCP4:= params1*POWER(xCP4,3)+params2*POWER(xCP4,2)+params3*xCP4+params4;
        //if imag(fCP)==0 && fCP < fmin
        cond4_2 := (fCP4<fmin4) AND ISrootsreal; // If the roots are imaginary so is FCP
        minpos5 := IF (cond4_1,IF (cond4_2, xCP4, minpos4),minpos4);
        fmin5 := IF (cond4_1,IF (cond4_2, fCP4, fmin4),fmin4);
        
        
        xCP5 := cp5;
        cond5_1 := xCP5 >= xminBound AND xCP5 <= xmaxBound; //???
        // fCP = polyval(params,xCP);
        fCP5:= params1*POWER(xCP5,3)+params2*POWER(xCP5,2)+params3*xCP5+params4;
        //if imag(fCP)==0 && fCP < fmin
        cond5_2 := (fCP5<fmin5) AND ISrootsreal; // If the roots are imaginary so is FCP
        minpos6 := IF (cond5_1,IF (cond5_2, xCP5, minpos5),minpos5);
        fmin6 := IF (cond5_1,IF (cond5_2, fCP5, fmin5),fmin5);
        
        xCP6 := cp6;
        cond6_1 := xCP6 >= xminBound AND xCP6 <= xmaxBound; //???
        // fCP = polyval(params,xCP);
        fCP6:= params1*POWER(xCP6,3)+params2*POWER(xCP6,2)+params3*xCP6+params4;
        //if imag(fCP)==0 && fCP < fmin
        cond6_2 := (fCP6<fmin6) AND ISrootsreal; // If the roots are imaginary so is FCP
        minpos7 := IF (cond6_1,IF (cond6_2, xCP6, minpos6),minpos6);
        fmin7 := IF (cond6_1,IF (cond6_2, fCP6, fmin6),fmin6);
        
        RETURN IF (ISrootsreal, minpos7, minpos5);
        
      END;

    polResult := out;
    RETURN polResult;
  END;//end polyinterp_both
  //polyinterp when no boundry values are provided
  EXPORT  polyinterp_noboundry (REAL8 t_1, REAL8 f_1, REAL8 gtd_1, REAL8 t_2, REAL8 f_2, REAL8 gtd_2) := FUNCTION
      tmin := IF (t_1<t_2,t_1 , t_2);
      tmax := IF (t_1<t_2,t_2 , t_1);
      fmin := IF (t_1<t_2,f_1 , f_2);
      fmax := IF (t_1<t_2,f_2 , f_1);
      gtdmin := IF (t_1<t_2,gtd_1 , gtd_2);
      gtdmax := IF (t_1<t_2,gtd_2 , gtd_1);
      // d1 = points(minPos,3) + points(notMinPos,3) - 3*(points(minPos,2)-points(notMinPos,2))/(points(minPos,1)-points(notMinPos,1));
      d1 := gtdmin + gtdmax - (3*(fmin-fmax)/(tmin-tmax));
      //d2 = sqrt(d1^2 - points(minPos,3)*points(notMinPos,3));
      d2 := SQRT ((d1*d1)-(gtdmin*gtdmax));
      d2real := TRUE; //check it ???
      //t = points(notMinPos,1) - (points(notMinPos,1) - points(minPos,1))*((points(notMinPos,3) + d2 - d1)/(points(notMinPos,3) - points(minPos,3) + 2*d2));
      temp := tmax - ((tmax-tmin)*((gtdmax+d2-d1)/(gtdmax-gtdmin+(2*d2))));
      //min(max(t,points(minPos,1)),points(notMinPos,1));
      minpos1 := MIN([MAX([temp,tmin]),tmax]);
      minpos2 := (t_1+t_2)/2;
      pol2Result := IF (d2real,minpos1,minpos2);
    RETURN pol2Result;
    //RETURN d1;
  END;//end polyinterp_noboundry
  //polyinterp when no boundry values is provided  and gtd_2 is imaginary (used in Armijo Back Tracking)
  EXPORT  polyinterp_img (REAL8 t_1, REAL8 f_1, REAL8 gtd_1, REAL8 t_2, REAL8 f_2, REAL8 gtd_2) := FUNCTION
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
    tmin := orderedp (id=1)[1].value;
    tmax := orderedp (id=2)[1].value;
    fmin := orderedp (id=3)[1].value;
    fmax := orderedp (id=4)[1].value;
    gtdmin := orderedp (id=5)[1].value;
    gtdmax := orderedp (id=6)[1].value;
    xminBound := tmin;
    xmaxBound := tmax;
    // A= [t_1^3 t_1^2 t_1 1
    //    t_2^3 t_2^2 t_2 1
    //    3*t_1^2 2*t_1 t_1 0]
    //b = [f_1 f_2 dtg_1]'
    A := DATASET([
    {1,1,POWER(t_1,2)},
    {1,2,POWER(t_1,3)},
    {1,3,1},
    {2,1,POWER(t_2,2)},
    {2,2,POWER(t_2,1)},
    {2,3,1},
    {3,1,2*t_1},
    {3,2,1},
    {3,3,0}],
    Types.NumericField);
    Aset := [POWER(t_1,2), POWER(t_2,2), 2*t_1,
    POWER(t_1,3), POWER(t_2,1), 1,
    1, 1, 0];
    b := DATASET([
    {1,1,f_1},
    {2,1,f_2},
    {3,1,gtd_1}],
    Types.NumericField);
    Bset := [f_1, f_2, gtd_1];
    // Find interpolating polynomial
    A_map := PBblas.Matrix_Map(3, 3, 3, 3);
    b_map := PBblas.Matrix_Map(3, 1, 3, 1);
    A_part := ML.DMat.Converted.FromNumericFieldDS (A, A_map);
    b_part := ML.DMat.Converted.FromNumericFieldDS (b, b_map);
    //params = A\b;
    params_partset := PBblas.BLAS.solvelinear (Aset, Bset, 3, 1, 3 , 3);
    params1 := params_partset[1];
    params2 := params_partset[2];
    params3 := params_partset[3];
    dParams1 := 2*params_partset[1];
    dparams2 := params_partset[2];
    
    
    
    Rvalues := -1*dparams2/dParams1;

    // Compute Critical Points
    INANYINF := FALSE; //????for now
    cp1 := xminBound;
    cp2 := xmaxBound;
    cp3 := t_1;
    cp4 := t_2;
    cp5 := Rvalues;
    ISrootsreal := TRUE;
    cp_real := DATASET([
    {1,1,cp1},
    {2,1,cp2},
    {3,1,cp3},
    {4,1,cp4},
    {5,1,cp5}],
    Types.NumericField);
    cp_imag := DATASET([
    {1,1,cp1},
    {2,1,cp2},
    {3,1,cp3},
    {4,1,cp4}],
    Types.NumericField);
    cp := IF (ISrootsreal, cp_real, cp_imag);
    itr := IF (ISrootsreal, 5, 4);
    // Test Critical Points
    topa :=  DATASET([{1,1,(xminBound+xmaxBound)/2},{2,1,1000000}], Types.NumericField);
    Resultstep (DATASET(Types.NumericField) x, UNSIGNED coun) := FUNCTION
      minPos := x(id=1)[1].value;
      f_min := x(id=2)[1].value;
      // if imag(xCP)==0 && xCP >= xminBound && xCP <= xmaxBound
      xCP := cp(id=coun)[1].value;
      cond := xCP >= xminBound AND xCP <= xmaxBound; //???
      // fCP = polyval(params,xCP);
      fCP := params1*POWER(xCP,2)+params2*xCP+params3;
      //if imag(fCP)==0 && fCP < fmin
      cond2 := (coun=1 OR fCP<f_min) AND ISrootsreal; // If the roots are imaginary so is fCP
      rr := IF (cond,IF (cond2, xCP, minPos),minPos);
      ff := IF (cond,IF (cond2, fCP, f_min),f_min);
      RETURN DATASET([{1,1,rr},{2,1,ff}], Types.NumericField);
    END;
    finalresult := LOOP(topa, COUNTER <= itr, Resultstep(ROWS(LEFT),COUNTER));
    //RETURN IF(t_1=0, 10, 100);
    // RETURN DATASET([
    // {1,1,dParams1},
    // {2,1,dParams2},
    // {3,1,dParams3}],
    // Types.NumericField);
    RETURN finalresult;
    END;//END poly1
    polResult := poly1(id=1)[1].value;
    RETURN polResult;  
  END;//end polyinterp_img
  EXPORT Limited_Memory_BFGS (UNSIGNED P, UNSIGNED K) := MODULE
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
    SHARED sizeTable := DATASET([{derivemap.matrix_rows,derivemap.matrix_cols,derivemap.part_rows(1),derivemap.part_cols(1)}], sizeRec);
    SHARED ColMap := PBblas.Matrix_Map (1,K,1,sizeTable[1].f_b_cols);
    SHARED RowMap := PBblas.Matrix_Map (P,1,sizeTable[1].f_b_rows, 1);
    SHARED OnevalueMap := PBblas.Matrix_Map (1,1,1, 1);

    //this function is used for the very first iteration in lbfgs algorithm
    EXPORT Steepest_Descent (DATASET(Mat.Types.Element) g) := FUNCTION
      gdist := DMAT.Converted.FromElement(g, RowMap);
      gdist_ := PBblas.PB_dscal(-1, gdist);
      RETURN gdist_;
    END;
    //Implementation of Limited_Memory BFGS algorithm
    //The implementation is done based on "Numerical Optimization Authors: Nocedal, Jorge, Wright, Stephen"
    //corrections : number of corrections to store in memory
    //MaxIter : Maximum number of iterations allowed
    //This function returns the approximate inverse Hessian, multiplied by the gradient multiplied by -1
    //g : the gradient values
    // s: old steps values ("s" in the book)
    //d: old dir values ("y" in the book)
    //Hdiag value to initialize Hessian0 as Hdiag*I
    EXPORT lbfgs (DATASET(Mat.Types.Element) g, DATASET(Mat.Types.Element) s, DATASET(Mat.Types.Element) d, REAL8 Hdiag) := FUNCTION
      //Functions needed in calculations
      PBblas.Types.value_t Reciprocal(PBblas.Types.value_t v, 
                                      PBblas.Types.dimension_t r, 
                                      PBblas.Types.dimension_t c) := IF (v!=0, 1/v, 0); //based on the current implementation of the lbfgs fucntion, I want the values to be zero where Reciprocal is used and division by zero occurs. Even though this is the default behaviour of ECL, I have changed this default behaviour to produce NAN when division by zero occurs in the entire MinFunc so here I explecitly have to say that if division by zero occures it should equal to 0.
      //maps used
      MainMap := PBblas.Matrix_Map (P,K,sizeTable[1].f_b_rows,sizeTable[1].f_b_cols);
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
      stepdir_sumcol_Reciprocal := PBblas.Apply2Elements(ColMap, stepdir_sumcol, Reciprocal);
      rho := ML.DMat.Converted.FromPart2Elm (stepdir_sumcol_Reciprocal); //rho is in Mat.element format so we can retrive the ith rho values as rho(y=i)[1].value
      // // Algorithm 9.1 (L-BFGS two-loop recursion)
      //first loop : q calculation
      q0 := g;
      q0dist := DMAT.Converted.FromElement(q0,RowMap);
      q0distno := PBblas.MU.TO(q0dist,0);
      loop1 (DATASET(PBblas.Types.MUElement) inputin, INTEGER coun) := FUNCTION
        q := PBblas.MU.FROM(inputin,0);
        i := K-coun+1;//assumption is that in the steps and dir matrices the highest the index the more recent the vector is
        si := PROJECT(s(y=i),TRANSFORM(Mat.Types.Element,SELF.y :=1;SELF:=LEFT));//retrive the ith column (the y value should be 1)
        sidist := DMAT.Converted.FromElement(si,RowMap);//any way to extract sidist directly from sdist?
        //ai=rhoi*siT*q
        ai := PBblas.PB_dgemm(TRUE, FALSE, rho(y=i)[1].value, RowMap, sidist, RowMap, q, OnevalueMap);
        aiValue := ai[1].mat_part[1];
        //q=q-ai*yi
        yi := PROJECT(d(y=i),TRANSFORM(Mat.Types.Element,SELF.y :=1;SELF:=LEFT));//retrive the ith column and the y value should be 1
        yidist := DMAT.Converted.FromElement(yi,RowMap);//any way to extract yidist directly from ydist?
        qout := PBblas.PB_daxpy(-1*aiValue, yidist, q);
        qoutno := PBblas.MU.TO(qout,0);
        RETURN qoutno+inputin(no>0)+PBblas.MU.TO(ai,i);
      END;
      R1 := LOOP(q0distno, COUNTER <= K, loop1(ROWS(LEFT),COUNTER));
      finalq := PBblas.MU.from(R1(no=0),0);
      Aivalues := R1(no>0);
      //r=Hdiag*q
      r0 := PBblas.PB_dscal(Hdiag, finalq);
      loop2 (DATASET(Layout_Part) r, INTEGER coun) := FUNCTION
        i := coun;
        yi := PROJECT(d(y=i),TRANSFORM(Mat.Types.Element,SELF.y :=1;SELF:=LEFT));
        yidist := DMAT.Converted.FromElement(yi,RowMap);
        bi := PBblas.PB_dgemm(TRUE, FALSE, rho(y=i)[1].value, RowMap, yidist, RowMap, r, OnevalueMap);
        bivalue := bi[1].mat_part[1];
        ai := PBblas.MU.From(Aivalues,i);
        aivalue := ai[1].mat_part[1];
        si := PROJECT(s(y=i),TRANSFORM(Mat.Types.Element,SELF.y :=1;SELF:=LEFT));
        sidist := DMAT.Converted.FromElement(si,RowMap);
        rout := PBblas.PB_daxpy(aivalue-bivalue, sidist, r);
      RETURN rout;
      END;
      R2 := LOOP(r0, COUNTER <= K, loop2(ROWS(LEFT),COUNTER));
      //R2_ := PBblas.PB_dscal(-1, R2) ;
      RETURN R2;
    END;// END lbfgs
    //This function adds the most recent vectors differences (vec_next-vec_pre) to the matrix (old_mat) and removes the oldest vector from the matrix
    //old_mat can be a P*K matrix that each column can be a parameter vector or a gradient vector in lbfgs algorithm i.e. Steps and Dirs matrices respectively
    //in each step of the lbfgs algorithm Steps and Dirs matrices should be updated. The most recent vector should be added and the oldest vector should be removed
    //The most recent vector should apear as the last column in the matrix and the oldest vector is actually the first column that should be removed
    //MATLAB code is as below:
    // old_dirs = [old_dirs(:,2:corrections) s];
    // old_stps = [old_stps(:,2:corrections) y];
    //corr in the namae of the fucntion stands for "corrections"
    EXPORT lbfgsUpdate_corr (DATASET(Mat.Types.Element) vec_pre, DATASET(Mat.Types.Element) vec_next, DATASET(Mat.Types.Element) old_mat) := FUNCTION
      //vec = vec_next-vec_pre
      vec_predist := DMAT.Converted.FromElement(vec_pre, RowMap);
      vec_nextdist := DMAT.Converted.FromElement(vec_next, RowMap);
      vecdist := PBblas.PB_daxpy(-1, vec_predist, vec_nextdist);
      vec := ML.DMat.Converted.FromPart2Elm (vecdist);
      //remove the first column from the matrix
      old_mat_firstColrmv := old_mat (y>1);
      //decrease column values by one (shift the columsn to the left)
      Mat.Types.Element colMinus (Mat.Types.Element l) := TRANSFORM
        SELF.y := l.y-1;
        SELF := l;
      END;
      old_mat_firstColrmv_:=PROJECT(old_mat_firstColrmv,colMinus(LEFT));
      //add vec to the last column
      vec_K := PROJECT (vec, TRANSFORM(Mat.Types.Element, SELF.y :=K; SELF:=LEFT));
      New_mat := old_mat_firstColrmv_+vec_K;
      RETURN New_mat;
    END;//END lbfgsUpdate_corr
    //Calculate the next Hessian diagonal value for the next iteration of the lbfgs algorithm based on the current parameter vector (s) and gradient vector (y)
    //Formula 9.6 in the book :  Hdiag = y'*s/(y'*y);
    EXPORT lbfgsUpdate_Hdiag (DATASET(Mat.Types.Element) s1, DATASET(Mat.Types.Element) s2, DATASET(Mat.Types.Element) y1, DATASET(Mat.Types.Element) y2, REAL8 hdiaginput) := FUNCTION
      vec_predist := DMAT.Converted.FromElement(s1, RowMap);
      vec_nextdist := DMAT.Converted.FromElement(s2, RowMap);
      sdist := PBblas.PB_daxpy(-1, vec_predist, vec_nextdist);
      vec_predist2 := DMAT.Converted.FromElement(y1, RowMap);
      vec_nextdist2 := DMAT.Converted.FromElement(y2, RowMap);
      ydist := PBblas.PB_daxpy(-1, vec_predist2, vec_nextdist2);
      first_term := PBblas.PB_dgemm (TRUE, FALSE, 1.0, RowMap, ydist, RowMap, sdist, OnevalueMap );
      first_term_M := ML.DMat.Converted.FromPart2Elm (first_term);
      Second_Term := PBblas.PB_dgemm (TRUE, FALSE, 1.0, RowMap, ydist, RowMap, ydist, OnevalueMap );
      Second_Term_M := ML.DMat.Converted.FromPart2Elm (Second_Term);
      HD := IF (First_Term_M[1].value>0.0000000001, First_Term_M[1].value/Second_Term_M[1].value, hdiaginput);
      RETURN HD;
    END; // END lbfgsUpdate_Hdiag
    EXPORT lbfgsUpdate_Dirs (DATASET(Mat.Types.Element) s1, DATASET(Mat.Types.Element) s2, DATASET(Mat.Types.Element) y1, DATASET(Mat.Types.Element) y2, DATASET(Mat.Types.Element) old_dir_mat) := FUNCTION
      vec_predist := DMAT.Converted.FromElement(s1, RowMap);
      vec_nextdist := DMAT.Converted.FromElement(s2, RowMap);
      sdist := PBblas.PB_daxpy(-1, vec_predist, vec_nextdist);
      vec_predist2 := DMAT.Converted.FromElement(y1, RowMap);
      vec_nextdist2 := DMAT.Converted.FromElement(y2, RowMap);
      ydist := PBblas.PB_daxpy(-1, vec_predist2, vec_nextdist2);
      ys_term := PBblas.PB_dgemm (TRUE, FALSE, 1.0, RowMap, ydist, RowMap, sdist, OnevalueMap );
      ys_term_M := ML.DMat.Converted.FromPart2Elm (ys_term);
      RETURN IF(ys_term_M[1].value>0.0000000001,lbfgsUpdate_corr (s1, s2, old_dir_mat) ,old_dir_mat );
    END; // END lbfgsUpdate_Dirs
    EXPORT lbfgsUpdate_Stps (DATASET(Mat.Types.Element) s1, DATASET(Mat.Types.Element) s2, DATASET(Mat.Types.Element) y1, DATASET(Mat.Types.Element) y2, DATASET(Mat.Types.Element) old_stp_mat) := FUNCTION
      vec_predist := DMAT.Converted.FromElement(s1, RowMap);
      vec_nextdist := DMAT.Converted.FromElement(s2, RowMap);
      sdist := PBblas.PB_daxpy(-1, vec_predist, vec_nextdist);
      vec_predist2 := DMAT.Converted.FromElement(y1, RowMap);
      vec_nextdist2 := DMAT.Converted.FromElement(y2, RowMap);
      ydist := PBblas.PB_daxpy(-1, vec_predist2, vec_nextdist2);
      ys_term := PBblas.PB_dgemm (TRUE, FALSE, 1.0, RowMap, ydist, RowMap, sdist, OnevalueMap );
      ys_term_M := ML.DMat.Converted.FromPart2Elm (ys_term);
      RETURN IF(ys_term_M[1].value>0.0000000001,lbfgsUpdate_corr (y1, y2, old_stp_mat) ,old_stp_mat );
    END; // END lbfgsUpdate_Stps
  END;//END Limited_Memory_BFGS
  
  
  
    
  //
  //WolfeLineSearch
  // Bracketing Line Search to Satisfy Wolfe Conditions
  //Source "Numerical Optimization Book" and Matlab implementaion of minFunc :
  // M. Schmidt. minFunc: unconstrained differentiable multivariate optimization in Matlab. http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html, 2005.
  //  x:  Starting Location (numeric field format)
  //  t: Initial step size (its a number, usually 1)
  //  d:  descent direction  (Pk in formula 3.1 in the book) (numeric field format)
  //  g: gradient at starting location (numeric field format)
  //  gtd:  directional derivative at starting location :  gtd = g'*d (its a number), TRANS(Deltafk)*Pk in formula 3.6a
  //  c1: sufficient decrease parameter (c1 in formula 3.6a, its a number)
  //  c2: curvature parameter (c2 in formula 3.6b, its a number)
  //  maxLS: maximum number of iterations in WOLFE algorithm
  //  tolX: minimum allowable step length
  //  CostFunc: objective function(it returns the gradient and cost value in numeric field format, cost value has the highest
  //  id in the returned numeric field structure
  //  TrainData and TrainLabel: train and label data for the objective fucntion (numeric field format)
  //  The rest are PBblas parameters
  //  Define a general FunVAL and add it in wolfebracketing and WolfeZoom ??????
  //  WolfeOut is what the macro returns, it include t,f_new,g_new,funEvals (t the calculated step size
  //  f_new the cost value in the new point, g_new is the gradient value in the new point and funevals is the number of
  //EXPORT WolfeLineSearch(DATASET(Types.NumericField)x, REAL8 t, DATASET(Types.NumericField)d, REAL8 f, DATASET(Types.NumericField) g, REAL8 gtd, REAL8 c1=0.0001, REAL8 c2=0.9, INTEGER maxLS=25, REAL8 tolX=0.000000001,DATASET(Types.NumericField) CostFunc_params, DATASET(Types.NumericField) TrainData , DATASET(Types.NumericField) TrainLabel, DATASET(Types.NumericField) CostFunc (DATASET(Types.NumericField) x, DATASET(Types.NumericField) CostFunc_params, DATASET(Types.NumericField) TrainData , DATASET(Types.NumericField) TrainLabel), prows=0, pcols=0, Maxrows=0, Maxcols=0):=FUNCTION orig

  
   EXPORT WolfeLineSearch2(INTEGER cccc, DATASET(Types.NumericField)x, REAL8 t, DATASET(Types.NumericField)d, REAL8 f, DATASET(Types.NumericField) g, REAL8 gtd, REAL8 c1=0.0001, REAL8 c2=0.9, INTEGER maxLS=25, REAL8 tolX=0.000000001,DATASET(Types.NumericField) CostFunc_params, DATASET(Types.NumericField) TrainData , DATASET(Types.NumericField) TrainLabel, DATASET(Types.NumericField) CostFunc (DATASET(Types.NumericField) x, DATASET(Types.NumericField) CostFunc_params, DATASET(Types.NumericField) TrainData , DATASET(Types.NumericField) TrainLabel), prows=0, pcols=0, Maxrows=0, Maxcols=0):=FUNCTION
    //initial parameters
    P_num := Max (x, id); //the length of the parameters vector (number of parameters)
    emptyE := DATASET([], Mat.Types.Element);
    LSiter := 0;

    calculate_gtdNew (DATASET(Types.NumericField) g_in, DATASET(Types.NumericField) d_in) := FUNCTION
      Types.NumericField Mu(g_in le,d_in ri) := TRANSFORM
        SELF.id := le.id;
        SELF.number := ri.number;
        SELF.value := le.value * ri.value;
      END;
      element_mul := JOIN(g_in,d_in,LEFT.id=RIGHT.id,Mu(LEFT,RIGHT));
      r := RECORD
        t_RecordID id := 1 ;
        t_FieldNumber number := 1;
        t_FieldReal value := SUM(GROUP,element_mul.value);
      END;
      SumMulElm := TABLE(element_mul,r);
      RETURN SumMulElm[1].value;
    END;
    calculate_xNew (DATASET(Types.NumericField) d_in, REAL8 t_in) := FUNCTION
      //x_new = x+t*d
      Types.NumericField xnew_tran (x l, d_in r):= TRANSFORM
        SELF.value := l.value+(t_in*r.value);
        SELF := l;
      END;
      Result := JOIN(x,d_in,LEFT.id=RIGHT.id AND LEFT.number=RIGHT.number,xnew_tran(LEFT,RIGHT));
      RETURN Result;
    END;
    ExtractGrad (DATASET(Types.NumericField) inp) := FUNCTION
      RETURN inp (id <= P_num);
    END;
    ExtractCost (DATASET(Types.NumericField) inp) := FUNCTION
      RETURN inp (id = (P_num+1))[1].value;
    END;
    IsNotLegal (DATASET (Mat.Types.Element) Mat) := FUNCTION //???to be defined
      RETURN FALSE;
    END;
    IsNotLegal_real (REAL8 v) := FUNCTION //???to be defined
      RETURN FALSE;
    END;
  

    
    //% Evaluate the Objective and Gradient at the Initial Step
    //x_new = x+t*d
    //x_new := ML.Types.FromMatrix (ML.Mat.Add(ML.Types.ToMatrix(x),ML.Mat.Scale(ML.Types.ToMatrix(d),t)));
    x_new := calculate_xNew (d,t);
    CostGrad_new := CostFunc (x_new ,CostFunc_params,TrainData, TrainLabel);
    g_new := ExtractGrad (CostGrad_new);
    f_new := ExtractCost (CostGrad_new);
    funEvals := 1;
    //gtd_new = g_new'*d;
    //gtd_new := ML.Mat.Mul (ML.Mat.Trans(ML.Types.ToMatrix(g_new)),ML.Types.ToMatrix(d));
    gtd_new := calculate_gtdNew (g_new, d);
    
    // Bracket an Interval containing a point satisfying the Wolfe Criteria
    t_prev := 0;
    f_prev := f;
    g_prev := g;
    gtd_prev := gtd;

    //Bracketing algorithm, either produces the final t value or a bracket that contains the final t value
    
    bracketing_record := RECORD
      REAL8 f_prev_;
      REAL8 f_new_;
      DATASET(Mat.Types.Element) g_prev_;
      DATASET(Mat.Types.Element) g_new_;
      REAL8 t_;
      REAL8 t_prev_;
      INTEGER8 funEvals_;
      REAL8 gtd_prev_;
      REAL8 gtd_new_;
      REAL8 bracket1_;
      REAL8 bracket2_;
      REAL8 bracket1_f_;
      REAL8 bracket2_f_;
      DATASET(Mat.Types.Element) bracket1_g_;
      DATASET(Mat.Types.Element) bracket2_g_;
      INTEGER8 c; //Counter
    END;
    
    ToPass_bracketing := DATASET ([{f_prev,f_new,ML.Types.ToMatrix(g_prev),ML.Types.ToMatrix(g_new),t,t_prev,funEvals,gtd_prev,gtd_new,-1,-1,-1,-1,emptyE,emptyE,0 }],bracketing_record);
    
    
    
    BracketingStep (DATASET (bracketing_record) inputp, INTEGER coun) := FUNCTION
      AreTheyLegal := IsNotLegal_real(inputp.f_new_) | IsNotLegal(inputp.g_new_);
      WolfeStep := FUNCTION
        fNew := inputp[1].f_new_;
        fPrev := inputp[1].f_prev_;
        gtdNew := inputp[1].gtd_new_;
        gtdPrev := inputp[1].gtd_prev_;
        tt := inputp[1].t_;
        tPrev := inputp[1].t_prev_;
        gNew := inputp[1].g_new_;
        gPrev := inputp[1].g_prev_;
        inputFunEval := inputp[1].funEvals_;
        BrackLSiter := coun-1;
        
        //If the strong wolfe conditions satisfies then retun the final t or the bracket, otherwise do the next iteration
        //f_new > f + c1*t*gtd || (LSiter > 1 && f_new >= f_prev)
        con1 := (fNew > f + c1 * tt* gtd)| ((BrackLSiter > 1) & (fNew >= fPrev)) ;
        //abs(gtd_new) <= -c2*gtd
        con2 := ABS(gtdNew) <= (-1*c2*gtd);
        // gtd_new >= 0
        con3 := gtdNew >= 0;
        bracketing_record bracketing_con1_3 (bracketing_record l) := TRANSFORM
        /* bracket = [t_prev t];
        bracketFval = [f_prev f_new];
        bracketGval = [g_prev g_new];*/
          SELF.bracket1_ := tPrev;
          SELF.bracket2_ := tt;
          SELF.bracket1_f_ := fPrev;
          SELF.bracket2_f_ := fNew;
          SELF.bracket1_g_ := gPrev;
          SELF.bracket2_g_ := gNew;
          SELF := l;
        END;
        bracketing_record bracketing_con2 (bracketing_record l) := TRANSFORM
        /*  bracket = t;
        bracketFval = f_new;
        bracketGval = g_new;
        done = 1;*/
          SELF.bracket1_ := tt;
          SELF.bracket2_ := -1;
          SELF.bracket1_f_ := fNew;
          SELF.bracket2_f_ := -1;
          SELF.bracket1_g_ := gNew;
          SELF := l;
        END;

        bracketing_record bracketing_Nocon (bracketing_record l) := TRANSFORM
          //calculate new t
          minstep := tt + 0.01* (tt-tPrev);
          maxstep := tt*10;
          newt := polyinterp_both (tPrev, fPrev,gtdPrev, tt, fNew, gtdNew, minstep, maxstep);
          //calculate fnew gnew gtdnew
          xNew := calculate_xNew (d, newt);
          CostGradNew := CostFunc (xNew ,CostFunc_params,TrainData, TrainLabel);
          gNewwolfe := ExtractGrad (CostGradNew);
          fNewWolfe := ExtractCost (CostGradNew);
          gtdNewWolfe := ML.Mat.Mul (ML.Mat.Trans(ML.Types.ToMatrix(gNewwolfe)),ML.Types.ToMatrix(d));
          SELF.t_prev_ := tt; //t_prev = t;
          SELF.f_prev_ := fNew;
          SELF.g_prev_ := gNew;
          SELF.gtd_prev_ := gtdNew;
          SELF.t_ := newt;
          SELF.f_new_ := fNewWolfe;
          SELF.g_new_ := ML.Types.ToMatrix(gNewwolfe);
          SELF.gtd_new_ := gtdNewWolfe[1].value;
          SELF.bracket1_ := -1;
          SELF.bracket2_ := -1;
          SELF.funEvals_ := inputFunEval+1;
          SELF.c := l.c+1;
          SELF := l;
        END;
        con1_3_output := PROJECT(inputp,bracketing_con1_3(LEFT));
        con2_output := PROJECT(inputp,bracketing_con2(LEFT));
        Nocon_output := PROJECT(inputp,bracketing_Nocon(LEFT));
        RETURN IF (con1, con1_3_output, IF (con2, con2_output, IF (con3, con1_3_output, Nocon_output)));
      END;//END WolfeStep
      
      
      
      WolfeStep2 := FUNCTION
        fNew := inputp[1].f_new_;
        fPrev := inputp[1].f_prev_;
        gtdNew := inputp[1].gtd_new_;
        gtdPrev := inputp[1].gtd_prev_;
        tt := inputp[1].t_;
        tPrev := inputp[1].t_prev_;
        gNew := inputp[1].g_new_;
        gPrev := inputp[1].g_prev_;
        inputFunEval := inputp[1].funEvals_;
        BrackLSiter := coun-1;
        
        //If the strong wolfe conditions satisfies then retun the final t or the bracket, otherwise do the next iteration
        //f_new > f + c1*t*gtd || (LSiter > 1 && f_new >= f_prev)
        con1 := (fNew > f + c1 * tt* gtd)| ((BrackLSiter > 1) & (fNew >= fPrev)) ;
        //abs(gtd_new) <= -c2*gtd
        con2 := ABS(gtdNew) <= (-1*c2*gtd);
        // gtd_new >= 0
        con3 := gtdNew >= 0;
        bracketing_record bracketing_con1_3 (bracketing_record l) := TRANSFORM
        /* bracket = [t_prev t];
        bracketFval = [f_prev f_new];
        bracketGval = [g_prev g_new];*/
          SELF.bracket1_ := l.t_prev_;
          SELF.bracket2_ := l.t_;
          SELF.bracket1_f_ := l.f_prev_;
          SELF.bracket2_f_ := l.f_new_;
          SELF.bracket1_g_ := l.g_prev_;
          SELF.bracket2_g_ := l.g_new_;
          SELF := l;
        END;
        bracketing_record bracketing_con2 (bracketing_record l) := TRANSFORM
        /*  bracket = t;
        bracketFval = f_new;
        bracketGval = g_new;
        done = 1;*/
          SELF.bracket1_ := tt;
          SELF.bracket2_ := -1;
          SELF.bracket1_f_ := fNew;
          SELF.bracket2_f_ := -1;
          SELF.bracket1_g_ := gNew;
          SELF := l;
        END;

        bracketing_record bracketing_Nocon (bracketing_record l) := TRANSFORM
          //calculate new t
          minstep := tt + 0.01* (tt-tPrev);
          maxstep := tt*10;
          newt := polyinterp_both (tPrev, fPrev,gtdPrev, tt, fNew, gtdNew, minstep, maxstep);
          //calculate fnew gnew gtdnew
          xNew := calculate_xNew (d, newt);
          CostGradNew := CostFunc (xNew ,CostFunc_params,TrainData, TrainLabel);
          gNewwolfe := ExtractGrad (CostGradNew);
          fNewWolfe := ExtractCost (CostGradNew);
          gtdNewWolfe := ML.Mat.Mul (ML.Mat.Trans(ML.Types.ToMatrix(gNewwolfe)),ML.Types.ToMatrix(d));
          SELF.t_prev_ := tt; //t_prev = t;
          SELF.f_prev_ := fNew;
          SELF.g_prev_ := gNew;
          SELF.gtd_prev_ := gtdNew;
          SELF.t_ := newt;
          SELF.f_new_ := fNewWolfe;
          SELF.g_new_ := ML.Types.ToMatrix(gNewwolfe);
          SELF.gtd_new_ := gtdNewWolfe[1].value;
          SELF.bracket1_ := -1;
          SELF.bracket2_ := -1;
          SELF.funEvals_ := inputFunEval+1;
          SELF.c := l.c+1;
          SELF := l;
        END;
        con1_3_output := PROJECT(inputp,bracketing_con1_3(LEFT));
        con2_output := PROJECT(inputp,bracketing_con2(LEFT));
        Nocon_output := PROJECT(inputp,bracketing_Nocon(LEFT));
        RETURN IF (con1, con1_3_output, IF (con2, con2_output, IF (con3, con1_3_output, Nocon_output)));
      END;//END WolfeStep2
      ArmijoStep := FUNCTION
        RETURN inputp;
      END;// END ArmijoStep
      WolfeBracket := WolfeStep;
      ArmijoBracket := ArmijoStep;
      RETURN IF(AreTheyLegal,ArmijoBracket,WolfeBracket);
    END; // END BracketingStep
    
    
    
    ZoomingRecord := RECORD
      REAL8 bracket1_;
      REAL8 bracket2_;
      REAL8 bracket1_f_;
      REAL8 bracket2_f_;
      DATASET(Mat.Types.Element) bracket1_g_;
      DATASET(Mat.Types.Element) bracket2_g_;
      INTEGER8 funEvals_;
      BOOLEAN InsufProg:= FALSE;
      BOOLEAN Done := FALSE;
      BOOLEAN Break := FALSE;
    END;
    
    ZoomingStep (DATASET (ZoomingRecord) inputp, INTEGER coun) := FUNCTION
      t_first  := inputp[1].bracket1_;
      t_second := inputp[1].bracket2_;
      f_first  := inputp[1].bracket1_f_;
      f_second := inputp[1].bracket2_f_;
      g_first  := inputp[1].bracket1_g_;
      g_second := inputp[1].bracket2_g_;
      gtd_first := ML.Mat.Mul (ML.Mat.Trans (g_first),ML.Types.ToMatrix(d));
      gtd_second := ML.Mat.Mul (ML.Mat.Trans (g_second),ML.Types.ToMatrix(d));
      insufProgress := inputp[1].InsufProg;
      inputZFunEval := inputp[1].funEvals_;
      //
      // Find High and Low Points in bracket
      LOt := IF (f_first < f_second, t_first , t_second);
      HIt := IF (f_first < f_second, t_second, t_first );
      LOf := IF (f_first < f_second, f_first, f_second);
      HIf := IF (f_first < f_second,  f_second, f_first);
      LO_g := IF (f_first < f_second,  g_first, g_second);
      HIg := IF (f_first < f_second, g_second,   g_first);
      // Compute new trial value
      //t = polyinterp([bracket(1) bracketFval(1) bracketGval(:,1)'*d bracket(2) bracketFval(2) bracketGval(:,2)'*d],doPlot);
      tTmp := polyinterp_noboundry (t_first, f_first, gtd_first[1].value, t_second, f_second, gtd_second[1].value);
      //Test that we are making sufficient progress
      
      BList := [t_first,t_second];
      MaxB := MAX (BList);
      MinB := MIN (BList);
      //if min(max(bracket)-t,t-min(bracket))/(max(bracket)-min(bracket)) < 0.1
      MainPCondterm := (MIN ((MAXB - tTmp) , (tTmp - MINB)) / (MAXB - MINB) );
      MainPCond := MainPCondterm < 0.1 ;
      //if insufProgress || t>=max(bracket) || t <= min(bracket)
      PCond2 := insufProgress | (tTmp >= MAXB) | (tTmp <= MINB);
      //abs(t-max(bracket)) < abs(t-min(bracket))
      PCond2_1 := ABS (tTMP - MAXB) < ABS (tTmp - MINB);
      // t = max(bracket)-0.1*(max(bracket)-min(bracket));
      MMTemp := 0.1 * (MAXB - MINB);
      tIF    := MAXB - MMTemp;
      // t = min(bracket)+0.1*(max(bracket)-min(bracket));
      tELSE := MINB + MMTemp;
      tZOOM := IF (MainPCond,IF (PCond2, IF (PCond2_1, tIF, tELSE) , tTmp),tTmp);
      insufProgress_new := IF (MainPCond, IF (PCond2, FALSE, TRUE) , FALSE);
      //
      // Evaluate new point with tZoom
      x_td := ML.Types.FromMatrix (ML.Mat.Add(ML.Types.ToMatrix(x),ML.Mat.Scale(ML.Types.ToMatrix(d),tZOOM)));
      CG_New := CostFunc (x_td ,CostFunc_params,TrainData, TrainLabel);
            
      gNew := ExtractGrad (CG_New);
      fNew := ExtractCost(CG_New);
      gtdNew := ML.Mat.Mul (ML.Mat.Trans(ML.Types.ToMatrix(gNew)),ML.Types.ToMatrix(d));
      New_FunEval := inputZFunEval + 1;
      
      //conditions
      //IF f_new > f + c1*t*gtd || f_new >= f_LO
      ZoomCon1 := (fNew > f + c1 * tZoom * gtd) | (fNew >LOf);
      //if abs(gtd_new) <= - c2*gtd
      ZOOMCon1_1 := ABS (gtdNew[1].value) <= (-1 * c2 * gtd); 
      //gtd_new*(bracket(HIpos)-bracket(LOpos)) >= 0
      ZOOMCon1_2 := (gtdNew[1].value * (HIt - LOt)) >= 0; 
      ZoomingRecord zooming_Con1_f1 (ZoomingRecord l) := TRANSFORM //when Con1 is satisfied and f_first < f_second
      /*
        bracket(HIpos) = t;
        bracketFval(HIpos) = f_new;
        bracketGval(:,HIpos) = g_new;
        Tpos = HIpos;
      */
        SELF.bracket2_ := tZOOM;
        SELF.bracket2_f_ := fNew;
        SELF.bracket2_g_ := ML.Types.ToMatrix(gNew);
        SELF.funEvals_ := New_FunEval;
        SELF.InsufProg := insufProgress_new;
        SELF := l;
      END;
      ZoomingRecord zooming_Con1_f2 (ZoomingRecord l) := TRANSFORM //when Con1 is satisfied and f_second < f_first
              /*
        bracket(HIpos) = t;
        bracketFval(HIpos) = f_new;
        bracketGval(:,HIpos) = g_new;
        Tpos = HIpos;
      */
        SELF.bracket1_ := tZOOM;
        SELF.bracket1_f_ := fNew;
        SELF.bracket1_g_ := ML.Types.ToMatrix(gNew);
        SELF.funEvals_ := New_FunEval;
        SELF.InsufProg := insufProgress_new;
        SELF := l;
      END;

      ZoomingRecord zooming_Con1_1_f1 (ZoomingRecord l) := TRANSFORM ////when Con1_1 is satisfied and f_first < f_second
      /*
 % Wolfe conditions satisfied
            done = 1;
 % New point becomes new LO
        bracket(LOpos) = t;
        bracketFval(LOpos) = f_new;
        bracketGval(:,LOpos) = g_new;*/
        SELF.bracket1_ := tZOOM;
        SELF.bracket1_f_ := fNew;
        SELF.bracket1_g_ := ML.Types.ToMatrix(gNew);
        SELF.funEvals_ := New_FunEval;
        SELF.InsufProg := insufProgress_new;
        SELF.done := TRUE;
        SELF := l;
      END;
      ZoomingRecord zooming_Con1_1_f2 (ZoomingRecord l) := TRANSFORM ////when Con1_1 is satisfied and f_second < f_first
      /*
 % Wolfe conditions satisfied
            done = 1;
 % New point becomes new LO
        bracket(LOpos) = t;
        bracketFval(LOpos) = f_new;
        bracketGval(:,LOpos) = g_new;*/
        SELF.bracket2_ := tZOOM;
        SELF.bracket2_f_ := fNew;
        SELF.bracket2_g_ := ML.Types.ToMatrix(gNew);
        SELF.funEvals_ := New_FunEval;
        SELF.InsufProg := insufProgress_new;
        SELF.done := TRUE;
        SELF := l;
      END;
      
      
      ZoomingRecord zooming_Con1_2_f1 (ZoomingRecord l) := TRANSFORM ////when Con1_1 is satisfied and  f_first <f_second 
      /*
 
           
% Old HI becomes new LO
bracket(HIpos) = bracket(LOpos);
            bracketFval(HIpos) = bracketFval(LOpos);
            bracketGval(:,HIpos) = bracketGval(:,LOpos);
 % New point becomes new LO
        bracket(LOpos) = t;
        bracketFval(LOpos) = f_new;
        bracketGval(:,LOpos) = g_new;
*/
        SELF.bracket2_ := l.bracket1_;
        SELF.bracket2_f_ := l.bracket1_f_;
        SELF.bracket2_g_ := l.bracket1_g_;
        SELF.bracket1_ := tZOOM;
        SELF.bracket1_f_ := fNew;
        SELF.bracket1_g_ := ML.Types.ToMatrix(gNew);
        SELF.funEvals_ := New_FunEval;
        SELF.InsufProg := insufProgress_new;
        SELF := l;
      END;
      
        ZoomingRecord zooming_Con1_2_f2 (ZoomingRecord l) := TRANSFORM ////when Con1_1 is satisfied and  f_second <f_first
      /*       
% Old HI becomes new LO
bracket(HIpos) = bracket(LOpos);
            bracketFval(HIpos) = bracketFval(LOpos);
            bracketGval(:,HIpos) = bracketGval(:,LOpos);
 % New point becomes new LO
        bracket(LOpos) = t;
        bracketFval(LOpos) = f_new;
        bracketGval(:,LOpos) = g_new;
*/
        SELF.bracket1_ := l.bracket2_;
        SELF.bracket1_f_ := l.bracket2_f_;
        SELF.bracket1_g_ := l.bracket2_g_;
        SELF.bracket2_ := tZOOM;
        SELF.bracket2_f_ := fNew;
        SELF.bracket2_g_ := ML.Types.ToMatrix(gNew);
        SELF.funEvals_ := New_FunEval;
        SELF.InsufProg := insufProgress_new;
        SELF := l;
      END;
      
      ZoomingRecord zooming_NoCon_f1 (ZoomingRecord l) := TRANSFORM //// f_first < f_second
      /*
 % New point becomes new LO
        bracket(LOpos) = t;
        bracketFval(LOpos) = f_new;
        bracketGval(:,LOpos) = g_new;*/
        SELF.bracket1_ := tZOOM;
        SELF.bracket1_f_ := fNew;
        SELF.bracket1_g_ := ML.Types.ToMatrix(gNew);
        SELF.funEvals_ := New_FunEval;
        SELF.InsufProg := insufProgress_new;
        SELF := l;
      END;
      ZoomingRecord zooming_NoCon_f2 (ZoomingRecord l) := TRANSFORM //// f_second < f_first
      /*

 % New point becomes new LO
        bracket(LOpos) = t;
        bracketFval(LOpos) = f_new;
        bracketGval(:,LOpos) = g_new;*/
        SELF.bracket2_ := tZOOM;
        SELF.bracket2_f_ := fNew;
        SELF.bracket2_g_ := ML.Types.ToMatrix(gNew);
        SELF.funEvals_ := New_FunEval;
        SELF.InsufProg := insufProgress_new;
        SELF := l;
      END;

      output_Con1 := IF (f_first < f_second, PROJECT(inputp,zooming_Con1_f1(LEFT)), PROJECT(inputp,zooming_Con1_f2(LEFT)));
      output_Con1_1 := IF (f_first < f_second,PROJECT(inputp,zooming_Con1_1_f1(LEFT)), PROJECT(inputp,zooming_Con1_1_f2(LEFT)));
      output_Con1_2 := IF (f_first < f_second,PROJECT(inputp,zooming_Con1_2_f1(LEFT)), PROJECT(inputp,zooming_Con1_2_f2(LEFT)));
      output_NoCon := IF (f_first < f_second,PROJECT(inputp,zooming_NoCon_f1(LEFT)),PROJECT(inputp,zooming_NoCon_f2(LEFT)));
      IFOut := IF (ZoomCon1, output_Con1, IF (ZOOMCon1_1, output_Con1_1, IF (ZOOMCon1_2, output_Con1_2, output_NoCon ) ));
      ZoomingRecord BreakTran (IFOut l) := TRANSFORM
        // IF ~done && abs((bracket(1)-bracket(2))*gtd_new) < tolX then break
        SELF.break := (~l.done) & (abs((l.bracket1_-l.bracket2_)*gtdNew[1].value) < tolX);
        SELF := l;
      END;
      
      ZoomOut := PROJECT (IFOut , BreakTran (LEFT) );

      RETURN ZoomOut;
    END; // END ZoomingStep
    
    //Run the Loop as long as counter is less than or equal to maxLS and the bracket_t1 is not still found
    BracketingResult := LOOP(ToPass_bracketing, COUNTER <= maxLS AND ROWS(LEFT)[1].bracket1_ = -1, BracketingStep(ROWS(LEFT),COUNTER));
    //BracketingResult :=  BracketingStep(ToPass_bracketing,1);
    //BracketingResult := LOOP(ToPass_bracketing,COUNTER <= 1, BracketingStep(ROWS(LEFT),COUNTER)); 
    interval_found := BracketingResult[1].bracket1_ != -1 AND BracketingResult[1].bracket2_  !=-1;
    final_t_found := BracketingResult[1].bracket1_ != -1 AND BracketingResult[1].bracket2_  =-1;
    Zoom_Max_itr_tmp :=  maxLS - BracketingResult[1].c;
    Zoom_Max_Itr := IF (Zoom_Max_itr_tmp >0, Zoom_Max_itr_tmp, 0);
    
    
    ToPass_Zooming := PROJECT(BracketingResult, TRANSFORM(ZoomingRecord, SELF := LEFT));
    ZoomingResult := LOOP(ToPass_Zooming, COUNTER <= Zoom_Max_Itr AND ~ROWS(LEFT)[1].done AND ~ROWS(LEFT)[1].break, ZoomingStep(ROWS(LEFT),COUNTER));
    //ZoomingResult := ZoomingStep(ToPass_Zooming,1);

    
    thisoutputrecord := DATASET([{1,1,DATASET([{1,1,1},{1,2,3},{3,4,5}],Mat.Types.Element),7}],OutputRecord);
    OutputRecord finaltTran (BracketingResult l) := TRANSFORM
      SELF.t := l.bracket1_ ;
      SELF.f_new := l.bracket1_f_ ;
      SELF.g_new := l.bracket1_g_;
      SELF.WolfeFunEval := l.funEvals_;
    END;
    final_t_output := PROJECT (BracketingResult, finaltTran(LEFT));
    
    OutputRecord MaxItrTran (BracketingResult l) := TRANSFORM
      con := f < l.f_new_;
      SELF.t := IF (con, 0, l.t_) ;
      SELF.f_new := IF (con, f, l.f_new_) ;
      SELF.g_new := IF (con, ML.Types.ToMatrix(g), l.g_new_ );
      SELF.WolfeFunEval := l.funEvals_;
    END;
    MaxItr_output := PROJECT (BracketingResult, MaxItrTran(LEFT));
    
    OutputRecord ZoomTran (ZoomingResult l) := TRANSFORM
      con := l.bracket1_f_ < l.bracket2_f_;
      SELF.t := IF (con, l.bracket1_, l.bracket2_) ;
      SELF.f_new := IF (con, l.bracket1_f_, l.bracket2_f_) ;
      SELF.g_new := IF (con, l.bracket1_g_, l.bracket2_g_ );
      SELF.WolfeFunEval := l.funEvals_;
    END;
    zoom_output := PROJECT (ZoomingResult, ZoomTran(LEFT));
    FinalResult := IF (final_t_found,final_t_output , IF (Zoom_Max_itr_tmp=0,MaxItr_output,zoom_output));
    //ZoomingResult :=  ZoomingStep(ToPass_Zooming,1);
    
    //RETURN FinalResult; orig
    thisout := DATASET ([{1,1,ML.Types.ToMatrix(x),1}],OutputRecord);
    RETURN final_t_output;
    
  END;// END WolfeLineSearch2
  

   SHARED IdElementRec := RECORD
      INTEGER1 id;
      Mat.Types.Element;
   END;
   SHARED globalID := 1;
   SHARED appendID2mat (DATASET (Mat.Types.Element) mat) := FUNCTION
      IdElementRec addID (mat l) := TRANSFORM
        SELF.id := globalID;
        SELF := l;
      END;
      RETURN PROJECT(mat, addID(LEFT));
    END;
  
   EXPORT WolfeLineSearch3(INTEGER cccc, DATASET(Types.NumericField)x, REAL8 t, DATASET(Types.NumericField)d, REAL8 f, DATASET(Types.NumericField) g, REAL8 gtd, REAL8 c1=0.0001, REAL8 c2=0.9, INTEGER maxLS=25, REAL8 tolX=0.000000001,DATASET(Types.NumericField) CostFunc_params, DATASET(Types.NumericField) TrainData , DATASET(Types.NumericField) TrainLabel, DATASET(Types.NumericField) CostFunc (DATASET(Types.NumericField) x, DATASET(Types.NumericField) CostFunc_params, DATASET(Types.NumericField) TrainData , DATASET(Types.NumericField) TrainLabel), prows=0, pcols=0, Maxrows=0, Maxcols=0):=FUNCTION
    //initial parameters
    P_num := Max (x, id); //the length of the parameters vector (number of parameters)
    emptyE := DATASET([], Mat.Types.Element);

    emptyEid := DATASET([], IdElementRec);

    LSiter := 0;

    calculate_gtdNew (DATASET(Types.NumericField) g_in, DATASET(Types.NumericField) d_in) := FUNCTION
      Types.NumericField Mu(g_in le,d_in ri) := TRANSFORM
        SELF.id := le.id;
        SELF.number := ri.number;
        SELF.value := le.value * ri.value;
      END;
      element_mul := JOIN(g_in,d_in,LEFT.id=RIGHT.id,Mu(LEFT,RIGHT));
      r := RECORD
        t_RecordID id := 1 ;
        t_FieldNumber number := 1;
        t_FieldReal value := SUM(GROUP,element_mul.value);
      END;
      SumMulElm := TABLE(element_mul,r);
      RETURN SumMulElm[1].value;
    END;
    calculate_xNew (DATASET(Types.NumericField) d_in, REAL8 t_in) := FUNCTION
      //x_new = x+t*d
      Types.NumericField xnew_tran (x l, d_in r):= TRANSFORM
        SELF.value := l.value+(t_in*r.value);
        SELF := l;
      END;
      Result := JOIN(x,d_in,LEFT.id=RIGHT.id AND LEFT.number=RIGHT.number,xnew_tran(LEFT,RIGHT));
      RETURN Result;
    END;
    ExtractGrad (DATASET(Types.NumericField) inp) := FUNCTION
      RETURN inp (id <= P_num);
    END;
    ExtractCost (DATASET(Types.NumericField) inp) := FUNCTION
      RETURN inp (id = (P_num+1))[1].value;
    END;
    IsNotLegal (DATASET (Mat.Types.Element) Mat) := FUNCTION //???to be defined
      RETURN FALSE;
    END;
    IsNotLegal_real (REAL8 v) := FUNCTION //???to be defined
      RETURN FALSE;
    END;
  

    
    //% Evaluate the Objective and Gradient at the Initial Step
    //x_new = x+t*d
    //x_new := ML.Types.FromMatrix (ML.Mat.Add(ML.Types.ToMatrix(x),ML.Mat.Scale(ML.Types.ToMatrix(d),t)));
    x_new := calculate_xNew (d,t);
    CostGrad_new := CostFunc (x_new ,CostFunc_params,TrainData, TrainLabel);
    g_new := ExtractGrad (CostGrad_new);
    f_new := ExtractCost (CostGrad_new);
    funEvals := 1;
    //gtd_new = g_new'*d;
    //gtd_new := ML.Mat.Mul (ML.Mat.Trans(ML.Types.ToMatrix(g_new)),ML.Types.ToMatrix(d));
    gtd_new := calculate_gtdNew (g_new, d);
    
    // Bracket an Interval containing a point satisfying the Wolfe Criteria
    t_prev := 0;
    f_prev := f;
    g_prev := g;
    gtd_prev := gtd;
   
    //Bracketing algorithm, either produces the final t value or a bracket that contains the final t value
     bracketing_record_nomat := RECORD
      INTEGER1 id;
      REAL8 f_prev_;
      REAL8 f_new_;
      REAL8 t_;
      REAL8 t_prev_;
      INTEGER8 funEvals_;
      REAL8 gtd_prev_;
      REAL8 gtd_new_;
      REAL8 bracket1_;
      REAL8 bracket2_;
      REAL8 bracket1_f_;
      REAL8 bracket2_f_;
      INTEGER8 c; //Counter
    END;
    bracketing_record := RECORD
      INTEGER1 id;
      REAL8 f_prev_;
      REAL8 f_new_;
      DATASET(IdElementRec) g_prev_;
      DATASET(IdElementRec) g_new_;
      REAL8 t_;
      REAL8 t_prev_;
      INTEGER8 funEvals_;
      REAL8 gtd_prev_;
      REAL8 gtd_new_;
      REAL8 bracket1_;
      REAL8 bracket2_;
      REAL8 bracket1_f_;
      REAL8 bracket2_f_;
      DATASET(IdElementRec) bracket1_g_;
      DATASET(IdElementRec) bracket2_g_;
      INTEGER8 c; //Counter
    END;
    


    ToPass_bracketing_nomat := DATASET ([{globalID,f_prev,f_new,t,t_prev,funEvals,gtd_prev,gtd_new,-1,-1,-1,-1,0}],bracketing_record_nomat);
    bracketing_record DeNorm_gp(bracketing_record L, IdElementRec R) := TRANSFORM
      SELF.g_prev_ := L.g_prev_ + R;
      SELF := L;
    END;
    bracketing_record DeNorm_gn(bracketing_record L, IdElementRec R) := TRANSFORM
      SELF.g_new_ := L.g_new_ + R;
      SELF := L;
    END;
    bracketing_record DeNorm_bracket_g1 (bracketing_record L, IdElementRec R) := TRANSFORM
      SELF.bracket1_g_ := L.bracket1_g_ + R;
      SELF := L;
     END;
     bracketing_record DeNorm_bracket_g2 (bracketing_record L, IdElementRec R) := TRANSFORM
      SELF.bracket2_g_ := L.bracket2_g_ + R;
      SELF := L;
     END;
    BuildBracketingData (DATASET(bracketing_record_nomat) p, DATASET (Mat.Types.Element) gp, DATASET (Mat.Types.Element) gn)  := FUNCTION //this function returns a bracketing_record dataset in which the two matrices gp and gn are nested in g_prev_ and g_new_ fields relatively (the matrices are actually nested in the IdElementRec fromat)
      //add id to the two input matrices
      gp_id := appendID2mat (gp);
      gn_id := appendID2mat (gn);
      //load parent p with empty datasets for the dataset fields
      bracketing_record LoadParent(bracketing_record_nomat L) := TRANSFORM
        SELF.g_prev_ := [];
        SELF.g_new_ := [];
        SELF.bracket1_g_ := [];
        SELF.bracket2_g_ := [];
        SELF := L;
      END;

      //1 - fill in the p with Empty datasets
      p_ready := PROJECT(p,LoadParent(LEFT));
      // 1- fill in p_ready with gp
      p_gp := DENORMALIZE(p_ready, gp_id, LEFT.id = RIGHT.id, DeNorm_gp(LEFT,RIGHT));
      // 2- fill in p_gp with gn
      p_gp_gn := DENORMALIZE(p_gp, gn_id, LEFT.id = RIGHT.id, DeNorm_gn(LEFT,RIGHT));
      
      RETURN p_gp_gn;
    END;
    
    g_prev_ext (DATASET (bracketing_record) br) := FUNCTION
      IdElementRec NewChildren(IdElementRec R) := TRANSFORM
        SELF := R;
      END;
      NewChilds := NORMALIZE(br,LEFT.g_prev_,NewChildren(RIGHT));

      RETURN PROJECT(NewChilds, TRANSFORM(Mat.Types.Element, SELF := LEFT));
    END; // END g_prev_ext
    g_new_ext (DATASET (bracketing_record) br) := FUNCTION
      IdElementRec NewChildren(IdElementRec R) := TRANSFORM
        SELF := R;
      END;
      NewChilds := NORMALIZE(br,LEFT.g_new_,NewChildren(RIGHT));
  
      RETURN PROJECT(NewChilds, TRANSFORM(Mat.Types.Element, SELF := LEFT));
    END; // END g_new_ext
    
    bracket1g_ext (DATASET (bracketing_record) br) := FUNCTION
      IdElementRec NewChildren(IdElementRec R) := TRANSFORM
        SELF := R;
      END;
      NewChilds := NORMALIZE(br,LEFT.bracket1_g_,NewChildren(RIGHT));
  
      RETURN PROJECT(NewChilds, TRANSFORM(Mat.Types.Element, SELF := LEFT));
    END; // END bracket1g_ext

    SetBracket (DATASET (bracketing_record) B, REAL8 b1, REAL8 b2, REAL8 f1, REAL8 f2, DATASET (Mat.Types.Element) g1, DATASET (Mat.Types.Element) g2  ) := FUNCTION
      g1_id := appendID2mat (g1);
      g2_id := appendID2mat (g2);
      bracketing_record set_b_f (bracketing_record l) := TRANSFORM
        SELF.bracket1_ := b1;
        SELF.bracket2_ := b2;
        SELF.bracket1_f_ := f1;
        SELF.bracket2_f_ := f2;
        SELF.bracket1_g_ := [];
        SELF.bracket2_g_ := [];
        SELF := l;
      END;

      
      B_ready := PROJECT (B, set_b_f (LEFT) );
      B_g1 := DENORMALIZE(B_ready, g1_id, LEFT.id = RIGHT.id, DeNorm_bracket_g1(LEFT,RIGHT));
      B_g1_g2 := DENORMALIZE(B_g1, g2_id, LEFT.id = RIGHT.id, DeNorm_bracket_g2(LEFT,RIGHT));
      RETURN B_g1_g2;
    END;
    SetBracket1 (DATASET (bracketing_record) B, REAL8 b1, REAL8 f1, DATASET (Mat.Types.Element) g1 ) := FUNCTION
      g1_id := appendID2mat (g1);
      bracketing_record set_b_f (bracketing_record l) := TRANSFORM
        SELF.bracket1_ := b1;
        SELF.bracket2_ := -1;
        SELF.bracket1_f_ := f1;
        SELF.bracket2_f_ := -1;
        SELF.bracket1_g_ := [];
        SELF.bracket2_g_ := [];
        SELF := l;
      END;

     
      B_ready := PROJECT (B, set_b_f (LEFT) );
      B_g1 := DENORMALIZE(B_ready, g1_id, LEFT.id = RIGHT.id, DeNorm_bracket_g1(LEFT,RIGHT));
      RETURN B_g1;
    END;
    ToPassBracketing := BuildBracketingData (ToPass_bracketing_nomat,ML.Types.ToMatrix(g_prev), ML.Types.ToMatrix(g_new) );
  
    BracketingStep (DATASET (bracketing_record) inputp, INTEGER coun) := FUNCTION
      inputp1 := inputp[1];
      AreTheyLegal := TRUE; // to be evaluated ???
      WolfeStep := FUNCTION
        fNew := inputp1.f_new_;
        fPrev := inputp1.f_prev_;
        gtdNew := inputp1.gtd_new_;
        gtdPrev := inputp1.gtd_prev_;
        tt := inputp1.t_;
        tPrev := inputp1.t_prev_;
        gNew := g_new_ext(inputp);
        gPrev := g_prev_ext(inputp);
        inputFunEval := inputp1.funEvals_;
        BrackLSiter := coun-1;
        
        //If the strong wolfe conditions satisfies then retun the final t or the bracket, otherwise do the next iteration
        //f_new > f + c1*t*gtd || (LSiter > 1 && f_new >= f_prev)
        con1 := (fNew > f + c1 * tt* gtd)| ((BrackLSiter > 1) & (fNew >= fPrev)) ;
        //abs(gtd_new) <= -c2*gtd
        con2 := ABS(gtdNew) <= (-1*c2*gtd);
        // gtd_new >= 0
        con3 := gtdNew >= 0;
        /* bracket = [t_prev t];
        bracketFval = [f_prev f_new];
        bracketGval = [g_prev g_new];*/
        bracketing_con1_3 := SetBracket(inputp, tPrev, tt, fPrev, fNew, gPrev, gNew);
        
        
        /*  bracket = t;
        bracketFval = f_new;
        bracketGval = g_new;
        done = 1;*/
        bracketing_con2 := SetBracket1(inputp, tt, fNew, gNew);

        bracketing_Nocon := FUNCTION
          //calculate new t
          minstep := tt + 0.01* (tt-tPrev);
          maxstep := tt*10;
          newt := polyinterp_both (tPrev, fPrev,gtdPrev, tt, fNew, gtdNew, minstep, maxstep);
          //calculate fnew gnew gtdnew
          xNew := calculate_xNew (d, newt);
          CostGradNew := CostFunc (xNew ,CostFunc_params,TrainData, TrainLabel);
          gNewwolfe := ExtractGrad (CostGradNew);
          fNewWolfe := ExtractCost (CostGradNew);
          gtdNewWolfe := ML.Mat.Mul (ML.Mat.Trans(ML.Types.ToMatrix(gNewwolfe)),ML.Types.ToMatrix(d));
          bracketing_record SetValues (bracketing_record l) := TRANSFORM
            SELF.t_prev_ := tt; //t_prev = t;
            SELF.f_prev_ := fNew;
            SELF.g_prev_ := []; //gNew;
            SELF.gtd_prev_ := gtdNew;
            SELF.t_ := newt;
            SELF.f_new_ := fNewWolfe;
            SELF.g_new_ := []; // ML.Types.ToMatrix(gNewwolfe);
            SELF.gtd_new_ := gtdNewWolfe[1].value;
            SELF.bracket1_ := -1;
            SELF.bracket2_ := -1;
            SELF.funEvals_ := inputFunEval+1;
            SELF.c := l.c+1;
            SELF := l;
          END;
          B_ready := PROJECT (inputp, SetValues (LEFT) );
          B_gp := DENORMALIZE(B_ready, appendID2mat (gNew), LEFT.id = RIGHT.id, DeNorm_gp(LEFT,RIGHT));
          B_gp_gn := DENORMALIZE(B_gp, appendID2mat(ML.Types.ToMatrix(gNewwolfe)), LEFT.id = RIGHT.id, DeNorm_gn(LEFT,RIGHT));
          RETURN B_gp_gn;
        END;
        RETURN IF (con1, bracketing_con1_3, IF (con2, bracketing_con2, IF (con3, bracketing_con1_3, bracketing_Nocon)));        
      END;//END WolfeStep
      ArmijoStep := FUNCTION
        RETURN inputp;
      END;// END ArmijoStep
      RETURN IF(AreTheyLegal,WolfeStep,ArmijoStep);
    END; // END BracketingStep
    BracketingResult := LOOP(ToPassBracketing, COUNTER <= maxLS AND ROWS(LEFT)[1].bracket1_ = -1, BracketingStep(ROWS(LEFT),COUNTER));
    interval_found := BracketingResult[1].bracket1_ != -1 AND BracketingResult[1].bracket2_  !=-1;
    final_t_found := BracketingResult[1].bracket1_ != -1 AND BracketingResult[1].bracket2_  =-1;
    Zoom_Max_itr_tmp :=  maxLS - BracketingResult[1].c;
    Zoom_Max_Itr := IF (Zoom_Max_itr_tmp >0, Zoom_Max_itr_tmp, 0);
/* OutputRecord finaltTran (BracketingResult l) := TRANSFORM
      SELF.t := l.bracket1_ ;
      SELF.f_new := l.bracket1_f_ ;
      SELF.g_new := l.bracket1_g_;
      SELF.WolfeFunEval := l.funEvals_;
    END;*/
    final_t_output := DATASET ([{P_num+1,1,BracketingResult[1].bracket1_},{P_num+2,1,BracketingResult[1].bracket1_f_},{P_num+3,1,BracketingResult[1].funEvals_}],Mat.Types.Element) + bracket1g_ext(BracketingResult);
    RETURN final_t_output;
  END;// END WolfeLineSearch3
  EXPORT wolfe_gnew_ext  (DATASET (Mat.Types.Element) wolfeout) := FUNCTION
    ind := MAX (wolfeout,x)-3;
    RETURN wolfeout (x>=1 and x <=ind);
  END;
  EXPORT wolfe_t_ext  (DATASET (Mat.Types.Element) wolfeout) := FUNCTION
    ind := MAX (wolfeout,x)-2;
    RETURN wolfeout (x = ind)[1].value;
  END;
  EXPORT wolfe_fnew_ext  (DATASET (Mat.Types.Element) wolfeout) := FUNCTION
    ind := MAX (wolfeout,x)-1;
    RETURN wolfeout (x = ind)[1].value;
  END;
  EXPORT wolfe_funeval_ext  (DATASET (Mat.Types.Element) wolfeout) := FUNCTION
    ind := MAX (wolfeout,x);
    RETURN wolfeout (x = ind)[1].value;
  END;
  //x,t,d,f,fr,g,gtd,c1,LS,tolX,debug,doPlot,saveHessianComp,funObj,varargin)
  // Backtracking linesearch to satisfy Armijo condition
  //
  // Inputs:
  //   x: starting location
  //   t: initial step size
  //   d: descent direction
  //   f: function value at starting location
  //   fr: reference function value (usually funObj(x))
  //   gtd: directional derivative at starting location
  //   c1: sufficient decrease parameter
  //   debug: display debugging information
  //   LS: type of interpolation
  //   tolX: minimum allowable step length
  //   doPlot: do a graphical display of interpolation
  //   funObj: objective function
  //   varargin: parameters of objective function
  EXPORT ArmijoBacktrack(DATASET(Types.NumericField)x, REAL8 t, DATASET(Types.NumericField)d, REAL8 f, REAL8 fr, DATASET(Types.NumericField) g, REAL8 gtd, REAL8 c1=0.0001, REAL8 tolX=0.000000001,DATASET(Types.NumericField) CostFunc_params, DATASET(Types.NumericField) TrainData , DATASET(Types.NumericField) TrainLabel, DATASET(Types.NumericField) CostFunc (DATASET(Types.NumericField) x, DATASET(Types.NumericField) CostFunc_params, DATASET(Types.NumericField) TrainData , DATASET(Types.NumericField) TrainLabel), prows=0, pcols=0, Maxrows=0, Maxcols=0):=FUNCTION
    
     P_num := Max (x, id); //the length of the parameters vector (number of parameters)
      IsNotLegal (DATASET (Mat.Types.Element) Mat) := FUNCTION //???to be defined
        RETURN FALSE;
      END;
      ExtractGrad (DATASET(Types.NumericField) inp) := FUNCTION
        RETURN inp (id <= P_num);
      END;
      ExtractCost (DATASET(Types.NumericField) inp) := FUNCTION
        RETURN inp (id = (P_num+1))[1].value;
      END;
      // Evaluate the Objective and Gradient at the Initial Step
      xNew := ML.Types.FromMatrix (ML.Mat.Add(ML.Types.ToMatrix(x),ML.Mat.Scale(ML.Types.ToMatrix(d),t)));
      CostGradNew := CostFunc (xNew ,CostFunc_params,TrainData, TrainLabel);
      gNew := ExtractGrad (CostGradNew);
      fNew := ExtractCost (CostGradNew);
      //loop term to be used in the loop condition
      LoopTerm := fr + c1*t*gtd;
      //build what should be passed to the bracketing loop
      fNewno := DATASET([{1,1,fNew,1}], Mat.Types.MUElement);
      gNewno := Mat.MU.To (ML.Types.ToMatrix(gNew),2);
      funevalno := DATASET([{1,1,1,3}], Mat.Types.MUElement);
      tno := DATASET([{1,1,t,4}], Mat.Types.MUElement);
      fPrevno := DATASET([{1,1,-1,5}], Mat.Types.MUElement);
      tPrevno := DATASET([{1,1,0,6}], Mat.Types.MUElement);
      breakno := DATASET([{1,1,0,7}], Mat.Types.MUElement);
      toPass := fNewno + gNewno + funevalno + tno + fPrevno + tPrevno + breakno;
      Armijo (DATASET (Mat.Types.MUElement) Inp) := FUNCTION
        //calculate new t
        fNewit := Mat.MU.From (Inp,1);
        gNewit := Mat.MU.FROM (Inp,2);
        tit := Mat.MU.From (Inp,4)[1].value;
        tPrevit := Mat.MU.From (Inp,6)[1].value;
        gtdNewit := ML.Mat.Mul (ML.Mat.Trans((gNewit)),ML.Types.ToMatrix(d));
        FEval := Mat.MU.FROM (Inp,3)[1].value;
        fPrevit := Mat.MU.FROM (Inp,5);
        cond := IsNotLegal (fPrevit) | (FEval <2);
        // t = polyinterp([0 f gtd; t f_new sqrt(-1)],doPlot);
        tTemp := polyinterp_img (0, f, gtd, tit, fNewit[1].value, gtdNewit[1].value);
        tTemp2 := tTemp;//t = polyinterp([0 f gtd; t f_new sqrt(-1); t_prev f_prev sqrt(-1)],doPlot);
        tNew := IF (IsNotLegal(fNewit),tit*0.5,IF (cond,tTemp,tTemp2));
        //Adjust tNew if change in t is too small/large
        T1 := tit*0.001;
        T2 := tit*0.6;
        AdCond1 := tNew < T1 ; //t < temp*1e-3
        AdCond2 := tNew > T2; //t > temp*0.6
        AdtNew := IF (AdCond1, T1, IF (AdCond2, T2, tNew));
        //Calculate new f and g values
        fPrvno := DATASET([{1,1,fNewit[1].value,5}], Mat.Types.MUElement);
        tPrvno := DATASET([{1,1,tit,6}], Mat.Types.MUElement);
        //calculate new f and g
        xN := ML.Types.FromMatrix (ML.Mat.Add(ML.Types.ToMatrix(x),ML.Mat.Scale(ML.Types.ToMatrix(d),AdtNew)));
        CGN := CostFunc (xN ,CostFunc_params,TrainData, TrainLabel);
        gN := ExtractGrad (CGN);
        fN := ExtractCost (CGN);
        gNno := Mat.MU.To (ML.Types.ToMatrix(gN),2);
        fNno := DATASET([{1,1,fN,1}], Mat.Types.MUElement);
        FEalno := DATASET([{1,1,FEval+1,3}], Mat.Types.MUElement);
        tNno := DATASET([{1,1,AdtNew,4}], Mat.Types.MUElement);
        //sum(abs(t*d))???? to be calculated
        sumA := 3;
        brk := IF(sumA<=tolX,1,0);
        brkno := DATASET([{1,1,brk,7}], Mat.Types.MUElement);
        RETURN fNno + gNno + FEalno + tNno + fPrvno +  tPrvno + brkno;
      END;
      //f_new > fr + c1*t*gtd || ~isLegal(f_new)
      LoopResult := LOOP (toPass, (Mat.MU.From (ROWS(LEFT),7)[1].value = 0) & (IsNotLegal (Mat.MU.From (ROWS(LEFT),1)) & (Mat.MU.From (ROWS(LEFT),1)[1].value < LoopTerm ) ), Armijo(ROWS(LEFT)));
      BCond := Mat.MU.From (LoopResult,7)[1].value = 1;
      //[t,f_new,g_new,funEvals] return x_new too????
      RegularResult := LoopResult (no=4) + LoopResult (no=1) + LoopResult (no=2) + LoopResult (no=3);
      Breakresult := DATASET([{1,1,0,4}], Mat.Types.MUElement) + DATASET([{1,1,f,1}], Mat.Types.MUElement) + Mat.MU.To (ML.Types.ToMatrix(g),2) + LoopResult (no=3);
      FinalResult := IF (BCond, Breakresult, RegularResult);
    RETURN FinalResult;
  END; // END ArmijoBacktrack

  //LBFGS algorithm
  //Returns final updated parameter: numericField foramt
  //x0: input parameter vector (column)
  //CostFunc : function handler , it should return a Cost value and the gradient values which is a vector with the same size of x0
  //The output of the CostFunc function should be in numericField format where the last id's value (the maximum id) represents cost value and the rest
  //represent the gradient vector
  //So basically CostFunc should recive all its parameters in one single numericField structure + training data + training labels and return a vector of the gradients+costvalue
  //Cost function should have a universal interface, so it recives all parameters in numericfield format and returns in numericfield format
  //CostFunc_params : parameters that need to be passed to the CostFunc
  //TrainData : Train data in numericField format
  //TrainLabel : labels asigned to the train data ( if it is an un-supervised task this parameter would be empty dataset)
  //MaxIter: Maximum number of iteration allowed in the optimization algorithm
  //tolFun : Termination tolerance on the first-order optimality (1e-5)
  //TolX : Termination tolerance on progress in terms of function/parameter changes (1e-9)
  //maxFunEvals : Maximum number of function evaluations allowed (1000)
  //corrections : number of corrections to store in memory (default: 100) higher numbers converge faster but use more memory)
  //this Macro recives all the parameters in numeric field fromat and returns in numeric field format
  //prows and maxrows related to "numer of parameters (P)" which is actually the length of the x0 vector
  //pcols and Maxcols are relaetd to "number of correstions to store in the memory (corrections)" which is in the MethodOptions
  //In all operation I want to f or g get nan value if it is devided by zero (do I need to include #option on top of the CostFunc)????????

//LBFGS algorithm
  //Returns final updated parameter: numericField foramt
  //x0: input parameter vector (column)
  //CostFunc : function handler , it should return a Cost value and the gradient values which is a vector with the same size of x0
  //The output of the CostFunc function should be in numericField format where the last id's value (the maximum id) represents cost value and the rest
  //represent the gradient vector
  //So basically CostFunc should recive all its parameters in one single numericField structure + training data + training labels and return a vector of the gradients+costvalue
  //Cost function should have a universal interface, so it recives all parameters in numericfield format and returns in numericfield format
  //CostFunc_params : parameters that need to be passed to the CostFunc
  //TrainData : Train data in numericField format
  //TrainLabel : labels asigned to the train data ( if it is an un-supervised task this parameter would be empty dataset)
  //MaxIter: Maximum number of iteration allowed in the optimization algorithm
  //tolFun : Termination tolerance on the first-order optimality (1e-5)
  //TolX : Termination tolerance on progress in terms of function/parameter changes (1e-9)
  //maxFunEvals : Maximum number of function evaluations allowed (1000)
  //corrections : number of corrections to store in memory (default: 100) higher numbers converge faster but use more memory)
  //this Macro recives all the parameters in numeric field fromat and returns in numeric field format
  //prows and maxrows related to "numer of parameters (P)" which is actually the length of the x0 vector
  //pcols and Maxcols are relaetd to "number of correstions to store in the memory (corrections)" which is in the MethodOptions
  //In all operation I want to f or g get nan value if it is devided by zero (do I need to include #option on top of the CostFunc)????????
  EXPORT MinFUNC(DATASET(Types.NumericField) x0, DATASET(Types.NumericField) CostFunc (DATASET(Types.NumericField) x, DATASET(Types.NumericField) CostFunc_params, DATASET(Types.NumericField) TrainData , DATASET(Types.NumericField) TrainLabel), DATASET(Types.NumericField) CostFunc_params, DATASET(Types.NumericField) TrainData , DATASET(Types.NumericField) TrainLabel, INTEGER MaxIter = 100, REAL8 tolFun = 0.00001, REAL8 TolX = 0.000000001, INTEGER maxFunEvals = 1000, INTEGER corrections = 100, prows=0, pcols=0, Maxrows=0, Maxcols=0) := FUNCTION
  //#option ('divideByZero', 'nan'); //In all operation I want to f or g get nan value if it is devided by zero
  //Functions used
  SumABSg (DATASET (Types.NumericField) g_temp) := FUNCTION
    r := RECORD
      t_RecordID id := 1 ;
      t_FieldNumber number := 1;
      t_FieldReal value := SUM(GROUP,ABS(g_temp.value));
    END;
    SumABSCol := TABLE(g_temp,r);
    RETURN SumABSCol[1].value;
  END;
  SumABSg_matrix (DATASET (Mat.Types.Element) g_temp) := FUNCTION
    r := RECORD
      Mat.Types.t_Index x := 1 ;
      Mat.Types.t_Index y := 1;
      Mat.Types.t_value value := SUM(GROUP,ABS(g_temp.value));
    END;
    SumABSCol := TABLE(g_temp,r);
    RETURN SumABSCol[1].value;
  END;
  // Check Optimality Condition: if sum(abs(g)) <= tolFun return True
  OptimalityCond (DATASET (Types.NumericField) g_temp) := FUNCTION
    sag := SumABSg (g_temp);
    RETURN IF( sag <= tolFun , TRUE, FALSE);
  END;
  //Check Optimality Condition for the Loop : if sum(abs(g)) <= tolFun return TRUE
  OptimalityCond_mat (DATASET (Mat.Types.Element) g_t) := FUNCTION
    sag := SumABSg_matrix (g_t);
    RETURN IF( sag <= tolFun , TRUE, FALSE);
  END;//END OptimalityCond_mat
  //Check for lack of progress 1: if sum(abs(t*d)) <= tolX return TRUE
  LackofProgress1 (DATASET (Mat.Types.Element) d_temp, REAL8 t_temp) := FUNCTION
    r := RECORD
      Mat.Types.t_Index x := 1 ;
      Mat.Types.t_Index y := d_temp.y;
      Mat.Types.t_Value value := SUM(GROUP,ABS(d_temp.value * t_temp));
    END;
    SumABSCol := TABLE(d_temp,r,d_temp.y);
    RETURN IF( SumABSCol[1].value <= tolX , TRUE, FALSE);
  END;
  //Check for lack of progress 2: if abs(f-f_old) < tolX return TRUE
  LackofProgress2 (REAL8 f_f_temp) := FUNCTION
    RETURN IF (ABS (f_f_temp) < tolX, TRUE, FALSE);
  END;
  //Check for going over evaluation limit
  //if funEvals*funEvalMultiplier > maxFunEvals return TRUE  (funEvalMultiplier=1)
  EvaluationLimit (INTEGER8 fun_temp) := FUNCTION
    RETURN IF (fun_temp > maxFunEvals, TRUE, FALSE);
  END;
  IsLegal (DATASET (Types.NumericField) inp) := FUNCTION //???to be defined
    RETURN TRUE;
  END;
  //the length of the parameters vector (for example in an neural network algorith, all the weight parameters are passed in ONE vector
  //to the optimization algorithm to get updated

  P := Max (x0, id); // the maximum matrix id in the parameters numericfield dataset
  ExtractGrad (DATASET(Types.NumericField) inp) := FUNCTION
    RETURN inp (id <= P);
  END;
  ExtractCost (DATASET(Types.NumericField) inp) := FUNCTION
    RETURN inp (id = (P+1))[1].value;
  END;
  //Evaluate Initial Point
  CostGrad0 := CostFunc (x0,CostFunc_params,TrainData, TrainLabel);

  g0 := ExtractGrad (CostGrad0);
  Cost0 := ExtractCost (CostGrad0);
  //Check the optimality of the initial point (if sum(abs(g)) <= tolFun)
  IsInitialPointOptimal := OptimalityCond (g0);

  //LBFGS Module
  O := Limited_Memory_BFGS (P, corrections);
  //initialize Hdiag,old_dir,old_steps, gradient and cost
  //The size of the old_dir and old_steps matrices are "number of parameters" * "number of corrections to store in memory (corrections)"
  // old_dir0 := Mat.Zeros(P, corrections);
  // old_steps0 := old_dir0;
  //New Matrix Generator
  Mat.Types.Element gen(UNSIGNED4 c, UNSIGNED4 NumRows, REAL8 v=0) := TRANSFORM
    SELF.x := ((c-1) % NumRows) + 1;
    SELF.y := ((c-1) DIV NumRows) + 1;
    SELF.value := v;
  END;
  // Initialize dirs and steps matrices
  old_dir0   := DATASET(P * corrections, gen(COUNTER, P));
  old_steps0 := DATASET(P * corrections, gen(COUNTER, P));
  Hdiag0 := 1;
  //initialize hessian diag as 1
  Hdiag0no := DATASET([{1,1,1,5}], Mat.Types.MUElement);
  //number of times the cost function is evaluated
  FunEval := 1; 
  //Perform up to a maximum of 'maxIter' descent steps:
  //put all the parameters that need to be sent to the step fucntion in a Mat.Types.MUElement format
  
  emptyE := DATASET([], Mat.Types.Element);
  MinFRecord := RECORD
    DATASET(Mat.Types.Element) x;
    DATASET(Mat.Types.Element) g;
    DATASET(Mat.Types.Element) old_steps;
    DATASET(Mat.Types.Element) old_dirs;
    REAL8 Hdiag;
    REAL8 Cost;
    INTEGER8 funEvals_;
    DATASET(Mat.Types.Element) d;
    REAL8 fnew_fold; //f_new-f_old in that iteration
    REAL8 t_;
    BOOLEAN dLegal;
    BOOLEAN ProgAlongDir; //Progress Along Directionno //Check that progress can be made along direction ( if gtd > -tolX)
    BOOLEAN optcond; // Check Optimality Condition
    BOOLEAN lackprog1; //Check for lack of progress 1
    BOOLEAN lackprog2; //Check for lack of progress 2
    BOOLEAN exceedfuneval; //Check for going over evaluation limit
  END;

  ToPassMinF := DATASET ([{ML.Types.ToMatrix(x0),ML.Types.ToMatrix(g0),old_steps0,old_dir0,Hdiag0,Cost0,FunEval,emptyE,100 +tolFun,1,TRUE,TRUE,FALSE,FALSE,FALSE,FALSE}],MinFRecord);


  MinFstep (DATASET (MinFRecord) inputp, INTEGER coun) := FUNCTION
  /*
    x_pre := inputp[1].x;
    g_pre := inputp[1].g;
    g_pre_ := ML.Mat.Scale (g_pre , -1);
    step_pre := inputp[1].old_steps;
    dir_pre := inputp[1].old_dirs;
    H_pre := inputp[1].Hdiag;
    f_pre := inputp[1].Cost;
    FunEval_pre := inputp[1].funEvals_;
    //HG_ is actually search direction in fromual 3.1
    HG_ :=  IF (coun = 1, O.Steepest_Descent (g_pre), O.lbfgs(g_pre_,Dir_pre, Step_pre,H_pre));//the result is the approximate inverse Hessian, multiplied by the gradient and it is in PBblas.layout format
    d := ML.DMat.Converted.FromPart2DS(HG_);
    dlegalstep := IsLegal (d);
    d_next := ML.Types.ToMatrix(d);
    //Directional Derivative : gtd = g'*d;
    //calculate gtd to be passed to the wolfe algortihm
    gtd := ML.Mat.Mul(ML.Mat.Trans((g_pre)),ML.Types.ToMatrix(d));
    //Check that progress can be made along direction : if gtd > -tolX then break!
    gtdprogress := IF (gtd[1].value > -1*tolX, FALSE, TRUE);
    // Select Initial Guess If coun =1 then t = min(1,1/sum(abs(g)));
    t := IF (coun = 1,MIN ([1, 1/SumABSg (ML.Types.FromMatrix (g_pre))]),1);
     //find point satisfiying wolfe
    //[t,f,g,LSfunEvals] = WolfeLineSearch(x,t,d,f,g,gtd,c1,c2,LS,25,tolX,debug,doPlot,1,funObj,varargin{:});
    t_neworig := WolfeLineSearch2(1, ML.Types.FromMatrix(x_pre), t, d, f_pre, ML.Types.FromMatrix(g_pre), gtd[1].value, 0.0001, 0.9, 25, 0.000000001, CostFunc_params, TrainData , TrainLabel, CostFunc , prows, pcols, Maxrows, Maxcols);
    t_new := t_neworig[1].t;
    g_Next := t_neworig[1].g_new;
    Cost_Next := t_neworig[1].f_new;
    FunEval_Wolfe := t_neworig[1].WolfeFunEval;
    FunEval_next := FunEval_pre + FunEval_Wolfe;
    x_pre_updated :=  ML.Mat.Add((x_pre),ML.Mat.Scale(ML.Types.ToMatrix(d),t_new));
    x_Next := ML.Types.FromMatrix(x_pre_updated);
    fpre_fnext := Cost_Next-f_pre;
    //calculate new Hessian diag, dir and steps
    //Step_Next :=  O.lbfgsUpdate_corr (g_pre, g_Next, Step_pre);
    Step_Next := O.lbfgsUpdate_Stps (x_pre, x_pre_updated, g_pre, g_Next, Step_pre);
    //Dir_Next := O.lbfgsUpdate_corr(x_pre, x_pre_updated, Dir_pre);
    Dir_Next := O.lbfgsUpdate_Dirs (x_pre, x_pre_updated, g_pre, g_Next, Dir_pre);
    H_Next := O.lbfgsUpdate_Hdiag (x_pre, x_pre_updated, g_pre, g_Next, H_pre);
    optcond := OptimalityCond_mat (g_Next);
    lack1 := LackofProgress1 (d_next, t_new);
    lack2 := LackofProgress2 (fpre_fnext);
    evalimit := EvaluationLimit (FunEval_next);
    //ToReturn := x_Next_no + g_Nextno + Step_Nextno + Dir_Nextno + H_Nextno+  C_Nextno + FunEval_Nextno + d_Nextno + fpre_fnextno + t_newno + dlegalstepno + gtdprogressno;
    MinFRecord MF (MinFRecord l) := TRANSFORM
      SELF.x := x_pre_updated;
      SELF.g := g_Next;
      SELF.old_steps := Step_Next;
      SELF.old_dirs := Dir_Next;
      SELF.Hdiag := H_Next;
      SELF.Cost := Cost_Next;
      SELF.funEvals_ := FunEval_next;
      SELF.d := d_next;
      SELF.fnew_fold := fpre_fnext;
      SELF.t_ := t_new;
      SELF.dLegal := dlegalstep;
      SELF.ProgAlongDir := gtdprogress;
      SELF.optcond := optcond; // Check Optimality Condition
      SELF.lackprog1 := lack1; //Check for lack of progress 1
      SELF.lackprog2 := lack2; //Check for lack of progress 2
      SELF.exceedfuneval := evalimit;
      SELF := l;
    END;
    MinFRecord MF_dnotleg (MinFRecord l) := TRANSFORM
      SELF.d := d_next;
      SELF.dLegal := FALSE;
      SELF := l;
    END;
    MFreturn := PROJECT (inputp,MF(LEFT));
    MFndreturn := PROJECT (inputp,MF_dnotleg(LEFT));
    
   orig */
    
    MinFRecord MinF_tran (MinFRecord l) := TRANSFORM
    x_pre := l.x;
    g_pre := l.g;
    g_pre_ := ML.Mat.Scale (g_pre , -1);
    step_pre := l.old_steps;
    dir_pre := l.old_dirs;
    H_pre := l.Hdiag;
    f_pre := l.Cost;
    FunEval_pre := l.funEvals_;
    //HG_ is actually search direction in fromual 3.1
    HG_ :=  IF (coun = 1, O.Steepest_Descent (g_pre), O.lbfgs(g_pre_,Dir_pre, Step_pre,H_pre));//the result is the approximate inverse Hessian, multiplied by the gradient and it is in PBblas.layout format
    d := ML.DMat.Converted.FromPart2DS(HG_);
    dlegalstep := IsLegal (d);
    d_next := ML.Types.ToMatrix(d);
    //Directional Derivative : gtd = g'*d;
    //calculate gtd to be passed to the wolfe algortihm
    gtd := ML.Mat.Mul(ML.Mat.Trans((g_pre)),ML.Types.ToMatrix(d));
    //Check that progress can be made along direction : if gtd > -tolX then break!
    gtdprogress := IF (gtd[1].value > -1*tolX, FALSE, TRUE);
    // Select Initial Guess If coun =1 then t = min(1,1/sum(abs(g)));
    t := IF (coun = 1,MIN ([1, 1/SumABSg (ML.Types.FromMatrix (g_pre))]),1);
     //find point satisfiying wolfe
    //[t,f,g,LSfunEvals] = WolfeLineSearch(x,t,d,f,g,gtd,c1,c2,LS,25,tolX,debug,doPlot,1,funObj,varargin{:});
    t_neworig := WolfeLineSearch3(1, ML.Types.FromMatrix(x_pre), t, d, f_pre, ML.Types.FromMatrix(g_pre), gtd[1].value, 0.0001, 0.9, 25, 0.000000001, CostFunc_params, TrainData , TrainLabel, CostFunc , prows, pcols, Maxrows, Maxcols);
    t_new := wolfe_t_ext(t_neworig);
    g_Next := wolfe_gnew_ext(t_neworig); // := g_pre works here
    Cost_Next := wolfe_fnew_ext(t_neworig);
    FunEval_Wolfe := wolfe_funeval_ext(t_neworig);
    FunEval_next := FunEval_pre + FunEval_Wolfe;
    x_pre_updated :=  ML.Mat.Add((x_pre),ML.Mat.Scale(ML.Types.ToMatrix(d),t_new));
    x_Next := ML.Types.FromMatrix(x_pre_updated);
    fpre_fnext := Cost_Next-f_pre;
    //calculate new Hessian diag, dir and steps
    //Step_Next :=  O.lbfgsUpdate_corr (g_pre, g_Next, Step_pre);
    Step_Next := O.lbfgsUpdate_Stps (x_pre, x_pre_updated, g_pre, g_Next, Step_pre);
    //Dir_Next := O.lbfgsUpdate_corr(x_pre, x_pre_updated, Dir_pre);
    Dir_Next := O.lbfgsUpdate_Dirs (x_pre, x_pre_updated, g_pre, g_Next, Dir_pre);
    H_Next := O.lbfgsUpdate_Hdiag (x_pre, x_pre_updated, g_pre, g_Next, H_pre);
    optcond := OptimalityCond_mat (g_Next);
    lack1 := LackofProgress1 (d_next, t_new);
    lack2 := LackofProgress2 (fpre_fnext);
    evalimit := EvaluationLimit (FunEval_next);
    //ToReturn := x_Next_no + g_Nextno + Step_Nextno + Dir_Nextno + H_Nextno+  C_Nextno + FunEval_Nextno + d_Nextno + fpre_fnextno + t_newno + dlegalstepno + gtdprogressno;
    
      SELF.x := x_pre_updated;
      SELF.g := g_Next;
      SELF.old_steps := Step_Next;
      SELF.old_dirs := Dir_Next;
      SELF.Hdiag := H_Next;
      SELF.Cost := Cost_Next;
      SELF.funEvals_ := FunEval_next;
      SELF.d := d_next;
      SELF.fnew_fold := fpre_fnext;
      SELF.t_ := t_new;
      SELF.dLegal := dlegalstep;
      SELF.ProgAlongDir := gtdprogress;
      SELF.optcond := optcond; // Check Optimality Condition
      SELF.lackprog1 := lack1; //Check for lack of progress 1
      SELF.lackprog2 := lack2; //Check for lack of progress 2
      SELF.exceedfuneval := evalimit;
      SELF := l;
    END;
    
    
    MinFRecord MinF_dnotleg_tran (MinFRecord l) := TRANSFORM
    x_pre := l.x;
    g_pre := l.g;
    g_pre_ := ML.Mat.Scale (g_pre , -1);
    step_pre := l.old_steps;
    dir_pre := l.old_dirs;
    H_pre := l.Hdiag;
    f_pre := l.Cost;
    FunEval_pre := l.funEvals_;
    //HG_ is actually search direction in fromual 3.1
    HG_ :=  IF (coun = 1, O.Steepest_Descent (g_pre), O.lbfgs(g_pre_,Dir_pre, Step_pre,H_pre));//the result is the approximate inverse Hessian, multiplied by the gradient and it is in PBblas.layout format
    d := ML.DMat.Converted.FromPart2DS(HG_);
    dlegalstep := IsLegal (d);
    d_next := ML.Types.ToMatrix(d);
    SELF.d := d_next;
    SELF.dLegal := FALSE;
    SELF := l;
    END;
    MFreturn := PROJECT (inputp,MinF_tran(LEFT));
    MFndreturn := PROJECT (inputp,MinF_dnotleg_tran(LEFT));
    RETURN MFreturn;
    //RETURN IF(dlegalstep,MFreturn,MFndreturn); orig
  END;
  //updating step function


 // MinFstepout := MinFstep(ToPassMinF,1);
  // MinFstepout := LOOP(ToPassMinF, COUNTER <= MaxIter AND ROWS(LEFT)[1].dLegal AND ROWS(LEFT)[1].ProgAlongDir   
  // AND ~ROWS(LEFT)[1].optcond AND ~ROWS(LEFT)[1].lackprog1 AND ~ROWS(LEFT)[1].lackprog2 AND ~ROWS(LEFT)[1].exceedfuneval  , MinFstep(ROWS(LEFT),COUNTER)); orig

  MinFstepout := LOOP(ToPassMinF, COUNTER <= 1, MinFstep(ROWS(LEFT),COUNTER));
/*
  outrec := RECORD
    DATASET(Mat.Types.Element) x;
    REAL8 cost;
  END;
  output_xfinal_costfinal := PROJECT(MinFstepout, TRANSFORM(outrec, SELF := LEFT));
  output_x0_cost0 := DATASET([{ML.Types.ToMatrix(x0),90}],outrec);
  FinalResult := IF (IsInitialPointOptimal,output_x0_cost0,output_xfinal_costfinal ); orig */
  //RETURN FinalResult; orig
  RETURN MinFstepout;

  END;//END MinFUNC



EXPORT MinFUNC3(DATASET(Types.NumericField) x0, DATASET(Types.NumericField) CostFunc (DATASET(Types.NumericField) x, DATASET(Types.NumericField) CostFunc_params, DATASET(Types.NumericField) TrainData , DATASET(Types.NumericField) TrainLabel), DATASET(Types.NumericField) CostFunc_params, DATASET(Types.NumericField) TrainData , DATASET(Types.NumericField) TrainLabel, INTEGER MaxIter = 100, REAL8 tolFun = 0.00001, REAL8 TolX = 0.000000001, INTEGER maxFunEvals = 1000, INTEGER corrections = 100, prows=0, pcols=0, Maxrows=0, Maxcols=0) := FUNCTION
  //#option ('divideByZero', 'nan'); //In all operation I want to f or g get nan value if it is devided by zero
  //Functions used
  SumABSg (DATASET (Types.NumericField) g_temp) := FUNCTION
    r := RECORD
      t_RecordID id := 1 ;
      t_FieldNumber number := 1;
      t_FieldReal value := SUM(GROUP,ABS(g_temp.value));
    END;
    SumABSCol := TABLE(g_temp,r);
    RETURN SumABSCol[1].value;
  END;
  SumABSg_matrix (DATASET (Mat.Types.Element) g_temp) := FUNCTION
    r := RECORD
      Mat.Types.t_Index x := 1 ;
      Mat.Types.t_Index y := 1;
      Mat.Types.t_value value := SUM(GROUP,ABS(g_temp.value));
    END;
    SumABSCol := TABLE(g_temp,r);
    RETURN SumABSCol[1].value;
  END;
  // Check Optimality Condition: if sum(abs(g)) <= tolFun return True
  OptimalityCond (DATASET (Types.NumericField) g_temp) := FUNCTION
    sag := SumABSg (g_temp);
    RETURN IF( sag <= tolFun , TRUE, FALSE);
  END;
  //Check Optimality Condition for the Loop : if sum(abs(g)) <= tolFun return TRUE
  OptimalityCond_mat (DATASET (Mat.Types.Element) g_t) := FUNCTION
    sag := SumABSg_matrix (g_t);
    RETURN IF( sag <= tolFun , TRUE, FALSE);
  END;//END OptimalityCond_mat
  //Check for lack of progress 1: if sum(abs(t*d)) <= tolX return TRUE
  LackofProgress1 (DATASET (Mat.Types.Element) d_temp, REAL8 t_temp) := FUNCTION
    r := RECORD
      Mat.Types.t_Index x := 1 ;
      Mat.Types.t_Index y := d_temp.y;
      Mat.Types.t_Value value := SUM(GROUP,ABS(d_temp.value * t_temp));
    END;
    SumABSCol := TABLE(d_temp,r,d_temp.y);
    RETURN IF( SumABSCol[1].value <= tolX , TRUE, FALSE);
  END;
  //Check for lack of progress 2: if abs(f-f_old) < tolX return TRUE
  LackofProgress2 (REAL8 f_f_temp) := FUNCTION
    RETURN IF (ABS (f_f_temp) < tolX, TRUE, FALSE);
  END;
  //Check for going over evaluation limit
  //if funEvals*funEvalMultiplier > maxFunEvals return TRUE  (funEvalMultiplier=1)
  EvaluationLimit (INTEGER8 fun_temp) := FUNCTION
    RETURN IF (fun_temp > maxFunEvals, TRUE, FALSE);
  END;
  IsLegal (DATASET (Types.NumericField) inp) := FUNCTION //???to be defined
    RETURN TRUE;
  END;
  //the length of the parameters vector (for example in an neural network algorith, all the weight parameters are passed in ONE vector
  //to the optimization algorithm to get updated

  P := Max (x0, id); // the maximum matrix id in the parameters numericfield dataset
  ExtractGrad (DATASET(Types.NumericField) inp) := FUNCTION
    RETURN inp (id <= P);
  END;
  ExtractCost (DATASET(Types.NumericField) inp) := FUNCTION
    RETURN inp (id = (P+1))[1].value;
  END;
  //Evaluate Initial Point
  CostGrad0 := CostFunc (x0,CostFunc_params,TrainData, TrainLabel);

  g0 := ExtractGrad (CostGrad0);
  Cost0 := ExtractCost (CostGrad0);
  //Check the optimality of the initial point (if sum(abs(g)) <= tolFun)
  IsInitialPointOptimal := OptimalityCond (g0);

  //LBFGS Module
  O := Limited_Memory_BFGS (P, corrections);
  //initialize Hdiag,old_dir,old_steps, gradient and cost
  //The size of the old_dir and old_steps matrices are "number of parameters" * "number of corrections to store in memory (corrections)"
  // old_dir0 := Mat.Zeros(P, corrections);
  // old_steps0 := old_dir0;
  //New Matrix Generator
  Mat.Types.Element gen(UNSIGNED4 c, UNSIGNED4 NumRows, REAL8 v=0) := TRANSFORM
    SELF.x := ((c-1) % NumRows) + 1;
    SELF.y := ((c-1) DIV NumRows) + 1;
    SELF.value := v;
  END;
  // Initialize dirs and steps matrices
  old_dir0   := DATASET(P * corrections, gen(COUNTER, P));
  old_steps0 := DATASET(P * corrections, gen(COUNTER, P));
  Hdiag0 := 1;
  //initialize hessian diag as 1
  Hdiag0no := DATASET([{1,1,1,5}], Mat.Types.MUElement);
  //number of times the cost function is evaluated
  FunEval := 1; 
  //Perform up to a maximum of 'maxIter' descent steps:
  //put all the parameters that need to be sent to the step fucntion in a Mat.Types.MUElement format
  
  emptyE := DATASET([], Mat.Types.Element);
  MinFRecord := RECORD
    DATASET(Mat.Types.Element) x;
    DATASET(Mat.Types.Element) g;
    DATASET(Mat.Types.Element) old_steps;
    DATASET(Mat.Types.Element) old_dirs;
    REAL8 Hdiag;
    REAL8 Cost;
    INTEGER8 funEvals_;
    DATASET(Mat.Types.Element) d;
    REAL8 fnew_fold; //f_new-f_old in that iteration
    REAL8 t_;
    BOOLEAN dLegal;
    BOOLEAN ProgAlongDir; //Progress Along Directionno //Check that progress can be made along direction ( if gtd > -tolX)
    BOOLEAN optcond; // Check Optimality Condition
    BOOLEAN lackprog1; //Check for lack of progress 1
    BOOLEAN lackprog2; //Check for lack of progress 2
    BOOLEAN exceedfuneval; //Check for going over evaluation limit
  END;

  ToPassMinF := DATASET ([{ML.Types.ToMatrix(x0),ML.Types.ToMatrix(g0),old_steps0,old_dir0,Hdiag0,Cost0,FunEval,emptyE,100 +tolFun,1,TRUE,TRUE,FALSE,FALSE,FALSE,FALSE}],MinFRecord);


  MinFstep (DATASET (MinFRecord) inputp, INTEGER coun) := FUNCTION

    
     
    x_pre := inputp[1].x;
    g_pre := inputp[1].g;
    g_pre_ := ML.Mat.Scale (g_pre , -1);
    step_pre := inputp[1].old_steps;
    dir_pre := inputp[1].old_dirs;
    H_pre := inputp[1].Hdiag;
    f_pre := inputp[1].Cost;
    FunEval_pre := inputp[1].funEvals_;
    //HG_ is actually search direction in fromual 3.1
    HG_ :=  IF (coun = 1, O.Steepest_Descent (g_pre), O.lbfgs(g_pre_,Dir_pre, Step_pre,H_pre));//the result is the approximate inverse Hessian, multiplied by the gradient and it is in PBblas.layout format
    d := ML.DMat.Converted.FromPart2DS(HG_);
    dlegalstep := IsLegal (d);
    d_next := ML.Types.ToMatrix(d);
    //Directional Derivative : gtd = g'*d;
    //calculate gtd to be passed to the wolfe algortihm
    gtd := ML.Mat.Mul(ML.Mat.Trans((g_pre)),ML.Types.ToMatrix(d));
    //Check that progress can be made along direction : if gtd > -tolX then break!
    gtdprogress := IF (gtd[1].value > -1*tolX, FALSE, TRUE);
    // Select Initial Guess If coun =1 then t = min(1,1/sum(abs(g)));
    t := IF (coun = 1,MIN ([1, 1/SumABSg (ML.Types.FromMatrix (g_pre))]),1);
     //find point satisfiying wolfe
    //[t,f,g,LSfunEvals] = WolfeLineSearch(x,t,d,f,g,gtd,c1,c2,LS,25,tolX,debug,doPlot,1,funObj,varargin{:});
    t_neworig := WolfeLineSearch2(1, ML.Types.FromMatrix(x_pre), t, d, f_pre, ML.Types.FromMatrix(g_pre), gtd[1].value, 0.0001, 0.9, 25, 0.000000001, CostFunc_params, TrainData , TrainLabel, CostFunc , prows, pcols, Maxrows, Maxcols);
    t_new := t_neworig[1].t;
    g_Next := t_neworig[1].g_new;
    Cost_Next := t_neworig[1].f_new;
    FunEval_Wolfe := t_neworig[1].WolfeFunEval;
    FunEval_next := FunEval_pre + FunEval_Wolfe;
    x_pre_updated :=  ML.Mat.Add((x_pre),ML.Mat.Scale(ML.Types.ToMatrix(d),t_new));
    x_Next := ML.Types.FromMatrix(x_pre_updated);
    fpre_fnext := Cost_Next-f_pre;
    //calculate new Hessian diag, dir and steps
    //Step_Next :=  O.lbfgsUpdate_corr (g_pre, g_Next, Step_pre);
    Step_Next := O.lbfgsUpdate_Stps (x_pre, x_pre_updated, g_pre, g_Next, Step_pre);
    //Dir_Next := O.lbfgsUpdate_corr(x_pre, x_pre_updated, Dir_pre);
    Dir_Next := O.lbfgsUpdate_Dirs (x_pre, x_pre_updated, g_pre, g_Next, Dir_pre);
    H_Next := O.lbfgsUpdate_Hdiag (x_pre, x_pre_updated, g_pre, g_Next, H_pre);
    optcond := OptimalityCond_mat (g_Next);
    lack1 := LackofProgress1 (d_next, t_new);
    lack2 := LackofProgress2 (fpre_fnext);
    evalimit := EvaluationLimit (FunEval_next);
    //ToReturn := x_Next_no + g_Nextno + Step_Nextno + Dir_Nextno + H_Nextno+  C_Nextno + FunEval_Nextno + d_Nextno + fpre_fnextno + t_newno + dlegalstepno + gtdprogressno;
    MinFRecord MF (MinFRecord l) := TRANSFORM
      SELF.x := x_pre_updated;
      SELF.g := g_Next;
      SELF.old_steps := Step_Next;
      SELF.old_dirs := Dir_Next;
      SELF.Hdiag := H_Next;
      SELF.Cost := Cost_Next;
      SELF.funEvals_ := FunEval_next;
      SELF.d := d_next;
      SELF.fnew_fold := fpre_fnext;
      SELF.t_ := t_new;
      SELF.dLegal := dlegalstep;
      SELF.ProgAlongDir := gtdprogress;
      SELF.optcond := optcond; // Check Optimality Condition
      SELF.lackprog1 := lack1; //Check for lack of progress 1
      SELF.lackprog2 := lack2; //Check for lack of progress 2
      SELF.exceedfuneval := evalimit;
      SELF := l;
    END;
    MinFRecord MF_dnotleg (MinFRecord l) := TRANSFORM
      SELF.d := d_next;
      SELF.dLegal := FALSE;
      SELF := l;
    END;
    MFreturn := PROJECT (inputp,MF(LEFT));
    MFndreturn := PROJECT (inputp,MF_dnotleg(LEFT));
    

    
    RETURN MFreturn;
    //RETURN IF(dlegalstep,MFreturn,MFndreturn); orig
  END;
  //updating step function


 // MinFstepout := MinFstep(ToPassMinF,1);
  // MinFstepout := LOOP(ToPassMinF, COUNTER <= MaxIter AND ROWS(LEFT)[1].dLegal AND ROWS(LEFT)[1].ProgAlongDir   
  // AND ~ROWS(LEFT)[1].optcond AND ~ROWS(LEFT)[1].lackprog1 AND ~ROWS(LEFT)[1].lackprog2 AND ~ROWS(LEFT)[1].exceedfuneval  , MinFstep(ROWS(LEFT),COUNTER)); orig

  MinFstepout := LOOP(ToPassMinF, COUNTER <= 1, MinFstep(ROWS(LEFT),COUNTER));
/*
  outrec := RECORD
    DATASET(Mat.Types.Element) x;
    REAL8 cost;
  END;
  output_xfinal_costfinal := PROJECT(MinFstepout, TRANSFORM(outrec, SELF := LEFT));
  output_x0_cost0 := DATASET([{ML.Types.ToMatrix(x0),90}],outrec);
  FinalResult := IF (IsInitialPointOptimal,output_x0_cost0,output_xfinal_costfinal ); orig */
  //RETURN FinalResult; orig
  RETURN MinFstepout;

  END;//END MinFUNC3


END;// END Optimization2









