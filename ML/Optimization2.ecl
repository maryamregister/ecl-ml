//#option ('divideByZero', 'nan');
IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat;
IMPORT * FROM ML.Types;
IMPORT PBblas;
Layout_Cell := PBblas.Types.Layout_Cell;
Layout_Part := PBblas.Types.Layout_Part;

//Func : handle to the function we want to minimize it, its output should be the error cost and the error gradient
EXPORT Optimization2 (UNSIGNED4 prows=0, UNSIGNED4 pcols=0, UNSIGNED4 Maxrows=0, UNSIGNED4 Maxcols=0) := MODULE

  //polyinterp when the boundry values are provided
  EXPORT  polyinterp_both (REAL8 t_1, REAL8 f_1, REAL8 gtd_1, REAL8 t_2, REAL8 f_2, REAL8 gtd_2, REAL8 xminBound, REAL8 xmaxBound) := FUNCTION
    poly1 := FUNCTION
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
     RETURN out;
    END;//END poly1
    polResult := poly1;
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
  EXPORT WolfeLineSearch(INTEGER cccc, DATASET(Types.NumericField)x, REAL8 t, DATASET(Types.NumericField)d, REAL8 f, DATASET(Types.NumericField) g, REAL8 gtd, REAL8 c1=0.0001, REAL8 c2=0.9, INTEGER maxLS=25, REAL8 tolX=0.000000001,DATASET(Types.NumericField) CostFunc_params, DATASET(Types.NumericField) TrainData , DATASET(Types.NumericField) TrainLabel, DATASET(Types.NumericField) CostFunc (DATASET(Types.NumericField) x, DATASET(Types.NumericField) CostFunc_params, DATASET(Types.NumericField) TrainData , DATASET(Types.NumericField) TrainLabel), prows=0, pcols=0, Maxrows=0, Maxcols=0):=FUNCTION
    //initial parameters
    P_num := Max (x, id); //the length of the parameters vector (number of parameters)
    emptyE := DATASET([], Mat.Types.Element);
    LSiter := 0;
    Bracket1no := DATASET([{1,1,-1,10}], Mat.Types.MUElement); //the result of the bracketing algorithm
    Bracket2no := DATASET([{1,1,-1,11}], Mat.Types.MUElement); //the result of the bracketing algorithm
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
    ArmijoBacktrack4 (DATASET (Mat.Types.MUElement) inputpp) := FUNCTION // to be defined with recieving real parameters (should be a macro similar to this one)
      RETURN inputpp;
    END;

    WolfeBracketing ( Real8 fNew, Real8 fPrev, Real8 gtdNew, REAL8 gtdPrev, REAL8 tt, REAL8 tPrev, DATASET(Mat.Types.Element) gNew, DATASET(Mat.Types.Element) gPrev, UNSIGNED8 inputFunEval, UNSIGNED8 BrackLSiter) := FUNCTION
      SetBrackets (REAL8 t1, REAL8 t2, REAL8 fval1, REAL8 fval2, DATASET(Mat.Types.Element) gval1 , DATASET(Mat.Types.Element) gval2) := FUNCTION
        t1no := DATASET([{1,1,t1,10}], Mat.Types.MUElement); //the result of the bracketing algorithm
        t2no := DATASET([{1,1,t2,11}], Mat.Types.MUElement); //the result of the bracketing algorithm
        fval1no := DATASET([{1,1,fval1,12}], Mat.Types.MUElement); //the result of the bracketing algorithm
        fval2no := DATASET([{1,1,fval2,13}], Mat.Types.MUElement); //the result of the bracketing algorithm
        gval1no := Mat.MU.To (gval1,14); //the result of the bracketing algorithm
        gval2no := Mat.MU.To (gval2,15); //the result of the bracketing algorithm
        FEnochange := DATASET([{1,1,inputFunEval,7}], Mat.Types.MUElement);
        RETURN t1no + t2no + fval1no + fval2no + gval1no + gval2no +FEnochange;
      END;

      SetNewValues () := FUNCTION
        //t_prev = t;
        tPrevno := DATASET([{1,1,tt,6}], Mat.Types.MUElement);
        // minStep = t + 0.01*(t-temp);
        // maxStep = t*10;
        minstep := tt + 0.01* (tt-tPrev);
        maxstep := tt*10;
        //t = polyinterp([temp f_prev gtd_prev; t f_new gtd_new],doPlot,minStep,maxStep);
        newt := polyinterp_both (tPrev, fPrev,gtdPrev, tt, fNew, gtdNew, minstep, maxstep);
        newtno := DATASET([{1,1,newt,5}], Mat.Types.MUElement);
        // f_prev = f_new;
        // g_prev = g_new;
        // gtd_prev = gtd_new;
        fPrevno := DATASET([{1,1,fNew,1}], Mat.Types.MUElement);
        gPrevno := Mat.MU.To (gNew,3);
        gtdPrevno:= DATASET([{1,1,gtdNew,8}], Mat.Types.MUElement);
        //calculate fnew gnew gtdnew
        //xNew := ML.Types.FromMatrix (ML.Mat.Add(ML.Types.ToMatrix(x),ML.Mat.Scale(ML.Types.ToMatrix(d),newt)));
        xNew := calculate_xNew (d, newt);
        CostGradNew := CostFunc (xNew ,CostFunc_params,TrainData, TrainLabel);
        gNewwolfe := ExtractGrad (CostGradNew);
        fNewWolfe := ExtractCost (CostGradNew);
        gtdNewWolfe := ML.Mat.Mul (ML.Mat.Trans(ML.Types.ToMatrix(gNewwolfe)),ML.Types.ToMatrix(d));
        gNewwolfeno := Mat.MU.To (ML.Types.ToMatrix(gNewwolfe),4);
        fNewWolfeno := DATASET([{1,1,fNewWolfe,2}], Mat.Types.MUElement);
        gtdNewWolfeno := Mat.MU.To (gtdNewWolfe,9);
        FEno := DATASET([{1,1,inputFunEval + 1,7}], Mat.Types.MUElement);
        Rno := fPrevno + fNewWolfeno + gPrevno + gNewwolfeno + newtno + tPrevno + FEno + gtdPrevno + gtdNewWolfeno + Bracket1no + Bracket2no;
       RETURN Rno;
      END;

      //If the strong wolfe conditions satisfies then retun the final t or the bracket, otherwise do the next iteration
      //f_new > f + c1*t*gtd || (LSiter > 1 && f_new >= f_prev)
      con1 := (fNew > f + c1 * tt* gtd)| ((BrackLSiter > 1) & (fNew >= fPrev)) ;
      //abs(gtd_new) <= -c2*gtd
      con2 := ABS(gtdNew) <= (-1*c2*gtd);
      // gtd_new >= 0
      con3 := gtdNew >= 0;
      //
      BracketValues := IF (con1, SetBrackets (tPrev,tt,fPrev,fNew, gPrev, gNew), IF (con2,SetBrackets (tt,-1,fNew,-1, gNew, EmptyE),IF (con3, SetBrackets (tPrev,tt,fPrev,fNew, gPrev, gNew),SetNewValues ()) ));
      //If the conditions have been satisfied then only the final interval or final t is returned in the BracketValues, otherwise
      //a new t is evaluated and all the new values for f_new, f_prev, etc. are returned in the BracketValues
      RETURN BracketValues; 
      //RETURN (DATASET([{1,1,gtdNew,1}], Mat.Types.MUElement)); test
    END; // END WolfeBracketing
    //WI : Wolfe Interval
    WolfeZooming (DATASET (Mat.Types.MUElement) WI, INTEGER coun) := FUNCTION
      t_first  := Mat.MU.From (WI,10)[1].value;
      t_second := Mat.MU.From (WI,11)[1].value;
      f_first  := Mat.MU.From (WI,12)[1].value;
      f_second := Mat.MU.From (WI,13)[1].value;
      g_first  := Mat.MU.From (WI,14);
      g_second := Mat.MU.From (WI,15);
      gtd_first := ML.Mat.Mul (ML.Mat.Trans (g_first),ML.Types.ToMatrix(d));
      gtd_second := ML.Mat.Mul (ML.Mat.Trans (g_second),ML.Types.ToMatrix(d));
      //
      // Find High and Low Points in bracket
      LOt_i := IF (f_first < f_second, 10 , 11);
      LOf_i := LOt_i + 2;
      LOg_i := LOt_i + 4;
      HIt_i := -1 * LOt_i + 21;
      HIf_i := -1 * LOf_i + 25;
      HIg_i := -1 * LOg_i + 29;
      //
      LOt := Mat.MU.From (WI,LOt_i);
      HIt := Mat.MU.From (WI,HIt_i);
      LOf := Mat.MU.From (WI,LOf_i);
      HIf := Mat.MU.From (WI,HIf_i);
      LO_g := Mat.MU.From (WI,LOg_i);
      HIg := Mat.MU.From (WI,HIg_i);
      //
      // Compute new trial value
      //t = polyinterp([bracket(1) bracketFval(1) bracketGval(:,1)'*d bracket(2) bracketFval(2) bracketGval(:,2)'*d],doPlot);
      tTmp := polyinterp_noboundry (t_first, f_first, gtd_first[1].value, t_second, f_second, gtd_second[1].value);
      //Test that we are making sufficient progress
      insufProgress := (BOOLEAN)Mat.MU.From (WI,300)[1].value;
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
      insufProgress_new := IF (MainPCond, IF (PCond2, 0, 1) , 0);
      //
      // Evaluate new point with tZoom
      x_td := ML.Types.FromMatrix (ML.Mat.Add(ML.Types.ToMatrix(x),ML.Mat.Scale(ML.Types.ToMatrix(d),tZOOM)));
      //CG_New := CostFunc (x_td ,CostFunc_params,TrainData, TrainLabel); orig
      
CF_new_mine := DATASET ([  
 { 1  ,  1  ,  0.0014},
   { 2 ,   1  ,  0.0003},
    {3  ,  1  ,  0.0015},
   { 4,    1  ,  0.0002},
   { 5,    1  ,  0.0012},
   { 6,    1  ,  0.0002},
   { 7,    1  , -0.0002},
  {  8,    1  , -0.0002},
   { 9,    1 ,  -0.0002},
  { 10,    1  , -0.0002},
  { 11 ,   1 ,  -0.0001},
  { 12,    1 ,  -0.0002},
  { 13,    1 ,   0.0011},
   {14,    1 ,  -0.0011},
   {15,    1 ,   0.0003},
  { 16,    1 ,   0.0003},
   {17,    1 ,   0.0003},
  { 18,    1 ,   0.1398}],Types.NumericField);      
      
      
      // CostFunc_params_test := DATASET([{1, 11, 3},{2,11,0.1},{3,11,0.1}], Types.NumericField);
      // CFP := IF (cccc=11,CostFunc_params_test,CostFunc_params);
      // CG_New := CostFunc (x_td ,CFP,TrainData, TrainLabel);
      CG_New := IF (cccc=11,CF_new_mine, CostFunc (x_td ,CostFunc_params,TrainData, TrainLabel)); 
      
      gNew := ExtractGrad (CG_New);
      fNew := ExtractCost(CG_New);
      gtdNew := ML.Mat.Mul (ML.Mat.Trans(ML.Types.ToMatrix(gNew)),ML.Types.ToMatrix(d));
      inputZFunEval := Mat.MU.From (WI,7)[1].value;
      ZoomFunEvalno := DATASET([{1,1,inputZFunEval + 1,7}], Mat.Types.MUElement);
      SetIntervalIF1 := FUNCTION
        //bracket(HIpos) = t;
        bracket_HIt := DATASET([{1,1,tZOOM,HIt_i}], Mat.Types.MUElement);;
        //bracketFval(HIpos) = f_new;
        bracket_HIf := DATASET([{1,1,fNew,HIf_i}], Mat.Types.MUElement);
        //bracketGval(:,HIpos) = g_new;
        bracket_HIg := Mat.MU.To (ML.Types.ToMatrix(gNew),HIg_i);
        done := DATASET([{1,1,0,200}], Mat.Types.MUElement);
        RETURN bracket_HIt + bracket_HIf + bracket_HIg + WI(no = LOt_i) + WI (no = LOf_i) + WI (no = LOg_i) + done ;
      END;
      SetIntervalELSE1 := FUNCTION
       //bracket(LOpos) = t;
        bracket_LOt := DATASET([{1,1,tZOOM,LOt_i}], Mat.Types.MUElement);
        //bracketFval(LOpos) = f_new;
        bracket_LOf := DATASET([{1,1,fNew,LOf_i}], Mat.Types.MUElement);
        //bracketGval(:,LOpos) = g_new;
        bracket_LOg := Mat.MU.To (ML.Types.ToMatrix(gNew),LOg_i);
        done := DATASET([{1,1,0,200}], Mat.Types.MUElement);
        RETURN bracket_LOt + bracket_LOf + bracket_LOg + WI (no = HIt_i) + WI (no = HIf_i) + WI (no = HIg_i) + done;
      END;//yes
      SETIntervalELSE1_1 := FUNCTION
        //bracket(LOpos) = t;
        bracket_LOt := DATASET([{1,1,tZOOM,LOt_i}], Mat.Types.MUElement);
        //bracketFval(LOpos) = f_new;
        bracket_LOf := DATASET([{1,1,fNew,LOf_i}], Mat.Types.MUElement);
        //bracketGval(:,LOpos) = g_new;
        bracket_LOg := Mat.MU.To (ML.Types.ToMatrix(gNew),LOg_i);
        done := DATASET([{1,1,1,200}], Mat.Types.MUElement);
        RETURN bracket_LOt + bracket_LOf + bracket_LOg + WI (no = HIt_i) + WI (no = HIf_i) + WI (no = HIg_i) + done;
      END;//yes
      SETIntervalELSE1_2 := FUNCTION
        //bracket(LOpos) = t;
        bracket_LOt := DATASET([{1,1,tZOOM,LOt_i}], Mat.Types.MUElement);
        //bracketFval(LOpos) = f_new;
        bracket_LOf := DATASET([{1,1,fNew,LOf_i}], Mat.Types.MUElement);
        //bracketGval(:,LOpos) = g_new;
        bracket_LOg := Mat.MU.To (ML.Types.ToMatrix(gNew),LOg_i);
        // bracket(HIpos) = bracket(LOpos);
        bracket_HIt := DATASET([{1,1,LOt[1].value,HIt_i}], Mat.Types.MUElement);
        // bracketFval(HIpos) = bracketFval(LOpos);
        bracket_HIf := DATASET([{1,1,LOf[1].value,HIf_i}], Mat.Types.MUElement);;
        // bracketGval(:,HIpos) = bracketGval(:,LOpos);
        bracket_HIg := Mat.MU.To (LO_g,HIg_i);
        done := DATASET([{1,1,0,200}], Mat.Types.MUElement);
        RETURN bracket_LOt + bracket_LOf + bracket_LOg + bracket_HIt + bracket_HIf + bracket_HIg + done;
      END;//yes
      //IF f_new > f + c1*t*gtd || f_new >= f_LO
      ZoomCon1 := (fNew > f + c1 * tZoom * gtd) | (fNew >LOf[1].value);
      //if abs(gtd_new) <= - c2*gtd
      ZOOMCon1_1 := ABS (gtdNew[1].value) <= (-1 * c2 * gtd); 
      //gtd_new*(bracket(HIpos)-bracket(LOpos)) >= 0
      ZOOMCon1_2 := (gtdNew[1].value * (HIt[1].value - LOt[1].value)) >= 0; 

      ZOOOMResult := IF (ZoomCon1, SetIntervalIF1, (IF(ZOOMCon1_1, SETIntervalELSE1_1, IF (ZOOMCon1_2,SETIntervalELSE1_2, SetIntervalELSE1 ))));
      //~done && abs((bracket(1)-bracket(2))*gtd_new) < tolX
       ZOOMTermination :=( (Mat.MU.FROM (ZOOOMResult,200)[1].value = 0) & (ABS((gtdNew[1].value * (Mat.MU.From (ZOOOMResult,10)[1].value-Mat.MU.From (ZOOOMResult,11)[1].value)))<tolX) ) | (Mat.MU.FROM (ZOOOMResult,200)[1].value = 1);
      //ZOOMTermination :=( (Mat.MU.FROM (ZOOOMResult,200)[1].value = 0) & (ABS((gtdNew[1].value * (t_first-t_second)))<tolX) ) | (Mat.MU.FROM (ZOOOMResult,200)[1].value = 1); orig
      //ZOOMTermination_num := (INTEGER)ZOOMTermination; orig
      ZOOMTermination_num := IF(ZOOMTermination,1,0);
      ZOOMFinalResult := ZOOOMResult (no<200) + DATASET([{1,1,ZOOMTermination_num,200}], Mat.Types.MUElement)+ DATASET([{1,1,insufProgress_new,300}], Mat.Types.MUElement) +ZoomFunEvalno ;
      //RETURN ZOOMFinalResult; orig
      //t_first, f_first, gtd_first[1].value, t_second, f_second, gtd_second[1].value
      //RETURN IF(cccc=11,WI(no=12)+DATASET([{1,1,t_first,5},{2,1,f_first,6},{3,1,gtd_first[1].value,7},{4,1,t_second,8},{5,1,f_second,9},{6,1,gtd_second[1].value ,10}], Mat.Types.MUElement)+DATASET([{1,1,tTmp,1}], Mat.Types.MUElement) +DATASET([{1,1,IF(ZoomCon1,3.1,4.1),2}], Mat.Types.MUElement)+DATASET([{1,1,IF(ZOOMCon1_1,3.1,4.1),3}], Mat.Types.MUElement)+DATASET([{1,1,IF(ZOOMCon1_2,3.1,4.1),4}], Mat.Types.MUElement) , ZOOMFinalResult);
//RETURN IF (cccc=11, DATASET([{2,1,8,60},{5,1,8,90},{4,3,tTmp,10}], Mat.Types.MUElement), ZOOMFinalResult); 
// RETURN IF (cccc=11, DATASET([{2,1,f_first,60},{5,1,f_second,90},{4,3,tTmp,10}], Mat.Types.MUElement), ZOOMFinalResult); // 0 0 0.3236 
//IF (cccc=11, DATASET([{50,1,(REAL8)WI(no=12)[1].value,60}], Mat.Types.MUElement), ZOOMFinalResult);  0
WI12 := WI(no=12);
RETURN WI;
//RETURN DATASET ([{1,1,WI(no=12)[1].value,10}],Mat.Types.MUElement);
//RETURN WI(n=12)+ DATASET ([{1,1,WI(no=12)[1].value,10}],Mat.Types.MUElement);
//IF(COUNT(ds) > 0, ds[1].x, 0);
//RETURN IF (cccc=11,klk+ DATASET([{50,1,klk[1].value,60}], Mat.Types.MUElement), ZOOMFinalResult); w 
//RETURN IF (cccc=11,DATASET([{40,2,ttmp,30}], Mat.Types.MUElement), ZOOMFinalResult);//W20151219-163059 , klk+klk[1].value W20151219-153356
//RETURN IF (cccc=11,DATASET([{40,2,WI(no=12)[1].value,30}], Mat.Types.MUElement), ZOOMFinalResult);
//RETURN IF (cccc=11,WI(no=12)+DATASET([{40,2,WI(no=12)[1].value,30}], Mat.Types.MUElement), ZOOMFinalResult);
//Return DATASET([{40,2,10000*WI(no=12)[1].value,30}], Mat.Types.MUElement);
//RETURN IF (cccc=11, WI(no=12), ZOOMFinalResult);  W
      //RETURN Mat.MU.To (ML.Types.ToMatrix(CG_New),1);
      //RETURN DATASET([{1,1,tTmp,1}], Mat.Types.MUElement); 0
      
    END;// END WolfeZooming
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
    REAL8 funEvals_;
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
    
    //prepare the parameters to be passed to the bracketing algorithm
    f_prevno := DATASET([{1,1,f_prev,1}], Mat.Types.MUElement);
    f_newno := DATASET([{1,1,f_new,2}], Mat.Types.MUElement);
    g_prevno := Mat.MU.To (ML.Types.ToMatrix(g_prev),3);
    g_newno := Mat.MU.To (ML.Types.ToMatrix(g_new),4);
    tno := DATASET([{1,1,t,5}], Mat.Types.MUElement);
    t_prevno := DATASET([{1,1,t_prev,6}], Mat.Types.MUElement);
    funEvalsno := DATASET([{1,1,funEvals,7}], Mat.Types.MUElement);
    gtd_prevno := DATASET([{1,1,gtd_prev,8}], Mat.Types.MUElement);
    gtd_newno := DATASET([{1,1,gtd_new,9}], Mat.Types.MUElement);
    //at the begining no interval or final t value is found so the assigned values to Bracket1no and Bracket2no are -1
    //In each iteration these two values are checked to see if the interval is found (both should be ~-1) or the final t is found (just the first one should be ~-1)
    // f_prev 1
    // f_new 2
    // g_prev 3
    // g_new 4
    // t 5
    // t_prev 6
    // funeval 7
    // gtd_prev 8
    // gtd_new 9
    // bracket1 10
    // bracket2 11
    //Topass := f_prevno + f_newno + g_prevno + g_newno + tno + t_prevno + funEvalsno + gtd_prevno + gtd_newno + Bracket1no + Bracket2no + Mat.MU.To (ML.Types.ToMatrix(x_new),103);
    Topass := f_prevno + f_newno + g_prevno + g_newno + tno + t_prevno + funEvalsno + gtd_prevno + gtd_newno + Bracket1no + Bracket2no ;
    
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
          SELF.c := coun;
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
          SELF.c := coun;
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
          SELF.c := coun;
          SELF := l;
        END;
        con1_3_output := PROJECT(inputp,bracketing_con1_3(LEFT));
        con2_output := PROJECT(inputp,bracketing_con2(LEFT));
        Nocon_output := PROJECT(inputp,bracketing_Nocon(LEFT));
        RETURN IF (con1, con1_3_output, IF (con2, con2_output, IF (con3, con1_3_output, Nocon_output)));
      END;//END WolfeStep
      ArmijoStep := FUNCTION
        RETURN inputp;
      END;// END ArmijoStep
      WolfeBracket := WolfeStep;
      ArmijoBracket := ArmijoStep;
      RETURN IF(AreTheyLegal,ArmijoBracket,WolfeBracket);
    END; // END BracketingStep
    
    BracketingResult := LOOP(ToPass_bracketing, COUNTER <= maxLS AND ROWS(LEFT)[1].bracket1_ = -1, BracketingStep(ROWS(LEFT),COUNTER));
    
    Bracketing (DATASET (Mat.Types.MUElement) inputp, INTEGER coun) := FUNCTION
      fi_prev :=  Mat.MU.From (inputp,1);
      fi_new := Mat.MU.From (inputp,2);
      gi_prev :=  Mat.MU.From (inputp,3);
      gi_new := Mat.MU.From (inputp,4);
      ti :=  Mat.MU.From (inputp,5);
      ti_prev :=  Mat.MU.From (inputp,6);
      FunEvalsi := Mat.MU.From (inputp,7);
      gtdi_prev := Mat.MU.From (inputp,8);
      gtdi_new := Mat.MU.From (inputp,9);
      AreTheyLegal := IsNotLegal(fi_new) | IsNotLegal(gi_new);
      //armijo only returns final t results and then the loop will stop becasue bracket1 would be ~-1 nad the wolfelinesearch has to return
      WolfeH := WolfeBracketing ( fi_new[1].value, fi_prev[1].value, gtdi_new[1].value, gtdi_prev[1].value, ti[1].value, ti_prev[1].value, gi_new, gi_prev, FunEvalsi[1].value, (coun-1));
      Bracketing_Result := IF (AreTheyLegal, ArmijoBacktrack4(inputp), WolfeH );
      tobereturn := Bracketing_Result + DATASET([{1,1,coun-1,100}], Mat.Types.MUElement);
      RETURN tobereturn;  
    END;
    Bracketing_Result := LOOP(Topass, COUNTER <= maxLS AND Mat.MU.From (ROWS(LEFT),10)[1].value = -1, Bracketing(ROWS(LEFT),COUNTER));
    //Bracketing_Result := LOOP(Topass, (COUNTER <= maxLS) AND ((ROWS(LEFT)(no=10)[1].value) = -1), Bracketing(ROWS(LEFT),COUNTER));
    //Bracketing_Result := Bracketing(Topass,1); //the problem is loop
    //Bracketing_Result := LOOP(Topass, COUNTER <= 1 , Bracketing(ROWS(LEFT),COUNTER));
    
    
    //FoundInterval := Bracketing_Result (no = 10) + Bracketing_Result (no = 11) + Bracketing_Result (no = 12) + Bracketing_Result (no = 13) + Bracketing_Result (no = 14) + Bracketing_Result (no = 15); orig
    FoundInterval := Bracketing_Result (no = 10 OR no=11 OR no=12 OR no=13 OR no=14 OR no=15);
    alakiha := Bracketing_Result (no = 11 OR no=10 OR no=7 OR no=12 OR no=13 OR no =14 OR no=15);
    
    //FinaltInterval := Bracketing_Result (no = 10) + Bracketing_Result (no = 12) + Bracketing_Result (no = 14) + Bracketing_Result (no = 7); orig
    FinaltInterval := Bracketing_Result (no = 10 OR no=12 OR no=14 OR no=7);
    Interval_Found := Mat.MU.From (Bracketing_Result,10)[1].value != -1 AND Mat.MU.From (Bracketing_Result,11)[1].value !=-1;
    final_t_found := Mat.MU.From (Bracketing_Result,10)[1].value != -1 AND Mat.MU.From (Bracketing_Result,11)[1].value =-1;
    ItrExceedInterval := DATASET([{1,1,0,10},
    {1,1,Mat.MU.From (Bracketing_Result,5)[1].value ,11},
    {1,1,f ,12},
    {1,1,Mat.MU.From (Bracketing_Result,2)[1].value ,13}
    ], Mat.Types.MUElement) + Mat.MU.To (ML.Types.ToMatrix(g),14) + Mat.MU.To (Mat.MU.FROM(Bracketing_Result,4),15) + Bracketing_Result (no = 7);
    //
    Zoom_Max_itr_tmp :=  maxLS - Mat.MU.From (Bracketing_Result,100)[1].value;
    Zoom_Max_Itr := IF (Zoom_Max_itr_tmp >0, Zoom_Max_itr_tmp, 0);
    
   TOpassZOOM :=  Bracketing_Result (no=10 OR no = 11 OR no=12 OR no=13 OR no=14 OR no=15 OR no=7) + DATASET([{1,1,0,200}], Mat.Types.MUElement) + DATASET([{1,1,0,300}], Mat.Types.MUElement);
   
   //TOpassZOOM := FoundInterval + DATASET([{1,1,0,200}], Mat.Types.MUElement) + DATASET([{1,1,0,300}], Mat.Types.MUElement) + Bracketing_Result (no = 7); orig// pass the found interval + {zoomtermination=0} to Zoom LOOP +insufficientProgress+FunEval
   ZOOMInterval := LOOP(TOpassZOOM, COUNTER <= Zoom_Max_Itr AND Mat.MU.From (ROWS(LEFT),200)[1].value = 0, WolfeZooming(ROWS(LEFT), COUNTER));
   // ZOOMInterval := IF (cccc=11, WolfeZooming(TOpassZOOM, 1),LOOP(TOpassZOOM, COUNTER <= Zoom_Max_Itr AND Mat.MU.From (ROWS(LEFT),200)[1].value = 0, WolfeZooming(ROWS(LEFT), COUNTER)));
    
    //ZOOMInterval := WolfeZooming(TOpassZOOM, 1);
    //ZOOMInterval := LOOP(TOpassZOOM, COUNTER <= Zoom_Max_Itr , WolfeZooming(ROWS(LEFT), COUNTER));
    //ZOOMInterval := LOOP(TOpassZOOM,  1, WolfeZooming(ROWS(LEFT), COUNTER));
    FinalBracket := IF (final_t_found, FinaltInterval, IF (Interval_Found,ZOOMInterval,ItrExceedInterval));
    //The final t value is the t value which has the lowest f value
    WolfeT1 := DATASET([{1,1,FinalBracket(no=10)[1].value,1},
    {1,1,FinalBracket(no=12)[1].value,2},
    {1,1,FinalBracket(no=7)[1].value,4}], Mat.Types.MUElement) + Mat.MU.To (Mat.MU.FROM (FinalBracket,14),3) ;
    WolfeT2 := DATASET([{1,1,FinalBracket(no=11)[1].value,1},
    {1,1,FinalBracket(no=13)[1].value,2},
    {1,1,FinalBracket(no=7)[1].value,4}], Mat.Types.MUElement) + Mat.MU.To (Mat.MU.FROM (FinalBracket,15),3);
    //WolfeOut := IF (final_t_found | (FinalBracket(no=12)[1].value < FinalBracket(no=13)[1].value), WolfeT1, WolfeT2); orig
    //WOlfeOut := IF (cccc=11, ZOOMInterval, IF (final_t_found | (FinalBracket(no=12)[1].value < FinalBracket(no=13)[1].value), WolfeT1, WolfeT2));
    Wolfeout := Bracketing_Result;
    //WOlfeOut := IF (cccc=11, Bracketing_Result (no = 11 OR no=10 OR no=7 OR no=12 OR no=13), IF (final_t_found | (FinalBracket(no=12)[1].value < FinalBracket(no=13)[1].value), WolfeT1, WolfeT2));
FoundInterval_funcount := Bracketing_Result(no = 11 OR no=10 OR no=7 OR no=12 OR no=13 OR no =14 OR no=15);
//W // WOlfeOut := IF (cccc=11, DATASET([{1,1,0,200}], Mat.Types.MUElement) + DATASET([{1,1,0,300}], Mat.Types.MUElement), IF (final_t_found | (FinalBracket(no=12)[1].value < FinalBracket(no=13)[1].value), WolfeT1, WolfeT2));
topasszoomme := Bracketing_Result (no=10 OR no = 11 OR no=12 OR no=13 OR no=14 OR no=15 OR no=7) + DATASET([{1,1,0,200}], Mat.Types.MUElement) + DATASET([{1,1,0,300}], Mat.Types.MUElement);
//N W // WOlfeOut := IF (cccc=11,  topasszoomme, IF (final_t_found | (FinalBracket(no=12)[1].value < FinalBracket(no=13)[1].value), WolfeT1, WolfeT2));

g11 := DATASET ([{1   , 1  ,  0.0014},
   { 2  ,  1  ,  0.0003},
   { 3  ,  1  ,  0.0015},
   { 4   , 1   , 0.0002},
   { 5  ,  1   , 0.0012},
    {6  ,  1  ,  0.0002},
    {7  ,  1  , -0.0002},
    {8  ,  1  , -0.0002},
    {9   , 1  , -0.0002},
   {10 ,   1  , -0.0002},
   {11  ,  1 ,  -0.0001},
   {12  ,  1  , -0.0002},
   {13  ,  1  ,  0.0011},
   {14   , 1  , -0.0011},
   {15   , 1  ,  0.0003},
   {16  ,  1  ,  0.0003},
   {17   , 1 ,   0.0003}],Mat.Types.Element);
WolfeOut11:= DATASET([{1,1,  0.3236,1},
    {1,1, 0.1398,2},
    {1,1,2,4}], Mat.Types.MUElement) + Mat.MU.To (g11,3) ;
    
//WOlfeOut := IF (cccc=11,WolfeOut11 , IF (final_t_found | (FinalBracket(no=12)[1].value < FinalBracket(no=13)[1].value), WolfeT1, WolfeT2));    
    
//WOlfeOut := IF (cccc=11,  Bracketing_Result (no=10 OR no = 11 OR no=12 OR no=13 OR no=14 OR no=15 OR no=7) + DATASET([{1,1,0,200}], Mat.Types.MUElement) + DATASET([{1,1,0,300}], Mat.Types.MUElement), IF (final_t_found | (FinalBracket(no=12)[1].value < FinalBracket(no=13)[1].value), WolfeT1, WolfeT2));
     //W20151204-102550 N//WOlfeOut := IF (cccc=11, Bracketing_Result(no in [10,11,12,13,14,15,7]), IF (final_t_found | (FinalBracket(no=12)[1].value < FinalBracket(no=13)[1].value), WolfeT1, WolfeT2));
     //W20151203-152637 W//WOlfeOut := IF (cccc=11, Bracketing_Result(no = 11 OR no=10 OR no=7 OR no=12 OR no=13 OR no =14 OR no=15), IF (final_t_found | (FinalBracket(no=12)[1].value < FinalBracket(no=13)[1].value), WolfeT1, WolfeT2));
    //W20151203-144738 N//WOlfeOut := IF (cccc=11, Bracketing_Result (no = 10 OR no=11 OR no=12 OR no=13 OR no=14 OR no=15 OR no=7), IF (final_t_found | (FinalBracket(no=12)[1].value < FinalBracket(no=13)[1].value), WolfeT1, WolfeT2));
    //W20151203-195832 N//WOlfeOut := IF (cccc=11, Bracketing_Result(no = 11)+ Bracketing_Result( no=10 ) + Bracketing_Result( no=7 ) + Bracketing_Result( no=12 )+Bracketing_Result( no=13 )+Bracketing_Result( no =14 )+ Bracketing_Result( no=15), IF (final_t_found | (FinalBracket(no=12)[1].value < FinalBracket(no=13)[1].value), WolfeT1, WolfeT2));
    
    // br10:= Bracketing_Result (no = 10);
    // br11:= Bracketing_Result (no = 11);
    // br := br10 + br11;
    // WOlfeOut := IF (cccc=11, br, IF (final_t_found | (FinalBracket(no=12)[1].value < FinalBracket(no=13)[1].value), WolfeT1, WolfeT2));
    //WolfeOut := Bracketing_Result;
    //WolfeOut := Bracketing_Result;
    //Topass2 := f_prevno +  f_newno+ g_prevno  + tno + t_prevno +  gtd_prevno  + Bracket1no + Bracket2no;
    //topass3 := f_prevno + f_newno + g_prevno + g_newno + tno + t_prevno + funEvalsno + gtd_prevno + gtd_newno + Bracket1no + Bracket2no ;
    //WolfeOut := f_prevno + f_newno + g_prevno + g_newno + tno + t_prevno + funEvalsno + gtd_prevno + gtd_newno + Bracket1no + Bracket2no ;
    //WolfeOut := WolfeT1 + WolfeT2;
    //WolfeOut := ZOOMInterval;
    AppendID(WolfeOut, id, WolfeOut_id);
    ToField (WolfeOut_id, WolfeOut_id_out, id, 'x,y,value,no');//WolfeOut_id_out is the numeric field format of WolfeOut
    RETURN WolfeOut_id_out;
   // RETURN DATASET([{1,1,Zoom_Max_Itr,200}], Mat.Types.MUElement) ;
  // RETURN ZOOMInterval;
  END;// END WolfeLineSearch
  
   EXPORT WolfeLineSearch2(INTEGER cccc, DATASET(Types.NumericField)x, REAL8 t, DATASET(Types.NumericField)d, REAL8 f, DATASET(Types.NumericField) g, REAL8 gtd, REAL8 c1=0.0001, REAL8 c2=0.9, INTEGER maxLS=25, REAL8 tolX=0.000000001,DATASET(Types.NumericField) CostFunc_params, DATASET(Types.NumericField) TrainData , DATASET(Types.NumericField) TrainLabel, DATASET(Types.NumericField) CostFunc (DATASET(Types.NumericField) x, DATASET(Types.NumericField) CostFunc_params, DATASET(Types.NumericField) TrainData , DATASET(Types.NumericField) TrainLabel), prows=0, pcols=0, Maxrows=0, Maxcols=0):=FUNCTION
    //initial parameters
    P_num := Max (x, id); //the length of the parameters vector (number of parameters)
    emptyE := DATASET([], Mat.Types.Element);
    LSiter := 0;
    Bracket1no := DATASET([{1,1,-1,10}], Mat.Types.MUElement); //the result of the bracketing algorithm
    Bracket2no := DATASET([{1,1,-1,11}], Mat.Types.MUElement); //the result of the bracketing algorithm
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
        //RETURN IF (con1, con1_3_output, IF (con2, con2_output, IF (con3, con1_3_output, Nocon_output))); orig
        RETURN Nocon_output;

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
      //RETURN IF(AreTheyLegal,ArmijoBracket,WolfeBracket); orig
      RETURN WolfeBracket;
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
    //BracketingResult := LOOP(ToPass_bracketing, COUNTER <= maxLS AND ROWS(LEFT)[1].bracket1_ = -1, BracketingStep(ROWS(LEFT),COUNTER)); orig
    //BracketingResult :=  BracketingStep(ToPass_bracketing,1);
    BracketingResult := LOOP(ToPass_bracketing,COUNTER <= 1, BracketingStep(ROWS(LEFT),COUNTER)); 
    interval_found := BracketingResult[1].bracket1_ != -1 AND BracketingResult[1].bracket2_  !=-1;
    final_t_found := BracketingResult[1].bracket1_ != -1 AND BracketingResult[1].bracket2_  =-1;
    Zoom_Max_itr_tmp :=  maxLS - BracketingResult[1].c;
    Zoom_Max_Itr := IF (Zoom_Max_itr_tmp >0, Zoom_Max_itr_tmp, 0);
    
    
    ToPass_Zooming := PROJECT(BracketingResult, TRANSFORM(ZoomingRecord, SELF := LEFT));
    ZoomingResult := LOOP(ToPass_Zooming, COUNTER <= Zoom_Max_Itr AND ~ROWS(LEFT)[1].done AND ~ROWS(LEFT)[1].break, ZoomingStep(ROWS(LEFT),COUNTER));
    OutputRecord := RECORD
      REAL8 t;
      REAL8 f_new;
      DATASET(Mat.Types.Element) g_new;
      INTEGER8 WolfeFunEval;
    END;
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
    RETURN BracketingResult;
    
  END;// END WolfeLineSearch2
  


END;// END Optimization2









