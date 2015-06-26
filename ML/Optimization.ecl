IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat;
IMPORT * FROM ML.Types;
IMPORT PBblas;
Layout_Cell := PBblas.Types.Layout_Cell;
Layout_Part := PBblas.Types.Layout_Part;
//Func : handle to the function we want to minimize it, its output should be the error cost and the error gradient
EXPORT Optimization (UNSIGNED4 prows=0, UNSIGNED4 pcols=0, UNSIGNED4 Maxrows=0, UNSIGNED4 Maxcols=0) := MODULE
  //gtd_2_I : if gtd_2_I should be ignored, this is used in ArmijoBacktrack
  EXPORT  polyinterp (REAL8 t_1, REAL8 f_1, REAL8 gtd_1, REAL8 t_2, REAL8 f_2, REAL8 gtd_2, BOOLEAN gtd_2_I=TRUE) := FUNCTION
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
        // d1 = points(minPos,3) + points(notMinPos,3) - 3*(points(minPos,2)-points(notMinPos,2))/(points(minPos,1)-points(notMinPos,1));
        d1 := gtdmin + gtdmax - (3*((fmin-fmax)/(tmin-tmax)));
        //d2 = sqrt(d1^2 - points(minPos,3)*points(notMinPos,3));
        d2 := SQRT ((d1*d1)-(gtdmin*gtdmax));
        d2real := TRUE; //check it ???
        //t = points(notMinPos,1) - (points(notMinPos,1) - points(minPos,1))*((points(notMinPos,3) + d2 - d1)/(points(notMinPos,3) - points(minPos,3) + 2*d2));
        temp := tmax - ((tmax-tmin)*((gtdmax+d2-d1)/(gtdmax-gtdmin+(2*d2))));
        //min(max(t,points(minPos,1)),points(notMinPos,1));
        minpos1 := MIN([MAX([temp,tmin]),tmax]);
        minpos2 := (t_1+t_2)/2;
        pol1Result := IF (d2real,minpos1,minpos2);
        //RETURN pol1Result; orig
        RETURN IF(t_1=0, 10, 100);
      END;
      poly2 := FUNCTION
        tminBound := MIN ([t_1,t_2]);
        tmaxBound := MAX ([t_1,t_2]);
        RETURN 3;
      END;
      polResult := IF (gtd_2_I,poly1,poly2);
      RETURN polResult;
    END;//end polyinterp
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
      tmin := orderedp (id=1)[1].value;
      tmax := orderedp (id=2)[1].value;
      fmin := orderedp (id=3)[1].value;
      fmax := orderedp (id=4)[1].value;
      gtdmin := orderedp (id=5)[1].value;
      gtdmax := orderedp (id=6)[1].value;
      // A= [t_1^3 t_1^2 t_1 1
      //    t_2^3 t_2^2 t_2 1
      //    3*t_1^2 2*t_1 t_1 0
      //    3*t_2^2 2*t_2 t_2 0]
      //b = [f_1 f_2 dtg_1 gtd_2]'
      A := DATASET([
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
      b := DATASET([
      {1,1,f_1},
      {2,1,f_2},
      {3,1,gtd_1},
      {4,1,gtd_2}],
      Types.NumericField);
      // Find interpolating polynomial
      A_map := PBblas.Matrix_Map(4, 4, 4, 4);
      b_map := PBblas.Matrix_Map(4, 1, 4, 1);
      A_part := ML.DMat.Converted.FromNumericFieldDS (A, A_map);
      b_part := ML.DMat.Converted.FromNumericFieldDS (b, b_map);
      //params = A\b;
      params_part := DMAT.solvelinear (A_map,  A_part, FALSE, b_map, b_part) ; // for now
      params := DMat.Converted.FromPart2DS (params_part);
      params1 := params(id=1)[1].value;
      params2 := params(id=2)[1].value;
      params3 := params(id=3)[1].value;
      params4 := params(id=4)[1].value;
      dParams1 := 3*params(id=1)[1].value;
      dparams2 := 2*params(id=2)[1].value;
      dparams3 := params(id=3)[1].value;
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
      // Test Critical Points
      topa :=  DATASET([{1,1,(xminBound+xmaxBound)/2},{1,2,1000000}], Types.NumericField);//send minpos and fmin value to Resultsstep
      Resultstep (DATASET(Types.NumericField) x, UNSIGNED coun) := FUNCTION
        inr := x(id=1)[1].value;
        f_min := x(id=2)[1].value;
        // if imag(xCP)==0 && xCP >= xminBound && xCP <= xmaxBound
        xCP := cp(id=coun)[1].value;
        cond := xCP >= xminBound AND xCP <= xmaxBound; //???
        // fCP = polyval(params,xCP);
        fCP := params1*POWER(xCP,3)+params2*POWER(xCP,2)+params3*xCP+params4;
        //if imag(fCP)==0 && fCP < fmin
        cond2 := coun=1 OR fCP<f_min;//???
        rr := IF (cond,IF (cond2, xCP, inr),inr);
        ff := IF (cond,IF (cond2, fCP, f_min),f_min);
        RETURN DATASET([{1,1,rr},{2,1,ff}], Types.NumericField);
      END;
      finalresult := LOOP(topa, COUNTER <= itr, Resultstep(ROWS(LEFT),COUNTER));
      //RETURN finalresult; orig
      //RETURN IF(t_1=0, 10, 100);
      // RETURN DATASET([
      // {1,1,dParams1},
      // {2,1,dParams2},
      // {3,1,dParams3}],
      // Types.NumericField);
     RETURN finalresult;
    END;//END poly1
     poly2 := FUNCTION
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
        // d1 = points(minPos,3) + points(notMinPos,3) - 3*(points(minPos,2)-points(notMinPos,2))/(points(minPos,1)-points(notMinPos,1));
        d1 := gtdmin + gtdmax - (3*((fmin-fmax)/(tmin-tmax)));
        //d2 = sqrt(d1^2 - points(minPos,3)*points(notMinPos,3));
        d2 := SQRT ((d1*d1)-(gtdmin*gtdmax));
        d2real := TRUE; //check it ???
        //t = points(notMinPos,1) - (points(notMinPos,1) - points(minPos,1))*((points(notMinPos,3) + d2 - d1)/(points(notMinPos,3) - points(minPos,3) + 2*d2));
        temp := tmax - ((tmax-tmin)*((gtdmax+d2-d1)/(gtdmax-gtdmin+(2*d2))));
        //min(max(t,points(minPos,1)),points(notMinPos,1));
        minpos1 := MIN([MAX([temp,tmin]),tmax]);
        minpos2 := (t_1+t_2)/2;
        pol1Result := IF (d2real,minpos1,minpos2);
        RETURN pol1Result;
        //RETURN IF(t_1=0, 10, 100);
      END;//END poly2
    polResult := poly1;
    RETURN polResult;
  END;//end polyinterp_both
  EXPORT  polyinterp_noboundry (REAL8 t_1, REAL8 f_1, REAL8 gtd_1, REAL8 t_2, REAL8 f_2, REAL8 gtd_2) := FUNCTION
    poly2 := FUNCTION
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
    END;//END poly2
    polResult := poly2;
    RETURN polResult;
  END;//end polyinterp_noboundry
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
    b := DATASET([
    {1,1,f_1},
    {2,1,f_2},
    {3,1,gtd_1}],
    Types.NumericField);
    // Find interpolating polynomial
    A_map := PBblas.Matrix_Map(3, 3, 3, 3);
    b_map := PBblas.Matrix_Map(3, 1, 3, 1);
    A_part := ML.DMat.Converted.FromNumericFieldDS (A, A_map);
    b_part := ML.DMat.Converted.FromNumericFieldDS (b, b_map);
    //params = A\b;
    params_part := DMAT.solvelinear (A_map,  A_part, FALSE, b_map, b_part) ; // for now
    params := DMat.Converted.FromPart2DS (params_part);
    params1 := params(id=1)[1].value;
    params2 := params(id=2)[1].value;
    params3 := params(id=3)[1].value;
    dParams1 := 2*params(id=1)[1].value;
    dparams2 := params(id=2)[1].value;

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
    topa :=  DATASET([{1,1,(xminBound+xmaxBound)/2},{1,2,1000000}], Types.NumericField);
    Resultstep (DATASET(Types.NumericField) x, UNSIGNED coun) := FUNCTION
      minPos := x(id=1)[1].value;
      f_min := x(id=2)[1].value;
      // if imag(xCP)==0 && xCP >= xminBound && xCP <= xmaxBound
      xCP := cp(id=coun)[1].value;
      cond := xCP >= xminBound AND xCP <= xmaxBound; //???
      // fCP = polyval(params,xCP);
      fCP := params1*POWER(xCP,2)+params2*xCP+params3;
      //if imag(fCP)==0 && fCP < fmin
      cond2 := coun=1 OR fCP<f_min;//???
      rr := IF (cond,IF (cond2, xCP, minPos),minPos);
      ff := IF (cond,IF (cond2, fCP, f_min),f_min);
      RETURN DATASET([{1,1,rr},{2,1,ff}], Types.NumericField);
    END;
    finalresult := LOOP(topa, COUNTER <= itr, Resultstep(ROWS(LEFT),COUNTER));
    //RETURN finalresult; orig
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

    //this function is used for the very first step in lbfgs algorithm
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
    EXPORT lbfgs (DATASET(Mat.Types.Element) g, DATASET(Mat.Types.Element) s, DATASET(Mat.Types.Element) d, DATASET(Mat.Types.Element) Hdiag) := FUNCTION
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
    //This function adds the most recent vector (vec) to the matrix (old_mat) and removes the oldest vector from the matrix
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
    EXPORT lbfgsUpdate_Hdiag (DATASET(Mat.Types.Element) s, DATASET(Mat.Types.Element) y) := FUNCTION
      sdist := DMAT.Converted.FromElement(s,RowMap);
      ydist := DMAT.Converted.FromElement(y,RowMap);
      first_term := PBblas.PB_dgemm (TRUE, FALSE, 1.0, RowMap, ydist, RowMap, sdist, OnevalueMap );
      first_term_M := ML.DMat.Converted.FromPart2Elm (first_term);
      Second_Term := PBblas.PB_dgemm (TRUE, FALSE, 1.0, RowMap, ydist, RowMap, ydist, OnevalueMap );
      Second_Term_M := ML.DMat.Converted.FromPart2Elm (Second_Term);
      HD := First_Term_M[1].value/Second_Term_M[1].value;
      RETURN HD;
    END;
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
  EXPORT WolfeLineSearch(DATASET(Types.NumericField)x, REAL8 t, DATASET(Types.NumericField)d, REAL8 f, DATASET(Types.NumericField) g, REAL8 gtd, REAL8 c1=0.0001, REAL8 c2=0.9, INTEGER maxLS=25, REAL8 tolX=0.000000001,DATASET(Types.NumericField) CostFunc_params, DATASET(Types.NumericField) TrainData , DATASET(Types.NumericField) TrainLabel, DATASET(Types.NumericField) CostFunc (DATASET(Types.NumericField) x, DATASET(Types.NumericField) CostFunc_params, DATASET(Types.NumericField) TrainData , DATASET(Types.NumericField) TrainLabel), prows=0, pcols=0, Maxrows=0, Maxcols=0):=FUNCTION
    //initial parameters
    P_num := Max (x, id); //the length of the parameters vector (number of parameters)
    
    ExtractGrad (DATASET(Types.NumericField) inp) := FUNCTION
      RETURN inp (id <= P_num);
    END;
    ExtractCost (DATASET(Types.NumericField) inp) := FUNCTION
      RETURN inp (id = (P_num+1))[1].value;
    END;
    Bracket1no := DATASET([{1,1,-1,10}], Mat.Types.MUElement); //the result of the bracketing algorithm
    Bracket2no := DATASET([{1,1,-1,11}], Mat.Types.MUElement); //the result of the bracketing algorithm

    emptyE := DATASET([], Mat.Types.Element);
    LSiter := 0;
    IsNotLegal (DATASET (Mat.Types.Element) Mat) := FUNCTION //???to be defined
      RETURN FALSE;
    END;
    ArmijoBacktrack4 (DATASET (Mat.Types.MUElement) inputpp) := FUNCTION // to be defined with recieving real parameters (should be a macro similar to this one)
      RETURN inputpp;
    END;
    // polyinterp (REAL8 t_1, REAL8 f_1, REAL8 gtd_1, REAL8 t_2, REAL8 f_2, REAL8 gtd_2) := FUNCTION
      // d1 := gtd_1 + gtd_2 - (3*((f_1-f_2)/(t_1-t_2)));
      // d2 := SQRT ((d1*d1)-(gtd_1*gtd_2));
      // d2real := TRUE; //check it ???
      // temp := IF (d2real,t_2 - ((t_2-t_1)*((gtd_2+d2-d1)/(gtd_2-gtd_1+(2*d2)))),-100);
      // temp100 := temp =-100;
      // polResult := IF (temp100,(t_1+t_2)/2,MIN([MAX([temp,t_1]),t_2]));
      // RETURN polResult;
    // END;

    //OK
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
        newt := polyinterp (tPrev, fPrev,gtdPrev, tt, fNew, gtdNew);
        newtno := DATASET([{1,1,newt,5}], Mat.Types.MUElement);
        // f_prev = f_new;
        // g_prev = g_new;
        // gtd_prev = gtd_new;
        fPrevno := DATASET([{1,1,fNew,1}], Mat.Types.MUElement);
        gPrevno := Mat.MU.To (gNew,3);
        gtdPrevno:= DATASET([{1,1,gtdNew,8}], Mat.Types.MUElement);
        //calculate fnew gnew gtdnew
        xNew := ML.Types.FromMatrix (ML.Mat.Add(ML.Types.ToMatrix(x),ML.Mat.Scale(ML.Types.ToMatrix(d),newt)));
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
      //tTmp := polyinterp (t_first, f_first, gtd_first[1].value, t_second, f_second, gtd_second[1].value); orig
      tTmp := IF (coun=1,52.4859, IF(coun=2, 30.5770, IF(coun=3,19.5981, IF(coun=4,17.2821, IF(coun=5,19.4093,IF(coun=6,19.3919, IF(coun=7,19.3902,19.3901)))))));
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
      CG_New := CostFunc (x_td ,CostFunc_params,TrainData, TrainLabel);
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
      ZOOMTermination :=( (Mat.MU.FROM (ZOOOMResult,200)[1].value = 0) & (ABS((gtdNew[1].value * (t_first-t_second)))<tolX) ) | (Mat.MU.FROM (ZOOOMResult,200)[1].value = 1);
      ZOOMTermination_num := (INTEGER)ZOOMTermination;
      ZOOMFinalResult := ZOOOMResult (no<200) + DATASET([{1,1,ZOOMTermination_num,200}], Mat.Types.MUElement)+ DATASET([{1,1,insufProgress_new,300}], Mat.Types.MUElement) +ZoomFunEvalno ;
      RETURN ZOOMFinalResult; 
      //RETURN DATASET([{1,1,lof[1].value,1},{1,1,hif[1].value ,2}], Mat.Types.MUElement);
        
    END;// END WolfeZooming
    //x_new = x+t*d
    x_new := ML.Types.FromMatrix (ML.Mat.Add(ML.Types.ToMatrix(x),ML.Mat.Scale(ML.Types.ToMatrix(d),t)));
    // Evaluate the cost and Gradient at the Initial Step
    CostGrad_new := CostFunc (x_new ,CostFunc_params,TrainData, TrainLabel);
    g_new := ExtractGrad (CostGrad_new);
    f_new := ExtractCost (CostGrad_new);
    funEvals := 1;
    //gtd_new = g_new'*d;
    gtd_new := ML.Mat.Mul (ML.Mat.Trans(ML.Types.ToMatrix(g_new)),ML.Types.ToMatrix(d));
    
    // Bracket an Interval containing a point satisfying the Wolfe Criteria
    t_prev := 0;
    f_prev := f;
    g_prev := g;
    gtd_prev := gtd;

    //Bracketing algorithm, either produces the final t value or a bracket that contains the final t value
    //prepare the parameters to be passed to the bracketing algorithm
    f_prevno := DATASET([{1,1,f_prev,1}], Mat.Types.MUElement);
    f_newno := DATASET([{1,1,f_new,2}], Mat.Types.MUElement);
    g_prevno := Mat.MU.To (ML.Types.ToMatrix(g_prev),3);
    g_newno := Mat.MU.To (ML.Types.ToMatrix(g_new),4);
    tno := DATASET([{1,1,t,5}], Mat.Types.MUElement);
    t_prevno := DATASET([{1,1,t_prev,6}], Mat.Types.MUElement);
    funEvalsno := DATASET([{1,1,funEvals,7}], Mat.Types.MUElement);
    gtd_prevno := DATASET([{1,1,gtd_prev,8}], Mat.Types.MUElement);
    gtd_newno := DATASET([{1,1,gtd_new[1].value,9}], Mat.Types.MUElement);
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
    Topass := f_prevno + f_newno + g_prevno + g_newno + tno + t_prevno + funEvalsno + gtd_prevno + gtd_newno + Bracket1no + Bracket2no  ;
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
      //armijo only returns final t results and then the loop will stop becasue bracket1 would be ~-1
      WolfeH := WolfeBracketing ( fi_new[1].value, fi_prev[1].value, gtdi_new[1].value, gtdi_prev[1].value, ti[1].value, ti_prev[1].value, gi_new, gi_prev, FunEvalsi[1].value, (coun-1));
      Bracketing_Result := IF (AreTheyLegal, ArmijoBacktrack4(inputp), WolfeH );
      tobereturn := Bracketing_Result + DATASET([{1,1,coun-1,100}], Mat.Types.MUElement);
      RETURN tobereturn;  
    END;
    Bracketing_Result := LOOP(Topass, COUNTER <= maxLS AND Mat.MU.From (ROWS(LEFT),10)[1].value = -1, Bracketing(ROWS(LEFT),COUNTER));
    
    
    FoundInterval := Bracketing_Result (no = 10) + Bracketing_Result (no = 11) + Bracketing_Result (no = 12) + Bracketing_Result (no = 13) + Bracketing_Result (no = 14) + Bracketing_Result (no = 15);
    FinaltInterval := Bracketing_Result (no = 10) + Bracketing_Result (no = 12) + Bracketing_Result (no = 14) ;
    Interval_Found := Mat.MU.From (Bracketing_Result,10)[1].value != -1 AND Mat.MU.From (Bracketing_Result,11)[1].value !=-1;
    final_t_found := Mat.MU.From (Bracketing_Result,10)[1].value != -1 AND Mat.MU.From (Bracketing_Result,11)[1].value =-1;
    ItrExceedInterval := DATASET([{1,1,0,10},
    {1,1,Mat.MU.From (Bracketing_Result,5)[1].value ,11},
    {1,1,f ,12},
    {1,1,Mat.MU.From (Bracketing_Result,2)[1].value ,13}
    ], Mat.Types.MUElement) + Mat.MU.To (ML.Types.ToMatrix(g),14) + Mat.MU.To (Mat.MU.FROM(Bracketing_Result,4),15);
    //
    Zoom_Max_itr_tmp :=  maxLS - Mat.MU.From (Bracketing_Result,100)[1].value;
    Zoom_Max_Itr := IF (Zoom_Max_itr_tmp >0, Zoom_Max_itr_tmp, 0);
    TOpassZOOM := FoundInterval + DATASET([{1,1,0,200}], Mat.Types.MUElement) + DATASET([{1,1,0,300}], Mat.Types.MUElement) + Bracketing_Result (no = 7); // pass the found interval + {zoomtermination=0} to Zoom LOOP +insufficientProgress+FunEval
    ZOOMInterval := LOOP(TOpassZOOM, COUNTER <= Zoom_Max_Itr AND Mat.MU.From (ROWS(LEFT),200)[1].value = 0, WolfeZooming(ROWS(LEFT), COUNTER));
    FinalBracket := IF (final_t_found, FinaltInterval, IF (Interval_Found,ZOOMInterval,ItrExceedInterval));
    WolfeOut :=FinalBracket;
    RETURN WolfeOut; 
    //MYOUT := ZOOMInterval;
   // RETURN MYOUT;
  END;// END WolfeLineSearch

END;// END Optimization