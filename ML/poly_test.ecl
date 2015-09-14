IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat;
IMPORT * FROM ML.Types;
IMPORT PBblas;
Layout_Cell := PBblas.Types.Layout_Cell;
Layout_Part := PBblas.Types.Layout_Part;
emptyC := DATASET([], Types.NumericField);

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
      params_part := DMAT.solvelinear (A_map,  A_part, FALSE, b_map, b_part) ; // Takes around 12 seconds to be calculated
      params := DMat.Converted.FromPart2DS (params_part);
      //If I just assign some constant value to "params" parameter in order to avoid using DMAT.solvelinear  in calculation of this parameter for the sake of comparing the 
      //overall algorithm time with/without using DMAT.solvelinear  then the algorithm takes no time to be run.
      // params := DATASET([
      // {1,1,0.28},
      // {2,1,-3.04},
      // {3,1, 10.24},
      // {4,1,-7.48}],
      // Types.NumericField);

      params1 := params(id=1)[1].value;
      params2 := params(id=2)[1].value;
      params3 := params(id=3)[1].value;
      params4 := params(id=4)[1].value;
      dParams1 := 3*params(id=1)[1].value;
      dparams2 := 2*params(id=2)[1].value;
      dparams3 := params(id=3)[1].value;
      Rvalues := roots (dParams1, dparams2, dparams3);
      // Compute Critical Points
      INANYINF := FALSE;
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
      topa :=  DATASET([{1,1,(xminBound+xmaxBound)/2},{2,1,1000000}], Types.NumericField);//send minpos and fmin value to Resultsstep
      Resultstep (DATASET(Types.NumericField) x, UNSIGNED coun) := FUNCTION
        inr := x(id=1)[1].value;
        f_min := x(id=2)[1].value;
        // if imag(xCP)==0 && xCP >= xminBound && xCP <= xmaxBound
        xCP := cp(id=coun)[1].value;
        cond := xCP >= xminBound AND xCP <= xmaxBound; 
        // fCP = polyval(params,xCP);
        fCP := params1*POWER(xCP,3)+params2*POWER(xCP,2)+params3*xCP+params4;
        //if imag(fCP)==0 && fCP < fmin
        cond2 := (coun=1 OR fCP<f_min) AND ISrootsreal; // If the roots are imaginary so is FCP
        rr := IF (cond,IF (cond2, xCP, inr),inr);
        ff := IF (cond,IF (cond2, fCP, f_min),f_min);
        RETURN DATASET([{1,1,rr},{2,1,ff}], Types.NumericField);
      END;
      finalresult := LOOP(topa, COUNTER <= itr, Resultstep(ROWS(LEFT),COUNTER));
     RETURN finalresult(id=1)[1].value;
     //RETURN params;
    END;//END poly1
    polResult := poly1;
    RETURN polResult;
  END;//end polyinterp_both
  
  newt := polyinterp_both (1, 0,5, 6, 5, 4, 2, 1);
  output(newt, named('newt'));
