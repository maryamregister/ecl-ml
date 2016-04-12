IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat;
IMPORT * FROM ML.Types;
IMPORT PBblas;
Layout_Cell := PBblas.Types.Layout_Cell;
Layout_Part := PBblas.Types.Layout_Part;
emptyC := DATASET([], Types.NumericField);

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
      POWER(t_1,1),POWER(t_2,1), 1, 1,
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
    RETURN params_partset;
  END;//end polyinterp_both
  
  
  
   
  newt := polyinterp_both (10.0000  , 17.4416  , -0.0235,    100.0000 ,  12.4581 ,  -0.0101, 100.9000, 1000);
  output(newt, named('newt'));
