
//#option ('divideByZero', 'nan');
IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat;
IMPORT * FROM ML.Types;
IMPORT PBblas;
Layout_Cell := PBblas.Types.Layout_Cell;
Layout_Part := PBblas.Types.Layout_Part;
matrix_t     := SET OF REAL8;
     OutputRecord := RECORD
      REAL8 t;
      REAL8 f_new;
      DATASET(Mat.Types.Element) g_new;
      INTEGER8 WolfeFunEval;
    END;
      

//Func : handle to the function we want to minimize it, its output should be the error cost and the error gradient
EXPORT Optimization (UNSIGNED4 prows=0, UNSIGNED4 pcols=0, UNSIGNED4 Maxrows=0, UNSIGNED4 Maxcols=0) := MODULE
    // BFGS Search Direction
    //
    // This function returns the (L-BFGS) approximate inverse Hessian,
    // multiplied by the gradient
    //
    // If you pass in all previous directions/sizes, it will be the same as full BFGS
    // If you truncate to the k most recent directions/sizes, it will be L-BFGS
    //
    // s - previous search directions (p by k) , p : length of the parameter vector , k : number of corrcetions
    // y - previous step sizes (p by k)
    // g - gradient (p by 1)
    // Hdiag - value of initial Hessian diagonal elements (scalar)

    
    
    //s : old_dirs: no starts from 1 to k
    //d: old_steps : no starts from k+1 to 2*k
    
    //this function calculates SUM(inp1.*inp2)
    SHARED SumProduct (DATASET(Layout_Part) inp1, DATASET(Layout_Part) inp2) := FUNCTION

      Product(REAL8 val1, REAL8 val2) := val1 * val2;
      Elem := {REAL8 v};  //short-cut record def
      Elem hadPart(Layout_Part xrec, Layout_Part yrec) := TRANSFORM //hadamard product
        elemsX := DATASET(xrec.mat_part, Elem);
        elemsY := DATASET(yrec.mat_part, Elem);
        new_elems := COMBINE(elemsX, elemsY, TRANSFORM(Elem, SELF.v := Product(LEFT.v,RIGHT.v)));
        SELF.v :=  SUM(new_elems,new_elems.v);
      END;

      prod := JOIN(inp1, inp2, LEFT.partition_id=RIGHT.partition_id, hadPart(LEFT,RIGHT), FULL OUTER, LOCAL);
      RETURN SUM(prod, prod.v);
    END;//END SumProduct
    

    EXPORT lbfgs_4 (DATASET(Layout_Part) g, DATASET(PBblas.Types.MUElement) s, DATASET(PBblas.Types.MUElement) y, REAL8 Hdiag) := FUNCTION
      dot_tmp_rec := RECORD
        UNSIGNED2 id;
        REAL8 ro;
      END;
      one_map := PBblas.Matrix_Map(1,1,1,1);
      k := MAX(s,no); // k is the number of previous step vectors included in the s strcuture
      Product(REAL8 val1, REAL8 val2) := val1 * val2;
      Elem := {REAL8 v};  //short-cut record def
      //I am implementing this myself instead of using PBblas library because this can be done in paralel for all the vectors in s and y recordsest (the vector with corresponding ids can dot product simultaniously)
      dot_tmp_rec hadPart(PBblas.Types.MUElement xrec, PBblas.Types.MUElement yrec) := TRANSFORM //hadamard product
        elemsX := DATASET(xrec.mat_part, Elem);
        elemsY := DATASET(yrec.mat_part, Elem);
        new_elems := COMBINE(elemsX, elemsY, TRANSFORM(Elem, SELF.v := Product(LEFT.v,RIGHT.v)));
        SELF.ro :=  SUM(new_elems,new_elems.v);
        SELF.id := xrec.no;
      END;
      //calculate ro values,  ro(i) = 1/(y(i)'*s(i)); 
      ro_tmp := JOIN(s, y, LEFT.partition_id=RIGHT.partition_id AND LEFT.no=RIGHT.no-k, hadPart(LEFT,RIGHT), FULL OUTER, LOCAL);
      ro_rec := RECORD
        ro_tmp.id;
        REAL ro_val := 1/SUM(GROUP,ro_tmp.ro) ;
      END; 
      ro := TABLE(ro_tmp,ro_rec,id,FEW);
      
      //calculate q and al
      //inp has the previous q value with no=1 and the al values start from no+1 to no+k+1 which correponds to al[1] to al[k], each al[i] is calculated in one iteration from i=k to i=1
      q_step (DATASET(PBblas.Types.MUElement) inp, unsigned4 coun) := FUNCTION
        inp_ := PBblas.MU.From(inp,1); // this is actually old_q, its has no=1
        ind := k-coun+1;
        s_ := PBblas.Mu.From(s,ind);
        y_ := PBblas.MU.From(y,ind+k);
        ro_ := ro(id=ind)[1].ro_val;
        //calculate al_ 
        //al_ = ro(i)*s(:,i)'*q(:,i+1);
        //al_tmp := Pbblas.PB_dgemm(TRUE, FALSE, ro_, param_map, s_, param_map, inp_, one_map); orig
        al_ := ro_ * SumProduct (s_ ,inp_ );
        //al_ := al_tmp[1].mat_part[1]; orig
        al_tmp := ML.DMat.Converted.FromElement(DATASET ([{1,1,al_}],MAT.Types.Element),one_map);
        al_no := Pbblas.MU.TO(al_tmp,ind+1); //+1 is added to make sure that the last al[1] gets no=1 so it does not gets mixed up with q that has no=1
        //calculate q
        //q(:,i) = q(:,i+1)-al(i)*y(:,i); // new_q = old_q - al_ * y_
        new_q := PBblas.PB_daxpy(-1*al_, y_, inp_);
        new_q_no := Pbblas.MU.TO(new_q,1); // this goanna be old_q (no=1) for the next loop iteration
        RETURN al_no+new_q_no+inp(no!=1);
      END; //END q_step
      g_ind := 2*k+1;
      topass_q := PBblas.MU.TO(g,1);
      q_tmp := LOOP(topass_q, COUNTER <= k, q_step(ROWS(LEFT),COUNTER));
      //q_tmp : no=1 has the q vector, n=2 to n=k+1 have al values corespondant ro al[1] tp al[k]
      q := Pbblas.MU.FROM(q_tmp,1);
      al_tmp := q_tmp (no>1);
      al_rec := RECORD
        INTEGER8 id;
        REAL al_val;
      END; 
      al := PROJECT(al_tmp, TRANSFORM(al_rec, SELF.id:=LEFT.no-1, SELF.al_val := LEFT.mat_part[1])); //al contains al values with id starting from 1 to k
      
      //calculate r
      //inp(no=1) includes r calculate in teh previous step
      //inp(n=2) to inp(n=k+1) includes be values, which each one is calculated in one step
      r_step (DATASET(Layout_Part) inp, unsigned4 coun) := FUNCTION
        inp_ := inp; // this is actually old_r, it has no=1
        y_ := PBblas.MU.From(y,coun+k);
        s_ := PBblas.MU.From(s,coun);
        ro_ := ro(id=coun)[1].ro_val;
        al_ := al (id=coun)[1].al_val;
        // be(i) = ro(i)*y(:,i)'*r(:,i);
        //be_tmp := Pbblas.PB_dgemm(TRUE, FALSE, ro_, param_map, y_, param_map, inp_, one_map); orig
        be_tmp := ro_ * SumProduct (y_ ,inp_ );
        be_ := be_tmp;//this covers r(:,1) = Hdiag*q(:,1);
        //r(:,i+1) = r(:,i) + s(:,i)*(al(i)-be(i));
        al_be := al_ - be_;
        new_r := PBblas.PB_daxpy(al_be, s_, inp_);
        RETURN new_r ;
      END;
      //Functions needed in calculations
      PBblas.Types.value_t h_mul(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := v * Hdiag;
      topass_r := PBblas.PB_dscal(Hdiag,q); //r(:,1) = Hdiag*q(:,1);
      d := LOOP(topass_r, COUNTER <= k, r_step(ROWS(LEFT),COUNTER));
      
     //RETURN PBblas.PB_dscal (al (id=1)[1].al_val,q);
     //RETURN PBblas.PB_dscal(Hdiag,g);
     RETURN d; 
    END; // END lbfgs_4
    
    //olsy include old s and y values, where s vectors have no=1 to no =k and y values have no=k+1 to k+k (k is the number of vector  we are storing by now (k<= corrections)
    //s is the new vector s which we need to add to the oldsy dataset
    //y is the new vector y which we need to add to the oldsy dataset
    //corrections is the lbfgs parameter value for the number of s and y vectore we are saving in the memory ( that's why it is called limited memory-bfgs), if we save all the s and y vector it would be bfgs 
    EXPORT lbfgsUpdate ( DATASET(PBblas.Types.MUElement) oldsy, DATASET(Layout_Part) y,DATASET(Layout_Part) s , INTEGER8 corrections, REAL8 ys) := FUNCTION
      one_map := PBblas.Matrix_Map(1,1,1,1);
      k := MAX(oldsy,no)/2;
      K_corr := k<corrections;
      s_new_ind := IF (K_corr, k+1, k);
      y_new_ind := IF (K_corr, 2*(k+1), 2*k);
      s_new_no := PBblas.MU.TO(s,s_new_ind);
      y_new_no := PBBlas.MU.TO(y, y_new_ind);
      //IF the dataset is not still full (k<corrections)
      PBblas.Types.MUElement no_update_notfull (PBblas.Types.MUElement old) := TRANSFORM
        SELF.no := IF (old.no <=k, old.no, old.no+1);
        SELF := old;
      END;
      //IF the dataset is full (k<corrections)
      PBblas.Types.MUElement no_update_full (PBblas.Types.MUElement old) := TRANSFORM
        SELF.no := old.no-1;
        SELF := old;
      END;
      oldsy_reduced := oldsy(no != 1 AND no != (k+1));
      old_notfull_updated := PROJECT(oldsy, no_update_notfull(LEFT),LOCAL);
      old_full_updated := PROJECT(oldsy_reduced, no_update_full(LEFT),LOCAL);
      old_updated := IF (K_corr, old_notfull_updated, old_full_updated);
      //calculate ys
      //ys = y'*s;
      // ys_ := PBblas.PB_dgemm (TRUE, FALSE, 1.0, param_map, y, param_map, s, one_map);
      // ys := ys_[1].mat_part[1];
      R := IF (ys<10^(-10), oldsy, old_updated + s_new_no + y_new_no);
      RETURN R; 
    END; // END lbfgsUpdate 
    
    //update HDiag value
    EXPORT lbfgsUpdate_Hdiag ( DATASET(Layout_Part) y , PBblas.IMatrix_Map param_map, REAL8 ys, REAL8 hdiag) := FUNCTION
      one_map := PBblas.Matrix_Map(1,1,1,1);
      yy_ := PBblas.PB_dgemm (TRUE, FALSE, 1.0, param_map, y, param_map, y, one_map);
      yy := yy_[1].mat_part[1];
      hdiag_new := ys/yy;
      R := IF (ys<10^(-10), hdiag_new, hdiag);
      RETURN R;
    END;

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
      
      // I was using LOOP before to implement this part and it make some LOOP related errors in the wolfesearch function later, so I changed the code in a way that it does not use LOOP
      out := FUNCTION
        fmin1 := 100000000;
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
      d2real := IF((d1*d1)-(gtdmin*gtdmax) >=0 , TRUE, FALSE);
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
    
  //
  //WolfeLineSearch
  // Bracketing Line Search to Satisfy Wolfe Conditions
  //Source "Numerical Optimization Book" and Matlab implementaion of minFunc :
  // M. Schmidt. minFunc: unconstrained differentiable multivariate optimization in Matlab. http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html, 2005.
  //  x:  Starting Location 
  //  t: Initial step size (its a number, usually 1)
  //  d:  descent direction  (Pk in formula 3.1 in the book) 
  //  g: gradient at starting location 
  //  gtd:  directional derivative at starting location :  gtd = g'*d (its a number), TRANS(Deltafk)*Pk in formula 3.6a
  //  c1: sufficient decrease parameter (c1 in formula 3.6a, its a number)
  //  c2: curvature parameter (c2 in formula 3.6b, its a number)
  //  maxLS: maximum number of iterations in WOLFE algorithm
  //  tolX: minimum allowable step length
  //  CostFunc: objective function(it returns the gradient and cost value in numeric field format, cost value has the highest
  //  id in the returned numeric field structure
  //  TrainData and TrainLabel: train and label data for the objective fucntion 
  //  The rest are PBblas parameters
  //  Define a general FunVAL and add it in wolfebracketing and WolfeZoom ??????
  //  WolfeOut includes t,f_new,g_new,funEvals (t the calculated step size
  //  f_new the cost value in the new point, g_new is the gradient value in the new point and funevals is the number of
  

   // id =1 for g_prev and id=2 for g_next 
   SHARED  bracketing_record4 := RECORD(Layout_Part)
        INTEGER8 id; // id=1 means prev values, id=2 means new values
        REAL8 f_;
        REAL8 t_;
        INTEGER8 funEvals_;
        REAL8 gtd_;
        INTEGER8 c; //Counter
        INTEGER8 Which_cond; // which bracketing condition is satisfied
        //-1 :- no condition satisfied, continue the loop
        // 1 : if f_new > f + c1*t*gtd || (LSiter > 1 && f_new >= f_prev) is satisfied, break!
        // 2 : if abs(gtd_new) <= -c2*gtd is satisfied , break!
        // 3 : if gtd_new >= 0 is satisfied, break!
    END;
    SHARED zooming_record4 := RECORD (bracketing_record4)
      BOOLEAN insufProgress := FALSE;
      BOOLEAN LoopTermination := FALSE;
    END;
    
 EXPORT WolfeLineSearch4(INTEGER cccc, DATASET(Layout_Part) x, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Part) TrainData , DATASET(Layout_Part) TrainLabel,DATASET(PBblas.Types.MUElement) CostFunc (DATASET(Layout_Part) x, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Part) TrainData , DATASET(Layout_Part) TrainLabel), UNSIGNED param_num, REAL8 t, DATASET(Layout_Part) d, REAL8 f, DATASET(Layout_Part) g, REAL8 gtd, REAL8 c1=0.0001, REAL8 c2=0.9, INTEGER maxLS=25, REAL8 tolX=0.000000001):=FUNCTION
    //maps used
    one_map := PBblas.Matrix_Map(1,1,1,1);
    //Extract the gradient layout_part from the cost function result
    ExtractGrad (DATASET(PBblas.Types.MUElement) inp) := FUNCTION
      RETURN PBblas.MU.FROM(inp,1); 
    END;
    //Extract the gradient part from the cost value result
    ExtractCost (DATASET(PBblas.Types.MUElement) inp) := FUNCTION
      inp2 := inp (no=2);
      RETURN inp2[1].mat_part[1]; 
    END;
    Extractvalue (DATASET(Layout_Part) inp) := FUNCTION
      RETURN inp[1].mat_part[1]; 
    END;

  
    // Evaluate the Objective and Gradient at the Initial Step
    //x_new = x+t*d
    x_new := PBblas.PB_daxpy(t, d, x);
    CostGrad_new := CostFunc(x_new,CostFunc_params,TrainData, TrainLabel);
    //CostGrad_new := myfunc(x_new,param_map,param_num);
    g_new := ExtractGrad (CostGrad_new);
    f_new := ExtractCost (CostGrad_new);
    //gtd_new = g_new'*d;
    //gtd_new := Extractvalue(PBblas.PB_dgemm(TRUE, FALSE, 1.0,param_map, g_new,param_map, d,one_map ));
    gtd_new := SumProduct (g_new,d);
    funEvals := 1;
    
    // Bracket an Interval containing a point satisfying the Wolfe criteria
    t_prev := 0;
    f_prev := f;
    g_prev := g;
    gtd_prev := gtd;
    
   //Bracketing algorithm, either produces the final t value or a bracket that contains the final t value
   Load_bracketing_record := FUNCTION
   
     bracketing_record4 load_scalars_g_prev (Layout_Part l) := TRANSFORM
        SELF.id := 1;
        SELF.f_ := f_prev;
        SELF.t_ := t_prev;
        SELF.funEvals_ := funEvals;
        SELF.gtd_ := gtd_prev;
        SELF.c := 0; //Counter
        SELF.Which_cond := -1;
        SELF := l;
      END;
    R1 := PROJECT(g_prev, load_scalars_g_prev(LEFT) );
    bracketing_record4 load_scalars_g_new (Layout_Part l) := TRANSFORM
        SELF.id := 2;
        SELF.f_ := f_new;
        SELF.t_ := t;
        SELF.funEvals_ := funEvals;
        SELF.gtd_ := gtd_new;
        SELF.c := 0; //Counter
        SELF.Which_cond := -1;
        SELF := l;
      END;
    R2 := PROJECT(g_new, load_scalars_g_new(LEFT) );
    RETURN R1+R2;
   END; // END Load_bracketing_record
   ToPassBracketing := Load_bracketing_record;
   BracketingStep (DATASET (bracketing_record4) inputp, INTEGER coun) := FUNCTION
    // if ~isLegal(f_new) || ~isLegal(g_new) ????
    in_table := TABLE(inputp, {id, f_,t_,funevals_,gtd_}, id, FEW);
    in1 := in_table(id=1);
    in2 := in_table (id=2);
    
    fPrev := in1[1].f_;
    
    fNew := in2[1].f_;
    
    gtdPrev := in1[1].gtd_;

    gtdNew := in2[1].gtd_;
    
    tPrev := in1[1].t_;
    
    tt := in2[1].t_;
    
    BrackfunEval := in1[1].funEvals_;

    BrackLSiter := coun-1;
    
    // check conditions
    // 1- f_new > f + c1*t*gtd || (LSiter > 1 && f_new >= f_prev)
    con1 := (fNew > f + c1 * tt* gtd)| ((BrackLSiter > 1) & (fNew >= fPrev));
   //con1 := FALSE;
    //2- abs(gtd_new) <= -c2*gtd orig
    con2 := ABS(gtdNew) <= (-1*c2*gtd);
   //con2 := TRUE;
    // 3- gtd_new >= 0
    con3 := gtdNew >= 0;
    WhichCon := IF (con1, 1, IF(con2, 2, IF (con3,3,-1)));
    
    //update which_cond in the input dataset. If which_cond != -1 the loop ends (break)
    inputp_con :=  PROJECT(inputp, TRANSFORM(bracketing_record4, SELF.Which_cond := WhichCon; SELF:=LEFT));
    
    //the bracketing results when none of the above conditions are satsfied (calculate a new t and update f,g, etc. values)
    bracketing_Nocon := FUNCTION
      //calculate new t
      minstep := tt + 0.01* (tt-tPrev);
      maxstep := tt*10;
      newt := polyinterp_both (tPrev, fPrev,gtdPrev, tt, fNew, gtdNew, minstep, maxstep);
      //calculate fnew gnew gtdnew
      xNew := PBblas.PB_daxpy(newt, d, x);
      CostGradNew := CostFunc(xNew,CostFunc_params,TrainData, TrainLabel);
      //CostGradNew := myfunc(xNew,param_map,param_num);
      gNewbrack := ExtractGrad (CostGradNew);
      fNewbrack := ExtractCost (CostGradNew);
      //gtdNewbrack := Extractvalue(PBblas.PB_dgemm(TRUE, FALSE, 1.0,param_map, gNewbrack,param_map, d,one_map ));
      gtdNewbrack := SumProduct (gNewbrack, d);
      //update inputp :
      //current _prev values in inputp are replaced with current _new values
      //current _new values in inputp are replaced with the new calculated values based on newt are
      //funEvals_ is increased by 1
      BrackfunEval_1 := BrackfunEval + 1;
      bracketing_record4 load_scalars_g_prev (bracketing_record4 l) := TRANSFORM
        SELF.id := 1;
        SELF.funEvals_ := BrackfunEval_1;
        SELF.c := coun; //Counter
        SELF := l;
      END;
      R_id_1 := PROJECT(inputp(id=2), load_scalars_g_prev(LEFT) ); //id value is changed from 2 to 1. It is the same as :  f_prev = f_new;g_prev = g_new; gtd_prev = gtd_new; : the new values in the current loop iteration are actually prev values for the next iteration

      bracketing_record4 load_scalars_g_new (Layout_Part l) := TRANSFORM
        SELF.id := 2;
        SELF.f_ := fNewbrack;
        SELF.t_ := newt;
        SELF.funEvals_ := BrackfunEval_1;
        SELF.gtd_ := gtdNewbrack;
        SELF.c := coun; //Counter
        SELF.Which_cond := -1;
        SELF := l;
      END;
      R_id_2 := PROJECT(gNewbrack, load_scalars_g_new(LEFT) ); // scalar values are wrapped around gNewbrack with id=2 , these are actually the new values for the next iteration
     
      RETURN R_id_1 + R_id_2;
    END;

    LoopResult := IF (WhichCon=-1, bracketing_Nocon, inputp_con);
    RETURN IF (COUNT(inputp)=0,inputp,LoopResult);
   END;//END BracketingStep
   
 extr_which_cond (DATASET(bracketing_record4) i) := FUNCTION
    in_table := TABLE(i, {which_cond}, FEW);
    RETURN in_table[1].which_cond;
   END;
   extr_loop_term (DATASET(zooming_record4) i) := FUNCTION
    in_table := TABLE(i, {LoopTermination}, FEW);
    RETURN in_table[1].LoopTermination;
   END;
  
  // BracketingResult := LOOP(ToPassBracketing, COUNTER <= maxLS AND loopcond(ROWS(LEFT))=-1 , BracketingStep(ROWS(LEFT),COUNTER)); 
  //BracketingResult := LOOP(ToPassBracketing, maxLS, LEFT.Which_cond = -1, BracketingStep(ROWS(LEFT),COUNTER)); 
  BracketingResult := LOOP(ToPassBracketing, maxLS, LEFT.Which_cond = -1, BracketingStep(ROWS(LEFT),COUNTER));
  //BracketingResult :=  LOOP(Topassbracketing, extr_which_cond(ROWS(LEFT))=-1 AND  COUNTER <maxLS  , bracketingstep(ROWS(LEFT),COUNTER)); orig
  brack_table := TABLE(BracketingResult, {id, c}, id, FEW);
 
  Zoom_Max_itr_tmp :=   maxLS - brack_table[1].c; // orig ???
  Zoom_Max_Itr := IF (Zoom_Max_itr_tmp >0, Zoom_Max_itr_tmp, 0);

   // Zoom Phase
   
   //We now either have a point satisfying the criteria, or a bracket
   //surrounding a point satisfying the criteria
   // Refine the bracket until we find a point satisfying the criteria
   toto := PROJECT (BracketingResult, TRANSFORM(zooming_record4 ,SELF.LoopTermination:=(LEFT.which_cond=2) | (LEFT.c=MaxLS); SELF := LEFT)); // If in the bracketing step condition 2 has been meet or we have reaached MaxLS then we don't need to pass zoom step, so the termination condition for soom will be set as true here
   toto2 := PROJECT (toto, TRANSFORM(zooming_record4 ,SELF.c:=100; SELF := LEFT)); 
   ZoomingStep (DATASET (zooming_record4) inputp, INTEGER coun) := FUNCTION
    // At the begining of the loop find High and Low Points in bracket:
    // Assign id=1 to the low point
    // Assign id=2 to the high point
    // pass_thru := inputp0(LoopTermination = TRUE);
    // inputp:= inputp0(LoopTermination = FALSE);
    in_table := TABLE(inputp, {id, f_,t_,funevals_,gtd_, insufProgress,c}, id, FEW);
    in1 := in_table (id=1);
    in2 := in_table (id=2);

    bracketFval_1 := in1[1].f_;
    
    bracketFval_2 := in2[1].f_;
    
    bracket_1 := in1[1].t_;
    
    bracket_2 := in2[1].t_;
    
    bracketGval_1 := PROJECT(inputp(id=1),TRANSFORM(Layout_Part,SELF := LEFT),LOCAL);
    
    bracketGval_2 := PROJECT(inputp(id=2),TRANSFORM(Layout_Part,SELF := LEFT),LOCAL);
    
    //bracketGTDval_1  := Extractvalue(PBblas.PB_dgemm(TRUE, FALSE, 1.0,param_map, bracketGval_1,param_map, d,one_map ));
    bracketGTDval_1 := SumProduct (bracketGval_1, d);
    //bracketGTDval_2  := Extractvalue(PBblas.PB_dgemm(TRUE, FALSE, 1.0,param_map, bracketGval_2,param_map, d,one_map ));
    bracketGTDval_2 := SumProduct (bracketGval_2, d);
    insufprog := in1[1].insufProgress;
    
    zoom_funevals := in1[1].funEvals_;
    zoom_c := in1[1].c;
    
    // Find High and Low Points in bracket
    LO_id := IF (bracketFval_1 < bracketFval_2, 1, 2);
    HI_id := 3 - LO_id;
    
    // Compute new trial value
    // t = polyinterp([bracket(1) bracketFval(1) bracketGval(:,1)'*d bracket(2) bracketFval(2) bracketGval(:,2)'*d],doPlot);
    tTmp := polyinterp_noboundry (bracket_1, bracketFval_1, bracketGTDval_1, bracket_2, bracketFval_2, bracketGTDval_2);
    BList := [bracket_1,bracket_2];
    max_bracket := MAX(Blist);
    min_bracket := MIN(Blist);
    
    //Test that we are making sufficient progress
    // if min(max(bracket)-t,t-min(bracket))/(max(bracket)-min(bracket)) < 0.1
    insuf_cond_1 := MIN ((max_bracket-tTmp),(tTmp-min_bracket))/(max_bracket - min_bracket) < 0.1 ;
    //if insufProgress || t>=max(bracket) || t <= min(bracket)
    insuf_cond_2 := insufprog | (tTmp >= max_bracket ) | (tTmp <= min_bracket);
    //abs(t-max(bracket)) < abs(t-min(bracket))
    insuf_cond_3 := ABS (tTmp-max_bracket) < ABS (tTmp-min_bracket);
    
    max_min_bracket := 0.1 * (max_bracket - min_bracket);
    //t = max(bracket)-0.1*(max(bracket)-min(bracket));
    tIF := max_bracket - max_min_bracket;
    //t = min(bracket)+0.1*(max(bracket)-min(bracket));
    tELSE := min_bracket + max_min_bracket;
    tZoom := IF (insuf_cond_1, IF (insuf_cond_2,  IF (insuf_cond_3, tIF, tELSE) , tTmp), tTmp);
    insufprog_new := IF (insuf_cond_1, IF (insuf_cond_2, FALSE, TRUE) , FALSE);
    zoom_c_new := zoom_c + 1;
    // Evaluate new point with tZoom
    
    xNew := PBblas.PB_daxpy(tZoom, d, x);
    CostGradNew := CostFunc(xNew,CostFunc_params,TrainData, TrainLabel);
    //CostGradNew := myfunc(xNew,param_map,param_num);
    gNewZoom := ExtractGrad (CostGradNew);
    fNewZoom := ExtractCost (CostGradNew);
    //gtd_new = g_new'*d;
    //gtdNewZoom := Extractvalue(PBblas.PB_dgemm(TRUE, FALSE, 1.0,param_map, gNewZoom,param_map, d,one_map ));
    gtdNewZoom := SumProduct (gNewZoom, d);
    zoom_funevals_new := zoom_funevals + 1;
    //Zoom Conditions
    max_bracketFval := IF (HI_id = 1 , bracketFval_1, bracketFval_2);
    min_bracketFval := IF (HI_id = 1 , bracketFval_2, bracketFval_1);
    
    HI_bracket := IF (HI_id = 1 , bracket_1, bracket_2);
    LO_bracket := IF (HI_id = 1 , bracket_2, bracket_1);
    //if f_new > f + c1*t*gtd || f_new >= f_LO
    zoom_cond_1 := (fNewZoom > f + c1 * tZoom * gtd) | (fNewZoom >= min_bracketFval);
    // if abs(gtd_new) <= - c2*gtd
    zoom_cond_2 := ABS (gtdNewZoom) <= (-1 * c2 * gtd);
    //if gtd_new*(bracket(HIpos)-bracket(LOpos)) >= 0
    zoom_cond_3 := gtdNewZoom * (max_bracket - min_bracket) >= 0; 
 
    whichcond := IF (zoom_cond_1, 11, IF (zoom_cond_2, 12, IF (zoom_cond_3, 13, -2)));
    
    zooming_cond_1 := FUNCTION
      // ~ done & abs((bracket(1)-bracket(2))*gtd_new) < tolX
      //Since we are in zooming_cond_1 it means that condition 2 is already not satisfied (~done is true) so we only check the other condition
      zoomter := ABS ((tZoom-LO_bracket)*gtdNewZoom) < tolX;
      zooming_record4 load_scalars_g_new (Layout_Part l) := TRANSFORM
        SELF.id := HI_id;
        SELF.f_ := fNewZoom;
        SELF.t_ := tZoom;
        SELF.funEvals_ := zoom_funevals_new;
        SELF.gtd_ := gtdNewZoom;
        SELF.c := zoom_c_new; //Counter
        SELF.insufProgress := insufprog_new;
        SELF.which_cond := whichcond;
        SELF.LoopTermination := zoomter;
        SELF := l;
      END;
      R_HI_id := PROJECT(gNewZoom, load_scalars_g_new(LEFT) );
      zooming_record4 load_scalars_LOID (zooming_record4 l) := TRANSFORM
        SELF.funEvals_ := zoom_funevals_new;
        SELF.c := zoom_c_new; //Counter
        SELF.insufProgress := insufprog_new;
        SELF.which_cond := whichcond;
        SELF.LoopTermination := zoomter;
        SELF := l;
      END;
      R_LO_id := PROJECT (inputp (id=LO_id), load_scalars_LOID(LEFT) );
      RETURN R_HI_id + R_LO_id ;
    END; // END zooming_cond_1
    
    zooming_cond_2 := FUNCTION
      zoomter := TRUE; // IF condition 2 is correct, then loop should terminates, in case that other conditions are corect abs((bracket(1)-bracket(2))*gtd_new) < tolX should be checked for the zoom termination
      // Old HI becomes new LO
      zooming_record4 HIID (zooming_record4 l) := TRANSFORM
        SELF.funEvals_ := zoom_funevals_new;
        SELF.c := zoom_c_new; //Counter
        SELF.insufProgress := insufprog_new;
        SELF.which_cond := whichcond;
        SELF.LoopTermination := TRUE;
        SELF := l;
      END;
      R_HI_id := PROJECT(inputp(id=HI_id), HIID(LEFT) );
      zooming_record4 load_scalars_g_new (Layout_Part l) := TRANSFORM
        SELF.id := LO_id;
        SELF.f_ := fNewZoom;
        SELF.t_ := tZoom;
        SELF.funEvals_ := zoom_funevals_new;
        SELF.gtd_ := gtdNewZoom;
        SELF.c := zoom_c_new; //Counter
        SELF.insufProgress := insufprog_new;
        SELF.which_cond := whichcond;
        SELF.LoopTermination := TRUE;
        SELF := l;
      END;
      // New point becomes new LO
      R_LO_id := PROJECT(gNewZoom, load_scalars_g_new(LEFT) );
      RETURN R_HI_id + R_LO_id;
    END;// END zooming_cond_2
    
    zooming_cond_3 := FUNCTION
      //Since we are in zooming_cond_1 it means that condition 2 is already not satisfied (~done is true) so we only check the other condition
      zoomter := ABS ((tZoom-LO_bracket)*gtdNewZoom) < tolX;
      zooming_record4 LOID (zooming_record4 l) := TRANSFORM
        SELF.id := HI_id;
        SELF.funEvals_ := zoom_funevals_new;
        SELF.c := zoom_c_new; //Counter
        SELF.insufProgress := insufprog_new;
        SELF.which_cond := whichcond;
        SELF.LoopTermination := zoomter;
        SELF := l;
      END;
      R_HI_id := PROJECT(inputp(id=LO_id), LOID(LEFT) );
      zooming_record4 load_scalars_g_new (Layout_Part l) := TRANSFORM
        SELF.id := LO_id;
        SELF.f_ := fNewZoom;
        SELF.t_ := tZoom;
        SELF.funEvals_ := zoom_funevals_new;
        SELF.gtd_ := gtdNewZoom;
        SELF.c := zoom_c_new; //Counter
        SELF.insufProgress := insufprog_new;
        SELF.which_cond := whichcond;
        SELF.LoopTermination := zoomter;
        SELF := l;
      END; //END zooming_cond_3
      // New point becomes new LO
      R_LO_id := PROJECT(gNewZoom, load_scalars_g_new(LEFT) );
      RETURN R_HI_id + R_LO_id;
    END;
    zooming_nocond := zooming_cond_2;
    
    zooming_result := IF (zoom_cond_1, zooming_cond_1, IF (zoom_cond_2, zooming_cond_2, IF (zoom_cond_3, zooming_cond_3, zooming_nocond )));
    
    zooming_result2 :=  PROJECT (toto, TRANSFORM(zooming_record4 ,SELF.f_:=bracketGTDval_1; SELF := LEFT)); 
    RETURN IF (COUNT(inputp)=0,inputp,zooming_result); //ZoomingStep produces an output even when it recives an empty dataset, I would like to avoid that so I can make sure when loopfilter avoids a dataset to be passed to the loop (in case of which_cond==2 which means we already have found final t) then zooming_step would not produce any output  
    //RETURN zooming_result;
   END; // END ZoomingStep
     
   //BracketingResult is provided as input to the Zooming LOOP 
   //Since in the bracketing results in case of cond 1 and cond2 the prev and new values are assigned to bracket_1 and bracket_2 so when we pass the bracketing result to the zooming loop, in fact we have:
   //id = 1 indicates bracket_1 
   //id = 2 indicates bracket_2
  // LOOP( dataset, loopcount, loopfilter, loopbody [, PARALLEL( iterations | iterationlist [, default ] ) ] )
  Topass_zooming := PROJECT (BracketingResult, TRANSFORM(zooming_record4 ,SELF.LoopTermination:=(LEFT.which_cond=2) | (LEFT.c=MaxLS); SELF := LEFT)); // If in the bracketing step condition 2 has been meet or we have reaached MaxLS then we don't need to pass zoom step, so the termination condition for soom will be set as true here
  //ZoomingResult := IF (extr_loop_term(Topass_zooming)=TRUE, Topass_zooming , LOOP(Topass_zooming, extr_loop_term(ROWS(LEFT))=FALSE AND  COUNTER <=Zoom_Max_Itr  , zoomingstep(ROWS(LEFT),COUNTER))); orig
   ZoomingResult := IF (extr_loop_term(Topass_zooming)=TRUE, Topass_zooming , LOOP(Topass_zooming, Zoom_Max_Itr, LEFT.LoopTermination=FALSE , zoomingstep(ROWS(LEFT),COUNTER)));
   //ZoomingResult := IF (extr_loop_term(Topass_zooming)=TRUE, Topass_zooming , LOOP(Topass_zooming, 1 , zoomingstep(ROWS(LEFT),COUNTER))); 
   //BracketingResult := LOOP(ToPassBracketing, maxLS, LEFT.Which_cond = -1, BracketingStep(ROWS(LEFT),COUNTER));
  //ZoomingResult := Topass_zooming; //added
// BracketingResult := LOOP(ToPassBracketing, LEFT.Which_cond = -1, COUNTER <= maxLS AND EXISTS(ROWS(LEFT)) , BracketingStep(ROWS(LEFT),COUNTER)); 
// based on whichcond value in the very final result we figure out the flow of the data and what we should return as output
   // 2: final t is found in the bracketing step


    bracketing_record4 load_scalars_g (Layout_Part l) := TRANSFORM
      SELF.id := 1;
      SELF.f_ := f;
      SELF.t_ := 0;
      SELF.funEvals_ := maxLS + 1;//when the bracketing loop get to MAX_itr number of iterations, it means that funcation has been evaluated Max_Itr + 1 (one time before the loop starts) Times.
      SELF.gtd_ := gtd;
      SELF.c := maxLS; //Counter
      SELF.which_cond := -1; // when the bracketing loop get to MAX_itr number of iterations, it means that no condition in the bracketing_step has ever been satisfied for the loop to break
      SELF := l;
    END; //END zooming_cond_3
   //wolfecond :
   // -1 : begining of the wolfe algorithm and when we are still in the bracketing step and no condition in satisfied, or we are out of bracketing step with cond=-1 which means we reached MAX_LS
   // 1 : we're out of bracketing step and the condition that break the bracketing step loop was condition number 1 -> go to zooming loop
   // 2 : we're out of bracketing step and the condition that break the bracketing step loop was condition number 2 -> final t found
   // 3 : we're out of bracketing step and the condition that break the bracketing step loop was condition number 3 -> go to zooming loop
   //11, 12, 13, -2 means we are in the zooming loop and condition 1, condition 2 , condition 3 and no condition have been satisfied respectively
   zoomTBL :=  TABLE(ZoomingResult, {id, which_cond, f_}, id, FEW);
   zoomfnew := zoomTBL(id=2)[1].f_;
   zoomfold := zoomTBL(id=1)[1].f_;
   wolfe_cond := zoomTBL[1].which_cond;
   final_t_found := wolfe_cond = 2;
   t_new_result := PROJECT (ZoomingResult (id=2), TRANSFORM(bracketing_record4 ,SELF := LEFT));
   t_old_result := PROJECT (ZoomingResult (id=1), TRANSFORM(bracketing_record4 ,SELF := LEFT));
   t_0_result := PROJECT(g, load_scalars_g(LEFT) );
   final_t_result := t_new_result;
   MaxLS_result := IF ( zoomfnew < f, t_new_result , t_0_result);
   zoom_result := IF ( zoomfnew < zoomfold, t_new_result , t_old_result);
   wolfe_result := IF (final_t_found,final_t_result , IF (Zoom_Max_itr_tmp=0,MaxLS_result,zoom_result));
   RETURN wolfe_result;
   //RETURN bracketingresult;
   
   
   
  
   
   //RETURN  LOOP(Topassbracketing, extr_which_cond(ROWS(LEFT))=5 AND  COUNTER <0  , bracketingstep(ROWS(LEFT),COUNTER)); 
   //RETURN IF (extr_loop_term(Topass_zooming)=TRUE,Topass_zooming , LOOP(Topass_zooming, extr_loop_term(ROWS(LEFT))=FALSE AND  COUNTER <3  , zoomingstep(ROWS(LEFT),COUNTER)));
   //RETURN LOOP(topass_zooming,  LEFT.which_cond = 5, COUNTER <= 1 AND EXISTS(ROWS(LEFT)) , zoomingStep(ROWS(LEFT),COUNTER));
   //RETURN ToPassBracketing;
  //RETURN  LOOP(DATASET([],zooming_record4),COUNTER <= 1 AND EXISTS(ROWS(LEFT)) , ZoomingStep2(ROWS(LEFT),COUNTER));
//RETURN LOOP(DATASET([],zooming_record4),COUNTER <= 1 ,ZoomingStep2(ROWS(LEFT),COUNTER));
//RETURN LOOP( topass_zooming, LEFT.LoopTermination =FALSE , ZoomingStep2(ROWS(LEFT),COUNTER) );
//RETURN LOOP(Topass_zooming, LEFT.LoopTermination =FALSE , ZoomingStep(ROWS(LEFT),COUNTER));
   //RETURN LOOP( Topass_zooming, 1,  ZoomingStep(ROWS(LEFT),COUNTER) );
   //ZoomingResult := LOOP(Topass_zooming, LEFT.LoopTermination =FALSE , COUNTER <= 1 AND EXISTS(ROWS(LEFT)) , ZoomingStep(ROWS(LEFT),COUNTER));
   //RETURN LOOP( Topass_zooming, 1, LEFT.LoopTermination =FALSE, ZoomingStep(ROWS(LEFT),COUNTER));
   //RETURN ZoomingStep(DATASET([],zooming_record4),1);

    
 END;// END WolfeLineSearch4
EXPORT wolfe_g (DATASET(bracketing_record4) wolfeout) := FUNCTION
  RETURN PROJECT(wolfeout, TRANSFORM(Layout_Part, SELF:=LEFT));
END;

EXPORT wolfe_f (DATASET(bracketing_record4) wolfeout) := FUNCTION
  t := TABLE(wolfeout, {f_}, FEW);
  RETURN t[1].f_;
END;

EXPORT wolfe_t (DATASET(bracketing_record4) wolfeout) := FUNCTION
  t := TABLE(wolfeout, {t_}, FEW);
  RETURN t[1].t_;
END;

EXPORT wolfe_funEvals (DATASET(bracketing_record4) wolfeout) := FUNCTION
  t := TABLE(wolfeout, {funEvals_}, FEW);
  RETURN t[1].funEvals_;
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
  //x0: input parameters in Layout_Part format 
  //Cost function should have a universal interface : ( DATASET(Layout_Part) theta, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Part) TrainData , DATASET(Layout_Part) TrainLabel)
  //CostFunc_params : parameters that need to be passed to the CostFunc
  //TrainData : Train data 
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
  EXPORT MinFUNC(DATASET(Layout_Part) x0,DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Part) TrainData , DATASET(Layout_Part) TrainLabel,DATASET(PBblas.Types.MUElement) CostFunc (DATASET(Layout_Part) x0, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Part) TrainData , DATASET(Layout_Part) TrainLabel), INTEGER8 param_num, INTEGER8 MaxIter = 100, REAL8 tolFun = 0.00001, REAL8 TolX = 0.000000001, INTEGER maxFunEvals = 1000, INTEGER corrections = 100, prows=0, pcols=0, Maxrows=0, Maxcols=0) := FUNCTION
    //calculate sum(abs(g_in))
   sum_abs (DATASET(Layout_Part) g_in) := FUNCTION
      Elem := {REAL8 v};  //short-cut record def
      Elem su(Layout_Part xrec) := TRANSFORM //hadamard product
        elemsX := DATASET(xrec.mat_part, Elem);
        elemX_abs := PROJECT(elemsX,TRANSFORM(Elem,SELF.v := ABS(LEFT.v)));
        SELF.v :=  SUM(elemX_abs,elemX_abs.v);
      END;
      ss_ := PROJECT (g_in, su (LEFT), LOCAL);
      ss := SUM (ss_, ss_.v);
      RETURN ss;
    END;//sum_abs
    //check optimality 
    //if sum(abs(g)) <= tolFun
    optimality_check (DATASET(Layout_Part) g_in) := FUNCTION
      Elem := {REAL8 v};  //short-cut record def
      Elem su(Layout_Part xrec) := TRANSFORM //hadamard product
        elemsX := DATASET(xrec.mat_part, Elem);
        elemX_abs := PROJECT(elemsX,TRANSFORM(Elem,SELF.v := ABS(LEFT.v)));
        SELF.v :=  SUM(elemX_abs,elemX_abs.v);
      END;
      ss_ := PROJECT (g_in, su (LEFT), LOCAL);
      ss := SUM (ss_, ss_.v);
      RETURN ss<tolFun;
    END;//END optimality_check
    //maps used
    one_map := PBblas.Matrix_Map(1,1,1,1);
    //Extract the gradient layout_part from the cost function result
    ExtractGrad (DATASET(PBblas.Types.MUElement) inp) := FUNCTION
      RETURN PBblas.MU.FROM(inp,1); 
    END;// END ExtractGrad
    //Extract the gradient part from the cost value result
    ExtractCost (DATASET(PBblas.Types.MUElement) inp) := FUNCTION
      inp2 := inp (no=2);
      RETURN inp2[1].mat_part[1]; 
    END;//End ExtractCost
    Extractvalue (DATASET(Layout_Part) inp) := FUNCTION
      RETURN inp[1].mat_part[1]; 
    END;// END Extractvalue
    IsLegal (DATASET(Layout_Part) inp) := FUNCTION
      RETURN TRUE;
    END;//END IsLegal ???
  
    // Evaluate Initial Point
    CostGrad := CostFunc(x0,CostFunc_params,TrainData, TrainLabel);
    //CostGrad := myfunc(x0,param_map,param_num);
    g0 := ExtractGrad (CostGrad);
    f0 := ExtractCost (CostGrad);
    funEvals := 1;
    Is_Initialpoint_Optimal := optimality_check (g0);
    g0_no := PBblas.MU.TO(g0,1);
    x0_no := PBblas.MU.TO(x0,2);
    minfRec := RECORD (PBblas.Types.MUElement)
      REAL8 h ;//hdiag value
      REAL8 f ; //f from the previous iteration
      INTEGER8 min_funEval ;
      INTEGER break_cond ;
    END;
    topass_min := PROJECT(g0_no+x0_no, TRANSFORM(minfrec, SELF.h := 1; SELF.f := f0, SELF.min_funEval:=funEvals, SELF.break_cond := -1, SELF:= LEFT));
    //in min_step:
    //first d is calculated based on whether it is the first or not_first iteration
    //d is checked to be legal, if it is not legal we return the same input (inp) with break code =1
    //gtd if calculated
    //IF gtd satisfies that there is no progress along the direction then again we return the input inp with break code =2
    // If none of the condition above are satisfied then we calculate new step length that satisfies wolfe condition
    // After Wolfe calulation gives us the new f, g, and t values we check some other break conditions
    //If any of them are satisfied we retunr new g, new f, and the corresponding break code
    //if no break condition is satisfied we retunr the same things as above plus the updated hdiag and sy values which will be used in the next iteraion (if a break cond is satisfied then thers is not gonna
    //be any next iteration so there is no need to calculate updated hdiga nd sy values
    //inp includes : no[1 k] is old_dirs matrix or s. no[k+1 2*k] is old steps or y, no[2*k+1] is g from previous iteration, no [2*k+2] is x from previous iteration
 min_step_firstitr := FUNCTION
     
      d := PBblas.PB_dscal(-1, g0); // Steepest_Descent
      dlegalstep := IsLegal (d);
      // Computer step length
      // Directional Derivative : gtd = g'*d;
      //gtd := Extractvalue(PBblas.PB_dgemm(TRUE, FALSE, 1.0,param_map, g_pre,param_map, d,one_map )); orig
      gtd := SumProduct (g0, d);
      //Check that progress can be made along direction : if gtd > -tolX then break!
      gtdprogress := IF (gtd > -1*tolX, FALSE, TRUE);
      // Select Initial Guess If coun =1 then t = min(1,1/sum(abs(g)));
      t_init := MIN ([1, 1/(sum_abs(g0))]);
      // Find Point satisfying Wolfe
      w := WolfeLineSearch4(1, x0,CostFunc_params,TrainData, TrainLabel,CostFunc, param_num, t_init, d, f0, g0,gtd, 0.0001, 0.9, 25, 0.000000001);
      w_feval := wolfe_funEvals (w);
      w_t := wolfe_t (w);
      w_f := wolfe_f (w);
      w_g := wolfe_g (w);
      x_updated := PBblas.PB_daxpy(w_t, d, x0);
      //update hdiag, s and y
      // lbfgsUpdate ( DATASET(PBblas.Types.MUElement) oldsy, DATASET(Layout_Part) s,DATASET(Layout_Part) y , INTEGER8 corrections, PBblas.IMatrix_Map param_map, REAL8 ys) := FUNCTION
      sy_pre := DATASET([],PBblas.Types.MUElement);
      //s_s_ = t*d
      s_s := PBblas.PB_dscal(w_t, d);
      //y_y := g_new - g_old
      y_y := PBblas.PB_daxpy(-1, g0, w_g);
      //y_s = y_y'*s_s
      //y_s := PBblas.PB_dgemm(TRUE, FALSE, 1.0,param_map, y_y,param_map, s_s,one_map );
      y_s := SumProduct (y_y, s_s);
      sy_updated := lbfgsUpdate ( sy_pre, y_y, s_s, corrections,  y_s);
      //y_y_ := PBblas.PB_dgemm(TRUE, FALSE, 1.0,param_map, y_y,param_map, y_y,one_map );
      y_y_ := SumProduct (y_y, y_y);
      hdiag_updated := y_s/ y_y_;
      w_g_ := PBblas.PB_dscal(-1, w_g);
      // do the optimality (break condition) checks
      //~isLegal(d)
      IsLegald := IsLegal (d);
      legal_d_code := IF (IsLegald, -1, 1);
      //% Check that progress can be made along direction: if gtd > -tolX
      Isprog1 := gtd > -1 * tolX;
      prog1_code := IF (Isprog1, 2, -1 );
      // % Check Optimality Condition : sum(abs(g)) <= tolFun
      Isprog2 := sum_abs(w_g)<= tolFun;
      opt1_code := IF (Isprog2, 3, -1);
      //% ******************* Check for lack of progress ******************* : if sum(abs(t*d)) <= tolX
      Islack1 := ABS(w_t)*sum_abs(d)<= tolX;
      lack1_code := IF (Islack1, 4, -1);
      //if abs(f-f_old) < tolX
      Islack2 := ABS(w_f-f0) < tolX;
      lack2_code := IF (Islack2, 5, -1);
      // Check for going over iteration/evaluation limit *******************
      // if funEvals > maxFunEvals
      updated_funEvals := w_feval + funEvals;
      Isoverfun := updated_funEvals > maxFunEvals;
      fun_code := IF (Isoverfun, 6, -1 );
      break_code := IF(Isprog2, 3, IF (Islack1, 4, IF (Islack2, 5, IF(Isoverfun, 6, -1) )));
      //If breakcode!=-1 it means that we will exit the loop so there is no need to update hdiag and sy
      no_part_break := Pbblas.MU.TO(w_g, 1) + PBblas.MU.TO(x_updated, 2);
      to_return_break := PROJECT(no_part_break, TRANSFORM(minfrec, SELF.h := 1; SELF.f := w_f, SELF.min_funEval:=updated_funEvals, SELF.break_cond := break_code, SELF:= LEFT));
      //IF none of break conditions are satisfied, we will continue the loop. Means that we have to update hdiag and sy values for the next iteration
      max_no := MAX(sy_updated, sy_updated.no);
      no_part := sy_updated + Pbblas.MU.TO(w_g, max_no+1)+ PBblas.MU.TO(x_updated, max_no+2);
      to_return := PROJECT(no_part, TRANSFORM(minfrec, SELF.h := hdiag_updated; SELF.f := w_f, SELF.min_funEval:=updated_funEvals, SELF.break_cond := break_code, SELF:= LEFT));
      //After d is calculated check whether it is legal, if itis legal continue, otherwise return with the break_cond=1
      dnot_legal_return := PROJECT(topass_min, TRANSFORM(minfrec, SELF.break_cond := 1, SELF:= LEFT ));
      Noprogalong_return := PROJECT(topass_min, TRANSFORM(minfrec, SELF.break_cond := 2, SELF:= LEFT ));
      FinalResult := IF (IsLegald, IF (Isprog1, Noprogalong_return, IF (break_code!=-1, to_return_break, to_return)), dnot_legal_return);
      //RETURN IF (EXISTS(inp),FinalResult,inp); orig
     RETURN FinalResult;
   
    END;// min_step_firstitr
    min_result_firstitr := min_step_firstitr;
    min_step (DATASET(minfRec) inp,unsigned4 c) := FUNCTION
      k_ := MAX(inp,no);
      k := (k_-2)/2; //k_ is the index for s which is from 1 to k, y is from k+1 to 2*k, then g has 2*k+1 and the index for x is 2*k+2
      g_pre_ind :=k_-1;
      x_pre_ind := k_;
      inp_ :=  PROJECT(inp, TRANSFORM(PBblas.Types.MUElement, SELF:=LEFT));
      g_pre := PBblas.MU.FROM (inp_,g_pre_ind); // This is the g calculated in the preceding iteration
      x_pre := PBblas.MU.FROM (inp_,x_pre_ind); // This is the x calculated in the preceding iteration
      d_steap := PBblas.PB_dscal(-1, g_pre); // Steepest_Descent
      s_pre := inp_(no>=1 AND no<=k); // old_dirs
      y_pre := inp_(no>k AND no <= 2*k); // old_steps
      funmin_table := TABLE (inp, {min_funEval}, LOCAL);
      funmin_pre := funmin_table[1].min_funEval;
      h_table := TABLE(inp, {h}, LOCAL);
      hDiag_pre := h_table[1].h;
      f_table := TABLE(inp, {f}, LOCAL);
      f_pre := f_table[1].f;
      d_lbfgs := lbfgs_4(d_steap, s_pre, y_pre, hDiag_pre);// orig Hdiga should be defined 
      hrec := RECORD 
        REAL8 h ;//hdiag value
      END;
      hproj := PROJECT(inp, TRANSFORM(hrec, SELF.h:=LEFT.h));
      hdup := dedup(hproj, hproj.h);
      //d_lbfgs := PBblas.PB_dscal(hdup[1].h,d_steap);
      //d_lbfgs := PBblas.PB_dscal (SumProduct(y_pre,s_pre),g_pre);
      //d:= IF (coun=1, d_steap, d_lbfgs);
      d:=  d_lbfgs;
      dlegalstep := IsLegal (d);
      // Computer step length
      // Directional Derivative : gtd = g'*d;
      //gtd := Extractvalue(PBblas.PB_dgemm(TRUE, FALSE, 1.0,param_map, g_pre,param_map, d,one_map )); orig
      gtd := SumProduct (g_pre, d);
      //Check that progress can be made along direction : if gtd > -tolX then break!
      gtdprogress := IF (gtd > -1*tolX, FALSE, TRUE);
      // Select Initial Guess If coun =1 then t = min(1,1/sum(abs(g)));
      //t_init := IF (coun = 1,MIN ([1, 1/(sum_abs(g_pre))]),1);
      t_init := 1;
      // Find Point satisfying Wolfe
      w := WolfeLineSearch4(1, x_pre,CostFunc_params,TrainData, TrainLabel,CostFunc, param_num, t_init, d, f_pre, g_pre,gtd, 0.0001, 0.9, 25, 0.000000001);
      w_feval := wolfe_funEvals (w);
      w_t := wolfe_t (w);
      w_f := wolfe_f (w);
      w_g := wolfe_g (w);
      x_updated := PBblas.PB_daxpy(w_t, d, x_pre);
      //update hdiag, s and y
      // lbfgsUpdate ( DATASET(PBblas.Types.MUElement) oldsy, DATASET(Layout_Part) s,DATASET(Layout_Part) y , INTEGER8 corrections, PBblas.IMatrix_Map param_map, REAL8 ys) := FUNCTION
      sy_pre := inp_(no<=2*k);
      //s_s_ = t*d
      s_s := PBblas.PB_dscal(w_t, d);
      //y_y := g_new - g_old
      y_y := PBblas.PB_daxpy(-1, g_pre, w_g);
      //y_s = y_y'*s_s
      //y_s := PBblas.PB_dgemm(TRUE, FALSE, 1.0,param_map, y_y,param_map, s_s,one_map );
      y_s := SumProduct (y_y, s_s);
      sy_updated := lbfgsUpdate ( sy_pre, y_y, s_s, corrections,  y_s);
      //y_y_ := PBblas.PB_dgemm(TRUE, FALSE, 1.0,param_map, y_y,param_map, y_y,one_map );
      y_y_ := SumProduct (y_y, y_y);
      hdiag_updated := y_s/ y_y_;
      w_g_ := PBblas.PB_dscal(-1, w_g);
      // do the optimality (break condition) checks
      //~isLegal(d)
      IsLegald := IsLegal (d);
      legal_d_code := IF (IsLegald, -1, 1);
      //% Check that progress can be made along direction: if gtd > -tolX
      Isprog1 := gtd > -1 * tolX;
      prog1_code := IF (Isprog1, 2, -1 );
      // % Check Optimality Condition : sum(abs(g)) <= tolFun
      Isprog2 := sum_abs(w_g)<= tolFun;
      opt1_code := IF (Isprog2, 3, -1);
      //% ******************* Check for lack of progress ******************* : if sum(abs(t*d)) <= tolX
      Islack1 := ABS(w_t)*sum_abs(d)<= tolX;
      lack1_code := IF (Islack1, 4, -1);
      //if abs(f-f_old) < tolX
      Islack2 := ABS(w_f-f_pre) < tolX;
      lack2_code := IF (Islack2, 5, -1);
      // Check for going over iteration/evaluation limit *******************
      // if funEvals > maxFunEvals
      updated_funEvals := w_feval + funmin_pre;
      Isoverfun := updated_funEvals > maxFunEvals;
      fun_code := IF (Isoverfun, 6, -1 );
      break_code := IF(Isprog2, 3, IF (Islack1, 4, IF (Islack2, 5, IF(Isoverfun, 6, -1) )));
      //If breakcode!=-1 it means that we will exit the loop so there is no need to update hdiag and sy
      no_part_break := inp_ (no <= 2*k) + Pbblas.MU.TO(w_g, k_-1) + PBblas.MU.TO(x_updated, k_);
      to_return_break := PROJECT(no_part_break, TRANSFORM(minfrec, SELF.h := hDiag_pre; SELF.f := w_f, SELF.min_funEval:=updated_funEvals, SELF.break_cond := break_code, SELF:= LEFT));
      //IF none of break conditions are satisfied, we will continue the loop. Means that we have to update hdiag and sy values for the next iteration
      max_no := MAX(sy_updated, sy_updated.no);
      no_part := sy_updated + Pbblas.MU.TO(w_g, max_no+1)+ PBblas.MU.TO(x_updated, max_no+2);
      to_return := PROJECT(no_part, TRANSFORM(minfrec, SELF.h := hdiag_updated; SELF.f := w_f, SELF.min_funEval:=updated_funEvals, SELF.break_cond := break_code, SELF:= LEFT));
      //After d is calculated check whether it is legal, if itis legal continue, otherwise return with the break_cond=1
      dnot_legal_return := PROJECT(inp, TRANSFORM(minfrec, SELF.break_cond := 1, SELF:= LEFT ));
      Noprogalong_return := PROJECT(inp, TRANSFORM(minfrec, SELF.break_cond := 2, SELF:= LEFT ));
      FinalResult := IF (IsLegald, IF (Isprog1, Noprogalong_return, IF (break_code!=-1, to_return_break, to_return)), dnot_legal_return);
      //RETURN IF (EXISTS(inp),FinalResult,inp); orig
      RETURN FinalResult;
     //RETURN d;
     //RETURN PROJECT(inp, TRANSFORM(minfrec, SELF.h := w[1].t_, SELF:= LEFT ));
     // RETURN w;

      //RETURN (IF (coun=6,PROJECT(Pbblas.MU.TO(d, 100), TRANSFORM(minfrec, SELF.h := hdiag_updated; SELF.f := w_f, SELF.min_funEval:=w_feval, SELF.break_cond := break_code, SELF:= LEFT)) , FinalResult));
      //RETURN FinalResult;
    END;// min_step
    
    //min_result := LOOP(topass_min, LEFT.break_cond=-1, COUNTER <= 14 AND EXISTS(ROWS(LEFT)) , min_step(ROWS(LEFT),COUNTER)); orig
    break_table := TABLE(min_result_firstitr, {break_cond}, LOCAL);
    first_itr_break := break_table[1].break_cond;
    min_result_nextitr := LOOP(min_result_firstitr, MaxIter ,LEFT.break_cond=-1, min_step(ROWS(LEFT),COUNTER));      
    min_result := IF(first_itr_break!=-1, min_result_firstitr, min_result_nextitr);
    RETURN min_result;
    //RETURN  LOOP(min_result_firstitr, 400 ,LEFT.break_cond=-1, min_step(ROWS(LEFT),COUNTER));   
  END;// END MinFUNC_4
END;// END Optimization