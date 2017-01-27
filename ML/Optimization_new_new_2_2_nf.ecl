//this version is a newer version of Optimization_new_new_2_2 where I am trying to figure out why my program cause thor to crash
//#option ('divideByZero', 'nan');
IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat;
IMPORT * FROM ML.Types;
IMPORT PBblas;
IMPORT STD;
IMPORT std.system.Thorlib;
SHARED nodes_available := STD.system.Thorlib.nodes();
SHARED Layout_Cell_nid := RECORD (Pbblas.Types.Layout_Cell)
UNSIGNED4 node_id;
END;
Layout_Cell := PBblas.Types.Layout_Cell;
Layout_Part := PBblas.Types.Layout_Part;
matrix_t     := SET OF REAL8;
     OutputRecord := RECORD
      REAL8 t;
      REAL8 f_new;
      DATASET(Mat.Types.Element) g_new;
      INTEGER8 WolfeFunEval;
    END;
      
// A version of Optimization_new_new_2_2 where the costfuc has a different format than costfun in Optimization_new_new_2_2
// the train data is provided in Layout_Cell_nid which has been converted from numericfield format (_nf)
//Func : handle to the function we want to minimize it, its output should be the error cost and the error gradient
EXPORT Optimization_new_new_2_2_nf (UNSIGNED4 prows=0, UNSIGNED4 pcols=0, UNSIGNED4 Maxrows=0, UNSIGNED4 Maxcols=0) := MODULE
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
		SHARED 	costgrad_record := RECORD (Layout_Part)
			REAL8 cost_value;
	  END;
		
		EXPORT minfRec := RECORD (costgrad_record)
      REAL8 h ;//hdiag value
      INTEGER8 min_funEval;
      INTEGER break_cond ;
			REAL8 sty  ;
			PBblas.Types.t_mu_no no;
			INTEGER8 update_itr ; //This value is increased whenever a update is done and s and y vectors are added to the corrections. If no update is done due to the condition ys > 1e-10 then this value is not increased
			// we use this value to update the corrections vectors as well as in the lbfgs algorithm
			INTEGER8 itr_counter;
    END;
		//BoundProvided = 1 -> xminBound and xmaxBound values are provided
		//BoundProvided = 0 -> xminBound and xmaxBound values are not provided
		//set  f or g related values to 2 if f or g are not known ( f values are _2 values and g values are _3 values)
    // the order of the polynomial is the number of known f and g values minus 1. for example if first f (p1_2) does not exist the value of f1_2 will be equal to 0, otherwise it would be 1
		
    SHARED SumProduct (DATASET(Layout_Part) inp1, DATASET(Layout_Part) inp2) := FUNCTION
		  REAL8 sumpro(PBblas.Types.dimension_t N, PBblas.Types.matrix_t M, PBblas.Types.matrix_t V) := BEGINC++

				#body
				double result = 0;
				double tmpp ;
				double *cellm = (double*) m;
				double *cellv = (double*) v;
				uint32_t i;
				for (i=0; i<n; i++) {
					tmpp =(cellm[i] * cellv [i]);
					result = result + tmpp;
				}
				return(result);

			ENDC++;
      Product(REAL8 val1, REAL8 val2) := val1 * val2;
      Elem := {REAL8 v};  //short-cut record def
      Elem hadPart(Layout_Part xrec, Layout_Part yrec) := TRANSFORM //hadamard product
        SELF.v :=  sumpro(xrec.part_rows * xrec.part_cols, yrec.mat_part, xrec.mat_part);
      END;

      prod := JOIN(inp1, inp2, LEFT.partition_id=RIGHT.partition_id, hadPart(LEFT,RIGHT), LOCAL);
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
        REAL8 ro_val := 1/SUM(GROUP,ro_tmp.ro) ;
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
        REAL8 al_val;
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
    
		
		 EXPORT lbfgs_4_ (DATASET(Layout_Part) g, DATASET(PBblas.Types.MUElement) s, DATASET(PBblas.Types.MUElement) y, REAL8 Hdiag) := FUNCTION
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
        REAL8 ro_val := 1/SUM(GROUP,ro_tmp.ro) ;
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
        REAL8 al_val;
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
    END; // END lbfgs_4_
    //olsy include old s and y values, where s vectors have no=1 to no =k and y values have no=k+1 to k+k (k is the number of vector  we are storing by now (k<= corrections)
    //s is the new vector s which we need to add to the oldsy dataset
    //y is the new vector y which we need to add to the oldsy dataset
    //corrections is the lbfgs parameter value for the number of s and y vectore we are saving in the memory ( that's why it is called limited memory-bfgs), if we save all the s and y vector it would be bfgs 
    EXPORT lbfgsUpdate_before ( DATASET(PBblas.Types.MUElement) oldsy, DATASET(Layout_Part) y,DATASET(Layout_Part) s , INTEGER8 corrections, REAL8 ys) := FUNCTION
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
    END; // END lbfgsUpdate_before 
    
    
		
		MU_nID_REC := RECORD (PBblas.Types.MUElement)
		UNSIGNED2 nID := 0;
		END;
EXPORT lbfgsUpdate_ ( DATASET(PBblas.Types.MUElement) oldsy, DATASET(Layout_Part) y,DATASET(Layout_Part) s , INTEGER8 corrections, REAL8 ys, UNSIGNED nodes_used_vector, UNSIGNED itr_num ) := FUNCTION
			old_sy := PROJECT (oldsy, TRANSFORM(MU_nID_REC, SELF:= LEFT));
      k := IF (itr_num < corrections, itr_num, corrections);// The number of vectors (either s or y) stored by now
      K_corr := k<corrections;
      s_new_ind := IF (K_corr, k+1, k);
      y_new_ind := IF (K_corr, 2*(k+1), 2*k);
      s_new_no := PBblas.MU.TO(s,s_new_ind);
      y_new_no := PBBlas.MU.TO(y, y_new_ind);
			//IF the dataset is not still full (k<corrections)
      MU_nID_REC snew_trans (Layout_Part old) := TRANSFORM
        SELF.no  := s_new_ind;
				SELF.nID := (old.node_id + (s_new_ind-1)*nodes_used_vector) % nodes_available;
        SELF     := old;
      END;
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
    END; // END lbfgsUpdate_ 		
		
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
  
	
	EXPORT  polyinterp_noboundry_3points_imag2_3 (REAL8 t_1, REAL8 f_1, REAL8 gtd_1, REAL8 t_2, REAL8 f_2, REAL8 t_3, REAL8 f_3) := FUNCTION
		order := 4-1;
		xminBound := MIN ([t_1, t_2, t_3]);
		xmaxBound := MAX ([t_1, t_2, t_3]);
		// A= [t_1^3 t_1^2 t_1 1
		//    t_2^3 t_2^2 t_2 1
		//    3*t_1^2 2*t_1 t_1 0
		//    3*t_2^2 2*t_2 t_2 0]
		//b = [f_1 f_2 dtg_1 gtd_2]'
		Aset := [POWER(t_1,3),POWER(t_2,3), POWER(t_3,3),3*POWER(t_1,2),
		POWER(t_1,2),POWER(t_2,2), POWER(t_3,2),2*t_1,
		POWER(t_1,1),POWER(t_2,1), POWER(t_3,1), 1,
		1, 1, 1, 0]; // A 4*4 Matrix      
		Bset := [f_1, f_2, f_3, gtd_1]; // A 4*1 Matrix
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
	END;
	
	//polyinterp when no boundry values are provided
 //polyinterp when 2 points are provided and gtd2 and gtd3 are imaginary

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
	// no boundry is provided and gtd2 is imaginary value.	it is used in armijo backtracking.
	EXPORT  polyinterp_noboundry_img2 (REAL8 t_1, REAL8 f_1, REAL8 gtd_1, REAL8 t_2, REAL8 f_2) := FUNCTION
		order := 3-1; // number of known f ang gtd values (3) minus 1
		xmin := IF (t_1 < t_2, t_1, t_2);
		xmax := IF (t_1 < t_2, t_2, t_1);
		xminBound := xmin;
		xmaxBound := xmax;
		// build A and b Matrices based on available function values and gtd values
		// here A will be a 3(number of available f and gtd values)*3(order+1) matrix and b will be a 3(number of available f and gtd values)*1 array
		A := [POWER(t_1,2), POWER (t_2, 2), 2*t_1, t_1, t_2, 1, 1, 1, 0];
		b := [f_1,f_2,gtd_1];
		// Find interpolating polynomial
    //params = A\b;
    params_partset := PBblas.BLAS.solvelinear (A, B, 3,1,3,3);
		params1 := params_partset[1];
		params2 := params_partset[2];
		params3 := params_partset[3];
		
		dParams1 := 2*params_partset[1];
    dparams2 := params_partset[2];
		// Compute Critical Points
		root := (-1*dparams2)/dParams1;
		cp1 := xminBound;
		cp2 := xmaxBound;
		cp3 := t_1;
		cp4 := t_2;
		cp5 := root;
		ISrootsreal := TRUE;
		//Test Critical Points
		fmin1 := 100000000;
    minpos1 := (xminBound+xmaxBound)/2;
    xCP1 := cp1;
    cond1_1 := xCP1 >= xminBound AND xCP1 <= xmaxBound; //???
    // fCP = polyval(params,xCP);
    fCP1 := params1*POWER(xCP1,2)+params2*xCP1+params3;
		//if imag(fCP)==0 && fCP < fmin
		cond1_2 := TRUE; // If the root is real real so is FCP
		minpos2 := IF (cond1_1,IF (cond1_2, xCP1, minpos1),minpos1);
    fmin2 := IF (cond1_1,IF (cond1_2, fCP1, fmin1),fmin1);
        
		xCP2 := cp2;
		cond2_1 := xCP2 >= xminBound AND xCP2 <= xmaxBound; //???
		// fCP = polyval(params,xCP);
		fCP2:= params1*POWER(xCP2,2)+params2*xCP2+params3;
		//if imag(fCP)==0 && fCP < fmin
		cond2_2 := (fCP2<fmin2) AND ISrootsreal;
		minpos3 := IF (cond2_1,IF (cond2_2, xCP2, minpos2),minpos2);
		fmin3 := IF (cond2_1,IF (cond2_2, fCP2, fmin2),fmin2);
		xCP3 := cp3;
		cond3_1 := xCP3 >= xminBound AND xCP3 <= xmaxBound; //???
		// fCP = polyval(params,xCP);
		fCP3:= params1*POWER(xCP3,2)+params2*xCP3+params3;
		//if imag(fCP)==0 && fCP < fmin
		cond3_2 := (fCP3<fmin3) AND ISrootsreal; // If the roots are imaginary so is FCP
		minpos4 := IF (cond3_1,IF (cond3_2, xCP3, minpos3),minpos3);
		fmin4 := IF (cond3_1,IF (cond3_2, fCP3, fmin3),fmin3);
		xCP4 := cp4;
		cond4_1 := xCP4 >= xminBound AND xCP4 <= xmaxBound; //???
		// fCP = polyval(params,xCP);
		fCP4:= params1*POWER(xCP4,2)+params2*xCP4+params3;
		//if imag(fCP)==0 && fCP < fmin
		cond4_2 := (fCP4<fmin4) AND ISrootsreal; // If the roots are imaginary so is FCP
		minpos5 := IF (cond4_1,IF (cond4_2, xCP4, minpos4),minpos4);
		fmin5 := IF (cond4_1,IF (cond4_2, fCP4, fmin4),fmin4);
		xCP5 := cp5;
		cond5_1 := xCP5 >= xminBound AND xCP5 <= xmaxBound; //???
		// fCP = polyval(params,xCP);
		fCP5:= params1*POWER(xCP5,2)+params2*xCP5+params3;
		//if imag(fCP)==0 && fCP < fmin
		cond5_2 := (fCP5<fmin5) AND ISrootsreal; // If the roots are imaginary so is FCP
		minpos6 := IF (cond5_1,IF (cond5_2, xCP5, minpos5),minpos5);
		fmin6 := IF (cond5_1,IF (cond5_2, fCP5, fmin5),fmin5);

		RETURN fmin6;  
  END;//end polyinterp_noboundry_img2
	



	// no boundry is provided and gtd1 and gtd2 is imaginary value.
	EXPORT  polyinterp_noboundry_img1_2 (REAL8 t_1, REAL8 f_1, REAL8 t_2, REAL8 f_2) := FUNCTION
		order := 2-1; // number of known f ang gtd values (3) minus 1
		xmin := IF (t_1 < t_2, t_1, t_2);
		xmax := IF (t_1 < t_2, t_2, t_1);
		xminBound := xmin;
		xmaxBound := xmax;
		// build A and b Matrices based on available function values and gtd values
		// here A will be a 2(number of available f and gtd values)*2(order+1) matrix and b will be a 2(number of available f and gtd values)*1 array
		A := [POWER(t_1,2), POWER (t_2, 2), t_1, t_2, 1, 1];
		b := [f_1,f_2];
		// Find interpolating polynomial
    //params = A\b;
    params_partset := PBblas.BLAS.solvelinear (A, B, 2,1,2,2);
		params1 := params_partset[1];
		params2 := params_partset[2];
		
		cp1 := xminBound;
		cp2 := xmaxBound;
		cp3 := t_1;
		cp4 := t_2;
		
		ISrootsreal := TRUE;
		//Test Critical Points
		fmin1 := 100000000;
    minpos1 := (xminBound+xmaxBound)/2;
    xCP1 := cp1;
    cond1_1 := xCP1 >= xminBound AND xCP1 <= xmaxBound; //???
    // fCP = polyval(params,xCP);
    fCP1 := params1*xCP1+params2;
		//if imag(fCP)==0 && fCP < fmin
		cond1_2 := TRUE; // If the root is real real so is FCP
		minpos2 := IF (cond1_1,IF (cond1_2, xCP1, minpos1),minpos1);
    fmin2 := IF (cond1_1,IF (cond1_2, fCP1, fmin1),fmin1);
        
		xCP2 := cp2;
		cond2_1 := xCP2 >= xminBound AND xCP2 <= xmaxBound; //???
		// fCP = polyval(params,xCP);
		fCP2:= params1*xCP2+params2;
		//if imag(fCP)==0 && fCP < fmin
		cond2_2 := (fCP2<fmin2) AND ISrootsreal;
		minpos3 := IF (cond2_1,IF (cond2_2, xCP2, minpos2),minpos2);
		fmin3 := IF (cond2_1,IF (cond2_2, fCP2, fmin2),fmin2);
		xCP3 := cp3;
		cond3_1 := xCP3 >= xminBound AND xCP3 <= xmaxBound; //???
		// fCP = polyval(params,xCP);
		fCP3:= params1*xCP3+params2;
		//if imag(fCP)==0 && fCP < fmin
		cond3_2 := (fCP3<fmin3) AND ISrootsreal; // If the roots are imaginary so is FCP
		minpos4 := IF (cond3_1,IF (cond3_2, xCP3, minpos3),minpos3);
		fmin4 := IF (cond3_1,IF (cond3_2, fCP3, fmin3),fmin3);
		xCP4 := cp4;
		cond4_1 := xCP4 >= xminBound AND xCP4 <= xmaxBound; //???
		// fCP = polyval(params,xCP);
		fCP4:= params1*xCP4+params2;
		//if imag(fCP)==0 && fCP < fmin
		cond4_2 := (fCP4<fmin4) AND ISrootsreal; // If the roots are imaginary so is FCP
		minpos5 := IF (cond4_1,IF (cond4_2, xCP4, minpos4),minpos4);
		fmin5 := IF (cond4_1,IF (cond4_2, fCP4, fmin4),fmin4);


		RETURN fmin5;  
  END;//end polyinterp_noboundry_img1_2

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
  
		SHARED arm_t_rec := RECORD
			real8 init_arm_t;
			Layout_part.partition_id;
		END;
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
		
		SHARED  bracketing_record4_4 := RECORD(Layout_Part)
        INTEGER8 id; // id=1 means prev values, id=2 means new values
        REAL8 f_;
        REAL8 t_;
				REAL8 t_next;
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
    SHARED zooming_record4_4 := RECORD (bracketing_record4_4)
		  REAL8 t_high := -1;
			REAL8 f_high := -1;
			REAL8 gtd_high := -1;
      BOOLEAN insufProgress := FALSE;
      BOOLEAN LoopTermination := FALSE;
    END;
		
		SHARED ZoomingRecord := RECORD (CostGrad_Record)
			INTEGER id;
			REAL8 prev_t;
			REAL8 prev_gtd;
			INTEGER wolfe_funEvals;
			UNSIGNED8 c;
			INTEGER bracketing_cond;
			INTEGER zooming_cond := 0;
			REAL8 next_t;
			REAL8 high_t;
			REAL8 high_cost_value;
			REAL8 high_gtd;
			REAL8 glob_f; // this is the f value we recive through wolfelinesearch function call
			BOOLEAN insufProgress;
			BOOLEAN zoomtermination;
		END;
EXPORT WolfeLineSearch4(INTEGER cccc, DATASET(Layout_Part) x, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Cell_nid) TrainData , DATASET(Layout_Part) TrainLabel,DATASET(PBblas.Types.MUElement) CostFunc (DATASET(Layout_Part) x, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Cell_nid) TrainData , DATASET(Layout_Part) TrainLabel), UNSIGNED param_num, REAL8 t, DATASET(Layout_Part) d, REAL8 f, DATASET(Layout_Part) g, REAL8 gtd, REAL8 c1=0.0001, REAL8 c2=0.9, INTEGER maxLS=25, REAL8 tolX=0.000000001):=FUNCTION
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
    R1 := PROJECT(g_prev, load_scalars_g_prev(LEFT),LOCAL );
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
    R2 := PROJECT(g_new, load_scalars_g_new(LEFT),LOCAL );
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
    inputp_con :=  PROJECT(inputp, TRANSFORM(bracketing_record4, SELF.Which_cond := WhichCon; SELF:=LEFT),LOCAL);
    
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
      R_id_1 := PROJECT(inputp(id=2), load_scalars_g_prev(LEFT),LOCAL ); //id value is changed from 2 to 1. It is the same as :  f_prev = f_new;g_prev = g_new; gtd_prev = gtd_new; : the new values in the current loop iteration are actually prev values for the next iteration

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
      R_id_2 := PROJECT(gNewbrack, load_scalars_g_new(LEFT),LOCAL ); // scalar values are wrapped around gNewbrack with id=2 , these are actually the new values for the next iteration
     
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
   toto := PROJECT (BracketingResult, TRANSFORM(zooming_record4 ,SELF.LoopTermination:=(LEFT.which_cond=2) | (LEFT.c=MaxLS); SELF := LEFT)); // If in the bracketing step condition 2 has been met or we have reaached MaxLS then we don't need to pass zoom step, so the termination condition for zoom will be set as true here
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
      R_HI_id := PROJECT(gNewZoom, load_scalars_g_new(LEFT),LOCAL );
      zooming_record4 load_scalars_LOID (zooming_record4 l) := TRANSFORM
        SELF.funEvals_ := zoom_funevals_new;
        SELF.c := zoom_c_new; //Counter
        SELF.insufProgress := insufprog_new;
        SELF.which_cond := whichcond;
        SELF.LoopTermination := zoomter;
        SELF := l;
      END;
      R_LO_id := PROJECT (inputp (id=LO_id), load_scalars_LOID(LEFT),LOCAL );
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
      R_HI_id := PROJECT(inputp(id=HI_id), HIID(LEFT),LOCAL );
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
      R_LO_id := PROJECT(gNewZoom, load_scalars_g_new(LEFT),LOCAL );
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
      R_HI_id := PROJECT(inputp(id=LO_id), LOID(LEFT) ,LOCAL);
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
      R_LO_id := PROJECT(gNewZoom, load_scalars_g_new(LEFT) ,LOCAL);
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
  Topass_zooming := PROJECT (BracketingResult, TRANSFORM(zooming_record4 ,SELF.LoopTermination:=(LEFT.which_cond=2) | (LEFT.c=MaxLS); SELF := LEFT),LOCAL); // If in the bracketing step condition 2 has been meet or we have reaached MaxLS then we don't need to pass zoom step, so the termination condition for soom will be set as true here
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
   t_new_result := PROJECT (ZoomingResult (id=2), TRANSFORM(bracketing_record4 ,SELF := LEFT),LOCAL);
   t_old_result := PROJECT (ZoomingResult (id=1), TRANSFORM(bracketing_record4 ,SELF := LEFT),LOCAL);
   t_0_result := PROJECT(g, load_scalars_g(LEFT),LOCAL );
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


EXPORT WolfeLineSearch4_4_2_test(INTEGER cccc, DATASET(Layout_Part) x, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Cell_nid) TrainData , DATASET(Layout_Part) TrainLabel,DATASET(CostGrad_Record) CostFunc (DATASET(Layout_Part) x, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Cell_nid) TrainData , DATASET(Layout_Part) TrainLabel), REAL8 t, DATASET(Layout_Part) d, DATASET(costgrad_record) g, REAL8 gtd, REAL8 c1=0.0001, REAL8 c2=0.9, INTEGER maxLS=25, REAL8 tolX=0.000000001):=FUNCTION
	//C++ functions used
	//sum (M.*V)
	REAL8 sump(PBblas.Types.dimension_t N, PBblas.Types.matrix_t M, PBblas.Types.matrix_t V) := BEGINC++
		#body
		double result = 0;
		double tmpp ;
		double *cellm = (double*) m;
		double *cellv = (double*) v;
		uint32_t i;
		for (i=0; i<n; i++) {
			tmpp =(cellm[i] * cellv [i]);
			result = result + tmpp;
		}
		return(result);

	ENDC++;
	
	// We pass the prev g and cost values along with the next t value to the bracketing loop. The new g and cost values will be calculated at the begining of the loop
	//id field shows which side of the bracket prev values belong to (left (1) or right (2))
	//we embed the f value recived in wolfe function in what we pass to bracketing loop, so in the rest of the algorithm we have access to it in each node -> SELF.glob_f :=LEFT.cost_value
	topass_bracketing := PROJECT (g, TRANSFORM (ZoomingRecord, SELF.zooming_cond := -1; SELF.zoomtermination := FALSE; SELF.insufProgress := FALSE; SELF.id := 1; SELF.glob_f :=LEFT.cost_value; SELF.high_t := -1 ; SELF.high_gtd := -1; SELF.high_cost_value := -1; SELF.prev_t := 0; SELF.prev_gtd := gtd; SELF.wolfe_funEvals := 0; SELF.c := 0; SELF.bracketing_cond := -1; SELF.next_t := t; SELF := LEFT), LOCAL);
	BracketingStep (DATASET (ZoomingRecord) inputp, UNSIGNED coun) := FUNCTION
		//the idea is to calculate the new values at the begining of the bracket and then JOIN the loop input and this new values in order to generate the loop output. The JOIN TRANSFORM contains all the condition checks
		// calculate new g and cost value at the begining of the loop using next_t value in the input dataset
		// calculate t*d
		Elem := {REAL8 v};
		Layout_Part td_tran (inputp le, d ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := PBBlas.BLAS.dscal(cells, le.next_t, ri.mat_part, 1);
			SELF := le;
		END;
		td := JOIN (inputp, d, LEFT.partition_id = RIGHT.partition_id, td_tran (LEFT, RIGHT), LOCAL);
		//calculate x_new = x0 + td
		Layout_Part x_new_tran (Layout_part le, Layout_part ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := PBblas.BLAS.daxpy(cells, 1.0, le.mat_part, 1, ri.mat_part, 1);
			SELF := le;
		END;
		x_new := JOIN (x, td, LEFT.partition_id = RIGHT.partition_id, x_new_tran(LEFT, RIGHT), LOCAL);
		// calculated new g and cost values
		CostGrad_new := CostFunc(x_new,CostFunc_params,TrainData, TrainLabel);
		//calculated gtd_new (it will be used in IF conditions)
		Elem gtd_tran (CostGrad_new le, d ri) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.v := sump(cells, le.mat_part, ri.mat_part);
		END;
		gtd_new_ := JOIN (CostGrad_new, d, LEFT.partition_id = RIGHT.partition_id, gtd_tran (LEFT, RIGHT), LOCAL);
		gtd_new := SUM (gtd_new_, gtd_new_.v);
		//now that new values are calculated, JOIN prev (inputp) values and the new values considering all the conditions in order to produce the loop results
		//based on which condition is satisfied and whether previous cost (the cost in inputp) is less than the newly calculated cost (CostGrad_new) the return result is decided
		//we always return the braket side with smaller cost value, the values for the Hi sie of the bracket are in high_ fields. we don't return gradient vector for the high side
		BrackLSiter := coun - 1;
		ZoomingRecord main_tran (inputp le_prev, CostGrad_new ri_new) := TRANSFORM
			// calculate t for the next iteration (this value only will be calculated if there is going to be a next iteration)
			f := le_prev.glob_f; // the f value recived by wolfe function
			newt := le_prev.next_t; // the new values are calculated based on next_t so next_t is actually the new t in this iteration
			fNew := ri_new.cost_value;
			gtdNew := gtd_new;
			tPrev := le_prev.prev_t;
			fPrev := le_prev.cost_value;
			gtdPrev := le_prev.prev_gtd;
			minstep := newt + 0.01* (newt-tPrev);
			maxstep := newt*10;
			nextitr_t := polyinterp_both (tPrev, fPrev,gtdPrev, newt, fNew, gtdNew, minstep, maxstep);
			//specify the conditions along that which cost value is smaller
			cost_cond :=  (le_prev.cost_value < ri_new.cost_value);
			// f_new > f + c1*t*gtd || (LSiter > 1 && f_new >= f_prev)
			//cond1 := ((fNew > f + c1 * newt* gtd)| ((BrackLSiter > 1) & (fNew >= fPrev))) & (coun < (maxLS+1) );
			cond1 := FALSE;
			// abs(gtd_new) <= -c2*gtd
			//cond2 := (ABS(gtdNew) <= (-1*c2*gtd)) & (coun < (maxLS+1) );
			cond2 := FALSE;
			// gtd_new >= 0
			cond3 := TRUE;
			//cond3 := (gtdNew >= 0 ) &(coun < (maxLS+1) ); // by adding & (coun < (maxLS+1) ) at the end of each condition I make sure that at the very last iteration no condition is satisfied and i just return the new value which make my algorithm works like the matlab algorithm
			//  I calculate new values at the begining of each loop itr, where the matlab code calculates it at end
		  // so at the maxLS I calculate next iteration t and go to the next iteration, then I calculate the new value which corresponds to new value calculated at the end of maxLS iteration in the matlab code, then I make sure no condition is satified
			// and I just return the new value at the (maxLS+1)  iteration which corresponds to new value at the end of maxLS iteration in matlab code
			WhichCond := IF (coun = (MAXLS+1),4 , IF (cond1, 1, IF(cond2, 2, IF (cond3,3,-1)))); // if we reach the maximum number of iterations or if one of the conditions is satified then we loopfilter retunr empty set and we exit the loop
			//decide whether we should return prev or new values
			prev_mat := le_prev.mat_part;
			new_mat := ri_new.mat_part;
			prev_num := 1;
			new_num := 2;
			prev_or_new := IF (cond1, (IF (cost_cond, prev_num, new_num)) , IF (cond2, new_num, IF (cond3, IF (cost_cond, prev_num, new_num) , new_num)));
			SELF.mat_part := IF (prev_or_new = 1 , prev_mat, new_mat);
			SELF.cost_value := IF (prev_or_new = 1 , fPrev, fNew);
			SELF.high_cost_value := IF (prev_or_new = 1 , fNew, fPrev);
			SELF.prev_gtd := IF (prev_or_new = 1 , gtdPrev, gtdNew);
			SELF.high_gtd := IF (prev_or_new = 1 , gtdNew, gtdPrev);
			SELF.prev_t := IF (prev_or_new = 1 , tPrev, newt);
			SELF.high_t := IF (prev_or_new = 1 , newt, tPrev);
			SELF.id := prev_or_new; // thd id shows whether the associated value to prev is the left or riht side of the bracket, the hight value will be the other value of id (if id=1 it means the prev values start the bracket and high ends the brackte) 
			//actually the id only matters when we exit the loop because of cond1 or cond3. cond2 and reaching maxitr casue program to end so id does not matter becasue we don't go to zooming loop
			SELF.bracketing_cond := WhichCond;
			SELF.c := coun;
			SELF.next_t := IF (WhichCond = -1 AND coun < (maxLS+1), nextitr_t, -1);// next_t is calculated only if we are going to the next iteration which means WhichCond should be -1 ( no condition has been satisfied) and we have not rached the Max number of iterations
			SELF.wolfe_funEvals := le_prev.wolfe_funEvals + 1;
			SELF := le_prev;
		END;
		Result := JOIN (inputp, CostGrad_new, LEFT.partition_id = RIGHT.partition_id, main_tran(LEFT, RIGHT), LOCAL);
		RETURN Result;
	END; // END BracketingStep
	ZoomingRecord maxitr_tran (ZoomingRecord le_brack , CostGrad_record ri_g) := TRANSFORM
		itr_num := le_brack.c;
		maxitr_cond := (itr_num = (maxLS+1));
		cost_cond_0 := ri_g.cost_value < le_brack.cost_value;
		SELF.mat_part := IF (maxitr_cond , IF (cost_cond_0, ri_g.mat_part, le_brack.mat_part) , le_brack.mat_part);
		SELF.cost_value := IF (maxitr_cond , IF (cost_cond_0, ri_g.cost_value, le_brack.cost_value) , le_brack.cost_value);
		SELF.prev_t := IF (maxitr_cond , IF (cost_cond_0, 0 , le_brack.prev_t) , le_brack.prev_t);
		SELF.prev_gtd := IF (maxitr_cond , IF (cost_cond_0, gtd, le_brack.prev_gtd) , le_brack.prev_gtd);
		SELF.zoomtermination := maxitr_cond OR (le_brack.bracketing_cond=2); // if we reach maximum number of iterations in the bracketing loop, or if condition 2 is satisfied in the bracketing loop. it means we don't go to the zooming loop and the wolfe algorithms ends
		SELF := le_brack;
	END;
	//bracketing_result_ := LOOP(topass_bracketing, LEFT.bracketing_cond = -1, EXISTS(ROWS(LEFT)) and COUNTER < (maxLS+1), BracketingStep(ROWS(LEFT), COUNTER)); orig, not work
	//bracketing_result_ := LOOP(topass_bracketing, (maxLS+1), LEFT.bracketing_cond = -1, BracketingStep(ROWS(LEFT), COUNTER)); works but runs maxls+1 time
	bracketing_result_ := LOOP(topass_bracketing, LEFT.bracketing_cond = -1, BracketingStep(ROWS(LEFT), COUNTER));
	//check if MAXITR is reached and decide between bracketing result and f (t=0) to be returned
	bracketing_result := JOIN (bracketing_result_ , g, LEFT.partition_id = RIGHT.partition_id, maxitr_tran (LEFT, RIGHT), LOCAL);//?? it should be the loop result not topassbracketing
	topass_zooming := bracketing_result;
	ZoomingStep (DATASET (ZoomingRecord) inputp, UNSIGNED coun) := FUNCTION
		//similar to BracketingStep, first we calculate the new point (for that we need to calculate the new t using interpolation)
		//after the new point is calculated we JOIN the input and the new point in order to decide which one to return
		// Compute new trial value & Test that we are making sufficient progress
		//The following transform calculates the new trial value as well as insufProgress value by doing a PROJECT on the input, the new t is then saved in next_t field
		ZoomingRecord newt_tran (ZoomingRecord le) := TRANSFORM
			insufprog := le.insufProgress;
			LO_bracket := le.prev_t;
			HI_bracket := le.high_t; 
			HI_f := le.high_cost_value;
			HI_gtd := le.high_gtd;
			BList := [LO_bracket,HI_bracket];
			max_bracket := MAX(Blist);
			min_bracket := MIN(Blist);
			current_LO_id := le.id;
			gtd_LO := le.prev_gtd;
			new_id := IF (current_LO_id=1,2,1);
			min_bracketFval := le.cost_value;
			tTmp1 := polyinterp_noboundry (LO_bracket, min_bracketFval, gtd_LO, HI_bracket, HI_f, HI_gtd);
			tTmp2 := polyinterp_noboundry (HI_bracket, HI_f, HI_gtd, LO_bracket, min_bracketFval, gtd_LO);
			tTmp := IF (current_LO_id = 1, tTmp1, tTmp2);
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
			SELF.next_t := tZoom;
			SELF.insufProgress := insufprog_new;
			SELF := le;
		END;
		inputp_t_insuf := PROJECT (inputp, newt_tran(LEFT) ,LOCAL);//tzoom and insufficient progres values are calculated and embedded in the inputp dataset
		//evaluate new point
		// calculate t*d
		Elem := {REAL8 v};
		Layout_Part td_tran (inputp_t_insuf le, d ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := PBBlas.BLAS.dscal(cells, le.next_t, ri.mat_part, 1);
			SELF := le;
		END;
		td := JOIN (inputp_t_insuf, d, LEFT.partition_id = RIGHT.partition_id, td_tran (LEFT, RIGHT), LOCAL);
		// calculate x_new = x0 + td
		Layout_Part x_new_tran (Layout_part le, Layout_part ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := PBblas.BLAS.daxpy(cells, 1.0, le.mat_part, 1, ri.mat_part, 1);
			SELF := le;
		END;
		x_new := JOIN (x, td, LEFT.partition_id = RIGHT.partition_id, x_new_tran(LEFT, RIGHT), LOCAL);
		// calculated new g and cost values
		CostGrad_new := CostFunc(x_new,CostFunc_params,TrainData, TrainLabel);
		Elem gtd_tran (CostGrad_new le, d ri) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.v := sump(cells, le.mat_part, ri.mat_part);
		END;
		gtd_new_ := JOIN (CostGrad_new, d, LEFT.partition_id = RIGHT.partition_id, gtd_tran (LEFT, RIGHT), LOCAL);
		gtd_new := SUM (gtd_new_, gtd_new_.v);
		ZoomingRecord main_tran (inputp le_prev, CostGrad_new ri_new) := TRANSFORM
			f := le_prev.glob_f;
			current_LO_id := le_prev.id;
			new_id := IF (current_LO_id=1,2,1);
			fNewZoom := ri_new.cost_value;
			tZoom := le_prev.next_t;// the new point is calculates based on next_t value which was calculated at the begining of the zoom loop
			min_bracketFval := le_prev.cost_value;
			LO_bracket := le_prev.prev_t;
			HI_bracket := le_prev.high_t;
			gtdNewZoom := gtd_new;
			// f_new > f + c1*t*gtd || f_new >= f_LO
			//zoom_cond_1 := (fNewZoom > f + c1 * tZoom * gtd) | (fNewZoom >= min_bracketFval);
			zoom_cond_1 := FALSE;
			// abs(gtd_new) <= - c2*gtd
			// zoom_cond_2 := ABS (gtdNewZoom) <= (-1 * c2 * gtd);
			zoom_cond_2 := FALSE;
			// gtd_new*(bracket(HIpos)-bracket(LOpos)) >= 0
			// zoom_cond_3 := gtdNewZoom * (HI_bracket - LO_bracket) >= 0; 
			zoom_cond_3 := FALSE;
			whichcond := IF (zoom_cond_1, 11, IF (zoom_cond_2, 12, IF (zoom_cond_3, 13, -1)));
			cost_cond := min_bracketFval < fNewZoom; // f_lo < f_new
			lo_num := 1;
			new_num := 2;
			high_num := 3;
			lo_or_new := IF (zoom_cond_1, IF (cost_cond, lo_num, new_num), IF (zoom_cond_2, new_num, IF (zoom_cond_3, IF (cost_cond, lo_num, new_num), new_num)) );
			lo_or_new_id := IF (zoom_cond_1, IF (cost_cond, current_LO_id, new_id), IF (zoom_cond_2, current_LO_id, IF (zoom_cond_3, IF (cost_cond, new_id, current_LO_id), current_LO_id)) );
			lo_gtd := le_prev.prev_gtd;
			lo_t := le_prev.prev_t;
			//~done && abs((bracket(1)-bracket(2))*gtd_new) < tolX	
			zoom_term_cond := (~zoom_cond_1 AND zoom_cond_2) OR (ABS((tZoom-LO_bracket)*gtdNewZoom) < tolX);//if zoom_term_cond=TRUE then no need to assigne high values
			pre_mat := le_prev.mat_part;
			new_mat := ri_new.mat_part;
			//SELF.mat_part := IF (lo_or_new =1 , le_prev.mat_part, ri_new.mat_part);
			SELF.mat_part := IF (zoom_cond_1, IF (cost_cond, pre_mat, new_mat),IF (zoom_cond_2, new_mat, IF (zoom_cond_3, IF (cost_cond, pre_mat, new_mat), new_mat)));
			SELF.cost_value := IF (zoom_cond_1, IF (cost_cond, min_bracketFval, fNewZoom),IF (zoom_cond_2, fNewZoom, IF (zoom_cond_3, IF (cost_cond, min_bracketFval, fNewZoom), fNewZoom)));
			SELF.high_cost_value := IF (zoom_cond_1, IF (cost_cond, fNewZoom, min_bracketFval),IF (zoom_cond_2, le_prev.high_cost_value, IF (zoom_cond_3, IF (cost_cond, fNewZoom, min_bracketFval), le_prev.high_cost_value)));
			SELF.prev_gtd := IF (zoom_cond_1, IF (cost_cond, lo_gtd, gtdNewZoom),IF (zoom_cond_2, gtdNewZoom, IF (zoom_cond_3, IF (cost_cond, lo_gtd, gtdNewZoom), gtdNewZoom)));
			SELF.high_gtd := IF (zoom_cond_1, IF (cost_cond, gtdNewZoom, lo_gtd),IF (zoom_cond_2, le_prev.high_gtd, IF (zoom_cond_3, IF (cost_cond, gtdNewZoom, lo_gtd), le_prev.high_gtd)));
			SELF.prev_t := IF (zoom_cond_1, IF (cost_cond, lo_t, tZoom),IF (zoom_cond_2, tZoom, IF (zoom_cond_3, IF (cost_cond, lo_t, tZoom), tZoom)));
			SELF.high_t := IF (zoom_cond_1, IF (cost_cond, tZoom, lo_t),IF (zoom_cond_2, le_prev.high_t, IF (zoom_cond_3, IF (cost_cond, tZoom, lo_t), le_prev.high_t)));
			SELF.id := IF (zoom_cond_1, IF (cost_cond, current_LO_id, new_id), IF (zoom_cond_2, current_LO_id, IF (zoom_cond_3, IF (cost_cond, new_id, current_LO_id), current_LO_id)) );//here
			SELF.c := le_prev.c + 1;
			SELF.zoomtermination := zoom_term_cond;
			SELF.wolfe_funEvals := le_prev.wolfe_funEvals + 1;
			SELF.zooming_cond := whichcond;
			SELF := le_prev;
		END;
		Result := JOIN (inputp_t_insuf, CostGrad_new, LEFT.partition_id = RIGHT.partition_id, main_tran(LEFT, RIGHT), LOCAL);		
		RETURN Result;
	END;
	//zooming_result := LOOP(topass_zooming, (LEFT.zoomtermination = FALSE) AND (LEFT.c < (maxLS+1)), EXISTS(ROWS(LEFT)) , ZoomingStep(ROWS(LEFT), COUNTER)); orig
	zooming_result := LOOP(topass_zooming, (LEFT.zoomtermination = FALSE) AND (LEFT.c < (maxLS+1)), ZoomingStep(ROWS(LEFT), COUNTER));
	//RETURN BracketingStep (topass_bracketing, 1);
	//RETURN zooming_result;
	//RETURN LOOP(topass_bracketing, 1 , BracketingStep(ROWS(LEFT), COUNTER));
	//RETURN LOOP(topass_bracketing, (maxLS+1), LEFT.bracketing_cond = -1, BracketingStep(ROWS(LEFT), COUNTER));
	//RETURN bracketing_result;// works
	//RETURN zooming_result;
	RETURN  ZoomingStep(topass_zooming, 1);
	//RETURN BracketingStep(topass_bracketing, 1);
	//RETURN LOOP(topass_bracketing, LEFT.bracketing_cond = -1, BracketingStep(ROWS(LEFT), COUNTER)); // works correctly
	//RETURN BracketingStep(topass_bracketing, 1); works correctly
	//RETURN LOOP(topass_bracketing, 1, BracketingStep(ROWS(LEFT), COUNTER));works correctly
	//RETURN topass_zooming;
	//RETURN LOOP(topass_zooming, (LEFT.zoomtermination = FALSE) AND (LEFT.c < (maxLS+1)), FALSE , ZoomingStep(ROWS(LEFT), COUNTER)); // works
	//RETURN LOOP(topass_zooming, (LEFT.zoomtermination = FALSE) AND (LEFT.c < (maxLS+1)) , ZoomingStep(ROWS(LEFT), COUNTER));//works
	// At the end the final result returned from wolfe function contains the t value in prev_t field, the g vector in the Layout_part section and the f value in cost_value field and the gtd in the prev_gtd
END;// END WolfeLineSearch4_4_2_test
//this ArmijoBacktrack is called from wolfelinesearch function, the initial t is provided in arm_t_rec format arm_t_rec
EXPORT ArmijoBacktrack_fromwolfe(DATASET(Layout_Part) x, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Cell_nid) TrainData , DATASET(Layout_Part) TrainLabel,DATASET(CostGrad_Record) CostFunc (DATASET(Layout_Part) x, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Cell_nid) TrainData , DATASET(Layout_Part) TrainLabel), DATASET(arm_t_rec) t, DATASET(Layout_Part) d, DATASET(costgrad_record) g, REAL8 gtd, REAL8 c1=0.0001, REAL8 c2=0.9, REAL8 tolX=0.000000001):=FUNCTION
  // C++ functions
	REAL8 sumabs(PBblas.Types.dimension_t N, PBblas.Types.matrix_t M) := BEGINC++

    #body
    double result = 0;
		double tmpp ;
    double *cellm = (double*) m;
    uint32_t i;
		for (i=0; i<n; i++) {
		tmpp = (cellm[i]>=0) ? (cellm[i]) : (-1 * cellm[i]);
      result = result + tmpp;
    }
		return(result);

   ENDC++;
	REAL8 sump(PBblas.Types.dimension_t N, PBblas.Types.matrix_t M, PBblas.Types.matrix_t V) := BEGINC++
		#body
		double result = 0;
		double tmpp ;
		double *cellm = (double*) m;
		double *cellv = (double*) v;
		uint32_t i;
		for (i=0; i<n; i++) {
			tmpp =(cellm[i] * cellv [i]);
			result = result + tmpp;
		}
		return(result);

	ENDC++;
  // Evaluate the Objective and Gradient at the Initial Step
	Elem := {REAL8 v};
	//calculate x_new = x + td
	//first calculate td
	Layout_Part td_tran (t le, d ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := PBBlas.BLAS.dscal(cells, le.init_arm_t, ri.mat_part, 1);
			SELF := ri;
		END;
		td := JOIN (t, d, LEFT.partition_id = RIGHT.partition_id, td_tran (LEFT, RIGHT), LOCAL);
		//calculate x_new = x0 + td
		Layout_Part x_new_tran (Layout_part le, Layout_part ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := PBblas.BLAS.daxpy(cells, 1.0, le.mat_part, 1, ri.mat_part, 1);
			SELF := le;
		END;
		x_new := JOIN (x, td, LEFT.partition_id = RIGHT.partition_id, x_new_tran(LEFT, RIGHT), LOCAL);
		// calculated new g and cost values at x_new
	CostGrad_new := CostFunc(x_new,CostFunc_params,TrainData, TrainLabel);
	Elem gtdnew_tran(CostGrad_new inrec, d drec) := TRANSFORM //hadamard product
		cells := inrec.part_rows * inrec.part_cols;
		SELF.v :=  sump(cells, inrec.mat_part, drec.mat_part);
	END;
	gtd_new_ := JOIN (CostGrad_new, d, LEFT.partition_id = RIGHT.partition_id, gtdnew_tran (LEFT, RIGHT), LOCAL);
	gtd_new := SUM (gtd_new_, gtd_new_.v);
	armfunEvals := 1;
	// a copy of each of the following values should be on each node in order to calculate the new t:
	// f_prev t_prev  f_new ->t_new
	ArmijoRecord := RECORD (CostGrad_Record)
		REAL8 fprev;
		REAL8 tprev;// this is the actualy previous t calculated in the previous iteration
		REAL8 prev_t;// this should actually be tnew, however in order for this record format to be consistent with the ourput of wolfe line search , the new t has to be named prev_t
		UNSIGNED wolfe_funevals;
		INTEGER armCond;
		REAL8 glob_f;
		REAL8 gtdnew;
		BOOLEAN islegal_gnew := TRUE;
		REAL8 local_sumd;
	END; // ArmijoRec
	ArmijoRecord_shorten := RECORD
		REAL8 fprev;
		REAL8 tprev;// this is the actualy previous t calculated in the previous iteration
		REAL8 prev_t;// this should actually be tnew, however in order for this record format to be consistent with the ourput of wolfe line search , the new t has to be named prev_t
		UNSIGNED wolfe_funevals;
		INTEGER armCond;
		REAL8 glob_f;
		REAL8 gtdnew;
		BOOLEAN islegal_gnew := TRUE;
		Layout_Part.partition_id;
		CostGrad_Record.cost_value;
		REAL8 local_sumd;
	END; // ArmijoRecord_shorten
	f_table := TABLE (g, {g.cost_value, g.partition_id}, LOCAL);
	topass_BackTracking_init := JOIN (CostGrad_new, t, LEFT.partition_id = RIGHT.partition_id , TRANSFORM(ArmijoRecord, SELF.fprev := -1; SELF.tprev := -1; SELF.prev_t := RIGHT.init_arm_t; SELF.wolfe_funevals:=0;SELF.armCond := -1; SELF.glob_f := -1 ; SELF.gtdnew := -1; SELF.local_sumd := -1; SELF:=LEFT), LOCAL);
	topass_BackTracking_ := JOIN (topass_BackTracking_init, f_table , LEFT.partition_id = RIGHT.partition_id, TRANSFORM (ArmijoRecord, SELF.gtdnew := gtd_new; SELF.glob_f := RIGHT.cost_value; SELF.armCond := IF (( LEFT.cost_value > RIGHT.cost_value + c1*LEFT.prev_t*gtd) OR (NOT ISVALID (LEFT.cost_value)), -1, 2);
	SELF.wolfe_funevals := armfunEvals; SELF.fprev := -1; SELF.tprev := -1; SELF.local_sumd := -1; SELF := LEFT), LOCAL);
	topass_BackTracking := JOIN (topass_BackTracking_, d, LEFT.partition_id = RIGHT.partition_id, TRANSFORM (ArmijoRecord, SELF.local_sumd:=IF (LEFT.armCond=-1, sumabs(RIGHT.part_rows * RIGHT.part_cols, RIGHT.mat_part), -1) , SELF:=LEFT), LOCAL);// if we enter the loop (armCond==-1) then we will need sumabs(t*d) = abs(t) * sumabs(d) inside the loop in each iteration. so we calculate the local sumabs(d) values here and use them in the loop as t*sum(sumabs_local(d))
	
	BackTracking (DATASET (ArmijoRecord) armin, UNSIGNED armc) := FUNCTION
		//calculate new t value
		ArmijoRecord_shorten newt_tran (ArmijoRecord le) := TRANSFORM
			temp := le.prev_t;
			f_new := le.cost_value;
			funEvals := le.wolfe_funevals;
			f_prev := le.fprev;
			//if LS == 0 || ~isLegal(f_Nnew), since in my implementation I assumed LS=4 we only need to check second condition
			cond1 := NOT isvalid(f_new);
			//LS == 2 && isLegal(g_new)
			cond2 := le.islegal_gnew;
			//funEvals < 2 || ~isLegal(f_prev)
			cond3 := (funEvals<2) OR (NOT ISVALID (f_prev));
			armt1 := temp*0.5;
			// t = polyinterp([0 f gtd; t f_new g_new'*d],doPlot);
			armt2 := polyinterp_noboundry (0, le.glob_f, gtd, temp, le.cost_value, le.gtdnew) ;
			//t = polyinterp([0 f gtd; t f_new sqrt(-1)],doPlot);
			armt3 := polyinterp_noboundry_img2 (0, le.glob_f, gtd, temp, le.cost_value);
			// t = polyinterp([0 f gtd; t f_new sqrt(-1); t_prev f_prev sqrt(-1)],doPlot);
			armtelse := polyinterp_noboundry_3points_imag2_3 (0, le.glob_f, gtd, temp, le.cost_value, le.tprev, f_prev) ;
			// armt_tmp := IF (cond1, armt1, IF (cond2, armt2, IF (cond3, armt3, armtelse)));
			armt_tmp := IF (cond1, IF ( armt1 < temp*0.001 , temp*0.001, IF (armt1 >= temp*0.6, temp*0.6, armt1))
			, IF (cond2, IF ( armt2 < temp*0.001 , temp*0.001, IF (armt2 >= temp*0.6, temp*0.6, armt2)), IF (cond3, IF ( armt3 < temp*0.001 , temp*0.001, IF (armt3 >= temp*0.6, temp*0.6, armt3)), 
			IF ( armtelse < temp*0.001 , temp*0.001, IF (armtelse >= temp*0.6, temp*0.6, armtelse)))));
			//Adjust if change in t is too small/large
			armt := IF ( armt_tmp < temp*0.001 , temp*0.001, IF (armt_tmp >= temp*0.6, temp*0.6, armt_tmp));
			// SELF.prev_t := armt;// the new t value
			SELF.prev_t := armt_tmp; // armt causes error, somehow when I was using the old sintax armt_tmp := IF (cond1, armt1, IF (cond2, armt2, IF (cond3, armt3, armtelse))); followed by armt := IF ( armt_tmp < temp*0.001 , temp*0.001, IF (armt_tmp >= temp*0.6, temp*0.6, armt_tmp)); 
			// th compiler would compile all armt"i" values and it was cuasing matrix not positive ... error
			SELF.tprev := temp;
			SELF.fprev := le.cost_value;
			SELF.wolfe_funevals := le.wolfe_funevals;
			SELF := le;
		END;
		armin_t := PROJECT (armin, newt_tran(LEFT), LOCAL);

		// calculate t*d
		Elem := {REAL8 v};
		Layout_Part td_tran (armin_t le, d ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := PBBlas.BLAS.dscal(cells, le.prev_t, ri.mat_part, 1);
			SELF := ri;
		END;
		armtd := JOIN (armin_t, d, LEFT.partition_id = RIGHT.partition_id, td_tran (LEFT, RIGHT), LOCAL);
		//calculate sum(abs(armtd))
		
		 Elem abs_td_tran (armin_t le) := TRANSFORM
				SELF.v:= ABS (le.prev_t) * le.local_sumd;
		 END;
		 sum_abs_td_ := PROJECT (armin_t, abs_td_tran(LEFT), LOCAL);
		 sum_abs_td := SUM (sum_abs_td_, sum_abs_td_.v);
		 //evaluate new point
		 // calculate x_new = x + td
		 Layout_Part x_new_tran (Layout_part le, Layout_part ri) := TRANSFORM
				cells := le.part_rows * le.part_cols;
				SELF.mat_part := PBblas.BLAS.daxpy(cells, 1.0, le.mat_part, 1, ri.mat_part, 1);
				SELF := le;
			END;
			armx_new := JOIN (x, armtd, LEFT.partition_id = RIGHT.partition_id, x_new_tran(LEFT, RIGHT), LOCAL);
			armCostGrad_new := CostFunc(armx_new,CostFunc_params,TrainData, TrainLabel);
			//calculate gtdnew
			Elem armgtdnew_tran(armCostGrad_new inrec, d drec) := TRANSFORM //hadamard product
			cells := inrec.part_rows * inrec.part_cols;
				SELF.v :=  sump(cells, inrec.mat_part, drec.mat_part);
			END;
			armgtd_new_ := JOIN (armCostGrad_new, d, LEFT.partition_id = RIGHT.partition_id, gtdnew_tran (LEFT, RIGHT), LOCAL);
			armgtd_new := SUM (armgtd_new_, armgtd_new_.v);
			
			ArmijoRecord armmain_tran (armCostGrad_new le, armin_t ri) := TRANSFORM
			//while f_new > fr + c1*t*gtd || ~isLegal(f_new)
			SELF.armCond := IF ( (le.cost_value > (ri.glob_f + (c1*ri.prev_t*gtd)) OR (NOT ISVALID(le.cost_value))) ,-1,2);
			SELF.tprev := ri.tprev;
			SELF.fprev := ri.fprev;
			SELF.wolfe_funevals := ri.wolfe_funevals + 1;
			SELF.prev_t := ri.prev_t;
			SELF.glob_f := ri.glob_f;
			//calculate islegalgnew ??
			islegal_gnew := TRUE;
			SELF.gtdnew := IF (ISVALID (le.cost_value) AND islegal_gnew, armgtd_new, -1);//gtd value is assigned/calculated only if it will be used in the next iteration. It will be used if the first arm cond is not satisfied and the second one is satisfied ??
			SELF.local_sumd := ri.local_sumd;
			SELF := le;
		END;
		arm_nextitr_out := JOIN (armCostGrad_new, armin_t, LEFT.partition_id = RIGHT.partition_id, armmain_tran(LEFT,RIGHT) , LOCAL);
		armshort_format := RECORD
			armin_t.wolfe_funevals;
			armin_t.partition_id;
		END;
		armfunevals_table := TABLE (armin_t, armshort_format, LOCAL);
		steptoosmall_out := JOIN (g, armfunevals_table, LEFT.partition_id = RIGHT.partition_id,TRANSFORM (ArmijoRecord, SELF.local_sumd := -1; SELF.gtdnew := -1; SELF.glob_f := LEFT.cost_value; SELF.armCond := 1; SELF.prev_t:=0; SELF.tprev := -1; SELF.fprev:= -1; SELF.wolfe_funevals := RIGHT.wolfe_funevals; SELF := LEFT), LOCAL);// in case we are returning this dataset, it means step size had been too small so the armCond should be 1
		//Check whether step size has become too small
    //if sum(abs(t*d)) <= tolX
		toosmall_cond := sum_abs_td <= tolX;
		// loop continues until f_new > fr + c1*t*gtd || ~isLegal(f_new) is TRUE, so the loop has to stop when NOT (f_new > fr + c1*t*gtd || ~isLegal(f_new)), in the other word when NOT (f_new > fr + c1*t*gtd ) AND isLegal(f_new)
		
		arm_out := IF (toosmall_cond, steptoosmall_out, arm_nextitr_out);

		
		RETURN arm_out; 
	END;// END BackTracking
	ar1 := BackTracking(topass_BackTracking,1);
	// RETURN BackTracking(ar1,2);
	RETURN LOOP (topass_BackTracking, LEFT.armCond = -1 , BackTracking(ROWS(LEFT),COUNTER));
END;// END ArmijoBacktrack_fromwolfe
EXPORT WolfeLineSearch4_4_2(INTEGER cccc, DATASET(Layout_Part) x, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Cell_nid) TrainData , DATASET(Layout_Part) TrainLabel,DATASET(CostGrad_Record) CostFunc (DATASET(Layout_Part) x, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Cell_nid) TrainData , DATASET(Layout_Part) TrainLabel), REAL8 t, DATASET(Layout_Part) d, DATASET(costgrad_record) g, REAL8 gtd, REAL8 c1=0.0001, REAL8 c2=0.9, INTEGER maxLS=25, REAL8 tolX=0.000000001):=FUNCTION
	//C++ functions used
	//sum (M.*V)
	REAL8 sump(PBblas.Types.dimension_t N, PBblas.Types.matrix_t M, PBblas.Types.matrix_t V) := BEGINC++
		#body
		double result = 0;
		double tmpp ;
		double *cellm = (double*) m;
		double *cellv = (double*) v;
		uint32_t i;
		for (i=0; i<n; i++) {
			tmpp =(cellm[i] * cellv [i]);
			result = result + tmpp;
		}
		return(result);

	ENDC++;
	
	// We pass the prev g and cost values along with the next t value to the bracketing loop. The new g and cost values will be calculated at the begining of the loop
	//id field shows which side of the bracket prev values belong to (left (1) or right (2))
	//we embed the f value recived in wolfe function in what we pass to bracketing loop, so in the rest of the algorithm we have access to it in each node -> SELF.glob_f :=LEFT.cost_value
	topass_bracketing := PROJECT (g, TRANSFORM (ZoomingRecord, SELF.zooming_cond := -1; SELF.zoomtermination := FALSE; SELF.insufProgress := FALSE; SELF.id := 1; SELF.glob_f :=LEFT.cost_value; SELF.high_t := -1 ; SELF.high_gtd := -1; SELF.high_cost_value := -1; SELF.prev_t := 0; SELF.prev_gtd := gtd; SELF.wolfe_funEvals := 0; SELF.c := 0; SELF.bracketing_cond := -1; SELF.next_t := t; SELF := LEFT), LOCAL);
	BracketingStep (DATASET (ZoomingRecord) inputp, UNSIGNED coun) := FUNCTION
		//the idea is to calculate the new values at the begining of the bracket and then JOIN the loop input and this new values in order to generate the loop output. The JOIN TRANSFORM contains all the condition checks
		// calculate new g and cost value at the begining of the loop using next_t value in the input dataset
		// calculate t*d
		Elem := {REAL8 v};
		Layout_Part td_tran (inputp le, d ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := PBBlas.BLAS.dscal(cells, le.next_t, ri.mat_part, 1);
			SELF := le;
		END;
		td := JOIN (inputp, d, LEFT.partition_id = RIGHT.partition_id, td_tran (LEFT, RIGHT), LOCAL);
		//calculate x_new = x0 + td
		Layout_Part x_new_tran (Layout_part le, Layout_part ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := PBblas.BLAS.daxpy(cells, 1.0, le.mat_part, 1, ri.mat_part, 1);
			SELF := le;
		END;
		x_new := JOIN (x, td, LEFT.partition_id = RIGHT.partition_id, x_new_tran(LEFT, RIGHT), LOCAL);
		// calculated new g and cost values
		CostGrad_new := CostFunc(x_new,CostFunc_params,TrainData, TrainLabel);
		//calculated gtd_new (it will be used in IF conditions)
		Elem gtd_tran (CostGrad_new le, d ri) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.v := sump(cells, le.mat_part, ri.mat_part);
		END;
		gtd_new_ := JOIN (CostGrad_new, d, LEFT.partition_id = RIGHT.partition_id, gtd_tran (LEFT, RIGHT), LOCAL);
		gtd_new := SUM (gtd_new_, gtd_new_.v);
		//now that new values are calculated, JOIN prev (inputp) values and the new values considering all the conditions in order to produce the loop results
		//based on which condition is satisfied and whether previous cost (the cost in inputp) is less than the newly calculated cost (CostGrad_new) the return result is decided
		//we always return the braket side with smaller cost value, the values for the Hi sie of the bracket are in high_ fields. we don't return gradient vector for the high side
		BrackLSiter := coun - 1;
		//before we enter the main_tran we check whether f_new or g_new are not legal. In that case we switch to ArmijoBackTracking and return the results and end the wolfe line search
		b_rec := {INTEGER check};
		b_rec check_fnew_gnew_legal (CostGrad_new le) := TRANSFORM
			// SELF.check:= ISVALID (le.cost_value) AND islegal_mat (le.part_rows*le.part_cols, le.mat_part);
			SELF.check:= IF (ISVALID (le.cost_value), 0, 1) ;//if the values are legal then check will be zero so the summation of check values in f_g_new_legal will end up being zero which means or local values are legal. If only one local value is not legal, the summation will be more than zero and it means we had observed illegal value
		END;
		f_g_new_legal_ := PROJECT (CostGrad_new, check_fnew_gnew_legal(LEFT) , LOCAL);
		f_g_new_legal := SUM (f_g_new_legal_, f_g_new_legal_.check);
		IS_f_g_new_legal := IF (f_g_new_legal=0, TRUE, FALSE);
		arm_t_rec arm_init_t_tran (inputp le) := TRANSFORM
		  //t = (t + t_prev)/2;
			SELF.init_arm_t := (le.next_t + le.prev_t)/2;
			SELF := le;
		END;
		init_arm_t := PROJECT (inputp, arm_init_t_tran(LEFT), LOCAL);
		Arm_Result := ArmijoBacktrack_fromwolfe( x,  CostFunc_params, TrainData , TrainLabel, CostFunc , init_arm_t, d,  g, gtd,  c1,  c2, tolX);
		Arm_result_zoomingformat := PROJECT (Arm_Result, TRANSFORM (ZoomingRecord,
			SELF.high_cost_value := -1;
			SELF.prev_gtd := -1;
			SELF.high_gtd := -1;
			SELF.high_t := -1;
			SELF.id := 1; 
			SELF.bracketing_cond := 10;
			SELF.c := coun;
			SELF.next_t := -1;
			SELF.zoomtermination := TRUE;
			SELF.insufProgress := FALSE;
		  SELF:=LEFT), LOCAL);// This transfomr is converting the output of the Armijo to wolfe record format which is "ZoomingRecord"
		
		ZoomingRecord main_tran (inputp le_prev, CostGrad_new ri_new) := TRANSFORM
			// calculate t for the next iteration (this value only will be calculated if there is going to be a next iteration)
			f := le_prev.glob_f; // the f value recived by wolfe function
			newt := le_prev.next_t; // the new values are calculated based on next_t so next_t is actually the new t in this iteration
			fNew := ri_new.cost_value;
			gtdNew := gtd_new;
			tPrev := le_prev.prev_t;
			fPrev := le_prev.cost_value;
			gtdPrev := le_prev.prev_gtd;
			minstep := newt + 0.01* (newt-tPrev);
			maxstep := newt*10;
			nextitr_t := polyinterp_both (tPrev, fPrev,gtdPrev, newt, fNew, gtdNew, minstep, maxstep);
			//specify the conditions along that which cost value is smaller
			cost_cond :=  (le_prev.cost_value < ri_new.cost_value);
			// f_new > f + c1*t*gtd || (LSiter > 1 && f_new >= f_prev)
			cond1 := ((fNew > f + c1 * newt* gtd)| ((BrackLSiter > 1) & (fNew >= fPrev))) & (coun < (maxLS+1) );
			// abs(gtd_new) <= -c2*gtd
			cond2 := (ABS(gtdNew) <= (-1*c2*gtd)) & (coun < (maxLS+1) );
			// gtd_new >= 0
			cond3 := (gtdNew >= 0 ) &(coun < (maxLS+1) ); // by adding & (coun < (maxLS+1) ) at the end of each condition I make sure that at the very last iteration no condition is satisfied and i just return the new value which make my algorithm works like the matlab algorithm
			//  I calculate new values at the begining of each loop itr, where the matlab code calculates it at end
		  // so at the maxLS I calculate next iteration t and go to the next iteration, then I calculate the new value which corresponds to new value calculated at the end of maxLS iteration in the matlab code, then I make sure no condition is satified
			// and I just return the new value at the (maxLS+1)  iteration which corresponds to new value at the end of maxLS iteration in matlab code
			WhichCond := IF (coun = (MAXLS+1),4 , IF (cond1, 1, IF(cond2, 2, IF (cond3,3,-1)))); // if we reach the maximum number of iterations or if one of the conditions is satified then the loopfilter returns empty set and we exit the loop
			//decide whether we should return prev or new values
			prev_mat := le_prev.mat_part;
			new_mat := ri_new.mat_part;
			prev_num := 1;
			new_num := 2;
			prev_or_new := IF (cond1, (IF (cost_cond, prev_num, new_num)) , IF (cond2, new_num, IF (cond3, IF (cost_cond, prev_num, new_num) , new_num)));
			SELF.mat_part := IF (prev_or_new = 1 , prev_mat, new_mat);
			SELF.cost_value := IF (prev_or_new = 1 , fPrev, fNew);
			SELF.high_cost_value := IF (prev_or_new = 1 , fNew, fPrev);
			SELF.prev_gtd := IF (prev_or_new = 1 , gtdPrev, gtdNew);
			SELF.high_gtd := IF (prev_or_new = 1 , gtdNew, gtdPrev);
			SELF.prev_t := IF (prev_or_new = 1 , tPrev, newt);
			SELF.high_t := IF (prev_or_new = 1 , newt, tPrev);
			SELF.id := prev_or_new; // thd id shows whether the associated value to prev is the left or riht side of the bracket, the hight value will be the other value of id (if id=1 it means the prev values start the bracket and high ends the brackte) 
			//actually the id only matters when we exit the loop because of cond1 or cond3. cond2 and reaching maxitr casue program to end so id does not matter becasue we don't go to zooming loop
			SELF.bracketing_cond := WhichCond;
			SELF.c := coun;
			SELF.next_t := IF (WhichCond = -1 AND coun < (maxLS+1), nextitr_t, -1);// next_t is calculated only if we are going to the next iteration which means WhichCond should be -1 ( no condition has been satisfied) and we have not rached the Max number of iterations
			SELF.wolfe_funEvals := le_prev.wolfe_funEvals + 1;
			SELF := le_prev;
		END;
		Result := JOIN (inputp, CostGrad_new, LEFT.partition_id = RIGHT.partition_id, main_tran(LEFT, RIGHT), LOCAL);
		final_result := IF (IS_f_g_new_legal, result, Arm_result_zoomingformat); // if either of f_new or g_new are not legal return the armijobacktracking instead of wolfelinesearch
		RETURN final_result;
	END; // END BracketingStep
	//after the bracketing step check whether we have reached maxitr , however if we left bracketing step with which_cond==10 which means we used armijobacktracking then this is the final result and does not need to go for maxitr check
	ZoomingRecord maxitr_tran (ZoomingRecord le_brack , CostGrad_record ri_g) := TRANSFORM
		itr_num := le_brack.c;
		maxitr_cond := (itr_num = (maxLS+1)) AND le_brack.bracketing_cond !=10; // itr number should be equal to (maxLS+1) and also bracketing_cond should not be 10, if bracketing_cond is equal to 10 it means that in the bracketing step we entered ArmijoBackTrackign and soomterminaition is true and we will return
		cost_cond_0 := ri_g.cost_value < le_brack.cost_value;
		SELF.mat_part := IF (maxitr_cond , IF (cost_cond_0, ri_g.mat_part, le_brack.mat_part) , le_brack.mat_part);
		SELF.cost_value := IF (maxitr_cond , IF (cost_cond_0, ri_g.cost_value, le_brack.cost_value) , le_brack.cost_value);
		SELF.prev_t := IF (maxitr_cond , IF (cost_cond_0, 0 , le_brack.prev_t) , le_brack.prev_t);
		SELF.prev_gtd := IF (maxitr_cond , IF (cost_cond_0, gtd, le_brack.prev_gtd) , le_brack.prev_gtd);
		SELF.zoomtermination := le_brack.zoomtermination OR maxitr_cond OR (le_brack.bracketing_cond=2); // if we already have the zoomtermination=TRUE (armijo) or if we reach maximum number of iterations in the bracketing loop, or if condition 2 is satisfied in the bracketing loop. it means we don't go to the zooming loop and the wolfe algorithms ends
		SELF := le_brack;
	END;
	//bracketing_result_ := LOOP(topass_bracketing, LEFT.bracketing_cond = -1, EXISTS(ROWS(LEFT)) and COUNTER < (maxLS+1), BracketingStep(ROWS(LEFT), COUNTER)); orig, not work
	//bracketing_result_ := LOOP(topass_bracketing, (maxLS+1), LEFT.bracketing_cond = -1, BracketingStep(ROWS(LEFT), COUNTER)); works but runs maxls+1 time
	bracketing_result_ := LOOP(topass_bracketing, LEFT.bracketing_cond = -1, BracketingStep(ROWS(LEFT), COUNTER));
	//check if MAXITR is reached and decide between bracketing result and f (t=0) to be returned
	bracketing_result := JOIN (bracketing_result_ , g, LEFT.partition_id = RIGHT.partition_id, maxitr_tran (LEFT, RIGHT), LOCAL);
	topass_zooming := bracketing_result;
	ZoomingStep (DATASET (ZoomingRecord) inputp, UNSIGNED coun) := FUNCTION
		//similar to BracketingStep, first we calculate the new point (for that we need to calculate the new t using interpolation)
		//after the new point is calculated we JOIN the input and the new point in order to decide which one to return
		// Compute new trial value & Test that we are making sufficient progress
		//The following transform calculates the new trial value as well as insufProgress value by doing a PROJECT on the input, the new t is then saved in next_t field
		ZoomingRecord newt_tran (ZoomingRecord le) := TRANSFORM
			insufprog := le.insufProgress;
			LO_bracket := le.prev_t;
			HI_bracket := le.high_t; 
			HI_f := le.high_cost_value;
			HI_gtd := le.high_gtd;
			BList := [LO_bracket,HI_bracket];
			max_bracket := MAX(Blist);
			min_bracket := MIN(Blist);
			current_LO_id := le.id;
			gtd_LO := le.prev_gtd;
			new_id := IF (current_LO_id=1,2,1);
			min_bracketFval := le.cost_value;
			tTmp1 := polyinterp_noboundry (LO_bracket, min_bracketFval, gtd_LO, HI_bracket, HI_f, HI_gtd);
			tTmp2 := polyinterp_noboundry (HI_bracket, HI_f, HI_gtd, LO_bracket, min_bracketFval, gtd_LO);
			tTmp := IF (current_LO_id = 1, tTmp1, tTmp2);
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
			SELF.next_t := tZoom;
			SELF.insufProgress := insufprog_new;
			SELF := le;
		END;
		inputp_t_insuf := PROJECT (inputp, newt_tran(LEFT) ,LOCAL);//tzoom and insufficient progres values are calculated and embedded in the inputp dataset
		//evaluate new point
		// calculate t*d
		Elem := {REAL8 v};
		Layout_Part td_tran (inputp_t_insuf le, d ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := PBBlas.BLAS.dscal(cells, le.next_t, ri.mat_part, 1);
			SELF := le;
		END;
		td := JOIN (inputp_t_insuf, d, LEFT.partition_id = RIGHT.partition_id, td_tran (LEFT, RIGHT), LOCAL);
		// calculate x_new = x0 + td
		Layout_Part x_new_tran (Layout_part le, Layout_part ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := PBblas.BLAS.daxpy(cells, 1.0, le.mat_part, 1, ri.mat_part, 1);
			SELF := le;
		END;
		x_new := JOIN (x, td, LEFT.partition_id = RIGHT.partition_id, x_new_tran(LEFT, RIGHT), LOCAL);
		// calculated new g and cost values
		CostGrad_new := CostFunc(x_new,CostFunc_params,TrainData, TrainLabel);
		Elem gtd_tran (CostGrad_new le, d ri) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.v := sump(cells, le.mat_part, ri.mat_part);
		END;
		gtd_new_ := JOIN (CostGrad_new, d, LEFT.partition_id = RIGHT.partition_id, gtd_tran (LEFT, RIGHT), LOCAL);
		gtd_new := SUM (gtd_new_, gtd_new_.v);
		ZoomingRecord main_tran (inputp le_prev, CostGrad_new ri_new) := TRANSFORM
			f := le_prev.glob_f;
			current_LO_id := le_prev.id;
			new_id := IF (current_LO_id=1,2,1);
			fNewZoom := ri_new.cost_value;
			tZoom := le_prev.next_t;// the new point is calculates based on next_t value which was calculated at the begining of the zoom loop
			min_bracketFval := le_prev.cost_value;
			LO_bracket := le_prev.prev_t;
			HI_bracket := le_prev.high_t;
			gtdNewZoom := gtd_new;
			// f_new > f + c1*t*gtd || f_new >= f_LO
			zoom_cond_1 := (fNewZoom > f + c1 * tZoom * gtd) | (fNewZoom >= min_bracketFval);
			// abs(gtd_new) <= - c2*gtd
			zoom_cond_2 := ABS (gtdNewZoom) <= (-1 * c2 * gtd);
			// gtd_new*(bracket(HIpos)-bracket(LOpos)) >= 0
			zoom_cond_3 := gtdNewZoom * (HI_bracket - LO_bracket) >= 0; 
			whichcond := IF (zoom_cond_1, 11, IF (zoom_cond_2, 12, IF (zoom_cond_3, 13, -1)));
			cost_cond := min_bracketFval < fNewZoom; // f_lo < f_new
			lo_num := 1;
			new_num := 2;
			high_num := 3;
			lo_or_new := IF (zoom_cond_1, IF (cost_cond, lo_num, new_num), IF (zoom_cond_2, new_num, IF (zoom_cond_3, IF (cost_cond, lo_num, new_num), new_num)) );
			lo_or_new_id := IF (zoom_cond_1, IF (cost_cond, current_LO_id, new_id), IF (zoom_cond_2, current_LO_id, IF (zoom_cond_3, IF (cost_cond, new_id, current_LO_id), current_LO_id)) );
			lo_gtd := le_prev.prev_gtd;
			lo_t := le_prev.prev_t;
			//~done && abs((bracket(1)-bracket(2))*gtd_new) < tolX	
			zoom_term_cond := (~zoom_cond_1 AND zoom_cond_2) OR (ABS((tZoom-LO_bracket)*gtdNewZoom) < tolX);//if zoom_term_cond=TRUE then no need to assigne high values
			pre_mat := le_prev.mat_part;
			new_mat := ri_new.mat_part;
			//SELF.mat_part := IF (lo_or_new =1 , le_prev.mat_part, ri_new.mat_part);
			SELF.mat_part := IF (zoom_cond_1, IF (cost_cond, pre_mat, new_mat),IF (zoom_cond_2, new_mat, IF (zoom_cond_3, IF (cost_cond, pre_mat, new_mat), new_mat)));
			SELF.cost_value := IF (zoom_cond_1, IF (cost_cond, min_bracketFval, fNewZoom),IF (zoom_cond_2, fNewZoom, IF (zoom_cond_3, IF (cost_cond, min_bracketFval, fNewZoom), fNewZoom)));
			SELF.high_cost_value := IF (zoom_cond_1, IF (cost_cond, fNewZoom, min_bracketFval),IF (zoom_cond_2, le_prev.high_cost_value, IF (zoom_cond_3, IF (cost_cond, fNewZoom, min_bracketFval), le_prev.high_cost_value)));
			SELF.prev_gtd := IF (zoom_cond_1, IF (cost_cond, lo_gtd, gtdNewZoom),IF (zoom_cond_2, gtdNewZoom, IF (zoom_cond_3, IF (cost_cond, lo_gtd, gtdNewZoom), gtdNewZoom)));
			SELF.high_gtd := IF (zoom_cond_1, IF (cost_cond, gtdNewZoom, lo_gtd),IF (zoom_cond_2, le_prev.high_gtd, IF (zoom_cond_3, IF (cost_cond, gtdNewZoom, lo_gtd), le_prev.high_gtd)));
			SELF.prev_t := IF (zoom_cond_1, IF (cost_cond, lo_t, tZoom),IF (zoom_cond_2, tZoom, IF (zoom_cond_3, IF (cost_cond, lo_t, tZoom), tZoom)));
			SELF.high_t := IF (zoom_cond_1, IF (cost_cond, tZoom, lo_t),IF (zoom_cond_2, le_prev.high_t, IF (zoom_cond_3, IF (cost_cond, tZoom, lo_t), le_prev.high_t)));
			SELF.id := IF (zoom_cond_1, IF (cost_cond, current_LO_id, new_id), IF (zoom_cond_2, current_LO_id, IF (zoom_cond_3, IF (cost_cond, new_id, current_LO_id), current_LO_id)) );//here
			SELF.c := le_prev.c + 1;
			SELF.zoomtermination := zoom_term_cond;
			SELF.wolfe_funEvals := le_prev.wolfe_funEvals + 1;
			SELF.zooming_cond := whichcond;
			SELF := le_prev;
		END;
		Result := JOIN (inputp_t_insuf, CostGrad_new, LEFT.partition_id = RIGHT.partition_id, main_tran(LEFT, RIGHT), LOCAL);		
		RETURN Result;
	END;
	//zooming_result := LOOP(topass_zooming, (LEFT.zoomtermination = FALSE) AND (LEFT.c < (maxLS+1)), EXISTS(ROWS(LEFT)) , ZoomingStep(ROWS(LEFT), COUNTER)); orig
	zooming_result := LOOP(topass_zooming, (LEFT.zoomtermination = FALSE) AND (LEFT.c < (maxLS+1)), ZoomingStep(ROWS(LEFT), COUNTER));
	//RETURN BracketingStep (topass_bracketing, 1);
	//RETURN zooming_result;
	//RETURN LOOP(topass_bracketing, 1 , BracketingStep(ROWS(LEFT), COUNTER));
	//RETURN LOOP(topass_bracketing, (maxLS+1), LEFT.bracketing_cond = -1, BracketingStep(ROWS(LEFT), COUNTER));
	//RETURN bracketing_result;// works
	//RETURN zooming_result;
	RETURN zooming_result;
	//RETURN LOOP(topass_bracketing, LEFT.bracketing_cond = -1, BracketingStep(ROWS(LEFT), COUNTER)); // works correctly
	//RETURN BracketingStep(topass_bracketing, 1); works correctly
	//RETURN LOOP(topass_bracketing, 1, BracketingStep(ROWS(LEFT), COUNTER));works correctly
	//RETURN topass_zooming;
	//RETURN LOOP(topass_zooming, (LEFT.zoomtermination = FALSE) AND (LEFT.c < (maxLS+1)), FALSE , ZoomingStep(ROWS(LEFT), COUNTER)); // works
	//RETURN LOOP(topass_zooming, (LEFT.zoomtermination = FALSE) AND (LEFT.c < (maxLS+1)) , ZoomingStep(ROWS(LEFT), COUNTER));//works
	// At the end the final result returned from wolfe function contains the t value in prev_t field, the g vector in the Layout_part section and the f value in cost_value field and the gtd in the prev_gtd
END;// END WolfeLineSearch4_4_2


EXPORT WolfeLineSearch4_4_2_ttest(INTEGER cccc, DATASET(Layout_Part) x, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Cell_nid) TrainData , DATASET(Layout_Part) TrainLabel,DATASET(CostGrad_Record) CostFunc (DATASET(Layout_Part) x, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Cell_nid) TrainData , DATASET(Layout_Part) TrainLabel), REAL8 t, DATASET(Layout_Part) d, DATASET(costgrad_record) g, REAL8 gtd, REAL8 c1=0.0001, REAL8 c2=0.9, INTEGER maxLS=25, REAL8 tolX=0.000000001):=FUNCTION
	//C++ functions used
	//sum (M.*V)
	REAL8 sump(PBblas.Types.dimension_t N, PBblas.Types.matrix_t M, PBblas.Types.matrix_t V) := BEGINC++
		#body
		double result = 0;
		double tmpp ;
		double *cellm = (double*) m;
		double *cellv = (double*) v;
		uint32_t i;
		for (i=0; i<n; i++) {
			tmpp =(cellm[i] * cellv [i]);
			result = result + tmpp;
		}
		return(result);

	ENDC++;
	
	// We pass the prev g and cost values along with the next t value to the bracketing loop. The new g and cost values will be calculated at the begining of the loop
	//id field shows which side of the bracket prev values belong to (left (1) or right (2))
	//we embed the f value recived in wolfe function in what we pass to bracketing loop, so in the rest of the algorithm we have access to it in each node -> SELF.glob_f :=LEFT.cost_value
	topass_bracketing := PROJECT (g, TRANSFORM (ZoomingRecord, SELF.zooming_cond := -1; SELF.zoomtermination := FALSE; SELF.insufProgress := FALSE; SELF.id := 1; SELF.glob_f :=LEFT.cost_value; SELF.high_t := -1 ; SELF.high_gtd := -1; SELF.high_cost_value := -1; SELF.prev_t := 0; SELF.prev_gtd := gtd; SELF.wolfe_funEvals := 0; SELF.c := 0; SELF.bracketing_cond := -1; SELF.next_t := t; SELF := LEFT), LOCAL);
	BracketingStep (DATASET (ZoomingRecord) inputp, UNSIGNED coun) := FUNCTION
		//the idea is to calculate the new values at the begining of the bracket and then JOIN the loop input and this new values in order to generate the loop output. The JOIN TRANSFORM contains all the condition checks
		// calculate new g and cost value at the begining of the loop using next_t value in the input dataset
		// calculate t*d
		Elem := {REAL8 v};
		Layout_Part td_tran (inputp le, d ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := PBBlas.BLAS.dscal(cells, le.next_t, ri.mat_part, 1);
			SELF := le;
		END;
		td := JOIN (inputp, d, LEFT.partition_id = RIGHT.partition_id, td_tran (LEFT, RIGHT), LOCAL);
		//calculate x_new = x0 + td
		Layout_Part x_new_tran (Layout_part le, Layout_part ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := PBblas.BLAS.daxpy(cells, 1.0, le.mat_part, 1, ri.mat_part, 1);
			SELF := le;
		END;
		x_new := JOIN (x, td, LEFT.partition_id = RIGHT.partition_id, x_new_tran(LEFT, RIGHT), LOCAL);
		// calculated new g and cost values
		CostGrad_new := CostFunc(x_new,CostFunc_params,TrainData, TrainLabel);
		//calculated gtd_new (it will be used in IF conditions)
		Elem gtd_tran (CostGrad_new le, d ri) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.v := sump(cells, le.mat_part, ri.mat_part);
		END;
		gtd_new_ := JOIN (CostGrad_new, d, LEFT.partition_id = RIGHT.partition_id, gtd_tran (LEFT, RIGHT), LOCAL);
		gtd_new := SUM (gtd_new_, gtd_new_.v);
		//now that new values are calculated, JOIN prev (inputp) values and the new values considering all the conditions in order to produce the loop results
		//based on which condition is satisfied and whether previous cost (the cost in inputp) is less than the newly calculated cost (CostGrad_new) the return result is decided
		//we always return the braket side with smaller cost value, the values for the Hi sie of the bracket are in high_ fields. we don't return gradient vector for the high side
		BrackLSiter := coun - 1;
		//before we enter the main_tran we check whether f_new or g_new are not legal. In that case we switch to ArmijoBackTracking and return the results and end the wolfe line search
		b_rec := {INTEGER check};
		b_rec check_fnew_gnew_legal (CostGrad_new le) := TRANSFORM
			// SELF.check:= ISVALID (le.cost_value) AND islegal_mat (le.part_rows*le.part_cols, le.mat_part);
			SELF.check:= IF (ISVALID (le.cost_value), 0, 1) ;//if the values are legal then check will be zero so the summation of check values in f_g_new_legal will end up being zero which means or local values are legal. If only one local value is not legal, the summation will be more than zero and it means we had observed illegal value
		END;
		f_g_new_legal_ := PROJECT (CostGrad_new, check_fnew_gnew_legal(LEFT) , LOCAL);
		f_g_new_legal := SUM (f_g_new_legal_, f_g_new_legal_.check);
		IS_f_g_new_legal := IF (f_g_new_legal=0, TRUE, FALSE);
		arm_t_rec arm_init_t_tran (inputp le) := TRANSFORM
		  //t = (t + t_prev)/2;
			SELF.init_arm_t := (le.next_t + le.prev_t)/2;
			SELF := le;
		END;
		init_arm_t := PROJECT (inputp, arm_init_t_tran(LEFT), LOCAL);
		Arm_Result := ArmijoBacktrack_fromwolfe( x,  CostFunc_params, TrainData , TrainLabel, CostFunc , init_arm_t, d,  g, gtd,  c1,  c2, tolX);
		Arm_result_zoomingformat := PROJECT (Arm_Result, TRANSFORM (ZoomingRecord,
			SELF.high_cost_value := -1;
			SELF.prev_gtd := -1;
			SELF.high_gtd := -1;
			SELF.high_t := -1;
			SELF.id := 1; 
			SELF.bracketing_cond := 10;
			SELF.c := coun;
			SELF.next_t := -1;
			SELF.zoomtermination := TRUE;
			SELF.insufProgress := FALSE;
		  SELF:=LEFT), LOCAL);// This transfomr is converting the output of the Armijo to wolfe record format which is "ZoomingRecord"
		
		ZoomingRecord main_tran (inputp le_prev, CostGrad_new ri_new) := TRANSFORM
			// calculate t for the next iteration (this value only will be calculated if there is going to be a next iteration)
			f := le_prev.glob_f; // the f value recived by wolfe function
			newt := le_prev.next_t; // the new values are calculated based on next_t so next_t is actually the new t in this iteration
			fNew := ri_new.cost_value;
			gtdNew := gtd_new;
			tPrev := le_prev.prev_t;
			fPrev := le_prev.cost_value;
			gtdPrev := le_prev.prev_gtd;
			minstep := newt + 0.01* (newt-tPrev);
			maxstep := newt*10;
			nextitr_t := polyinterp_both (tPrev, fPrev,gtdPrev, newt, fNew, gtdNew, minstep, maxstep);
			//specify the conditions along that which cost value is smaller
			cost_cond :=  (le_prev.cost_value < ri_new.cost_value);
			// f_new > f + c1*t*gtd || (LSiter > 1 && f_new >= f_prev)
			cond1 := ((fNew > f + c1 * newt* gtd)| ((BrackLSiter > 1) & (fNew >= fPrev))) & (coun < (maxLS+1) );
			// abs(gtd_new) <= -c2*gtd
			cond2 := (ABS(gtdNew) <= (-1*c2*gtd)) & (coun < (maxLS+1) );
			// gtd_new >= 0
			cond3 := (gtdNew >= 0 ) &(coun < (maxLS+1) ); // by adding & (coun < (maxLS+1) ) at the end of each condition I make sure that at the very last iteration no condition is satisfied and i just return the new value which make my algorithm works like the matlab algorithm
			//  I calculate new values at the begining of each loop itr, where the matlab code calculates it at end
		  // so at the maxLS I calculate next iteration t and go to the next iteration, then I calculate the new value which corresponds to new value calculated at the end of maxLS iteration in the matlab code, then I make sure no condition is satified
			// and I just return the new value at the (maxLS+1)  iteration which corresponds to new value at the end of maxLS iteration in matlab code
			WhichCond := IF (coun = (MAXLS+1),4 , IF (cond1, 1, IF(cond2, 2, IF (cond3,3,-1)))); // if we reach the maximum number of iterations or if one of the conditions is satified then the loopfilter returns empty set and we exit the loop
			//decide whether we should return prev or new values
			prev_mat := le_prev.mat_part;
			new_mat := ri_new.mat_part;
			prev_num := 1;
			new_num := 2;
			prev_or_new := IF (cond1, (IF (cost_cond, prev_num, new_num)) , IF (cond2, new_num, IF (cond3, IF (cost_cond, prev_num, new_num) , new_num)));
			SELF.mat_part := IF (prev_or_new = 1 , prev_mat, new_mat);
			SELF.cost_value := IF (prev_or_new = 1 , fPrev, fNew);
			SELF.high_cost_value := IF (prev_or_new = 1 , fNew, fPrev);
			SELF.prev_gtd := IF (prev_or_new = 1 , gtdPrev, gtdNew);
			SELF.high_gtd := IF (prev_or_new = 1 , gtdNew, gtdPrev);
			SELF.prev_t := IF (prev_or_new = 1 , tPrev, newt);
			SELF.high_t := IF (prev_or_new = 1 , newt, tPrev);
			SELF.id := prev_or_new; // thd id shows whether the associated value to prev is the left or riht side of the bracket, the hight value will be the other value of id (if id=1 it means the prev values start the bracket and high ends the brackte) 
			//actually the id only matters when we exit the loop because of cond1 or cond3. cond2 and reaching maxitr casue program to end so id does not matter becasue we don't go to zooming loop
			SELF.bracketing_cond := WhichCond;
			SELF.c := coun;
			SELF.next_t := IF (WhichCond = -1 AND coun < (maxLS+1), nextitr_t, -1);// next_t is calculated only if we are going to the next iteration which means WhichCond should be -1 ( no condition has been satisfied) and we have not rached the Max number of iterations
			SELF.wolfe_funEvals := le_prev.wolfe_funEvals + 1;
			SELF := le_prev;
		END;
		Result := JOIN (inputp, CostGrad_new, LEFT.partition_id = RIGHT.partition_id, main_tran(LEFT, RIGHT), LOCAL);
		final_result := IF (IS_f_g_new_legal, result, Arm_result_zoomingformat); // if either of f_new or g_new are not legal return the armijobacktracking instead of wolfelinesearch
		RETURN final_result;
	END; // END BracketingStep
	//after the bracketing step check whether we have reached maxitr , however if we left bracketing step with which_cond==10 which means we used armijobacktracking then this is the final result and does not need to go for maxitr check
	ZoomingRecord maxitr_tran (ZoomingRecord le_brack , CostGrad_record ri_g) := TRANSFORM
		itr_num := le_brack.c;
		maxitr_cond := (itr_num = (maxLS+1)) AND le_brack.bracketing_cond !=10; // itr number should be equal to (maxLS+1) and also bracketing_cond should not be 10, if bracketing_cond is equal to 10 it means that in the bracketing step we entered ArmijoBackTrackign and soomterminaition is true and we will return
		cost_cond_0 := ri_g.cost_value < le_brack.cost_value;
		SELF.mat_part := IF (maxitr_cond , IF (cost_cond_0, ri_g.mat_part, le_brack.mat_part) , le_brack.mat_part);
		SELF.cost_value := IF (maxitr_cond , IF (cost_cond_0, ri_g.cost_value, le_brack.cost_value) , le_brack.cost_value);
		SELF.prev_t := IF (maxitr_cond , IF (cost_cond_0, 0 , le_brack.prev_t) , le_brack.prev_t);
		SELF.prev_gtd := IF (maxitr_cond , IF (cost_cond_0, gtd, le_brack.prev_gtd) , le_brack.prev_gtd);
		SELF.zoomtermination := le_brack.zoomtermination OR maxitr_cond OR (le_brack.bracketing_cond=2); // if we already have the zoomtermination=TRUE (armijo) or if we reach maximum number of iterations in the bracketing loop, or if condition 2 is satisfied in the bracketing loop. it means we don't go to the zooming loop and the wolfe algorithms ends
		SELF := le_brack;
	END;
	//bracketing_result_ := LOOP(topass_bracketing, LEFT.bracketing_cond = -1, EXISTS(ROWS(LEFT)) and COUNTER < (maxLS+1), BracketingStep(ROWS(LEFT), COUNTER)); orig, not work
	//bracketing_result_ := LOOP(topass_bracketing, (maxLS+1), LEFT.bracketing_cond = -1, BracketingStep(ROWS(LEFT), COUNTER)); works but runs maxls+1 time
	bracketing_result_ := LOOP(topass_bracketing, LEFT.bracketing_cond = -1, BracketingStep(ROWS(LEFT), COUNTER));
	//check if MAXITR is reached and decide between bracketing result and f (t=0) to be returned
	bracketing_result := JOIN (bracketing_result_ , g, LEFT.partition_id = RIGHT.partition_id, maxitr_tran (LEFT, RIGHT), LOCAL);
	topass_zooming := bracketing_result;
	ZoomingStep (DATASET (ZoomingRecord) inputp, UNSIGNED coun) := FUNCTION
		//similar to BracketingStep, first we calculate the new point (for that we need to calculate the new t using interpolation)
		//after the new point is calculated we JOIN the input and the new point in order to decide which one to return
		// Compute new trial value & Test that we are making sufficient progress
		//The following transform calculates the new trial value as well as insufProgress value by doing a PROJECT on the input, the new t is then saved in next_t field
		ZoomingRecord newt_tran (ZoomingRecord le) := TRANSFORM
			insufprog := le.insufProgress;
			LO_bracket := le.prev_t;
			HI_bracket := le.high_t; 
			HI_f := le.high_cost_value;
			HI_gtd := le.high_gtd;
			BList := [LO_bracket,HI_bracket];
			max_bracket := MAX(Blist);
			min_bracket := MIN(Blist);
			current_LO_id := le.id;
			gtd_LO := le.prev_gtd;
			new_id := IF (current_LO_id=1,2,1);
			min_bracketFval := le.cost_value;
			tTmp1 := polyinterp_noboundry (LO_bracket, min_bracketFval, gtd_LO, HI_bracket, HI_f, HI_gtd);
			tTmp2 := polyinterp_noboundry (HI_bracket, HI_f, HI_gtd, LO_bracket, min_bracketFval, gtd_LO);
			tTmp := IF (current_LO_id = 1, tTmp1, tTmp2);
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
			SELF.next_t := tZoom;
			SELF.insufProgress := insufprog_new;
			SELF := le;
		END;
		inputp_t_insuf := PROJECT (inputp, newt_tran(LEFT) ,LOCAL);//tzoom and insufficient progres values are calculated and embedded in the inputp dataset
		//evaluate new point
		// calculate t*d
		Elem := {REAL8 v};
		Layout_Part td_tran (inputp_t_insuf le, d ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := PBBlas.BLAS.dscal(cells, le.next_t, ri.mat_part, 1);
			SELF := le;
		END;
		td := JOIN (inputp_t_insuf, d, LEFT.partition_id = RIGHT.partition_id, td_tran (LEFT, RIGHT), LOCAL);
		// calculate x_new = x0 + td
		Layout_Part x_new_tran (Layout_part le, Layout_part ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := PBblas.BLAS.daxpy(cells, 1.0, le.mat_part, 1, ri.mat_part, 1);
			SELF := le;
		END;
		x_new := JOIN (x, td, LEFT.partition_id = RIGHT.partition_id, x_new_tran(LEFT, RIGHT), LOCAL);
		// calculated new g and cost values
		CostGrad_new := CostFunc(x_new,CostFunc_params,TrainData, TrainLabel);
		Elem gtd_tran (CostGrad_new le, d ri) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.v := sump(cells, le.mat_part, ri.mat_part);
		END;
		gtd_new_ := JOIN (CostGrad_new, d, LEFT.partition_id = RIGHT.partition_id, gtd_tran (LEFT, RIGHT), LOCAL);
		gtd_new := SUM (gtd_new_, gtd_new_.v);
		ZoomingRecord main_tran (inputp le_prev, CostGrad_new ri_new) := TRANSFORM
			f := le_prev.glob_f;
			current_LO_id := le_prev.id;
			new_id := IF (current_LO_id=1,2,1);
			fNewZoom := ri_new.cost_value;
			tZoom := le_prev.next_t;// the new point is calculates based on next_t value which was calculated at the begining of the zoom loop
			min_bracketFval := le_prev.cost_value;
			LO_bracket := le_prev.prev_t;
			HI_bracket := le_prev.high_t;
			gtdNewZoom := gtd_new;
			// f_new > f + c1*t*gtd || f_new >= f_LO
			zoom_cond_1 := (fNewZoom > f + c1 * tZoom * gtd) | (fNewZoom >= min_bracketFval);
			// abs(gtd_new) <= - c2*gtd
			zoom_cond_2 := ABS (gtdNewZoom) <= (-1 * c2 * gtd);
			// gtd_new*(bracket(HIpos)-bracket(LOpos)) >= 0
			zoom_cond_3 := gtdNewZoom * (HI_bracket - LO_bracket) >= 0; 
			whichcond := IF (zoom_cond_1, 11, IF (zoom_cond_2, 12, IF (zoom_cond_3, 13, -1)));
			cost_cond := min_bracketFval < fNewZoom; // f_lo < f_new
			lo_num := 1;
			new_num := 2;
			high_num := 3;
			lo_or_new := IF (zoom_cond_1, IF (cost_cond, lo_num, new_num), IF (zoom_cond_2, new_num, IF (zoom_cond_3, IF (cost_cond, lo_num, new_num), new_num)) );
			lo_or_new_id := IF (zoom_cond_1, IF (cost_cond, current_LO_id, new_id), IF (zoom_cond_2, current_LO_id, IF (zoom_cond_3, IF (cost_cond, new_id, current_LO_id), current_LO_id)) );
			lo_gtd := le_prev.prev_gtd;
			lo_t := le_prev.prev_t;
			//~done && abs((bracket(1)-bracket(2))*gtd_new) < tolX	
			zoom_term_cond := (~zoom_cond_1 AND zoom_cond_2) OR (ABS((tZoom-LO_bracket)*gtdNewZoom) < tolX);//if zoom_term_cond=TRUE then no need to assigne high values
			pre_mat := le_prev.mat_part;
			new_mat := ri_new.mat_part;
			//SELF.mat_part := IF (lo_or_new =1 , le_prev.mat_part, ri_new.mat_part);
			SELF.mat_part := IF (zoom_cond_1, IF (cost_cond, pre_mat, new_mat),IF (zoom_cond_2, new_mat, IF (zoom_cond_3, IF (cost_cond, pre_mat, new_mat), new_mat)));
			SELF.cost_value := IF (zoom_cond_1, IF (cost_cond, min_bracketFval, fNewZoom),IF (zoom_cond_2, fNewZoom, IF (zoom_cond_3, IF (cost_cond, min_bracketFval, fNewZoom), fNewZoom)));
			SELF.high_cost_value := IF (zoom_cond_1, IF (cost_cond, fNewZoom, min_bracketFval),IF (zoom_cond_2, le_prev.high_cost_value, IF (zoom_cond_3, IF (cost_cond, fNewZoom, min_bracketFval), le_prev.high_cost_value)));
			SELF.prev_gtd := IF (zoom_cond_1, IF (cost_cond, lo_gtd, gtdNewZoom),IF (zoom_cond_2, gtdNewZoom, IF (zoom_cond_3, IF (cost_cond, lo_gtd, gtdNewZoom), gtdNewZoom)));
			SELF.high_gtd := IF (zoom_cond_1, IF (cost_cond, gtdNewZoom, lo_gtd),IF (zoom_cond_2, le_prev.high_gtd, IF (zoom_cond_3, IF (cost_cond, gtdNewZoom, lo_gtd), le_prev.high_gtd)));
			SELF.prev_t := IF (zoom_cond_1, IF (cost_cond, lo_t, tZoom),IF (zoom_cond_2, tZoom, IF (zoom_cond_3, IF (cost_cond, lo_t, tZoom), tZoom)));
			SELF.high_t := IF (zoom_cond_1, IF (cost_cond, tZoom, lo_t),IF (zoom_cond_2, le_prev.high_t, IF (zoom_cond_3, IF (cost_cond, tZoom, lo_t), le_prev.high_t)));
			SELF.id := IF (zoom_cond_1, IF (cost_cond, current_LO_id, new_id), IF (zoom_cond_2, current_LO_id, IF (zoom_cond_3, IF (cost_cond, new_id, current_LO_id), current_LO_id)) );//here
			SELF.c := le_prev.c + 1;
			SELF.zoomtermination := zoom_term_cond;
			SELF.wolfe_funEvals := le_prev.wolfe_funEvals + 1;
			SELF.zooming_cond := whichcond;
			SELF := le_prev;
		END;
		Result := JOIN (inputp_t_insuf, CostGrad_new, LEFT.partition_id = RIGHT.partition_id, main_tran(LEFT, RIGHT), LOCAL);		
		RETURN Result;
	END;
	//zooming_result := LOOP(topass_zooming, (LEFT.zoomtermination = FALSE) AND (LEFT.c < (maxLS+1)), EXISTS(ROWS(LEFT)) , ZoomingStep(ROWS(LEFT), COUNTER)); orig
	zooming_result := LOOP(topass_zooming, (LEFT.zoomtermination = FALSE) AND (LEFT.c < (maxLS+1)), ZoomingStep(ROWS(LEFT), COUNTER));
	//RETURN BracketingStep (topass_bracketing, 1);
	//RETURN zooming_result;
	//RETURN LOOP(topass_bracketing, 1 , BracketingStep(ROWS(LEFT), COUNTER));
	//RETURN LOOP(topass_bracketing, (maxLS+1), LEFT.bracketing_cond = -1, BracketingStep(ROWS(LEFT), COUNTER));
	//RETURN bracketing_result;// works
	//RETURN zooming_result;
	RETURN zooming_result;
	//RETURN LOOP(topass_bracketing, LEFT.bracketing_cond = -1, BracketingStep(ROWS(LEFT), COUNTER)); // works correctly
	//RETURN BracketingStep(topass_bracketing, 1); works correctly
	//RETURN LOOP(topass_bracketing, 1, BracketingStep(ROWS(LEFT), COUNTER));works correctly
	//RETURN topass_zooming;
	//RETURN LOOP(topass_zooming, (LEFT.zoomtermination = FALSE) AND (LEFT.c < (maxLS+1)), FALSE , ZoomingStep(ROWS(LEFT), COUNTER)); // works
	//RETURN LOOP(topass_zooming, (LEFT.zoomtermination = FALSE) AND (LEFT.c < (maxLS+1)) , ZoomingStep(ROWS(LEFT), COUNTER));//works
	// At the end the final result returned from wolfe function contains the t value in prev_t field, the g vector in the Layout_part section and the f value in cost_value field and the gtd in the prev_gtd
END;// END WolfeLineSearch4_4_2_ttest


//x : starting location
// CostFunc_params : hyperparameters fo rthe cost function
// fr is equal to f so we do not pass fr

EXPORT ArmijoBacktrack(DATASET(Layout_Part) x, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Cell_nid) TrainData , DATASET(Layout_Part) TrainLabel,DATASET(CostGrad_Record) CostFunc (DATASET(Layout_Part) x, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Cell_nid) TrainData , DATASET(Layout_Part) TrainLabel), REAL8 t, DATASET(Layout_Part) d, DATASET(costgrad_record) g, REAL8 gtd, REAL8 c1=0.0001, REAL8 c2=0.9, INTEGER maxLS=25, REAL8 tolX=0.000000001):=FUNCTION
  // C++ functions
	REAL8 sumabs(PBblas.Types.dimension_t N, PBblas.Types.matrix_t M) := BEGINC++

    #body
    double result = 0;
		double tmpp ;
    double *cellm = (double*) m;
    uint32_t i;
		for (i=0; i<n; i++) {
		tmpp = (cellm[i]>=0) ? (cellm[i]) : (-1 * cellm[i]);
      result = result + tmpp;
    }
		return(result);

   ENDC++;
	REAL8 sump(PBblas.Types.dimension_t N, PBblas.Types.matrix_t M, PBblas.Types.matrix_t V) := BEGINC++
		#body
		double result = 0;
		double tmpp ;
		double *cellm = (double*) m;
		double *cellv = (double*) v;
		uint32_t i;
		for (i=0; i<n; i++) {
			tmpp =(cellm[i] * cellv [i]);
			result = result + tmpp;
		}
		return(result);

	ENDC++;
  // Evaluate the Objective and Gradient at the Initial Step
	Elem := {REAL8 v};
	//calculate x_new = x + td
	Layout_Part x_new_tran (Layout_part le, Layout_part ri) := TRANSFORM
		cells := ri.part_rows * ri.part_cols;
		SELF.mat_part := PBblas.BLAS.daxpy(cells, t, ri.mat_part, 1, le.mat_part, 1);
		SELF := le;
	END;
	x_new := JOIN (x, d, LEFT.partition_id = RIGHT.partition_id, x_new_tran(LEFT, RIGHT), LOCAL);
	// calculated new g and cost values at x_new
	CostGrad_new := CostFunc(x_new,CostFunc_params,TrainData, TrainLabel);
	Elem gtdnew_tran(CostGrad_new inrec, d drec) := TRANSFORM //hadamard product
		cells := inrec.part_rows * inrec.part_cols;
		SELF.v :=  sump(cells, inrec.mat_part, drec.mat_part);
	END;
	gtd_new_ := JOIN (CostGrad_new, d, LEFT.partition_id = RIGHT.partition_id, gtdnew_tran (LEFT, RIGHT), LOCAL);
	gtd_new := SUM (gtd_new_, gtd_new_.v);
	armfunEvals := 1;
	// a copy of each of the following values should be on each node in order to calculate the new t:
	// f_prev t_prev  f_new ->t_new
	ArmijoRecord := RECORD (CostGrad_Record)
		REAL8 fprev;
		REAL8 tprev;// this is the actualy previous t calculated in the previous iteration
		REAL8 prev_t;// this should actually be tnew, however in order for this record format to be consistent with the ourput of wolfe line search , the new t has to be named prev_t
		UNSIGNED wolfe_funevals;
		INTEGER armCond;
		REAL8 glob_f;
		REAL8 gtdnew;
		BOOLEAN islegal_gnew := TRUE;
		REAL8 local_sumd;
	END; // ArmijoRec
	ArmijoRecord_shorten := RECORD
		REAL8 fprev;
		REAL8 tprev;// this is the actualy previous t calculated in the previous iteration
		REAL8 prev_t;// this should actually be tnew, however in order for this record format to be consistent with the ourput of wolfe line search , the new t has to be named prev_t
		UNSIGNED wolfe_funevals;
		INTEGER armCond;
		REAL8 glob_f;
		REAL8 gtdnew;
		BOOLEAN islegal_gnew := TRUE;
		Layout_Part.partition_id;
		CostGrad_Record.cost_value;
		REAL8 local_sumd;
	END; // ArmijoRecord_shorten
	f_table := TABLE (g, {g.cost_value, g.partition_id}, LOCAL);

	topass_BackTracking_ := JOIN (CostGrad_new , f_table , LEFT.partition_id = RIGHT.partition_id, TRANSFORM (ArmijoRecord, SELF.gtdnew := gtd_new; SELF.glob_f := RIGHT.cost_value; SELF.armCond := IF (( LEFT.cost_value > RIGHT.cost_value + c1*t*gtd) OR (NOT ISVALID (LEFT.cost_value)), -1, 2);
	SELF.wolfe_funevals := armfunEvals; SELF.fprev := -1; SELF.tprev := -1; SELF.prev_t := t; SELF.local_sumd := -1; SELF := LEFT), LOCAL);
	topass_BackTracking := JOIN (topass_BackTracking_, d, LEFT.partition_id = RIGHT.partition_id, TRANSFORM (ArmijoRecord, SELF.local_sumd:=IF (LEFT.armCond=-1, sumabs(RIGHT.part_rows * RIGHT.part_cols, RIGHT.mat_part), -1) , SELF:=LEFT), LOCAL);// if we enter the loop (armCond==-1) then we will need sumabs(t*d) = abs(t) * sumabs(d) inside the loop in each iteration. so we calculate the local sumabs(d) values here and use them in the loop as t*sum(sumabs_local(d))
	BackTracking (DATASET (ArmijoRecord) armin, UNSIGNED armc) := FUNCTION
		//calculate new t value
		ArmijoRecord_shorten newt_tran (ArmijoRecord le) := TRANSFORM
			temp := le.prev_t;
			f_new := le.cost_value;
			funEvals := le.wolfe_funevals;
			f_prev := le.fprev;
			//if LS == 0 || ~isLegal(f_Nnew), since in my implementation I assumed LS=4 we only need to check second condition
			cond1 := NOT isvalid(f_new);
			//LS == 2 && isLegal(g_new)
			cond2 := le.islegal_gnew;
			//funEvals < 2 || ~isLegal(f_prev)
			cond3 := (funEvals<2) OR (NOT ISVALID (f_prev));
			armt1 := temp*0.5;
			// t = polyinterp([0 f gtd; t f_new g_new'*d],doPlot);
			armt2 := polyinterp_noboundry (0, le.glob_f, gtd, temp, le.cost_value, le.gtdnew) ;
			//t = polyinterp([0 f gtd; t f_new sqrt(-1)],doPlot);
			armt3 := polyinterp_noboundry_img2 (0, le.glob_f, gtd, temp, le.cost_value);
			// t = polyinterp([0 f gtd; t f_new sqrt(-1); t_prev f_prev sqrt(-1)],doPlot);
			armtelse := polyinterp_noboundry_3points_imag2_3 (0, le.glob_f, gtd, temp, le.cost_value, le.tprev, f_prev) ;
			armt_tmp := IF (cond1, armt1, IF (cond2, armt2, IF (cond3, armt3, armtelse)));
			//Adjust if change in t is too small/large
			armt := IF ( armt_tmp < temp*0.001 , temp*0.001, IF (armt_tmp >= temp*0.6, temp*0.6, armt_tmp));
			SELF.prev_t := armt;// the new t value
			SELF.tprev := temp;
			SELF.fprev := le.cost_value;
			SELF.wolfe_funevals := le.wolfe_funevals;
			SELF := le;
		END;
		armin_t := PROJECT (armin, newt_tran(LEFT), LOCAL);

		// calculate t*d
		Elem := {REAL8 v};
		Layout_Part td_tran (armin_t le, d ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := PBBlas.BLAS.dscal(cells, le.prev_t, ri.mat_part, 1);
			SELF := ri;
		END;
		armtd := JOIN (armin_t, d, LEFT.partition_id = RIGHT.partition_id, td_tran (LEFT, RIGHT), LOCAL);
		//calculate sum(abs(armtd))
		
		 Elem abs_td_tran (armin_t le) := TRANSFORM
				SELF.v:= ABS (le.prev_t) * le.local_sumd;
		 END;
		 sum_abs_td_ := PROJECT (armin_t, abs_td_tran(LEFT), LOCAL);
		 sum_abs_td := SUM (sum_abs_td_, sum_abs_td_.v);
		 //evaluate new point
		 // calculate x_new = x + td
		 Layout_Part x_new_tran (Layout_part le, Layout_part ri) := TRANSFORM
				cells := le.part_rows * le.part_cols;
				SELF.mat_part := PBblas.BLAS.daxpy(cells, 1.0, le.mat_part, 1, ri.mat_part, 1);
				SELF := le;
			END;
			armx_new := JOIN (x, armtd, LEFT.partition_id = RIGHT.partition_id, x_new_tran(LEFT, RIGHT), LOCAL);
			armCostGrad_new := CostFunc(armx_new,CostFunc_params,TrainData, TrainLabel);
			//calculate gtdnew
			Elem armgtdnew_tran(armCostGrad_new inrec, d drec) := TRANSFORM //hadamard product
			cells := inrec.part_rows * inrec.part_cols;
				SELF.v :=  sump(cells, inrec.mat_part, drec.mat_part);
			END;
			armgtd_new_ := JOIN (armCostGrad_new, d, LEFT.partition_id = RIGHT.partition_id, gtdnew_tran (LEFT, RIGHT), LOCAL);
			armgtd_new := SUM (armgtd_new_, armgtd_new_.v);
			
			ArmijoRecord armmain_tran (armCostGrad_new le, armin_t ri) := TRANSFORM
			//while f_new > fr + c1*t*gtd || ~isLegal(f_new)
			SELF.armCond := IF ( (le.cost_value > (ri.glob_f + (c1*ri.prev_t*gtd)) OR (NOT ISVALID(le.cost_value))) ,-1,2);
			SELF.tprev := ri.tprev;
			SELF.fprev := ri.fprev;
			SELF.wolfe_funevals := ri.wolfe_funevals + 1;
			SELF.prev_t := ri.prev_t;
			SELF.glob_f := ri.glob_f;
			//calculate islegalgnew ??
			islegal_gnew := TRUE;
			SELF.gtdnew := IF (ISVALID (le.cost_value) AND islegal_gnew, armgtd_new, -1);//gtd value is assigned/calculated only if it will be used in the next iteration. It will be used if the first arm cond is not satisfied and the second one is satisfied ??
			SELF.local_sumd := ri.local_sumd;
			SELF := le;
		END;
		arm_nextitr_out := JOIN (armCostGrad_new, armin_t, LEFT.partition_id = RIGHT.partition_id, armmain_tran(LEFT,RIGHT) , LOCAL);
		armshort_format := RECORD
			armin_t.wolfe_funevals;
			armin_t.partition_id;
		END;
		armfunevals_table := TABLE (armin_t, armshort_format, LOCAL);
		steptoosmall_out := JOIN (g, armfunevals_table, LEFT.partition_id = RIGHT.partition_id,TRANSFORM (ArmijoRecord, SELF.local_sumd := -1; SELF.gtdnew := -1; SELF.glob_f := LEFT.cost_value; SELF.armCond := 1; SELF.prev_t:=0; SELF.tprev := -1; SELF.fprev:= -1; SELF.wolfe_funevals := RIGHT.wolfe_funevals; SELF := LEFT), LOCAL);// in case we are returning this dataset, it means step size had been too small so the armCond should be 1
		//Check whether step size has become too small
    //if sum(abs(t*d)) <= tolX
		toosmall_cond := sum_abs_td <= tolX;
		// loop continues until f_new > fr + c1*t*gtd || ~isLegal(f_new) is TRUE, so the loop has to stop when NOT (f_new > fr + c1*t*gtd || ~isLegal(f_new)), in the other word when NOT (f_new > fr + c1*t*gtd ) AND isLegal(f_new)
		
		arm_out := IF (toosmall_cond, steptoosmall_out, arm_nextitr_out);

		
		RETURN arm_out; 
	END;// END BackTracking
	ar1 := BackTracking(topass_BackTracking,1);
	// RETURN BackTracking(ar1,2);
	RETURN LOOP (topass_BackTracking, LEFT.armCond = -1 , BackTracking(ROWS(LEFT),COUNTER));
	// RETURN LOOP (topass_BackTracking, 6 , BackTracking(ROWS(LEFT),COUNTER));
	// RETURN topass_BackTracking;
	// RETURN ar1;
	// RETURN topass_BackTracking;
END;// END ArmijoBacktrack
 
EXPORT WolfeLineSearch4_4_2_no(INTEGER cccc, DATASET(Layout_Part) x, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Cell_nid) TrainData , DATASET(Layout_Part) TrainLabel,DATASET(CostGrad_Record) CostFunc (DATASET(Layout_Part) x, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Cell_nid) TrainData , DATASET(Layout_Part) TrainLabel), REAL8 t, DATASET(Layout_Part) d, DATASET(costgrad_record) g, REAL8 gtd, REAL8 c1=0.0001, REAL8 c2=0.9, INTEGER maxLS=25, REAL8 tolX=0.000000001):=FUNCTION
	//C++ functions used
	//sum (M.*V)
	REAL8 sump(PBblas.Types.dimension_t N, PBblas.Types.matrix_t M, PBblas.Types.matrix_t V) := BEGINC++
		#body
		double result = 0;
		double tmpp ;
		double *cellm = (double*) m;
		double *cellv = (double*) v;
		uint32_t i;
		for (i=0; i<n; i++) {
			tmpp =(cellm[i] * cellv [i]);
			result = result + tmpp;
		}
		return(result);

	ENDC++;
	
	// We pass the prev g and cost values along with the next t value to the bracketing loop. The new g and cost values will be calculated at the begining of the loop
	//id field shows which side of the bracket prev values belong to (left (1) or right (2))
	//we embed the f value recived in wolfe function in what we pass to bracketing loop, so in the rest of the algorithm we have access to it in each node -> SELF.glob_f :=LEFT.cost_value
	topass_bracketing := PROJECT (g, TRANSFORM (ZoomingRecord, SELF.zooming_cond := -1; SELF.zoomtermination := FALSE; SELF.insufProgress := FALSE; SELF.id := 1; SELF.glob_f :=LEFT.cost_value; SELF.high_t := -1 ; SELF.high_gtd := -1; SELF.high_cost_value := -1; SELF.prev_t := 0; SELF.prev_gtd := gtd; SELF.wolfe_funEvals := 0; SELF.c := 0; SELF.bracketing_cond := -1; SELF.next_t := t; SELF := LEFT), LOCAL);
	BracketingStep (DATASET (ZoomingRecord) inputp, UNSIGNED coun) := FUNCTION
		//the idea is to calculate the new values at the begining of the bracket and then JOIN the loop input and this new values in order to generate the loop output. The JOIN TRANSFORM contains all the condition checks
		// calculate new g and cost value at the begining of the loop using next_t value in the input dataset
		// calculate t*d
		Elem := {REAL8 v};
		Layout_Part td_tran (inputp le, d ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := PBBlas.BLAS.dscal(cells, le.next_t, ri.mat_part, 1);
			SELF := le;
		END;
		td := JOIN (inputp, d, LEFT.partition_id = RIGHT.partition_id, td_tran (LEFT, RIGHT), LOCAL);
		//calculate x_new = x0 + td
		Layout_Part x_new_tran (Layout_part le, Layout_part ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := PBblas.BLAS.daxpy(cells, 1.0, le.mat_part, 1, ri.mat_part, 1);
			SELF := le;
		END;
		x_new := JOIN (x, td, LEFT.partition_id = RIGHT.partition_id, x_new_tran(LEFT, RIGHT), LOCAL);
		// calculated new g and cost values
		CostGrad_new := CostFunc(x_new,CostFunc_params,TrainData, TrainLabel);
		//calculated gtd_new (it will be used in IF conditions)
		Elem gtd_tran (CostGrad_new le, d ri) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.v := sump(cells, le.mat_part, ri.mat_part);
		END;
		gtd_new_ := JOIN (CostGrad_new, d, LEFT.partition_id = RIGHT.partition_id, gtd_tran (LEFT, RIGHT), LOCAL);
		gtd_new := SUM (gtd_new_, gtd_new_.v);
		//now that new values are calculated, JOIN prev (inputp) values and the new values considering all the conditions in order to produce the loop results
		//based on which condition is satisfied and whether previous cost (the cost in inputp) is less than the newly calculated cost (CostGrad_new) the return result is decided
		//we always return the braket side with smaller cost value, the values for the Hi sie of the bracket are in high_ fields. we don't return gradient vector for the high side
		BrackLSiter := coun - 1;
		ZoomingRecord main_tran (inputp le_prev, CostGrad_new ri_new) := TRANSFORM
			// calculate t for the next iteration (this value only will be calculated if there is going to be a next iteration)
			f := le_prev.glob_f; // the f value recived by wolfe function
			newt := le_prev.next_t; // the new values are calculated based on next_t so next_t is actually the new t in this iteration
			fNew := ri_new.cost_value;
			gtdNew := gtd_new;
			tPrev := le_prev.prev_t;
			fPrev := le_prev.cost_value;
			gtdPrev := le_prev.prev_gtd;
			minstep := newt + 0.01* (newt-tPrev);
			maxstep := newt*10;
			nextitr_t := polyinterp_both (tPrev, fPrev,gtdPrev, newt, fNew, gtdNew, minstep, maxstep);
			//specify the conditions along that which cost value is smaller
			cost_cond :=  (le_prev.cost_value < ri_new.cost_value);
			// f_new > f + c1*t*gtd || (LSiter > 1 && f_new >= f_prev)
			cond1 := ((fNew > f + c1 * newt* gtd)| ((BrackLSiter > 1) & (fNew >= fPrev))) & (coun < (maxLS+1) );
			// abs(gtd_new) <= -c2*gtd
			cond2 := (ABS(gtdNew) <= (-1*c2*gtd)) & (coun < (maxLS+1) );
			// gtd_new >= 0
			cond3 := (gtdNew >= 0 ) &(coun < (maxLS+1) ); // by adding & (coun < (maxLS+1) ) at the end of each condition I make sure that at the very last iteration no condition is satisfied and i just return the new value which make my algorithm works like the matlab algorithm
			//  I calculate new values at the begining of each loop itr, where the matlab code calculates it at end
		  // so at the maxLS I calculate next iteration t and go to the next iteration, then I calculate the new value which corresponds to new value calculated at the end of maxLS iteration in the matlab code, then I make sure no condition is satified
			// and I just return the new value at the (maxLS+1)  iteration which corresponds to new value at the end of maxLS iteration in matlab code
			WhichCond := IF (coun = (MAXLS+1),4 , IF (cond1, 1, IF(cond2, 2, IF (cond3,3,-1)))); // if we reach the maximum number of iterations or if one of the conditions is satified then we loopfilter retunr empty set and we exit the loop
			//decide whether we should return prev or new values
			prev_mat := le_prev.mat_part;
			new_mat := ri_new.mat_part;
			prev_num := 1;
			new_num := 2;
			prev_or_new := IF (cond1, (IF (cost_cond, prev_num, new_num)) , IF (cond2, new_num, IF (cond3, IF (cost_cond, prev_num, new_num) , new_num)));
			SELF.mat_part := IF (prev_or_new = 1 , prev_mat, new_mat);
			SELF.cost_value := IF (prev_or_new = 1 , fPrev, fNew);
			SELF.high_cost_value := IF (prev_or_new = 1 , fNew, fPrev);
			SELF.prev_gtd := IF (prev_or_new = 1 , gtdPrev, gtdNew);
			SELF.high_gtd := IF (prev_or_new = 1 , gtdNew, gtdPrev);
			SELF.prev_t := IF (prev_or_new = 1 , tPrev, newt);
			SELF.high_t := IF (prev_or_new = 1 , newt, tPrev);
			SELF.id := prev_or_new; // thd id shows whether the associated value to prev is the left or riht side of the bracket, the hight value will be the other value of id (if id=1 it means the prev values start the bracket and high ends the brackte) 
			//actually the id only matters when we exit the loop because of cond1 or cond3. cond2 and reaching maxitr casue program to end so id does not matter becasue we don't go to zooming loop
			SELF.bracketing_cond := WhichCond;
			SELF.c := coun;
			SELF.next_t := IF (WhichCond = -1 AND coun < (maxLS+1), nextitr_t, -1);// next_t is calculated only if we are going to the next iteration which means WhichCond should be -1 ( no condition has been satisfied) and we have not rached the Max number of iterations
			SELF.wolfe_funEvals := le_prev.wolfe_funEvals + 1;
			SELF := le_prev;
		END;
		Result := JOIN (inputp, CostGrad_new, LEFT.partition_id = RIGHT.partition_id, main_tran(LEFT, RIGHT), LOCAL);
		RETURN Result;
	END; // END BracketingStep
	ZoomingRecord maxitr_tran (ZoomingRecord le_brack , CostGrad_record ri_g) := TRANSFORM
		itr_num := le_brack.c;
		maxitr_cond := (itr_num = (maxLS+1));
		cost_cond_0 := ri_g.cost_value < le_brack.cost_value;
		SELF.mat_part := IF (maxitr_cond , IF (cost_cond_0, ri_g.mat_part, le_brack.mat_part) , le_brack.mat_part);
		SELF.cost_value := IF (maxitr_cond , IF (cost_cond_0, ri_g.cost_value, le_brack.cost_value) , le_brack.cost_value);
		SELF.prev_t := IF (maxitr_cond , IF (cost_cond_0, 0 , le_brack.prev_t) , le_brack.prev_t);
		SELF.prev_gtd := IF (maxitr_cond , IF (cost_cond_0, gtd, le_brack.prev_gtd) , le_brack.prev_gtd);
		SELF.zoomtermination := maxitr_cond OR (le_brack.bracketing_cond=2); // if we reach maximum number of iterations in the bracketing loop, or if condition 2 is satisfied in the bracketing loop. it means we don't go to the zooming loop and the wolfe algorithms ends
		SELF := le_brack;
	END;
	//bracketing_result_ := LOOP(topass_bracketing, LEFT.bracketing_cond = -1, EXISTS(ROWS(LEFT)) and COUNTER < (maxLS+1), BracketingStep(ROWS(LEFT), COUNTER)); orig, not work
	//bracketing_result_ := LOOP(topass_bracketing, (maxLS+1), LEFT.bracketing_cond = -1, BracketingStep(ROWS(LEFT), COUNTER)); works but runs maxls+1 time
	bracketing_result_ := LOOP(topass_bracketing, LEFT.bracketing_cond = -1, BracketingStep(ROWS(LEFT), COUNTER));
	//check if MAXITR is reached and decide between bracketing result and f (t=0) to be returned
	bracketing_result := JOIN (bracketing_result_ , g, LEFT.partition_id = RIGHT.partition_id, maxitr_tran (LEFT, RIGHT), LOCAL);//?? it should be the loop result not topassbracketing
	topass_zooming := bracketing_result;
	ZoomingStep (DATASET (ZoomingRecord) inputp, UNSIGNED coun) := FUNCTION
		//similar to BracketingStep, first we calculate the new point (for that we need to calculate the new t using interpolation)
		//after the new point is calculated we JOIN the input and the new point in order to decide which one to return
		// Compute new trial value & Test that we are making sufficient progress
		//The following transform calculates the new trial value as well as insufProgress value by doing a PROJECT on the input, the new t is then saved in next_t field
		ZoomingRecord newt_tran (ZoomingRecord le) := TRANSFORM
			insufprog := le.insufProgress;
			LO_bracket := le.prev_t;
			HI_bracket := le.high_t; 
			HI_f := le.high_cost_value;
			HI_gtd := le.high_gtd;
			BList := [LO_bracket,HI_bracket];
			max_bracket := MAX(Blist);
			min_bracket := MIN(Blist);
			current_LO_id := le.id;
			gtd_LO := le.prev_gtd;
			new_id := IF (current_LO_id=1,2,1);
			min_bracketFval := le.cost_value;
			tTmp1 := polyinterp_noboundry (LO_bracket, min_bracketFval, gtd_LO, HI_bracket, HI_f, HI_gtd);
			tTmp2 := polyinterp_noboundry (HI_bracket, HI_f, HI_gtd, LO_bracket, min_bracketFval, gtd_LO);
			tTmp := IF (current_LO_id = 1, tTmp1, tTmp2);
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
			SELF.next_t := tZoom;
			SELF.insufProgress := insufprog_new;
			SELF := le;
		END;
		inputp_t_insuf := PROJECT (inputp, newt_tran(LEFT) ,LOCAL);//tzoom and insufficient progres values are calculated and embedded in the inputp dataset
		//evaluate new point
		// calculate t*d
		Elem := {REAL8 v};
		Layout_Part td_tran (inputp_t_insuf le, d ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := PBBlas.BLAS.dscal(cells, le.next_t, ri.mat_part, 1);
			SELF := le;
		END;
		td := JOIN (inputp_t_insuf, d, LEFT.partition_id = RIGHT.partition_id, td_tran (LEFT, RIGHT), LOCAL);
		// calculate x_new = x0 + td
		Layout_Part x_new_tran (Layout_part le, Layout_part ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := PBblas.BLAS.daxpy(cells, 1.0, le.mat_part, 1, ri.mat_part, 1);
			SELF := le;
		END;
		x_new := JOIN (x, td, LEFT.partition_id = RIGHT.partition_id, x_new_tran(LEFT, RIGHT), LOCAL);
		// calculated new g and cost values
		CostGrad_new := CostFunc(x_new,CostFunc_params,TrainData, TrainLabel);
		Elem gtd_tran (CostGrad_new le, d ri) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.v := sump(cells, le.mat_part, ri.mat_part);
		END;
		gtd_new_ := JOIN (CostGrad_new, d, LEFT.partition_id = RIGHT.partition_id, gtd_tran (LEFT, RIGHT), LOCAL);
		gtd_new := SUM (gtd_new_, gtd_new_.v);
		ZoomingRecord main_tran (inputp le_prev, CostGrad_new ri_new) := TRANSFORM
			f := le_prev.glob_f;
			current_LO_id := le_prev.id;
			new_id := IF (current_LO_id=1,2,1);
			fNewZoom := ri_new.cost_value;
			tZoom := le_prev.next_t;// the new point is calculates based on next_t value which was calculated at the begining of the zoom loop
			min_bracketFval := le_prev.cost_value;
			LO_bracket := le_prev.prev_t;
			HI_bracket := le_prev.high_t;
			gtdNewZoom := gtd_new;
			// f_new > f + c1*t*gtd || f_new >= f_LO
			zoom_cond_1 := (fNewZoom > f + c1 * tZoom * gtd) | (fNewZoom >= min_bracketFval);
			// abs(gtd_new) <= - c2*gtd
			zoom_cond_2 := ABS (gtdNewZoom) <= (-1 * c2 * gtd);
			// gtd_new*(bracket(HIpos)-bracket(LOpos)) >= 0
			zoom_cond_3 := gtdNewZoom * (HI_bracket - LO_bracket) >= 0; 
			whichcond := IF (zoom_cond_1, 11, IF (zoom_cond_2, 12, IF (zoom_cond_3, 13, -2)));
			cost_cond := min_bracketFval < fNewZoom; // f_lo < f_new
			lo_num := 1;
			new_num := 2;
			lo_or_new := IF (zoom_cond_1, IF (cost_cond, lo_num, new_num), IF (zoom_cond_2, new_num, IF (zoom_cond_3, IF (cost_cond, lo_num, new_num), new_num)) );
			lo_or_new_id := IF (zoom_cond_1, IF (cost_cond, current_LO_id, new_id), IF (zoom_cond_2, current_LO_id, IF (zoom_cond_3, IF (cost_cond, new_id, current_LO_id), current_LO_id)) );
			//~done && abs((bracket(1)-bracket(2))*gtd_new) < tolX	
			zoom_term_cond := zoom_cond_2 OR (ABS((tZoom-LO_bracket)*gtdNewZoom) < tolX);//if zoom_term_cond=TRUE then no need to assigne high values
			SELF.mat_part := IF (lo_or_new =1 , le_prev.mat_part, ri_new.mat_part);
			SELF.cost_value := IF (lo_or_new =1 , le_prev.cost_value, ri_new.cost_value);
			SELF.high_cost_value := IF (lo_or_new =1 , ri_new.cost_value, le_prev.cost_value);
			SELF.prev_gtd := IF (lo_or_new =1 , le_prev.prev_gtd, gtdNewZoom);
			SELF.high_gtd := IF (lo_or_new =1 , gtdNewZoom, le_prev.prev_gtd);
			SELF.prev_t := IF (lo_or_new =1 , le_prev.prev_t, tZoom);
			SELF.high_t := IF (lo_or_new =1 , tZoom, le_prev.prev_t);
			SELF.id := lo_or_new_id;
			SELF.c := le_prev.c + 1;
			SELF.zoomtermination := zoom_term_cond;
			SELF.wolfe_funEvals := le_prev.wolfe_funEvals + 1;
			SELF := le_prev;
		END;
		Result := JOIN (inputp_t_insuf, CostGrad_new, LEFT.partition_id = RIGHT.partition_id, main_tran(LEFT, RIGHT), LOCAL);		
		RETURN Result;
	END;
	//zooming_result := LOOP(topass_zooming, (LEFT.zoomtermination = FALSE) AND (LEFT.c < (maxLS+1)), EXISTS(ROWS(LEFT)) , ZoomingStep(ROWS(LEFT), COUNTER)); orig
	zooming_result := LOOP(topass_zooming, (LEFT.zoomtermination = FALSE) AND (LEFT.c < (maxLS+1)), ZoomingStep(ROWS(LEFT), COUNTER));
	//RETURN BracketingStep (topass_bracketing, 1);
	//RETURN zooming_result;
	//RETURN LOOP(topass_bracketing, 1 , BracketingStep(ROWS(LEFT), COUNTER));
	//RETURN LOOP(topass_bracketing, (maxLS+1), LEFT.bracketing_cond = -1, BracketingStep(ROWS(LEFT), COUNTER));
	//RETURN bracketing_result;// works
	RETURN zooming_result;
	//RETURN LOOP(topass_bracketing, LEFT.bracketing_cond = -1, BracketingStep(ROWS(LEFT), COUNTER)); // works correctly
	//RETURN BracketingStep(topass_bracketing, 1); works correctly
	//RETURN LOOP(topass_bracketing, 1, BracketingStep(ROWS(LEFT), COUNTER));works correctly
	//RETURN topass_zooming;
	//RETURN LOOP(topass_zooming, (LEFT.zoomtermination = FALSE) AND (LEFT.c < (maxLS+1)), FALSE , ZoomingStep(ROWS(LEFT), COUNTER)); // works
	//RETURN LOOP(topass_zooming, (LEFT.zoomtermination = FALSE) AND (LEFT.c < (maxLS+1)) , ZoomingStep(ROWS(LEFT), COUNTER));//works
	// At the end the final result returned from wolfe function contains the t value in prev_t field, the g vector in the Layout_part section and the f value in cost_value field and the gtd in the prev_gtd
END;// END WolfeLineSearch4_4_2_no
    
 EXPORT WolfeLineSearch4_4(INTEGER cccc, DATASET(Layout_Part) x, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Part) TrainData , DATASET(Layout_Part) TrainLabel,DATASET(PBblas.Types.MUElement) CostFunc (DATASET(Layout_Part) x, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Part) TrainData , DATASET(Layout_Part) TrainLabel), UNSIGNED param_num, REAL8 t, DATASET(Layout_Part) d, REAL8 f, DATASET(Layout_Part) g, REAL8 gtd, REAL8 c1=0.0001, REAL8 c2=0.9, INTEGER maxLS=25, REAL8 tolX=0.000000001):=FUNCTION
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
    R1 := PROJECT(g_prev, load_scalars_g_prev(LEFT),LOCAL );
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
    R2 := PROJECT(g_new, load_scalars_g_new(LEFT),LOCAL );
    RETURN R1+R2;
   END; // END Load_bracketing_record
   ToPassBracketing := Load_bracketing_record;
	 
	 
	  bracketing_record4_4 load_scalars_g_prev_one (Layout_Part l) := TRANSFORM
        SELF.id := 1;
        SELF.f_ := f_prev;
        SELF.t_ := t_prev;
        SELF.funEvals_ := 0;
        SELF.gtd_ := gtd_prev;
        SELF.c := 0; //Counter
        SELF.Which_cond := -1;
				SELF.t_next := t;
        SELF := l;
      END;
	   zooming_record4_4 load_scalars_g_prev_one_2 (Layout_Part l) := TRANSFORM
        SELF.id := 1;
        SELF.f_ := f_prev;
        SELF.t_ := t_prev;
        SELF.funEvals_ := 0;
        SELF.gtd_ := gtd_prev;
        SELF.c := 0; //Counter
        SELF.Which_cond := -1;
				SELF.t_next := t;
        SELF := l;
      END;
  topassbracketing_one := PROJECT(g_prev, load_scalars_g_prev_one(LEFT),LOCAL );
	topassbracketing_one_2 := PROJECT(g_prev, load_scalars_g_prev_one_2(LEFT),LOCAL );
	// this function only passes one g to each iteration (does not  pass gPrev as apoosed to  previouse implementation of BracketingStep
	// If its the last iteration (one of the conditions is satisfied) then the function returns the input in addition to the newly calculated g
	BracketingStep_one (DATASET (bracketing_record4_4) inputp, INTEGER coun) := FUNCTION
		//calculate the next cost, gradient, gtd at the next t value
		in_table := TABLE(inputp, {id, f_,t_,t_next, funevals_,gtd_}, id, FEW);
		BrackfunEval := in_table[1].funEvals_;
		tPrev := in_table[1].t_;
		fPrev := in_table[1].f_;
		gtdPrev := in_table[1].gtd_;
		newt := in_table[1].t_next;
		xNew := PBblas.PB_daxpy(newt, d, x);
		CostGradNew := CostFunc(xNew,CostFunc_params,TrainData, TrainLabel);
		gNew := ExtractGrad (CostGradNew);
		fNew := ExtractCost (CostGradNew);
		gtdNew := SumProduct (gNew, d);
		// check conditions
    // 1- f_new > f + c1*t*gtd || (LSiter > 1 && f_new >= f_prev)
		BrackLSiter := coun-1;
    con1 := (fNew > f + c1 * newt* gtd)| ((BrackLSiter > 1) & (fNew >= fPrev));
    //2- abs(gtd_new) <= -c2*gtd orig
    con2 := ABS(gtdNew) <= (-1*c2*gtd);
    // 3- gtd_new >= 0
    con3 := gtdNew >= 0;
    WhichCon := IF (con1, 1, IF(con2, 2, IF (con3,3,-1)));
		//generate the results for the next iteration:
		//wrap gNew and other new values for the next iteration, the next value of t shuold be calculated (This is our return recordset when this is not the last iteration)
		minstep := newt + 0.01* (newt-tPrev);
		maxstep := newt*10;
		next_t := polyinterp_both (tPrev, fPrev,gtdPrev, newt, fNew, gtdNew, minstep, maxstep);
	  bracketing_record4_4 load_scalars_gNew (Layout_Part l) := TRANSFORM
        SELF.id := 1;
        SELF.f_ := fNew;
        SELF.t_ := newt;
        SELF.funEvals_ := BrackfunEval + 1;
        SELF.gtd_ := gtdNew;
        SELF.c := coun; //Counter
        SELF.Which_cond := -1; // there is a next step, so no condition has been satisfied and which_cond==-1
				SELF.t_next := next_t;
        SELF := l;
    END;
		next_itr_result := PROJECT(gNew, load_scalars_gNew(LEFT), LOCAL);
		//generate the results for the last iteration (in case one of conditions satisfies and this is our last iteration). we have to return the input in addition to the newly calculated g
		bracketing_record4_4 load_scalars_gNew_lastitr (Layout_Part l) := TRANSFORM
			SELF.id := 2;
			SELF.f_ := fNew;
			SELF.t_ := newt;
			SELF.funEvals_ := BrackfunEval + 1;
			SELF.gtd_ := gtdNew;
			SELF.c := coun; //Counter
			SELF.Which_cond := WhichCon;
			SELF.t_next := -1; // since this is the last iteration, t_next does not mean anything, there is no next iteration
			SELF := l;
    END;
		last_itr_result_2 := PROJECT(gNew, load_scalars_gNew_lastitr(LEFT), LOCAL);
		last_itr_result_1 := PROJECT(inputp, TRANSFORM(bracketing_record4_4, SELF.Which_cond := WhichCon; SELF.c := LEFT.c +1; SELF.funEvals_:= LEFT.funEvals_ +1; SELF := LEFT;), LOCAL);
		last_itr_result := last_itr_result_1 + last_itr_result_2;
		LoopResult := IF (WhichCon=-1, next_itr_result, IF (WhichCon=2, last_itr_result_2, last_itr_result));
    RETURN LoopResult;
	END;// END BracketingStep_one
	BracketingStep_one_2 (DATASET (zooming_record4_4) inputp, INTEGER coun) := FUNCTION
		inputp_ := inputp (which_cond=-1);
		//calculate the next cost, gradient, gtd at the next t value
		in_table := TABLE(inputp, {id, f_,t_,t_next, funevals_,gtd_}, id, FEW);
		BrackfunEval := in_table[1].funEvals_;
		tPrev := in_table[1].t_;
		fPrev := in_table[1].f_;
		gtdPrev := in_table[1].gtd_;
		newt := in_table[1].t_next;
		xNew := PBblas.PB_daxpy(newt, d, x);
		CostGradNew := CostFunc(xNew,CostFunc_params,TrainData, TrainLabel);
		gNew := ExtractGrad (CostGradNew);
		fNew := ExtractCost (CostGradNew);
		gtdNew := SumProduct (gNew, d);
		// check conditions
    // 1- f_new > f + c1*t*gtd || (LSiter > 1 && f_new >= f_prev)
		BrackLSiter := coun-1;
    con1 := (fNew > f + c1 * newt* gtd)| ((BrackLSiter > 1) & (fNew >= fPrev));
    //2- abs(gtd_new) <= -c2*gtd orig
    con2 := ABS(gtdNew) <= (-1*c2*gtd);
    // 3- gtd_new >= 0
    con3 := gtdNew >= 0;
    WhichCon := IF (con1, 1, IF(con2, 2, IF (con3,3,-1)));
		//generate the results for the next iteration:
		//wrap gNew and other new values for the next iteration, the next value of t shuold be calculated (This is our return recordset when this is not the last iteration)
		minstep := newt + 0.01* (newt-tPrev);
		maxstep := newt*10;
		next_t := polyinterp_both (tPrev, fPrev,gtdPrev, newt, fNew, gtdNew, minstep, maxstep);
	  zooming_record4_4 load_scalars_gNew (Layout_Part l) := TRANSFORM // if not condition has been satisfied and we are going for the next iteration (have not reached MAXITR yet)
        SELF.id := 1;
        SELF.f_ := fNew;
        SELF.t_ := newt;
        SELF.funEvals_ := BrackfunEval + 1;
        SELF.gtd_ := gtdNew;
        SELF.c := coun; //Counter
        SELF.Which_cond := -1; // there is a next step, so no condition has been satisfied and which_cond==-1
				SELF.t_next := next_t;
        SELF := l;
    END;
		next_itr_result := PROJECT(gNew, load_scalars_gNew(LEFT), LOCAL);
		//in case condition 1 or 3 is satisfied we have to return [pre new]. we need the LO between pre and new to pass to zooming so we return the one with lowet f value
		zooming_record4_4 load_scalars_gNew_lastitr (Layout_Part l) := TRANSFORM // condition 1 or 3 is satisfied and fnew is less than fpre
			SELF.id := 2;
			SELF.f_ := fNew;
			SELF.t_ := newt;
			SELF.funEvals_ := BrackfunEval + 1;
			SELF.gtd_ := gtdNew;
			SELF.c := coun; //Counter
			SELF.Which_cond := WhichCon;
			SELF.t_next := -1; // since this is the last iteration, t_next does not mean anything, there is no next iteration
			SELF.f_high := fPrev;
			SELF.t_high := tPrev;
			SELF.gtd_high := gtdPrev;
			SELF := l;
    END;
		zooming_record4_4 load_scalars_gNew_lastitr_cond2 (Layout_Part l) := TRANSFORM //// condition 2 is satisfied and we return gnew, the algorithm ends and no need to calculate next or high values
			SELF.id := 2;
			SELF.f_ := fNew;
			SELF.t_ := newt;
			SELF.funEvals_ := BrackfunEval + 1;
			SELF.gtd_ := gtdNew;
			SELF.c := coun; //Counter
			SELF.Which_cond := WhichCon;
			SELF.t_next := -1;
			SELF.LoopTermination := TRUE;
			SELF := l;
    END;
		// we have reached the maximum number of iteration, it means no condition has been satified for us to break the loop
		// the algorithm has to return [0 new]. since at the end we need the LO value, we return the lowest between f and fNew
		zooming_record4_4 load_scalars_gNew_maxitr (Layout_Part l) := TRANSFORM // we have reached maxitr and fNew is less than f
			SELF.id := 2;
			SELF.f_ := fNew;
			SELF.t_ := newt;
			SELF.funEvals_ := BrackfunEval + 1;
			SELF.gtd_ := gtdNew;
			SELF.c := coun; //Counter
			SELF.Which_cond := WhichCon;
			SELF.LoopTermination := TRUE;
			SELF.t_next := -1; 
			SELF := l;
    END;
		
		zooming_record4_4 load_scalars_g_maxitr (Layout_Part l) := TRANSFORM // we have reached maxitr and f is less than fNew
			SELF.id := 1;
			SELF.f_ := f;
			SELF.t_ := 0;
			SELF.funEvals_ := BrackfunEval + 1;
			SELF.gtd_ := gtd;
			SELF.c := coun; //Counter
			SELF.Which_cond := WhichCon;
			SELF.LoopTermination := TRUE;
			SELF.t_next := -1; // since this is the last iteration, t_next does not mean anything, there is no next iteration
			SELF := l;
    END; 
		
		max_itr_result_2 := PROJECT(gNew, load_scalars_gNew_maxitr(LEFT), LOCAL);
		max_itr_result_1 := PROJECT(g, load_scalars_g_maxitr(LEFT), LOCAL);
		max_itr_result := IF (fNew <= f, max_itr_result_2, max_itr_result_1);
		last_itr_cond2_result := PROJECT(gNew, load_scalars_gNew_lastitr_cond2(LEFT), LOCAL);
		last_itr_result_2 := PROJECT(gNew, load_scalars_gNew_lastitr(LEFT), LOCAL);
		last_itr_result_1 := PROJECT(inputp, TRANSFORM(zooming_record4_4, SELF.Which_cond := WhichCon; SELF.c := LEFT.c +1; SELF.funEvals_:= LEFT.funEvals_ +1; SELF.t_high := newt; SELF.f_high := fNew; SELF.gtd_high := gtdNew; SELF := LEFT;), LOCAL);
		last_itr_result := IF (fNew <= fPrev, last_itr_result_2, last_itr_result_1);
		//LoopResult := IF (WhichCon=-1, next_itr_result, IF (WhichCon=2, last_itr_result_2, last_itr_result));
		LoopResult := IF (WhichCon=1,last_itr_result, IF (WhichCon=2, last_itr_cond2_result, IF (WhichCon=3, last_itr_result, IF (coun = (maxLS+1), max_itr_result, next_itr_result ))) ); // orig maxls+1
		//LoopResult := IF (WhichCond = -1, last_itr_result, IF (WhichCond = -2, last_itr_cond2_result, IF (WhichCond = -3, last_itr_result, IF (coun = (maxLS+1), next_itr_result ,next_itr_result))));
   // RETURN LoopResult;
		RETURN IF (COUNT(inputp_)=0,inputp,LoopResult);
	END;// END BracketingStep_one_2
	
	
		BracketingStep_one_2_100 (DATASET (zooming_record4_4) inputp, INTEGER coun) := FUNCTION
		//calculate the next cost, gradient, gtd at the next t value
		in_table := TABLE(inputp, {id, f_,t_,t_next, funevals_,gtd_}, id, FEW);
		BrackfunEval := in_table[1].funEvals_;
		tPrev := in_table[1].t_;
		fPrev := in_table[1].f_;
		gtdPrev := in_table[1].gtd_;
		newt := in_table[1].t_next;
		xNew := PBblas.PB_daxpy(newt, d, x);
		CostGradNew := CostFunc(xNew,CostFunc_params,TrainData, TrainLabel);
		gNew := ExtractGrad (CostGradNew);
		fNew := ExtractCost (CostGradNew);
		gtdNew := SumProduct (gNew, d);
		// check conditions
    // 1- f_new > f + c1*t*gtd || (LSiter > 1 && f_new >= f_prev)
		BrackLSiter := coun-1;
    con1 := (fNew > f + c1 * newt* gtd)| ((BrackLSiter > 1) & (fNew >= fPrev));
    //2- abs(gtd_new) <= -c2*gtd orig
    con2 := ABS(gtdNew) <= (-1*c2*gtd);
    // 3- gtd_new >= 0
    con3 := gtdNew >= 0;
    WhichCon := IF (con1, 1, IF(con2, 2, IF (con3,3,-1)));
		//generate the results for the next iteration:
		//wrap gNew and other new values for the next iteration, the next value of t shuold be calculated (This is our return recordset when this is not the last iteration)
		minstep := newt + 0.01* (newt-tPrev);
		maxstep := newt*10;
		next_t := polyinterp_both (tPrev, fPrev,gtdPrev, newt, fNew, gtdNew, minstep, maxstep);
	  zooming_record4_4 load_scalars_gNew (Layout_Part l) := TRANSFORM // if not condition has been satisfied and we are going for the next iteration (have not reached MAXITR yet)
        SELF.id := 1;
        SELF.f_ := fNew;
        SELF.t_ := newt;
        SELF.funEvals_ := BrackfunEval + 1;
        SELF.gtd_ := gtdNew;
        SELF.c := coun; //Counter
        SELF.Which_cond := -1; // there is a next step, so no condition has been satisfied and which_cond==-1
				SELF.t_next := next_t;
        SELF := l;
    END;
		next_itr_result := PROJECT(gNew, load_scalars_gNew(LEFT), LOCAL);
		//in case condition 1 or 3 is satisfied we have to return [pre new]. we need the LO between pre and new to pass to zooming so we return the one with lowet f value
		zooming_record4_4 load_scalars_gNew_lastitr (Layout_Part l) := TRANSFORM // condition 1 or 3 is satisfied and fnew is less than fpre
			SELF.id := 2;
			SELF.f_ := fNew;
			SELF.t_ := newt;
			SELF.funEvals_ := BrackfunEval + 1;
			SELF.gtd_ := gtdNew;
			SELF.c := coun; //Counter
			SELF.Which_cond := WhichCon;
			SELF.t_next := -1; // since this is the last iteration, t_next does not mean anything, there is no next iteration
			SELF.f_high := fPrev;
			SELF.t_high := tPrev;
			SELF.gtd_high := gtdPrev;
			SELF := l;
    END;
		zooming_record4_4 load_scalars_gNew_lastitr_cond2 (Layout_Part l) := TRANSFORM //// condition 2 is satisfied and we return gnew, the algorithm ends and no need to calculate next or high values
			SELF.id := 2;
			SELF.f_ := fNew;
			SELF.t_ := newt;
			SELF.funEvals_ := BrackfunEval + 1;
			SELF.gtd_ := gtdNew;
			SELF.c := coun; //Counter
			SELF.Which_cond := WhichCon;
			SELF.t_next := -1;
			SELF.LoopTermination := TRUE;
			SELF := l;
    END;
		// we have reached the maximum number of iteration, it means no condition has been satified for us to break the loop
		// the algorithm has to return [0 new]. since at the end we need the LO value, we return the lowest between f and fNew
		zooming_record4_4 load_scalars_gNew_maxitr (Layout_Part l) := TRANSFORM // we have reached maxitr and fNew is less than f
			SELF.id := 2;
			SELF.f_ := fNew;
			SELF.t_ := newt;
			SELF.funEvals_ := BrackfunEval + 1;
			SELF.gtd_ := gtdNew;
			SELF.c := coun; //Counter
			SELF.Which_cond := WhichCon;
			SELF.LoopTermination := TRUE;
			SELF.t_next := -1; 
			SELF := l;
    END;
		
		zooming_record4_4 load_scalars_g_maxitr (Layout_Part l) := TRANSFORM // we have reached maxitr and f is less than fNew
			SELF.id := 1;
			SELF.f_ := f;
			SELF.t_ := t;
			SELF.funEvals_ := BrackfunEval + 1;
			SELF.gtd_ := gtd;
			SELF.c := coun; //Counter
			SELF.Which_cond := WhichCon;
			SELF.LoopTermination := TRUE;
			SELF.t_next := -1; // since this is the last iteration, t_next does not mean anything, there is no next iteration
			SELF := l;
    END;
		
		max_itr_result_2 := PROJECT(gNew, load_scalars_gNew_maxitr(LEFT), LOCAL);
		max_itr_result_1 := PROJECT(g, load_scalars_g_maxitr(LEFT), LOCAL);
		max_itr_result := IF (fNew <= f, max_itr_result_2, max_itr_result_1);
		last_itr_cond2_result := PROJECT(gNew, load_scalars_gNew_lastitr_cond2(LEFT), LOCAL);
		last_itr_result_2 := PROJECT(gNew, load_scalars_gNew_lastitr(LEFT), LOCAL);
		last_itr_result_1 := PROJECT(inputp, TRANSFORM(zooming_record4_4, SELF.Which_cond := WhichCon; SELF.c := LEFT.c +1; SELF.funEvals_:= LEFT.funEvals_ +1; SELF.t_high := newt; SELF.f_high := fNew; SELF.gtd_high := gtdNew; SELF := LEFT;), LOCAL);
		last_itr_result := IF (fNew <= fPrev, last_itr_result_2, last_itr_result_1);
		//LoopResult := IF (WhichCon=-1, next_itr_result, IF (WhichCon=2, last_itr_result_2, last_itr_result));
		LoopResult := IF (WhichCon=1,last_itr_result, IF (WhichCon=2, last_itr_cond2_result, IF (WhichCon=3, last_itr_result, IF (coun = (maxLS+1), max_itr_result, next_itr_result ))) ); // orig maxls+1
		//LoopResult := IF (WhichCond = -1, last_itr_result, IF (WhichCond = -2, last_itr_cond2_result, IF (WhichCond = -3, last_itr_result, IF (coun = (maxLS+1), next_itr_result ,next_itr_result))));
   // RETURN LoopResult;
		//RETURN IF (cccc=100 and coun=2,last_itr_cond2_result,LoopResult);
		RETURN LoopResult;
	END;// END BracketingStep_one_2_100
	 
	//BracketingResult_one := LOOP(topassbracketing_one, LEFT.Which_cond = -1,  EXISTS(ROWS(LEFT)) AND COUNTER <= (maxLS+1), BracketingStep_one(ROWS(LEFT),COUNTER)); orig we go one iteration more because gnew is calculated at the first iteration inside the loop instead of being calculated outside the loop
	BracketingResult_one := LOOP(topassbracketing_one, LEFT.Which_cond = -1,  EXISTS(ROWS(LEFT)) AND COUNTER <=3 , BracketingStep_one(ROWS(LEFT),COUNTER));
	//BracketingResult_one_2 := LOOP(topassbracketing_one_2, LEFT.Which_cond = -1,  EXISTS(ROWS(LEFT)) AND COUNTER <=(maxLS+1) , BracketingStep_one_2(ROWS(LEFT),COUNTER));//the orig 
	//BracketingResult_one_2 := LOOP(topassbracketing_one_2, LEFT.Which_cond = -1,  COUNTER <=(maxLS+1) , BracketingStep_one_2(ROWS(LEFT),COUNTER));
	//BracketingResult_one_2 := LOOP(topassbracketing_one_2,  LEFT.Which_cond = -1,  EXISTS(ROWS(LEFT)) AND COUNTER <=(maxLS+1) , BracketingStep_one_2(ROWS(LEFT),COUNTER));//this was first
	//BracketingResult_one_2 := LOOP(topassbracketing_one_2,  1, BracketingStep_one_2(ROWS(LEFT),COUNTER));
	BracketingResult_one_1 := BracketingStep_one_2(topassbracketing_one_2,1);
	BracketingResult_one_2 := BracketingStep_one_2(BracketingResult_one_1,2);
	BracketingResult_one_3 := BracketingStep_one_2(BracketingResult_one_2,3);
	BracketingResult_one_4 := BracketingStep_one_2(BracketingResult_one_3,4);
	BracketingResult_one_5 := BracketingStep_one_2(BracketingResult_one_4,5);
	final_BracketingResult_one := BracketingResult_one_5;
	//BracketingResult_one_2 := LOOP(topassbracketing_one_2, 1, LEFT.Which_cond = -1, BracketingStep_one_2(ROWS(LEFT),COUNTER));
	//BracketingResult_one_2 := LOOP(topassbracketing_one_2, 1 , BracketingStep_one_2(ROWS(LEFT),COUNTER));
	//BracketingResult_one is a bracket surrounding a point satisfying the criteria
	//We only pass the bracket side with lower cost (f) value, inside the loop we also need the t value for the higher side of bracket as well as the
	//interpolated t 
	//calculate the interpolated t count

	zooming_record4_4 topass_zoom_first_irt (bracketing_record4_4 le, bracketing_record4_4 ri):= TRANSFORM
		f_cond :=  (le.f_ <= ri.f_);
		SELF.t_next := polyinterp_noboundry (le.t_, le.f_, le.gtd_, ri.t_, ri.f_, ri.gtd_);
		SELF.LoopTermination:=(le.which_cond=2) | (le.c=(MaxLS+1));//if condition 2 is met in the bracketing step, or if we have reached to maximum number of iterations in the bracketing step, then there is no need to go to the zooming step and we set the zoomingtermination as TRUE
		SELF.t_high := IF(f_cond, ri.t_, le.t_);// return the t for the other side of the bracket with higher f value
		SELF.f_high := IF(f_cond, ri.f_, le.f_);
		SELF.gtd_high := IF(f_cond, ri.gtd_, le.gtd_);
		SELF := IF(f_cond, le, ri);// return the bracket side with the lowest f value
	END;
	
	//This function also pass on the LO bracket side
	ZoomingStep_one (DATASET (zooming_record4_4) inputp, INTEGER coun) := FUNCTION
	  inputp_ := inputp(LoopTermination = FALSE);
	  // extract new trial value
		in_table := TABLE(inputp, {id, f_,t_,t_high,f_high,gtd_high, t_next,funevals_,gtd_, insufProgress,c}, id, FEW);
		zoom_FunEval := in_table[1].funevals_;
		zoom_c := in_table[1].c;
		insufprog := in_table[1].insufProgress;
		LO_bracket := in_table[1].t_;
		HI_bracket := in_table[1].t_high; 
		HI_f := in_table[1].f_high;
		HI_gtd := in_table[1].gtd_high;
    BList := [LO_bracket,HI_bracket];
    max_bracket := MAX(Blist);
    min_bracket := MIN(Blist);
		current_LO_id := in_table[1].id;
		gtd_LO := in_table[1].gtd_;
		new_id := IF (current_LO_id=1,2,1);
		min_bracketFval := in_table[1].f_;
		tTmp1 := polyinterp_noboundry (LO_bracket, min_bracketFval, gtd_LO, HI_bracket, HI_f, HI_gtd);
		tTmp2 := polyinterp_noboundry (HI_bracket, HI_f, HI_gtd, LO_bracket, min_bracketFval, gtd_LO);
    //tTmp := in_table[1].t_next; orig
		tTmp := IF (current_LO_id = 1, tTmp1, tTmp2);
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
		
		// Evaluate new point with tZoom
    xNew := PBblas.PB_daxpy(tZoom, d, x);
    CostGradNew := CostFunc(xNew,CostFunc_params,TrainData, TrainLabel);
    //CostGradNew := myfunc(xNew,param_map,param_num);
    gNewZoom := ExtractGrad (CostGradNew);
    fNewZoom := ExtractCost (CostGradNew);
    //gtd_new = g_new'*d;
    //gtdNewZoom := Extractvalue(PBblas.PB_dgemm(TRUE, FALSE, 1.0,param_map, gNewZoom,param_map, d,one_map ));
    gtdNewZoom := SumProduct (gNewZoom, d);
		
		//Zoom Conditions
    
    //if f_new > f + c1*t*gtd || f_new >= f_LO
    zoom_cond_1 := (fNewZoom > f + c1 * tZoom * gtd) | (fNewZoom >= min_bracketFval);
    // if abs(gtd_new) <= - c2*gtd
    zoom_cond_2 := ABS (gtdNewZoom) <= (-1 * c2 * gtd);
    //if gtd_new*(bracket(HIpos)-bracket(LOpos)) >= 0
    zoom_cond_3 := gtdNewZoom * (HI_bracket - LO_bracket) >= 0; 
 
    whichcond := IF (zoom_cond_1, 11, IF (zoom_cond_2, 12, IF (zoom_cond_3, 13, -2)));
		

		//calculate next t for the next iteration (if there is going to be a next iteration)
		next_t_1 := polyinterp_noboundry (tZoom, fNewZoom, gtdNewZoom, LO_bracket, min_bracketFval, gtd_LO);
		next_t_2 := polyinterp_noboundry (LO_bracket, min_bracketFval, gtd_LO, tZoom, fNewZoom, gtdNewZoom);
		next_t_HI_1 := polyinterp_noboundry (tZoom, fNewZoom, gtdNewZoom, HI_bracket, HI_f, HI_gtd);
		next_t_HI_2 := polyinterp_noboundry (HI_bracket, HI_f, HI_gtd, tZoom, fNewZoom, gtdNewZoom);
		//cond1 := if f_new > f + c1*t*gtd || f_new >= f_LO
		zooming_record4_4 load_scalars_g_new_cond1 (Layout_Part l) := TRANSFORM // if we are wrapping g_new for cond1, it means that f_new has been less than the current f_LO: condition 1 is true and f_new is less than f_Lo so we return g_new for the next iteration
			zoom_term := ABS ((tZoom-LO_bracket)*gtdNewZoom) < tolX;
			SELF.id := new_id;
			SELF.f_ := fNewZoom;
			SELF.t_ := tZoom;
			SELF.funEvals_ := zoom_FunEval + 1;
			SELF.gtd_ := gtdNewZoom;
			SELF.c := zoom_c + 1; //Counter
			SELF.insufProgress := insufprog_new;
			SELF.which_cond := whichcond;
			SELF.LoopTermination := zoom_term;
			//SELF.t_next := IF(zoom_term, -1, IF (new_id=1, next_t_1,next_t_2)); // if this is the last iteration (zoom_term=TRUE) then there is no need to calculate these values for the next iteration
			SELF.t_next := 1;
			SELF.t_high := IF(zoom_term, -1, LO_bracket);
			SELF.f_high := IF(zoom_term, -1, min_bracketFval);
			SELF.gtd_high := IF(zoom_term, -1, gtd_LO);
			SELF := l;
		END;
		
		zooming_record4_4 LO_proj_cond1 (zooming_record4_4 l) := TRANSFORM // condition 1 is true and f_LO is less than f_new so we return the g_LO things for the next iteration
			zoom_term := ABS ((tZoom-LO_bracket)*gtdNewZoom) < tolX;
			SELF.funEvals_ := l.funEvals_ + 1;
			SELF.c := l.c + 1; //Counter
			SELF.insufProgress := insufprog_new;
			SELF.which_cond := whichcond;
			SELF.LoopTermination := zoom_term;
			//SELF.t_next := IF(zoom_term, -1, IF (new_id=1, next_t_1,next_t_2));
			SELF.t_high := IF(zoom_term, -1, tZoom);
			SELF.f_high := IF(zoom_term, -1, fNewZoom);
			SELF.gtd_high := IF(zoom_term, -1, gtdNewZoom);
			SELF := l;
		END;
		
		cond1_zoomresult := IF (fNewzoom <= min_bracketFval,Project(gNewZoom, load_scalars_g_new_cond1(LEFT),LOCAL), PROJECT(inputp, LO_proj_cond1(LEFT), LOCAL));
		
		zooming_record4_4 load_scalars_g_new_cond2 (Layout_Part l) := TRANSFORM // if condition 2 is satified we return g_new because we are sure than f_new is less than f_hi (otherwise codnition 1 would have been corret -> f_new>f_hi => f_new>f_lo)
			SELF.id := current_LO_id; // New point becomes new LO
			SELF.f_ := fNewZoom;
			SELF.t_ := tZoom;
			SELF.funEvals_ := zoom_FunEval + 1;
			SELF.gtd_ := gtdNewZoom;
			SELF.c := zoom_c + 1; //Counter
			SELF.insufProgress := insufprog_new;
			SELF.which_cond := whichcond;
			SELF.LoopTermination := TRUE;// sicne there is no other next iteration, we don't need to calculate the following values
			SELF.t_next := -1;
			SELF.t_high := -1;
			SELF.f_high := -1;
			SELF.gtd_high := -1;
			SELF := l;
		END; 
		cond2_zoomresult := PROJECT(gNewZoom, load_scalars_g_new_cond2(LEFT), LOCAL);

		zooming_record4_4 load_scalars_g_new_cond3 (Layout_Part l) := TRANSFORM //condition 3 is satified and f_new is less than f_lo
			zoom_term := ABS ((tZoom-LO_bracket)*gtdNewZoom) < tolX;
			SELF.id := current_LO_id;
			SELF.f_ := fNewZoom;
			SELF.t_ := tZoom;
			SELF.funEvals_ := zoom_FunEval + 1;
			SELF.gtd_ := gtdNewZoom;
			SELF.c := zoom_c + 1; //Counter
			SELF.insufProgress := insufprog_new;
			SELF.which_cond := whichcond;
			SELF.LoopTermination := zoom_term;
			//SELF.t_next := IF(zoom_term, -1, IF (new_id=1, next_t_2,next_t_1));
			SELF.t_next := -1;
			SELF.t_high := IF(zoom_term, -1, LO_bracket);
			SELF.f_high := IF(zoom_term, -1, min_bracketFval);
			SELF.gtd_high := IF(zoom_term, -1, gtd_LO);
			SELF := l;
		END;
		
		zooming_record4_4 LO_proj_cond3 (zooming_record4_4 l) := TRANSFORM
			zoom_term := ABS ((tZoom-LO_bracket)*gtdNewZoom) < tolX;
			SELF.id := new_id;
			SELF.funEvals_ := l.funEvals_ + 1;
			SELF.c := l.c + 1; //Counter
			SELF.insufProgress := insufprog_new;
			SELF.which_cond := whichcond;
			SELF.LoopTermination := zoom_term;
			//SELF.t_next := IF(zoom_term, -1, IF (new_id=1, next_t_2,next_t_1));
			SELF.t_high := IF(zoom_term, -1, tZoom);
			SELF.f_high := IF(zoom_term, -1, fNewZoom);
			SELF.gtd_high := IF(zoom_term, -1, gtdNewZoom);
			SELF := l;
		END;
		cond3_zoomresult := IF (fNewzoom <= min_bracketFval,Project(gNewZoom, load_scalars_g_new_cond3(LEFT),LOCAL),PROJECT(inputp, LO_proj_cond3(LEFT), LOCAL));
		
		zooming_record4_4 load_scalars_g_new_nocond (Layout_Part l) := TRANSFORM
			zoom_term := ABS ((tZoom-HI_bracket)*gtdNewZoom) < tolX;
			SELF.id := current_LO_id;
			SELF.f_ := fNewZoom;
			SELF.t_ := tZoom;
			SELF.funEvals_ := zoom_FunEval + 1;
			SELF.gtd_ := gtdNewZoom;
			SELF.c := zoom_c + 1; //Counter
			SELF.insufProgress := insufprog_new;
			SELF.which_cond := whichcond;
			SELF.LoopTermination := zoom_term;
			// SELF.t_next := IF(zoom_term, -1, IF (current_LO_id=1, next_t_HI_1,next_t_HI_2));
			SELF.t_next := -1;
			SELF.t_high := HI_bracket;
			SELF.f_high := HI_f;
			SELF.gtd_high := HI_gtd;
			SELF := l;
		END;
		nocond_zoomresult := PROJECT(gNewZoom, load_scalars_g_new_nocond(LEFT), LOCAL);
		
		zooming_result := IF (zoom_cond_1, cond1_zoomresult, IF (zoom_cond_2, cond2_zoomresult, IF (zoom_cond_3, cond3_zoomresult, nocond_zoomresult)));
		RETURN IF (COUNT(inputp_)=0,inputp,zooming_result);
	END;//END ZoomingStep_one
	
	brack_one_table := TABLE(BracketingResult_one, {id, c}, id, FEW);
  Zoom_one_Max_itr_tmp :=   maxLS + 1 - brack_one_table[1].c;
  Zoom_one_Max_Itr := IF (Zoom_one_Max_itr_tmp >0, Zoom_one_Max_itr_tmp, 0);
  topasszooming_one := final_BracketingResult_one;
	// ZoomingResult_one := LOOP(topasszooming_one, LEFT.LoopTermination = FALSE,  EXISTS(ROWS(LEFT)) AND COUNTER < Zoom_one_Max_Itr, ZoomingStep_one(ROWS(LEFT),COUNTER));
	do_zoom := EXISTS(topasszooming_one(LoopTermination=FALSE AND c < (maxLS + 1)));
	//ZoomingResult_one := LOOP(topasszooming_one, LEFT.LoopTermination = FALSE,  EXISTS(ROWS(LEFT)) AND COUNTER < Zoom_one_Max_Itr, ZoomingStep_one(ROWS(LEFT),COUNTER)); orig
	//ZoomingResult_one := LOOP(topasszooming_one, (maxLS + 1), LEFT.LoopTermination = FALSE , ZoomingStep_one(ROWS(LEFT),COUNTER));
	ZoomingResult_one := LOOP(topasszooming_one, 1 , ZoomingStep_one(ROWS(LEFT),COUNTER));//works
	ZoomingResult_one_1 := ZoomingStep_one(topasszooming_one,1);
	ZoomingResult_one_2 := ZoomingStep_one(ZoomingResult_one_1,2);
	final_ZoomingResult_one := ZoomingResult_one_2;
	//topasszooming_one_ := PROJECT (topasszooming_one, TRANSFORM(zooming_record4_4,SELF.LoopTermination:=FALSE; SELF:=LEFT),local);
	//ZoomingResult_one := LOOP(topasszooming_one_, LEFT.LoopTermination = FALSE,  EXISTS(ROWS(LEFT)) AND COUNTER < 2 , ZoomingStep_one(ROWS(LEFT),COUNTER));
	
	final_result_one := IF (do_zoom, final_ZoomingResult_one, topasszooming_one);

	// final_result := IF ((extr_loop_term_one(topasszooming_one)=TRUE), topasszooming_one , ZoomingResult_one);
	
	
   BracketingStep (DATASET (bracketing_record4) inputp, INTEGER coun) := FUNCTION
    // if ~isLegal(f_new) || ~isLegal(g_new) ????
    in_table := TABLE(inputp, {id, f_,t_,funevals_,gtd_}, id, FEW, LOCAL);// mylocal
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
    inputp_con :=  PROJECT(inputp, TRANSFORM(bracketing_record4, SELF.Which_cond := WhichCon; SELF:=LEFT),LOCAL);
    
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
      R_id_1 := PROJECT(inputp(id=2), load_scalars_g_prev(LEFT),LOCAL ); //id value is changed from 2 to 1. It is the same as :  f_prev = f_new;g_prev = g_new; gtd_prev = gtd_new; : the new values in the current loop iteration are actually prev values for the next iteration

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
      R_id_2 := PROJECT(gNewbrack, load_scalars_g_new(LEFT),LOCAL ); // scalar values are wrapped around gNewbrack with id=2 , these are actually the new values for the next iteration
     
      RETURN R_id_1 + R_id_2;
    END;

    LoopResult := IF (WhichCon=-1, bracketing_Nocon, inputp_con);
    //RETURN IF (COUNT(inputp)=0,inputp,LoopResult); orig
		RETURN LoopResult;
   END;//END BracketingStep
   
 // extr_which_cond (DATASET(bracketing_record4) i) := FUNCTION
    // in_table := TABLE(i, {which_cond}, FEW);
    // RETURN in_table[1].which_cond;
   // END;
   extr_loop_term (DATASET(zooming_record4) i) := FUNCTION
    in_table := TABLE(i, {LoopTermination}, FEW, LOCAL);//mylocal
    RETURN in_table[1].LoopTermination;
   END;
  
  // BracketingResult := LOOP(ToPassBracketing, COUNTER <= maxLS AND loopcond(ROWS(LEFT))=-1 , BracketingStep(ROWS(LEFT),COUNTER)); 
  //BracketingResult := LOOP(ToPassBracketing, maxLS, LEFT.Which_cond = -1, BracketingStep(ROWS(LEFT),COUNTER)); 
  BracketingResult_ := LOOP(ToPassBracketing, maxLS, LEFT.Which_cond = -1, BracketingStep(ROWS(LEFT),COUNTER));
	//BracketingResult := LOOP(ToPassBracketing, maxLS, LEFT.Which_cond = -1,  EXISTS(ROWS(LEFT)) AND COUNTER <= maxLS, BracketingStep(ROWS(LEFT),COUNTER)); myorig
	BracketingResult := LOOP(ToPassBracketing, LEFT.Which_cond = -1,  EXISTS(ROWS(LEFT)) AND COUNTER <= maxLS , BracketingStep(ROWS(LEFT),COUNTER));
  //BracketingResult :=  LOOP(Topassbracketing, extr_which_cond(ROWS(LEFT))=-1 AND  COUNTER <maxLS  , bracketingstep(ROWS(LEFT),COUNTER)); orig
  brack_table := TABLE(BracketingResult, {id, c}, id, FEW, LOCAL);//mylocal
 
  Zoom_Max_itr_tmp :=   maxLS - brack_table[1].c;
  Zoom_Max_Itr := IF (Zoom_Max_itr_tmp >0, Zoom_Max_itr_tmp, 0);

   // Zoom Phase
   
   //We now either have a point satisfying the criteria, or a bracket
   //surrounding a point satisfying the criteria
   // Refine the bracket until we find a point satisfying the criteria
   toto := PROJECT (BracketingResult, TRANSFORM(zooming_record4 ,SELF.LoopTermination:=(LEFT.which_cond=2) | (LEFT.c=MaxLS); SELF := LEFT), LOCAL); // myloca, If in the bracketing step condition 2 has been met or we have reaached MaxLS then we don't need to pass zoom step, so the termination condition for zoom will be set as true here
   //toto2 := PROJECT (toto, TRANSFORM(zooming_record4 ,SELF.c:=100; SELF := LEFT)); 
   ZoomingStep (DATASET (zooming_record4) inputp, INTEGER coun) := FUNCTION
    // At the begining of the loop find High and Low Points in bracket:
    // Assign id=1 to the low point
    // Assign id=2 to the high point
    // pass_thru := inputp0(LoopTermination = TRUE);
    // inputp:= inputp0(LoopTermination = FALSE);
    in_table := TABLE(inputp, {id, f_,t_,funevals_,gtd_, insufProgress,c}, id, FEW, LOCAL);// mylocal
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
      R_HI_id := PROJECT(gNewZoom, load_scalars_g_new(LEFT),LOCAL );
      zooming_record4 load_scalars_LOID (zooming_record4 l) := TRANSFORM
        SELF.funEvals_ := zoom_funevals_new;
        SELF.c := zoom_c_new; //Counter
        SELF.insufProgress := insufprog_new;
        SELF.which_cond := whichcond;
        SELF.LoopTermination := zoomter;
        SELF := l;
      END;
      R_LO_id := PROJECT (inputp (id=LO_id), load_scalars_LOID(LEFT),LOCAL );
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
      R_HI_id := PROJECT(inputp(id=HI_id), HIID(LEFT),LOCAL );
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
      R_LO_id := PROJECT(gNewZoom, load_scalars_g_new(LEFT),LOCAL );
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
      R_HI_id := PROJECT(inputp(id=LO_id), LOID(LEFT) ,LOCAL);
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
      R_LO_id := PROJECT(gNewZoom, load_scalars_g_new(LEFT) ,LOCAL);
      RETURN R_HI_id + R_LO_id;
    END;
    zooming_nocond := zooming_cond_2;
    
    zooming_result := IF (zoom_cond_1, zooming_cond_1, IF (zoom_cond_2, zooming_cond_2, IF (zoom_cond_3, zooming_cond_3, zooming_nocond )));
    
    //zooming_result2 :=  PROJECT (toto, TRANSFORM(zooming_record4 ,SELF.f_:=bracketGTDval_1; SELF := LEFT)); 
    RETURN IF (COUNT(inputp)=0,inputp,zooming_result); //ZoomingStep produces an output even when it recives an empty dataset, I would like to avoid that so I can make sure when loopfilter avoids a dataset to be passed to the loop (in case of which_cond==2 which means we already have found final t) then zooming_step would not produce any output  
    //RETURN zooming_result;
   END; // END ZoomingStep
     
   //BracketingResult is provided as input to the Zooming LOOP 
   //Since in the bracketing results in case of cond 1 and cond2 the prev and new values are assigned to bracket_1 and bracket_2 so when we pass the bracketing result to the zooming loop, in fact we have:
   //id = 1 indicates bracket_1 
   //id = 2 indicates bracket_2
  // LOOP( dataset, loopcount, loopfilter, loopbody [, PARALLEL( iterations | iterationlist [, default ] ) ] )
  Topass_zooming := PROJECT (BracketingResult, TRANSFORM(zooming_record4 ,SELF.LoopTermination:=(LEFT.which_cond=2) | (LEFT.c=MaxLS); SELF := LEFT),LOCAL); // If in the bracketing step condition 2 has been meet or we have reaached MaxLS then we don't need to pass zoom step, so the termination condition for soom will be set as true here
  //ZoomingResult := IF (extr_loop_term(Topass_zooming)=TRUE, Topass_zooming , LOOP(Topass_zooming, extr_loop_term(ROWS(LEFT))=FALSE AND  COUNTER <=Zoom_Max_Itr  , zoomingstep(ROWS(LEFT),COUNTER))); orig
  // ZoomingResult := IF (extr_loop_term(Topass_zooming)=TRUE, Topass_zooming , LOOP(Topass_zooming, Zoom_Max_Itr, LEFT.LoopTermination=FALSE , zoomingstep(ROWS(LEFT),COUNTER))); myorig
	 //ZoomingResult := LOOP(topasszooming_one, LEFT.LoopTermination = FALSE AND LEFT.c < (maxLS + 1),  EXISTS(ROWS(LEFT)), ZoomingStep_one(ROWS(LEFT),COUNTER));
	ZoomingResult:= IF (extr_loop_term(Topass_zooming)=TRUE, Topass_zooming , LOOP(Topass_zooming, LEFT.LoopTermination = FALSE AND LEFT.c < (maxLS + 1),  EXISTS(ROWS(LEFT)), zoomingstep(ROWS(LEFT),COUNTER)));
	 
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
   zoomTBL :=  TABLE(ZoomingResult, {id, which_cond, f_}, id, FEW, LOCAL);//mylocal
   zoomfnew := zoomTBL(id=2)[1].f_;
   zoomfold := zoomTBL(id=1)[1].f_;
   wolfe_cond := zoomTBL[1].which_cond;
   final_t_found := wolfe_cond = 2;
   t_new_result := PROJECT (ZoomingResult (id=2), TRANSFORM(bracketing_record4 ,SELF := LEFT),LOCAL);
   t_old_result := PROJECT (ZoomingResult (id=1), TRANSFORM(bracketing_record4 ,SELF := LEFT),LOCAL);
   t_0_result := PROJECT(g, load_scalars_g(LEFT),LOCAL );
   final_t_result := t_new_result;
   MaxLS_result := IF ( zoomfnew < f, t_new_result , t_0_result);
   zoom_result := IF ( zoomfnew < zoomfold, t_new_result , t_old_result);
   wolfe_result := IF (final_t_found,final_t_result , IF (Zoom_Max_itr_tmp=0,MaxLS_result,zoom_result));
   //RETURN wolfe_result; orig
	 //RETURN BracketingStep(topassbracketing,1);
	 brackR1 := BracketingStep(topassbracketing,1);
	 
	 //RETURN BracketingStep(brackR1,2);
	 //RETURN BracketingResult;
	 //RETURN LOOP(ToPassBracketing, 2, BracketingStep(ROWS(LEFT),COUNTER));
	 //RETURN LOOP(ToPassBracketing, 3, LEFT.Which_cond = -1, BracketingStep(ROWS(LEFT),COUNTER));
	 //RETURN  LOOP(ToPassBracketing, 2, LEFT.Which_cond = -1, BracketingStep(ROWS(LEFT),COUNTER));
	 AAA := BracketingStep_one (topassbracketing_one, 1);
	 //RETURN LOOP(ToPassBracketing, LEFT.Which_cond = -1,  COUNTER <= 2, BracketingStep(ROWS(LEFT),COUNTER));
	 //RETURN LOOP(topassbracketing_one, LEFT.Which_cond = -1,  EXISTS(ROWS(LEFT)) AND COUNTER <= 20, BracketingStep_one(ROWS(LEFT),COUNTER));
	// RETURN LOOP(topassbracketing, LEFT.Which_cond = -1,  EXISTS(ROWS(LEFT)) AND COUNTER <= 20, BracketingStep(ROWS(LEFT),COUNTER));
	// RETURN IF (cccc=100,LOOP(topassbracketing_one_2,   COUNTER <=1 , BracketingStep_one_2_100(ROWS(LEFT),COUNTER)) ,final_result_one);
	// RETURN topasszooming_one;
	RETURN final_result_one;
	//RETURN topassbracketing_one_2;
	//RETURN BracketingResult_one_2;
	
	 //RETURN LOOP(topassbracketing, LEFT.Which_cond = -1,  COUNTER <= 2, BracketingStep(ROWS(LEFT),COUNTER));
	 //RETURN BracketingStep_one (topassbracketing_one ,LEFT.Which_cond = -1,  COUNTER <= 2, BracketingStep_one(ROWS(LEFT),COUNTER));
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

    
 END;// END WolfeLineSearch4_4






EXPORT wolfe_g_4_4 (DATASET(zooming_record4_4) wolfeout) := FUNCTION
  RETURN PROJECT(wolfeout, TRANSFORM(Layout_Part, SELF:=LEFT), LOCAL);
END;
EXPORT wolfe_f_4_4 (DATASET(zooming_record4_4) wolfeout) := FUNCTION
  t := TABLE(wolfeout, {f_}, FEW, LOCAL);
  RETURN t[1].f_;
END;

EXPORT wolfe_t_4_4 (DATASET(zooming_record4_4) wolfeout) := FUNCTION
  t := TABLE(wolfeout, {t_}, FEW);
  RETURN t[1].t_;
END;

EXPORT wolfe_funEvals_4_4 (DATASET(zooming_record4_4) wolfeout) := FUNCTION
  t := TABLE(wolfeout, {funEvals_});
  RETURN t[1].funEvals_;
END;






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
  EXPORT ArmijoBacktrack2(DATASET(Types.NumericField)x, REAL8 t, DATASET(Types.NumericField)d, REAL8 f, REAL8 fr, DATASET(Types.NumericField) g, REAL8 gtd, REAL8 c1=0.0001, REAL8 tolX=0.000000001,DATASET(Types.NumericField) CostFunc_params, DATASET(Types.NumericField) TrainData , DATASET(Types.NumericField) TrainLabel, DATASET(Types.NumericField) CostFunc (DATASET(Types.NumericField) x, DATASET(Types.NumericField) CostFunc_params, DATASET(Types.NumericField) TrainData , DATASET(Types.NumericField) TrainLabel), prows=0, pcols=0, Maxrows=0, Maxcols=0):=FUNCTION
    
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
  END; // END ArmijoBacktrack2

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
EXPORT lbfgs_ex (DATASET(minfRec) min_in) := FUNCTION
REAL8 sump(PBblas.Types.dimension_t N, PBblas.Types.matrix_t M, PBblas.Types.matrix_t V) := BEGINC++

					#body
					double result = 0;
					double tmpp ;
					double *cellm = (double*) m;
					double *cellv = (double*) v;
					uint32_t i;
					for (i=0; i<n; i++) {
						tmpp =(cellm[i] * cellv [i]);
						result = result + tmpp;
					}
					return(result);

				ENDC++;
corrections := 3;
			itr_n := MAX (min_in, min_in.update_itr);
      k := IF (itr_n >corrections, corrections, itr_n); // k is the number of previous step vectors which are already stored
			// q LOOP
			w_2part := MAX(min_in, min_in.partition_id)-2;
			q_step (DATASET(minfRec) q_inp, unsigned4 q_c) := FUNCTION
				q_itr := itr_n- q_c + 1;
				q := q_inp(no=1);//this is the q vector
				s_tmp := min_in (no = 3 AND update_itr = q_itr);
				y_tmp := min_in (no = 4 AND update_itr = q_itr);
				//calculate al : al(i) = ro(i)*s(:,i)'*q(:,i+1);
				simple := {REAL8 v};
				simple al_tran (minfRec q_in, minfRec s_in) := TRANSFORM
				  cells := q_in.part_rows * q_in.part_cols;
					SELF.v := (1/s_in.sty) * sump(cells, q_in.mat_part, s_in.mat_part);
				END;
				
				al_ := JOIN (q, s_tmp, LEFT.partition_id = RIGHT.partition_id, al_tran (LEFT, RIGHT), LOCAL);
				al := SUM (al_, al_.v);
				// calculate new q vector : q(:,i) = q(:,i+1)-al(i)*y(:,i);
				minfRec new_q_tran (minfRec le, minfRec ri):= TRANSFORM
					cells := le.part_rows * le.part_cols;
			    SELF.mat_part := PBblas.BLAS.daxpy(cells, -1 * al, le.mat_part, 1, ri.mat_part, 1);
					SELF := ri;//by assigning q part (right hand side) to SELF we make sure we are keeping funevals and other information
				END;
				new_q := JOIN (y_tmp, q, LEFT.partition_id = RIGHT.partition_id, new_q_tran (LEFT, RIGHT), LOCAL);
				//normalize al to new_q and return the result
				minfRec norm_al (minfRec le) := TRANSFORM
					SELF.mat_part := [al];
					SELF.no := q_itr + 1;// 1 is added because the final q_itr will be 1 and we have already reserved no=1 for the q vector itself
					SELF := le;
				END;
				al_norm_ := NORMALIZE(new_q(no=1), 1,norm_al(LEFT));
				al_norm := ASSERT(al_norm_, node_id = Thorlib.node() and node_id=(partition_id-1), 'al is not well distributed in the lbfgs function', FAIL);
				
				// al_norm := ASSERT(al_norm_, (partition_id<=w_2part and node_id=Thorlib.node() and node_id=(partition_id-1)) or (partition_id>w_2part and node_id=Thorlib.node() and node_id = w_2part ), 'al is not well distributed in the lbfgs function', FAIL);
				
        RETURN new_q + al_norm + q_inp (no != 1);
      END; //END q_step
			minfRec steep_tran (minfRec le) := TRANSFORM
				cells := le.part_rows * le.part_cols;
				SELF.mat_part := PBBlas.BLAS.dscal(cells, -1, le.mat_part, 1);//multiply by -1
				SELF := le;
			END;
			topass_q := PROJECT (min_in(no=1), steep_tran(LEFT),LOCAL);//contains funevals and cost_value, h information from the previous iteration in minfunc function. So basically the h field (hdiag) belongs to what has been calculated in the previous iteratio and can be used here in topass_r
			q_result := LOOP(topass_q, COUNTER <= k, q_step(ROWS(LEFT),COUNTER));
			// r loop
			r_step (DATASET(minfRec) r_inp, unsigned4 r_c) := FUNCTION
				r_itr := r_c;
				s_tmp := min_in (no = 3 AND update_itr = r_itr);
				y_tmp := min_in (no = 4 AND update_itr = r_itr);
				//calculate be be(i) = ro(i)*y(:,i)'*r(:,i);
				simple := {REAL8 v};
				simple be_tran (minfRec r_in, minfRec y_in) := TRANSFORM
				  cells := r_in.part_rows * r_in.part_cols;
					SELF.v := (1/y_in.sty) * sump(cells, r_in.mat_part, y_in.mat_part);
				END;
				be_ := JOIN (r_inp, y_tmp, LEFT.partition_id = right.partition_id, be_tran (LEFT, RIGHT), LOCAL);
				be := SUM (be_, be_.v);
				// calculate (al (i) - be ) * s (i)
				minfRec s_tran (minfRec s_in, minfRec al_in) := TRANSFORM
					cells := s_in.part_rows * s_in.part_cols;
					SELF.mat_part := PBBlas.BLAS.dscal(cells, (al_in.mat_part[1] - be), s_in.mat_part, 1);
					SELF := s_in;
				END;
				al_be_s := JOIN (s_tmp, q_result (no = (r_itr + 1)) , LEFT.partition_id = RIGHT. partition_id, s_tran (LEFT, RIGHT), LOCAL);
				// calculate new_r := r + al_be_s
				minfRec new_r_tran (minfRec le, minfRec ri):= TRANSFORM
					cells := le.part_rows * le.part_cols;
			    SELF.mat_part := PBblas.BLAS.daxpy(cells, 1.0, le.mat_part, 1, ri.mat_part, 1);
					SELF := ri;//again here by assigning the r part (RIGHT) we are making sure we are keeping all the field initially comming from g (min_in (no=1)) such as funevals field and etc.
				END;
				new_r := JOIN (al_be_s, r_inp, LEFT.partition_id = RIGHT.partition_id, new_r_tran (LEFT, RIGHT), LOCAL);
				RETURN new_r; 
			END;// END r_step
			minfRec r_pass_tran (minfRec le) := TRANSFORM
				cells := le.part_rows * le.part_cols;
				SELF.mat_part := PBBlas.BLAS.dscal(cells, le.h, le.mat_part, 1);
				SELF := le;
			END;
			//r(:,1) = Hdiag*q(:,1);
			topass_r := PROJECT (q_result (no=1), r_pass_tran (LEFT), LOCAL);
			final_d := LOOP(topass_r, COUNTER <= k, r_step(ROWS(LEFT),COUNTER));
		  //RETURN q_result;
			// q1 := q_step(topass_q,1);
			r1 := r_step(topass_r,1);
			r2 := r_step(r1,2);
			RETURN q_result(no=1);
    END; // END lbfgs_ex
  EXPORT MinFUNC(DATASET(Layout_Part) x0,DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Cell_nid) TrainData , DATASET(Layout_Part) TrainLabel,DATASET(costgrad_record) CostFunc (DATASET(Layout_Part) x0, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Cell_nid) TrainData , DATASET(Layout_Part) TrainLabel), INTEGER8 param_num, INTEGER8 MaxIter = 100, REAL8 tolFun = 0.00001, REAL8 TolX = 0.000000001, INTEGER maxFunEvals = 1000, INTEGER corrections = 100, prows=0, pcols=0, Maxrows=0, Maxcols=0) := FUNCTION
		//C++ function used
		//sum(abs(M(:)))
		REAL8 sumabs(PBblas.Types.dimension_t N, PBblas.Types.matrix_t M) := BEGINC++

    #body
    double result = 0;
		double tmpp ;
    double *cellm = (double*) m;
    uint32_t i;
		for (i=0; i<n; i++) {
		tmpp = (cellm[i]>=0) ? (cellm[i]) : (-1 * cellm[i]);
      result = result + tmpp;
    }
		return(result);

   ENDC++;
		REAL8 sum_sq(PBblas.Types.dimension_t N, PBblas.Types.matrix_t M) := BEGINC++

			#body
			double result = 0;
			double tmpp ;
			double *cellm = (double*) m;
			uint32_t i;
			for (i=0; i<n; i++) {
				result = result + (cellm[i]*cellm[i]);
			}
			return(result);

		ENDC++;
		//sum (M.*V)
		REAL8 sump(PBblas.Types.dimension_t N, PBblas.Types.matrix_t M, PBblas.Types.matrix_t V) := BEGINC++

					#body
					double result = 0;
					double tmpp ;
					double *cellm = (double*) m;
					double *cellv = (double*) v;
					uint32_t i;
					for (i=0; i<n; i++) {
						tmpp =(cellm[i] * cellv [i]);
						result = result + tmpp;
					}
					return(result);

				ENDC++;
			
		//sum(gin(:).^2)
		sum_square (DATASET(Layout_Part) g_in) := FUNCTION
      Elem := {REAL8 v};  //short-cut record def
      Elem su(Layout_Part xrec) := TRANSFORM //hadamard product
        SELF.v :=  sum_sq(xrec.part_rows * xrec.part_cols, xrec.mat_part);
      END;
      ss_ := PROJECT (g_in, su (LEFT), LOCAL);
      ss := SUM (ss_, ss_.v);
      RETURN ss;
    END;//sum_square
		
		sum_abs (DATASET(Layout_Part) g_in) := FUNCTION
      Elem := {REAL8 v};  //short-cut record def
      Elem su(Layout_Part xrec) := TRANSFORM //hadamard product
        SELF.v :=  sumabs(xrec.part_rows*xrec.part_cols, xrec.mat_part);
      END;
      ss_ := PROJECT (g_in, su (LEFT), LOCAL);
      ss := SUM (ss_, ss_.v);
      RETURN ss;
    END;//sum_abs
    
		//l-bfgs algorithm
		lbfgs (DATASET(minfRec) min_in) := FUNCTION
			itr_n := MAX (min_in, min_in.update_itr);
      k := IF (itr_n >corrections, corrections, itr_n); // k is the number of previous step vectors which are already stored
			// q LOOP
			q_step (DATASET(minfRec) q_inp, unsigned4 q_c) := FUNCTION
				q_itr := itr_n- q_c + 1;
				q := q_inp(no=1);//this is the q vector
				s_tmp := min_in (no = 3 AND update_itr = q_itr);
				y_tmp := min_in (no = 4 AND update_itr = q_itr);
				//calculate al : al(i) = ro(i)*s(:,i)'*q(:,i+1);
				simple := {REAL8 v};
				simple al_tran (minfRec q_in, minfRec s_in) := TRANSFORM
				  cells := q_in.part_rows * q_in.part_cols;
					SELF.v := (1/s_in.sty) * sump(cells, q_in.mat_part, s_in.mat_part);
				END;
				
				al_ := JOIN (q, s_tmp, LEFT.partition_id = RIGHT.partition_id, al_tran (LEFT, RIGHT), LOCAL);
				al := SUM (al_, al_.v);
				// calculate new q vector : q(:,i) = q(:,i+1)-al(i)*y(:,i);
				minfRec new_q_tran (minfRec le, minfRec ri):= TRANSFORM
					cells := le.part_rows * le.part_cols;
			    SELF.mat_part := PBblas.BLAS.daxpy(cells, -1 * al, le.mat_part, 1, ri.mat_part, 1);
					SELF := ri;//by assigning q part (right hand side) to SELF we make sure we are keeping funevals and other information
				END;
				new_q := JOIN (y_tmp, q_inp, RIGHT.no =1 AND (LEFT.partition_id = RIGHT.partition_id), new_q_tran (LEFT, RIGHT), LOCAL);
				//normalize al to new_q and return the result
				minfRec norm_al (minfRec le) := TRANSFORM
					SELF.mat_part := [al];
					SELF.no := q_c + 1;
					SELF := le;
				END;
				al_norm_ := NORMALIZE(new_q(no=1), 1,norm_al(LEFT));
				al_norm := ASSERT(al_norm_, node_id = Thorlib.node() and node_id=(partition_id-1), 'al is not well distributed in the lbfgs function', FAIL);
        RETURN new_q + al_norm;
      END; //END q_step
			minfRec steep_tran (minfRec le) := TRANSFORM
				cells := le.part_rows * le.part_cols;
				SELF.mat_part := PBBlas.BLAS.dscal(cells, -1, le.mat_part, 1);//multiply by -1
				SELF := le;
			END;
			topass_q := PROJECT (min_in(no=1), steep_tran(LEFT),LOCAL);//contains funevals and cost_value, h information from the previous iteration in minfunc function. So basically the h field (hdiag) belongs to what has been calculated in the previous iteratio and can be used here in topass_r
			q_result := LOOP(topass_q, COUNTER <= k, q_step(ROWS(LEFT),COUNTER));
			// r loop
			r_step (DATASET(minfRec) r_inp, unsigned4 r_c) := FUNCTION
				r_itr := r_c;
				s_tmp := min_in (no = 3 AND update_itr = r_itr);
				y_tmp := min_in (no = 4 AND update_itr = r_itr);
				//calculate be be(i) = ro(i)*y(:,i)'*r(:,i);
				simple := {REAL8 v};
				simple be_tran (minfRec r_in, minfRec y_in) := TRANSFORM
				  cells := r_in.part_rows * r_in.part_cols;
					SELF.v := (1/y_in.sty) * sump(cells, r_in.mat_part, y_in.mat_part);
				END;
				be_ := JOIN (r_inp, y_tmp, LEFT.partition_id = right.partition_id, be_tran (LEFT, RIGHT), LOCAL);
				be := SUM (be_, be_.v);
				// calculate (al (i) - be ) * s (i)
				minfRec s_tran (minfRec s_in, minfRec al_in) := TRANSFORM
					cells := s_in.part_rows * s_in.part_cols;
					SELF.mat_part := PBBlas.BLAS.dscal(cells, (al_in.mat_part[1] - be), s_in.mat_part, 1);
					SELF := s_in;
				END;
				al_be_s := JOIN (s_tmp, q_result (no = (r_itr + 1)) , LEFT.partition_id = RIGHT. partition_id, s_tran (LEFT, RIGHT), LOCAL);
				// calculate new_r := r + al_be_s
				minfRec new_r_tran (minfRec le, minfRec ri):= TRANSFORM
					cells := le.part_rows * le.part_cols;
			    SELF.mat_part := PBblas.BLAS.daxpy(cells, 1.0, le.mat_part, 1, ri.mat_part, 1);
					SELF := ri;//again here by assigning the r part (RIGHT) we are making sure we are keeping all the field initially comming from g (min_in (no=1)) such as funevals field and etc.
				END;
				new_r := JOIN (al_be_s, r_inp, LEFT.partition_id = RIGHT.partition_id, new_r_tran (LEFT, RIGHT), LOCAL);
				RETURN new_r; 
			END;// END r_step
			minfRec r_pass_tran (minfRec le) := TRANSFORM
				cells := le.part_rows * le.part_cols;
				SELF.mat_part := PBBlas.BLAS.dscal(cells, le.h, le.mat_part, 1);
				SELF := le;
			END;
			//r(:,1) = Hdiag*q(:,1);
			topass_r := PROJECT (q_result (no=1), r_pass_tran (LEFT), LOCAL);
			final_d := LOOP(topass_r, COUNTER <= k, r_step(ROWS(LEFT),COUNTER));
		  RETURN final_d;
    END; // END lbfgs
		lbfgs2_backup (DATASET(minfRec) min_in) := FUNCTION
			itr_n := MAX (min_in, min_in.update_itr);
      k := IF (itr_n >corrections, corrections, itr_n); // k is the number of previous step vectors which are already stored
			// q LOOP
			q_step (DATASET(minfRec) q_inp, unsigned4 q_c) := FUNCTION
				q_itr := itr_n- q_c + 1;
				q := q_inp(no=1);//this is the q vector
				s_tmp := min_in (no = 3 AND update_itr = q_itr);
				y_tmp := min_in (no = 4 AND update_itr = q_itr);
				//calculate al : al(i) = ro(i)*s(:,i)'*q(:,i+1);
				simple := {REAL8 v};
				simple al_tran (minfRec q_in, minfRec s_in) := TRANSFORM
				  cells := q_in.part_rows * q_in.part_cols;
					SELF.v := (1/s_in.sty) * sump(cells, q_in.mat_part, s_in.mat_part);
				END;
				
				al_ := JOIN (q, s_tmp, LEFT.partition_id = RIGHT.partition_id, al_tran (LEFT, RIGHT), LOCAL);
				al := SUM (al_, al_.v);
				// calculate new q vector : q(:,i) = q(:,i+1)-al(i)*y(:,i);
				minfRec new_q_tran (minfRec le, minfRec ri):= TRANSFORM
					cells := le.part_rows * le.part_cols;
			    SELF.mat_part := PBblas.BLAS.daxpy(cells, -1 * al, le.mat_part, 1, ri.mat_part, 1);
					SELF := ri;//by assigning q part (right hand side) to SELF we make sure we are keeping funevals and other information
				END;
				new_q := JOIN (y_tmp, q, LEFT.partition_id = RIGHT.partition_id, new_q_tran (LEFT, RIGHT), LOCAL);
				//normalize al to new_q and return the result
				minfRec norm_al (minfRec le) := TRANSFORM
					SELF.mat_part := [al];
					SELF.no := q_itr + 1;// 1 is added because the final q_itr will be 1 and we have already reserved no=1 for the q vector itself
					SELF := le;
				END;
				al_norm_ := NORMALIZE(new_q(no=1), 1,norm_al(LEFT));
				al_norm := ASSERT(al_norm_, node_id = Thorlib.node() and node_id=(partition_id-1), 'al is not well distributed in the lbfgs function', FAIL);
        RETURN new_q + al_norm + q_inp (no != 1);
      END; //END q_step
			minfRec steep_tran (minfRec le) := TRANSFORM
				cells := le.part_rows * le.part_cols;
				SELF.mat_part := PBBlas.BLAS.dscal(cells, -1, le.mat_part, 1);//multiply by -1
				SELF := le;
			END;
			topass_q := PROJECT (min_in(no=1), steep_tran(LEFT),LOCAL);//contains funevals and cost_value, h information from the previous iteration in minfunc function. So basically the h field (hdiag) belongs to what has been calculated in the previous iteratio and can be used here in topass_r
			q_result := LOOP(topass_q, COUNTER <= k, q_step(ROWS(LEFT),COUNTER));
			// r loop
			r_step (DATASET(minfRec) r_inp, unsigned4 r_c) := FUNCTION
				r_itr := r_c;
				s_tmp := min_in (no = 3 AND update_itr = r_itr);
				y_tmp := min_in (no = 4 AND update_itr = r_itr);
				//calculate be be(i) = ro(i)*y(:,i)'*r(:,i);
				simple := {REAL8 v};
				simple be_tran (minfRec r_in, minfRec y_in) := TRANSFORM
				  cells := r_in.part_rows * r_in.part_cols;
					SELF.v := (1/y_in.sty) * sump(cells, r_in.mat_part, y_in.mat_part);
				END;
				be_ := JOIN (r_inp, y_tmp, LEFT.partition_id = right.partition_id, be_tran (LEFT, RIGHT), LOCAL);
				be := SUM (be_, be_.v);
				// calculate (al (i) - be ) * s (i)
				minfRec s_tran (minfRec s_in, minfRec al_in) := TRANSFORM
					cells := s_in.part_rows * s_in.part_cols;
					SELF.mat_part := PBBlas.BLAS.dscal(cells, (al_in.mat_part[1] - be), s_in.mat_part, 1);
					SELF := s_in;
				END;
				al_be_s := JOIN (s_tmp, q_result (no = (r_itr + 1)) , LEFT.partition_id = RIGHT. partition_id, s_tran (LEFT, RIGHT), LOCAL);
				// calculate new_r := r + al_be_s
				minfRec new_r_tran (minfRec le, minfRec ri):= TRANSFORM
					cells := le.part_rows * le.part_cols;
			    SELF.mat_part := PBblas.BLAS.daxpy(cells, 1.0, le.mat_part, 1, ri.mat_part, 1);
					SELF := ri;//again here by assigning the r part (RIGHT) we are making sure we are keeping all the field initially comming from g (min_in (no=1)) such as funevals field and etc.
				END;
				new_r := JOIN (al_be_s, r_inp, LEFT.partition_id = RIGHT.partition_id, new_r_tran (LEFT, RIGHT), LOCAL);
				RETURN new_r; 
			END;// END r_step
			minfRec r_pass_tran (minfRec le) := TRANSFORM
				cells := le.part_rows * le.part_cols;
				SELF.mat_part := PBBlas.BLAS.dscal(cells, le.h, le.mat_part, 1);
				SELF := le;
			END;
			//r(:,1) = Hdiag*q(:,1);
			topass_r := PROJECT (q_result (no=1), r_pass_tran (LEFT), LOCAL);
			final_d := LOOP(topass_r, COUNTER <= k, r_step(ROWS(LEFT),COUNTER));
		  //RETURN q_result;
			// q1 := q_step(topass_q,1);
			r1 := r_step(topass_r,1);
			r2 := r_step(r1,2);
			RETURN final_d;
    END; // END lbfgs2_backup
		
		
		lbfgs2 (DATASET(minfRec) min_in, UNSIGNED rec_c) := FUNCTION
			//itr_n := MAX (min_in, min_in.update_itr);
			itr_n := rec_c;
      k := IF (itr_n >corrections, corrections, itr_n); // k is the number of previous step vectors which are already stored
			//k := itr_n;
			// q LOOP
			q_step (DATASET(minfRec) q_inp, unsigned4 q_c) := FUNCTION
				q_itr := itr_n- q_c + 1;
				q := q_inp(no=1);//this is the q vector
				s_tmp := min_in (no = 3 AND update_itr = q_itr);
				y_tmp := min_in (no = 4 AND update_itr = q_itr);
				//calculate al : al(i) = ro(i)*s(:,i)'*q(:,i+1);
				simple := {REAL8 v};
				simple al_tran (minfRec q_in, minfRec s_in) := TRANSFORM
				  cells := q_in.part_rows * q_in.part_cols;
					SELF.v := (1/s_in.sty) * sump(cells, q_in.mat_part, s_in.mat_part);
				END;
				
				al_ := JOIN (q, s_tmp, LEFT.partition_id = RIGHT.partition_id, al_tran (LEFT, RIGHT), LOCAL);
				al := SUM (al_, al_.v);
				// calculate new q vector : q(:,i) = q(:,i+1)-al(i)*y(:,i);
				minfRec new_q_tran (minfRec le, minfRec ri):= TRANSFORM
					cells := le.part_rows * le.part_cols;
			    SELF.mat_part := PBblas.BLAS.daxpy(cells, -1 * al, le.mat_part, 1, ri.mat_part, 1);
					SELF := ri;//by assigning q part (right hand side) to SELF we make sure we are keeping funevals and other information
				END;
				new_q := JOIN (y_tmp, q, LEFT.partition_id = RIGHT.partition_id, new_q_tran (LEFT, RIGHT), LOCAL);
				//normalize al to new_q and return the result
				minfRec norm_al (minfRec le) := TRANSFORM
					SELF.mat_part := [al];
					SELF.no := q_itr + 1;// 1 is added because the final q_itr will be 1 and we have already reserved no=1 for the q vector itself
					SELF := le;
				END;
				al_norm_ := NORMALIZE(new_q(no=1), 1,norm_al(LEFT));
				al_norm := ASSERT(al_norm_, node_id = Thorlib.node() and node_id=(partition_id-1), 'al is not well distributed in the lbfgs function', FAIL);
        RETURN new_q + al_norm + q_inp (no != 1);
      END; //END q_step
			minfRec steep_tran (minfRec le) := TRANSFORM
				cells := le.part_rows * le.part_cols;
				SELF.mat_part := PBBlas.BLAS.dscal(cells, -1, le.mat_part, 1);//multiply by -1
				SELF := le;
			END;
			topass_q := PROJECT (min_in(no=1), steep_tran(LEFT),LOCAL);//contains funevals and cost_value, h information from the previous iteration in minfunc function. So basically the h field (hdiag) belongs to what has been calculated in the previous iteratio and can be used here in topass_r
			//q_result := LOOP(topass_q, COUNTER <= k, q_step(ROWS(LEFT),COUNTER)); orig
			q_result := LOOP(topass_q, k, q_step(ROWS(LEFT),COUNTER));
			// r loop
			r_step (DATASET(minfRec) r_inp, unsigned4 r_c) := FUNCTION
				r_itr := r_c + (itr_n - k);
				s_tmp := min_in (no = 3 AND update_itr = r_itr);
				y_tmp := min_in (no = 4 AND update_itr = r_itr);
				//calculate be be(i) = ro(i)*y(:,i)'*r(:,i);
				simple := {REAL8 v};
				simple be_tran (minfRec r_in, minfRec y_in) := TRANSFORM
				  cells := r_in.part_rows * r_in.part_cols;
					SELF.v := (1/y_in.sty) * sump(cells, r_in.mat_part, y_in.mat_part);
				END;
				be_ := JOIN (r_inp, y_tmp, LEFT.partition_id = right.partition_id, be_tran (LEFT, RIGHT), LOCAL);
				be := SUM (be_, be_.v);
				// calculate (al (i) - be ) * s (i)
				minfRec s_tran (minfRec s_in, minfRec al_in) := TRANSFORM
					cells := s_in.part_rows * s_in.part_cols;
					SELF.mat_part := PBBlas.BLAS.dscal(cells, (al_in.mat_part[1] - be), s_in.mat_part, 1);
					SELF := s_in;
				END;
				al_be_s := JOIN (s_tmp, q_result (no = (r_itr + 1)) , LEFT.partition_id = RIGHT. partition_id, s_tran (LEFT, RIGHT), LOCAL);
				// calculate new_r := r + al_be_s
				minfRec new_r_tran (minfRec le, minfRec ri):= TRANSFORM
					cells := le.part_rows * le.part_cols;
			    SELF.mat_part := PBblas.BLAS.daxpy(cells, 1.0, le.mat_part, 1, ri.mat_part, 1);
					SELF := ri;//again here by assigning the r part (RIGHT) we are making sure we are keeping all the field initially comming from g (min_in (no=1)) such as funevals field and etc.
				END;
				new_r := JOIN (al_be_s, r_inp, LEFT.partition_id = RIGHT.partition_id, new_r_tran (LEFT, RIGHT), LOCAL);
				RETURN new_r; 
			END;// END r_step
			minfRec r_pass_tran (minfRec le) := TRANSFORM
				cells := le.part_rows * le.part_cols;
				SELF.mat_part := PBBlas.BLAS.dscal(cells, le.h, le.mat_part, 1);
				SELF := le;
			END;
			//r(:,1) = Hdiag*q(:,1);
			topass_r := PROJECT (q_result (no=1), r_pass_tran (LEFT), LOCAL);
			//final_d := LOOP(topass_r, COUNTER <= k, r_step(ROWS(LEFT),COUNTER)); orig
			final_d := LOOP(topass_r, k, r_step(ROWS(LEFT),COUNTER));
		  //RETURN q_result;
			// q1 := q_step(topass_q,1);
			r1 := r_step(topass_r,1);
			r2 := r_step(r1,2);
			RETURN final_d;
    END; // END lbfgs2
		
		lbfgs4 (DATASET(minfRec) min_in, UNSIGNED rec_c) := FUNCTION
			//itr_n := MAX (min_in, min_in.update_itr);
			itr_n := rec_c;
      k := IF (itr_n >corrections, corrections, itr_n); // k is the number of previous step vectors which are already stored
			//k := itr_n;
			// q LOOP
			q_step (DATASET(minfRec) q_inp, unsigned4 q_c) := FUNCTION
				q_itr := itr_n- q_c + 1;
				q := q_inp;//this is the q vector
				//calculate al : al(i) = ro(i)*s(:,i)'*q(:,i+1);
				simple := {REAL8 v};
				simple al_tran (minfRec q_in, minfRec s_in) := TRANSFORM
				  cells := q_in.part_rows * q_in.part_cols;
					SELF.v := (1/s_in.sty) * sump(cells, q_in.mat_part, s_in.mat_part);
				END;
				
				al_ := JOIN (q, min_in, RIGHT.no = 3 AND (RIGHT.update_itr=LEFT.update_itr) AND (LEFT.partition_id = RIGHT.partition_id), al_tran (LEFT, RIGHT), LOCAL);
				al := SUM (al_, al_.v);
				// calculate new q vector : q(:,i) = q(:,i+1)-al(i)*y(:,i);
				minfRec new_q_tran (minfRec le, minfRec ri):= TRANSFORM
					cells := le.part_rows * le.part_cols;
			    SELF.mat_part := PBblas.BLAS.daxpy(cells, -1 * al, le.mat_part, 1, ri.mat_part, 1);
					SELF.update_itr :=  IF (q_c =1 , IF (ri.update_itr < corrections , corrections - ri.update_itr +1 , 1) , ri.update_itr + 1 ) ;
					SELF := ri;//by assigning q part (right hand side) to SELF we make sure we are keeping funevals and other information
				END;
				new_q := JOIN (min_in, q, RIGHT.no = 4 AND (LEFT.update_itr=RIGHT.update_itr) AND (LEFT.partition_id = RIGHT.partition_id), new_q_tran (LEFT, RIGHT), LOCAL);
				//normalize al to new_q and return the result
				minfRec norm_al (minfRec le) := TRANSFORM
					SELF.mat_part := [al];
					SELF.no := le.update_itr+ 1;// 1 is added because the final q_itr will be 1 and we have already reserved no=1 for the q vector itself
					SELF := le;
				END;
				al_norm_ := NORMALIZE(new_q(no=1), 1,norm_al(LEFT));
				al_norm := ASSERT(al_norm_, node_id = Thorlib.node() and node_id=(partition_id-1), 'al is not well distributed in the lbfgs function', FAIL);
        RETURN new_q + al_norm;
      END; //END q_step
			minfRec steep_tran (minfRec le) := TRANSFORM
				cells := le.part_rows * le.part_cols;
				SELF.mat_part := PBBlas.BLAS.dscal(cells, -1, le.mat_part, 1);//multiply by -1
				SELF := le;
			END;
			topass_q := PROJECT (min_in(no=1), steep_tran(LEFT),LOCAL);//contains funevals and cost_value, h information from the previous iteration in minfunc function. So basically the h field (hdiag) belongs to what has been calculated in the previous iteratio and can be used here in topass_r
			//q_result := LOOP(topass_q, COUNTER <= k, q_step(ROWS(LEFT),COUNTER)); orig
			q_result := LOOP(topass_q, (LEFT.no =1) AND (LEFT.update_itr - corrections + 1 > 0) , q_step(ROWS(LEFT),COUNTER));
			// r loop
			r_step (DATASET(minfRec) r_inp, unsigned4 r_c) := FUNCTION
				r_itr := r_c + (itr_n - k);
				s_tmp := min_in (no = 3 AND update_itr = r_itr);
				y_tmp := min_in (no = 4 AND update_itr = r_itr);
				//calculate be be(i) = ro(i)*y(:,i)'*r(:,i);
				simple := {REAL8 v};
				simple be_tran (minfRec r_in, minfRec y_in) := TRANSFORM
				  cells := r_in.part_rows * r_in.part_cols;
					SELF.v := (1/y_in.sty) * sump(cells, r_in.mat_part, y_in.mat_part);
				END;
				be_ := JOIN (r_inp, y_tmp, LEFT.partition_id = right.partition_id, be_tran (LEFT, RIGHT), LOCAL);
				be := SUM (be_, be_.v);
				// calculate (al (i) - be ) * s (i)
				minfRec s_tran (minfRec s_in, minfRec al_in) := TRANSFORM
					cells := s_in.part_rows * s_in.part_cols;
					SELF.mat_part := PBBlas.BLAS.dscal(cells, (al_in.mat_part[1] - be), s_in.mat_part, 1);
					SELF := s_in;
				END;
				al_be_s := JOIN (s_tmp, q_result (no = (r_itr + 1)) , LEFT.partition_id = RIGHT. partition_id, s_tran (LEFT, RIGHT), LOCAL);
				// calculate new_r := r + al_be_s
				minfRec new_r_tran (minfRec le, minfRec ri):= TRANSFORM
					cells := le.part_rows * le.part_cols;
			    SELF.mat_part := PBblas.BLAS.daxpy(cells, 1.0, le.mat_part, 1, ri.mat_part, 1);
					SELF := ri;//again here by assigning the r part (RIGHT) we are making sure we are keeping all the field initially comming from g (min_in (no=1)) such as funevals field and etc.
				END;
				new_r := JOIN (al_be_s, r_inp, LEFT.partition_id = RIGHT.partition_id, new_r_tran (LEFT, RIGHT), LOCAL);
				RETURN new_r; 
			END;// END r_step
			minfRec r_pass_tran (minfRec le) := TRANSFORM
				cells := le.part_rows * le.part_cols;
				SELF.mat_part := PBBlas.BLAS.dscal(cells, le.h, le.mat_part, 1);
				SELF := le;
			END;
			//r(:,1) = Hdiag*q(:,1);
			topass_r := PROJECT (q_result (no=1), r_pass_tran (LEFT), LOCAL);
			//final_d := LOOP(topass_r, COUNTER <= k, r_step(ROWS(LEFT),COUNTER)); orig
			final_d := LOOP(topass_r, k, r_step(ROWS(LEFT),COUNTER));
		  //RETURN q_result;
			// q1 := q_step(topass_q,1);
			r1 := r_step(topass_r,1);
			r2 := r_step(r1,2);
			RETURN q_result;
    END; // END lbfgs4
		
		
			lbfgs3 (DATASET(minfRec) min_in) := FUNCTION
			itr_n := MAX (min_in, min_in.update_itr);
      k := IF (itr_n >corrections, corrections, itr_n); // k is the number of previous step vectors which are already stored
			// q LOOP
			q_step (DATASET(minfRec) q_inp, unsigned4 q_c) := FUNCTION
				q_itr := itr_n- q_c + 1;
				q := q_inp(no=1);//this is the q vector
				s_tmp := min_in (no = 3 AND update_itr = q_itr);
				y_tmp := min_in (no = 4 AND update_itr = q_itr);
				//calculate al : al(i) = ro(i)*s(:,i)'*q(:,i+1);
				simple := {REAL8 v};
				simple al_tran (minfRec q_in, minfRec s_in) := TRANSFORM
				  cells := q_in.part_rows * q_in.part_cols;
					SELF.v := (1/s_in.sty) * sump(cells, q_in.mat_part, s_in.mat_part);
				END;
				
				al_ := JOIN (q, s_tmp, LEFT.partition_id = RIGHT.partition_id, al_tran (LEFT, RIGHT), LOCAL);
				al := SUM (al_, al_.v);
				// calculate new q vector : q(:,i) = q(:,i+1)-al(i)*y(:,i);
				minfRec new_q_tran (minfRec le, minfRec ri):= TRANSFORM
					cells := le.part_rows * le.part_cols;
			    SELF.mat_part := PBblas.BLAS.daxpy(cells, -1 * al, le.mat_part, 1, ri.mat_part, 1);
					SELF := ri;//by assigning q part (right hand side) to SELF we make sure we are keeping funevals and other information
				END;
				new_q := JOIN (y_tmp, q, LEFT.partition_id = RIGHT.partition_id, new_q_tran (LEFT, RIGHT), LOCAL);
				//normalize al to new_q and return the result
				minfRec norm_al (minfRec le) := TRANSFORM
					SELF.mat_part := [al];
					SELF.no := q_itr + 1;// 1 is added because the final q_itr will be 1 and we have already reserved no=1 for the q vector itself
					SELF := le;
				END;
				al_norm_ := NORMALIZE(new_q(no=1), 1,norm_al(LEFT));
				al_norm := ASSERT(al_norm_, node_id = Thorlib.node() and node_id=(partition_id-1), 'al is not well distributed in the lbfgs function', FAIL);
        RETURN new_q + al_norm + q_inp (no != 1);
      END; //END q_step
			minfRec steep_tran (minfRec le) := TRANSFORM
				cells := le.part_rows * le.part_cols;
				SELF.mat_part := PBBlas.BLAS.dscal(cells, -1, le.mat_part, 1);//multiply by -1
				SELF := le;
			END;
			topass_q := PROJECT (min_in(no=1), steep_tran(LEFT),LOCAL);//contains funevals and cost_value, h information from the previous iteration in minfunc function. So basically the h field (hdiag) belongs to what has been calculated in the previous iteratio and can be used here in topass_r
			q_result := LOOP(topass_q, COUNTER <= k, q_step(ROWS(LEFT),COUNTER));
			// r loop
			r_step (DATASET(minfRec) r_inp, unsigned4 r_c) := FUNCTION
				r_itr := r_c;
				s_tmp := min_in (no = 3 AND update_itr = r_itr);
				y_tmp := min_in (no = 4 AND update_itr = r_itr);
				//calculate be be(i) = ro(i)*y(:,i)'*r(:,i);
				simple := {REAL8 v};
				simple be_tran (minfRec r_in, minfRec y_in) := TRANSFORM
				  cells := r_in.part_rows * r_in.part_cols;
					SELF.v := (1/y_in.sty) * sump(cells, r_in.mat_part, y_in.mat_part);
				END;
				be_ := JOIN (r_inp, y_tmp, LEFT.partition_id = right.partition_id, be_tran (LEFT, RIGHT), LOCAL);
				be := SUM (be_, be_.v);
				// calculate (al (i) - be ) * s (i)
				minfRec s_tran (minfRec s_in, minfRec al_in) := TRANSFORM
					cells := s_in.part_rows * s_in.part_cols;
					SELF.mat_part := PBBlas.BLAS.dscal(cells, (al_in.mat_part[1] - be), s_in.mat_part, 1);
					SELF := s_in;
				END;
				al_be_s := JOIN (s_tmp, q_result (no = (r_itr + 1)) , LEFT.partition_id = RIGHT. partition_id, s_tran (LEFT, RIGHT), LOCAL);
				// calculate new_r := r + al_be_s
				minfRec new_r_tran (minfRec le, minfRec ri):= TRANSFORM
					cells := le.part_rows * le.part_cols;
			    SELF.mat_part := PBblas.BLAS.daxpy(cells, 1.0, le.mat_part, 1, ri.mat_part, 1);
					SELF := ri;//again here by assigning the r part (RIGHT) we are making sure we are keeping all the field initially comming from g (min_in (no=1)) such as funevals field and etc.
				END;
				new_r := JOIN (al_be_s, r_inp, LEFT.partition_id = RIGHT.partition_id, new_r_tran (LEFT, RIGHT), LOCAL);
				RETURN new_r; 
			END;// END r_step
			minfRec r_pass_tran (minfRec le) := TRANSFORM
				cells := le.part_rows * le.part_cols;
				SELF.mat_part := PBBlas.BLAS.dscal(cells, le.h, le.mat_part, 1);
				SELF := le;
			END;
			//r(:,1) = Hdiag*q(:,1);
			topass_r := PROJECT (q_result (no=1), r_pass_tran (LEFT), LOCAL);
			final_d := LOOP(topass_r, COUNTER <= k, r_step(ROWS(LEFT),COUNTER));
		  //RETURN q_result;
			// q1 := q_step(topass_q,1);
			r1 := r_step(topass_r,1);
			r2 := r_step(r1,2);
			RETURN q_result(no=1);
    END; // END lbfgs3

		//check optimality 
    //if sum(abs(g)) <= tolFun
    optimality_check (DATASET(Layout_Part) g_in) := FUNCTION
      ss := sum_abs(g_in);
      RETURN ss<tolFun;
    END;//END optimality_check
    IsLegal (DATASET(costgrad_record) inp) := FUNCTION
      RETURN TRUE;
    END;//END IsLegal ???
		   // Evaluate Initial Point
    CostGrad := CostFunc(x0,CostFunc_params,TrainData, TrainLabel);
    funEvals := 1;
		Hdiag := 1;
		wolfe_max_itr := 25;
	 min_step_firstitr := FUNCTION
		//calculate d
		//Steepest_Descent
		costgrad_record steep_tran (costgrad_record le) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.mat_part := PBBlas.BLAS.dscal(cells, -1, le.mat_part, 1);//multiply by -1
			SELF := le;
		END;
		d := PROJECT (CostGrad, steep_tran (LEFT) , LOCAL);
		//check whether d is legal, if not return
		dlegalstep := IsLegal (d);
		// Directional Derivative : gtd = g'*d;
		//Since d = -g -> gtd = -sum (g.^2)
		Elem := {REAL8 v};
		Elem g2_tran (costgrad_record le) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.v := sum_sq(cells, le.mat_part);
		END;
		gtd_ := PROJECT (CostGrad, g2_tran (LEFT), LOCAL);
		gtd := -1*SUM (gtd_, gtd_.v);
		//Check that progress can be made along direction : if gtd > -tolX then break!
		gtdprogress := gtd > -1*tolX;
		// Select Initial Guess If coun =1 then t = min(1,1/sum(abs(g)));
		Elem gabs_tran (costgrad_record le) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.v := sumabs(cells, le.mat_part);
		END;
		sum_abs_g0_ := PROJECT (CostGrad, gabs_tran (LEFT), LOCAL);
		sum_abs_g0 := SUM (sum_abs_g0_, sum_abs_g0_.v);
		t_init := MIN ([1, 1/(sum_abs_g0)]);
		// Find Point satisfying Wolfe
		w := WolfeLineSearch4_4_2(1, x0,CostFunc_params,TrainData, TrainLabel,CostFunc, t_init, d, CostGrad,gtd, 0.0001, 0.9, wolfe_max_itr, 0.000000001);
		//calculate new oint
		//calculate td
		minfRec td_tran (w le, d ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := PBBlas.BLAS.dscal(cells, le.prev_t, ri.mat_part, 1);
			SELF.no := 3;
			SELF.h := -1; //not calculated yet
			SELF.min_funEval := funEvals + le.wolfe_funevals;
			SELF.break_cond := -1;
			SELF.sty := -1;// not calculated yet
			SELF.update_itr := -1; // not calculated yet
			SELF.cost_value := le.cost_value;
			SELF.itr_counter := 1;
			SELF := ri;
		END;
		td := JOIN (w, d, LEFT.partition_id = RIGHT.partition_id, td_tran (LEFT, RIGHT), LOCAL);
		//calculate x_new = x0 + td
		minfRec x_new_tran (Layout_part le, minfRec ri) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.mat_part := PBblas.BLAS.daxpy(cells, 1.0, le.mat_part, 1, ri.mat_part, 1);
			SELF.no := 2;
			SELF.h := Hdiag; //not calculated yet
			SELF.min_funEval := ri.min_funEval;
			SELF.break_cond := -1;
			SELF.sty := -1;// NA
			SELF.update_itr := -1; // not calculated yet
			SELF.cost_value := ri.cost_value;
			SELF.itr_counter := 1;
			SELF := le;
		END;
		x_new := JOIN (x0, td, LEFT.partition_id = RIGHT.partition_id, x_new_tran(LEFT, RIGHT), LOCAL);
		//s_s = t*d
		s_s := td;
		//y_y := g_new - g_old
		minfRec y_y_tran (costgrad_record le, w ri) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.mat_part := PBblas.BLAS.daxpy(cells, -1.0, le.mat_part, 1, ri.mat_part, 1);
			SELF.no := 4;
			SELF.h := -1; //not calculated yer
			SELF.min_funEval := funEvals + ri.wolfe_funevals;
			SELF.break_cond := -1;
			SELF.sty := -1;// not calculated yet
			SELF.update_itr := -1; // not calculated yet
			SELF.cost_value := ri.cost_value;
			SELF.itr_counter := 1;
			SELF := le;
		END;
		y_y := JOIN (CostGrad, w, LEFT.partition_id = RIGHT.partition_id, y_y_tran(LEFT, RIGHT), LOCAL);
		//y_s = y_y'*s_s
		Elem y_s_tran (y_y le, s_s ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.v:= sump(cells, le.mat_part, ri.mat_part);
		END;
		y_s_ := JOIN (y_y, s_s, LEFT.partition_id = RIGHT.partition_id, y_s_tran (LEFT, RIGHT), LOCAL);
		y_s := SUM (y_s_, y_s_.v);
		//calculated y_y^2
		Elem yy2_tran (Layout_Part le) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.v := sum_sq(cells, le.mat_part);
		END;
		y_y_2_ := PROJECT (y_y, yy2_tran(LEFT), LOCAL);
		y_y_2 := SUM (y_y_2_, y_y_2_.v);
		hdiag_updated := y_s/ y_y_2;
		update_cond := y_s>10^(-10);
		// calculate sum(abs(g_new))
		Elem abs_g_tran (w le) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.v:= sumabs(cells, le.mat_part);
		END;
		sum_abs_g_new_ := PROJECT (w, abs_g_tran (LEFT), LOCAL);
		sum_abs_g_new := SUM (sum_abs_g_new_, sum_abs_g_new_.v);
		//calculate sum(abs(td))
		Elem abs_td_tran (td le) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.v:= sumabs(cells, le.mat_part);
		END;
		sum_abs_td_ := PROJECT (td, abs_td_tran(LEFT), LOCAL);
		sum_abs_td := SUM (sum_abs_td_, sum_abs_td_.v);
		//Optimality Condition sum(abs(g)) <= tolFun
		optimality_cond := sum_abs_g_new <= tolFun;
		//Check for lack of progress sum(abs(t*d)) <= tolX
		lack_prog_cond := sum_abs_td <= tolX;
		// -1: no condition
		// 1 : when d is not legal
		// 2 : when progress along direction not allowed
		// 3 : optimality cond
		// 4 : lack of progress
		minfunc_cond := IF (optimality_cond, 3, IF (lack_prog_cond, 4, -1));
		//when d is not legal, no other calculation is done and we return the current g and x values
		g0_dnotlegal_return := PROJECT (Costgrad, TRANSFORM(minfrec, SELF.itr_counter := 1; SELF.sty := -1; SELF.update_itr := 0; SELF.h := Hdiag; SELF.break_cond := 1, SELF.no := 1; SELF.min_funEval:= funEvals; SELF:= LEFT ), LOCAL);
		x0_dnotlegal_return := PROJECT (x0,       TRANSFORM(minfrec, SELF.itr_counter := 1; SELF.sty := -1; SELF.update_itr := 0; SELF.h := Hdiag; SELF.break_cond := 1, SELF.no := 2; SELF.min_funEval:= funEvals; SELF.cost_value := -1; SELF:= LEFT ), LOCAL);//cost_value for x (when no =2 ) does not mean any thing, the correct cost) value is shown when no=1 (along with gradient value)
		dnotlegal_return := g0_dnotlegal_return + x0_dnotlegal_return;
		//After gtd is calculated check whether progress along the direction is possible, if it is not, break
		g0_noprogalong_return := PROJECT (Costgrad, TRANSFORM(minfrec, SELF.itr_counter := 1; SELF.sty := -1; SELF.update_itr := 0; SELF.h := Hdiag; SELF.break_cond := 2, SELF.no := 1; SELF.min_funEval:= funEvals; SELF:= LEFT ), LOCAL);
		x0_noprogalong_return := PROJECT (x0,       TRANSFORM(minfrec, SELF.itr_counter := 1; SELF.sty := -1; SELF.update_itr := 0; SELF.h := Hdiag; SELF.break_cond := 2, SELF.no := 2; SELF.min_funEval:= funEvals; SELF.cost_value := -1; SELF:= LEFT ), LOCAL);//cost_value for x (when no =2 ) does not mean any thing, the correct cost) value is shown when no=1 (along with gradient value)
		noprogalong_return := g0_noprogalong_return + x0_noprogalong_return;
		x_new_break := PROJECT (x_new, TRANSFORM (minfRec, SELF.itr_counter := 1; SELF.break_cond := minfunc_cond; SELF := LEFT) ,LOCAL); // if after calculating new point and calculating x_new and g_new one of the conditions is satisfied we return and no need to calculate s ad y and updated_hdiag fo rthe next itr
		g_new_break := PROJECT (w,TRANSFORM (minfRec, SELF.itr_counter := 1; SELF.min_funEval := funEvals + LEFT.wolfe_funevals; SELF.break_cond := minfunc_cond; SELF.sty := -1; SELF.update_itr := 0;  SELF.no :=1; SELF.h := Hdiag; SELF := LEFT) ,LOCAL);
		break_result := x_new_break + g_new_break;
		x_new_nextitr := PROJECT (x_new, TRANSFORM (minfRec, SELF.itr_counter := 1; SELF.update_itr := IF (update_cond,1,0); SELF.h := IF (update_cond, hdiag_updated, Hdiag); SELF.break_cond := minfunc_cond; SELF := LEFT) ,LOCAL);// IF (minfunc_cond=-1,hdiag_updated, -1) : if a breack condition is satisfied, then there is not going to be a next iteration, so no need to calculate updated hidiag
		g_new_nextitr := PROJECT (w,TRANSFORM (minfRec, SELF.itr_counter := 1; SELF.min_funEval := funEvals + LEFT.wolfe_funevals; SELF.break_cond := minfunc_cond; SELF.sty := -1; SELF.update_itr := IF (update_cond,1,0);  SELF.no :=1; SELF.h := IF (update_cond, hdiag_updated, Hdiag); SELF := LEFT) ,LOCAL);
		g_x_nextitr := x_new_nextitr + g_new_nextitr;
		s_s_corr := PROJECT (s_s,TRANSFORM (minfRec, SELF.itr_counter := 1; SELF.sty := y_s; SELF.h := hdiag_updated; SELF.update_itr := 1; SELF := LEFT) ,LOCAL);//update_itr:=1 becasue if we are adding s_s_corr to the loop data it means the condition has been accepted
		y_y_corr := PROJECT (y_y,TRANSFORM (minfRec, SELF.itr_counter := 1; SELF.sty := y_s; SELF.h := hdiag_updated; SELF.update_itr := 1; SELF := LEFT) ,LOCAL);//update_itr:=1 becasue if we are adding s_s_corr to the loop data it means the condition has been accepted
		// IF one of the break conditions is satisfied and minfunc!=-1 we need to return only x_new and g_new and no need to calculated Hdiag, s, y fo rthe next iteration
		//Conversly, if no condition is satisfied, it means we are going to the next iteration so we need to update Hdiag, s and y if and only if update_cond is satisfied, otherwise no update is done and we just return the current values to the next iteration
		TheReturn := IF (minfunc_cond = -1, IF (update_cond, g_x_nextitr + s_s_corr + y_y_corr, g_x_nextitr), break_result);
		Rresult := IF (dlegalstep, IF (gtdprogress, noprogalong_return, TheReturn) , dnotlegal_return);
		RETURN Rresult;
		//RETURN g_x_nextitr + s_s_corr + y_y_corr;
	 END;// END min_step_firstitr
	 
	 min_step_firstitr_ttest := FUNCTION
		//calculate d
		//Steepest_Descent
		costgrad_record steep_tran (costgrad_record le) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.mat_part := PBBlas.BLAS.dscal(cells, -1, le.mat_part, 1);//multiply by -1
			SELF := le;
		END;
		d := PROJECT (CostGrad, steep_tran (LEFT) , LOCAL);
		//check whether d is legal, if not return
		dlegalstep := IsLegal (d);
		// Directional Derivative : gtd = g'*d;
		//Since d = -g -> gtd = -sum (g.^2)
		Elem := {REAL8 v};
		Elem g2_tran (costgrad_record le) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.v := sum_sq(cells, le.mat_part);
		END;
		gtd_ := PROJECT (CostGrad, g2_tran (LEFT), LOCAL);
		gtd := -1*SUM (gtd_, gtd_.v);
		//Check that progress can be made along direction : if gtd > -tolX then break!
		gtdprogress := gtd > -1*tolX;
		// Select Initial Guess If coun =1 then t = min(1,1/sum(abs(g)));
		Elem gabs_tran (costgrad_record le) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.v := sumabs(cells, le.mat_part);
		END;
		sum_abs_g0_ := PROJECT (CostGrad, gabs_tran (LEFT), LOCAL);
		sum_abs_g0 := SUM (sum_abs_g0_, sum_abs_g0_.v);
		t_init := MIN ([1, 1/(sum_abs_g0)]);
		// Find Point satisfying Wolfe
		w := WolfeLineSearch4_4_2_ttest(1, x0,CostFunc_params,TrainData, TrainLabel,CostFunc, t_init, d, CostGrad,gtd, 0.0001, 0.9, wolfe_max_itr, 0.000000001);
		//calculate new oint
		//calculate td
		minfRec td_tran (w le, d ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := PBBlas.BLAS.dscal(cells, le.prev_t, ri.mat_part, 1);
			SELF.no := 3;
			SELF.h := -1; //not calculated yet
			SELF.min_funEval := funEvals + le.wolfe_funevals;
			SELF.break_cond := -1;
			SELF.sty := -1;// not calculated yet
			SELF.update_itr := -1; // not calculated yet
			SELF.cost_value := le.cost_value;
			SELF.itr_counter := 1;
			SELF := ri;
		END;
		td := JOIN (w, d, LEFT.partition_id = RIGHT.partition_id, td_tran (LEFT, RIGHT), LOCAL);
		//calculate x_new = x0 + td
		minfRec x_new_tran (Layout_part le, minfRec ri) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.mat_part := PBblas.BLAS.daxpy(cells, 1.0, le.mat_part, 1, ri.mat_part, 1);
			SELF.no := 2;
			SELF.h := Hdiag; //not calculated yet
			SELF.min_funEval := ri.min_funEval;
			SELF.break_cond := -1;
			SELF.sty := -1;// NA
			SELF.update_itr := -1; // not calculated yet
			SELF.cost_value := ri.cost_value;
			SELF.itr_counter := 1;
			SELF := le;
		END;
		x_new := JOIN (x0, td, LEFT.partition_id = RIGHT.partition_id, x_new_tran(LEFT, RIGHT), LOCAL);
		//s_s = t*d
		s_s := td;
		//y_y := g_new - g_old
		minfRec y_y_tran (costgrad_record le, w ri) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.mat_part := PBblas.BLAS.daxpy(cells, -1.0, le.mat_part, 1, ri.mat_part, 1);
			SELF.no := 4;
			SELF.h := -1; //not calculated yer
			SELF.min_funEval := funEvals + ri.wolfe_funevals;
			SELF.break_cond := -1;
			SELF.sty := -1;// not calculated yet
			SELF.update_itr := -1; // not calculated yet
			SELF.cost_value := ri.cost_value;
			SELF.itr_counter := 1;
			SELF := le;
		END;
		y_y := JOIN (CostGrad, w, LEFT.partition_id = RIGHT.partition_id, y_y_tran(LEFT, RIGHT), LOCAL);
		//y_s = y_y'*s_s
		Elem y_s_tran (y_y le, s_s ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.v:= sump(cells, le.mat_part, ri.mat_part);
		END;
		y_s_ := JOIN (y_y, s_s, LEFT.partition_id = RIGHT.partition_id, y_s_tran (LEFT, RIGHT), LOCAL);
		y_s := SUM (y_s_, y_s_.v);
		//calculated y_y^2
		Elem yy2_tran (Layout_Part le) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.v := sum_sq(cells, le.mat_part);
		END;
		y_y_2_ := PROJECT (y_y, yy2_tran(LEFT), LOCAL);
		y_y_2 := SUM (y_y_2_, y_y_2_.v);
		hdiag_updated := y_s/ y_y_2;
		update_cond := y_s>10^(-10);
		// calculate sum(abs(g_new))
		Elem abs_g_tran (w le) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.v:= sumabs(cells, le.mat_part);
		END;
		sum_abs_g_new_ := PROJECT (w, abs_g_tran (LEFT), LOCAL);
		sum_abs_g_new := SUM (sum_abs_g_new_, sum_abs_g_new_.v);
		//calculate sum(abs(td))
		Elem abs_td_tran (td le) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.v:= sumabs(cells, le.mat_part);
		END;
		sum_abs_td_ := PROJECT (td, abs_td_tran(LEFT), LOCAL);
		sum_abs_td := SUM (sum_abs_td_, sum_abs_td_.v);
		//Optimality Condition sum(abs(g)) <= tolFun
		optimality_cond := sum_abs_g_new <= tolFun;
		//Check for lack of progress sum(abs(t*d)) <= tolX
		lack_prog_cond := sum_abs_td <= tolX;
		// -1: no condition
		// 1 : when d is not legal
		// 2 : when progress along direction not allowed
		// 3 : optimality cond
		// 4 : lack of progress
		minfunc_cond := IF (optimality_cond, 3, IF (lack_prog_cond, 4, -1));
		//when d is not legal, no other calculation is done and we return the current g and x values
		g0_dnotlegal_return := PROJECT (Costgrad, TRANSFORM(minfrec, SELF.itr_counter := 1; SELF.sty := -1; SELF.update_itr := 0; SELF.h := Hdiag; SELF.break_cond := 1, SELF.no := 1; SELF.min_funEval:= funEvals; SELF:= LEFT ), LOCAL);
		x0_dnotlegal_return := PROJECT (x0,       TRANSFORM(minfrec, SELF.itr_counter := 1; SELF.sty := -1; SELF.update_itr := 0; SELF.h := Hdiag; SELF.break_cond := 1, SELF.no := 2; SELF.min_funEval:= funEvals; SELF.cost_value := -1; SELF:= LEFT ), LOCAL);//cost_value for x (when no =2 ) does not mean any thing, the correct cost) value is shown when no=1 (along with gradient value)
		dnotlegal_return := g0_dnotlegal_return + x0_dnotlegal_return;
		//After gtd is calculated check whether progress along the direction is possible, if it is not, break
		g0_noprogalong_return := PROJECT (Costgrad, TRANSFORM(minfrec, SELF.itr_counter := 1; SELF.sty := -1; SELF.update_itr := 0; SELF.h := Hdiag; SELF.break_cond := 2, SELF.no := 1; SELF.min_funEval:= funEvals; SELF:= LEFT ), LOCAL);
		x0_noprogalong_return := PROJECT (x0,       TRANSFORM(minfrec, SELF.itr_counter := 1; SELF.sty := -1; SELF.update_itr := 0; SELF.h := Hdiag; SELF.break_cond := 2, SELF.no := 2; SELF.min_funEval:= funEvals; SELF.cost_value := -1; SELF:= LEFT ), LOCAL);//cost_value for x (when no =2 ) does not mean any thing, the correct cost) value is shown when no=1 (along with gradient value)
		noprogalong_return := g0_noprogalong_return + x0_noprogalong_return;
		x_new_break := PROJECT (x_new, TRANSFORM (minfRec, SELF.itr_counter := 1; SELF.break_cond := minfunc_cond; SELF := LEFT) ,LOCAL); // if after calculating new point and calculating x_new and g_new one of the conditions is satisfied we return and no need to calculate s ad y and updated_hdiag fo rthe next itr
		g_new_break := PROJECT (w,TRANSFORM (minfRec, SELF.itr_counter := 1; SELF.min_funEval := funEvals + LEFT.wolfe_funevals; SELF.break_cond := minfunc_cond; SELF.sty := -1; SELF.update_itr := 0;  SELF.no :=1; SELF.h := Hdiag; SELF := LEFT) ,LOCAL);
		break_result := x_new_break + g_new_break;
		x_new_nextitr := PROJECT (x_new, TRANSFORM (minfRec, SELF.itr_counter := 1; SELF.update_itr := IF (update_cond,1,0); SELF.h := IF (update_cond, hdiag_updated, Hdiag); SELF.break_cond := minfunc_cond; SELF := LEFT) ,LOCAL);// IF (minfunc_cond=-1,hdiag_updated, -1) : if a breack condition is satisfied, then there is not going to be a next iteration, so no need to calculate updated hidiag
		g_new_nextitr := PROJECT (w,TRANSFORM (minfRec, SELF.itr_counter := 1; SELF.min_funEval := funEvals + LEFT.wolfe_funevals; SELF.break_cond := minfunc_cond; SELF.sty := -1; SELF.update_itr := IF (update_cond,1,0);  SELF.no :=1; SELF.h := IF (update_cond, hdiag_updated, Hdiag); SELF := LEFT) ,LOCAL);
		g_x_nextitr := x_new_nextitr + g_new_nextitr;
		s_s_corr := PROJECT (s_s,TRANSFORM (minfRec, SELF.itr_counter := 1; SELF.sty := y_s; SELF.h := hdiag_updated; SELF.update_itr := 1; SELF := LEFT) ,LOCAL);//update_itr:=1 becasue if we are adding s_s_corr to the loop data it means the condition has been accepted
		y_y_corr := PROJECT (y_y,TRANSFORM (minfRec, SELF.itr_counter := 1; SELF.sty := y_s; SELF.h := hdiag_updated; SELF.update_itr := 1; SELF := LEFT) ,LOCAL);//update_itr:=1 becasue if we are adding s_s_corr to the loop data it means the condition has been accepted
		// IF one of the break conditions is satisfied and minfunc!=-1 we need to return only x_new and g_new and no need to calculated Hdiag, s, y fo rthe next iteration
		//Conversly, if no condition is satisfied, it means we are going to the next iteration so we need to update Hdiag, s and y if and only if update_cond is satisfied, otherwise no update is done and we just return the current values to the next iteration
		TheReturn := IF (minfunc_cond = -1, IF (update_cond, g_x_nextitr + s_s_corr + y_y_corr, g_x_nextitr), break_result);
		Rresult := IF (dlegalstep, IF (gtdprogress, noprogalong_return, TheReturn) , dnotlegal_return);
		RETURN w;
		//RETURN g_x_nextitr + s_s_corr + y_y_corr;
	 END;// END min_step_firstitr_ttest
	 
	 //inp contains g (nno=1), x (no=2), correction vectors where s has no =3 and y has no=4 and different vectors are distingished with update_itr value
	 min_step (DATASET(minfRec) inp,unsigned4 min_c) := FUNCTION
		g_pre := inp (no = 1);
		x_pre := inp (no = 2);
		//calculate d
		// OUTPUT(inp,,'~thor::maryam::mytest::minfunc_inp',CSV(HEADING(SINGLE)), OVERWRITE);		
		whatever := MAX ([5, MAX (inp, update_itr)]);
		lbfgs_d := lbfgs2 (inp, min_c);
		d := lbfgs_d;
		// is d legal
		dlegalstep := IsLegal (d);// lbfgs algorithm keeps the funevals, cost_value and other fields for final calculated d same as what it is recieved intitially (inp(no=1))
		// Directional Derivative : gtd = g_pre'*d;
		Elem := {REAL8 v};
		Elem gtd_tran(minfRec inrec, minfRec drec) := TRANSFORM //hadamard product
			cells := inrec.part_rows * inrec.part_cols;
			SELF.v :=  sump(cells, inrec.mat_part, drec.mat_part);
		END;
		gtd_ := JOIN (g_pre, d, LEFT.partition_id = RIGHT.partition_id, gtd_tran (LEFT, RIGHT), LOCAL);
		gtd := SUM (gtd_, gtd_.v);
		//Check that progress can be made along direction : if gtd > -tolX then break!
		gtdprogress := gtd > -1*tolX;
		// Find Point satisfying Wolfe
		t_init := 1;
		w := WolfeLineSearch4_4_2(1, x_pre, CostFunc_params, TrainData, TrainLabel,CostFunc, t_init, d, g_pre, gtd, 0.0001, 0.9, wolfe_max_itr, 0.000000001);
		//w := ASSERT(w_, EXISTS(w_), 'w has zero rows', FAIL);
		//calculate new oint
		//calculate td
		minfRec td_tran (w le, d ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := PBBlas.BLAS.dscal(cells, le.prev_t, ri.mat_part, 1);
			SELF.no := 3;
			SELF.h := -1; //not calculated yet
			SELF.min_funEval := ri.min_funEval + le.wolfe_funevals;//ri.min_funEval is meaningful because in calculating d in the lbfgs function, we reserved the funeval value
			SELF.break_cond := -1;
			SELF.sty := -1;// not calculated yet
			SELF.update_itr := ri.update_itr; // will be increased if we decide to return s vector for the next iteration
			SELF.cost_value := le.cost_value; // new cost value calculated in wolfe function
			SELF := ri;
		END;
		td := JOIN (w, d, LEFT.partition_id = RIGHT.partition_id, td_tran (LEFT, RIGHT), LOCAL);
		//calculate x_new = x0 + td
		minfRec x_new_tran (minfRec le, minfRec ri) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.mat_part := PBblas.BLAS.daxpy(cells, 1.0, le.mat_part, 1, ri.mat_part, 1);
			SELF.no := 2;
			SELF.h := -1; //not calculated yer
			SELF.min_funEval := ri.min_funEval;
			SELF.break_cond := -1;
			SELF.sty := -1;// NA
			SELF.update_itr := ri.update_itr; // only will get increased if  we are not breaking the loop and update_cond is true
			SELF.cost_value := ri.cost_value;
			SELF := le;
		END;
		x_new := JOIN (x_pre, td, LEFT.partition_id = RIGHT.partition_id, x_new_tran(LEFT, RIGHT), LOCAL);
		//s_s = t*d
		s_s := td;
    //y_y := g_new - g_old
		minfRec y_y_tran (minfRec le, w ri) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.mat_part := PBblas.BLAS.daxpy(cells, -1.0, le.mat_part, 1, ri.mat_part, 1);
			SELF.no := 4;
			SELF.h := -1; //not calculated yer
			SELF.min_funEval := le.min_funEval + ri.wolfe_funevals;
			SELF.break_cond := -1;
			SELF.sty := -1;// not yet calculated
			SELF.update_itr := le.update_itr; // will get increased if we return the y vector for the next iteration
			SELF.cost_value := ri.cost_value;
			SELF := le;
		END;
		y_y := JOIN (g_pre, w, LEFT.partition_id = RIGHT.partition_id, y_y_tran(LEFT, RIGHT), LOCAL);
		//y_s = y_y'*s_s
		Elem y_s_tran (y_y le, s_s ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.v:= sump(cells, le.mat_part, ri.mat_part);
		END;
		y_s_ := JOIN (y_y, s_s, LEFT.partition_id = RIGHT.partition_id, y_s_tran (LEFT, RIGHT), LOCAL);
		y_s := SUM (y_s_, y_s_.v);
		//calculated y_y^2
		Elem yy2_tran (Layout_Part le) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.v := sum_sq(cells, le.mat_part);
		END;
		y_y_2_ := PROJECT (y_y, yy2_tran(LEFT), LOCAL);
		y_y_2 := SUM (y_y_2_, y_y_2_.v);
		hdiag_updated := y_s/ y_y_2;
		update_cond := y_s>10^(-10);
		// calculate sum(abs(g_new))
		Elem abs_g_tran (w le) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.v:= sumabs(cells, le.mat_part);
		END;
		sum_abs_g_new_ := PROJECT (w, abs_g_tran (LEFT), LOCAL);
		sum_abs_g_new := SUM (sum_abs_g_new_, sum_abs_g_new_.v);
		//calculate sum(abs(td))
		Elem abs_td_tran (td le) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.v:= sumabs(cells, le.mat_part);
		END;
		sum_abs_td_ := PROJECT (td, abs_td_tran(LEFT), LOCAL);
		sum_abs_td := SUM (sum_abs_td_, sum_abs_td_.v);
		//Optimality Condition sum(abs(g)) <= tolFun
		optimality_cond := sum_abs_g_new <= tolFun;
		//Check for lack of progress sum(abs(t*d)) <= tolX
		lack_prog_cond := sum_abs_td <= tolX;
		// -1: no condition
		// 1 : when d is not legal
		// 2 : when progress along direction not allowed
		// 3 : optimality cond
		// 4 : lack of progress
		minfunc_cond := IF (optimality_cond, 3, IF (lack_prog_cond, 4, -1));
		//when d is not legal, no other calculation is done and we return the current g and x values recived in the loop input with break_cond updated 
		dnotlegal_return := PROJECT (inp (no =1 OR no =2), TRANSFORM(minfrec, SELF.itr_counter := min_c + 1; SELF.break_cond := 1; SELF:= LEFT ), LOCAL);
		//After gtd is calculated check whether progress along the direction is possible, if it is not, the current g and x values recived in the loop input with break_cond updated 
		noprogalong_return := PROJECT (inp (no =1 OR no =2), TRANSFORM(minfrec, SELF.itr_counter := min_c + 1; SELF.break_cond := 2; SELF:= LEFT ), LOCAL);
		// if d is legal and progress along direction is allowed, then calculate new point using wolfe algorithm. Next check for minfunc termination conditions (break conditions) if any of them is satisfied
		//no need to calculate new hdiag , s and y values, just retunr new calculated x and g and return with break_cond updated
		x_new_break := PROJECT (x_new, TRANSFORM (minfRec, SELF.itr_counter := min_c + 1; SELF.break_cond := minfunc_cond; SELF := LEFT) ,LOCAL);
		//when passing g_new to wolfe function we lost the funevals, h and other values, so here we need to JOIN g_new with x_new in order to get those values back
		//g_new_break := PROJECT (w,TRANSFORM (minfRec, SELF.min_funEval := -1; SELF.break_cond := minfunc_cond; SELF.sty := -1; SELF.update_itr := -1;  SELF.no :=1; SELF.h := -1; SELF := LEFT) ,LOCAL);
		g_new_break := JOIN (w, x_new_break, LEFT.partition_id = RIGHT.partition_id, TRANSFORM (minfRec, SELF.itr_counter := min_c + 1; SELF.min_funEval := RIGHT.min_funEval;  SELF.break_cond := minfunc_cond; SELF.sty := -1; SELF.update_itr := RIGHT.update_itr;  SELF.no :=1; SELF.h := RIGHT.h; SELF:= LEFT), LOCAL);
		break_result := x_new_break + g_new_break;
		// id d is legal, progress along direction is allowed, no break condition is satified then return x_new, g_new values along with newly calculated Hdiag and s and y value only if update_cond is satified
		x_new_nextitr := PROJECT (x_new, TRANSFORM (minfRec, SELF.itr_counter := min_c + 1; SELF.update_itr := IF (update_cond, LEFT.update_itr + 1, LEFT.update_itr); SELF.h := IF (update_cond, hdiag_updated, LEFT.h); SELF.break_cond := minfunc_cond; SELF := LEFT) ,LOCAL);
		g_new_nextitr := JOIN (w, x_new_nextitr, LEFT.partition_id = RIGHT.partition_id, TRANSFORM (minfRec, SELF.itr_counter := min_c + 1; SELF.min_funEval := RIGHT.min_funEval;  SELF.break_cond := minfunc_cond; SELF.sty := -1; SELF.update_itr := RIGHT.update_itr;  SELF.no :=1; SELF.h := RIGHT.h; SELF:= LEFT), LOCAL);
		g_x_nextitr := x_new_nextitr + g_new_nextitr + inp ((no = 3 OR no = 4) AND update_itr > (min_c+1-corrections));// retunr new_x + new_g + the n=correction recent correction vectors, the old correction vectors should not be returned -> limited memory
		s_s_corr := PROJECT (s_s,TRANSFORM (minfRec, SELF.itr_counter := min_c + 1; SELF.sty := y_s; SELF.h := hdiag_updated; SELF.update_itr := LEFT.update_itr + 1; SELF := LEFT) ,LOCAL);//update_itr:=1 becasue if we are adding s_s_corr to the loop data it means the condition has been accepted
		y_y_corr := PROJECT (y_y,TRANSFORM (minfRec, SELF.itr_counter := min_c + 1; SELF.sty := y_s; SELF.h := hdiag_updated; SELF.update_itr := LEFT.update_itr + 1; SELF := LEFT) ,LOCAL);//update_itr:=1 becasue if we are adding s_s_corr to the loop data it means the condition has been accepted
		// IF one of the break conditions is satisfied and minfunc!=-1 we need to return only x_new and g_new and no need to calculated Hdiag, s, y fo rthe next iteration
		//Conversly, if no condition is satisfied, it means we are going to the next iteration so we need to update Hdiag, s and y if and only if update_cond is satisfied, otherwise no update is done and we just return the current values to the next iteration
		TheReturn := IF (minfunc_cond = -1, IF (update_cond, g_x_nextitr + s_s_corr + y_y_corr, g_x_nextitr), break_result);
		Rresult := IF (dlegalstep, IF (gtdprogress, noprogalong_return, TheReturn) , dnotlegal_return);
		RETURN Rresult;
		//RETURN PROJECT (w, TRANSFORM(minfRec, SELF.sty:=LEFT.prev_t; SELF.break_cond := LEFT.bracketing_cond; SELF.h := -1; SELF.update_itr := 0; SELF.no := 10; SELF.min_funeval := LEFT.wolfe_funevals;SELF:= LEFT), LOCAL);
		//RETURN g_x_nextitr + s_s_corr + y_y_corr;
 END;//END min_step
 
   
	
	 min_step2 (DATASET(minfRec) inp,unsigned4 min_c) := FUNCTION
		g_pre := inp (no = 1);
		x_pre := inp (no = 2);
		//calculate d
		upitr := MAX (inp, inp.update_itr);
		// lbfgs_d := lbfgs2 (inp, min_c);
		lbfgs_d := lbfgs2 (inp, upitr);
		d := lbfgs_d;
		// is d legal
		dlegalstep := IsLegal (d);// lbfgs algorithm keeps the funevals, cost_value and other fields for final calculated d same as what it is recieved intitially (inp(no=1))
		// Directional Derivative : gtd = g_pre'*d;
		Elem := {REAL8 v};
		Elem gtd_tran(minfRec inrec, minfRec drec) := TRANSFORM //hadamard product
			cells := inrec.part_rows * inrec.part_cols;
			SELF.v :=  sump(cells, inrec.mat_part, drec.mat_part);
		END;
		gtd_ := JOIN (g_pre, d, LEFT.partition_id = RIGHT.partition_id, gtd_tran (LEFT, RIGHT), LOCAL);
		gtd := SUM (gtd_, gtd_.v);
		//Check that progress can be made along direction : if gtd > -tolX then break!
		gtdprogress := gtd > -1*tolX;
		// Find Point satisfying Wolfe
		t_init := 1;
		w := WolfeLineSearch4_4_2(1, x_pre, CostFunc_params, TrainData, TrainLabel,CostFunc, t_init, d, g_pre, gtd, 0.0001, 0.9, wolfe_max_itr, 0.000000001);
		//w := ASSERT(w_, EXISTS(w_), 'w has zero rows', FAIL);
		//calculate new oint
		//calculate td
		minfRec td_tran (w le, d ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := PBBlas.BLAS.dscal(cells, le.prev_t, ri.mat_part, 1);
			SELF.no := 3;
			SELF.h := -1; //not calculated yet
			SELF.min_funEval := ri.min_funEval + le.wolfe_funevals;//ri.min_funEval is meaningful because in calculating d in the lbfgs function, we reserved the funeval value
			SELF.break_cond := -1;
			SELF.sty := -1;// not calculated yet
			SELF.update_itr := ri.update_itr; // will be increased if we decide to return s vector for the next iteration
			SELF.cost_value := le.cost_value; // new cost value calculated in wolfe function
			SELF := ri;
		END;
		td := JOIN (w, d, LEFT.partition_id = RIGHT.partition_id, td_tran (LEFT, RIGHT), LOCAL);
		//calculate x_new = x0 + td
		minfRec x_new_tran (minfRec le, minfRec ri) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.mat_part := PBblas.BLAS.daxpy(cells, 1.0, le.mat_part, 1, ri.mat_part, 1);
			SELF.no := 2;
			SELF.h := -1; //not calculated yer
			SELF.min_funEval := ri.min_funEval;
			SELF.break_cond := -1;
			SELF.sty := -1;// NA
			SELF.update_itr := ri.update_itr; // only will get increased if  we are not breaking the loop and update_cond is true
			SELF.cost_value := ri.cost_value;
			SELF := le;
		END;
		x_new := JOIN (x_pre, td, LEFT.partition_id = RIGHT.partition_id, x_new_tran(LEFT, RIGHT), LOCAL);
		//s_s = t*d
		s_s := td;
    //y_y := g_new - g_old
		minfRec y_y_tran (minfRec le, w ri) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.mat_part := PBblas.BLAS.daxpy(cells, -1.0, le.mat_part, 1, ri.mat_part, 1);
			SELF.no := 4;
			SELF.h := -1; //not calculated yer
			SELF.min_funEval := le.min_funEval + ri.wolfe_funevals;
			SELF.break_cond := -1;
			SELF.sty := -1;// not yet calculated
			SELF.update_itr := le.update_itr; // will get increased if we return the y vector for the next iteration
			SELF.cost_value := ri.cost_value;
			SELF := le;
		END;
		y_y := JOIN (g_pre, w, LEFT.partition_id = RIGHT.partition_id, y_y_tran(LEFT, RIGHT), LOCAL);
		//y_s = y_y'*s_s
		Elem y_s_tran (y_y le, s_s ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.v:= sump(cells, le.mat_part, ri.mat_part);
		END;
		y_s_ := JOIN (y_y, s_s, LEFT.partition_id = RIGHT.partition_id, y_s_tran (LEFT, RIGHT), LOCAL);
		y_s := SUM (y_s_, y_s_.v);
		//calculated y_y^2
		Elem yy2_tran (Layout_Part le) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.v := sum_sq(cells, le.mat_part);
		END;
		y_y_2_ := PROJECT (y_y, yy2_tran(LEFT), LOCAL);
		y_y_2 := SUM (y_y_2_, y_y_2_.v);
		hdiag_updated := y_s/ y_y_2;
		update_cond := y_s>10^(-10);
		// calculate sum(abs(g_new))
		Elem abs_g_tran (w le) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.v:= sumabs(cells, le.mat_part);
		END;
		sum_abs_g_new_ := PROJECT (w, abs_g_tran (LEFT), LOCAL);
		sum_abs_g_new := SUM (sum_abs_g_new_, sum_abs_g_new_.v);
		//calculate sum(abs(td))
		Elem abs_td_tran (td le) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.v:= sumabs(cells, le.mat_part);
		END;
		sum_abs_td_ := PROJECT (td, abs_td_tran(LEFT), LOCAL);
		sum_abs_td := SUM (sum_abs_td_, sum_abs_td_.v);
		//Optimality Condition sum(abs(g)) <= tolFun
		optimality_cond := sum_abs_g_new <= tolFun;
		//Check for lack of progress sum(abs(t*d)) <= tolX
		lack_prog_cond := sum_abs_td <= tolX;
		// -1: no condition
		// 1 : when d is not legal
		// 2 : when progress along direction not allowed
		// 3 : optimality cond
		// 4 : lack of progress
		// 5 : maximum number of iterations reached
		// 6 : Check for going over iteration/evaluation limit
		// 7 : lack of progress 2 : abs(f-f_old) < tolX
		funevalCond := x_new[1].min_funEval > maxFunEvals;
		costCond := ABS(x_pre[1].cost_value - w[1].cost_value) < tolX;

		minfunc_cond := IF (optimality_cond, 3, IF (lack_prog_cond, 4, IF (min_c > MaxIter, 5, IF (funevalCond, 6, IF (costCond, 7, -1)))));
		//when d is not legal, no other calculation is done and we return the current g and x values recived in the loop input with break_cond updated 
		dnotlegal_return := PROJECT (inp (no =1 OR no =2), TRANSFORM(minfrec, SELF.itr_counter := min_c + 1; SELF.break_cond := 1; SELF:= LEFT ), LOCAL);
		//After gtd is calculated check whether progress along the direction is possible, if it is not, the current g and x values recived in the loop input with break_cond updated 
		noprogalong_return := PROJECT (inp (no =1 OR no =2), TRANSFORM(minfrec, SELF.itr_counter := min_c + 1; SELF.break_cond := 2; SELF:= LEFT ), LOCAL);
		// if d is legal and progress along direction is allowed, then calculate new point using wolfe algorithm. Next check for minfunc termination conditions (break conditions) if any of them is satisfied
		//no need to calculate new hdiag , s and y values, just retunr new calculated x and g and return with break_cond updated
		x_new_break := PROJECT (x_new, TRANSFORM (minfRec, SELF.itr_counter := min_c + 1; SELF.break_cond := minfunc_cond; SELF := LEFT) ,LOCAL);
		//when passing g_new to wolfe function we lost the funevals, h and other values, so here we need to JOIN g_new with x_new in order to get those values back
		//g_new_break := PROJECT (w,TRANSFORM (minfRec, SELF.min_funEval := -1; SELF.break_cond := minfunc_cond; SELF.sty := -1; SELF.update_itr := -1;  SELF.no :=1; SELF.h := -1; SELF := LEFT) ,LOCAL);
		g_new_break := JOIN (w, x_new_break, LEFT.partition_id = RIGHT.partition_id, TRANSFORM (minfRec, SELF.itr_counter := min_c + 1; SELF.min_funEval := RIGHT.min_funEval;  SELF.break_cond := minfunc_cond; SELF.sty := -1; SELF.update_itr := RIGHT.update_itr;  SELF.no :=1; SELF.h := RIGHT.h; SELF:= LEFT), LOCAL);
		break_result := x_new_break + g_new_break;
		// id d is legal, progress along direction is allowed, no break condition is satified then return x_new, g_new values along with newly calculated Hdiag and s and y value only if update_cond is satified
		x_new_nextitr := PROJECT (x_new, TRANSFORM (minfRec, SELF.itr_counter := min_c + 1; SELF.update_itr := IF (update_cond, LEFT.update_itr + 1, LEFT.update_itr); SELF.h := IF (update_cond, hdiag_updated, LEFT.h); SELF.break_cond := minfunc_cond; SELF := LEFT) ,LOCAL);
		g_new_nextitr := JOIN (w, x_new_nextitr, LEFT.partition_id = RIGHT.partition_id, TRANSFORM (minfRec, SELF.itr_counter := min_c + 1; SELF.min_funEval := RIGHT.min_funEval;  SELF.break_cond := minfunc_cond; SELF.sty := -1; SELF.update_itr := RIGHT.update_itr;  SELF.no :=1; SELF.h := RIGHT.h; SELF:= LEFT), LOCAL);
		g_x_nextitr := x_new_nextitr + g_new_nextitr + inp ((no = 3 OR no = 4) AND update_itr > (min_c+1-corrections));// retunr new_x + new_g + the n=correction recent correction vectors, the old correction vectors should not be returned -> limited memory
		s_s_corr := PROJECT (s_s,TRANSFORM (minfRec, SELF.itr_counter := min_c + 1; SELF.sty := y_s; SELF.h := hdiag_updated; SELF.update_itr := LEFT.update_itr + 1; SELF := LEFT) ,LOCAL);//update_itr:=1 becasue if we are adding s_s_corr to the loop data it means the condition has been accepted
		y_y_corr := PROJECT (y_y,TRANSFORM (minfRec, SELF.itr_counter := min_c + 1; SELF.sty := y_s; SELF.h := hdiag_updated; SELF.update_itr := LEFT.update_itr + 1; SELF := LEFT) ,LOCAL);//update_itr:=1 becasue if we are adding s_s_corr to the loop data it means the condition has been accepted
		// IF one of the break conditions is satisfied and minfunc!=-1 we need to return only x_new and g_new and no need to calculated Hdiag, s, y fo rthe next iteration
		//Conversly, if no condition is satisfied, it means we are going to the next iteration so we need to update Hdiag, s and y if and only if update_cond is satisfied, otherwise no update is done and we just return the current values to the next iteration
		TheReturn := IF (minfunc_cond = -1, IF (update_cond, g_x_nextitr + s_s_corr + y_y_corr, g_x_nextitr), break_result);
		Rresult := IF (dlegalstep, IF (gtdprogress, noprogalong_return, TheReturn) , dnotlegal_return);
		RETURN Rresult;
		//RETURN PROJECT (w, TRANSFORM(minfRec, SELF.sty:=LEFT.prev_t; SELF.break_cond := LEFT.bracketing_cond; SELF.h := -1; SELF.update_itr := 0; SELF.no := 10; SELF.min_funeval := LEFT.wolfe_funevals;SELF:= LEFT), LOCAL);
		//RETURN g_x_nextitr + s_s_corr + y_y_corr;
 END;//END min_step2
 
 
 	 min_step2_9 (DATASET(minfRec) inp,unsigned4 min_c) := FUNCTION
		g_pre := inp (no = 1);
		x_pre := inp (no = 2);
		//calculate d
		upitr := MAX (inp, inp.update_itr);
		// lbfgs_d := lbfgs2 (inp, min_c);
		lbfgs_d := lbfgs2 (inp, upitr);
		d := lbfgs_d;
		// is d legal
		dlegalstep := IsLegal (d);// lbfgs algorithm keeps the funevals, cost_value and other fields for final calculated d same as what it is recieved intitially (inp(no=1))
		// Directional Derivative : gtd = g_pre'*d;
		Elem := {REAL8 v};
		Elem gtd_tran(minfRec inrec, minfRec drec) := TRANSFORM //hadamard product
			cells := inrec.part_rows * inrec.part_cols;
			SELF.v :=  sump(cells, inrec.mat_part, drec.mat_part);
		END;
		gtd_ := JOIN (g_pre, d, LEFT.partition_id = RIGHT.partition_id, gtd_tran (LEFT, RIGHT), LOCAL);
		gtd := SUM (gtd_, gtd_.v);
		//Check that progress can be made along direction : if gtd > -tolX then break!
		gtdprogress := gtd > -1*tolX;
		// Find Point satisfying Wolfe
		t_init := 1;
		w := WolfeLineSearch4_4_2(1, x_pre, CostFunc_params, TrainData, TrainLabel,CostFunc, t_init, d, g_pre, gtd, 0.0001, 0.9, wolfe_max_itr, 0.000000001);
		//w := ASSERT(w_, EXISTS(w_), 'w has zero rows', FAIL);
		//calculate new oint
		//calculate td
		minfRec td_tran (w le, d ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := PBBlas.BLAS.dscal(cells, le.prev_t, ri.mat_part, 1);
			SELF.no := 3;
			SELF.h := -1; //not calculated yet
			SELF.min_funEval := ri.min_funEval + le.wolfe_funevals;//ri.min_funEval is meaningful because in calculating d in the lbfgs function, we reserved the funeval value
			SELF.break_cond := -1;
			SELF.sty := -1;// not calculated yet
			SELF.update_itr := ri.update_itr; // will be increased if we decide to return s vector for the next iteration
			SELF.cost_value := le.cost_value; // new cost value calculated in wolfe function
			SELF := ri;
		END;
		td := JOIN (w, d, LEFT.partition_id = RIGHT.partition_id, td_tran (LEFT, RIGHT), LOCAL);
		//calculate x_new = x0 + td
		minfRec x_new_tran (minfRec le, minfRec ri) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.mat_part := PBblas.BLAS.daxpy(cells, 1.0, le.mat_part, 1, ri.mat_part, 1);
			SELF.no := 2;
			SELF.h := -1; //not calculated yer
			SELF.min_funEval := ri.min_funEval;
			SELF.break_cond := -1;
			SELF.sty := -1;// NA
			SELF.update_itr := ri.update_itr; // only will get increased if  we are not breaking the loop and update_cond is true
			SELF.cost_value := ri.cost_value;
			SELF := le;
		END;
		x_new := JOIN (x_pre, td, LEFT.partition_id = RIGHT.partition_id, x_new_tran(LEFT, RIGHT), LOCAL);
		//s_s = t*d
		s_s := td;
    //y_y := g_new - g_old
		minfRec y_y_tran (minfRec le, w ri) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.mat_part := PBblas.BLAS.daxpy(cells, -1.0, le.mat_part, 1, ri.mat_part, 1);
			SELF.no := 4;
			SELF.h := -1; //not calculated yer
			SELF.min_funEval := le.min_funEval + ri.wolfe_funevals;
			SELF.break_cond := -1;
			SELF.sty := -1;// not yet calculated
			SELF.update_itr := le.update_itr; // will get increased if we return the y vector for the next iteration
			SELF.cost_value := ri.cost_value;
			SELF := le;
		END;
		y_y := JOIN (g_pre, w, LEFT.partition_id = RIGHT.partition_id, y_y_tran(LEFT, RIGHT), LOCAL);
		//y_s = y_y'*s_s
		Elem y_s_tran (y_y le, s_s ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.v:= sump(cells, le.mat_part, ri.mat_part);
		END;
		y_s_ := JOIN (y_y, s_s, LEFT.partition_id = RIGHT.partition_id, y_s_tran (LEFT, RIGHT), LOCAL);
		y_s := SUM (y_s_, y_s_.v);
		//calculated y_y^2
		Elem yy2_tran (Layout_Part le) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.v := sum_sq(cells, le.mat_part);
		END;
		y_y_2_ := PROJECT (y_y, yy2_tran(LEFT), LOCAL);
		y_y_2 := SUM (y_y_2_, y_y_2_.v);
		hdiag_updated := y_s/ y_y_2;
		update_cond := y_s>10^(-10);
		// calculate sum(abs(g_new))
		Elem abs_g_tran (w le) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.v:= sumabs(cells, le.mat_part);
		END;
		sum_abs_g_new_ := PROJECT (w, abs_g_tran (LEFT), LOCAL);
		sum_abs_g_new := SUM (sum_abs_g_new_, sum_abs_g_new_.v);
		//calculate sum(abs(td))
		Elem abs_td_tran (td le) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.v:= sumabs(cells, le.mat_part);
		END;
		sum_abs_td_ := PROJECT (td, abs_td_tran(LEFT), LOCAL);
		sum_abs_td := SUM (sum_abs_td_, sum_abs_td_.v);
		//Optimality Condition sum(abs(g)) <= tolFun
		optimality_cond := sum_abs_g_new <= tolFun;
		//Check for lack of progress sum(abs(t*d)) <= tolX
		lack_prog_cond := sum_abs_td <= tolX;
		// -1: no condition
		// 1 : when d is not legal
		// 2 : when progress along direction not allowed
		// 3 : optimality cond
		// 4 : lack of progress
		// 5 : maximum number of iterations reached
		// 6 : Check for going over iteration/evaluation limit
		// 7 : lack of progress 2 : abs(f-f_old) < tolX
		funevalCond := x_new[1].min_funEval > maxFunEvals;
		costCond := ABS(x_pre[1].cost_value - w[1].cost_value) < tolX;

		minfunc_cond := IF (optimality_cond, 3, IF (lack_prog_cond, 4, IF (min_c > MaxIter, 5, IF (funevalCond, 6, IF (costCond, 7, -1)))));
		//when d is not legal, no other calculation is done and we return the current g and x values recived in the loop input with break_cond updated 
		dnotlegal_return := PROJECT (inp (no =1 OR no =2), TRANSFORM(minfrec, SELF.itr_counter := min_c + 1; SELF.break_cond := 1; SELF:= LEFT ), LOCAL);
		//After gtd is calculated check whether progress along the direction is possible, if it is not, the current g and x values recived in the loop input with break_cond updated 
		noprogalong_return := PROJECT (inp (no =1 OR no =2), TRANSFORM(minfrec, SELF.itr_counter := min_c + 1; SELF.break_cond := 2; SELF:= LEFT ), LOCAL);
		// if d is legal and progress along direction is allowed, then calculate new point using wolfe algorithm. Next check for minfunc termination conditions (break conditions) if any of them is satisfied
		//no need to calculate new hdiag , s and y values, just retunr new calculated x and g and return with break_cond updated
		x_new_break := PROJECT (x_new, TRANSFORM (minfRec, SELF.itr_counter := min_c + 1; SELF.break_cond := minfunc_cond; SELF := LEFT) ,LOCAL);
		//when passing g_new to wolfe function we lost the funevals, h and other values, so here we need to JOIN g_new with x_new in order to get those values back
		//g_new_break := PROJECT (w,TRANSFORM (minfRec, SELF.min_funEval := -1; SELF.break_cond := minfunc_cond; SELF.sty := -1; SELF.update_itr := -1;  SELF.no :=1; SELF.h := -1; SELF := LEFT) ,LOCAL);
		g_new_break := JOIN (w, x_new_break, LEFT.partition_id = RIGHT.partition_id, TRANSFORM (minfRec, SELF.itr_counter := min_c + 1; SELF.min_funEval := RIGHT.min_funEval;  SELF.break_cond := minfunc_cond; SELF.sty := -1; SELF.update_itr := RIGHT.update_itr;  SELF.no :=1; SELF.h := RIGHT.h; SELF:= LEFT), LOCAL);
		break_result := x_new_break + g_new_break;
		// id d is legal, progress along direction is allowed, no break condition is satified then return x_new, g_new values along with newly calculated Hdiag and s and y value only if update_cond is satified
		x_new_nextitr := PROJECT (x_new, TRANSFORM (minfRec, SELF.itr_counter := min_c + 1; SELF.update_itr := IF (update_cond, LEFT.update_itr + 1, LEFT.update_itr); SELF.h := IF (update_cond, hdiag_updated, LEFT.h); SELF.break_cond := minfunc_cond; SELF := LEFT) ,LOCAL);
		g_new_nextitr := JOIN (w, x_new_nextitr, LEFT.partition_id = RIGHT.partition_id, TRANSFORM (minfRec, SELF.itr_counter := min_c + 1; SELF.min_funEval := RIGHT.min_funEval;  SELF.break_cond := minfunc_cond; SELF.sty := -1; SELF.update_itr := RIGHT.update_itr;  SELF.no :=1; SELF.h := RIGHT.h; SELF:= LEFT), LOCAL);
		g_x_nextitr := x_new_nextitr + g_new_nextitr + inp ((no = 3 OR no = 4) AND update_itr > (min_c+1-corrections));// retunr new_x + new_g + the n=correction recent correction vectors, the old correction vectors should not be returned -> limited memory
		s_s_corr := PROJECT (s_s,TRANSFORM (minfRec, SELF.itr_counter := min_c + 1; SELF.sty := y_s; SELF.h := hdiag_updated; SELF.update_itr := LEFT.update_itr + 1; SELF := LEFT) ,LOCAL);//update_itr:=1 becasue if we are adding s_s_corr to the loop data it means the condition has been accepted
		y_y_corr := PROJECT (y_y,TRANSFORM (minfRec, SELF.itr_counter := min_c + 1; SELF.sty := y_s; SELF.h := hdiag_updated; SELF.update_itr := LEFT.update_itr + 1; SELF := LEFT) ,LOCAL);//update_itr:=1 becasue if we are adding s_s_corr to the loop data it means the condition has been accepted
		// IF one of the break conditions is satisfied and minfunc!=-1 we need to return only x_new and g_new and no need to calculated Hdiag, s, y fo rthe next iteration
		//Conversly, if no condition is satisfied, it means we are going to the next iteration so we need to update Hdiag, s and y if and only if update_cond is satisfied, otherwise no update is done and we just return the current values to the next iteration
		TheReturn := IF (minfunc_cond = -1, IF (update_cond, g_x_nextitr + s_s_corr + y_y_corr, g_x_nextitr), break_result);
		Rresult := IF (dlegalstep, IF (gtdprogress, noprogalong_return, TheReturn) , dnotlegal_return);
		RETURN td;
		//RETURN PROJECT (w, TRANSFORM(minfRec, SELF.sty:=LEFT.prev_t; SELF.break_cond := LEFT.bracketing_cond; SELF.h := -1; SELF.update_itr := 0; SELF.no := 10; SELF.min_funeval := LEFT.wolfe_funevals;SELF:= LEFT), LOCAL);
		//RETURN g_x_nextitr + s_s_corr + y_y_corr;
 END;//END min_step2_9
 //RETURN lbfgs(min_step_firstitr);
 //RETURN min_step_firstitr;
 m1 := min_step_firstitr;
 // m2 := min_step (m1,1);
// l2 := lbfgs3 (m2);//works
//works
// m2 := min_step2 (m1,1);
 // l2 := min_step2 (m2,2);
// end works
// min_step (m2,2);
 //RETURN min_step (m1,1);
 //RETURN m1;
 m4 := LOOP(m1, COUNTER <= 3, min_step(ROWS(LEFT),COUNTER));
m28 :=  LOOP(m1, 28 , min_step(ROWS(LEFT),COUNTER));
m6 := LOOP(m1, COUNTER <= 6, min_step(ROWS(LEFT),COUNTER));
 // min_step29 (m28 ,29)
 //RETURN m1;
  //RETURN min_step29 (m28 ,29);
	//RETURN LOOP(m1, 3, min_step(ROWS(LEFT),COUNTER));
	m2 := LOOP(m1, 2, min_step(ROWS(LEFT),COUNTER));
	//RETURN min_step_test (m2, 3);
	//RETURN LOOP(m1, 10, min_step_maxup(ROWS(LEFT),COUNTER));//minstep works
	// RETURN LOOP(m1, 1, min_step(ROWS(LEFT),COUNTER));//minstep works
	
	//start here
	m9 := LOOP(m1, 9, min_step2(ROWS(LEFT),COUNTER));
	
 // RETURN LOOP (m1, LEFT.break_cond = -1 , min_step2(ROWS(LEFT),COUNTER)); //orig

 RETURN m1;
// LOOP(topass_zooming, (LEFT.zoomtermination = FALSE) AND (LEFT.c < (maxLS+1)), ZoomingStep(ROWS(LEFT), COUNTER));
  END;// END MinFUNC

END;// END Optimization

// in lbfgs2 k should be max(update_itr)
// f-fpre and one other condition should be added to minfunc
// program root and poly in c++
// The reason calculated cost value is not deterministic is that whenever there is a distribution followed by a roll up the results are not deterministic. Because distribution distributes record in different orders and adding floqting values a+b+c results in different value as adding c+a+b
// an example of this is d3a2t function in deep learning MODULE where the calclation of final results is not determinitstic due to roll up after a distribution
//I output the same mul_part_dist_sorted out multiple times, each time the order of records are different 
// in 