//this version is a newer version of Optimization_new_new_2_2_nf which works for REAL8 implementation of softmax_lbfgs compatible label partitioned cost function
//#option ('divideByZero', 'nan');
IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat;
IMPORT * FROM ML.Types;
IMPORT PBblas;
IMPORT STD;
IMPORT std.system.Thorlib;

Layout_Cell4 := PBblas.Types.Layout_Cell4;
Layout_part4 := PBblas.Types.Layout_part4;
SHARED nodes_available := STD.system.Thorlib.nodes();
SHARED Layout_Cell_nid := RECORD (Pbblas.Types.Layout_Cell)
UNSIGNED4 node_id;
END;
SHARED Layout_Cell_nid4 := RECORD (Layout_Cell4)
PBblas.Types.node_t node_id;
END;


      
// A version of Optimization_new_new_2_2 where the costfuc has a different format than costfun in Optimization_new_new_2_2
// the train data is provided in Layout_Cell_nid which has been converted from numericfield format (_nf)
//Func : handle to the function we want to minimize it, its output should be the error cost and the error gradient
EXPORT Optimization_new_new_2_2_nf4 (UNSIGNED4 prows=0, UNSIGNED4 pcols=0, UNSIGNED4 Maxrows=0, UNSIGNED4 Maxcols=0) := MODULE
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
	SHARED 	costgrad_record4 := RECORD (Layout_Part4)
			REAL4 cost_value;
	  END;

		
	
		EXPORT minfRec4 := RECORD (costgrad_record4)
      REAL4 h ;//hdiag value
      UNSIGNED min_funEval;
      INTEGER break_cond ;
			REAL4 sty  ;
			PBblas.Types.t_mu_no no;
			INTEGER8 update_itr ; //This value is increased whenever a update is done and s and y vectors are added to the corrections. If no update is done due to the condition ys > 1e-10 then this value is not increased
			// we use this value to update the corrections vectors as well as in the lbfgs algorithm
			UNSIGNED itr_counter;
    END;
		

		//BoundProvided = 1 -> xminBound and xmaxBound values are provided
		//BoundProvided = 0 -> xminBound and xmaxBound values are not provided
		//set  f or g related values to 2 if f or g are not known ( f values are _2 values and g values are _3 values)
    // the order of the polynomial is the number of known f and g values minus 1. for example if first f (p1_2) does not exist the value of f1_2 will be equal to 0, otherwise it would be 1
		
        
    
		
 

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
  //

		SHARED arm_t_rec4 := RECORD
			real4 init_arm_t;
			Layout_part4.partition_id;
		END;

		
	
		SHARED Z4oomingRecord := RECORD (CostGrad_Record4)
			INTEGER id;
			REAL4 prev_t;
			REAL4 prev_gtd;
			UNSIGNED wolfe_funEvals;
			UNSIGNED8 c;
			INTEGER bracketing_cond;
			INTEGER zooming_cond := 0;
			REAL4 next_t;
			REAL4 high_t;
			REAL4 high_cost_value;
			REAL4 high_gtd;
			REAL4 glob_f; // this is the f value we recive through wolfelinesearch function call
			BOOLEAN insufProgress;
			BOOLEAN zoomtermination;
		END;

EXPORT A4rmijoBacktrack_fromwolfe(DATASET(PBblas.Types.Layout_Part4) x, DATASET(Types.NumericField4) CostFunc_params, DATASET(Layout_Cell_nid4) TrainData , DATASET(PBblas.Types.Layout_Part4) TrainLabel,DATASET(CostGrad_Record4) CostFunc (DATASET(PBblas.Types.Layout_Part4) x, DATASET(Types.NumericField4) CostFunc_params, DATASET(Layout_Cell_nid4) TrainData , DATASET(PBblas.Types.Layout_Part4) TrainLabel), DATASET(arm_t_rec4) t, DATASET(Layout_Part4) d, DATASET(costgrad_record4) g, REAL4 gtd, REAL4 c1=0.0001, REAL4 c2=0.9, REAL4 tolX=0.000000001):=FUNCTION
  // C++ functions
	
	PBblas.Types.value_t4 sumabs(PBblas.Types.dimension_t N, PBblas.Types.matrix_t4 M) := BEGINC++

    #body
    float result = 0;
		float tmpp ;
    float *cellm = (float*) m;
    uint32_t i;
		for (i=0; i<n; i++) {
		tmpp = (cellm[i]>=0) ? (cellm[i]) : (-1 * cellm[i]);
      result = result + tmpp;
    }
		return(result);

   ENDC++;
	 
	PBblas.Types.value_t4 sump(PBblas.Types.dimension_t N, PBblas.Types.matrix_t4 M, PBblas.Types.matrix_t4 V) := BEGINC++
		#body
		float result = 0;
		float tmpp ;
		float *cellm = (float*) m;
		float *cellv = (float*) v;
		uint32_t i;
		for (i=0; i<n; i++) {
			tmpp =(cellm[i] * cellv [i]);
			result = result + tmpp;
		}
		return(result);

	ENDC++;
	 
	SET OF Pbblas.Types.value_t4 scale_mat (PBblas.Types.dimension_t N, Pbblas.Types.value_t4 c, PBblas.Types.matrix_t4 M ) := BEGINC++

    #body
    __lenResult = n * sizeof(float);
    __isAllResult = false;
    float * result = new float[n];
    __result = (void*) result;
    float *cellm = (float*) m;
    uint32_t i;
		for (i=0; i<n; i++){
			result[i] = cellm[i]*c;
		}

  ENDC++;
	// l*M + D
	SET OF Pbblas.Types.value_t4 summation(PBblas.types.dimension_t N, Pbblas.Types.value_t4 L, PBblas.types.matrix_t4 M, PBblas.types.matrix_t4 D) := BEGINC++

    #body
    __lenResult = n * sizeof(float);
    __isAllResult = false;
    float * result = new float[n];
    __result = (void*) result;
		float *cellm = (float*) m;
    float *celld = (float*) d;
    uint32_t i;
		for (i=0; i<n; i++) {
			result[i] = (l*cellm[i])+celld[i];
    }
		
  ENDC++;
	
  // Evaluate the Objective and Gradient at the Initial Step
	Elem := {PBblas.Types.value_t4 v};
	//calculate x_new = x + td
	//first calculate td
	Layout_Part4 td_tran (t le, d ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := scale_mat(cells, le.init_arm_t, ri.mat_part);
			SELF := ri;
		END;
		td := JOIN (t, d, LEFT.partition_id = RIGHT.partition_id, td_tran (LEFT, RIGHT), LOCAL);
		
		//calculate x_new = x0 + td
		Layout_Part4 x_new_tran (Layout_part4 le, Layout_part4 ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := summation (cells, 1.0, le.mat_part, ri.mat_part);
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
	ArmijoRecord := RECORD (CostGrad_Record4)
		REAL4 fprev;
		REAL4 tprev;// this is the actualy previous t calculated in the previous iteration
		REAL4 prev_t;// this should actually be tnew, however in order for this record format to be consistent with the ourput of wolfe line search , the new t has to be named prev_t
		UNSIGNED wolfe_funevals;
		INTEGER armCond;
		REAL4 glob_f;
		REAL4 gtdnew;
		BOOLEAN islegal_gnew := TRUE;
		REAL4 local_sumd;
	END; // ArmijoRec
	ArmijoRecord_shorten := RECORD
		REAL4 fprev;
		REAL4 tprev;// this is the actualy previous t calculated in the previous iteration
		REAL4 prev_t;// this should actually be tnew, however in order for this record format to be consistent with the ourput of wolfe line search , the new t has to be named prev_t
		UNSIGNED wolfe_funevals;
		INTEGER armCond;
		REAL4 glob_f;
		REAL4 gtdnew;
		BOOLEAN islegal_gnew := TRUE;
		Layout_Part4.partition_id;
		CostGrad_Record4.cost_value;
		REAL4 local_sumd;
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
		Elem := {REAL4 v};
		Layout_Part4 td_tran (armin_t le, d ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := scale_mat(cells, le.prev_t, ri.mat_part);
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
		 Layout_Part4 x_new_tran (Layout_part4 le, Layout_part4 ri) := TRANSFORM
				cells := le.part_rows * le.part_cols;
				SELF.mat_part := summation(cells, 1.0, le.mat_part, ri.mat_part);
				SELF := le;
			END;
			armx_new := JOIN (x, armtd, LEFT.partition_id = RIGHT.partition_id, x_new_tran(LEFT, RIGHT), LOCAL);
			armCostGrad_new := CostFunc(armx_new,CostFunc_params,TrainData, TrainLabel);
			//calculate gtdnew
			Elem armgtdnew_tran(armCostGrad_new inrec, d drec) := TRANSFORM
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

END;// END A4rmijoBacktrack_fromwolfe
//this ArmijoBacktrack is called from wolfelinesearch function, the initial t is provided in arm_t_rec format arm_t_rec


EXPORT W4olfeLineSearch4_4_2_test(INTEGER cccc, DATASET(PBblas.Types.Layout_Part4) x, DATASET(Types.NumericField4) CostFunc_params, DATASET(Layout_Cell_nid4) TrainData , DATASET(PBblas.Types.Layout_Part4) TrainLabel,DATASET(CostGrad_Record4) CostFunc (DATASET(PBblas.Types.Layout_Part4) x, DATASET(Types.NumericField4) CostFunc_params, DATASET(Layout_Cell_nid4) TrainData , DATASET(PBblas.Types.Layout_Part4) TrainLabel), REAL4 t, DATASET(Layout_Part4) d, DATASET(costgrad_record4) g, REAL4 gtd, REAL4 c1=0.0001, REAL4 c2=0.9, UNSIGNED maxLS=25, REAL4 tolX=0.000000001):=FUNCTION
	//C++ functions used
	//sum (M.*V)
	SET OF Pbblas.Types.value_t4 scale_mat (PBblas.Types.dimension_t N, Pbblas.Types.value_t4 c, PBblas.Types.matrix_t4 M ) := BEGINC++

    #body
    __lenResult = n * sizeof(float);
    __isAllResult = false;
    float * result = new float[n];
    __result = (void*) result;
    float *cellm = (float*) m;
    uint32_t i;
		for (i=0; i<n; i++){
			result[i] = cellm[i]*c;
		}

  ENDC++;
	// l*M + D
	SET OF Pbblas.Types.value_t4 summation(PBblas.types.dimension_t N, Pbblas.Types.value_t4 L, PBblas.types.matrix_t4 M, PBblas.types.matrix_t4 D) := BEGINC++

    #body
    __lenResult = n * sizeof(float);
    __isAllResult = false;
    float * result = new float[n];
    __result = (void*) result;
		float *cellm = (float*) m;
    float *celld = (float*) d;
    uint32_t i;
		for (i=0; i<n; i++) {
			result[i] = (l*cellm[i])+celld[i];
    }
		
  ENDC++;
	
	PBblas.Types.value_t4 sump(PBblas.Types.dimension_t N, PBblas.Types.matrix_t4 M, PBblas.Types.matrix_t4 V) := BEGINC++
		#body
		float result = 0;
		float tmpp ;
		float *cellm = (float*) m;
		float *cellv = (float*) v;
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
	topass_bracketing := PROJECT (g, TRANSFORM (Z4oomingRecord, SELF.zooming_cond := -1; SELF.zoomtermination := FALSE; SELF.insufProgress := FALSE; SELF.id := 1; SELF.glob_f :=LEFT.cost_value; SELF.high_t := -1 ; SELF.high_gtd := -1; SELF.high_cost_value := -1; SELF.prev_t := 0; SELF.prev_gtd := gtd; SELF.wolfe_funEvals := 0; SELF.c := 0; SELF.bracketing_cond := -1; SELF.next_t := t; SELF := LEFT), LOCAL);
	
	BracketingStep (DATASET (Z4oomingRecord) inputp, UNSIGNED coun) := FUNCTION
		//the idea is to calculate the new values at the begining of the bracket and then JOIN the loop input and this new values in order to generate the loop output. The JOIN TRANSFORM contains all the condition checks
		// calculate new g and cost value at the begining of the loop using next_t value in the input dataset
		// calculate t*d
		Elem := {REAL4 v};
		Layout_Part4 td_tran (inputp le, d ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := scale_mat(cells, le.next_t, ri.mat_part);
			SELF := le;
		END;
		td := JOIN (inputp, d, LEFT.partition_id = RIGHT.partition_id, td_tran (LEFT, RIGHT), LOCAL);
		//calculate x_new = x0 + td
		Layout_Part4 x_new_tran (Layout_part4 le, Layout_part4 ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := summation(cells, 1.0, le.mat_part, ri.mat_part);
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
		//we always return the braket side with smaller cost value, the values for the Hi side of the bracket are in high_ fields. we don't return gradient vector for the high side
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
		arm_t_rec4 arm_init_t_tran (inputp le) := TRANSFORM
		  //t = (t + t_prev)/2;
			SELF.init_arm_t := (le.next_t + le.prev_t)/2;
			SELF := le;
		END;
		init_arm_t := PROJECT (inputp, arm_init_t_tran(LEFT), LOCAL);
		Arm_Result := A4rmijoBacktrack_fromwolfe( x,  CostFunc_params, TrainData , TrainLabel, CostFunc , init_arm_t, d,  g, gtd,  c1,  c2, tolX);
		Arm_result_zoomingformat := PROJECT (Arm_Result, TRANSFORM (Z4oomingRecord,
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
		
		Z4oomingRecord main_tran (inputp le_prev, CostGrad_new ri_new) := TRANSFORM
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
			//SELF.zoomtermination := IS_f_g_new_legal;
			SELF := le_prev;
		END;
		Z4oomingRecord main_tran_test (inputp le_prev, CostGrad_new ri_new) := TRANSFORM
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
			// cond1 := ((fNew > f + c1 * newt* gtd)| ((BrackLSiter > 1) & (fNew >= fPrev))) & (coun < (maxLS+1) );
			cond1 := ((fNew > (f + c1 * newt* gtd)) );
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
			// SELF.cost_value := IF (prev_or_new = 1 , fPrev, fNew);
			SELF.cost_value :=  f;
			// SELF.high_cost_value := f-(f+ c1 * gtd); 
			SELF.high_cost_value :=( (f+ gtd))-f; 
			// SELF.prev_gtd := gtd;
			SELF.prev_gtd := ((REAL4)c1* gtd);
			// SELF.high_gtd :=   c1 ;
			SELF.high_gtd := f-(f+ ((REAL8)c1* gtd));
			xx := (((REAL4)0.0001* gtd));
			REAL4 freal := f;
			SELF.prev_t := freal-(freal+ xx);
			SELF.high_t :=  xx;
			SELF.id := prev_or_new; // thd id shows whether the associated value to prev is the left or riht side of the bracket, the hight value will be the other value of id (if id=1 it means the prev values start the bracket and high ends the brackte) 
			//actually the id only matters when we exit the loop because of cond1 or cond3. cond2 and reaching maxitr casue program to end so id does not matter becasue we don't go to zooming loop
			SELF.bracketing_cond := WhichCond;
			SELF.c := coun;
			// SELF.next_t := f-(f+ newt* gtd);// next_t is calculated only if we are going to the next iteration which means WhichCond should be -1 ( no condition has been satisfied) and we have not rached the Max number of iterations
			SELF.next_t :=newt;
			SELF.wolfe_funEvals := 87;
			SELF.zoomtermination := newt = (REAL8) 0.0;
			SELF := le_prev;
		END;
		Result := JOIN (inputp, CostGrad_new, LEFT.partition_id = RIGHT.partition_id, main_tran(LEFT, RIGHT), LOCAL);
		final_result := IF (IS_f_g_new_legal, result, Arm_result_zoomingformat); // if either of f_new or g_new are not legal return the armijobacktracking instead of wolfelinesearch
		
		Z4oomingRecord this_tran (CostGrad_new ri) := TRANSFORM

			SELF.high_cost_value := 1;
			SELF.glob_f := 10;
			SELF.insufprogress := 10;
			SELF.zoomtermination := FALSE;
			SELF.prev_gtd := 1;
			SELF.high_gtd := 1;
			SELF.prev_t := 1;
			SELF.high_t := 1;
			SELF.id := 1; // thd id shows whether the associated value to prev is the left or riht side of the bracket, the hight value will be the other value of id (if id=1 it means the prev values start the bracket and high ends the brackte) 
			//actually the id only matters when we exit the loop because of cond1 or cond3. cond2 and reaching maxitr casue program to end so id does not matter becasue we don't go to zooming loop
			SELF.bracketing_cond := 1;
			SELF.c := 1;
			SELF.next_t := 10;
			SELF.wolfe_funEvals := 1;
			SELF := ri;
		END;
		// RETURN final_result;
		// RETURN inputp+ PROJECT (CostGrad_new, this_tran(LEFT), LOCAL);
		// RETURN PROJECT (CostGrad_new, this_tran(LEFT), LOCAL)+inputp;
		RETURN JOIN (inputp, CostGrad_new, LEFT.partition_id = RIGHT.partition_id, main_tran_test(LEFT, RIGHT), LOCAL);
		// RETURN result;
	END; // END BracketingStep
	/*
	//after the bracketing step check whether we have reached maxitr , however if we left bracketing step with which_cond==10 which means we used armijobacktracking then this is the final result and does not need to go for maxitr check
	Z4oomingRecord maxitr_tran (Z4oomingRecord le_brack , CostGrad_record4 ri_g) := TRANSFORM
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
	
	bracketing_result_ := LOOP(topass_bracketing, LEFT.bracketing_cond = -1, BracketingStep(ROWS(LEFT), COUNTER));
	//check if MAXITR is reached and decide between bracketing result and f (t=0) to be returned
	bracketing_result := JOIN (bracketing_result_ , g, LEFT.partition_id = RIGHT.partition_id, maxitr_tran (LEFT, RIGHT), LOCAL);
	topass_zooming := bracketing_result;
	
	ZoomingStep (DATASET (Z4oomingRecord) inputp, UNSIGNED coun) := FUNCTION
		//similar to BracketingStep, first we calculate the new point (for that we need to calculate the new t using interpolation)
		//after the new point is calculated we JOIN the input and the new point in order to decide which one to return
		// Compute new trial value & Test that we are making sufficient progress
		//The following transform calculates the new trial value as well as insufProgress value by doing a PROJECT on the input, the new t is then saved in next_t field
		Z4oomingRecord newt_tran (Z4oomingRecord le) := TRANSFORM
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
		Elem := {REAL4 v};
		Layout_Part4 td_tran (inputp_t_insuf le, d ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := scale_mat(cells, le.next_t, ri.mat_part);
			SELF := le;
		END;
		td := JOIN (inputp_t_insuf, d, LEFT.partition_id = RIGHT.partition_id, td_tran (LEFT, RIGHT), LOCAL);
		// calculate x_new = x0 + td
		Layout_Part4 x_new_tran (Layout_part4 le, Layout_part4 ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := summation(cells, 1.0, le.mat_part, ri.mat_part);
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
		Z4oomingRecord main_tran (inputp le_prev, CostGrad_new ri_new) := TRANSFORM
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
	// RETURN zooming_result;
// RETURN BracketingStep(topass_bracketing, 1);

*/
// RETURN LOOP(topass_bracketing, LEFT.bracketing_cond = -1, BracketingStep(ROWS(LEFT), COUNTER));
RETURN BracketingStep(topass_bracketing, 1);
// RETURN topass_bracketing;
END;// END W4olfeLineSearch4_4_2_test
    


EXPORT W4olfeLineSearch4_4_2(INTEGER cccc, DATASET(PBblas.Types.Layout_Part4) x, DATASET(Types.NumericField4) CostFunc_params, DATASET(Layout_Cell_nid4) TrainData , DATASET(PBblas.Types.Layout_Part4) TrainLabel,DATASET(CostGrad_Record4) CostFunc (DATASET(PBblas.Types.Layout_Part4) x, DATASET(Types.NumericField4) CostFunc_params, DATASET(Layout_Cell_nid4) TrainData , DATASET(PBblas.Types.Layout_Part4) TrainLabel), REAL4 t, DATASET(Layout_Part4) d, DATASET(costgrad_record4) g, REAL4 gtd, REAL4 c1=0.0001, REAL4 c2=0.9, UNSIGNED maxLS=25, REAL4 tolX=0.000000001):=FUNCTION
	//C++ functions used
	//sum (M.*V)
	SET OF Pbblas.Types.value_t4 scale_mat (PBblas.Types.dimension_t N, Pbblas.Types.value_t4 c, PBblas.Types.matrix_t4 M ) := BEGINC++

    #body
    __lenResult = n * sizeof(float);
    __isAllResult = false;
    float * result = new float[n];
    __result = (void*) result;
    float *cellm = (float*) m;
    uint32_t i;
		for (i=0; i<n; i++){
			result[i] = cellm[i]*c;
		}

  ENDC++;
	// l*M + D
	SET OF Pbblas.Types.value_t4 summation(PBblas.types.dimension_t N, Pbblas.Types.value_t4 L, PBblas.types.matrix_t4 M, PBblas.types.matrix_t4 D) := BEGINC++

    #body
    __lenResult = n * sizeof(float);
    __isAllResult = false;
    float * result = new float[n];
    __result = (void*) result;
		float *cellm = (float*) m;
    float *celld = (float*) d;
    uint32_t i;
		for (i=0; i<n; i++) {
			result[i] = (l*cellm[i])+celld[i];
    }
		
  ENDC++;
	
	PBblas.Types.value_t4 sump(PBblas.Types.dimension_t N, PBblas.Types.matrix_t4 M, PBblas.Types.matrix_t4 V) := BEGINC++
		#body
		float result = 0;
		float tmpp ;
		float *cellm = (float*) m;
		float *cellv = (float*) v;
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
	topass_bracketing := PROJECT (g, TRANSFORM (Z4oomingRecord, SELF.zooming_cond := -1; SELF.zoomtermination := FALSE; SELF.insufProgress := FALSE; SELF.id := 1; SELF.glob_f :=LEFT.cost_value; SELF.high_t := -1 ; SELF.high_gtd := -1; SELF.high_cost_value := -1; SELF.prev_t := 0; SELF.prev_gtd := gtd; SELF.wolfe_funEvals := 0; SELF.c := 0; SELF.bracketing_cond := -1; SELF.next_t := t; SELF := LEFT), LOCAL);
	
	BracketingStep (DATASET (Z4oomingRecord) inputp, UNSIGNED coun) := FUNCTION
		//the idea is to calculate the new values at the begining of the bracket and then JOIN the loop input and this new values in order to generate the loop output. The JOIN TRANSFORM contains all the condition checks
		// calculate new g and cost value at the begining of the loop using next_t value in the input dataset
		// calculate t*d
		Elem := {REAL4 v};
		Layout_Part4 td_tran (inputp le, d ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := scale_mat(cells, le.next_t, ri.mat_part);
			SELF := le;
		END;
		td := JOIN (inputp, d, LEFT.partition_id = RIGHT.partition_id, td_tran (LEFT, RIGHT), LOCAL);
		//calculate x_new = x0 + td
		Layout_Part4 x_new_tran (Layout_part4 le, Layout_part4 ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := summation(cells, 1.0, le.mat_part, ri.mat_part);
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
		arm_t_rec4 arm_init_t_tran (inputp le) := TRANSFORM
		  //t = (t + t_prev)/2;
			SELF.init_arm_t := (le.next_t + le.prev_t)/2;
			SELF := le;
		END;
		init_arm_t := PROJECT (inputp, arm_init_t_tran(LEFT), LOCAL);
		Arm_Result := A4rmijoBacktrack_fromwolfe( x,  CostFunc_params, TrainData , TrainLabel, CostFunc , init_arm_t, d,  g, gtd,  c1,  c2, tolX);
		Arm_result_zoomingformat := PROJECT (Arm_Result, TRANSFORM (Z4oomingRecord,
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
		
		Z4oomingRecord main_tran (inputp le_prev, CostGrad_new ri_new) := TRANSFORM
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
	Z4oomingRecord maxitr_tran (Z4oomingRecord le_brack , CostGrad_record4 ri_g) := TRANSFORM
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
	
	bracketing_result_ := LOOP(topass_bracketing, LEFT.bracketing_cond = -1, BracketingStep(ROWS(LEFT), COUNTER));
	//check if MAXITR is reached and decide between bracketing result and f (t=0) to be returned
	bracketing_result := JOIN (bracketing_result_ , g, LEFT.partition_id = RIGHT.partition_id, maxitr_tran (LEFT, RIGHT), LOCAL);
	topass_zooming := bracketing_result;
	
	ZoomingStep (DATASET (Z4oomingRecord) inputp, UNSIGNED coun) := FUNCTION
		//similar to BracketingStep, first we calculate the new point (for that we need to calculate the new t using interpolation)
		//after the new point is calculated we JOIN the input and the new point in order to decide which one to return
		// Compute new trial value & Test that we are making sufficient progress
		//The following transform calculates the new trial value as well as insufProgress value by doing a PROJECT on the input, the new t is then saved in next_t field
		Z4oomingRecord newt_tran (Z4oomingRecord le) := TRANSFORM
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
		Elem := {REAL4 v};
		Layout_Part4 td_tran (inputp_t_insuf le, d ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := scale_mat(cells, le.next_t, ri.mat_part);
			SELF := le;
		END;
		td := JOIN (inputp_t_insuf, d, LEFT.partition_id = RIGHT.partition_id, td_tran (LEFT, RIGHT), LOCAL);
		// calculate x_new = x0 + td
		Layout_Part4 x_new_tran (Layout_part4 le, Layout_part4 ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := summation(cells, 1.0, le.mat_part, ri.mat_part);
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
		Z4oomingRecord main_tran (inputp le_prev, CostGrad_new ri_new) := TRANSFORM
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
	RETURN zooming_result;

END;// END W4olfeLineSearch4_4_2

EXPORT MinFUNC5(DATASET(PBblas.Types.Layout_Part4) x0,DATASET(Types.NumericField4) CostFunc_params, DATASET(Layout_Cell_nid4) TrainData , DATASET(PBblas.Types.Layout_Part4) TrainLabel,DATASET(costgrad_record4) CostFunc (DATASET(PBblas.Types.Layout_Part4) x0, DATASET(Types.NumericField4) CostFunc_params, DATASET(Layout_Cell_nid4) TrainData , DATASET(PBblas.Types.Layout_Part4) TrainLabel) ) := FUNCTION
RETURN 1;
END;
EXPORT MinFUNC6(DATASET(PBblas.Types.Layout_Part4) x0 , DATASET(costgrad_record4) CostFunc (DATASET(PBblas.Types.Layout_Part4) x0 , DATASET(Layout_Cell_nid4) TrainData )) := FUNCTION
RETURN 1;
END;

  EXPORT MinFUNC4(DATASET(PBblas.Types.Layout_Part4) x0,DATASET(Types.NumericField4) CostFunc_params, DATASET(Layout_Cell_nid4) TrainData , DATASET(PBblas.Types.Layout_Part4) TrainLabel,DATASET(costgrad_record4) CostFunc (DATASET(PBblas.Types.Layout_Part4) x0, DATASET(Types.NumericField4) CostFunc_params, DATASET(Layout_Cell_nid4) TrainData , DATASET(PBblas.Types.Layout_Part4) TrainLabel), INTEGER8 param_num, UNSIGNED MaxIter = 100, REAL4 tolFun = 0.00001, REAL4 TolX = 0.000000001, UNSIGNED maxFunEvals = 1000, UNSIGNED corrections = 100, prows=0, pcols=0, Maxrows=0, Maxcols=0) := FUNCTION
		//C++ function used
		//sum(abs(M(:)))
		
		PBblas.Types.value_t4 sumabs(PBblas.Types.dimension_t N, PBblas.Types.matrix_t4 M) := BEGINC++

    #body
    float result = 0;
		float tmpp ;
    float *cellm = (float*) m;
    uint32_t i;
		for (i=0; i<n; i++) {
		tmpp = (cellm[i]>=0) ? (cellm[i]) : (-1 * cellm[i]);
      result = result + tmpp;
    }
		return(result);

   ENDC++;
	 
		PBblas.Types.value_t4 sum_sq(PBblas.Types.dimension_t N, PBblas.Types.matrix_t4 M) := BEGINC++

			#body
			float result = 0;
			float tmpp ;
			float *cellm = (float*) m;
			uint32_t i;
			for (i=0; i<n; i++) {
				result = result + (cellm[i]*cellm[i]);
			}
			return(result);

		ENDC++;
		//sum (M.*V)
		PBblas.Types.value_t4 sump(PBblas.Types.dimension_t N, PBblas.Types.matrix_t4 M, PBblas.Types.matrix_t4 V) := BEGINC++

					#body
					float result = 0;
					float tmpp ;
					float *cellm = (float*) m;
					float *cellv = (float*) v;
					uint32_t i;
					for (i=0; i<n; i++) {
						tmpp =(cellm[i] * cellv [i]);
						result = result + tmpp;
					}
					return(result);

				ENDC++;
			
		//sum(gin(:).^2)
	sum_square (DATASET(Layout_Part4) g_in) := FUNCTION
      Elem := {REAL4 v};  //short-cut record def
      Elem su(Layout_Part4 xrec) := TRANSFORM //hadamard product
        SELF.v :=  sum_sq(xrec.part_rows * xrec.part_cols, xrec.mat_part);
      END;
      ss_ := PROJECT (g_in, su (LEFT), LOCAL);
      ss := SUM (ss_, ss_.v);
      RETURN ss;
    END;//sum_square
		
	sum_abs (DATASET(Layout_Part4) g_in) := FUNCTION
      Elem := {REAL4 v};  //short-cut record def
      Elem su(Layout_Part4 xrec) := TRANSFORM //hadamard product
        SELF.v :=  sumabs(xrec.part_rows*xrec.part_cols, xrec.mat_part);
      END;
      ss_ := PROJECT (g_in, su (LEFT), LOCAL);
      ss := SUM (ss_, ss_.v);
      RETURN ss;
    END;//sum_abs
		
	SET OF Pbblas.Types.value_t4 scale_mat (PBblas.Types.dimension_t N, Pbblas.Types.value_t4 c, PBblas.Types.matrix_t4 M ) := BEGINC++

    #body
    __lenResult = n * sizeof(float);
    __isAllResult = false;
    float * result = new float[n];
    __result = (void*) result;
    float *cellm = (float*) m;
    uint32_t i;
		for (i=0; i<n; i++){
			result[i] = cellm[i]*c;
		}

  ENDC++;
	// l*M + D
	SET OF Pbblas.Types.value_t4 summation(PBblas.types.dimension_t N, Pbblas.Types.value_t4 L, PBblas.types.matrix_t4 M, PBblas.types.matrix_t4 D) := BEGINC++

    #body
    __lenResult = n * sizeof(float);
    __isAllResult = false;
    float * result = new float[n];
    __result = (void*) result;
		float *cellm = (float*) m;
    float *celld = (float*) d;
    uint32_t i;
		for (i=0; i<n; i++) {
			result[i] = (l*cellm[i])+celld[i];
    }
		
  ENDC++;
l4bfgs2 (DATASET(minfRec4) min_in, UNSIGNED rec_c) := FUNCTION
			//itr_n := MAX (min_in, min_in.update_itr);
			itr_n := rec_c;
      k := IF (itr_n >corrections, corrections, itr_n); // k is the number of previous step vectors which are already stored
			//k := itr_n;
			// q LOOP
			q_step (DATASET(minfRec4) q_inp, unsigned q_c) := FUNCTION
				q_itr := itr_n- q_c + 1;
				q := q_inp(no=1);//this is the q vector
				s_tmp := min_in (no = 3 AND update_itr = q_itr);
				y_tmp := min_in (no = 4 AND update_itr = q_itr);
				//calculate al : al(i) = ro(i)*s(:,i)'*q(:,i+1);
				simple := {REAL4 v};
				simple al_tran (minfRec4 q_in, minfRec4 s_in) := TRANSFORM
				  cells := q_in.part_rows * q_in.part_cols;
					SELF.v := (1/s_in.sty) * sump(cells, q_in.mat_part, s_in.mat_part);
				END;
				
				al_ := JOIN (q, s_tmp, LEFT.partition_id = RIGHT.partition_id, al_tran (LEFT, RIGHT), LOCAL);
				al := SUM (al_, al_.v);
				// calculate new q vector : q(:,i) = q(:,i+1)-al(i)*y(:,i);
				minfRec4 new_q_tran (minfRec4 le, minfRec4 ri):= TRANSFORM
					cells := le.part_rows * le.part_cols;
			    SELF.mat_part := summation(cells, -1 * al, le.mat_part,  ri.mat_part);
					SELF := ri;//by assigning q part (right hand side) to SELF we make sure we are keeping funevals and other information
				END;
				new_q := JOIN (y_tmp, q, LEFT.partition_id = RIGHT.partition_id, new_q_tran (LEFT, RIGHT), LOCAL);
				//normalize al to new_q and return the result
				minfRec4 norm_al (minfRec4 le) := TRANSFORM
					SELF.mat_part := [al];
					SELF.no := q_itr + 1;// 1 is added because the final q_itr will be 1 and we have already reserved no=1 for the q vector itself
					SELF := le;
				END;
				al_norm_ := NORMALIZE(new_q(no=1), 1,norm_al(LEFT));
				al_norm := ASSERT(al_norm_, node_id = Thorlib.node() and node_id=(partition_id-1), 'al is not well distributed in the lbfgs function', FAIL);
        RETURN new_q + al_norm + q_inp (no != 1);
      END; //END q_step
			minfRec4 steep_tran (minfRec4 le) := TRANSFORM
				cells := le.part_rows * le.part_cols;
				SELF.mat_part := scale_mat(cells, -1.0, le.mat_part);//multiply by -1
				SELF := le;
			END;
			topass_q := PROJECT (min_in(no=1), steep_tran(LEFT),LOCAL);//contains funevals and cost_value, h information from the previous iteration in minfunc function. So basically the h field (hdiag) belongs to what has been calculated in the previous iteratio and can be used here in topass_r
			//q_result := LOOP(topass_q, COUNTER <= k, q_step(ROWS(LEFT),COUNTER)); orig
			q_result := LOOP(topass_q, k, q_step(ROWS(LEFT),COUNTER));
			// r loop
			r_step (DATASET(minfRec4) r_inp, unsigned8 r_c) := FUNCTION
				r_itr := r_c + (itr_n - k);
				s_tmp := min_in (no = 3 AND update_itr = r_itr);
				y_tmp := min_in (no = 4 AND update_itr = r_itr);
				//calculate be be(i) = ro(i)*y(:,i)'*r(:,i);
				simple := {REAL4 v};
				simple be_tran (minfRec4 r_in, minfRec4 y_in) := TRANSFORM
				  cells := r_in.part_rows * r_in.part_cols;
					SELF.v := (1/y_in.sty) * sump(cells, r_in.mat_part, y_in.mat_part);
				END;
				be_ := JOIN (r_inp, y_tmp, LEFT.partition_id = right.partition_id, be_tran (LEFT, RIGHT), LOCAL);
				be := SUM (be_, be_.v);
				// calculate (al (i) - be ) * s (i)
				minfRec4 s_tran (minfRec4 s_in, minfRec4 al_in) := TRANSFORM
					cells := s_in.part_rows * s_in.part_cols;
					SELF.mat_part := scale_mat(cells, (al_in.mat_part[1] - be), s_in.mat_part);
					SELF := s_in;
				END;
				al_be_s := JOIN (s_tmp, q_result (no = (r_itr + 1)) , LEFT.partition_id = RIGHT. partition_id, s_tran (LEFT, RIGHT), LOCAL);
				// calculate new_r := r + al_be_s
				minfRec4 new_r_tran (minfRec4 le, minfRec4 ri):= TRANSFORM
					cells := le.part_rows * le.part_cols;
			    SELF.mat_part := summation(cells, 1.0, le.mat_part,  ri.mat_part);
					SELF := ri;//again here by assigning the r part (RIGHT) we are making sure we are keeping all the field initially comming from g (min_in (no=1)) such as funevals field and etc.
				END;
				new_r := JOIN (al_be_s, r_inp, LEFT.partition_id = RIGHT.partition_id, new_r_tran (LEFT, RIGHT), LOCAL);
				RETURN new_r; 
			END;// END r_step
			minfRec4 r_pass_tran (minfRec4 le) := TRANSFORM
				cells := le.part_rows * le.part_cols;
				SELF.mat_part := scale_mat(cells, le.h, le.mat_part);
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
    END; // END l4bfgs2
		
		
		
	
		
		//check optimality 
    //if sum(abs(g)) <= tolFun
    // optimality_check (DATASET(Layout_Part) g_in) := FUNCTION
      // ss := sum_abs(g_in);
      // RETURN ss<tolFun;
    // END;//END optimality_check
optimality_check4 (DATASET(Layout_Part4) g_in) := FUNCTION
      ss := sum_abs(g_in);
      RETURN ss<tolFun;
    END;//END optimality_check4		

		
	IsLegal4 (DATASET(costgrad_record4) inp) := FUNCTION
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
		costgrad_record4 steep_tran (costgrad_record4 le) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.mat_part := scale_mat(cells, -1, le.mat_part);//multiply by -1
			SELF := le;
		END;
		d := PROJECT (CostGrad, steep_tran (LEFT) , LOCAL);
		//check whether d is legal, if not return
		dlegalstep := IsLegal4 (d);
		// Directional Derivative : gtd = g'*d;
		//Since d = -g -> gtd = -sum (g.^2)
		Elem := {REAL4 v};
		Elem g2_tran (costgrad_record4 le) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.v := sum_sq(cells, le.mat_part);
		END;
		gtd_ := PROJECT (CostGrad, g2_tran (LEFT), LOCAL);
		gtd := -1*SUM (gtd_, gtd_.v);
		//Check that progress can be made along direction : if gtd > -tolX then break!
		gtdprogress := gtd > -1*tolX;
		// Select Initial Guess If coun =1 then t = min(1,1/sum(abs(g)));
		Elem gabs_tran (costgrad_record4 le) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.v := sumabs(cells, le.mat_part);
		END;
		sum_abs_g0_ := PROJECT (CostGrad, gabs_tran (LEFT), LOCAL);
		sum_abs_g0 := SUM (sum_abs_g0_, sum_abs_g0_.v);
		t_init := MIN ([1, 1/(sum_abs_g0)]);
		// Find Point satisfying Wolfe
		w := W4olfeLineSearch4_4_2(1, x0,CostFunc_params,TrainData, TrainLabel,CostFunc, t_init, d, CostGrad,gtd, 0.0001, 0.9, wolfe_max_itr, 0.000000001);
		//calculate new oint
		//calculate td
		minfRec4 td_tran (w le, d ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := scale_mat(cells, le.prev_t, ri.mat_part);
			SELF.no := 3;
			SELF.h := -1; //not calculated yet
			SELF.min_funEval := funEvals + le.wolfe_funevals;
			SELF.break_cond := -1;
			SELF.sty := -1.0;// not calculated yet
			SELF.update_itr := -1; // not calculated yet
			SELF.cost_value := le.cost_value;
			SELF.itr_counter := 1;
			SELF := ri;
		END;
		td := JOIN (w, d, LEFT.partition_id = RIGHT.partition_id, td_tran (LEFT, RIGHT), LOCAL);
		//calculate x_new = x0 + td
		minfRec4 x_new_tran (Layout_part4 le, minfRec4 ri) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.mat_part := summation(cells, 1.0, le.mat_part,  ri.mat_part);
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
		minfRec4 y_y_tran (costgrad_record4 le, w ri) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.mat_part := summation(cells, -1.0, le.mat_part, ri.mat_part);
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
		Elem yy2_tran (Layout_Part4 le) := TRANSFORM
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
		g0_dnotlegal_return := PROJECT (Costgrad, TRANSFORM(minfrec4, SELF.itr_counter := 1; SELF.sty := -1; SELF.update_itr := 0; SELF.h := Hdiag; SELF.break_cond := 1, SELF.no := 1; SELF.min_funEval:= funEvals; SELF:= LEFT ), LOCAL);
		x0_dnotlegal_return := PROJECT (x0,       TRANSFORM(minfrec4, SELF.itr_counter := 1; SELF.sty := -1; SELF.update_itr := 0; SELF.h := Hdiag; SELF.break_cond := 1, SELF.no := 2; SELF.min_funEval:= funEvals; SELF.cost_value := -1; SELF:= LEFT ), LOCAL);//cost_value for x (when no =2 ) does not mean any thing, the correct cost) value is shown when no=1 (along with gradient value)
		dnotlegal_return := g0_dnotlegal_return + x0_dnotlegal_return;
		//After gtd is calculated check whether progress along the direction is possible, if it is not, break
		g0_noprogalong_return := PROJECT (Costgrad, TRANSFORM(minfrec4, SELF.itr_counter := 1; SELF.sty := -1; SELF.update_itr := 0; SELF.h := Hdiag; SELF.break_cond := 2, SELF.no := 1; SELF.min_funEval:= funEvals; SELF:= LEFT ), LOCAL);
		x0_noprogalong_return := PROJECT (x0,       TRANSFORM(minfrec4, SELF.itr_counter := 1; SELF.sty := -1; SELF.update_itr := 0; SELF.h := Hdiag; SELF.break_cond := 2, SELF.no := 2; SELF.min_funEval:= funEvals; SELF.cost_value := -1; SELF:= LEFT ), LOCAL);//cost_value for x (when no =2 ) does not mean any thing, the correct cost) value is shown when no=1 (along with gradient value)
		noprogalong_return := g0_noprogalong_return + x0_noprogalong_return;
		x_new_break := PROJECT (x_new, TRANSFORM (minfRec4, SELF.itr_counter := 1; SELF.break_cond := minfunc_cond; SELF := LEFT) ,LOCAL); // if after calculating new point and calculating x_new and g_new one of the conditions is satisfied we return and no need to calculate s ad y and updated_hdiag fo rthe next itr
		g_new_break := PROJECT (w,TRANSFORM (minfRec4, SELF.itr_counter := 1; SELF.min_funEval := funEvals + LEFT.wolfe_funevals; SELF.break_cond := minfunc_cond; SELF.sty := -1; SELF.update_itr := 0;  SELF.no :=1; SELF.h := Hdiag; SELF := LEFT) ,LOCAL);
		break_result := x_new_break + g_new_break;
		x_new_nextitr := PROJECT (x_new, TRANSFORM (minfRec4, SELF.itr_counter := 1; SELF.update_itr := IF (update_cond,1,0); SELF.h := IF (update_cond, hdiag_updated, Hdiag); SELF.break_cond := minfunc_cond; SELF := LEFT) ,LOCAL);// IF (minfunc_cond=-1,hdiag_updated, -1) : if a breack condition is satisfied, then there is not going to be a next iteration, so no need to calculate updated hidiag
		g_new_nextitr := PROJECT (w,TRANSFORM (minfRec4, SELF.itr_counter := 1; SELF.min_funEval := funEvals + LEFT.wolfe_funevals; SELF.break_cond := minfunc_cond; SELF.sty := -1; SELF.update_itr := IF (update_cond,1,0);  SELF.no :=1; SELF.h := IF (update_cond, hdiag_updated, Hdiag); SELF := LEFT) ,LOCAL);
		g_x_nextitr := x_new_nextitr + g_new_nextitr;
		s_s_corr := PROJECT (s_s,TRANSFORM (minfRec4, SELF.itr_counter := 1; SELF.sty := y_s; SELF.h := hdiag_updated; SELF.update_itr := 1; SELF := LEFT) ,LOCAL);//update_itr:=1 becasue if we are adding s_s_corr to the loop data it means the condition has been accepted
		y_y_corr := PROJECT (y_y,TRANSFORM (minfRec4, SELF.itr_counter := 1; SELF.sty := y_s; SELF.h := hdiag_updated; SELF.update_itr := 1; SELF := LEFT) ,LOCAL);//update_itr:=1 becasue if we are adding s_s_corr to the loop data it means the condition has been accepted
		// IF one of the break conditions is satisfied and minfunc!=-1 we need to return only x_new and g_new and no need to calculated Hdiag, s, y fo rthe next iteration
		//Conversly, if no condition is satisfied, it means we are going to the next iteration so we need to update Hdiag, s and y if and only if update_cond is satisfied, otherwise no update is done and we just return the current values to the next iteration
		TheReturn := IF (minfunc_cond = -1, IF (update_cond, g_x_nextitr + s_s_corr + y_y_corr, g_x_nextitr), break_result);
		Rresult := IF (dlegalstep, IF (gtdprogress, noprogalong_return, TheReturn) , dnotlegal_return);
		RETURN Rresult;
		//RETURN g_x_nextitr + s_s_corr + y_y_corr;
	 END;// END min_step_firstitr

 min_step_firstitr_test := FUNCTION
		//calculate d
		//Steepest_Descent
		costgrad_record4 steep_tran (costgrad_record4 le) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.mat_part := scale_mat(cells, -1, le.mat_part);//multiply by -1
			SELF := le;
		END;
		d := PROJECT (CostGrad, steep_tran (LEFT) , LOCAL);
		//check whether d is legal, if not return
		dlegalstep := IsLegal4 (d);
		// Directional Derivative : gtd = g'*d;
		//Since d = -g -> gtd = -sum (g.^2)
		Elem := {REAL4 v};
		Elem g2_tran (costgrad_record4 le) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.v := sum_sq(cells, le.mat_part);
		END;
		gtd_ := PROJECT (CostGrad, g2_tran (LEFT), LOCAL);
		gtd := -1.0*SUM (gtd_, gtd_.v);
		//Check that progress can be made along direction : if gtd > -tolX then break!
		gtdprogress := gtd > -1*tolX;
		// Select Initial Guess If coun =1 then t = min(1,1/sum(abs(g)));
		Elem gabs_tran (costgrad_record4 le) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.v := sumabs(cells, le.mat_part);
		END;
		sum_abs_g0_ := PROJECT (CostGrad, gabs_tran (LEFT), LOCAL);
		sum_abs_g0 := SUM (sum_abs_g0_, sum_abs_g0_.v);
		// sum_abs_g0 := (REAL4)SUM (sum_abs_g0_, sum_abs_g0_.v);
		t_init := MIN ([1, 1/(sum_abs_g0)]);
		// t_init := (REAL4)((REAL4)1.0/(sum_abs_g0)); before
		// t_init := (REAL4)((REAL4)1.0/(real4)(939578.9));
		// Find Point satisfying Wolfe
		w := W4olfeLineSearch4_4_2_test(1, x0,CostFunc_params,TrainData, TrainLabel,CostFunc, t_init, d, CostGrad,gtd, 0.0001, 0.9, wolfe_max_itr, 0.000000001);
		/*
		//calculate new oint
		//calculate td
		minfRec4 td_tran (w le, d ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := scale_mat(cells, le.prev_t, ri.mat_part);
			SELF.no := 3;
			SELF.h := -1; //not calculated yet
			SELF.min_funEval := funEvals + le.wolfe_funevals;
			SELF.break_cond := -1;
			SELF.sty := -1.0;// not calculated yet
			SELF.update_itr := -1; // not calculated yet
			SELF.cost_value := le.cost_value;
			SELF.itr_counter := 1;
			SELF := ri;
		END;
		td := JOIN (w, d, LEFT.partition_id = RIGHT.partition_id, td_tran (LEFT, RIGHT), LOCAL);
		//calculate x_new = x0 + td
		minfRec4 x_new_tran (Layout_part4 le, minfRec4 ri) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.mat_part := summation(cells, 1.0, le.mat_part,  ri.mat_part);
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
		minfRec4 y_y_tran (costgrad_record4 le, w ri) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.mat_part := summation(cells, -1.0, le.mat_part, ri.mat_part);
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
		Elem yy2_tran (Layout_Part4 le) := TRANSFORM
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
		g0_dnotlegal_return := PROJECT (Costgrad, TRANSFORM(minfrec4, SELF.itr_counter := 1; SELF.sty := -1; SELF.update_itr := 0; SELF.h := Hdiag; SELF.break_cond := 1, SELF.no := 1; SELF.min_funEval:= funEvals; SELF:= LEFT ), LOCAL);
		x0_dnotlegal_return := PROJECT (x0,       TRANSFORM(minfrec4, SELF.itr_counter := 1; SELF.sty := -1; SELF.update_itr := 0; SELF.h := Hdiag; SELF.break_cond := 1, SELF.no := 2; SELF.min_funEval:= funEvals; SELF.cost_value := -1; SELF:= LEFT ), LOCAL);//cost_value for x (when no =2 ) does not mean any thing, the correct cost) value is shown when no=1 (along with gradient value)
		dnotlegal_return := g0_dnotlegal_return + x0_dnotlegal_return;
		//After gtd is calculated check whether progress along the direction is possible, if it is not, break
		g0_noprogalong_return := PROJECT (Costgrad, TRANSFORM(minfrec4, SELF.itr_counter := 1; SELF.sty := -1; SELF.update_itr := 0; SELF.h := Hdiag; SELF.break_cond := 2, SELF.no := 1; SELF.min_funEval:= funEvals; SELF:= LEFT ), LOCAL);
		x0_noprogalong_return := PROJECT (x0,       TRANSFORM(minfrec4, SELF.itr_counter := 1; SELF.sty := -1; SELF.update_itr := 0; SELF.h := Hdiag; SELF.break_cond := 2, SELF.no := 2; SELF.min_funEval:= funEvals; SELF.cost_value := -1; SELF:= LEFT ), LOCAL);//cost_value for x (when no =2 ) does not mean any thing, the correct cost) value is shown when no=1 (along with gradient value)
		noprogalong_return := g0_noprogalong_return + x0_noprogalong_return;
		x_new_break := PROJECT (x_new, TRANSFORM (minfRec4, SELF.itr_counter := 1; SELF.break_cond := minfunc_cond; SELF := LEFT) ,LOCAL); // if after calculating new point and calculating x_new and g_new one of the conditions is satisfied we return and no need to calculate s ad y and updated_hdiag fo rthe next itr
		g_new_break := PROJECT (w,TRANSFORM (minfRec4, SELF.itr_counter := 1; SELF.min_funEval := funEvals + LEFT.wolfe_funevals; SELF.break_cond := minfunc_cond; SELF.sty := -1; SELF.update_itr := 0;  SELF.no :=1; SELF.h := Hdiag; SELF := LEFT) ,LOCAL);
		break_result := x_new_break + g_new_break;
		x_new_nextitr := PROJECT (x_new, TRANSFORM (minfRec4, SELF.itr_counter := 1; SELF.update_itr := IF (update_cond,1,0); SELF.h := IF (update_cond, hdiag_updated, Hdiag); SELF.break_cond := minfunc_cond; SELF := LEFT) ,LOCAL);// IF (minfunc_cond=-1,hdiag_updated, -1) : if a breack condition is satisfied, then there is not going to be a next iteration, so no need to calculate updated hidiag
		g_new_nextitr := PROJECT (w,TRANSFORM (minfRec4, SELF.itr_counter := 1; SELF.min_funEval := funEvals + LEFT.wolfe_funevals; SELF.break_cond := minfunc_cond; SELF.sty := -1; SELF.update_itr := IF (update_cond,1,0);  SELF.no :=1; SELF.h := IF (update_cond, hdiag_updated, Hdiag); SELF := LEFT) ,LOCAL);
		g_x_nextitr := x_new_nextitr + g_new_nextitr;
		s_s_corr := PROJECT (s_s,TRANSFORM (minfRec4, SELF.itr_counter := 1; SELF.sty := y_s; SELF.h := hdiag_updated; SELF.update_itr := 1; SELF := LEFT) ,LOCAL);//update_itr:=1 becasue if we are adding s_s_corr to the loop data it means the condition has been accepted
		y_y_corr := PROJECT (y_y,TRANSFORM (minfRec4, SELF.itr_counter := 1; SELF.sty := y_s; SELF.h := hdiag_updated; SELF.update_itr := 1; SELF := LEFT) ,LOCAL);//update_itr:=1 becasue if we are adding s_s_corr to the loop data it means the condition has been accepted
		// IF one of the break conditions is satisfied and minfunc!=-1 we need to return only x_new and g_new and no need to calculated Hdiag, s, y fo rthe next iteration
		//Conversly, if no condition is satisfied, it means we are going to the next iteration so we need to update Hdiag, s and y if and only if update_cond is satisfied, otherwise no update is done and we just return the current values to the next iteration
		TheReturn := IF (minfunc_cond = -1, IF (update_cond, g_x_nextitr + s_s_corr + y_y_corr, g_x_nextitr), break_result);
		Rresult := IF (dlegalstep, IF (gtdprogress, noprogalong_return, TheReturn) , dnotlegal_return);
		*/
		RETURN w;
		//RETURN g_x_nextitr + s_s_corr + y_y_corr;
	 END;// END min_step_firstitr_test

   
	
	 min_step2 (DATASET(minfRec4) inp,unsigned8 min_c) := FUNCTION
		g_pre := inp (no = 1);
		x_pre := inp (no = 2);
		//calculate d
		upitr := MAX (inp, inp.update_itr);
		// lbfgs_d := lbfgs2 (inp, min_c);
		lbfgs_d := l4bfgs2 (inp, upitr);
		d := lbfgs_d;
		// is d legal
		dlegalstep := IsLegal4 (d);// lbfgs algorithm keeps the funevals, cost_value and other fields for final calculated d same as what it is recieved intitially (inp(no=1))
		// Directional Derivative : gtd = g_pre'*d;
		Elem := {REAL4 v};
		Elem gtd_tran(minfRec4 inrec, minfRec4 drec) := TRANSFORM //hadamard product
			cells := inrec.part_rows * inrec.part_cols;
			SELF.v :=  sump(cells, inrec.mat_part, drec.mat_part);
		END;
		gtd_ := JOIN (g_pre, d, LEFT.partition_id = RIGHT.partition_id, gtd_tran (LEFT, RIGHT), LOCAL);
		gtd := SUM (gtd_, gtd_.v);
		//Check that progress can be made along direction : if gtd > -tolX then break!
		gtdprogress := gtd > -1*tolX;
		// Find Point satisfying Wolfe
		t_init := 1;
		w := W4olfeLineSearch4_4_2(1, x_pre, CostFunc_params, TrainData, TrainLabel,CostFunc, t_init, d, g_pre, gtd, 0.0001, 0.9, wolfe_max_itr, 0.000000001);
		//w := ASSERT(w_, EXISTS(w_), 'w has zero rows', FAIL);
		//calculate new oint
		//calculate td
		minfRec4 td_tran (w le, d ri) := TRANSFORM
			cells := ri.part_rows * ri.part_cols;
			SELF.mat_part := scale_mat(cells, le.prev_t, ri.mat_part);
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
		minfRec4 x_new_tran (minfRec4 le, minfRec4 ri) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.mat_part := summation(cells, 1.0, le.mat_part,  ri.mat_part);
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
		minfRec4 y_y_tran (minfRec4 le, w ri) := TRANSFORM
			cells := le.part_rows * le.part_cols;
			SELF.mat_part := summation(cells, -1.0, le.mat_part, ri.mat_part);
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
		Elem yy2_tran (Layout_Part4 le) := TRANSFORM
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
		dnotlegal_return := PROJECT (inp (no =1 OR no =2), TRANSFORM(minfrec4, SELF.itr_counter := min_c + 1; SELF.break_cond := 1; SELF:= LEFT ), LOCAL);
		//After gtd is calculated check whether progress along the direction is possible, if it is not, the current g and x values recived in the loop input with break_cond updated 
		noprogalong_return := PROJECT (inp (no =1 OR no =2), TRANSFORM(minfrec4, SELF.itr_counter := min_c + 1; SELF.break_cond := 2; SELF:= LEFT ), LOCAL);
		// if d is legal and progress along direction is allowed, then calculate new point using wolfe algorithm. Next check for minfunc termination conditions (break conditions) if any of them is satisfied
		//no need to calculate new hdiag , s and y values, just retunr new calculated x and g and return with break_cond updated
		x_new_break := PROJECT (x_new, TRANSFORM (minfRec4, SELF.itr_counter := min_c + 1; SELF.break_cond := minfunc_cond; SELF := LEFT) ,LOCAL);
		//when passing g_new to wolfe function we lost the funevals, h and other values, so here we need to JOIN g_new with x_new in order to get those values back
		//g_new_break := PROJECT (w,TRANSFORM (minfRec, SELF.min_funEval := -1; SELF.break_cond := minfunc_cond; SELF.sty := -1; SELF.update_itr := -1;  SELF.no :=1; SELF.h := -1; SELF := LEFT) ,LOCAL);
		g_new_break := JOIN (w, x_new_break, LEFT.partition_id = RIGHT.partition_id, TRANSFORM (minfRec4, SELF.itr_counter := min_c + 1; SELF.min_funEval := RIGHT.min_funEval;  SELF.break_cond := minfunc_cond; SELF.sty := -1; SELF.update_itr := RIGHT.update_itr;  SELF.no :=1; SELF.h := RIGHT.h; SELF:= LEFT), LOCAL);
		break_result := x_new_break + g_new_break;
		// id d is legal, progress along direction is allowed, no break condition is satified then return x_new, g_new values along with newly calculated Hdiag and s and y value only if update_cond is satified
		x_new_nextitr := PROJECT (x_new, TRANSFORM (minfRec4, SELF.itr_counter := min_c + 1; SELF.update_itr := IF (update_cond, LEFT.update_itr + 1, LEFT.update_itr); SELF.h := IF (update_cond, hdiag_updated, LEFT.h); SELF.break_cond := minfunc_cond; SELF := LEFT) ,LOCAL);
		g_new_nextitr := JOIN (w, x_new_nextitr, LEFT.partition_id = RIGHT.partition_id, TRANSFORM (minfRec4, SELF.itr_counter := min_c + 1; SELF.min_funEval := RIGHT.min_funEval;  SELF.break_cond := minfunc_cond; SELF.sty := -1; SELF.update_itr := RIGHT.update_itr;  SELF.no :=1; SELF.h := RIGHT.h; SELF:= LEFT), LOCAL);
		g_x_nextitr := x_new_nextitr + g_new_nextitr + inp ((no = 3 OR no = 4) AND update_itr > (min_c+1-corrections));// retunr new_x + new_g + the n=correction recent correction vectors, the old correction vectors should not be returned -> limited memory
		s_s_corr := PROJECT (s_s,TRANSFORM (minfRec4, SELF.itr_counter := min_c + 1; SELF.sty := y_s; SELF.h := hdiag_updated; SELF.update_itr := LEFT.update_itr + 1; SELF := LEFT) ,LOCAL);//update_itr:=1 becasue if we are adding s_s_corr to the loop data it means the condition has been accepted
		y_y_corr := PROJECT (y_y,TRANSFORM (minfRec4, SELF.itr_counter := min_c + 1; SELF.sty := y_s; SELF.h := hdiag_updated; SELF.update_itr := LEFT.update_itr + 1; SELF := LEFT) ,LOCAL);//update_itr:=1 becasue if we are adding s_s_corr to the loop data it means the condition has been accepted
		// IF one of the break conditions is satisfied and minfunc!=-1 we need to return only x_new and g_new and no need to calculated Hdiag, s, y fo rthe next iteration
		//Conversly, if no condition is satisfied, it means we are going to the next iteration so we need to update Hdiag, s and y if and only if update_cond is satisfied, otherwise no update is done and we just return the current values to the next iteration
		TheReturn := IF (minfunc_cond = -1, IF (update_cond, g_x_nextitr + s_s_corr + y_y_corr, g_x_nextitr), break_result);
		Rresult := IF (dlegalstep, IF (gtdprogress, noprogalong_return, TheReturn) , dnotlegal_return);
		RETURN Rresult;
		//RETURN PROJECT (w, TRANSFORM(minfRec, SELF.sty:=LEFT.prev_t; SELF.break_cond := LEFT.bracketing_cond; SELF.h := -1; SELF.update_itr := 0; SELF.no := 10; SELF.min_funeval := LEFT.wolfe_funevals;SELF:= LEFT), LOCAL);
		//RETURN g_x_nextitr + s_s_corr + y_y_corr;
 END;//END min_step2
 

 //RETURN lbfgs(min_step_firstitr);
 //RETURN min_step_firstitr;
 m1 := min_step_firstitr;
 
	m9 := LOOP(m1, 9, min_step2(ROWS(LEFT),COUNTER));
	
 // RETURN LOOP (m1, LEFT.break_cond = -1 , min_step2(ROWS(LEFT),COUNTER)); //orig

 RETURN min_step_firstitr_test;
// LOOP(topass_zooming, (LEFT.zoomtermination = FALSE) AND (LEFT.c < (maxLS+1)), ZoomingStep(ROWS(LEFT), COUNTER));

  END;// END MinFUNC4


END;// END Optimization

// in lbfgs2 k should be max(update_itr)
// f-fpre and one other condition should be added to minfunc
// program root and poly in c++
// The reason calculated cost value is not deterministic is that whenever there is a distribution followed by a roll up the results are not deterministic. Because distribution distributes record in different orders and adding floqting values a+b+c results in different value as adding c+a+b
// an example of this is d3a2t function in deep learning MODULE where the calclation of final results is not determinitstic due to roll up after a distribution
//I output the same mul_part_dist_sorted out multiple times, each time the order of records are different 
// in 
//make sure all -1 are asigned to int values, not unsigned value
//check for no double
// not real8, no Layout_Part
//no BLAS.
//check everything is unsigned, no unsigned4
//check integers and make sure they are integer because of -1
//in polyinterp_both functions, the function expecting REAL8 but the input is REAL4, is that ok?or do I need casting?
