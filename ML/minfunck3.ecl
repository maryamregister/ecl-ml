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
    INTEGER1 id;
    DATASET(IdElementRec) x;
    DATASET(IdElementRec) g;
    DATASET(IdElementRec) old_steps;
    DATASET(IdElementRec) old_dirs;
    REAL8 Hdiag;
    REAL8 Cost;
    INTEGER8 funEvals_;
    DATASET(IdElementRec) d;
    REAL8 fnew_fold; //f_new-f_old in that iteration
    REAL8 t_;
    BOOLEAN dLegal;
    BOOLEAN ProgAlongDir; //Progress Along Directionno //Check that progress can be made along direction ( if gtd > -tolX)
    BOOLEAN optcond; // Check Optimality Condition
    BOOLEAN lackprog1; //Check for lack of progress 1
    BOOLEAN lackprog2; //Check for lack of progress 2
    BOOLEAN exceedfuneval; //Check for going over evaluation limit
  END;
  MinFRecord_nomat := RECORD
    INTEGER1 id;
    REAL8 Hdiag;
    REAL8 Cost;
    INTEGER8 funEvals_;
    REAL8 fnew_fold; //f_new-f_old in that iteration
    REAL8 t_;
    BOOLEAN dLegal;
    BOOLEAN ProgAlongDir; //Progress Along Directionno //Check that progress can be made along direction ( if gtd > -tolX)
    BOOLEAN optcond; // Check Optimality Condition
    BOOLEAN lackprog1; //Check for lack of progress 1
    BOOLEAN lackprog2; //Check for lack of progress 2
    BOOLEAN exceedfuneval; //Check for going over evaluation limit
  END;


  MinFRecord DeNorm_x(MinFRecord L, IdElementRec R) := TRANSFORM
    SELF.x := L.x + R;
    SELF := L;
  END;
  MinFRecord DeNorm_g(MinFRecord L, IdElementRec R) := TRANSFORM
    SELF.g := L.g + R;
    SELF := L;
  END;
  MinFRecord DeNorm_old_steps(MinFRecord L, IdElementRec R) := TRANSFORM
    SELF.old_steps := L.old_steps + R;
    SELF := L;
  END;
  MinFRecord DeNorm_old_dirs(MinFRecord L, IdElementRec R) := TRANSFORM
    SELF.old_dirs := L.old_dirs + R;
    SELF := L;
  END;
  MinFRecord DeNorm_d(MinFRecord L, IdElementRec R) := TRANSFORM
    SELF.d := L.d + R;
    SELF := L;
  END;
  
  x_ext (DATASET (MinFRecord) br) := FUNCTION
    IdElementRec NewChildren(IdElementRec R) := TRANSFORM
    SELF := R;
    END;
    NewChilds := NORMALIZE(br,LEFT.x,NewChildren(RIGHT));

    RETURN PROJECT(NewChilds, TRANSFORM(Mat.Types.Element, SELF := LEFT));
  END; // END x_ext
  g_ext (DATASET (MinFRecord) br) := FUNCTION
    IdElementRec NewChildren(IdElementRec R) := TRANSFORM
    SELF := R;
    END;
    NewChilds := NORMALIZE(br,LEFT.g,NewChildren(RIGHT));

    RETURN PROJECT(NewChilds, TRANSFORM(Mat.Types.Element, SELF := LEFT));
  END; // END g_ext
  old_steps_ext (DATASET (MinFRecord) br) := FUNCTION
    IdElementRec NewChildren(IdElementRec R) := TRANSFORM
    SELF := R;
    END;
    NewChilds := NORMALIZE(br,LEFT.old_steps,NewChildren(RIGHT));

    RETURN PROJECT(NewChilds, TRANSFORM(Mat.Types.Element, SELF := LEFT));
  END; // END old_steps_ext
  old_dirs_ext (DATASET (MinFRecord) br) := FUNCTION
    IdElementRec NewChildren(IdElementRec R) := TRANSFORM
    SELF := R;
    END;
    NewChilds := NORMALIZE(br,LEFT.old_dirs,NewChildren(RIGHT));

    RETURN PROJECT(NewChilds, TRANSFORM(Mat.Types.Element, SELF := LEFT));
  END; // END old_dirs_ext
  d_ext (DATASET (MinFRecord) br) := FUNCTION
    IdElementRec NewChildren(IdElementRec R) := TRANSFORM
    SELF := R;
    END;
    NewChilds := NORMALIZE(br,LEFT.d,NewChildren(RIGHT));

    RETURN PROJECT(NewChilds, TRANSFORM(Mat.Types.Element, SELF := LEFT));
  END; // END d_ext
  BuildMinfuncData (DATASET(MinFRecord_nomat) p, DATASET (Mat.Types.Element) x_, DATASET (Mat.Types.Element) g_, DATASET (Mat.Types.Element) os, DATASET (Mat.Types.Element) od)  := FUNCTION //this function returns a MinFRecord dataset in which the four matrices are nested
      //add id to the two input matrices
      x_id := appendID2mat (x_);
      g_id := appendID2mat (g_);
      od_id := appendID2mat (od);
      os_id := appendID2mat (os);
      //load parent p with empty datasets for the dataset fields
      MinFRecord LoadParent(MinFRecord_nomat L) := TRANSFORM
        SELF.x := [];
        SELF.g := [];
        SELF.old_steps := [];
        SELF.old_dirs := [];
        SELF.d := [];
        SELF := L;
      END;

      //1 - fill in the p with Empty datasets
      p_ready := PROJECT(p,LoadParent(LEFT));
      // 2- fill in p_ready with x_
      p_x := DENORMALIZE(p_ready, x_id, LEFT.id = RIGHT.id, DeNorm_x(LEFT,RIGHT));
      // 3- fill in p_x with g_
      p_x_g := DENORMALIZE(p_x, g_id, LEFT.id = RIGHT.id, DeNorm_g(LEFT,RIGHT));
      // 4- fill p_x_g with od
      p_x_g_od := DENORMALIZE(p_x_g, od_id, LEFT.id = RIGHT.id, DeNorm_old_steps(LEFT,RIGHT));
      // 5- fill p_x_g_od with os
      p_x_g_od_os := DENORMALIZE(p_x_g_od, os_id, LEFT.id = RIGHT.id, DeNorm_old_dirs(LEFT,RIGHT));
      
      RETURN p_x_g_od_os;
    END; // END BuildMinfuncData
  
  TopassMinF_nomat := DATASET ([{GlobalID,Hdiag0,Cost0,FunEval,100 +tolFun,1,TRUE,TRUE,FALSE,FALSE,FALSE,FALSE}], MinFRecord_nomat);
  
  //ToPassMinF := DATASET ([{ML.Types.ToMatrix(x0),ML.Types.ToMatrix(g0),old_steps0,old_dir0,Hdiag0,Cost0,FunEval,emptyE,100 +tolFun,1,TRUE,TRUE,FALSE,FALSE,FALSE,FALSE}],MinFRecord);

  
  ToPassMinF := BuildMinfuncData (TopassMinF_nomat, ML.Types.ToMatrix(x0), ML.Types.ToMatrix(g0), old_steps0, old_dir0);
  
  MinFstep (DATASET (MinFRecord) inputp, INTEGER coun) := FUNCTION
    inputp1 := inputp[1];
    x_pre := x_ext(inputp);
    g_pre := g_ext(inputp);
    g_pre_ := ML.Mat.Scale (g_pre , -1);
    step_pre := old_steps_ext(inputp);
    dir_pre := old_dirs_ext(inputp);
    H_pre := inputp1.Hdiag;
    f_pre := inputp1.Cost;
    FunEval_pre := inputp1.funEvals_;
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
    // WolfeLineSearch3(INTEGER cccc, DATASET(Types.NumericField)x, REAL8 t, DATASET(Types.NumericField)d, REAL8 f, DATASET(Types.NumericField) g, REAL8 gtd, REAL8 c1=0.0001, REAL8 c2=0.9, INTEGER maxLS=25, REAL8 tolX=0.000000001,DATASET(Types.NumericField) CostFunc_params, DATASET(Types.NumericField) TrainData , DATASET(Types.NumericField) TrainLabel, DATASET(Types.NumericField) CostFunc (DATASET(Types.NumericField) x, DATASET(Types.NumericField) CostFunc_params, DATASET(Types.NumericField) TrainData , DATASET(Types.NumericField) TrainLabel), prows=0, pcols=0, Maxrows=0, Maxcols=0):=FUNCTION
    t_neworig := WolfeLineSearch3(1, ML.Types.FromMatrix(x_pre), t, d, f_pre, ML.Types.FromMatrix(g_pre), gtd[1].value, 0.0001, 0.9, 25, 0.000000001, CostFunc_params, TrainData , TrainLabel, CostFunc , prows, pcols, Maxrows, Maxcols);
    t_new := wolfe_t_ext2(t_neworig);
    g_Next := wolfe_gnew_ext2(t_neworig);
    Cost_Next := wolfe_fnew_ext2(t_neworig);
    FunEval_Wolfe := wolfe_funeval_ext2(t_neworig);
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
    MF := FUNCTION
      MinFRecord Loadinput(MinFRecord L) := TRANSFORM
        SELF.x := [];
        SELF.g := [];
        SELF.old_steps := [];
        SELF.old_dirs := [];
        SELF.Hdiag := H_Next;
        SELF.Cost := Cost_Next;
        SELF.funEvals_ := FunEval_next;
        SELF.d := [];
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
      //1 - fill in the inputp with Empty datasets
      p_ready := PROJECT(inputp,Loadinput(LEFT));
      //2- fill x
      p_x := DENORMALIZE(p_ready, appendID2mat (x_pre_updated), LEFT.id = RIGHT.id, DeNorm_x(LEFT,RIGHT));
      // 3- fill g
      p_x_g := DENORMALIZE(p_x, appendID2mat (g_Next), LEFT.id = RIGHT.id, DeNorm_g(LEFT,RIGHT));
      //4 fill old_steps
      p_x_g_os := DENORMALIZE(p_x_g, appendID2mat (Step_Next), LEFT.id = RIGHT.id, DeNorm_old_steps(LEFT,RIGHT));
      //5 fill old dirs
      p_x_g_os_od := DENORMALIZE(p_x_g_os, appendID2mat (Dir_Next), LEFT.id = RIGHT.id, DeNorm_old_dirs(LEFT,RIGHT));
      //6 fill d
      p_x_g_os_od_d := DENORMALIZE(p_x_g_os_od, appendID2mat (d_next), LEFT.id = RIGHT.id, DeNorm_d(LEFT,RIGHT)); 
 
      RETURN p_x_g_os_od_d;
    END; // END MF
    MF_dnotlegal := FUNCTION
      MinFRecord Loadinput(MinFRecord L) := TRANSFORM
        
        SELF.d := [];
        SELF.dLegal := FALSE;
        SELF := l;
      END;
      //1 - fill in the inputp with Empty datasets
      p_ready := PROJECT(inputp,Loadinput(LEFT));
      //6 fill d
      p_d := DENORMALIZE(p_ready, appendID2mat (d_next), LEFT.id = RIGHT.id, DeNorm_d(LEFT,RIGHT)); 
 
      RETURN p_d;
    END; // END MF_dnotlegal
   
    RETURN IF(dlegalstep,MF,MF_dnotlegal);
  END; // END MinFstep
  
  MinFstepout := LOOP(ToPassMinF, COUNTER <= 1 AND ROWS(LEFT)[1].dLegal AND ROWS(LEFT)[1].ProgAlongDir   
  AND ~ROWS(LEFT)[1].optcond AND ~ROWS(LEFT)[1].lackprog1 AND ~ROWS(LEFT)[1].lackprog2 AND ~ROWS(LEFT)[1].exceedfuneval  , MinFstep(ROWS(LEFT),COUNTER));
  
  //RETURN MinFstep(ToPassMinF,1);
  RETURN MinFstepout;
  END;//END MinFUNC3