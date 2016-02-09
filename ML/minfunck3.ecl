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