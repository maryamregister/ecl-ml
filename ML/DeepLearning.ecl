﻿IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat;
IMPORT * FROM ML.Types;
IMPORT PBblas;
Layout_Cell := PBblas.Types.Layout_Cell;
Layout_Part := PBblas.Types.Layout_Part;

EXPORT DeepLearning := MODULE
EXPORT Sparse_Autoencoder_IntWeights (INTEGER4 NumberofFeatures, INTEGER4 NumberofHiddenLayerNodes) := FUNCTION
  net := DATASET([
  {1, 1, NumberofFeatures},
  {2,1,NumberofHiddenLayerNodes},
  {3,1,NumberofFeatures}],
  Types.DiscreteField);
  RETURN NeuralNetworks(net).IntWeights;
END;
EXPORT Sparse_Autoencoder_IntBias (INTEGER4 NumberofFeatures, INTEGER4 NumberofHiddenLayerNodes) := FUNCTION
  net := DATASET([
  {1, 1, NumberofFeatures},
  {2,1,NumberofHiddenLayerNodes},
  {3,1,NumberofFeatures}],
  Types.DiscreteField);
  RETURN NeuralNetworks(net).IntBias;
END;
//Implementation of the Sparse Autoencoder based on the stanford Deep Learning tutorial
//beta: weight of sparsity penalty term
//sparsityParam: The desired average activation for the hidden units
EXPORT Sparse_Autoencoder (DATASET(Mat.Types.MUElement) IntW, DATASET(Mat.Types.MUElement) Intb,REAL8 BETA, REAL8 sparsityParam , REAL8 LAMBDA=0.001, REAL8 ALPHA=0.1, UNSIGNED2 MaxIter=100,
  UNSIGNED4 prows=0, UNSIGNED4 pcols=0, UNSIGNED4 Maxrows=0, UNSIGNED4 Maxcols=0) := MODULE
  //this is a un-supervised learning algorithm, no need for the labled data
  SHARED SA(DATASET(Types.NumericField) X) := MODULE
    dt := Types.ToMatrix (X);
    dTmp := dt;
    SHARED d := Mat.Trans(dTmp); //in the entire of the calculations we work with the d matrix that each sample is presented in one column
    SHARED m := MAX (d, d.y); //number of samples
    SHARED m_1 := 1/m;
    SHARED sparsityParam_ := -1*sparsityParam;
    SHARED sparsityParam_1 := 1-sparsityParam;
    SHARED sizeRec := RECORD
      PBblas.Types.dimension_t m_rows;
      PBblas.Types.dimension_t m_cols;
      PBblas.Types.dimension_t f_b_rows;
      PBblas.Types.dimension_t f_b_cols;
    END;
   //Map for Matrix d.
    SHARED havemaxrow := maxrows > 0;
    SHARED havemaxcol := maxcols > 0;
    SHARED havemaxrowcol := havemaxrow and havemaxcol;
    SHARED dstats := Mat.Has(d).Stats;
    SHARED d_n := dstats.XMax;
    SHARED d_m := dstats.YMax;
    SHARED output_num := d_n;
    derivemap := IF(havemaxrowcol, PBblas.AutoBVMap(d_n, d_m,prows,pcols,maxrows, maxcols),
                   IF(havemaxrow, PBblas.AutoBVMap(d_n, d_m,prows,pcols,maxrows),
                      IF(havemaxcol, PBblas.AutoBVMap(d_n, d_m,prows,pcols,,maxcols),
                      PBblas.AutoBVMap(d_n, d_m,prows,pcols))));
    SHARED sizeTable := DATASET([{derivemap.matrix_rows,derivemap.matrix_cols,derivemap.part_rows(1),derivemap.part_cols(1)}], sizeRec);
    //Create block matrix d
    dmap := PBblas.Matrix_Map(sizeTable[1].m_rows,sizeTable[1].m_cols,sizeTable[1].f_b_rows,sizeTable[1].f_b_cols);
    ddist := DMAT.Converted.FromElement(d,dmap);
    //Create block matrix Ytmp
    Ymap := dmap;
    Ydist := ddist;
    //Creat block matrices for weights
    w1_mat := Mat.MU.From(IntW,1);
    w1_mat_x := Mat.Has(w1_mat).Stats.Xmax;
    w1_mat_y := Mat.Has(w1_mat).Stats.Ymax;
    w1map := PBblas.Matrix_Map(w1_mat_x, w1_mat_y, sizeTable[1].f_b_rows, sizeTable[1].f_b_rows);
    w1dist := DMAT.Converted.FromElement(w1_mat,w1map);
    w2_mat := Mat.MU.From(IntW,2);
    w2_mat_x := w1_mat_y;
    w2_mat_y := w1_mat_x;
    w2map := PBblas.Matrix_Map(w2_mat_x, w2_mat_y, sizeTable[1].f_b_rows, sizeTable[1].f_b_rows);
    w2dist := DMAT.Converted.FromElement(w2_mat,w2map);
    //each bias vector is converted to block format
    b1vec := Mat.MU.From(Intb,1);
    b1vec_x := Mat.Has(b1vec).Stats.Xmax;
    b1vecmap := PBblas.Matrix_Map(b1vec_x, 1, sizeTable[1].f_b_rows, 1);
    b1vecdist := DMAT.Converted.FromElement(b1vec,b1vecmap);
    b2vec := Mat.MU.From(Intb,2);
    b2vec_x := Mat.Has(b2vec).Stats.Xmax;
    b2vecmap := PBblas.Matrix_Map(b2vec_x, 1, sizeTable[1].f_b_rows, 1);
    b2vecdist := DMAT.Converted.FromElement(b2vec,b2vecmap);

    //functions used
    PBblas.Types.value_t sp_reci(PBblas.Types.value_t v,PBblas.Types.dimension_t r,PBblas.Types.dimension_t c) := sparsityParam_/v;
    PBblas.Types.value_t sp_1_reci(PBblas.Types.value_t v,PBblas.Types.dimension_t r,PBblas.Types.dimension_t c) := sparsityParam_1/(1-v);
    //sparsity_delta=((-sparsityParam./rhohat)+((1-sparsityParam)./(1.-rhohat)));
    PBblas.Types.value_t sp_delta(PBblas.Types.value_t v,PBblas.Types.dimension_t r,PBblas.Types.dimension_t c) := (sparsityParam_/v)+(sparsityParam_1/(1-v));
    PBblas.Types.value_t siggrad(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := v*(1.0-v);
    PBblas.Types.value_t sigmoid(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := 1/(1+exp(-1*v));
    //maps used
    b1map := PBblas.Matrix_Map(b1vec_x, m, sizeTable[1].f_b_rows, sizeTable[1].f_b_cols);
    b2map := PBblas.Matrix_Map(b2vec_x, m, sizeTable[1].f_b_rows, sizeTable[1].f_b_cols);
    a2map := b1map;
    a3map := b2map;
    HL_nodes := w1_mat_x;//number of nodes in the hidden layer
    Hiddmap := b1vecmap;
    //onevec for calculating rhohat
    Ones_VecMap := PBblas.Matrix_Map(m, 1, sizeTable[1].f_b_cols, 1);
    //New Vector Generator
    Layout_Cell gen(UNSIGNED4 c, UNSIGNED4 NumRows) := TRANSFORM
      SELF.x := ((c-1) % NumRows) + 1;
      SELF.y := ((c-1) DIV NumRows) + 1;
      SELF.v := 1;
    END;
    //Create Ones Vector for the calculations in the step fucntion
    Ones_Vec := DATASET(m, gen(COUNTER, m),DISTRIBUTED);
    Ones_Vecdist := DMAT.Converted.FromCells(Ones_VecMap, Ones_Vec);
    //FF2 returns a2
    FF2(DATASET(Layout_Part) w1, DATASET(Layout_Part) b1v):= FUNCTION
      //b1m = repmat(b1v,1,m)
      b1m := PBblas.PB_dgemm(FALSE, TRUE, 1.0,b1vecmap, b1v, Ones_VecMap, Ones_Vecdist, b1map);
      //z2 = w1*X+b1;
      z2 := PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1, dmap, ddist, b1map, b1m, 1.0);
      //a2 = sigmoid (z2);
      a2 := PBblas.Apply2Elements(b1map, z2, sigmoid);
      RETURN a2;
    END;//END FF2
    //FF3 returns a3
    FF3(DATASET(Layout_Part) w2,DATASET(Layout_Part) b2v, DATASET(Layout_Part) a2 ):= FUNCTION
      //b2m = repmat(b2v,1,m)
      b2m := PBblas.PB_dgemm(FALSE, TRUE, 1.0,b2vecmap, b2v, Ones_VecMap, Ones_Vecdist, b2map);
      //z3 = w2*a2+b2;
      //z3 := PBblas.PB_dgemm(FALSE, FALSE,1.0,w2map, w2, a2map, a2, b2map,b2m, 1.0);
      z3_tmp := PBblas.PB_dgemm(FALSE, FALSE,1.0,w2map, w2, a2map, a2, b2map);
      z3 := PBblas.PB_daxpy(1.0, z3_tmp, b2m);
      //a3 = sigmoid (z3)
      a3 := PBblas.Apply2Elements(b2map, z3, sigmoid);
      RETURN a3;
    END;//END FF3
    //DELTA3 returns d3
    DELTA3 (DATASET(Layout_Part) a3 ) := FUNCTION
      //calculate delta for the last layer (3rd layer)
      //y=X;
      //d3=-(y-a3).*(a3.*(1-a3));
      siggrad_a3 := PBblas.Apply2Elements(a3map, a3, siggrad);
      a3_y := PBblas.PB_daxpy(-1, ddist, a3);
      d3 := PBblas.HadamardProduct(a3map, a3_y, siggrad_a3);
      RETURN d3 ;
    END;//END DELTA3
    //DELTA2 retunrs d2
    DELTA2 (DATASET(Layout_Part) w2, DATASET(Layout_Part) a2, DATASET(Layout_Part) d3) := FUNCTION
      //calculate delta for 2nd layer
      //rhohat=mean(a2,2);
      //sparsity_delta=((-sparsityParam./rhohat)+((1-sparsityParam)./(1.-rhohat)));
      //d2=((W2'*d3)+beta*repmat(sparsity_delta,1,m)).*(a2.*(1-a2));
      rhohat := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a2map, a2, Ones_VecMap, Ones_Vecdist, Hiddmap);
      sparsity_delta := PBblas.Apply2Elements(Hiddmap, rhohat, sp_delta);
      siggrad_a2 := PBblas.Apply2Elements(a2map, a2, siggrad);
      repmat_sparsity_delta := PBblas.PB_dgemm(FALSE, TRUE, 1.0,  Hiddmap, sparsity_delta, Ones_VecMap, Ones_Vecdist, a2map);
      //d2_firstterm = (W2'*d3)+beta*repmat(sparsity_delta,1,m);
      d2_firstterm := PBblas.PB_dgemm(TRUE, FALSE, 1.0, w2map, w2, a3map, d3, a2map, repmat_sparsity_delta, BETA);
      d2 := PBblas.HadamardProduct(a2map, d2_firstterm, siggrad_a2);
      RETURN d2 ;
    END;
    //WeightGrad1 returns gradient for w1
    WeightGrad1 (DATASET(Layout_Part) w1, DATASET(Layout_Part) d2) := FUNCTION
      w1_g := PBblas.PB_dgemm(FALSE, TRUE, m_1, a2map, d2, dmap, ddist, w1map, w1 ,LAMBDA );
      RETURN w1_g;
    END;
    //WeightGrad2 returns gradient for w2
    WeightGrad2 (DATASET(Layout_Part) w2, DATASET(Layout_Part) d3, DATASET(Layout_Part) a2) := FUNCTION
      w2_g := PBblas.PB_dgemm(FALSE, TRUE, m_1, a3map, d3, a2map, a2, w2map, w2 ,LAMBDA );
      RETURN w2_g;
    END;
    //BiasGrad1 calculates the bias gradients for b1
    BiasGrad1 (DATASET(Layout_Part) d2) := FUNCTION
      b1_g := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a2map, d2, Ones_VecMap, Ones_Vecdist, b1vecmap);
      RETURN b1_g;
    END;
    //BiasGrad2 calculates the bias gradients for b2
    BiasGrad2 (DATASET(Layout_Part) d3) := FUNCTION
      b2_g := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a3map, d3, Ones_VecMap, Ones_Vecdist, b2vecmap);
      RETURN b2_g;
    END;
    GradDesUpdate (DATASET(Layout_Part) tobeUpdated, DATASET(Layout_Part) GradDesTerm):= FUNCTION
      tmp_updated := PBblas.PB_daxpy(-1, PBblas.PB_dscal(ALPHA, GradDesTerm), tobeUpdated);
      RETURN tmp_updated;
    END;
    GradDesLoop (DATASET(Layout_Part) w1in, DATASET(Layout_Part) w2in, DATASET(Layout_Part) bvec1in, DATASET(Layout_Part) bvec2in):= FUNCTION
      w1inno := PBblas.MU.TO(w1in, 1);
      w2inno := PBblas.MU.TO(w2in, 2);
      bvec1inno := PBblas.MU.TO(bvec1in, 3);
      bvec2inno := PBblas.MU.TO(bvec2in, 4);
      prm := w1inno + w2inno + bvec1inno + bvec2inno;
      GradDesLoop_Step (DATASET(PBblas.Types.MUElement) Inputprm) := FUNCTION
        w1m := PBblas.MU.FROM (Inputprm, 1);
        w2m := PBblas.MU.FROM (Inputprm, 2);
        b1v := PBblas.MU.FROM (Inputprm, 3);
        b2v := PBblas.MU.FROM (Inputprm, 4);
        a2 := FF2 (w1m, b1v);
        a3 := FF3 (w2m, b2v, a2);
        d3 := DELTA3 (a3);
        d2 := DELTA2 (w2m, a2, d3);
        wg1 := WeightGrad1 (w1m, d2);
        wg2 := WeightGrad2 (w2m, d3, a2);
        bg1 := BiasGrad1 (d2);
        bg2 := BiasGrad2 (d3);
        w1u := GradDesUpdate (w1m, wg1);
        w2u := GradDesUpdate (w2m, wg2);
        b1u := GradDesUpdate (b1v, bg1);
        b2u := GradDesUpdate (b2v, bg2);
        w1uno := PBblas.MU.TO (w1u, 1);
        w2uno := PBblas.MU.TO (w2u, 2);
        b1uno := PBblas.MU.TO (b1u, 3);
        b2uno := PBblas.MU.TO (b2u, 4);
        // prmu := IF (coun=1, w1uno + w2uno + b1uno + b2uno,PBblas.MU.TO (a2, 2)+PBblas.MU.TO (w2m, 1)+PBblas.MU.TO (d3, 3)+PBblas.MU.TO (d2, 4));
        prmu := w1uno + w2uno + b1uno + b2uno;
        RETURN prmu;
      END;
      //finalprm := GradDesLoop_Step (prm);
      finalprm := LOOP(prm, COUNTER <= MaxIter, GradDesLoop_Step(ROWS(LEFT)));
      RETURN finalprm;
    END;//END GradDesLoop
    SAprm := GradDesLoop (w1dist, w2dist, b1vecdist, b2vecdist);// SAprm is in PBblas.Types.MUElement format convert it to
    //numericfield format
    SAprm1 := PBblas.MU.From (SAprm,1);
    SAprm2 := PBblas.MU.From (SAprm,2);
    SAprm3 := PBblas.MU.From (SAprm,3);
    SAprm4 := PBblas.MU.From (SAprm,4);
    SAprm1_mat := DMat.Converted.FromPart2Elm (SAprm1);
    SAprm2_mat := DMat.Converted.FromPart2Elm (SAprm2);
    SAprm3_mat := DMat.Converted.FromPart2Elm (SAprm3);
    SAprm4_mat := DMat.Converted.FromPart2Elm (SAprm4);
    SAprm1_mat_no := Mat.MU.TO(SAprm1_mat,1);
    SAprm2_mat_no := Mat.MU.TO(SAprm2_mat,2);
    SAprm3_mat_no := Mat.MU.TO(SAprm3_mat,3);
    SAprm4_mat_no := Mat.MU.TO(SAprm4_mat,4);
    SAprm_MUE := SAprm1_mat_no + SAprm2_mat_no + SAprm3_mat_no + SAprm4_mat_no;
    AppendID(SAprm_MUE, id, SAprm_MUE_id);
    ToField (SAprm_MUE_id, SAprm_MUE_out, id, 'x,y,value,no');
    EXPORT Mod := SAprm_MUE_out;
  END;//END SA
  EXPORT LearnC (DATASET(Types.NumericField) Indep) := SA(Indep).mod;
  EXPORT Model(DATASET(Types.NumericField) mod) := FUNCTION
    modelD_Map :=	DATASET([{'id','ID'},{'x','1'},{'y','2'},{'value','3'},{'no','4'}], {STRING orig_name; STRING assigned_name;});
    FromField(mod,Mat.Types.MUElement,dOut,modelD_Map);
    RETURN dOut;
  END;//END Model
  EXPORT SAOutput(DATASET(Types.NumericField) Indep,DATASET(Types.NumericField) mod) :=FUNCTION
    //Take the same steps in the FeedForward fucntions to calculate the output of the SparseAutoencoder
    X := Indep;
    Inputmod:= Model (mod);
    dt := Types.ToMatrix (X);
    dTmp := dt;
    d := Mat.Trans(dTmp); //in the entire of the calculations we work with the d matrix that each sample is presented in one column
    m := MAX (d, d.y); //number of samples
    sizeRec := RECORD
      PBblas.Types.dimension_t m_rows;
      PBblas.Types.dimension_t m_cols;
      PBblas.Types.dimension_t f_b_rows;
      PBblas.Types.dimension_t f_b_cols;
    END;
   //Map for Matrix d.
    havemaxrow := maxrows > 0;
    havemaxcol := maxcols > 0;
    havemaxrowcol := havemaxrow and havemaxcol;
    dstats := Mat.Has(d).Stats;
    d_n := dstats.XMax;
    d_m := dstats.YMax;
    derivemap := IF(havemaxrowcol, PBblas.AutoBVMap(d_n, d_m,prows,pcols,maxrows, maxcols),
                   IF(havemaxrow, PBblas.AutoBVMap(d_n, d_m,prows,pcols,maxrows),
                      IF(havemaxcol, PBblas.AutoBVMap(d_n, d_m,prows,pcols,,maxcols),
                      PBblas.AutoBVMap(d_n, d_m,prows,pcols))));
    sizeTable := DATASET([{derivemap.matrix_rows,derivemap.matrix_cols,derivemap.part_rows(1),derivemap.part_cols(1)}], sizeRec);
    //Create block matrix d
    dmap := PBblas.Matrix_Map(sizeTable[1].m_rows,sizeTable[1].m_cols,sizeTable[1].f_b_rows,sizeTable[1].f_b_cols);
    ddist := DMAT.Converted.FromElement(d,dmap);
    //Creat block matrices for weights
    w1_mat := Mat.MU.From(Inputmod,1);
    w1_mat_x := Mat.Has(w1_mat).Stats.Xmax;
    w1_mat_y := Mat.Has(w1_mat).Stats.Ymax;
    w1map := PBblas.Matrix_Map(w1_mat_x, w1_mat_y, sizeTable[1].f_b_rows, sizeTable[1].f_b_rows);
    w1dist := DMAT.Converted.FromElement(w1_mat,w1map);
    w2_mat := Mat.MU.From(Inputmod,2);
    w2_mat_x := w1_mat_y;
    w2_mat_y := w1_mat_x;
    w2map := PBblas.Matrix_Map(w2_mat_x, w2_mat_y, sizeTable[1].f_b_rows, sizeTable[1].f_b_rows);
    w2dist := DMAT.Converted.FromElement(w2_mat,w2map);
    //each bias vector is converted to block format
    b1vec := Mat.MU.From(Inputmod,3);
    b1vec_x := Mat.Has(b1vec).Stats.Xmax;
    b1vecmap := PBblas.Matrix_Map(b1vec_x, 1, sizeTable[1].f_b_rows, 1);
    b1vecdist := DMAT.Converted.FromElement(b1vec,b1vecmap);
    b2vec := Mat.MU.From(Inputmod,4);
    b2vec_x := Mat.Has(b2vec).Stats.Xmax;
    b2vecmap := PBblas.Matrix_Map(b2vec_x, 1, sizeTable[1].f_b_rows, 1);
    b2vecdist := DMAT.Converted.FromElement(b2vec,b2vecmap);
    //functions used
    PBblas.Types.value_t sigmoid(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := 1/(1+exp(-1*v));
    //maps used
    b1map := PBblas.Matrix_Map(b1vec_x, m, sizeTable[1].f_b_rows, sizeTable[1].f_b_cols);
    b2map := PBblas.Matrix_Map(b2vec_x, m, sizeTable[1].f_b_rows, sizeTable[1].f_b_cols);
    a2map := b1map;
    a3map := b2map;
    HL_nodes := w1_mat_x;//number of nodes in the hidden layer
    Hiddmap := b1vecmap;
    //onevec for calculating rhohat
    Ones_VecMap := PBblas.Matrix_Map(m, 1, sizeTable[1].f_b_cols, 1);
    //New Vector Generator
    Layout_Cell gen(UNSIGNED4 c, UNSIGNED4 NumRows) := TRANSFORM
      SELF.x := ((c-1) % NumRows) + 1;
      SELF.y := ((c-1) DIV NumRows) + 1;
      SELF.v := 1;
    END;
    //Create Ones Vector for the calculations in the step fucntion
    Ones_Vec := DATASET(m, gen(COUNTER, m),DISTRIBUTED);
    Ones_Vecdist := DMAT.Converted.FromCells(Ones_VecMap, Ones_Vec);
    //b1m = repmat(b1v,1,m)
    b1m := PBblas.PB_dgemm(FALSE, TRUE, 1.0,b1vecmap, b1vecdist, Ones_VecMap, Ones_Vecdist, b1map);
    //z2 = w1*X+b1;
    z2 := PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1dist, dmap, ddist, b1map, b1m, 1.0);
    //a2 = sigmoid (z2);
    a2 := PBblas.Apply2Elements(b1map, z2, sigmoid);
    //b2m = repmat(b2v,1,m)
    b2m := PBblas.PB_dgemm(FALSE, TRUE, 1.0,b2vecmap, b2vecdist, Ones_VecMap, Ones_Vecdist, b2map);
    //z3 = w2*a2+b2;
    z3 := PBblas.PB_dgemm(FALSE, FALSE,1.0,w2map, w2dist, a2map, a2, b2map, b2m, 1.0);
    //a3 = sigmoid (z3)
    a3 := PBblas.Apply2Elements(b2map, z3, sigmoid);
    a3_mat := DMat.Converted.FromPart2Elm(a3);
    Types.l_result tr(Mat.Types.Element le) := TRANSFORM
      SELF.value := le.x;
      SELF.id := le.y;
      SELF.number := 1; //number of class
      SELF.conf := le.value;
    END;
    RETURN PROJECT (a3_mat, tr(LEFT));
  END;//END SAOutput
END;//END Sparse_Autoencoder
END;//END DeepLearning