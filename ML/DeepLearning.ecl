IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat; 
IMPORT * FROM ML.Types;
IMPORT PBblas;
Layout_Cell := PBblas.Types.Layout_Cell;
Layout_Part := PBblas.Types.Layout_Part;
emptyC := DATASET([], Types.NumericField);
SHARED emptyMUelm := DATASET([], Mat.Types.MUElement);

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
//IntW : initial weights for the SparseAutoencoder Network
//IntW includes two matrices of size Number_of_hidden_layer_nodes * Number_of_features and the size Number_of_features * Number_of_hidden_layer_nodes (with having no =1 1 and no =1 respectively)
//IntB : Initial Bias for the SparseAutoencoder Network
//IntB includes two matrices of size Number_of_hidden_layer_nodes*1 and Number_of_features*1 (with having no =1 1 and no =1 respectively)
//LAMBDA : weight decay term
//ALPHA : learning rate
//MaxIter : Maximum number of iterations
//prows, pcols, Maxrows, Maxcols for the Pbblas partitioning:
// - prows: an optional parameter used to set the number of rows in partition blocks (Should be used in conjuction with pcols)
// - pcols: an optional parameter used to set the number of cols in partition blocks (Should be used in conjuction with prows)
// - Maxrows: an optional parameter used to set maximum rows allowed per block when using AutoBVMap
// - Maxcols: an optional parameter used to set maximum cols allowed per block when using AutoBVMap
EXPORT Sparse_Autoencoder (INTEGER4 NumberofFeatures, INTEGER4 NumberofHiddenLayerNodes, UNSIGNED4 prows=0, UNSIGNED4 pcols=0, UNSIGNED4 Maxrows=0, UNSIGNED4 Maxcols=0) := MODULE
  //this is a un-supervised learning algorithm, no need for the labled data
  SHARED SA(DATASET(Types.NumericField) X, DATASET(Mat.Types.MUElement) IntW, DATASET(Mat.Types.MUElement) Intb, REAL8 BETA=0.1, REAL8 sparsityParam=0.1 , REAL8 LAMBDA=0.001, REAL8 ALPHA=0.1, UNSIGNED2 MaxIter=100) := MODULE
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
    PBblas.Types.value_t pow2(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := v*v;
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
    rohat (DATASET(Layout_Part) a2) := FUNCTION
      rh := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a2map, a2, Ones_VecMap, Ones_Vecdist, Hiddmap);
      RETURN rh;
    END;
    DELTA2 (DATASET(Layout_Part) w2, DATASET(Layout_Part) a2, DATASET(Layout_Part) d3, DATASET(Layout_Part) rhat) := FUNCTION
      //calculate delta for 2nd layer
      //rhohat=mean(a2,2);
      //sparsity_delta=((-sparsityParam./rhohat)+((1-sparsityParam)./(1.-rhohat)));
      //d2=((W2'*d3)+beta*repmat(sparsity_delta,1,m)).*(a2.*(1-a2));
      //rhohat := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a2map, a2, Ones_VecMap, Ones_Vecdist, Hiddmap); orig
      rhohat := rhat;
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
    //Simple gradient descent algorithm : new_param = old_param - alpha*grad_param
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
        rohat_pass := rohat(a2);
        d2 := DELTA2 (w2m, a2, d3,rohat_pass);
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
    
    //This function returns the cost and gradient of the Sparse Autoencoder parameters
    //the output is in numericfield format where w1, w2, b1, b2 are listed columnwise and the cost value comes at the end and the ids are assigned in this order:
    //first column of w1, second column of w1,..., first column of w2, second colun of w2, ..., b1,b2,cost
    SparseParam_CostGradients :=  FUNCTION
      w1m := w1dist;
      w2m := w2dist;
      b1v := b1vecdist;
      b2v := b2vecdist;
      a2 := FF2 (w1m, b1v);
      a3 := FF3 (w2m, b2v, a2);
      d3 := DELTA3 (a3);
      rohat_a2 := rohat(a2);
      d2 := DELTA2 (w2m, a2, d3,rohat_a2);
      wg1 := WeightGrad1 (w1m, d2);
      wg2 := WeightGrad2 (w2m, d3, a2);
      bg1 := BiasGrad1 (d2);
      bg2 := BiasGrad2 (d3);
      wg1_mat := DMat.Converted.FromPart2Elm (wg1);
      wg2_mat := DMat.Converted.FromPart2Elm (wg2);
      bg1_mat := DMat.Converted.FromPart2Elm (bg1);
      bg2_mat := DMat.Converted.FromPart2Elm (bg2);
      wg1_mat_no := Mat.MU.TO(wg1_mat,1);
      wg2_mat_no := Mat.MU.TO(wg2_mat,2);
      bg1_mat_no := Mat.MU.TO(bg1_mat,3);
      bg2_mat_no := Mat.MU.TO(bg2_mat,4);
      prm_MUE := wg1_mat_no + wg2_mat_no + bg1_mat_no + bg2_mat_no;
      //calculate cost
      //PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1, dmap, ddist, b1map, b1m, 1.0);
      // squared_error_cost= 0.5*sum(sum((x-a3).^2));
      // cost=(1/m)*squared_error_cost+(lambda/2)*(sum(W2(:).^2)+sum(W1(:).^2))+beta*sum(KL(sparsityParam,rhohat));
      squared_error_cost := 0.5*PBblas.SumElements(PBblas.Apply2Elements(dmap, PBblas.PB_daxpy(-1.0, a3, ddist), pow2));
      cost_term1 := (1/m)*squared_error_cost;
      cost_term2 := (lambda/2)* PBblas.SumElements(PBblas.Apply2Elements(dmap, w2m, pow2));
      cost_term3 := (lambda/2)* PBblas.SumElements(PBblas.Apply2Elements(dmap, w1m, pow2));
      PBblas.Types.value_t klterm(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := sparsityParam * LN(sparsityParam/v) + (1-sparsityParam) * LN ((1-sparsityParam)/(1-v));
      KL := PBblas.Apply2Elements (Hiddmap,rohat_a2,klterm);
      cost_term4 := beta * PBblas.SumElements(KL);
      cost := cost_term1 + cost_term2 + cost_term3 +cost_term4;    
      //costfield := DATASET ([{1,1,cost,5}],ML.Mat.Types.MUElement);
      //first sort the retunring values, this makes sure that they are going to be retrived correctly later in SparseAutoencoderCost
      // sorted_prm_MUE := SORT (prm_MUE+costfield,no, y, x);
      // AppendID(sorted_prm_MUE, id, prm_MUE_id);
      // ToField (prm_MUE_id, costgrad_out, id, 'x,y,value,no');
      // toberetun :=costgrad_out(number=3);//return only value fields

      nf := NumberofFeatures;
      nh := NumberofHiddenLayerNodes;
      nfh := nf*nh;
      nfh_2 := 2*nfh;
      Types.NumericField Wshape (Mat.Types.MUElement l) := TRANSFORM
        SELF.id := IF (l.no=1,(l.y-1)*nh+l.x,nfh+(l.y-1)*nf+l.x);
        SELF.number := 1;
        SELF.value := l.value;
      END;
      W_field := PROJECT (wg1_mat_no + wg2_mat_no,Wshape(LEFT));
      
      Types.NumericField Bshape (Mat.Types.MUElement l) := TRANSFORM
        SELF.id := IF (l.no=3,nfh_2+l.x,nfh_2+l.x+nh);
        SELF.number := 1;
        SELF.value := l.value;
      END;
      B_field := PROJECT (bg1_mat_no + bg2_mat_no,Bshape(LEFT));
      cost_field := DATASET ([{nfh_2+nf+nh+1,1,cost}],Types.NumericField);
      RETURN W_field+B_field+cost_field;
      //w// RETURN IF (lambda=10 ,cost_field ,W_field+B_field+cost_field);
      //w//RETURN IF (lambda=10 ,W_field ,W_field+B_field+cost_field);
      //w// RETURN IF (lambda=10 ,B_field ,W_field+B_field+cost_field);
     //w// RETURN IF (lambda=10 ,B_field+cost_field ,W_field+B_field+cost_field);
    //w// RETURN IF (lambda=10 ,W_field+B_field+cost_field ,W_field+B_field+cost_field);
  //w//  RETURN IF (FALSE ,W_field+B_field+cost_field ,W_field+B_field+cost_field);
  //w//RETURN W_field+B_field+cost_field;
    END;//END SparseParam_CostGradients
    //this is actually the first implementation of SparseParam_CostGradients where it would simply convert MUE format to filed which was not consistent to what lbfgs expect the costfunc to do
    //no=2 belongs to w2
    //no=3 belongs to b1
    //no=4 belongas to b2
    //no=5 belongs to cost
    SparseParam_CostGradients2 :=  FUNCTION
      w1m := w1dist;
      w2m := w2dist;
      b1v := b1vecdist;
      b2v := b2vecdist;
      a2 := FF2 (w1m, b1v);
      a3 := FF3 (w2m, b2v, a2);
      d3 := DELTA3 (a3);
      rohat_a2 := rohat(a2);
      d2 := DELTA2 (w2m, a2, d3,rohat_a2);
      wg1 := WeightGrad1 (w1m, d2);
      wg2 := WeightGrad2 (w2m, d3, a2);
      bg1 := BiasGrad1 (d2);
      bg2 := BiasGrad2 (d3);
      wg1_mat := DMat.Converted.FromPart2Elm (wg1);
      wg2_mat := DMat.Converted.FromPart2Elm (wg2);
      bg1_mat := DMat.Converted.FromPart2Elm (bg1);
      bg2_mat := DMat.Converted.FromPart2Elm (bg2);
      wg1_mat_no := Mat.MU.TO(wg1_mat,1);
      wg2_mat_no := Mat.MU.TO(wg2_mat,2);
      bg1_mat_no := Mat.MU.TO(bg1_mat,3);
      bg2_mat_no := Mat.MU.TO(bg2_mat,4);
      prm_MUE := wg1_mat_no + wg2_mat_no + bg1_mat_no + bg2_mat_no;
      AppendID(prm_MUE, id, prm_MUE_id);
      
      
      //calculate cost
      //PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1, dmap, ddist, b1map, b1m, 1.0);
      // squared_error_cost= 0.5*sum(sum((x-a3).^2));
      // cost=(1/m)*squared_error_cost+(lambda/2)*(sum(W2(:).^2)+sum(W1(:).^2))+beta*sum(KL(sparsityParam,rhohat));
      squared_error_cost := 0.5*PBblas.SumElements(PBblas.Apply2Elements(dmap, PBblas.PB_daxpy(-1.0, a3, ddist), pow2));
      cost_term1 := (1/m)*squared_error_cost;
      cost_term2 := (lambda/2)* PBblas.SumElements(PBblas.Apply2Elements(dmap, w2m, pow2));
      cost_term3 := (lambda/2)* PBblas.SumElements(PBblas.Apply2Elements(dmap, w1m, pow2));
      PBblas.Types.value_t klterm(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := sparsityParam * LN(sparsityParam/v) + (1-sparsityParam) * LN ((1-sparsityParam)/(1-v));
      KL := PBblas.Apply2Elements (Hiddmap,rohat_a2,klterm);
      cost_term4 := beta * PBblas.SumElements(KL);
      cost := cost_term1 + cost_term2 + cost_term3 +cost_term4;    
      costidfield := DATASET ([{18,1,1,cost,5}],{ML.Types.t_RecordID id:=0;ML.Mat.Types.MUElement;});
      
      ToField (costidfield+prm_MUE_id, costgrad_out, id, 'x,y,value,no');
      RETURN costgrad_out;
    END;//END SparseParam_CostGradients2
    
    
    
    //if learning_param=simple then mod = SAprm_MUE_out, if learning_param = lbfgs then mod = SparseParam_CostGradients
    //EXPORT Mod := SAprm_MUE_out; orig , also make sure where "mod" is used and change them accordingly (mod is used in SparseAutoencoderCost)
    EXPORT mod := SAprm_MUE_out; // orig
  END;//END SA
  
  
  

  //theta includes the weight and bias matrices for the SparseAutoencoder in a numericfield dataset,below it is explained how this dataset is aquired
  //1- there is a Mat.Types.MUElement  dataset where that no=1 is w1, no=2 is w2, no =3 is b1 and no = 4 is b4
  //2-this dataset gets sorted based on (no->y->x)
  //3-the dataset is then converted to numeric field format
  //4-only the recordsets where number =3 (the corresponding "value" field in the Mat.Types.MUElement record) are returned 
  //CostFunc_params includes the parameters that the sparse autoencoder algortihm need : REAL8 BETA, REAL8 sparsityParam, , REAL8 LAMBDA,
  // CostFunc_params = DATASET([{1, 1, BETA},{2,1,sparsityParam},{3,1,LAMBDA}], Types.NumericField);
  SparseAutoencoderCost (DATASET(Types.NumericField) theta, DATASET(Types.NumericField) CostFunc_params, DATASET(Types.NumericField) TrainData , DATASET(Types.NumericField) TrainLabel=emptyC):= FUNCTION
    //Extract weights and bias matrices from theta by using the numebr of hidden and visible nodes
    nf := NumberofFeatures;
    nh := NumberofHiddenLayerNodes;
    nfh := nf*nh;
    nfh_2 := 2*nfh;
    //this transfrom converts the weight part of the theta (id>=1 and id <=2*nfh) to a MUE dataset where W1 has no=1 and W2 has no =2
    Mat.Types.MUElement Wreshape (Types.NumericField l) := TRANSFORM
      no_temp := (l.id DIV (nfh+1))+1;
      SELF.no := no_temp;
      SELF.x := IF (no_temp=1, 1+((l.id-1)%nh) , 1+((l.id-1-nfh)%nf));
      SELF.y := IF (no_temp=1, ((l.id-1) DIV nh)+1, ((l.id-1-nfh) DIV nf)+1);
      SELF.value := l.value;
    END;
    SA_W := PROJECT (theta(id<=2*nfh),Wreshape(LEFT));
    //this transfrom converts the bias part of the theta (id>=2*nfh+1) to a MUE dataset where b1 has no=1 and b2 has no =2
    Mat.Types.MUElement Breshape (Types.NumericField l) := TRANSFORM
      no_temp := IF (l.id-nfh_2<=nh,1,2);
      SELF.no := no_temp;
      SELF.x := IF (no_temp =1 ,l.id-nfh_2, l.id-nfh_2-nh);
      SELF.y := 1;
      SELF.value := l.value;
    END;
    SA_B := PROJECT (theta(id>nfh_2),Breshape(LEFT));

    
    // thetalD_Map :=	DATASET([{'id','ID'},{'x','1'},{'y','2'},{'value','3'},{'no','4'}], {STRING orig_name; STRING assigned_name;});
    // FromField(theta,Mat.Types.MUElement,params,thetalD_Map);
    // SA_weights := params (no<3);
    // B := params (no>2 AND no<5);
    // Mat.Types.MUElement Bno (Mat.Types.MUElement l) := TRANSFORM
      // SELF.no := l.no-2;
      // SELF := l;
    // END;
    // SA_bias := PROJECT (B,Bno(LEFT));
    //extract the sparseautoencoder parameters from CostFunc_params
    SA_BETA := CostFunc_params(id=1)[1].value;
    SA_sparsityparam := CostFunc_params(id=2)[1].value;
    SA_LAMBDA := CostFunc_params(id=3)[1].value;
    
    //orig    
    //my test starts
    mytest := CostFunc_params(id=1)[1].number;
    
    
    Cost_Grad := SA(TrainData,SA_W,SA_B, SA_BETA,SA_sparsityparam,SA_LAMBDA).mod;//orig , if you change the output of mod, don't forget to change it here as well
    Cost_Grad2 := SA(TrainData,SA_W,SA_B, SA_BETA,SA_sparsityparam,0.1).mod;
   //w// RETURN IF (mytest=11, DATASET([{100,200,300}],Types.NumericField), Cost_Grad);
   //w// RETURN IF (mytest=11, theta, Cost_Grad);
   //w//RETURN IF (mytest=180, Cost_Grad2, Cost_Grad);
   //NW// RETURN IF (FALSE, Cost_Grad2, Cost_Grad);
   RETURN IF (mytest=180, Cost_Grad2, Cost_Grad);
   //RETURN IF (FALSE, Cost_Grad2, Cost_Grad);
    //RETURN Cost_Grad; orig
  END; //end SparseAutoencoderCost
  EXPORT LearnC_lbfgs(DATASET(Types.NumericField) Indep,DATASET(Mat.Types.MUElement) IntW, DATASET(Mat.Types.MUElement) Intb, REAL8 BETA=0.1, REAL8 sparsityParam=0.1 , REAL8 LAMBDA=0.001, UNSIGNED2 MaxIter=100) := FUNCTION

    
    //prepare the parameters to be passed to MinFUNC
    //theta
    //convert IntW and Intb to NumericField format
    nf := NumberofFeatures;
    nh := NumberofHiddenLayerNodes;
    nfh := nf*nh;
    nfh_2 := 2*nfh;
    Types.NumericField Wshape (Mat.Types.MUElement l) := TRANSFORM
      SELF.id := IF (l.no=1,(l.y-1)*nh+l.x,nfh+(l.y-1)*nf+l.x);
      SELF.number := 1;
      SELF.value := l.value;
    END;
    W_field := PROJECT (IntW,Wshape(LEFT));
    
    Types.NumericField Bshape (Mat.Types.MUElement l) := TRANSFORM
      SELF.id := IF (l.no=1,nfh_2+l.x,nfh_2+l.x+nh);
      SELF.number := 1;
      SELF.value := l.value;
    END;
    B_field := PROJECT (Intb,Bshape(LEFT));    
    
    //CostFunc_params
    CostFunc_params_input := DATASET([{1, 1, BETA},{2,1,sparsityParam},{3,1,LAMBDA}], Types.NumericField);
    //MinFUNC( x0,CostFunc ,  CostFunc_params, TrainData ,  TrainLabel,  MaxIter = 500,  tolFun = 0.00001, TolX = 0.000000001,  maxFunEvals = 1000,  corrections = 100, prows=0, pcols=0, Maxrows=0, Maxcols=0) := FUNCTION
    LearntMod:=  Optimization (0, 0, 0, 0).MinFUNC (W_field+B_field, SparseAutoencoderCost, CostFunc_params_input, Indep , emptyC, MaxIter, 0.00001, 0.000000001,100, 3,0, 0, 0,0);
  RETURN LearntMod;
  
  
  
 /* 
   CostFunc_params_input := DATASET([{1, 1, BETA},{2,1,sparsityParam},{3,1,LAMBDA}], Types.NumericField);
  
x:= DATASET([
    {1,	1,	0.006044495885799916},
 {2	,1,	0.009318789063606572},
 {3	,1,	0.01018669626556341},
 {4	,1,	0.006171506619406089},
 {5	,1,	0.006128289843967584},
 {6	,1,	0.007025748362465625},
 {7,	1,	-0.004510862649502231},
 {8,	1,	-0.006469218178418376},
 {9,	1,	-0.006271200900827688},
 {10,	1,	-0.004941312737885555},
 {11,	1,	-0.004016700701096238},
 {12,	1,	-0.006086972665563899},
 {13,	1,	-2.215334899314491},
 {14	,1	,-2.222115123936652},
 {15,	1	,0.139947050029807},
 {16	,1	,0.07317247799046184},
 {17	,1,	-0.1951463119408574}

   ],Types.NumericField);
d:= DATASET ([
    {9	,1	,0.01310459857389977},
{10,	1	,0.0102596486212567},
{11,	1	,0.008379149719670206},
{12,	1,	0.01259729863630003},
{13,	1,	0.0277168106806592},
{14	,1,	0.02799687919365262},
{15	,1,	-0.003591687736872379},
{16	,1,	-0.00341460806949099},
{17,	1,	-0.001927009374861092},
{1	,1,	0.00791149140385727},
{2	,1,	-0.001359049696644065},
{3	,1,	-0.001355452314727256},
{4	,1,	0.004057234572179064},
{5	,1,	0.004584583276707946},
{6	,1	,0.0003721016522158791},
{7	,1	,0.008886026646056226},
{8	,1,	0.01366528895001707}

   ],Types.NumericField);
t:=1;

p_um := MAX (x,id);



g := DATASET([
   {13,	1,	-0.001845976252393845},
{14,	1	,-0.003641821060653878},
{15,	1	,0.0003392085641840357},
{16,	1	,0.0003392222731448309},
{17,	1	,0.0002635766539010679},
{1	,1	,-0.0003665246484459359},
{7,	1	,-0.0004390755921208298},
{2,	1	,-0.0009998691231218456},
{8,	1,	-0.0006480423312669389},
{3,	1	,8.728925193812395e-05},
{4,	1	,-0.001249349130128777},
{9	,1	,-0.0006171297638230355},
{5	,1	,-0.0002054446299479931},
{10,	1	,-0.0004867139520042107},
{11	,1	,-0.0003965699748878922},
{6	,1	,-0.0009270386230434895},
{12	,1,	-0.000595955057843322}

],Types.NumericField);

f := 0.1397830237894385;



//gtd = d'*d;
gtdT := ML.Mat.Mul (ML.Mat.Trans(ML.Types.ToMatrix(g)),ML.Types.ToMatrix(d));
gtd := -0.0002006921080210102;


//WolfeLineSearch(x, t,d, f,g,  gtd,  c1=0.0001,  c2=0.9,  maxLS=25,  tolX=0.000000001, CostFunc_params,  TrainData ,  TrainLabel,  CostFunc, prows=0, pcols=0, Maxrows=0, Maxcols=0):=FUNCTION
WResult := Optimization (0, 0, 0, 0).WolfeLineSearch(x,1,d,f,g,gtd,0.0001,0.9,25,0.000000001, CostFunc_params_input, Indep , emptyC,SparseAutoencoderCost,0,0,0,0);
WWWresult := Optimization (0, 0, 0, 0).WolfeOut_FromField(WResult);
  
  RETURN WWWresult;
  */
  END;//END LearnC_lbfgs

  EXPORT LearnC (DATASET(Types.NumericField) Indep,DATASET(Mat.Types.MUElement) IntW, DATASET(Mat.Types.MUElement) Intb, REAL8 BETA=3, REAL8 sparsityParam=0.1 , REAL8 LAMBDA=0.001, REAL8 ALPHA=0.1, UNSIGNED2 MaxIter=100) := SA(Indep,IntW,Intb, BETA,sparsityParam,LAMBDA, ALPHA,  MaxIter).mod;//orig
//SA(Indep,IntW,Intb, BETA,sparsityParam,LAMBDA, ALPHA,  MaxIter).mod;

  // EXPORT GradientCost(DATASET(Types.NumericField) Indep,DATASET(Mat.Types.MUElement) IntW, DATASET(Mat.Types.MUElement) Intb, REAL8 BETA=0.1, REAL8 sparsityParam=0.1 , REAL8 LAMBDA=0.001, REAL8 ALPHA=0.1, UNSIGNED2 MaxIter=100) := FUNCTION
    // result := SA(Indep,IntW,Intb, BETA,sparsityParam,LAMBDA, ALPHA,  MaxIter).mod;
    // RETURN result;
  // END;//END Model
  
  //convert the output to the more understandable format
  //no = 1 is the w1 matrix
  //no = 2 is the w2 matrix
  //no =3 is the b1 bias matrix
  //no = 4 is the b2 bias matrix
  // in case than there is a no = 5 it indicates the cost value
  EXPORT Model(DATASET(Types.NumericField) mod) := FUNCTION
    modelD_Map :=	DATASET([{'id','ID'},{'x','1'},{'y','2'},{'value','3'},{'no','4'}], {STRING orig_name; STRING assigned_name;});
    FromField(mod,Mat.Types.MUElement,dOut,modelD_Map);
    RETURN dOut;
  END;//END Model
  //the data and the SA model is fed to the function to calculate the output
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
    a2_mat := DMat.Converted.FromPart2Elm(a2);

    NumericField tr (Mat.Types.Element le) := TRANSFORM
      SELF.id := le.y;
      SELF.number := le.x;
      SELF := le;
    END;
    RETURN PROJECT (a2_mat, tr(LEFT));
  END;//END SAOutput
  EXPORT ExtractWeights (DATASET(Types.NumericField) mod) := FUNCTION
    SAmod := Model (mod);
    RETURN SAmod (no<3);
  END;//END ExtractWeights
  EXPORT ExtractBias (DATASET(Types.NumericField) mod) := FUNCTION
    SAmod := Model (mod);
    B := SAmod (no>2 AND no<5);
    Mat.Types.MUElement Sno (Mat.Types.MUElement l) := TRANSFORM
      SELF.no := l.no-2;
      SELF := l;
    END;
    RETURN PROJECT (B,Sno(LEFT));
  END;//END ExtractBias
  EXPORT ExtractW1 (DATASET(Types.NumericField) mod) := FUNCTION
    w1mod := mod (number = 4 and value = 1);
    Myid := RECORD
      w1mod.id;
    END;
    w1modid := TABLE(w1mod,Myid);
    w1r := JOIN (mod,w1modid,LEFT.id=RIGHT.id,TRANSFORM(LEFT) );
    RETURN w1r;
  END;
  EXPORT ExtractW2 (DATASET(Types.NumericField) mod) := FUNCTION
    w2mod := mod (number = 4 and value = 2);
    Myid := RECORD
      w2mod.id;
    END;
    w2modid := TABLE(w2mod,Myid);
    w2r := JOIN (mod,w2modid,LEFT.id=RIGHT.id,TRANSFORM(LEFT) );
    RETURN w2r;
  END;
  EXPORT Extractb1 (DATASET(Types.NumericField) mod) := FUNCTION
    b1mod := mod (number = 4 and value = 3);
    Myid := RECORD
      b1mod.id;
    END;
    b1modid := TABLE(b1mod,Myid);
    b1r := JOIN (mod,b1modid,LEFT.id=RIGHT.id,TRANSFORM(LEFT) );
    RETURN b1r;
  END;
  EXPORT Extractb2 (DATASET(Types.NumericField) mod) := FUNCTION
    b2mod := mod (number = 4 and value = 4);
    Myid := RECORD
      b2mod.id;
    END;
    b2modid := TABLE(b2mod,Myid);
    b2r := JOIN (mod,b2modid,LEFT.id=RIGHT.id,TRANSFORM(LEFT) );
    RETURN b2r;
  END;
END;//END Sparse_Autoencoder
//this function stack ups NumSAs sparse autoencoders to make a Deep Network of Sparse Autoencoders
//In this module we recive unsupervised data and pass it through NumSAs layers of sparseAutoencoders to initialize the weights in this network with a Greedy Layer-Wise manner
//data is passed to the first SA (Sparse Autoencoder) and SA is trained, i.e. the weights are learnt, when it is trained the output of it is passed to the second SA as input, the second SA is trained with 
//this data, then the output of this SA is passed as the input to the third SA, this continues until NumSAs of SAs are trained. At the end the end the end the whole network weighst are initialized
//with this method
//NumSAs : Number of SAs in the Deep Network, basically it means number of sparseautoencoders that need to stack up to make the deep network (the number of layers in the final Deep Learning models is
//NumSAs+1 because we have the input layer as well
//numHiddenNodes : number of hidden nodes in each Sparse Autoencoder
EXPORT StackedSA (UNSIGNED4 NumSAs, DATASET(Types.DiscreteField) numHiddenNodes, REAL8 BETA, REAL8 sparsityParam , REAL8 LAMBDA=0.001, REAL8 ALPHA=0.1, UNSIGNED2 MaxIter=100,
  UNSIGNED4 prows=0, UNSIGNED4 pcols=0, UNSIGNED4 Maxrows=0, UNSIGNED4 Maxcols=0) := MODULE
  NL := NumSAs+1;//number of layers in the final Deep Learning algorithm is 1 (input layer) + Number of SparseAutoencoders
  SSA (DATASET(Types.NumericField) X) := MODULE
      //TRANFFORM used
      Mat.Types.MUElement Addno (Mat.Types.MUElement l, UNSIGNED v) := TRANSFORM
        SELF.no := l.no+v;
        SELF := l;
      END;

    //number of features in the input independent data
    NumFeatures := MAX (X,number);
    //Define the first Sparse Autoencoder Module
    hd1 := numHiddenNodes(id=(1))[1].value;//number of hidden nodes in the first SA
    IntW1 := Sparse_Autoencoder_IntWeights(NumFeatures,hd1);//initialize weights
    Intb1 := Sparse_Autoencoder_IntBias(NumFeatures,hd1);//initialize bias
    SA1 := Sparse_Autoencoder (NumFeatures, hd1, prows, pcols, Maxrows, Maxcols);//SA module for the first SA
    //train the first Sparse Autoencoder
    LearntModel1 := SA1.LearnC(X,IntW1, Intb1, BETA, sparsityParam , LAMBDA, ALPHA, MaxIter); //learnt model in NumericFiled format
    Bias1 := SA1.ExtractBias (LearntModel1);
    Weight1 := SA1.ExtractWeights (LearntModel1);
    SAmodel1 := Weight1 (no=1) + PROJECT (Bias1 (no=1),Addno(LEFT,NL)); // Only weight and bias related to the first layer and hidden layer are considered for each SA to stack them up
    //produce the output of the first learnt Sparse Autoencoder
    Output1 := SA1.SAOutput (X, LearntModel1);
    MatrixOutput1 := ML.Types.ToMatrix (Output1);
    MatrixOutput1No := Mat.MU.To(MatrixOutput1, 0);
    StackedSA_Step(DATASET(Mat.Types.MUElement) MM, INTEGER coun) := FUNCTION
      L := coun + 1;
      //output of the previous SA which is gonna be the input of the next SA
      lastOutput := Mat.MU.From(MM, 0);
      lastOutputF := ML.Types.FromMatrix(lastOutput);
      //Define the Lth SaprseAutoencoder
      NFL := numHiddenNodes(id=(L-1))[1].value; //number of hidden layers of the last SA represents the number of input features for the next SA
      hdL := numHiddenNodes(id=(L))[1].value;
      IntWL := Sparse_Autoencoder_IntWeights(NFL,hdL);//initialize weights
      IntbL := Sparse_Autoencoder_IntBias(NFL,hdL);//initialize bias
      SAL := Sparse_Autoencoder (NFL, hdl, prows, pcols, Maxrows, Maxcols);//SA module for the Lth SA
      //Train the Lth SaprseAutoencoder (output of the last SA is fed as the input to the next SA)
      LearntModelL := SAL.LearnC(lastOutputF,IntWL, IntbL, BETA, sparsityParam , LAMBDA, ALPHA, MaxIter);
      BiasL := SAL.ExtractBias (LearntModelL);
      WeightL := SAL.ExtractWeights (LearntModelL);
      SAmodelL := PROJECT (WeightL (no=1),Addno(LEFT,coun)) + PROJECT (BiasL (no=1),Addno(LEFT,coun+NL));
      //produce the output of the Lth learnt Sparse Autoencoder
      OutputL := SAL.SAOutput (lastOutputF, LearntModelL);
      MatrixOutputL := ML.Types.ToMatrix (OutputL);
      MatrixOutputLNo := Mat.MU.To(MatrixOutputL, 0);
      RETURN SAmodelL + MatrixOutputLNo + MM (no > 0);
      //RETURN SAmodelL + MM + PROJECT (IntWL,Addno(LEFT,100)) + PROJECT (IntbL,Addno(LEFT,200));//the line I used to test the second SA's output with MATLAB code
    END;//END StackedSA_Step
    EXPORT SSA_prm := LOOP(SAmodel1 + MatrixOutput1No, COUNTER <= NumSAs-1, StackedSA_Step(ROWS(LEFT),COUNTER));//SSA_prm is in Mat.Types.MUElement format convert it to NumericFieldFormat
    AppendID(SSA_prm, id, SSA_prm_id);
    ToField (SSA_prm_id, mm, id, 'x,y,value,no');//convert the learnt model to numerifield before returning it
    EXPORT Mod := mm;
  END;//END SSA
  //LearnC returns the learnt model from Stacking up of SparseAutoencoders when some unsupervised data (Indep) are fed to it
  //the learn model contains one weight and one bias matrix correpondance to each SparseAutoencoder
  //the weight and bias matrix that correspond to each SA are actually the weight between first and hidden layer and the bias that goes to the hideen layer
  //the output of the Stacked Autoencoder (extracted feature) has no =0
  EXPORT LearnC (DATASET(Types.NumericField) Indep) := SSA(Indep).Mod;
  //Model converts the learnt model from Numeric field format to the Mat.Types.MUElement format
  //in the built model the no={1,2,..,NL-1} are the weight indexes
  //no={NL+1,NL+2,..,NL+NL-1} are bias indexes that go to the second, third, ..,NL)'s layer respectively
  //no={1,NL+1}: weight and bias belong to the first SA
  //no={2,NL+2}: weight and bias belong to the second SA
  //no={NL-1,NL+NL-1}: weight and bias belong to the second NL-1th SA
  EXPORT Model(DATASET(Types.NumericField) mod) := FUNCTION
    modelD_Map :=	DATASET([{'id','ID'},{'x','1'},{'y','2'},{'value','3'},{'no','4'}], {STRING orig_name; STRING assigned_name;});
    FromField(mod,Mat.Types.MUElement,dOut,modelD_Map);
    RETURN dOut;
  END;//END Model
  EXPORT SSAOutput(DATASET(Types.NumericField) Indep,DATASET(Types.NumericField) LearntMod) :=FUNCTION
    //The leartn model has the same format aa a model which is learnt by using NeuralNetwork.ecl
    //so we only need to feed this model and the input data to the NeuralNetwork.ecl to get the output
    Types.DiscreteField Addid (Types.DiscreteField l) := TRANSFORM
      SELF.id := l.id+1;
      SELF := l;
    END;
    NF := MAX (Indep, Indep.number);
    firstlayer := DATASET([{1, 1, NF}],Types.DiscreteField);//add the input layer information to the numHiddenNodes (numHiddenNodes only includes the SAs inforamtion)
    NNnet := firstlayer + PROJECT(numHiddenNodes,Addid(LEFT));
    NN := NeuralNetworks(NNnet,prows, pcols, Maxrows,  Maxcols);
    RR :=NN.NNOutput(Indep,LearntMod);
    RETURN RR;
  END;
END;//StackedSA


EXPORT Sparse_Autoencoder_mine (INTEGER4 NumberofFeatures, INTEGER4 NumberofHiddenLayerNodes, UNSIGNED4 prows=0, UNSIGNED4 pcols=0, UNSIGNED4 Maxrows=0, UNSIGNED4 Maxcols=0) := MODULE
  //this is a un-supervised learning algorithm, no need for the labled data
  
  SHARED SA(DATASET(Types.NumericField) X, DATASET(Mat.Types.Element) IntW1, DATASET(Mat.Types.Element) IntW2, DATASET(Mat.Types.Element) Intb1,  DATASET(Mat.Types.Element) Intb2, REAL8 BETA=0.1, REAL8 sparsityParam=0.1 , REAL8 LAMBDA=0.001, REAL8 ALPHA=0.1, UNSIGNED2 MaxIter=100) := MODULE
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
    w1_mat := IntW1;
    w1_mat_x := Mat.Has(w1_mat).Stats.Xmax;
    w1_mat_y := Mat.Has(w1_mat).Stats.Ymax;
    w1map := PBblas.Matrix_Map(w1_mat_x, w1_mat_y, sizeTable[1].f_b_rows, sizeTable[1].f_b_rows);
    w1dist := DMAT.Converted.FromElement(w1_mat,w1map);
    w2_mat := IntW2;
    w2_mat_x := w1_mat_y;
    w2_mat_y := w1_mat_x;
    w2map := PBblas.Matrix_Map(w2_mat_x, w2_mat_y, sizeTable[1].f_b_rows, sizeTable[1].f_b_rows);
    w2dist := DMAT.Converted.FromElement(w2_mat,w2map);
    //each bias vector is converted to block format
    b1vec := Intb1;
    b1vec_x := Mat.Has(b1vec).Stats.Xmax;
    b1vecmap := PBblas.Matrix_Map(b1vec_x, 1, sizeTable[1].f_b_rows, 1);
    b1vecdist := DMAT.Converted.FromElement(b1vec,b1vecmap);
    b2vec := Intb2;
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
    PBblas.Types.value_t pow2(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := v*v;
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
    rohat (DATASET(Layout_Part) a2) := FUNCTION
      rh := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a2map, a2, Ones_VecMap, Ones_Vecdist, Hiddmap);
      RETURN rh;
    END;
    DELTA2 (DATASET(Layout_Part) w2, DATASET(Layout_Part) a2, DATASET(Layout_Part) d3, DATASET(Layout_Part) rhat) := FUNCTION
      //calculate delta for 2nd layer
      //rhohat=mean(a2,2);
      //sparsity_delta=((-sparsityParam./rhohat)+((1-sparsityParam)./(1.-rhohat)));
      //d2=((W2'*d3)+beta*repmat(sparsity_delta,1,m)).*(a2.*(1-a2));
      //rhohat := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a2map, a2, Ones_VecMap, Ones_Vecdist, Hiddmap); orig
      rhohat := rhat;
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
    //Simple gradient descent algorithm : new_param = old_param - alpha*grad_param
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
        rohat_pass := rohat(a2);
        d2 := DELTA2 (w2m, a2, d3,rohat_pass);
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
    
    //This function returns the cost and gradient of the Sparse Autoencoder parameters
    //the output is in numericfield format where w1, w2, b1, b2 are listed columnwise and the cost value comes at the end and the ids are assigned in this order:
    //first column of w1, second column of w1,..., first column of w2, second colun of w2, ..., b1,b2,cost
    SparseParam_CostGradients :=  FUNCTION
      w1m := w1dist;
      w2m := w2dist;
      b1v := b1vecdist;
      b2v := b2vecdist;
      a2 := FF2 (w1m, b1v);
      a3 := FF3 (w2m, b2v, a2);
      d3 := DELTA3 (a3);
      rohat_a2 := rohat(a2);
      d2 := DELTA2 (w2m, a2, d3,rohat_a2);
      wg1 := WeightGrad1 (w1m, d2);
      wg2 := WeightGrad2 (w2m, d3, a2);
      bg1 := BiasGrad1 (d2);
      bg2 := BiasGrad2 (d3);
      wg1_mat := DMat.Converted.FromPart2Elm (wg1);
      wg2_mat := DMat.Converted.FromPart2Elm (wg2);
      bg1_mat := DMat.Converted.FromPart2Elm (bg1);
      bg2_mat := DMat.Converted.FromPart2Elm (bg2);
      wg1_mat_no := Mat.MU.TO(wg1_mat,1);
      wg2_mat_no := Mat.MU.TO(wg2_mat,2);
      bg1_mat_no := Mat.MU.TO(bg1_mat,3);
      bg2_mat_no := Mat.MU.TO(bg2_mat,4);
      prm_MUE := wg1_mat_no + wg2_mat_no + bg1_mat_no + bg2_mat_no;
      //calculate cost
      //PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1, dmap, ddist, b1map, b1m, 1.0);
      // squared_error_cost= 0.5*sum(sum((x-a3).^2));
      // cost=(1/m)*squared_error_cost+(lambda/2)*(sum(W2(:).^2)+sum(W1(:).^2))+beta*sum(KL(sparsityParam,rhohat));
      squared_error_cost := 0.5*PBblas.SumElements(PBblas.Apply2Elements(dmap, PBblas.PB_daxpy(-1.0, a3, ddist), pow2));
      cost_term1 := (1/m)*squared_error_cost;
      cost_term2 := (lambda/2)* PBblas.SumElements(PBblas.Apply2Elements(dmap, w2m, pow2));
      cost_term3 := (lambda/2)* PBblas.SumElements(PBblas.Apply2Elements(dmap, w1m, pow2));
      PBblas.Types.value_t klterm(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := sparsityParam * LN(sparsityParam/v) + (1-sparsityParam) * LN ((1-sparsityParam)/(1-v));
      KL := PBblas.Apply2Elements (Hiddmap,rohat_a2,klterm);
      cost_term4 := beta * PBblas.SumElements(KL);
      cost := cost_term1 + cost_term2 + cost_term3 +cost_term4;    
      //costfield := DATASET ([{1,1,cost,5}],ML.Mat.Types.MUElement);
      //first sort the retunring values, this makes sure that they are going to be retrived correctly later in SparseAutoencoderCost
      // sorted_prm_MUE := SORT (prm_MUE+costfield,no, y, x);
      // AppendID(sorted_prm_MUE, id, prm_MUE_id);
      // ToField (prm_MUE_id, costgrad_out, id, 'x,y,value,no');
      // toberetun :=costgrad_out(number=3);//return only value fields

      nf := NumberofFeatures;
      nh := NumberofHiddenLayerNodes;
      nfh := nf*nh;
      nfh_2 := 2*nfh;
      Types.NumericField Wshape (Mat.Types.MUElement l) := TRANSFORM
        SELF.id := IF (l.no=1,(l.y-1)*nh+l.x,nfh+(l.y-1)*nf+l.x);
        SELF.number := 1;
        SELF.value := l.value;
      END;
      W_field := PROJECT (wg1_mat_no + wg2_mat_no,Wshape(LEFT));
      
      Types.NumericField Bshape (Mat.Types.MUElement l) := TRANSFORM
        SELF.id := IF (l.no=3,nfh_2+l.x,nfh_2+l.x+nh);
        SELF.number := 1;
        SELF.value := l.value;
      END;
      B_field := PROJECT (bg1_mat_no + bg2_mat_no,Bshape(LEFT));
      cost_field := DATASET ([{nfh_2+nf+nh+1,1,cost}],Types.NumericField);
      RETURN W_field+B_field+cost_field;
      //w// RETURN IF (lambda=10 ,cost_field ,W_field+B_field+cost_field);
      //w//RETURN IF (lambda=10 ,W_field ,W_field+B_field+cost_field);
      //w// RETURN IF (lambda=10 ,B_field ,W_field+B_field+cost_field);
     //w// RETURN IF (lambda=10 ,B_field+cost_field ,W_field+B_field+cost_field);
    //w// RETURN IF (lambda=10 ,W_field+B_field+cost_field ,W_field+B_field+cost_field);
  //w//  RETURN IF (FALSE ,W_field+B_field+cost_field ,W_field+B_field+cost_field);
  //w//RETURN W_field+B_field+cost_field;
    END;//END SparseParam_CostGradients
    //this is actually the first implementation of SparseParam_CostGradients where it would simply convert MUE format to filed which was not consistent to what lbfgs expect the costfunc to do
    //no=2 belongs to w2
    //no=3 belongs to b1
    //no=4 belongas to b2
    //no=5 belongs to cost
    SparseParam_CostGradients2 :=  FUNCTION
      w1m := w1dist;
      w2m := w2dist;
      b1v := b1vecdist;
      b2v := b2vecdist;
      a2 := FF2 (w1m, b1v);
      a3 := FF3 (w2m, b2v, a2);
      d3 := DELTA3 (a3);
      rohat_a2 := rohat(a2);
      d2 := DELTA2 (w2m, a2, d3,rohat_a2);
      wg1 := WeightGrad1 (w1m, d2);
      wg2 := WeightGrad2 (w2m, d3, a2);
      bg1 := BiasGrad1 (d2);
      bg2 := BiasGrad2 (d3);
      wg1_mat := DMat.Converted.FromPart2Elm (wg1);
      wg2_mat := DMat.Converted.FromPart2Elm (wg2);
      bg1_mat := DMat.Converted.FromPart2Elm (bg1);
      bg2_mat := DMat.Converted.FromPart2Elm (bg2);
      wg1_mat_no := Mat.MU.TO(wg1_mat,1);
      wg2_mat_no := Mat.MU.TO(wg2_mat,2);
      bg1_mat_no := Mat.MU.TO(bg1_mat,3);
      bg2_mat_no := Mat.MU.TO(bg2_mat,4);
      prm_MUE := wg1_mat_no + wg2_mat_no + bg1_mat_no + bg2_mat_no;
      AppendID(prm_MUE, id, prm_MUE_id);
      
      
      //calculate cost
      //PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1, dmap, ddist, b1map, b1m, 1.0);
      // squared_error_cost= 0.5*sum(sum((x-a3).^2));
      // cost=(1/m)*squared_error_cost+(lambda/2)*(sum(W2(:).^2)+sum(W1(:).^2))+beta*sum(KL(sparsityParam,rhohat));
      squared_error_cost := 0.5*PBblas.SumElements(PBblas.Apply2Elements(dmap, PBblas.PB_daxpy(-1.0, a3, ddist), pow2));
      cost_term1 := (1/m)*squared_error_cost;
      cost_term2 := (lambda/2)* PBblas.SumElements(PBblas.Apply2Elements(dmap, w2m, pow2));
      cost_term3 := (lambda/2)* PBblas.SumElements(PBblas.Apply2Elements(dmap, w1m, pow2));
      PBblas.Types.value_t klterm(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := sparsityParam * LN(sparsityParam/v) + (1-sparsityParam) * LN ((1-sparsityParam)/(1-v));
      KL := PBblas.Apply2Elements (Hiddmap,rohat_a2,klterm);
      cost_term4 := beta * PBblas.SumElements(KL);
      cost := cost_term1 + cost_term2 + cost_term3 +cost_term4;    
      costidfield := DATASET ([{18,1,1,cost,5}],{ML.Types.t_RecordID id:=0;ML.Mat.Types.MUElement;});
      
      ToField (costidfield+prm_MUE_id, costgrad_out, id, 'x,y,value,no');
      RETURN costgrad_out;
    END;//END SparseParam_CostGradients2
    
    
    
    //if learning_param=simple then mod = SAprm_MUE_out, if learning_param = lbfgs then mod = SparseParam_CostGradients
    //EXPORT Mod := SAprm_MUE_out; orig , also make sure where "mod" is used and change them accordingly (mod is used in SparseAutoencoderCost)
    EXPORT mod := SparseParam_CostGradients;
  END;//END SA
  
  
  

  //theta includes the weight and bias matrices for the SparseAutoencoder in a numericfield dataset,below it is explained how this dataset is aquired
  //1- there is a Mat.Types.MUElement  dataset where that no=1 is w1, no=2 is w2, no =3 is b1 and no = 4 is b4
  //2-this dataset gets sorted based on (no->y->x)
  //3-the dataset is then converted to numeric field format
  //4-only the recordsets where number =3 (the corresponding "value" field in the Mat.Types.MUElement record) are returned 
  //CostFunc_params includes the parameters that the sparse autoencoder algortihm need : REAL8 BETA, REAL8 sparsityParam, , REAL8 LAMBDA,
  // CostFunc_params = DATASET([{1, 1, BETA},{2,1,sparsityParam},{3,1,LAMBDA}], Types.NumericField);
  SHARED SparseAutoencoderCost (DATASET(Types.NumericField) theta, DATASET(Types.NumericField) CostFunc_params, DATASET(Types.NumericField) TrainData , DATASET(Types.NumericField) TrainLabel=emptyC):= FUNCTION
    //Extract weights and bias matrices from theta by using the numebr of hidden and visible nodes
    nf := NumberofFeatures;
    nh := NumberofHiddenLayerNodes;
    nfh := nf*nh;
    nfh_2 := 2*nfh;
    //this transfrom converts the weight part of the theta to two matrices SA_W1 and SA_W2
    Mat.Types.Element Wreshape1 (Types.NumericField l) := TRANSFORM
      
      SELF.x :=  1+((l.id-1)%nh) ;
      SELF.y := ((l.id-1) DIV nh)+1;
      SELF.value := l.value;
    END;
    SA_W1 := PROJECT (theta(id<=nfh),Wreshape1(LEFT));
    
    Mat.Types.Element Wreshape2 (Types.NumericField l) := TRANSFORM
      SELF.x :=  1+((l.id-1-nfh)%nf);
      SELF.y :=  ((l.id-1-nfh) DIV nf)+1;
      SELF.value := l.value;
    END;
    SA_W2 := PROJECT (theta(nfh<id and id<=2*nfh),Wreshape2(LEFT));
    //this transfrom converts the bias part of the theta (id>=2*nfh+1) to two matrices SA_B1 and SA_B2
    Mat.Types.Element Breshape1 (Types.NumericField l) := TRANSFORM
      SELF.x := l.id-nfh_2;
      SELF.y := 1;
      SELF.value := l.value;
    END;
    SA_B1 := PROJECT (theta(id>nfh_2 and id<=nfh_2+nh ),Breshape1(LEFT));
    
    Mat.Types.Element Breshape2 (Types.NumericField l) := TRANSFORM
      SELF.x :=  l.id-nfh_2-nh;
      SELF.y := 1;
      SELF.value := l.value;
    END;
    SA_B2 := PROJECT (theta(id>nfh_2+nh),Breshape2(LEFT));

    SA_BETA := CostFunc_params(id=1)[1].value;
    SA_sparsityparam := CostFunc_params(id=2)[1].value;
    SA_LAMBDA := CostFunc_params(id=3)[1].value;
    

    
    Cost_Grad := SA(TrainData,SA_W1,SA_W2,SA_B1,SA_B2, SA_BETA,SA_sparsityparam,SA_LAMBDA).mod;//orig , if you change the output of mod, don't forget to change it here as well

    RETURN Cost_Grad;
  END; //end SparseAutoencoderCost
  
  
 
  





EXPORT LearnC_lbfgs(DATASET(Types.NumericField) Indep,DATASET(Mat.Types.Element) IntW1, DATASET(Mat.Types.Element) IntW2, DATASET(Mat.Types.Element) Intb1,DATASET(Mat.Types.Element) Intb2, REAL8 BETA=0.1, REAL8 sparsityParam=0.1 , REAL8 LAMBDA=0.001, UNSIGNED2 MaxIter=100) := FUNCTION

    
    //prepare the parameters to be passed to MinFUNC
    //theta
    //convert IntW and Intb to NumericField format
    nf := NumberofFeatures;
    nh := NumberofHiddenLayerNodes;
    nfh := nf*nh;
    nfh_2 := 2*nfh;
    Types.NumericField Wshape1 (Mat.Types.Element l) := TRANSFORM
      SELF.id := (l.y-1)*nh+l.x;
      SELF.number := 1;
      SELF.value := l.value;
    END;
    W1_field := PROJECT (IntW1,Wshape1(LEFT));
    
    
    
    Types.NumericField Wshape2 (Mat.Types.Element l) := TRANSFORM
      SELF.id := nfh+(l.y-1)*nf+l.x;
      SELF.number := 1;
      SELF.value := l.value;
    END;
    W2_field := PROJECT (IntW2,Wshape2(LEFT));    
    
    
    Types.NumericField Bshape1 (Mat.Types.Element l) := TRANSFORM
      SELF.id := nfh_2+l.x;
      SELF.number := 1;
      SELF.value := l.value;
    END;
    B1_field := PROJECT (Intb1,Bshape1(LEFT));
    
    Types.NumericField Bshape2 (Mat.Types.Element l) := TRANSFORM
      SELF.id := nfh_2+l.x+nh;
      SELF.number := 1;
      SELF.value := l.value;
    END;
    B2_field := PROJECT (Intb2,Bshape2(LEFT));  
    
    //CostFunc_params
    CostFunc_params_input := DATASET([{1, 1, BETA},{2,1,sparsityParam},{3,1,LAMBDA}], Types.NumericField);
    //MinFUNC( x0,CostFunc ,  CostFunc_params, TrainData ,  TrainLabel,  MaxIter = 500,  tolFun = 0.00001, TolX = 0.000000001,  maxFunEvals = 1000,  corrections = 100, prows=0, pcols=0, Maxrows=0, Maxcols=0) := FUNCTION
    LearntMod:=  Optimization2 (0, 0, 0, 0).MinFUNC3 (W1_field+W2_field+B1_field+B2_field, SparseAutoencoderCost, CostFunc_params_input, Indep , emptyC, MaxIter, 0.00001, 0.000000001,1000, 10,0, 0, 0,0); //orig correction is 100
  RETURN LearntMod;
  
 
  END;//END LearnC_lbfgs

  //EXPORT LearnC (DATASET(Types.NumericField) Indep,DATASET(Mat.Types.Element) IntW1, DATASET(Mat.Types.Element) IntW2, DATASET(Mat.Types.Element) Intb1, DATASET(Mat.Types.Element) Intb2, REAL8 BETA=3, REAL8 sparsityParam=0.1 , REAL8 LAMBDA=0.001, REAL8 ALPHA=0.1, UNSIGNED2 MaxIter=100) := SA(Indep,IntW1,IntW2,Intb1,Intb2, BETA,sparsityParam,LAMBDA, ALPHA,  MaxIter).mod;//orig
//SA(Indep,IntW,Intb, BETA,sparsityParam,LAMBDA, ALPHA,  MaxIter).mod;

  // EXPORT GradientCost(DATASET(Types.NumericField) Indep,DATASET(Mat.Types.MUElement) IntW, DATASET(Mat.Types.MUElement) Intb, REAL8 BETA=0.1, REAL8 sparsityParam=0.1 , REAL8 LAMBDA=0.001, REAL8 ALPHA=0.1, UNSIGNED2 MaxIter=100) := FUNCTION
    // result := SA(Indep,IntW,Intb, BETA,sparsityParam,LAMBDA, ALPHA,  MaxIter).mod;
    // RETURN result;
  // END;//END Model
  
  //convert the output to the more understandable format
  //no = 1 is the w1 matrix
  //no = 2 is the w2 matrix
  //no =3 is the b1 bias matrix
  //no = 4 is the b2 bias matrix
  // in case than there is a no = 5 it indicates the cost value
  EXPORT Model(DATASET(Types.NumericField) mod) := FUNCTION
    modelD_Map :=	DATASET([{'id','ID'},{'x','1'},{'y','2'},{'value','3'},{'no','4'}], {STRING orig_name; STRING assigned_name;});
    FromField(mod,Mat.Types.MUElement,dOut,modelD_Map);
    RETURN dOut;
  END;//END Model
  //the data and the SA model is fed to the function to calculate the output
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
    a2_mat := DMat.Converted.FromPart2Elm(a2);

    NumericField tr (Mat.Types.Element le) := TRANSFORM
      SELF.id := le.y;
      SELF.number := le.x;
      SELF := le;
    END;
    RETURN PROJECT (a2_mat, tr(LEFT));
  END;//END SAOutput
  EXPORT ExtractWeights (DATASET(Types.NumericField) mod) := FUNCTION
    SAmod := Model (mod);
    RETURN SAmod (no<3);
  END;//END ExtractWeights
  EXPORT ExtractBias (DATASET(Types.NumericField) mod) := FUNCTION
    SAmod := Model (mod);
    B := SAmod (no>2 AND no<5);
    Mat.Types.MUElement Sno (Mat.Types.MUElement l) := TRANSFORM
      SELF.no := l.no-2;
      SELF := l;
    END;
    RETURN PROJECT (B,Sno(LEFT));
  END;//END ExtractBias
  EXPORT ExtractW1 (DATASET(Types.NumericField) mod) := FUNCTION
    w1mod := mod (number = 4 and value = 1);
    Myid := RECORD
      w1mod.id;
    END;
    w1modid := TABLE(w1mod,Myid);
    w1r := JOIN (mod,w1modid,LEFT.id=RIGHT.id,TRANSFORM(LEFT) );
    RETURN w1r;
  END;
  EXPORT ExtractW2 (DATASET(Types.NumericField) mod) := FUNCTION
    w2mod := mod (number = 4 and value = 2);
    Myid := RECORD
      w2mod.id;
    END;
    w2modid := TABLE(w2mod,Myid);
    w2r := JOIN (mod,w2modid,LEFT.id=RIGHT.id,TRANSFORM(LEFT) );
    RETURN w2r;
  END;
  EXPORT Extractb1 (DATASET(Types.NumericField) mod) := FUNCTION
    b1mod := mod (number = 4 and value = 3);
    Myid := RECORD
      b1mod.id;
    END;
    b1modid := TABLE(b1mod,Myid);
    b1r := JOIN (mod,b1modid,LEFT.id=RIGHT.id,TRANSFORM(LEFT) );
    RETURN b1r;
  END;
  EXPORT Extractb2 (DATASET(Types.NumericField) mod) := FUNCTION
    b2mod := mod (number = 4 and value = 4);
    Myid := RECORD
      b2mod.id;
    END;
    b2modid := TABLE(b2mod,Myid);
    b2r := JOIN (mod,b2modid,LEFT.id=RIGHT.id,TRANSFORM(LEFT) );
    RETURN b2r;
  END;
  
  
  EXPORT CostGrad_cal (DATASET(Types.NumericField) Indep,DATASET(Mat.Types.Element) IntW1, DATASET(Mat.Types.Element) IntW2, DATASET(Mat.Types.Element) Intb1,DATASET(Mat.Types.Element) Intb2, REAL8 BETA=0.1, REAL8 sparsityParam=0.1 , REAL8 LAMBDA=0.001, UNSIGNED2 MaxIter=100) := SA (Indep,IntW1, IntW2,Intb1,Intb2, BETA, sparsityParam , LAMBDA,  MaxIter).mod;
END;//END Sparse_Autoencoder_mine




END;//END DeepLearning