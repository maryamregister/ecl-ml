
IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat; 
IMPORT * FROM ML.Types;
IMPORT PBblas;
Layout_Cell := PBblas.Types.Layout_Cell;
Layout_Part := PBblas.Types.Layout_Part;
emptyC := DATASET([], Types.NumericField);
SHARED emptyMUelm := DATASET([], Mat.Types.MUElement);
IMPORT STD;
IMPORT std.system.Thorlib;
SHARED Layout_Cell_nid := RECORD (Layout_Cell)
UNSIGNED4 node_id;
END;
// The REAL8 first implementation of Deep Learning

EXPORT DeepLearning8 := MODULE
EXPORT Sparse_Autoencoder_IntWeights (INTEGER4 NumberofFeatures, INTEGER4 NumberofHiddenLayerNodes) := FUNCTION
		W_Rec := RECORD
			STRING1 x:= '';
    END;
		W1_ := DATASET([{' '}], W_Rec);
		W2_ := DATASET([{' '}], W_Rec);
    r := SQRT(6) / SQRT (NumberofFeatures + NumberofHiddenLayerNodes + 1);
    //Generate a random number
    Produce_Random () := FUNCTION
      G := 1000000;
      Rnd_ := (RANDOM()%G) / (REAL8)G;
      Rnd := Rnd_ * 2 * r - r;
      RETURN Rnd;
    END;
    //New Randome Matrix Generator
    Mat.Types.Element RandGen(UNSIGNED4 c, UNSIGNED4 NumRows) := TRANSFORM
      SELF.x := ((c-1) % NumRows) + 1;
      SELF.y := ((c-1) DIV NumRows) + 1;
      SELF.value := Produce_Random();
    END;
    //Creat the first weight matrix with no=1 (weight matrix between layer 1 and layer 2)
    w1rows := NumberofHiddenLayerNodes;
    w1cols := NumberofFeatures;
    w1size := w1rows*w1cols;
		w1_test := NORMALIZE(W1_, w1size, RandGen(COUNTER, w1rows));
    w1 := DATASET(w1size, RandGen(COUNTER, w1rows));
    w1no := Mat.MU.To(w1_test, 1);
    
    w2rows := NumberofFeatures;
    w2cols := NumberofHiddenLayerNodes;
    w2size := w1rows*w1cols;
		w2_test := NORMALIZE(W2_, w2size, RandGen(COUNTER, w2rows));
    w2 := DATASET(w2size, RandGen(COUNTER, w2rows));
    w2no := Mat.MU.To(w2_test, 2);
  RETURN w1no+w2no;
END;
EXPORT Sparse_Autoencoder_IntWeights1 (INTEGER4 NumberofFeatures, INTEGER4 NumberofHiddenLayerNodes) := FUNCTION
		W_Rec := RECORD
			STRING1 x:= '';
    END;
		W1_ := DATASET([{' '}], W_Rec);
		W2_ := DATASET([{' '}], W_Rec);
    r := SQRT(6) / SQRT (NumberofFeatures + NumberofHiddenLayerNodes + 1);
    //Generate a random number
    Produce_Random () := FUNCTION
      G := 1000000;
      Rnd_ := (RANDOM()%G) / (REAL8)G;
      Rnd := Rnd_ * 2 * r - r;
      RETURN Rnd;
    END;
    //New Randome Matrix Generator
    Mat.Types.Element RandGen(UNSIGNED4 c, UNSIGNED4 NumRows) := TRANSFORM
      SELF.x := ((c-1) % NumRows) + 1;
      SELF.y := ((c-1) DIV NumRows) + 1;
      SELF.value := Produce_Random();
    END;
    //Creat the first weight matrix with no=1 (weight matrix between layer 1 and layer 2)
    w1rows := NumberofHiddenLayerNodes;
    w1cols := NumberofFeatures;
    w1size := w1rows*w1cols;
		w1_test := NORMALIZE(W1_, w1size, RandGen(COUNTER, w1rows));
    w1 := DATASET(w1size, RandGen(COUNTER, w1rows));
    w1no := Mat.MU.To(w1_test, 1);
    
    w2rows := NumberofFeatures;
    w2cols := NumberofHiddenLayerNodes;
    w2size := w1rows*w1cols;
		w2_test := NORMALIZE(W2_, w2size, RandGen(COUNTER, w2rows));
    w2 := DATASET(w2size, RandGen(COUNTER, w2rows));
    w2no := Mat.MU.To(w2_test, 2);
  RETURN w1;
END;
EXPORT Sparse_Autoencoder_IntWeights2 (INTEGER4 NumberofFeatures, INTEGER4 NumberofHiddenLayerNodes) := FUNCTION
		W_Rec := RECORD
			STRING1 x:= '';
    END;
		W1_ := DATASET([{' '}], W_Rec);
		W2_ := DATASET([{' '}], W_Rec);
    r := SQRT(6) / SQRT (NumberofFeatures + NumberofHiddenLayerNodes + 1);
    //Generate a random number
    Produce_Random () := FUNCTION
      G := 1000000;
      Rnd_ := (RANDOM()%G) / (REAL8)G;
      Rnd := Rnd_ * 2 * r - r;
      RETURN Rnd;
    END;
    //New Randome Matrix Generator
    Mat.Types.Element RandGen(UNSIGNED4 c, UNSIGNED4 NumRows) := TRANSFORM
      SELF.x := ((c-1) % NumRows) + 1;
      SELF.y := ((c-1) DIV NumRows) + 1;
      SELF.value := Produce_Random();
    END;
    //Creat the first weight matrix with no=1 (weight matrix between layer 1 and layer 2)
    w1rows := NumberofHiddenLayerNodes;
    w1cols := NumberofFeatures;
    w1size := w1rows*w1cols;
		w1_test := NORMALIZE(W1_, w1size, RandGen(COUNTER, w1rows));
    w1 := DATASET(w1size, RandGen(COUNTER, w1rows));
    w1no := Mat.MU.To(w1_test, 1);
    
    w2rows := NumberofFeatures;
    w2cols := NumberofHiddenLayerNodes;
    w2size := w1rows*w1cols;
		w2_test := NORMALIZE(W2_, w2size, RandGen(COUNTER, w2rows));
    w2 := DATASET(w2size, RandGen(COUNTER, w2rows));
    w2no := Mat.MU.To(w2_test, 2);
  RETURN w2;
END;

EXPORT Sparse_Autoencoder_IntBias (INTEGER4 NumberofFeatures, INTEGER4 NumberofHiddenLayerNodes) := FUNCTION
		B_Rec := RECORD
			STRING1 x:= '';
    END;
		B1_ := DATASET([{' '}], B_Rec);
		B2_ := DATASET([{' '}], B_Rec);

    Mat.Types.Element zeros(UNSIGNED4 c, UNSIGNED4 NumRows) := TRANSFORM
      SELF.x := ((c-1) % NumRows) + 1;
      SELF.y := 1;
      SELF.value := 0;
    END;
    //Creat the first bias vector with no=1 (bias that goes to second layer/hidden layer)
    b1 := NORMALIZE(B1_, NumberofHiddenLayerNodes, zeros(COUNTER, NumberofHiddenLayerNodes));
    b1no := Mat.MU.To(b1, 1);
    //Creat the first bias vector with no=1 (bias that goes to third layer/last layer)
    b2 := NORMALIZE(B2_, NumberofFeatures, zeros(COUNTER, NumberofFeatures));
    b2no := Mat.MU.To(b2, 2);
  RETURN b1no+b2no;
END;

EXPORT Sparse_Autoencoder_IntBias1 (INTEGER4 NumberofFeatures, INTEGER4 NumberofHiddenLayerNodes) := FUNCTION
		B_Rec := RECORD
			STRING1 x:= '';
    END;
		B1_ := DATASET([{' '}], B_Rec);
		B2_ := DATASET([{' '}], B_Rec);

    Mat.Types.Element zeros(UNSIGNED4 c, UNSIGNED4 NumRows) := TRANSFORM
      SELF.x := ((c-1) % NumRows) + 1;
      SELF.y := 1;
      SELF.value := 0;
    END;
    //Creat the first bias vector with no=1 (bias that goes to second layer/hidden layer)
    b1 := NORMALIZE(B1_, NumberofHiddenLayerNodes, zeros(COUNTER, NumberofHiddenLayerNodes));
    b1no := Mat.MU.To(b1, 1);
    //Creat the first bias vector with no=1 (bias that goes to third layer/last layer)
    b2 := NORMALIZE(B2_, NumberofFeatures, zeros(COUNTER, NumberofFeatures));
    b2no := Mat.MU.To(b2, 2);
  RETURN b1;
END;

EXPORT Sparse_Autoencoder_IntBias2 (INTEGER4 NumberofFeatures, INTEGER4 NumberofHiddenLayerNodes) := FUNCTION
		B_Rec := RECORD
			STRING1 x:= '';
    END;
		B1_ := DATASET([{' '}], B_Rec);
		B2_ := DATASET([{' '}], B_Rec);

    Mat.Types.Element zeros(UNSIGNED4 c, UNSIGNED4 NumRows) := TRANSFORM
      SELF.x := ((c-1) % NumRows) + 1;
      SELF.y := 1;
      SELF.value := 0;
    END;
    //Creat the first bias vector with no=1 (bias that goes to second layer/hidden layer)
    b1 := NORMALIZE(B1_, NumberofHiddenLayerNodes, zeros(COUNTER, NumberofHiddenLayerNodes));
    b1no := Mat.MU.To(b1, 1);
    //Creat the first bias vector with no=1 (bias that goes to third layer/last layer)
    b2 := NORMALIZE(B2_, NumberofFeatures, zeros(COUNTER, NumberofFeatures));
    b2no := Mat.MU.To(b2, 2);
  RETURN b2;
END;


EXPORT Sparse_Autoencoder_IntBias1_matrix (INTEGER4 NumberofFeatures, INTEGER4 NumberofHiddenLayerNodes, UNSIGNED8 m) := FUNCTION
		B_Rec := RECORD
			STRING1 x:= '';
    END;
		B1_ := DATASET([{' '}], B_Rec);
		
    Mat.Types.Element zeros(UNSIGNED4 c, UNSIGNED4 NumRows) := TRANSFORM
      SELF.x := ((c-1) % NumRows) + 1;
      SELF.y := ((c-1) DIV NumRows) + 1;
      SELF.value := 0;
    END;
    //Creat the first bias vector with no=1 (bias that goes to second layer/hidden layer)
    b1 := NORMALIZE(B1_, NumberofHiddenLayerNodes*m, zeros(COUNTER, NumberofHiddenLayerNodes));
    RETURN b1;
END;

EXPORT Sparse_Autoencoder_IntBias2_matrix (INTEGER4 NumberofFeatures, INTEGER4 NumberofHiddenLayerNodes, UNSIGNED8 m) := FUNCTION
		B_Rec := RECORD
			STRING1 x:= '';
    END;
		B2_ := DATASET([{' '}], B_Rec);
		
    Mat.Types.Element zeros(UNSIGNED4 c, UNSIGNED4 NumRows) := TRANSFORM
      SELF.x := ((c-1) % NumRows) + 1;
      SELF.y := ((c-1) DIV NumRows) + 1;
      SELF.value := 0;
    END;
    //Creat the first bias vector with no=1 (bias that goes to second layer/hidden layer)
    b2 := NORMALIZE(B2_, NumberofFeatures*m, zeros(COUNTER, NumberofFeatures));
    RETURN b2;
END;




EXPORT Sparse_Autoencoder_IntWeights_dataset (INTEGER4 NumberofFeatures, INTEGER4 NumberofHiddenLayerNodes) := FUNCTION
    r := SQRT(6) / SQRT (NumberofFeatures + NumberofHiddenLayerNodes + 1);
    //Generate a random number
    Produce_Random () := FUNCTION
      G := 1000000;
      Rnd_ := (RANDOM()%G) / (REAL8)G;
      Rnd := Rnd_ * 2 * r - r;
      RETURN Rnd;
    END;
    //New Randome Matrix Generator
    Mat.Types.Element RandGen(UNSIGNED4 c, UNSIGNED4 NumRows) := TRANSFORM
      SELF.x := ((c-1) % NumRows) + 1;
      SELF.y := ((c-1) DIV NumRows) + 1;
      SELF.value := Produce_Random();
    END;
    //Creat the first weight matrix with no=1 (weight matrix between layer 1 and layer 2)
    w1rows := NumberofHiddenLayerNodes;
    w1cols := NumberofFeatures;
    w1size := w1rows*w1cols;
    w1 := DATASET(w1size, RandGen(COUNTER, w1rows));
    w1no := Mat.MU.To(w1, 1);
    
    w2rows := NumberofFeatures;
    w2cols := NumberofHiddenLayerNodes;
    w2size := w1rows*w1cols;
    w2 := DATASET(w2size, RandGen(COUNTER, w2rows));
    w2no := Mat.MU.To(w2, 2);
  RETURN w1no + w2no;
END;
EXPORT Sparse_Autoencoder_IntBias_dataset (INTEGER4 NumberofFeatures, INTEGER4 NumberofHiddenLayerNodes) := FUNCTION
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
    Ones_Vec := DATASET(m, gen(COUNTER, m));
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
      //rhohat := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a2map, a2, Ones_VecMap, Ones_Vecdist, Hiddmap);
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
    EXPORT Mod := SAprm_MUE_out; 
  END;//END SA
  EXPORT LearnC (DATASET(Types.NumericField) Indep,DATASET(Mat.Types.MUElement) IntW, DATASET(Mat.Types.MUElement) Intb, REAL8 BETA=3, REAL8 sparsityParam=0.1 , REAL8 LAMBDA=0.001, REAL8 ALPHA=0.1, UNSIGNED2 MaxIter=100) := SA(Indep,IntW,Intb, BETA,sparsityParam,LAMBDA, ALPHA,  MaxIter).mod;

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
    Ones_Vec := DATASET(m, gen(COUNTER, m));
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

EXPORT SA_lbfgs_Compatible_alaki ( DATASET(Layout_Part) theta, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Part) TrainData , DATASET(Layout_Part) TrainLabel) := FUNCTION
    //extract sparse autoencoder parameters
    ddist := TrainData;
    m := CostFunc_params(id=1)[1].value;
    num_feat := CostFunc_params(id=2)[1].value;
    num_hid := CostFunc_params(id=3)[1].value;
    part_rows := CostFunc_params(id=4)[1].value;
    part_cols := CostFunc_params(id=5)[1].value;
    BETA := CostFunc_params(id=6)[1].value;
    sparsityParam := CostFunc_params(id=7)[1].value;
    LAMBDA := CostFunc_params(id=8)[1].value;
    
    m_1 := 1/m;
    sparsityParam_ := -1*sparsityParam;
    sparsityParam_1 := 1-sparsityParam;
    
     sizeRec := RECORD
      PBblas.Types.dimension_t m_rows;
      PBblas.Types.dimension_t m_cols;
      PBblas.Types.dimension_t f_b_rows;
      PBblas.Types.dimension_t f_b_cols;
    END;
    sizeTable := DATASET([{num_feat,m,part_rows,part_cols}], sizeRec);
    
    //Create block matrix d
    dmap := PBblas.Matrix_Map(sizeTable[1].m_rows,sizeTable[1].m_cols,sizeTable[1].f_b_rows,sizeTable[1].f_b_cols);
    
    //Create block matrix Ytmp
    Ymap := dmap;
    Ydist := ddist;
    //Creat block matrices for weights

    w1map := PBblas.Matrix_Map(num_hid, num_feat, num_hid, sizeTable[1].f_b_rows);
    w2map := PBblas.Matrix_Map(num_feat, num_hid, sizeTable[1].f_b_rows, num_hid);
     //each bias vector is converted to block format
    
    b1vecmap := PBblas.Matrix_Map(num_hid, 1, num_hid, 1);
    b2vecmap := PBblas.Matrix_Map(num_feat, 1, sizeTable[1].f_b_rows, 1);
    
    w1_partitions := w1map.partitions_used;
    w2_partitions := w2map.partitions_used;
    b1_partitions := b1vecmap.partitions_used;
    b2_partitions := b2vecmap.partitions_used;
    PBblas.Types.MUElement minuspart(Layout_Part l, UNSIGNED8 c ) := TRANSFORM
      SELF.partition_id := l.partition_id - c;
      SElF.no := 1;
      SELF := l;
    END;
    //w1m := w1dist;
    w1dist := theta (partition_id <= w1_partitions);
    //w2m := w2dist;
    w2m_ := theta (partition_id > w1_partitions AND partition_id <= w1_partitions + w2_partitions);
    w2dist  := PROJECT (w2m_, minuspart(LEFT, w1_partitions ),LOCAL);
    //b1v := b1vecdist;
    b1v_ := theta (partition_id > w1_partitions + w2_partitions AND partition_id <= w1_partitions + w2_partitions + b1_partitions);
    b1vecdist  := PROJECT (b1v_, minuspart (LEFT, w1_partitions + w2_partitions),LOCAL);
    
    //b2v := b2vecdist;
    b2v_ := theta (partition_id > w1_partitions + w2_partitions + b1_partitions AND partition_id <= w1_partitions + w2_partitions + b1_partitions + b2_partitions);
    b2vecdist  := PROJECT (b2v_, minuspart (LEFT, w1_partitions + w2_partitions + b1_partitions),LOCAL);
    
     //functions used
    PBblas.Types.value_t sp_reci(PBblas.Types.value_t v,PBblas.Types.dimension_t r,PBblas.Types.dimension_t c) := sparsityParam_/v;
    PBblas.Types.value_t sp_1_reci(PBblas.Types.value_t v,PBblas.Types.dimension_t r,PBblas.Types.dimension_t c) := sparsityParam_1/(1-v);
    //sparsity_delta=((-sparsityParam./rhohat)+((1-sparsityParam)./(1.-rhohat)));
    PBblas.Types.value_t sp_delta(PBblas.Types.value_t v,PBblas.Types.dimension_t r,PBblas.Types.dimension_t c) := (sparsityParam_/v)+(sparsityParam_1/(1-v));
    PBblas.Types.value_t siggrad(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := v*(1.0-v);
    PBblas.Types.value_t sigmoid(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := 1/(1+exp(-1*v));
    PBblas.Types.value_t pow2(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := v*v;
    //maps used
    b1map := PBblas.Matrix_Map(num_hid, m, num_hid, sizeTable[1].f_b_cols);
    b2map := PBblas.Matrix_Map(num_feat, m, sizeTable[1].f_b_rows, sizeTable[1].f_b_cols);
    a2map := b1map;
    a3map := b2map;
    HL_nodes := num_hid;//number of nodes in the hidden layer
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
    Ones_Vec := DATASET(m, gen(COUNTER, m));
    Ones_Vecdist := DMAT.Converted.FromCells(Ones_VecMap, Ones_Vec);
    //FF2 returns a2
    FF2(DATASET(Layout_Part) w1, DATASET(Layout_Part) b1v):= FUNCTION
      //b1m = repmat(b1v,1,m)
      b1m := PBblas.PB_dgemm(FALSE, TRUE, 1.0,b1vecmap, b1v, Ones_VecMap, Ones_Vecdist, b1map);
      //z2 = w1*X+b1;
      //z2 := PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1, dmap, ddist, b1map, b1m, 1.0); // gives MP closed error
      z2_ := PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1, dmap, ddist, b1map);
      z2 := PBblas.PB_daxpy(1.0, z2_, b1m);
      //a2 = sigmoid (z2);
      a2 := PBblas.Apply2Elements(b1map, z2, sigmoid);
      RETURN a2;
    END;//END FF2
    //FF3 returns a3
    FF3(DATASET(Layout_Part) w2,DATASET(Layout_Part) b2v, DATASET(Layout_Part) a2 ):= FUNCTION
      //b2m = repmat(b2v,1,m)
      b2m := PBblas.PB_dgemm(FALSE, TRUE, 1.0,b2vecmap, b2v, Ones_VecMap, Ones_Vecdist, b2map);
      //z3 = w2*a2+b2;
      //z3 := PBblas.PB_dgemm(FALSE, FALSE,1.0,w2map, w2, a2map, a2, b2map,b2m, 1.0); //gives MP closed error
      z3_ := PBblas.PB_dgemm(FALSE, FALSE,1.0,w2map, w2, a2map, a2, b2map);
      z3 := PBblas.PB_daxpy(1.0, z3_, b2m);
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
      //rhohat := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a2map, a2, Ones_VecMap, Ones_Vecdist, Hiddmap);
      rhohat := rhat;
      sparsity_delta := PBblas.Apply2Elements(Hiddmap, rhohat, sp_delta);
      siggrad_a2 := PBblas.Apply2Elements(a2map, a2, siggrad);
      repmat_sparsity_delta := PBblas.PB_dgemm(FALSE, TRUE, 1.0,  Hiddmap, sparsity_delta, Ones_VecMap, Ones_Vecdist, a2map);
      //d2_firstterm = (W2'*d3)+beta*repmat(sparsity_delta,1,m);
      //d2_firstterm := PBblas.PB_dgemm(TRUE, FALSE, 1.0, w2map, w2, a3map, d3, a2map, repmat_sparsity_delta, BETA); // MP close error
      d2_firstterm_ := PBblas.PB_dgemm(TRUE, FALSE, 1.0, w2map, w2, a3map, d3, a2map);
      d2_firstterm := PBblas.PB_daxpy(BETA, repmat_sparsity_delta, d2_firstterm_);
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
    emptyL := DATASET([], Layout_Part);
    //theta is the input weight and bias parameters for the sparseautoencoder
    //parameters are the parameters for the sparseautoencoder function
    //train_d is the train data to learn sparse autoencoder and calculate the gradient and the cost based on taht
    // train_l is empty in the sparse autoencoder (because it is an unsupervised learning)
    SparseParam_CostGradients4 :=  FUNCTION
      PBblas.Types.MUElement minuspart(Layout_Part l, UNSIGNED8 c ) := TRANSFORM
        SELF.partition_id := l.partition_id - c;
        SElF.no := 1;
        SELF := l;
      END;
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
      costField := DATASET ([{1,1,cost}],ML.Types.NumericField);
      one_map := PBblas.Matrix_Map(1,1,1,1);
      Cost_part_no := PBblas.MU.TO(ML.DMat.Converted.FromNumericFieldDS(costField,one_map),2);
      //convert w and b gradients to a datasets of layoutparts where partition_id differentiate them
      //w1_grad has partition_id from 1 to w1_partitions
      //w2_grad had partition_ds from w1_partitions+1 to w1_partitions + w2_partitions
      //b1_grad has partition_ids from w1_partitions + w2_partitions+1 to w1_partitions + w2_partitions+b1_partitions
      //b2_grad has partition_ids from w1_partitions + w2_partitions+b1_partitions+1 to w1_partitions + w2_partitions+b1_partitions+b2_partitions
      PBblas.Types.MUElement addpart(Layout_Part l, UNSIGNED8 c ) := TRANSFORM
        SELF.partition_id := l.partition_id + c;
        SElF.no := 1;
        SELF := l;
      END;
      wg1_reshape_no := Pbblas.MU.TO(wg1,1);
      wg2_reshape_no := PROJECT (wg2, addpart(LEFT, w1_partitions ), LOCAL);
      bg1_reshape_no := PROJECT (bg1, addpart (LEFT, w1_partitions + w2_partitions),LOCAL);
      bg2_reshape_no := PROJECT (bg2, addpart (LEFT, w1_partitions + w2_partitions + b1_partitions),LOCAL);
      theta_Part_no := wg1_reshape_no + wg2_reshape_no + bg1_reshape_no + bg2_reshape_no;
      RETURN wg1; 
			//RETURN theta_Part_no;
			
      //RETURN theta_Part_no;
    END;//END SparseParam_CostGradients4  
    RETURN SparseParam_CostGradients4;
   END;//END SA_lbfgs_Compatible_alaki


// This is Sparse Autoencoder function which is compatible with the function format the MinFunc (lbfgs algorithm) recives as input
// this function is later used in Sparse_Autoencoder_mine when calling LBFGS algorithm
// theta is the Sparse Autoencoder parameters (W1, W2, b1, b2) in Layout_Part format where different parameters are recognized by their partition_id
// w1 : partion_id is from 1 to w1.partitions_used
// w12 : parition_id is from w1.partitions_used+1 to w1.partitions_used+w2.partitions_used and so on
// CostFunc_params is the controlled parameters for Sparse Autoencoder, such as sparsityparam, etc.
EXPORT SA_lbfgs_Compatible ( DATASET(Layout_Part) theta, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Part) TrainData , DATASET(Layout_Part) TrainLabel) := FUNCTION
    //extract sparse autoencoder parameters
    ddist := TrainData;
    m := CostFunc_params(id=1)[1].value;
    num_feat := CostFunc_params(id=2)[1].value;
    num_hid := CostFunc_params(id=3)[1].value;
    part_rows := CostFunc_params(id=4)[1].value;
    part_cols := CostFunc_params(id=5)[1].value;
    BETA := CostFunc_params(id=6)[1].value;
    sparsityParam := CostFunc_params(id=7)[1].value;
    LAMBDA := CostFunc_params(id=8)[1].value;
    
    m_1 := 1/m;
    sparsityParam_ := -1*sparsityParam;
    sparsityParam_1 := 1-sparsityParam;
    
     sizeRec := RECORD
      PBblas.Types.dimension_t m_rows;
      PBblas.Types.dimension_t m_cols;
      PBblas.Types.dimension_t f_b_rows;
      PBblas.Types.dimension_t f_b_cols;
    END;
    sizeTable := DATASET([{num_feat,m,part_rows,part_cols}], sizeRec);
    
    //Create block matrix d
    dmap := PBblas.Matrix_Map(sizeTable[1].m_rows,sizeTable[1].m_cols,sizeTable[1].f_b_rows,sizeTable[1].f_b_cols);
    
    //Create block matrix Ytmp
    Ymap := dmap;
    Ydist := ddist;
    //Creat block matrices for weights

    w1map := PBblas.Matrix_Map(num_hid, num_feat, num_hid, sizeTable[1].f_b_rows);
    w2map := PBblas.Matrix_Map(num_feat, num_hid, sizeTable[1].f_b_rows, num_hid);
     //each bias vector is converted to block format
    
    b1vecmap := PBblas.Matrix_Map(num_hid, 1, num_hid, 1);
    b2vecmap := PBblas.Matrix_Map(num_feat, 1, sizeTable[1].f_b_rows, 1);
    
    w1_partitions := w1map.partitions_used;
    w2_partitions := w2map.partitions_used;
    b1_partitions := b1vecmap.partitions_used;
    b2_partitions := b2vecmap.partitions_used;
    PBblas.Types.MUElement minuspart(Layout_Part l, UNSIGNED8 c ) := TRANSFORM
      SELF.partition_id := l.partition_id - c;
      SElF.no := 1;
      SELF := l;
    END;
    //w1m := w1dist;
    w1dist := theta (partition_id <= w1_partitions);
    //w2m := w2dist;
    w2m_ := theta (partition_id > w1_partitions AND partition_id <= w1_partitions + w2_partitions);
    w2dist  := PROJECT (w2m_, minuspart(LEFT, w1_partitions ),LOCAL);
    //b1v := b1vecdist;
    b1v_ := theta (partition_id > w1_partitions + w2_partitions AND partition_id <= w1_partitions + w2_partitions + b1_partitions);
    b1vecdist  := PROJECT (b1v_, minuspart (LEFT, w1_partitions + w2_partitions),LOCAL);
    
    //b2v := b2vecdist;
    b2v_ := theta (partition_id > w1_partitions + w2_partitions + b1_partitions AND partition_id <= w1_partitions + w2_partitions + b1_partitions + b2_partitions);
    b2vecdist  := PROJECT (b2v_, minuspart (LEFT, w1_partitions + w2_partitions + b1_partitions),LOCAL);
    
     //functions used
    PBblas.Types.value_t sp_reci(PBblas.Types.value_t v,PBblas.Types.dimension_t r,PBblas.Types.dimension_t c) := sparsityParam_/v;
    PBblas.Types.value_t sp_1_reci(PBblas.Types.value_t v,PBblas.Types.dimension_t r,PBblas.Types.dimension_t c) := sparsityParam_1/(1-v);
    //sparsity_delta=((-sparsityParam./rhohat)+((1-sparsityParam)./(1.-rhohat)));
    PBblas.Types.value_t sp_delta(PBblas.Types.value_t v,PBblas.Types.dimension_t r,PBblas.Types.dimension_t c) := (sparsityParam_/v)+(sparsityParam_1/(1-v));
    PBblas.Types.value_t siggrad(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := v*(1.0-v);
    PBblas.Types.value_t sigmoid(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := 1/(1+exp(-1*v));
    PBblas.Types.value_t pow2(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := v*v;
    //maps used
    b1map := PBblas.Matrix_Map(num_hid, m, num_hid, sizeTable[1].f_b_cols);
    b2map := PBblas.Matrix_Map(num_feat, m, sizeTable[1].f_b_rows, sizeTable[1].f_b_cols);
    a2map := b1map;
    a3map := b2map;
    HL_nodes := num_hid;//number of nodes in the hidden layer
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
    Ones_Vec := DATASET(m, gen(COUNTER, m));
    Ones_Vecdist := DMAT.Converted.FromCells(Ones_VecMap, Ones_Vec);
    //FF2 returns a2
    FF2(DATASET(Layout_Part) w1, DATASET(Layout_Part) b1v):= FUNCTION
      //b1m = repmat(b1v,1,m)
      b1m := PBblas.PB_dgemm(FALSE, TRUE, 1.0,b1vecmap, b1v, Ones_VecMap, Ones_Vecdist, b1map);
      //z2 = w1*X+b1;
      //z2 := PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1, dmap, ddist, b1map, b1m, 1.0); // gives MP closed error
      z2_ := PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1, dmap, ddist, b1map);
      z2 := PBblas.PB_daxpy(1.0, z2_, b1m);
      //a2 = sigmoid (z2);
      a2 := PBblas.Apply2Elements(b1map, z2, sigmoid);
      RETURN a2;
    END;//END FF2
    //FF3 returns a3
    FF3(DATASET(Layout_Part) w2,DATASET(Layout_Part) b2v, DATASET(Layout_Part) a2 ):= FUNCTION
      //b2m = repmat(b2v,1,m)
      b2m := PBblas.PB_dgemm(FALSE, TRUE, 1.0,b2vecmap, b2v, Ones_VecMap, Ones_Vecdist, b2map);
      //z3 = w2*a2+b2;
      //z3 := PBblas.PB_dgemm(FALSE, FALSE,1.0,w2map, w2, a2map, a2, b2map,b2m, 1.0); //gives MP closed error
      z3_ := PBblas.PB_dgemm(FALSE, FALSE,1.0,w2map, w2, a2map, a2, b2map);
      z3 := PBblas.PB_daxpy(1.0, z3_, b2m);
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
      //rhohat := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a2map, a2, Ones_VecMap, Ones_Vecdist, Hiddmap);
      rhohat := rhat;
      sparsity_delta := PBblas.Apply2Elements(Hiddmap, rhohat, sp_delta);
      siggrad_a2 := PBblas.Apply2Elements(a2map, a2, siggrad);
      repmat_sparsity_delta := PBblas.PB_dgemm(FALSE, TRUE, 1.0,  Hiddmap, sparsity_delta, Ones_VecMap, Ones_Vecdist, a2map);
      //d2_firstterm = (W2'*d3)+beta*repmat(sparsity_delta,1,m);
      //d2_firstterm := PBblas.PB_dgemm(TRUE, FALSE, 1.0, w2map, w2, a3map, d3, a2map, repmat_sparsity_delta, BETA); // MP close error
      d2_firstterm_ := PBblas.PB_dgemm(TRUE, FALSE, 1.0, w2map, w2, a3map, d3, a2map);
      d2_firstterm := PBblas.PB_daxpy(BETA, repmat_sparsity_delta, d2_firstterm_);
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
    emptyL := DATASET([], Layout_Part);
    //theta is the input weight and bias parameters for the sparseautoencoder
    //parameters are the parameters for the sparseautoencoder function
    //train_d is the train data to learn sparse autoencoder and calculate the gradient and the cost based on taht
    // train_l is empty in the sparse autoencoder (because it is an unsupervised learning)
    SparseParam_CostGradients4 :=  FUNCTION
      PBblas.Types.MUElement minuspart(Layout_Part l, UNSIGNED8 c ) := TRANSFORM
        SELF.partition_id := l.partition_id - c;
        SElF.no := 1;
        SELF := l;
      END;
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
      costField := DATASET ([{1,1,cost}],ML.Types.NumericField);
      one_map := PBblas.Matrix_Map(1,1,1,1);
      Cost_part_no := PBblas.MU.TO(ML.DMat.Converted.FromNumericFieldDS(costField,one_map),2);
      //convert w and b gradients to a datasets of layoutparts where partition_id differentiate them
      //w1_grad has partition_id from 1 to w1_partitions
      //w2_grad had partition_ds from w1_partitions+1 to w1_partitions + w2_partitions
      //b1_grad has partition_ids from w1_partitions + w2_partitions+1 to w1_partitions + w2_partitions+b1_partitions
      //b2_grad has partition_ids from w1_partitions + w2_partitions+b1_partitions+1 to w1_partitions + w2_partitions+b1_partitions+b2_partitions
      PBblas.Types.MUElement addpart(Layout_Part l, UNSIGNED8 c ) := TRANSFORM
        SELF.partition_id := l.partition_id + c;
        SElF.no := 1;
        SELF := l;
      END;
      wg1_reshape_no := Pbblas.MU.TO(wg1,1);
      wg2_reshape_no := PROJECT (wg2, addpart(LEFT, w1_partitions ), LOCAL);
      bg1_reshape_no := PROJECT (bg1, addpart (LEFT, w1_partitions + w2_partitions),LOCAL);
      bg2_reshape_no := PROJECT (bg2, addpart (LEFT, w1_partitions + w2_partitions + b1_partitions),LOCAL);
      theta_Part_no := wg1_reshape_no + wg2_reshape_no + bg1_reshape_no + bg2_reshape_no;
      RETURN theta_Part_no + Cost_part_no; 
			//RETURN theta_Part_no;
			
      //RETURN theta_Part_no;
    END;//END SparseParam_CostGradients4  
    RETURN SparseParam_CostGradients4;
   END;//END SA_lbfgs_Compatible








EXPORT SA_lbfgs_Compatible_new ( DATASET(Layout_Part) theta, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Part) TrainData , DATASET(Layout_Part) TrainLabel) := FUNCTION
    //extract sparse autoencoder parameters
    ddist := TrainData;
    m := CostFunc_params(id=1)[1].value;
    num_feat := CostFunc_params(id=2)[1].value;
    num_hid := CostFunc_params(id=3)[1].value;
    part_rows := CostFunc_params(id=4)[1].value;
    part_cols := CostFunc_params(id=5)[1].value;
    BETA := CostFunc_params(id=6)[1].value;
    sparsityParam := CostFunc_params(id=7)[1].value;
    LAMBDA := CostFunc_params(id=8)[1].value;
    
    m_1 := 1/m;
    sparsityParam_ := -1*sparsityParam;
    sparsityParam_1 := 1-sparsityParam;
    
     sizeRec := RECORD
      PBblas.Types.dimension_t m_rows;
      PBblas.Types.dimension_t m_cols;
      PBblas.Types.dimension_t f_b_rows;
      PBblas.Types.dimension_t f_b_cols;
    END;
    sizeTable := DATASET([{num_feat,m,num_feat,part_cols}], sizeRec);
    
    //Create block matrix d
    dmap := PBblas.Matrix_Map(num_feat,m,num_feat,part_cols);
    
    //Create block matrix Ytmp
    Ymap := dmap;
    Ydist := ddist;
    //Creat block matrices for weights

    w1map := PBblas.Matrix_Map(num_hid, num_feat, num_hid, num_feat);
    w2map := PBblas.Matrix_Map(num_feat, num_hid, num_feat, num_hid);
     //each bias vector is converted to block format
    
    b1vecmap := PBblas.Matrix_Map(num_hid, 1, num_hid, 1);
    b2vecmap := PBblas.Matrix_Map(num_feat, 1, num_feat, 1);
    
    w1_partitions := w1map.partitions_used;
    w2_partitions := w2map.partitions_used;
    b1_partitions := b1vecmap.partitions_used;
    b2_partitions := b2vecmap.partitions_used;
    PBblas.Types.MUElement minuspart(Layout_Part l, UNSIGNED8 c ) := TRANSFORM
      SELF.partition_id := l.partition_id - c;
      SElF.no := 1;
      SELF := l;
    END;
    //w1m := w1dist;
    w1dist := theta (partition_id <= w1_partitions);
    //w2m := w2dist;
    w2m_ := theta (partition_id > w1_partitions AND partition_id <= w1_partitions + w2_partitions);
    w2dist  := PROJECT (w2m_, minuspart(LEFT, w1_partitions ),LOCAL);
    //b1v := b1vecdist;
    b1v_ := theta (partition_id > w1_partitions + w2_partitions AND partition_id <= w1_partitions + w2_partitions + b1_partitions);
    b1vecdist  := PROJECT (b1v_, minuspart (LEFT, w1_partitions + w2_partitions),LOCAL);
    
    //b2v := b2vecdist;
    b2v_ := theta (partition_id > w1_partitions + w2_partitions + b1_partitions AND partition_id <= w1_partitions + w2_partitions + b1_partitions + b2_partitions);
    b2vecdist  := PROJECT (b2v_, minuspart (LEFT, w1_partitions + w2_partitions + b1_partitions),LOCAL);
    
     //functions used
    PBblas.Types.value_t sp_reci(PBblas.Types.value_t v,PBblas.Types.dimension_t r,PBblas.Types.dimension_t c) := sparsityParam_/v;
    PBblas.Types.value_t sp_1_reci(PBblas.Types.value_t v,PBblas.Types.dimension_t r,PBblas.Types.dimension_t c) := sparsityParam_1/(1-v);
    //sparsity_delta=((-sparsityParam./rhohat)+((1-sparsityParam)./(1.-rhohat)));
    PBblas.Types.value_t sp_delta(PBblas.Types.value_t v,PBblas.Types.dimension_t r,PBblas.Types.dimension_t c) := (sparsityParam_/v)+(sparsityParam_1/(1-v));
    PBblas.Types.value_t siggrad(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := v*(1.0-v);
    PBblas.Types.value_t sigmoid(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := 1/(1+exp(-1*v));
    PBblas.Types.value_t pow2(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := v*v;
    //maps used
    b1map := PBblas.Matrix_Map(num_hid, m, num_hid, sizeTable[1].f_b_cols);
    b2map := PBblas.Matrix_Map(num_feat, m, sizeTable[1].f_b_rows, sizeTable[1].f_b_cols);
    a2map := b1map;
    a3map := b2map;
    HL_nodes := num_hid;//number of nodes in the hidden layer
    Hiddmap := b1vecmap;
    //onevec for calculating rhohat
    Ones_VecMap := PBblas.Matrix_Map(m, 1, part_cols, 1);
    //New Vector Generator
    Layout_Cell gen(UNSIGNED4 c, UNSIGNED4 NumRows) := TRANSFORM
      SELF.x := ((c-1) % NumRows) + 1;
      SELF.y := ((c-1) DIV NumRows) + 1;
      SELF.v := 1;
    END;
    //Create Ones Vector for the calculations in the step fucntion
    Ones_Vec := DATASET(m, gen(COUNTER, m));
    Ones_Vecdist := DMAT.Converted.FromCells(Ones_VecMap, Ones_Vec);
    //FF2 returns a2
    FF2(DATASET(Layout_Part) w1, DATASET(Layout_Part) b1v):= FUNCTION
      //b1m = repmat(b1v,1,m)
      b1m := PBblas.PB_dgemm(FALSE, TRUE, 1.0,b1vecmap, b1v, Ones_VecMap, Ones_Vecdist, b1map);
      //z2 = w1*X+b1;
      //z2 := PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1, dmap, ddist, b1map, b1m, 1.0); // gives MP closed error
      z2_ := PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1, dmap, ddist, b1map);
      z2 := PBblas.PB_daxpy(1.0, z2_, b1m);
      //a2 = sigmoid (z2);
      a2 := PBblas.Apply2Elements(b1map, z2, sigmoid);
      RETURN a2;
    END;//END FF2
		
		
		 FF2_(DATASET(Layout_Part) w1, DATASET(Layout_Part) b1v):= FUNCTION
      //b1m = repmat(b1v,1,m)
      b1m := PBblas.PB_dgemm(FALSE, TRUE, 1.0,b1vecmap, b1v, Ones_VecMap, Ones_Vecdist, b1map);
      //z2 = w1*X+b1;
      //z2 := PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1, dmap, ddist, b1map, b1m, 1.0); // gives MP closed error
      z2_ := PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1, dmap, ddist, b1map);
      z2 := PBblas.PB_daxpy(1.0, z2_, b1m);
      //a2 = sigmoid (z2);
      a2 := PBblas.Apply2Elements(b1map, z2, sigmoid);
      RETURN a2;
    END;//END FF2
		
		
    //FF3 returns a3
    FF3(DATASET(Layout_Part) w2,DATASET(Layout_Part) b2v, DATASET(Layout_Part) a2 ):= FUNCTION
      //b2m = repmat(b2v,1,m)
      b2m := PBblas.PB_dgemm(FALSE, TRUE, 1.0,b2vecmap, b2v, Ones_VecMap, Ones_Vecdist, b2map);
      //z3 = w2*a2+b2;
      //z3 := PBblas.PB_dgemm(FALSE, FALSE,1.0,w2map, w2, a2map, a2, b2map,b2m, 1.0); //gives MP closed error
      z3_ := PBblas.PB_dgemm(FALSE, FALSE,1.0,w2map, w2, a2map, a2, b2map);
      z3 := PBblas.PB_daxpy(1.0, z3_, b2m);
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
      //rhohat := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a2map, a2, Ones_VecMap, Ones_Vecdist, Hiddmap);
      rhohat := rhat;
      sparsity_delta := PBblas.Apply2Elements(Hiddmap, rhohat, sp_delta);
      siggrad_a2 := PBblas.Apply2Elements(a2map, a2, siggrad);
      repmat_sparsity_delta := PBblas.PB_dgemm(FALSE, TRUE, 1.0,  Hiddmap, sparsity_delta, Ones_VecMap, Ones_Vecdist, a2map);
      //d2_firstterm = (W2'*d3)+beta*repmat(sparsity_delta,1,m);
      //d2_firstterm := PBblas.PB_dgemm(TRUE, FALSE, 1.0, w2map, w2, a3map, d3, a2map, repmat_sparsity_delta, BETA); // MP close error
      d2_firstterm_ := PBblas.PB_dgemm(TRUE, FALSE, 1.0, w2map, w2, a3map, d3, a2map);
      d2_firstterm := PBblas.PB_daxpy(BETA, repmat_sparsity_delta, d2_firstterm_);
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
    emptyL := DATASET([], Layout_Part);
    //theta is the input weight and bias parameters for the sparseautoencoder
    //parameters are the parameters for the sparseautoencoder function
    //train_d is the train data to learn sparse autoencoder and calculate the gradient and the cost based on taht
    // train_l is empty in the sparse autoencoder (because it is an unsupervised learning)
    SparseParam_CostGradients4 :=  FUNCTION
      PBblas.Types.MUElement minuspart(Layout_Part l, UNSIGNED8 c ) := TRANSFORM
        SELF.partition_id := l.partition_id - c;
        SElF.no := 1;
        SELF := l;
      END;
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
      costField := DATASET ([{1,1,cost}],ML.Types.NumericField);
      one_map := PBblas.Matrix_Map(1,1,1,1);
      Cost_part_no := PBblas.MU.TO(ML.DMat.Converted.FromNumericFieldDS(costField,one_map),2);
      //convert w and b gradients to a datasets of layoutparts where partition_id differentiate them
      //w1_grad has partition_id from 1 to w1_partitions
      //w2_grad had partition_ds from w1_partitions+1 to w1_partitions + w2_partitions
      //b1_grad has partition_ids from w1_partitions + w2_partitions+1 to w1_partitions + w2_partitions+b1_partitions
      //b2_grad has partition_ids from w1_partitions + w2_partitions+b1_partitions+1 to w1_partitions + w2_partitions+b1_partitions+b2_partitions
      PBblas.Types.MUElement addpart(Layout_Part l, UNSIGNED8 c ) := TRANSFORM
        SELF.partition_id := l.partition_id + c;
        SElF.no := 1;
        SELF := l;
      END;
      wg1_reshape_no := Pbblas.MU.TO(wg1,1);
      wg2_reshape_no := PROJECT (wg2, addpart(LEFT, w1_partitions ), LOCAL);
      bg1_reshape_no := PROJECT (bg1, addpart (LEFT, w1_partitions + w2_partitions),LOCAL);
      bg2_reshape_no := PROJECT (bg2, addpart (LEFT, w1_partitions + w2_partitions + b1_partitions),LOCAL);
      theta_Part_no := wg1_reshape_no + wg2_reshape_no + bg1_reshape_no + bg2_reshape_no;
      //RETURN theta_Part_no + Cost_part_no; orig
			RETURN theta_Part_no + Cost_part_no;
      //RETURN theta_Part_no;
    END;//END SparseParam_CostGradients4  
    RETURN SparseParam_CostGradients4;
   END;//END SA_lbfgs_Compatible_new













EXPORT SA_lbfgs_Compatible2 ( DATASET(Layout_Part) theta, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Part) TrainData , DATASET(Layout_Part) TrainLabel) := FUNCTION

Layout_Target := PBblas.Types.Layout_Target;
WX(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_bb, DATASET(Layout_Part) bb_in, PBblas.IMatrix_Map map_c, REAL8 beta=0.0) := FUNCTION
SET OF PBblas.Types.value_t empty_array := [];
// First check maps for compatability.  Normalize for transpose operations.
  a_matrix_rows := map_a.matrix_rows;
  a_matrix_cols := map_a.matrix_cols;
  a_row_blocks  := map_a.row_blocks;
  a_col_blocks  := map_a.col_blocks;
  b_matrix_rows := map_b.matrix_rows;
  b_matrix_cols := map_b.matrix_cols;
  b_row_blocks  := map_b.row_blocks;
  b_col_blocks  := map_b.col_blocks;
  c_matrix_rows := map_c.matrix_rows;
  c_matrix_cols := map_c.matrix_cols;
  c_row_blocks  := map_c.row_blocks;
  c_col_blocks  := map_c.col_blocks;
  A := ASSERT(A_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' + ' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(a_matrix_rows=c_matrix_rows AND a_row_blocks=c_row_blocks,
                    'A-C: ' + 'Arows: ' + a_matrix_rows +' Crows '+c_matrix_rows+' '+ 
                              'Ablocks: ' + a_row_blocks +' Cblocks '+c_row_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	B := ASSERT(B_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' +' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(b_matrix_cols=c_matrix_cols AND b_col_blocks=c_col_blocks,
                    'B-C: ' + 'Brows: ' + b_matrix_cols +' Ccols '+c_matrix_cols+' '+ 
                              'Bblocks: ' + b_row_blocks +' Cblocks '+c_col_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));

//
  Layout_Target cvt(Layout_Part par, INTEGER c, BOOLEAN keepRow) := TRANSFORM
    s_block_row       := par.block_row;
    s_block_col       := par.block_col;
    part_id_new_row   := map_c.assigned_part(c, s_block_col);
    part_id_new_col   := map_c.assigned_part(s_block_row, c);
    partition_id      := IF(keepRow, part_id_new_col, part_id_new_row);
    SELF.t_node_id    := map_c.assigned_node(partition_id);
    SELF.t_part_id    := partition_id;
    SELF.t_block_row  := IF(keepRow, s_block_row, c);
    SELF.t_block_col  := IF(keepRow, c, s_block_col);
    SELF.t_term       := IF(keepRow, s_block_col, s_block_row);
    SELF              := par;
  END;

  // A: copy of weight matrix goes to each column of X
  a_fact := map_b.col_blocks; // The number of time weight matrix (A) has to be distributed is the number of columns on matrix X (B)
  a_work := NORMALIZE(A, a_fact, cvt(LEFT, COUNTER, TRUE));
  a_dist := DISTRIBUTE(a_work, t_node_id);
  a_sort := a_dist;// only one partition in each node, so no need to sort
  // B: copy of each cell in a column goes to a row
  b_fact := map_a.row_blocks;
  b_work := PROJECT(B, cvt(LEFT, COUNTER, FALSE), LOCAL);
  b_dist := b_work; // no need to distribute as it is already distributed
  b_sort := b_dist; // only one partition in each node, so no need to sort
	
	
	
	// Elem := {PBblas.Types.value_t v};
	// Elem_col := {PBblas.Types.value_t v, UNSIGNED8 v_col:=1};
	// Layout_Target rep_bb (Layout_Target x) := TRANSFORM
		// elemsX_ := DATASET(x.mat_part, Elem);
		// elemsX := PROJECT (elemsX_, TRANSFORM(Elem_col, SELF := LEFT));
		// Elem_col cvt2(Elem_col par, INTEGER c) := TRANSFORM
			// SELF := par;
		// END;
		// repeatedelemsX := NORMALIZE(elemsX, bb_fact, cvt2(LEFT));
		// self.mat_part := SET(repeatedelemsX, v);
		// SELF := x;
	
	// END;
	
	
  // Multiply
  Layout_Part mul(Layout_Target a_part, Layout_Target b_part):=TRANSFORM
    part_id     := a_part.t_part_id;    //arbitrary choice
    part_a_cols := a_part.part_cols;
    part_a_rows := a_part.part_rows;
    part_b_rows := b_part.part_rows;
    part_c_rows := map_c.part_rows(part_id);
    part_c_cols := map_c.part_cols(part_id);
    part_c_first_row  := map_c.first_row(part_id);
    part_c_first_col  := map_c.first_col(part_id);
    k := part_a_cols;
    SELF.partition_id := part_id;
    SELF.node_id      := a_part.t_node_id;
    SELF.block_row    := a_part.t_block_row;
    SELF.block_col    := a_part.t_block_col;
    SELF.first_row    := map_c.first_row(part_id);
    SELF.part_rows    := part_c_rows;
    SELF.first_col    := part_c_first_col;
    SELF.part_cols    := part_c_cols;
    SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
                                    part_c_rows, part_c_cols, k,
                                    1.0, a_part.mat_part, b_part.mat_part,
                                    0.0, empty_array);
  END;
  ab_prod := JOIN(a_sort, b_sort,
                  LEFT.t_part_id=RIGHT.t_part_id AND LEFT.t_term=RIGHT.t_term,
                  mul(LEFT,RIGHT), LOCAL);




   // Apply beta


	
	
	// N := 32*32*3*1000;
	// Layout_Target sumTerms(Layout_Target cumm, Layout_Target term) := TRANSFORM
    // SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term.mat_part, 1);
    // SELF := cumm;
  // END;
	// sumres := ROLLUP(a_sort, sumTerms(LEFT, RIGHT), partition_id);
	
	
	mymy := a_sort;
	myformat := RECORD
    mymy.node_id;
    mymy.partition_id;
    mymy.block_row;
    mymy.block_col;
    mymy.first_row;
    mymy.part_rows;
    mymy.first_col;
    mymy.part_cols;
		mymy.t_part_id;
    mymy.t_node_id;
    mymy.t_block_row;
    mymy.t_block_col;
    mymy.t_term;
		INTEGER real_node := STD.System.Thorlib.Node();
END;
mymy2 := ab_prod;
myformat2 := RECORD
    mymy2.node_id;
    mymy2.partition_id;
    mymy2.block_row;
    mymy2.block_col;
    mymy2.first_row;
    mymy2.part_rows;
    mymy2.first_col;
    mymy2.part_cols;
		INTEGER real_node := STD.System.Thorlib.Node();
END;
	rslt := TABLE(mymy2,myformat2,LOCAL); 
  RETURN rslt;
END; // END WX




WX_2(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_bb, DATASET(Layout_Part) bb_in, PBblas.IMatrix_Map map_c, REAL8 beta=0.0) := FUNCTION
SET OF PBblas.Types.value_t empty_array := [];
// First check maps for compatability.  Normalize for transpose operations.
  a_matrix_rows := map_a.matrix_rows;
  a_matrix_cols := map_a.matrix_cols;
  a_row_blocks  := map_a.row_blocks;
  a_col_blocks  := map_a.col_blocks;
  b_matrix_rows := map_b.matrix_rows;
  b_matrix_cols := map_b.matrix_cols;
  b_row_blocks  := map_b.row_blocks;
  b_col_blocks  := map_b.col_blocks;
  c_matrix_rows := map_c.matrix_rows;
  c_matrix_cols := map_c.matrix_cols;
  c_row_blocks  := map_c.row_blocks;
  c_col_blocks  := map_c.col_blocks;
  A := ASSERT(A_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' + ' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(a_matrix_rows=c_matrix_rows AND a_row_blocks=c_row_blocks,
                    'A-C: ' + 'Arows: ' + a_matrix_rows +' Crows '+c_matrix_rows+' '+ 
                              'Ablocks: ' + a_row_blocks +' Cblocks '+c_row_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	B := ASSERT(B_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' +' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(b_matrix_cols=c_matrix_cols AND b_col_blocks=c_col_blocks,
                    'B-C: ' + 'Brows: ' + b_matrix_cols +' Ccols '+c_matrix_cols+' '+ 
                              'Bblocks: ' + b_row_blocks +' Cblocks '+c_col_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));

//
  
	
	
	
	// Elem := {PBblas.Types.value_t v};
	// Elem_col := {PBblas.Types.value_t v, UNSIGNED8 v_col:=1};
	// Layout_Target rep_bb (Layout_Target x) := TRANSFORM
		// elemsX_ := DATASET(x.mat_part, Elem);
		// elemsX := PROJECT (elemsX_, TRANSFORM(Elem_col, SELF := LEFT));
		// Elem_col cvt2(Elem_col par, INTEGER c) := TRANSFORM
			// SELF := par;
		// END;
		// repeatedelemsX := NORMALIZE(elemsX, bb_fact, cvt2(LEFT));
		// self.mat_part := SET(repeatedelemsX, v);
		// SELF := x;
	
	// END;
	
	
  //multiply
	
	Layout_Part mul2(Layout_Part b_part, Layout_Part a_part):=TRANSFORM
    part_id     := b_part.partition_id;    //arbitrary choice
    part_a_cols := a_part.part_cols;
    part_a_rows := a_part.part_rows;
    part_b_rows := b_part.part_rows;
    part_c_rows := map_c.part_rows(part_id);
    part_c_cols := map_c.part_cols(part_id);
    part_c_first_row  := map_c.first_row(part_id);
    part_c_first_col  := map_c.first_col(part_id);
    k := part_a_cols;
    SELF.partition_id := b_part.partition_id;
    SELF.node_id      := b_part.node_id;
    SELF.block_row    := b_part.block_row;
    SELF.block_col    := b_part.block_col;
    SELF.first_row    := map_c.first_row(part_id);
    SELF.part_rows    := part_c_rows;
    SELF.first_col    := part_c_first_col;
    SELF.part_cols    := part_c_cols;
    SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
                                    part_c_rows, part_c_cols, k,
                                    1.0, a_part.mat_part, b_part.mat_part,
                                    0.0, empty_array);
  END;
  ab_prod := JOIN(B, A,TRUE , mul2(LEFT,RIGHT),ALL);




   // Apply beta

// Sum terms
  Layout_Part sumTerms(Layout_Part cumm, Layout_Part term) := TRANSFORM
		cumm_part_cols := cumm.part_cols;
    N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
		Elem := {PBblas.Types.value_t v};
		Elem_r := {PBblas.Types.value_t v, UNSIGNED r};
		elems := DATASET(term.mat_part, Elem);
		Elem_r rep (Elem l, UNSIGNED c) := TRANSFORM
			SELF.r := c;
			SELF := l;
		END;
		elems_rep := NORMALIZE(elems, cumm_part_cols, rep(LEFT, COUNTER));
		elems_rep_sort := SORT(elems_rep, r);
		term_rep_set := SET (elems_rep_sort, v);
    SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term_rep_set, 1);
		SELF.partition_id := cumm.partition_id;
    SELF := cumm;
  END;
	
	 ab_bb := JOIN(ab_prod, bb_in,TRUE , sumTerms(LEFT,RIGHT),ALL);

cumm := B[1];
cumm_part_cols := cumm.part_cols;
term := bb_in[1];
Elem := {PBblas.Types.value_t v};
		Elem_r := {PBblas.Types.value_t v, UNSIGNED r:=1};
		elems := DATASET(term.mat_part, Elem);
		Elem_r rep (Elem l, UNSIGNED c) := TRANSFORM
			SELF.r := c;
			SELF := l;
		END;
		elems_rep := NORMALIZE(elems, cumm_part_cols, rep(LEFT, COUNTER));
		elems_rep_sort := SORT(elems_rep, r);
		term_rep_set := SET (elems_rep_sort, v);
	
	// N := 32*32*3*1000;
	// Layout_Target sumTerms(Layout_Target cumm, Layout_Target term) := TRANSFORM
    // SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term.mat_part, 1);
    // SELF := cumm;
  // END;
	// sumres := ROLLUP(a_sort, sumTerms(LEFT, RIGHT), partition_id);
	
	
	
mymy2 := ab_bb;
myformat2 := RECORD
    mymy2.node_id;
    mymy2.partition_id;
    mymy2.block_row;
    mymy2.block_col;
    mymy2.first_row;
    mymy2.part_rows;
    mymy2.first_col;
    mymy2.part_cols;
		//mymy2.mat_part;
		INTEGER real_node := STD.System.Thorlib.Node();
END;
	rslt := TABLE(mymy2,myformat2,LOCAL);
	//rslt := A;
  RETURN rslt;
END; // END WX_2

// w*x + repmat (b, 1,m)
//A_in :w
//B_in :x
//bb_in : bias vector (b)
WX_repmatb(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_bb, DATASET(Layout_Part) bb_in, PBblas.IMatrix_Map map_c, REAL8 beta=0.0) := FUNCTION
SET OF PBblas.Types.value_t empty_array := [];
// First check maps for compatability.  Normalize for transpose operations.
  a_matrix_rows := map_a.matrix_rows;
  a_matrix_cols := map_a.matrix_cols;
  a_row_blocks  := map_a.row_blocks;
  a_col_blocks  := map_a.col_blocks;
  b_matrix_rows := map_b.matrix_rows;
  b_matrix_cols := map_b.matrix_cols;
  b_row_blocks  := map_b.row_blocks;
  b_col_blocks  := map_b.col_blocks;
  c_matrix_rows := map_c.matrix_rows;
  c_matrix_cols := map_c.matrix_cols;
  c_row_blocks  := map_c.row_blocks;
  c_col_blocks  := map_c.col_blocks;
  A := ASSERT(A_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' + ' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(a_matrix_rows=c_matrix_rows AND a_row_blocks=c_row_blocks,
                    'A-C: ' + 'Arows: ' + a_matrix_rows +' Crows '+c_matrix_rows+' '+ 
                              'Ablocks: ' + a_row_blocks +' Cblocks '+c_row_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	B := ASSERT(B_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' +' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(b_matrix_cols=c_matrix_cols AND b_col_blocks=c_col_blocks,
                    'B-C: ' + 'Brows: ' + b_matrix_cols +' Ccols '+c_matrix_cols+' '+ 
                              'Bblocks: ' + b_row_blocks +' Cblocks '+c_col_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	
  //multiply
	
	Layout_Part mul2(Layout_Part b_part, Layout_Part a_part):=TRANSFORM
    part_id     := b_part.partition_id;    //arbitrary choice
    part_a_cols := a_part.part_cols;
    part_a_rows := a_part.part_rows;
    part_b_rows := b_part.part_rows;
    part_c_rows := map_c.part_rows(part_id);
    part_c_cols := map_c.part_cols(part_id);
    part_c_first_row  := map_c.first_row(part_id);
    part_c_first_col  := map_c.first_col(part_id);
    k := part_a_cols;
    SELF.partition_id := b_part.partition_id;
    SELF.node_id      := b_part.node_id;
    SELF.block_row    := b_part.block_row;
    SELF.block_col    := b_part.block_col;
    SELF.first_row    := map_c.first_row(part_id);
    SELF.part_rows    := part_c_rows;
    SELF.first_col    := part_c_first_col;
    SELF.part_cols    := part_c_cols;
    SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
                                    part_c_rows, part_c_cols, k,
                                    1.0, a_part.mat_part, b_part.mat_part,
                                    0.0, empty_array);
  END;
  ab_prod := JOIN(B, A,TRUE , mul2(LEFT,RIGHT),ALL); // Each A (weight matrix) is copied in each B (X matrix) node

// add bias vector to each columns of X
// each bias vector (b) is copied (ALL JOIN) to each node of X. The bias vector is then normalized to repeat it to the number of columns of X in that node and then the two are added
  Layout_Part addbias(Layout_Part cumm, Layout_Part term) := TRANSFORM
		cumm_part_cols := cumm.part_cols;
    N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
		Elem := {PBblas.Types.value_t v};
		Elem_r := {PBblas.Types.value_t v, UNSIGNED r};
		elems := DATASET(term.mat_part, Elem);
		Elem_r rep (Elem l, UNSIGNED c) := TRANSFORM
			SELF.r := c;
			SELF := l;
		END;
		elems_rep := NORMALIZE(elems, cumm_part_cols, rep(LEFT, COUNTER));// each value in bias vector is repeated cumm_part_cols number of times (as the numebr of cumm columns in this node)
		elems_rep_sort := SORT(elems_rep, r);
		term_rep_set := SET (elems_rep_sort, v);
    SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term_rep_set, 1);
		SELF.partition_id := cumm.partition_id;
    SELF := cumm;
  END;
	
	 ab_bb := JOIN(ab_prod, bb_in,TRUE , addbias(LEFT,RIGHT),ALL);
	//rslt := A;
  RETURN ab_bb;
END; // END WX_repmatb
  //retunrs the sigmoid(WX+b)  
WX_repmatb_sig(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_bb, DATASET(Layout_Part) bb_in, PBblas.IMatrix_Map map_c, REAL8 beta=0.0) := FUNCTION
SET OF PBblas.Types.value_t empty_array := [];
// First check maps for compatability.  Normalize for transpose operations.
  a_matrix_rows := map_a.matrix_rows;
  a_matrix_cols := map_a.matrix_cols;
  a_row_blocks  := map_a.row_blocks;
  a_col_blocks  := map_a.col_blocks;
  b_matrix_rows := map_b.matrix_rows;
  b_matrix_cols := map_b.matrix_cols;
  b_row_blocks  := map_b.row_blocks;
  b_col_blocks  := map_b.col_blocks;
  c_matrix_rows := map_c.matrix_rows;
  c_matrix_cols := map_c.matrix_cols;
  c_row_blocks  := map_c.row_blocks;
  c_col_blocks  := map_c.col_blocks;
  A := ASSERT(A_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' + ' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(a_matrix_rows=c_matrix_rows AND a_row_blocks=c_row_blocks,
                    'A-C: ' + 'Arows: ' + a_matrix_rows +' Crows '+c_matrix_rows+' '+ 
                              'Ablocks: ' + a_row_blocks +' Cblocks '+c_row_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	B := ASSERT(B_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' +' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(b_matrix_cols=c_matrix_cols AND b_col_blocks=c_col_blocks,
                    'B-C: ' + 'Brows: ' + b_matrix_cols +' Ccols '+c_matrix_cols+' '+ 
                              'Bblocks: ' + b_row_blocks +' Cblocks '+c_col_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	
  //multiply
	
	Layout_Part mul2(Layout_Part b_part, Layout_Part a_part):=TRANSFORM
    part_id     := b_part.partition_id;    //arbitrary choice
    part_a_cols := a_part.part_cols;
    part_a_rows := a_part.part_rows;
    part_b_rows := b_part.part_rows;
    part_c_rows := map_c.part_rows(part_id);
    part_c_cols := map_c.part_cols(part_id);
    part_c_first_row  := map_c.first_row(part_id);
    part_c_first_col  := map_c.first_col(part_id);
    k := part_a_cols;
    SELF.partition_id := b_part.partition_id;
    SELF.node_id      := b_part.node_id;
    SELF.block_row    := b_part.block_row;
    SELF.block_col    := b_part.block_col;
    SELF.first_row    := map_c.first_row(part_id);
    SELF.part_rows    := part_c_rows;
    SELF.first_col    := part_c_first_col;
    SELF.part_cols    := part_c_cols;
    SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
                                    part_c_rows, part_c_cols, k,
                                    1.0, a_part.mat_part, b_part.mat_part,
                                    0.0, empty_array);
  END;
  ab_prod := JOIN(B, A,TRUE , mul2(LEFT,RIGHT),ALL); // Each A (weight matrix) is copied in each B (X matrix) node

// add bias vector to each columns of X
// each bias vector (b) is copied (ALL JOIN) to each node of X. The bias vector is then normalized to repeat it to the number of columns of X in that node and then the two are added
  Layout_Part addbias(Layout_Part cumm, Layout_Part term) := TRANSFORM
		cumm_part_cols := cumm.part_cols;
    N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
		Elem := {PBblas.Types.value_t v};
		Elem_r := {PBblas.Types.value_t v, UNSIGNED r};
		elems := DATASET(term.mat_part, Elem);
		Elem_r rep (Elem l, UNSIGNED c) := TRANSFORM
			SELF.r := c;
			SELF := l;
		END;
		elems_rep := NORMALIZE(elems, cumm_part_cols, rep(LEFT, COUNTER));// each value in bias vector is repeated cumm_part_cols number of times (as the numebr of cumm columns in this node)
		elems_rep_sort := SORT(elems_rep, r);
		term_rep_set := SET (elems_rep_sort, v);
    SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term_rep_set, 1);
		SELF.partition_id := cumm.partition_id;
    SELF := cumm;
  END;
	
	 ab_bb := JOIN(ab_prod, bb_in,TRUE , addbias(LEFT,RIGHT),ALL);
	//rslt := A;
  RETURN ab_bb;
END; // END WX_repmatb_sig
		
		

		
		
		
		
		//((W2'*d3)+beta*repmat(sparsity_delta,1,m))
		// w'*x + beta * repmat (b, 1,m)
		//A_in :w
		//B_in :x
		//bb_in : bias vector (b)
		
WtX_repmatb(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_bb, DATASET(Layout_Part) bb_in, PBblas.IMatrix_Map map_c, REAL8 beta=0.0) := FUNCTION
	SET OF PBblas.Types.value_t empty_array := [];
	// First check maps for compatability.  Normalize for transpose operations.
		a_matrix_rows := map_a.matrix_cols;
		a_matrix_cols := map_a.matrix_rows;
		a_row_blocks  := map_a.col_blocks;
		a_col_blocks  := map_a.row_blocks;
		b_matrix_rows := map_b.matrix_rows;
		b_matrix_cols := map_b.matrix_cols;
		b_row_blocks  := map_b.row_blocks;
		b_col_blocks  := map_b.col_blocks;
		c_matrix_rows := map_c.matrix_rows;
		c_matrix_cols := map_c.matrix_cols;
		c_row_blocks  := map_c.row_blocks;
		c_col_blocks  := map_c.col_blocks;
		A := ASSERT(A_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'YES' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'NO' + ' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(a_matrix_rows=c_matrix_rows AND a_row_blocks=c_row_blocks,
                    'A-C: ' + 'Arows: ' + a_matrix_rows +' Crows '+c_matrix_rows+' '+ 
                              'Ablocks: ' + a_row_blocks +' Cblocks '+c_row_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
  B := ASSERT(B_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'YES' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'NO' +' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(b_matrix_cols=c_matrix_cols AND b_col_blocks=c_col_blocks,
                    'B-C: ' + 'Brows: ' + b_matrix_cols +' Ccols '+c_matrix_cols+' '+ 
                              'Bblocks: ' + b_row_blocks +' Cblocks '+c_col_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
		
		//multiply
		
		Layout_Part mul2(Layout_Part b_part, Layout_Part a_part):=TRANSFORM
			part_id     := b_part.partition_id;    //arbitrary choice
			part_a_cols := a_part.part_cols;
			part_a_rows := a_part.part_rows;
			part_b_rows := b_part.part_rows;
			part_c_rows := map_c.part_rows(part_id);
			part_c_cols := map_c.part_cols(part_id);
			part_c_first_row  := map_c.first_row(part_id);
			part_c_first_col  := map_c.first_col(part_id);
			k := part_a_rows;
			SELF.partition_id := b_part.partition_id;
			SELF.node_id      := b_part.node_id;
			SELF.block_row    := b_part.block_row;
			SELF.block_col    := b_part.block_col;
			SELF.first_row    := map_c.first_row(part_id);
			SELF.part_rows    := part_c_rows;
			SELF.first_col    := part_c_first_col;
			SELF.part_cols    := part_c_cols;
			SELF.mat_part     := PBblas.BLAS.dgemm(TRUE, FALSE,
																			part_c_rows, part_c_cols, k,
																			1.0, a_part.mat_part, b_part.mat_part,
																			0.0, empty_array);
		END;
		ab_prod := JOIN(B, A,TRUE , mul2(LEFT,RIGHT),ALL); // Each A (weight matrix) is copied in each B (X matrix) node



// Apply beta
  Layout_Part applyBeta(Layout_Part part) := TRANSFORM
    SELF.mat_part := PBblas.BLAS.dscal(map_bb.matrix_rows*map_bb.matrix_cols,
                                beta, part.mat_part, 1);
    SELF:= part;
  END;
  bb_beta := PROJECT(bb_in, applyBeta(LEFT), LOCAL);
	// add the vector to each columns of X
	// each vector is copied (ALL JOIN) to each node of X. The vector is then normalized to repeat it to the number of columns of X in that node and then the two are added
		Layout_Part addvec(Layout_Part cumm, Layout_Part term) := TRANSFORM
			cumm_part_cols := cumm.part_cols;
			N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
			Elem := {PBblas.Types.value_t v};
			Elem_r := {PBblas.Types.value_t v, UNSIGNED r};
			elems := DATASET(term.mat_part, Elem);
			Elem_r rep (Elem l, UNSIGNED c) := TRANSFORM
				SELF.r := c;
				SELF := l;
			END;
			elems_rep := NORMALIZE(elems, cumm_part_cols, rep(LEFT, COUNTER));// each value in bias vector is repeated cumm_part_cols number of times (as the numebr of cumm columns in this node)
			elems_rep_sort := SORT(elems_rep, r);
			term_rep_set := SET (elems_rep_sort, v);
			SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term_rep_set, 1);
			SELF.partition_id := cumm.partition_id;
			SELF := cumm;
		END;
		
		 ab_bb := JOIN(ab_prod, bb_beta,TRUE , addvec(LEFT,RIGHT),ALL);

		//rslt := A;
		RETURN ab_bb;
END; // END WtX_repmatb
		
		
		
		// the input is a matrix in PBblas format where only columns are partitions
		//B_in : ones vector which is partitioned among nodes
		// map_c is the result's map
		col_sum(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_c) := FUNCTION
			SET OF PBblas.Types.value_t empty_array := [];
				Layout_Part mul(Layout_Part a_part, Layout_Part b_part):=TRANSFORM
					part_id     := 1;    //arbitrary choice
					part_a_cols := a_part.part_cols;
					part_a_rows := a_part.part_rows;
					part_b_rows := b_part.part_rows;
					part_c_rows := map_c.part_rows(part_id);
					part_c_cols := map_c.part_cols(part_id);
					part_c_first_row  := map_c.first_row(part_id);
					part_c_first_col  := map_c.first_col(part_id);
					k := part_a_cols;
					SELF.partition_id := 1;
					SELF.node_id      := a_part.node_id;
					SELF.block_row    := 1;
					SELF.block_col    := 1;
					SELF.first_row    := map_c.first_row(part_id);
					SELF.part_rows    := part_c_rows;
					SELF.first_col    := part_c_first_col;
					SELF.part_cols    := part_c_cols;
					SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
																					part_c_rows, part_c_cols, k,
																					1.0, a_part.mat_part, b_part.mat_part,
																					0.0, empty_array);
			END;
			col_sum_part := JOIN (A_in, B_in, LEFT.partition_id = RIGHT.partition_id, mul(LEFT, RIGHT), LOCAL );
			
			Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := le.part_rows ;
				SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1);
				SELF := le;
			END;
			//rslt := ROLLUP(col_sum_part, addup(LEFT, RIGHT), partition_id); // overload becasue of grouping
			rslt := ROLLUP(col_sum_part,TRUE, addup(LEFT, RIGHT)); // no groupijng, reduces overload
			//distribute to node one
			RETURN rslt; 
		END;//END Col_Sum
		
		
		col_mean(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_c) := FUNCTION
		
			Num := map_a.matrix_cols;
			Num_1 := 1/Num;
			SET OF PBblas.Types.value_t empty_array := [];
				Layout_Part mul(Layout_Part a_part, Layout_Part b_part):=TRANSFORM
					part_id     := 1;    //arbitrary choice
					part_a_cols := a_part.part_cols;
					part_a_rows := a_part.part_rows;
					part_b_rows := b_part.part_rows;
					part_c_rows := map_c.part_rows(part_id);
					part_c_cols := map_c.part_cols(part_id);
					part_c_first_row  := map_c.first_row(part_id);
					part_c_first_col  := map_c.first_col(part_id);
					k := part_a_cols;
					SELF.partition_id := 1;
					SELF.node_id      := 0;
					SELF.block_row    := 1;
					SELF.block_col    := 1;
					SELF.first_row    := map_c.first_row(part_id);
					SELF.part_rows    := part_c_rows;
					SELF.first_col    := part_c_first_col;
					SELF.part_cols    := part_c_cols;
					SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
																					part_c_rows, part_c_cols, k,
																					Num_1, a_part.mat_part, b_part.mat_part,
																					0.0, empty_array);
			END;
			col_sum_part := JOIN (A_in, B_in, LEFT.partition_id = RIGHT.partition_id, mul(LEFT, RIGHT), LOCAL );
			
			Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := le.part_rows ;
				SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1);
				SELF := le;
			END;
			rslt := ROLLUP(col_sum_part,TRUE, addup(LEFT, RIGHT)); // this form of rollup avoid grouping, ROLLUP(col_sum_part,addup(LEFT, RIGHT, partiion_id)) cause all the records with the same partiion_id
			//get grouped to one node which cause a lot of overload. The current rollup form avoidds grouping and improves performance
			final_rslt := DISTRIBUTE (rslt, node_id); 
			//distribute to node one
			RETURN final_rslt; 
		END;//END colmean
		
		
		// This function gets two big matrices which are distributed over all nodes and generate a final relatively smaller matrix which is on one node
		// this is used for weight gradient calculation where for example a h*m matrix is multiplied by a m*f matrix. PBblas will distribute all partitions in the first and second matrix to only one node which final matrix is in
		// this causes overhead, to avoid that we multiply each col partition of first matrix with a row partition of the second matrix in each node, the final generated matrices are added up to generate the final matrix
		// this way, we don't change the distribution of the first and second matrices
		big_big_small(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_c, PBblas.Types.value_t alph=1.0) := FUNCTION
			SET OF PBblas.Types.value_t empty_array := [];
				Layout_Part mul(Layout_Part a_part, Layout_Part b_part):=TRANSFORM
					part_id     := 1;    //arbitrary choice
					part_a_cols := a_part.part_cols;
					part_a_rows := a_part.part_rows;
					part_b_rows := b_part.part_rows;
					part_b_cols := b_part.part_cols;
					k := part_a_cols;
					SELF.partition_id := part_id;
					SELF.node_id      := 0;
					SELF.block_row    := 1;
					SELF.block_col    := 1;
					SELF.first_row    := map_c.first_row(part_id);
					SELF.part_rows    := map_c.part_rows(part_id);
					SELF.first_col    := map_c.first_col(part_id);
					SELF.part_cols    := map_c.part_cols(part_id);
					SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, TRUE,
																					part_a_rows, part_b_rows, k,
																					alph, a_part.mat_part, b_part.mat_part,
																					0.0, empty_array);
			END;
			mul_part := JOIN (A_in, B_in, LEFT.partition_id = RIGHT.partition_id, mul(LEFT, RIGHT), LOCAL );
			
			Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := le.part_rows * le.part_cols ;
				SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1);
				SELF := le;
			END;
			
			Layout_Part addup_it(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := ri.part_rows * ri.part_cols ;
				SELF.mat_part := IF (le.partition_id=0, ri.mat_part, PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1));
				SELF := ri;
			END;
			//rslt := ROLLUP(mul_part, addup(LEFT, RIGHT), partition_id); // since the results of rohat is used in a ALL join, no need to distribute this to node one to be consistent with PBblas
			//rslt := ITERATE(mul_part, addup_it(LEFT, RIGHT));// using rollup cause the graph to Group all the records which are distributed between all node to only one record and then do the operation, It takes a long time to GROUP all thoese partitions in one node and we avoid it by using ITERATE instead of ROLLUP

      rslt := ROLLUP(mul_part, TRUE, addup(LEFT, RIGHT));
			final_rslt := DISTRIBUTE (rslt, node_id); 

// a_part := A_in[1];
// b_part := B_in[1];
		 // RETURN PBblas.BLAS.dgemm(FALSE, TRUE,
																					// a_part.part_rows, b_part.part_rows, a_part.part_cols,
																					// 1.0, a_part.mat_part, b_part.mat_part,
																					// 0.0, empty_array);
																					
			RETURN final_rslt;
		END;// END big_big_small
		
		
		
		
		//extract sparse autoencoder parameters
    ddist := TrainData;
    m := CostFunc_params(id=1)[1].value;
    num_feat := CostFunc_params(id=2)[1].value;
    num_hid := CostFunc_params(id=3)[1].value;
    part_rows := CostFunc_params(id=4)[1].value;
    part_cols := CostFunc_params(id=5)[1].value;
    BETA := CostFunc_params(id=6)[1].value;
    sparsityParam := CostFunc_params(id=7)[1].value;
    LAMBDA := CostFunc_params(id=8)[1].value;
    
    m_1 := 1/m;
    sparsityParam_ := -1*sparsityParam;
    sparsityParam_1 := 1-sparsityParam;
    
     sizeRec := RECORD
      PBblas.Types.dimension_t m_rows;
      PBblas.Types.dimension_t m_cols;
      PBblas.Types.dimension_t f_b_rows;
      PBblas.Types.dimension_t f_b_cols;
    END;
    sizeTable := DATASET([{num_feat,m,num_feat,part_cols}], sizeRec);
    
    //Create block matrix d
    dmap := PBblas.Matrix_Map(num_feat,m,num_feat,part_cols);
    
    //Create block matrix Ytmp
    Ymap := dmap;
    Ydist := ddist;
    //Creat block matrices for weights

    w1map := PBblas.Matrix_Map(num_hid, num_feat, num_hid, num_feat);
    w2map := PBblas.Matrix_Map(num_feat, num_hid, num_feat, num_hid);
     //each bias vector is converted to block format
    
    b1vecmap := PBblas.Matrix_Map(num_hid, 1, num_hid, 1);
    b2vecmap := PBblas.Matrix_Map(num_feat, 1, num_feat, 1);
    
    w1_partitions := w1map.partitions_used;
    w2_partitions := w2map.partitions_used;
    b1_partitions := b1vecmap.partitions_used;
    b2_partitions := b2vecmap.partitions_used;
    PBblas.Types.MUElement minuspart(Layout_Part l, UNSIGNED8 c ) := TRANSFORM
      SELF.partition_id := l.partition_id - c;
      SElF.no := 1;
      SELF := l;
    END;
    //w1m := w1dist;
    w1dist := theta (partition_id <= w1_partitions);
    //w2m := w2dist;
    w2m_ := theta (partition_id > w1_partitions AND partition_id <= w1_partitions + w2_partitions);
    w2dist  := PROJECT (w2m_, minuspart(LEFT, w1_partitions ),LOCAL);
    //b1v := b1vecdist;
    b1v_ := theta (partition_id > w1_partitions + w2_partitions AND partition_id <= w1_partitions + w2_partitions + b1_partitions);
    b1vecdist  := PROJECT (b1v_, minuspart (LEFT, w1_partitions + w2_partitions),LOCAL);
    
    //b2v := b2vecdist;
    b2v_ := theta (partition_id > w1_partitions + w2_partitions + b1_partitions AND partition_id <= w1_partitions + w2_partitions + b1_partitions + b2_partitions);
    b2vecdist  := PROJECT (b2v_, minuspart (LEFT, w1_partitions + w2_partitions + b1_partitions),LOCAL);
    
     //functions used
    PBblas.Types.value_t sp_reci(PBblas.Types.value_t v,PBblas.Types.dimension_t r,PBblas.Types.dimension_t c) := sparsityParam_/v;
    PBblas.Types.value_t sp_1_reci(PBblas.Types.value_t v,PBblas.Types.dimension_t r,PBblas.Types.dimension_t c) := sparsityParam_1/(1-v);
    //sparsity_delta=((-sparsityParam./rhohat)+((1-sparsityParam)./(1.-rhohat)));
    PBblas.Types.value_t sp_delta(PBblas.Types.value_t v,PBblas.Types.dimension_t r,PBblas.Types.dimension_t c) := (sparsityParam_/v)+(sparsityParam_1/(1-v));
    PBblas.Types.value_t siggrad(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := v*(1.0-v);
    PBblas.Types.value_t sigmoid(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := 1/(1+exp(-1*v));
    PBblas.Types.value_t pow2(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := v*v;
    //maps used
    b1map := PBblas.Matrix_Map(num_hid, m, num_hid, part_cols);
    b2map := PBblas.Matrix_Map(num_feat, m, num_feat, part_cols);
    a2map := b1map;
    a3map := b2map;
    HL_nodes := num_hid;//number of nodes in the hidden layer
    Hiddmap := b1vecmap;
    //onevec for calculating rhohat
    Ones_VecMap := PBblas.Matrix_Map(m, 1, part_cols, 1);
    //New Vector Generator
    Layout_Cell gen(UNSIGNED4 c, UNSIGNED4 NumRows) := TRANSFORM
      SELF.x := ((c-1) % NumRows) + 1;
      SELF.y := ((c-1) DIV NumRows) + 1;
      SELF.v := 1;
    END;
    //Create Ones Vector for the calculations in the step fucntion
    Ones_Vec := DATASET(m, gen(COUNTER, m));
    Ones_Vecdist := DMAT.Converted.FromCells(Ones_VecMap, Ones_Vec);
    //FF2 returns a2
    FF2_(DATASET(Layout_Part) w1, DATASET(Layout_Part) b1v):= FUNCTION
      //b1m = repmat(b1v,1,m)
      b1m := PBblas.PB_dgemm(FALSE, TRUE, 1.0,b1vecmap, b1v, Ones_VecMap, Ones_Vecdist, b1map);
      //z2 = w1*X+b1;
      //z2 := PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1, dmap, ddist, b1map, b1m, 1.0); // gives MP closed error
      z2_ := PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1, dmap, ddist, b1map);
      z2 := PBblas.PB_daxpy(1.0, z2_, b1m);
      //a2 = sigmoid (z2);
      a2 := PBblas.Apply2Elements(b1map, z2, sigmoid);
      RETURN a2;
    END;//END FF2_
		
		
		
		//returns a2
		 FF2(DATASET(Layout_Part) w1, DATASET(Layout_Part) b1v):= FUNCTION
      //z2=w1*x+repmat(b1,1,m)
			z2 := WX_repmatb(w1map, w1, dmap, ddist, b1vecmap, b1v, b1map, 0.0);
      //a2 = sigmoid (z2);
      a2 := PBblas.Apply2Elements(b1map, z2, sigmoid);
      RETURN a2;
     END;//END FF2
		
		
    //FF3 returns a3
    FF3_(DATASET(Layout_Part) w2,DATASET(Layout_Part) b2v, DATASET(Layout_Part) a2 ):= FUNCTION
      //b2m = repmat(b2v,1,m)
      b2m := PBblas.PB_dgemm(FALSE, TRUE, 1.0,b2vecmap, b2v, Ones_VecMap, Ones_Vecdist, b2map);
      //z3 = w2*a2+b2;
      //z3 := PBblas.PB_dgemm(FALSE, FALSE,1.0,w2map, w2, a2map, a2, b2map,b2m, 1.0); //gives MP closed error
      z3_ := PBblas.PB_dgemm(FALSE, FALSE,1.0,w2map, w2, a2map, a2, b2map);
      z3 := PBblas.PB_daxpy(1.0, z3_, b2m);
      //a3 = sigmoid (z3)
      a3 := PBblas.Apply2Elements(b2map, z3, sigmoid);
      RETURN a3;
    END;//END FF3_
		
		 //FF3 returns a3
    FF3(DATASET(Layout_Part) w2,DATASET(Layout_Part) b2v, DATASET(Layout_Part) a2 ):= FUNCTION
		  //z3 = w2*a2 + repmat(b2,1,m)
			z3 := WX_repmatb(w2map, w2, a2map, a2, b2vecmap, b2v, b2map, 0.0);
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
		
		DELTA3_ (DATASET(Layout_Part) a3 ) := FUNCTION
      //calculate delta for the last layer (3rd layer)
      //y=X;
      //d3=-(y-a3).*(a3.*(1-a3));
      siggrad_a3 := PBblas.Apply2Elements(a3map, a3, siggrad);
      a3_y := PBblas.PB_daxpy(-1, ddist, a3);
      d3 := PBblas.HadamardProduct(a3map, a3_y, siggrad_a3);
      RETURN d3 ;
    END;//END DELTA3_
    //DELTA2 retunrs d2
    rohat_ (DATASET(Layout_Part) a2) := FUNCTION
      rh := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a2map, a2, Ones_VecMap, Ones_Vecdist, Hiddmap);
      RETURN rh;
    END;
		rohat (DATASET(Layout_Part) a2) := FUNCTION
			rh := col_mean(a2map, a2, Ones_VecMap, Ones_Vecdist, Hiddmap);
      RETURN rh;
    END;
    DELTA2_ (DATASET(Layout_Part) w2, DATASET(Layout_Part) a2, DATASET(Layout_Part) d3, DATASET(Layout_Part) rhat) := FUNCTION
      //calculate delta for 2nd layer
      //rhohat=mean(a2,2);
      //sparsity_delta=((-sparsityParam./rhohat)+((1-sparsityParam)./(1.-rhohat)));
      //d2=((W2'*d3)+beta*repmat(sparsity_delta,1,m)) .* (a2.*(1-a2));
      //rhohat := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a2map, a2, Ones_VecMap, Ones_Vecdist, Hiddmap);
      rhohat := rhat;
      sparsity_delta := PBblas.Apply2Elements(Hiddmap, rhohat, sp_delta);
      siggrad_a2 := PBblas.Apply2Elements(a2map, a2, siggrad);
      repmat_sparsity_delta := PBblas.PB_dgemm(FALSE, TRUE, 1.0,  Hiddmap, sparsity_delta, Ones_VecMap, Ones_Vecdist, a2map);
      //d2_firstterm = (W2'*d3)+beta*repmat(sparsity_delta,1,m);
      //d2_firstterm := PBblas.PB_dgemm(TRUE, FALSE, 1.0, w2map, w2, a3map, d3, a2map, repmat_sparsity_delta, BETA); // MP close error
      d2_firstterm_ := PBblas.PB_dgemm(TRUE, FALSE, 1.0, w2map, w2, a3map, d3, a2map);
      d2_firstterm := PBblas.PB_daxpy(BETA, repmat_sparsity_delta, d2_firstterm_);
      d2 := PBblas.HadamardProduct(a2map, d2_firstterm, siggrad_a2);
      RETURN d2 ;
    END;
		
		
		DELTA2 (DATASET(Layout_Part) w2, DATASET(Layout_Part) a2, DATASET(Layout_Part) d3, DATASET(Layout_Part) rhat) := FUNCTION
      //calculate delta for 2nd layer
      //rhohat=mean(a2,2);
      //sparsity_delta=((-sparsityParam./rhohat)+((1-sparsityParam)./(1.-rhohat)));
      //d2=((W2'*d3)+beta*repmat(sparsity_delta,1,m)) .* (a2.*(1-a2));
      //rhohat := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a2map, a2, Ones_VecMap, Ones_Vecdist, Hiddmap);
      rhohat := rhat;
      sparsity_delta := PBblas.Apply2Elements(Hiddmap, rhohat, sp_delta);
      siggrad_a2 := PBblas.Apply2Elements(a2map, a2, siggrad);
      //repmat_sparsity_delta := PBblas.PB_dgemm(FALSE, TRUE, 1.0,  Hiddmap, sparsity_delta, Ones_VecMap, Ones_Vecdist, a2map);
      //d2_firstterm = (W2'*d3)+beta*repmat(sparsity_delta,1,m);
			d2_firstterm := WtX_repmatb(w2map, w2, a3map, d3, Hiddmap, sparsity_delta, a2map, BETA);
      d2 := PBblas.HadamardProduct(a2map, d2_firstterm, siggrad_a2);
      RETURN d2 ;
    END;
    //WeightGrad1 returns gradient for w1
    WeightGrad1_ (DATASET(Layout_Part) w1, DATASET(Layout_Part) d2) := FUNCTION
      w1_g := PBblas.PB_dgemm(FALSE, TRUE, m_1, a2map, d2, dmap, ddist, w1map, w1 ,LAMBDA );
      RETURN w1_g;
    END;
		WeightGrad1 (DATASET(Layout_Part) w1, DATASET(Layout_Part) d2) := FUNCTION
			w1_g_ := big_big_small(a2map, d2, dmap, ddist, w1map, m_1);
			w1_g  := PBblas.PB_daxpy(LAMBDA, w1, w1_g_);
      RETURN w1_g;
    END;
		
    //WeightGrad2 returns gradient for w2
    WeightGrad2_ (DATASET(Layout_Part) w2, DATASET(Layout_Part) d3, DATASET(Layout_Part) a2) := FUNCTION
      w2_g := PBblas.PB_dgemm(FALSE, TRUE, m_1, a3map, d3, a2map, a2, w2map, w2 ,LAMBDA );
      RETURN w2_g;
    END;
		
		WeightGrad2 (DATASET(Layout_Part) w2, DATASET(Layout_Part) d3, DATASET(Layout_Part) a2) := FUNCTION
			w2_g_ := big_big_small(a3map, d3, a2map, a2, w2map, m_1);
			w2_g  := PBblas.PB_daxpy(LAMBDA, w2, w2_g_);
      RETURN w2_g;
    END;
    //BiasGrad1 calculates the bias gradients for b1
    BiasGrad1 (DATASET(Layout_Part) d2) := FUNCTION
			b1_g := col_mean(a2map, d2, Ones_VecMap, Ones_Vecdist, b1vecmap);
      RETURN b1_g;
    END;
		
		BiasGrad1_ (DATASET(Layout_Part) d2) := FUNCTION
      b1_g := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a2map, d2, Ones_VecMap, Ones_Vecdist, b1vecmap);
      RETURN b1_g;
    END;
    //BiasGrad2 calculates the bias gradients for b2
    BiasGrad2_ (DATASET(Layout_Part) d3) := FUNCTION
      b2_g := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a3map, d3, Ones_VecMap, Ones_Vecdist, b2vecmap);
      RETURN b2_g;
    END;
		
		BiasGrad2 (DATASET(Layout_Part) d3) := FUNCTION
      b2_g := col_mean(a3map, d3, Ones_VecMap, Ones_Vecdist, b2vecmap);
      RETURN b2_g;
    END;
		
		
    emptyL := DATASET([], Layout_Part);
    //theta is the input weight and bias parameters for the sparseautoencoder
    //parameters are the parameters for the sparseautoencoder function
    //train_d is the train data to learn sparse autoencoder and calculate the gradient and the cost based on taht
    // train_l is empty in the sparse autoencoder (because it is an unsupervised learning)
    SparseParam_CostGradients4 :=  FUNCTION
      PBblas.Types.MUElement minuspart(Layout_Part l, UNSIGNED8 c ) := TRANSFORM
        SELF.partition_id := l.partition_id - c;
        SElF.no := 1;
        SELF := l;
      END;
      w1m := w1dist;
      w2m := w2dist;
      b1v := b1vecdist;
      b2v := b2vecdist;
      a2 := FF2 (w1m, b1v);
			a2_ := FF2_ (w1m, b1v);
      a3 := FF3 (w2m, b2v, a2);
			a3_ := FF3_ (w2m, b2v, a2_);
      d3 := DELTA3 (a3);
			d3_ := DELTA3_ (a3_);
      rohat_a2 := rohat(a2);
			rohat_a2_ := rohat_(a2_);
      d2 := DELTA2 (w2m, a2, d3,rohat_a2);
			d2_ := DELTA2_ (w2m, a2_, d3_,rohat_a2_);
      wg1 := WeightGrad1 (w1m, d2);
      wg2 := WeightGrad2 (w2m, d3, a2);
      bg1 := BiasGrad1 (d2);
      bg2 := BiasGrad2 (d3);
			
			
			wg1_ := WeightGrad1_ (w1m, d2_);
      wg2_ := WeightGrad2_ (w2m, d3_, a2_);
      bg1_ := BiasGrad1_ (d2_);
      bg2_ := BiasGrad2_ (d3_);
      //calculate cost
      //PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1, dmap, ddist, b1map, b1m, 1.0);
      // squared_error_cost= 0.5*sum(sum((x-a3).^2));
      // cost=(1/m)*squared_error_cost+(lambda/2)*(sum(W2(:).^2)+sum(W1(:).^2))+beta*sum(KL(sparsityParam,rhohat));
      squared_error_cost := 0.5*PBblas.SumElements(PBblas.Apply2Elements(dmap, PBblas.PB_daxpy(-1.0, a3, ddist), pow2));
      cost_term1 := (1/m)*squared_error_cost;
      cost_term2 := (lambda/2)* PBblas.SumElements(PBblas.Apply2Elements(w2map, w2m, pow2));
      cost_term3 := (lambda/2)* PBblas.SumElements(PBblas.Apply2Elements(w1map, w1m, pow2));
      PBblas.Types.value_t klterm(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := sparsityParam * LN(sparsityParam/v) + (1-sparsityParam) * LN ((1-sparsityParam)/(1-v));
      KL := PBblas.Apply2Elements (Hiddmap,rohat_a2,klterm);
      cost_term4 := beta * PBblas.SumElements(KL);
      cost := cost_term1 + cost_term2 + cost_term3 +cost_term4;    
      costField := DATASET ([{1,1,cost}],ML.Types.NumericField);
      one_map := PBblas.Matrix_Map(1,1,1,1);
      Cost_part_no := PBblas.MU.TO(ML.DMat.Converted.FromNumericFieldDS(costField,one_map),2);
      //convert w and b gradients to a datasets of layoutparts where partition_id differentiate them
      //w1_grad has partition_id from 1 to w1_partitions
      //w2_grad had partition_ds from w1_partitions+1 to w1_partitions + w2_partitions
      //b1_grad has partition_ids from w1_partitions + w2_partitions+1 to w1_partitions + w2_partitions+b1_partitions
      //b2_grad has partition_ids from w1_partitions + w2_partitions+b1_partitions+1 to w1_partitions + w2_partitions+b1_partitions+b2_partitions
      PBblas.Types.MUElement addpart(Layout_Part l, UNSIGNED8 c ) := TRANSFORM
        SELF.partition_id := l.partition_id + c;
        SElF.no := 1;
        SELF := l;
      END;
      wg1_reshape_no := Pbblas.MU.TO(wg1,1);
      wg2_reshape_no := PROJECT (wg2, addpart(LEFT, w1_partitions ), LOCAL);
			wg2_reshape_no_ := PROJECT (wg2_, addpart(LEFT, w1_partitions ), LOCAL);
      bg1_reshape_no := PROJECT (bg1, addpart (LEFT, w1_partitions + w2_partitions),LOCAL);
      bg2_reshape_no := PROJECT (bg2, addpart (LEFT, w1_partitions + w2_partitions + b1_partitions),LOCAL);
      theta_Part_no := wg1_reshape_no + wg2_reshape_no + bg1_reshape_no + bg2_reshape_no;
      
			//RETURN PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1dist, dmap, ddist, b1map);
			//RETURN WX(w1map, w1dist, dmap, ddist, b1map, w1dist,  1);
			//RETURN WX_repmatb(w1map, w1dist, dmap, ddist, b1vecmap, b1vecdist, b1map, 1.0) ;
			//RETURN col_sum(dmap, ddist, Ones_VecMap, Ones_Vecdist, b2vecmap);
			w1x_b1 := WX_repmatb(w1map, w1dist, dmap, ddist, b1vecmap, b1vecdist, b1map, 1.0);
			thisis := big_big_small(b1map, w1x_b1, dmap, ddist, PBblas.Matrix_Map(num_hid, num_feat, num_hid, num_feat));
			thisone := WtX_repmatb(w2map,w2dist, dmap, ddist, b1vecmap, b1vecdist, b1map, 5);
			
			
			
			//mymy2 := DELTA2 (w2m, a2, d3,rohat_a2) + DELTA2_ (w2m, a2, d3,rohat_a2);
			//mymy2 := big_big_small(a2map, d2, dmap, ddist, w1map, 8);
			//mymy2 := col_mean(a3map, d3, Ones_VecMap, Ones_Vecdist, b2vecmap);
			mymy2 := wg1 + wg2  + bg1 + bg2;
myformat2 := RECORD
    mymy2.node_id;
    mymy2.partition_id;
    mymy2.block_row;
    mymy2.block_col;
    mymy2.first_row;
    mymy2.part_rows;
    mymy2.first_col;
    mymy2.part_cols;
		//mymy2.no;
		//mymy2.mat_part;
		INTEGER real_node := STD.System.Thorlib.Node();
END;
	thisR := TABLE(mymy2,myformat2,LOCAL); 
	RETURN  theta_Part_no + Cost_part_no;
	//RETURN  wg2_reshape_no_;

			//RETURN thisR;
      //RETURN theta_Part_no;
    END;//END SparseParam_CostGradients4  
    RETURN SparseParam_CostGradients4;
   END;//END SA_lbfgs_Compatible2
	 
	 
// in this implementation the matrix partitions are done based on partitioning m (number of samples) to partitions of size prow and f (the number of features) is partitioned to partitions of size pcol
// Based on this partitining, the main data (train data) is partitioned to partitions of size prow by pcol, however the distribution of it is based on block_col field which makes all the row partitions in one column block to end up in one node
// Based on what explained above:
//TrainData is partitioned to partitions of size prow by pcol, it is distributed based col_block (big column partitions, where each partion include smaller row partitions). The distribution is done based on node_id where node_id is calculated using a assigned_node(block_col)
//theta includes the two weight matrix as well as two bias vectors. W1 is partitioned to prow by numberofhiddennodes, W2 is partitioned by numberofhiddennodes by prow, bias1 is only one partition, bias two is partitioned to partitions of size prow
//in theta each matrix/vector can be distingished using partition numbers, also the distribution is done using the partition number (assigned_node(partition_id))
//TrainLabel is actually the onevecdist matrix which is used in the calculations, it is a vector of size m by1 and again partitioned based on partitions of size prow by1 and distributed using assigned_node(partition_id)




EXPORT SA_lbfgs_Compatible2_param_part_test ( DATASET(Layout_Part) theta, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Part) TrainData , DATASET(Layout_Part) TrainLabel) := FUNCTION
   

// This function repeats the bias vector of size r*1 , s times in a columnwise format, so the final results will be a r*s matrix
//D = [1,2,3], r=3, s=2 => output=[1,2,3,1,2,3]
  SET OF REAL8 repeatbias(PBblas.Types.dimension_t r, PBblas.Types.dimension_t s, PBblas.Types.matrix_t D) := BEGINC++

    #body
    __lenResult = r * s * sizeof(double);
    __isAllResult = false;
    double * result = new double[r*s];
    __result = (void*) result;
    double *cell = (double*) d;
    uint32_t cells =  r * s;
    uint32_t i;
    uint32_t pos;
    for (i=0; i<cells; i++) {
      pos = i % r;
      result[i] = cell[pos];
    }
  ENDC++;
	
//this function calculates d3=-(y-a3).*(a3.*(1-a3));
//N is the total number of elements in each matrix
//A is the a3 matrix in SET format
//Y is the y matrix in SET format
	SET OF REAL8 d3_cal(PBblas.Types.dimension_t N, PBblas.Types.matrix_t A, PBblas.Types.matrix_t Y) := BEGINC++

    #body
    __lenResult = n * sizeof(double);
    __isAllResult = false;
    double * result = new double[n];
    __result = (void*) result;
    double *cella = (double*) a;
		double *celly = (double*) y;
    uint32_t cells =  n;
    uint32_t i;
    for (i=0; i<cells; i++) {
      result[i] = (cella[i]-celly[i])*(cella[i]*(1-cella[i]));
    }
  ENDC++;
	SET OF REAL8 d2_cal(PBblas.Types.dimension_t N, PBblas.Types.matrix_t A, PBblas.Types.matrix_t Y) := BEGINC++

    #body
    __lenResult = n * sizeof(double);
    __isAllResult = false;
    double * result = new double[n];
    __result = (void*) result;
    double *cella = (double*) a;
		double *celly = (double*) y;
    uint32_t cells =  n;
    uint32_t i;
    for (i=0; i<cells; i++) {
      result[i] = (celly[i])*(cella[i]*(1-cella[i]));
    }
  ENDC++;
//result = M + bet * repmat (V, 1, r)
	SET OF REAL8 mat_vec_sum(PBblas.Types.dimension_t N, PBblas.Types.dimension_t r, PBblas.Types.matrix_t M, PBblas.Types.matrix_t V, PBblas.Types.value_t bet) := BEGINC++

    #body
    __lenResult = n * sizeof(double);
    __isAllResult = false;
    double * result = new double[n];
    __result = (void*) result;
    double *cellm = (double*) m;
		double *cellv = (double*) v;
    uint32_t cells =  n;
    uint32_t i;
		uint32_t pos;
    for (i=0; i<cells; i++) {
		  pos = i % r;
      result[i] = cellm[i] + (bet * cellv[pos]);
    }
  ENDC++;
	// //result = sigmoid (M + repmat (V, 1, c))
	SET OF REAL8 mat_vec_sum_sigmoid(PBblas.Types.dimension_t N, PBblas.Types.dimension_t r, PBblas.Types.matrix_t M, PBblas.Types.matrix_t V) := BEGINC++

    #body
    __lenResult = n * sizeof(double);
    __isAllResult = false;
    double * result = new double[n];
    __result = (void*) result;
    double *cellm = (double*) m;
		double *cellv = (double*) v;
    uint32_t cells =  n;
    uint32_t i;
		uint32_t pos;
    for (i=0; i<cells; i++) {
		  pos = i % r;
      result[i] = 1/(1 + exp(-1*(cellm[i] + cellv[pos])));
    }
  ENDC++;
	
	
	SET OF REAL8 sum_col (PBblas.Types.dimension_t r, PBblas.Types.dimension_t s, PBblas.Types.matrix_t D) := BEGINC++

    #body
    __lenResult = r * sizeof(double);
    __isAllResult = false;
    double * result = new double[r];
    __result = (void*) result;
    double *cell = (double*) d;
    uint32_t cells =  r * s;
    uint32_t i;
    uint32_t pos;
		for (i=0; i<r; i++) {
      result[i] = 0;
    }
    for (i=0; i<cells; i++) {
      pos = i % r;
      result[pos] = result[pos] + cell[i];
    }

  ENDC++;
	
		SET OF REAL8 sum_col_alpha (PBblas.Types.dimension_t r, PBblas.Types.dimension_t s, PBblas.Types.matrix_t D, REAL8 thisalpha) := BEGINC++

    #body
    __lenResult = r * sizeof(double);
    __isAllResult = false;
    double * result = new double[r];
    __result = (void*) result;
    double *cell = (double*) d;
    uint32_t cells =  r * s;
    uint32_t i;
    uint32_t pos;
		for (i=0; i<r; i++) {
      result[i] = 0;
    }
    for (i=0; i<cells; i++) {
      pos = i % r;
      result[pos] = result[pos] + cell[i];
    }
		for (i=0; i<r; i++) {
      result[i] = result[i] * thisalpha;
    }

  ENDC++;
	//0.5 * sum ((M-V).^2)
	REAL8 sum_pow2(PBblas.Types.dimension_t N, PBblas.Types.matrix_t M, PBblas.Types.matrix_t V) := BEGINC++

    #body
    double result = 0;
		double tmpp ;
    double *cellm = (double*) m;
		double *cellv = (double*) v;
    uint32_t i;
		for (i=0; i<n; i++) {
		  tmpp =(cellm[i] - cellv [i]);
      result = result + (tmpp*tmpp);
    }
		return(0.5*result);

  ENDC++;
	//sum(M.^2)
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
	// sum (kl(rho, M))
	REAL8 sum_kl(PBblas.Types.dimension_t N, PBblas.Types.matrix_t M, PBblas.Types.value_t rho) := BEGINC++

    #body
    double result = 0;
		double tmpp ;
    double *cellm = (double*) m;
    uint32_t i;
		for (i=0; i<n; i++) {
			result = result + (rho*log(rho/cellm[i])) + ((1-rho)*log((1-rho)/(1-cellm[i])));
    }
		return(result);

  ENDC++;
    ddist := TrainData;
    m := CostFunc_params(id=1)[1].value;
    num_feat := CostFunc_params(id=2)[1].value;
    num_hid := CostFunc_params(id=3)[1].value;
    part_rows := CostFunc_params(id=4)[1].value; // partition size for the features (number of rows)
    part_cols := CostFunc_params(id=5)[1].value; // partition size for the number of columns (samples) in the input data
    BETA := CostFunc_params(id=6)[1].value;
    sparsityParam := CostFunc_params(id=7)[1].value;
    LAMBDA := CostFunc_params(id=8)[1].value;
    
    m_1 := 1/m;
    sparsityParam_ := -1*sparsityParam;
    sparsityParam_1 := 1-sparsityParam;
    
    //Create map for block matrix d
    dmap := PBblas.Matrix_Map(num_feat,m,part_rows,part_cols);
    
    //Create block matrix Ytmp
    Ymap := dmap;
    Ydist := ddist;
    //Creat maps for block matrices for weights

    w1map := PBblas.Matrix_Map(num_hid, num_feat, num_hid, part_rows);
    w2map := PBblas.Matrix_Map(num_feat, num_hid, part_rows, num_hid);
     //each bias vector is converted to block format
    
    b1vecmap := PBblas.Matrix_Map(num_hid, 1, num_hid, 1);
    b2vecmap := PBblas.Matrix_Map(num_feat, 1, part_rows, 1);
    
    w1_partitions := w1map.partitions_used;
    w2_partitions := w2map.partitions_used;
    b1_partitions := b1vecmap.partitions_used;
    b2_partitions := b2vecmap.partitions_used;
    Layout_Part minuspart(Layout_Part l, UNSIGNED8 c ) := TRANSFORM
      SELF.partition_id := l.partition_id - c;
      SELF := l;
    END;
    //w1m := w1dist;
    w1dist := theta (partition_id <= w1_partitions);
    //w2m := w2dist;
    w2dist := theta (partition_id > w1_partitions AND partition_id <= w1_partitions + w2_partitions);
    //w2dist  := PROJECT (w2m_, minuspart(LEFT, w1_partitions ),LOCAL);
    //b1v := b1vecdist;
    b1vecdist := theta (partition_id > w1_partitions + w2_partitions AND partition_id <= w1_partitions + w2_partitions + b1_partitions);
    //b1vecdist  := PROJECT (b1v_, minuspart (LEFT, w1_partitions + w2_partitions),LOCAL);
    
    //b2v := b2vecdist;
    b2vecdist := theta (partition_id > w1_partitions + w2_partitions + b1_partitions AND partition_id <= w1_partitions + w2_partitions + b1_partitions + b2_partitions);
    //b2vecdist  := PROJECT (b2v_, minuspart (LEFT, w1_partitions + w2_partitions + b1_partitions),LOCAL);
    
     //functions used
    PBblas.Types.value_t sp_reci(PBblas.Types.value_t v,PBblas.Types.dimension_t r,PBblas.Types.dimension_t c) := sparsityParam_/v;
    PBblas.Types.value_t sp_1_reci(PBblas.Types.value_t v,PBblas.Types.dimension_t r,PBblas.Types.dimension_t c) := sparsityParam_1/(1-v);
    //sparsity_delta=((-sparsityParam./rhohat)+((1-sparsityParam)./(1.-rhohat)));
    PBblas.Types.value_t sp_delta(PBblas.Types.value_t v,PBblas.Types.dimension_t r,PBblas.Types.dimension_t c) := (sparsityParam_/v)+(sparsityParam_1/(1-v));
    PBblas.Types.value_t siggrad(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := v*(1.0-v);
    PBblas.Types.value_t sigmoid(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := 1/(1+exp(-1*v));
    PBblas.Types.value_t pow2(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := v*v;
    //maps used
    b1map := PBblas.Matrix_Map(num_hid, m, num_hid, part_cols);
    b2map := PBblas.Matrix_Map(num_feat, m, part_rows, part_cols);
    a2map := b1map;
    a3map := b2map;
    HL_nodes := num_hid;//number of nodes in the hidden layer
    Hiddmap := b1vecmap;
    //onevec for calculating rhohat	 
	 
	 Ones_VecMap := PBblas.Matrix_Map(m, 1, part_cols, 1);
    //New Vector Generator
    // Layout_Cell gen(UNSIGNED4 c, UNSIGNED4 NumRows) := TRANSFORM
      // SELF.x := ((c-1) % NumRows) + 1;
      // SELF.y := ((c-1) DIV NumRows) + 1;
      // SELF.v := 1;
    // END;
    //Create Ones Vector for the calculations in the step fucntion
    // Ones_Vec := DATASET(m, gen(COUNTER, m));
    Ones_Vecdist := TrainLabel;
Layout_Target := PBblas.Types.Layout_Target;
WX(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_bb, DATASET(Layout_Part) bb_in, PBblas.IMatrix_Map map_c, REAL8 beta=0.0) := FUNCTION
SET OF PBblas.Types.value_t empty_array := [];
// First check maps for compatability.  Normalize for transpose operations.
  a_matrix_rows := map_a.matrix_rows;
  a_matrix_cols := map_a.matrix_cols;
  a_row_blocks  := map_a.row_blocks;
  a_col_blocks  := map_a.col_blocks;
  b_matrix_rows := map_b.matrix_rows;
  b_matrix_cols := map_b.matrix_cols;
  b_row_blocks  := map_b.row_blocks;
  b_col_blocks  := map_b.col_blocks;
  c_matrix_rows := map_c.matrix_rows;
  c_matrix_cols := map_c.matrix_cols;
  c_row_blocks  := map_c.row_blocks;
  c_col_blocks  := map_c.col_blocks;
  A := ASSERT(A_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' + ' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(a_matrix_rows=c_matrix_rows AND a_row_blocks=c_row_blocks,
                    'A-C: ' + 'Arows: ' + a_matrix_rows +' Crows '+c_matrix_rows+' '+ 
                              'Ablocks: ' + a_row_blocks +' Cblocks '+c_row_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	B := ASSERT(B_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' +' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(b_matrix_cols=c_matrix_cols AND b_col_blocks=c_col_blocks,
                    'B-C: ' + 'Brows: ' + b_matrix_cols +' Ccols '+c_matrix_cols+' '+ 
                              'Bblocks: ' + b_row_blocks +' Cblocks '+c_col_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));

//
  Layout_Target cvt(Layout_Part par, INTEGER c, BOOLEAN keepRow) := TRANSFORM
    s_block_row       := par.block_row;
    s_block_col       := par.block_col;
    part_id_new_row   := map_c.assigned_part(c, s_block_col);
    part_id_new_col   := map_c.assigned_part(s_block_row, c);
    partition_id      := IF(keepRow, part_id_new_col, part_id_new_row);
    SELF.t_node_id    := map_c.assigned_node(partition_id);
    SELF.t_part_id    := partition_id;
    SELF.t_block_row  := IF(keepRow, s_block_row, c);
    SELF.t_block_col  := IF(keepRow, c, s_block_col);
    SELF.t_term       := IF(keepRow, s_block_col, s_block_row);
    SELF              := par;
  END;

  // A: copy of weight matrix goes to each column of X
  a_fact := map_b.col_blocks; // The number of time weight matrix (A) has to be distributed is the number of columns on matrix X (B)
  a_work := NORMALIZE(A, a_fact, cvt(LEFT, COUNTER, TRUE));
  a_dist := DISTRIBUTE(a_work, t_node_id);
  a_sort := a_dist;// only one partition in each node, so no need to sort
  // B: copy of each cell in a column goes to a row
  b_fact := map_a.row_blocks;
  b_work := PROJECT(B, cvt(LEFT, COUNTER, FALSE), LOCAL);
  b_dist := b_work; // no need to distribute as it is already distributed
  b_sort := b_dist; // only one partition in each node, so no need to sort
	
	
	
	// Elem := {PBblas.Types.value_t v};
	// Elem_col := {PBblas.Types.value_t v, UNSIGNED8 v_col:=1};
	// Layout_Target rep_bb (Layout_Target x) := TRANSFORM
		// elemsX_ := DATASET(x.mat_part, Elem);
		// elemsX := PROJECT (elemsX_, TRANSFORM(Elem_col, SELF := LEFT));
		// Elem_col cvt2(Elem_col par, INTEGER c) := TRANSFORM
			// SELF := par;
		// END;
		// repeatedelemsX := NORMALIZE(elemsX, bb_fact, cvt2(LEFT));
		// self.mat_part := SET(repeatedelemsX, v);
		// SELF := x;
	
	// END;
	
	
  // Multiply
  Layout_Part mul(Layout_Target a_part, Layout_Target b_part):=TRANSFORM
    part_id     := a_part.t_part_id;    //arbitrary choice
    part_a_cols := a_part.part_cols;
    part_a_rows := a_part.part_rows;
    part_b_rows := b_part.part_rows;
    part_c_rows := map_c.part_rows(part_id);
    part_c_cols := map_c.part_cols(part_id);
    part_c_first_row  := map_c.first_row(part_id);
    part_c_first_col  := map_c.first_col(part_id);
    k := part_a_cols;
    SELF.partition_id := part_id;
    SELF.node_id      := a_part.t_node_id;
    SELF.block_row    := a_part.t_block_row;
    SELF.block_col    := a_part.t_block_col;
    SELF.first_row    := map_c.first_row(part_id);
    SELF.part_rows    := part_c_rows;
    SELF.first_col    := part_c_first_col;
    SELF.part_cols    := part_c_cols;
    SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
                                    part_c_rows, part_c_cols, k,
                                    1.0, a_part.mat_part, b_part.mat_part,
                                    0.0, empty_array);
  END;
  ab_prod := JOIN(a_sort, b_sort,
                  LEFT.t_part_id=RIGHT.t_part_id AND LEFT.t_term=RIGHT.t_term,
                  mul(LEFT,RIGHT), LOCAL);




   // Apply beta


	
	
	// N := 32*32*3*1000;
	// Layout_Target sumTerms(Layout_Target cumm, Layout_Target term) := TRANSFORM
    // SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term.mat_part, 1);
    // SELF := cumm;
  // END;
	// sumres := ROLLUP(a_sort, sumTerms(LEFT, RIGHT), partition_id);
	
	
	mymy := a_sort;
	myformat := RECORD
    mymy.node_id;
    mymy.partition_id;
    mymy.block_row;
    mymy.block_col;
    mymy.first_row;
    mymy.part_rows;
    mymy.first_col;
    mymy.part_cols;
		mymy.t_part_id;
    mymy.t_node_id;
    mymy.t_block_row;
    mymy.t_block_col;
    mymy.t_term;
		INTEGER real_node := STD.System.Thorlib.Node();
END;
mymy2 := ab_prod;
myformat2 := RECORD
    mymy2.node_id;
    mymy2.partition_id;
    mymy2.block_row;
    mymy2.block_col;
    mymy2.first_row;
    mymy2.part_rows;
    mymy2.first_col;
    mymy2.part_cols;
		INTEGER real_node := STD.System.Thorlib.Node();
END;
	rslt := TABLE(mymy2,myformat2,LOCAL); 
  RETURN rslt;
END; // END WX




WX_2(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_bb, DATASET(Layout_Part) bb_in, PBblas.IMatrix_Map map_c, REAL8 beta=0.0) := FUNCTION
SET OF PBblas.Types.value_t empty_array := [];
// First check maps for compatability.  Normalize for transpose operations.
  a_matrix_rows := map_a.matrix_rows;
  a_matrix_cols := map_a.matrix_cols;
  a_row_blocks  := map_a.row_blocks;
  a_col_blocks  := map_a.col_blocks;
  b_matrix_rows := map_b.matrix_rows;
  b_matrix_cols := map_b.matrix_cols;
  b_row_blocks  := map_b.row_blocks;
  b_col_blocks  := map_b.col_blocks;
  c_matrix_rows := map_c.matrix_rows;
  c_matrix_cols := map_c.matrix_cols;
  c_row_blocks  := map_c.row_blocks;
  c_col_blocks  := map_c.col_blocks;
  A := ASSERT(A_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' + ' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(a_matrix_rows=c_matrix_rows AND a_row_blocks=c_row_blocks,
                    'A-C: ' + 'Arows: ' + a_matrix_rows +' Crows '+c_matrix_rows+' '+ 
                              'Ablocks: ' + a_row_blocks +' Cblocks '+c_row_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	B := ASSERT(B_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' +' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(b_matrix_cols=c_matrix_cols AND b_col_blocks=c_col_blocks,
                    'B-C: ' + 'Brows: ' + b_matrix_cols +' Ccols '+c_matrix_cols+' '+ 
                              'Bblocks: ' + b_row_blocks +' Cblocks '+c_col_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));

//
  
	
	
	
	// Elem := {PBblas.Types.value_t v};
	// Elem_col := {PBblas.Types.value_t v, UNSIGNED8 v_col:=1};
	// Layout_Target rep_bb (Layout_Target x) := TRANSFORM
		// elemsX_ := DATASET(x.mat_part, Elem);
		// elemsX := PROJECT (elemsX_, TRANSFORM(Elem_col, SELF := LEFT));
		// Elem_col cvt2(Elem_col par, INTEGER c) := TRANSFORM
			// SELF := par;
		// END;
		// repeatedelemsX := NORMALIZE(elemsX, bb_fact, cvt2(LEFT));
		// self.mat_part := SET(repeatedelemsX, v);
		// SELF := x;
	
	// END;
	
	
  //multiply
	
	Layout_Part mul2(Layout_Part b_part, Layout_Part a_part):=TRANSFORM
    part_id     := b_part.partition_id;    //arbitrary choice
    part_a_cols := a_part.part_cols;
    part_a_rows := a_part.part_rows;
    part_b_rows := b_part.part_rows;
    part_c_rows := map_c.part_rows(part_id);
    part_c_cols := map_c.part_cols(part_id);
    part_c_first_row  := map_c.first_row(part_id);
    part_c_first_col  := map_c.first_col(part_id);
    k := part_a_cols;
    SELF.partition_id := b_part.partition_id;
    SELF.node_id      := b_part.node_id;
    SELF.block_row    := b_part.block_row;
    SELF.block_col    := b_part.block_col;
    SELF.first_row    := map_c.first_row(part_id);
    SELF.part_rows    := part_c_rows;
    SELF.first_col    := part_c_first_col;
    SELF.part_cols    := part_c_cols;
    SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
                                    part_c_rows, part_c_cols, k,
                                    1.0, a_part.mat_part, b_part.mat_part,
                                    0.0, empty_array);
  END;
  ab_prod := JOIN(B, A,TRUE , mul2(LEFT,RIGHT),ALL);




   // Apply beta

// Sum terms
  Layout_Part sumTerms(Layout_Part cumm, Layout_Part term) := TRANSFORM
		cumm_part_cols := cumm.part_cols;
    N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
		Elem := {PBblas.Types.value_t v};
		Elem_r := {PBblas.Types.value_t v, UNSIGNED r};
		elems := DATASET(term.mat_part, Elem);
		Elem_r rep (Elem l, UNSIGNED c) := TRANSFORM
			SELF.r := c;
			SELF := l;
		END;
		elems_rep := NORMALIZE(elems, cumm_part_cols, rep(LEFT, COUNTER));
		elems_rep_sort := SORT(elems_rep, r);
		term_rep_set := SET (elems_rep_sort, v);
    SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term_rep_set, 1);
		SELF.partition_id := cumm.partition_id;
    SELF := cumm;
  END;
	
	 ab_bb := JOIN(ab_prod, bb_in,TRUE , sumTerms(LEFT,RIGHT),ALL);

cumm := B[1];
cumm_part_cols := cumm.part_cols;
term := bb_in[1];
Elem := {PBblas.Types.value_t v};
		Elem_r := {PBblas.Types.value_t v, UNSIGNED r:=1};
		elems := DATASET(term.mat_part, Elem);
		Elem_r rep (Elem l, UNSIGNED c) := TRANSFORM
			SELF.r := c;
			SELF := l;
		END;
		elems_rep := NORMALIZE(elems, cumm_part_cols, rep(LEFT, COUNTER));
		elems_rep_sort := SORT(elems_rep, r);
		term_rep_set := SET (elems_rep_sort, v);
	
	// N := 32*32*3*1000;
	// Layout_Target sumTerms(Layout_Target cumm, Layout_Target term) := TRANSFORM
    // SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term.mat_part, 1);
    // SELF := cumm;
  // END;
	// sumres := ROLLUP(a_sort, sumTerms(LEFT, RIGHT), partition_id);
	
	
	
mymy2 := ab_bb;
myformat2 := RECORD
    mymy2.node_id;
    mymy2.partition_id;
    mymy2.block_row;
    mymy2.block_col;
    mymy2.first_row;
    mymy2.part_rows;
    mymy2.first_col;
    mymy2.part_cols;
		//mymy2.mat_part;
		INTEGER real_node := STD.System.Thorlib.Node();
END;
	rslt := TABLE(mymy2,myformat2,LOCAL);
	//rslt := A;
  RETURN rslt;
END; // END WX_2



row_block_bigmat (UNSIGNED8 p, UNSIGNED8 nn) := FUNCTION // p is the partition number, nn is the number of rows
	RETURN ((p-1) % nn )+1;
END;

col_block_bigmat (UNSIGNED8 p, UNSIGNED8 nn) := FUNCTION // p is the partition number, nn is the number of rows
	RETURN ((p-1) DIV nn )+1;
END;

block_smallmat (UNSIGNED8 p, UNSIGNED8 offset) := FUNCTION 
	RETURN p-offset;
END;
// w*x + repmat (b, 1,m)
//A_in :w : by using ALL in JOIN we distribute all records in A_in to each node of B_in
//B_in :x : is already distributed, we don't change its distribution
//bb_in : bias vector (b): using ALL in JOIN, all its record will be distributed to nodes of B_in
WX_repmatb(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_bb, DATASET(Layout_Part) bb_in, PBblas.IMatrix_Map map_c, REAL8 beta=0.0) := FUNCTION
A_offset := 0; 
B_row_part := 2;
SET OF PBblas.Types.value_t empty_array := [];
// First check maps for compatability.  Normalize for transpose operations.
  a_matrix_rows := map_a.matrix_rows;
  a_matrix_cols := map_a.matrix_cols;
  a_row_blocks  := map_a.row_blocks;
  a_col_blocks  := map_a.col_blocks;
  b_matrix_rows := map_b.matrix_rows;
  b_matrix_cols := map_b.matrix_cols;
  b_row_blocks  := map_b.row_blocks;
  b_col_blocks  := map_b.col_blocks;
  c_matrix_rows := map_c.matrix_rows;
  c_matrix_cols := map_c.matrix_cols;
  c_row_blocks  := map_c.row_blocks;
  c_col_blocks  := map_c.col_blocks;
  A := ASSERT(A_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' + ' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(a_matrix_rows=c_matrix_rows AND a_row_blocks=c_row_blocks,
                    'A-C: ' + 'Arows: ' + a_matrix_rows +' Crows '+c_matrix_rows+' '+ 
                              'Ablocks: ' + a_row_blocks +' Cblocks '+c_row_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	B := ASSERT(B_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' +' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(b_matrix_cols=c_matrix_cols AND b_col_blocks=c_col_blocks,
                    'B-C: ' + 'Brows: ' + b_matrix_cols +' Ccols '+c_matrix_cols+' '+ 
                              'Bblocks: ' + b_row_blocks +' Cblocks '+c_col_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	
  //multiply
	
	Layout_Part mul2(Layout_Part b_part, Layout_Part a_part):=TRANSFORM
    real_part_id     := col_block_bigmat (b_part.partition_id, B_row_part);  //arbitrary choice
		part_id := real_part_id;
    part_a_cols := a_part.part_cols;
    part_a_rows := a_part.part_rows;
    part_b_rows := b_part.part_rows;
    part_c_rows := map_c.part_rows(part_id);
    part_c_cols := map_c.part_cols(part_id);
    part_c_first_row  := map_c.first_row(part_id);
    part_c_first_col  := map_c.first_col(part_id);
    k := part_a_cols;
    SELF.partition_id := part_id;
    SELF.node_id      := a_part.node_id;
    SELF.block_row    := a_part.block_row;
    SELF.block_col    := a_part.block_col;
    SELF.first_row    := map_c.first_row(part_id);
    SELF.part_rows    := part_c_rows;
    SELF.first_col    := part_c_first_col;
    SELF.part_cols    := part_c_cols;
    SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
                                    part_c_rows, part_c_cols, k,
                                    1.0, a_part.mat_part, b_part.mat_part,
                                    0.0, empty_array);
  END;
  ab_prod := JOIN(B, A,block_smallmat(LEFT.partition_id, A_offset) = row_block_bigmat (RIGHT.partition_id, B_row_part) , mul2(LEFT,RIGHT),ALL); // Each A (weight matrix) is copied in each B (X matrix) node

// add bias vector to each columns of X
// each bias vector (b) is copied (ALL JOIN) to each node of X. The bias vector is then normalized to repeat it to the number of columns of X in that node and then the two are added
  Layout_Part addbias(Layout_Part cumm, Layout_Part term) := TRANSFORM
		cumm_part_cols := cumm.part_cols;
    N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
		Elem := {PBblas.Types.value_t v};
		Elem_r := {PBblas.Types.value_t v, UNSIGNED r};
		elems := DATASET(term.mat_part, Elem);
		Elem_r rep (Elem l, UNSIGNED c) := TRANSFORM
			SELF.r := c;
			SELF := l;
		END;
		elems_rep := NORMALIZE(elems, cumm_part_cols, rep(LEFT, COUNTER));// each value in bias vector is repeated cumm_part_cols number of times (as the numebr of cumm columns in this node)
		elems_rep_sort := SORT(elems_rep, r);
		term_rep_set := SET (elems_rep_sort, v);
    SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term_rep_set, 1);
		SELF.partition_id := cumm.partition_id;
    SELF := cumm;
  END;
	
	 ab_bb := JOIN(ab_prod, bb_in,TRUE , addbias(LEFT,RIGHT),ALL);
	//rslt := A;
  RETURN ab_bb;
END; // END WX_repmatb



//A_in := w1 is h*f where f is divided to partitions of size prow
//B_in := data : ddist
// bb_in := bias1
//returns the sigmoid of the result
W1X_repmatb1(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_bb, DATASET(Layout_Part) bb_in, PBblas.IMatrix_Map map_c, REAL8 beta=0.0, UNSIGNED8 A_offset) := FUNCTION
SET OF PBblas.Types.value_t empty_array := [];
// First check maps for compatability.  Normalize for transpose operations.
  a_matrix_rows := map_a.matrix_rows;
  a_matrix_cols := map_a.matrix_cols;
  a_row_blocks  := map_a.row_blocks;
  a_col_blocks  := map_a.col_blocks;
  b_matrix_rows := map_b.matrix_rows;
  b_matrix_cols := map_b.matrix_cols;
  b_row_blocks  := map_b.row_blocks;
  b_col_blocks  := map_b.col_blocks;
  c_matrix_rows := map_c.matrix_rows;
  c_matrix_cols := map_c.matrix_cols;
  c_row_blocks  := map_c.row_blocks;
  c_col_blocks  := map_c.col_blocks;
  A := ASSERT(A_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' + ' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(a_matrix_rows=c_matrix_rows AND a_row_blocks=c_row_blocks,
                    'A-C: ' + 'Arows: ' + a_matrix_rows +' Crows '+c_matrix_rows+' '+ 
                              'Ablocks: ' + a_row_blocks +' Cblocks '+c_row_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	B := ASSERT(B_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' +' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(b_matrix_cols=c_matrix_cols AND b_col_blocks=c_col_blocks,
                    'B-C: ' + 'Brows: ' + b_matrix_cols +' Ccols '+c_matrix_cols+' '+ 
                              'Bblocks: ' + b_row_blocks +' Cblocks '+c_col_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	
  //multiply
	B_row_part := map_b.row_blocks;
	C_row_part := map_c.row_blocks;
	Layout_Part mul2(Layout_Part b_part, Layout_Part a_part):=TRANSFORM
    //real_part_id := col_block_bigmat (b_part.partition_id, B_row_part);  //arbitrary choice
		REAL_part_id := b_part.block_col;
		part_id := real_part_id;
    part_a_cols := a_part.part_cols;
    part_a_rows := a_part.part_rows;
    part_b_rows := b_part.part_rows;
    part_c_rows := map_c.part_rows(part_id);
    part_c_cols := map_c.part_cols(part_id);
    part_c_first_row  := map_c.first_row(part_id);
    part_c_first_col  := map_c.first_col(part_id);
    k := part_a_cols;
    SELF.partition_id := part_id;
    SELF.node_id      := b_part.node_id;
    SELF.block_row    := a_part.block_row;
    SELF.block_col    := b_part.block_col;
    SELF.first_row    := map_c.first_row(part_id);
    SELF.part_rows    := part_c_rows;
    SELF.first_col    := part_c_first_col;
    SELF.part_cols    := part_c_cols;
    SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
                                    part_c_rows, part_c_cols, k,
                                    1.0, a_part.mat_part, b_part.mat_part,
                                    0.0, empty_array);
  END;
  //ab_prod_ := JOIN(B, A, row_block_bigmat (LEFT.partition_id, B_row_part) = RIGHT.partition_id , mul2(LEFT,RIGHT),ALL); // Each A (weight matrix) is copied in each B (X matrix) node
	ab_prod_ := JOIN(B, A, LEFT.block_row = RIGHT.block_col , mul2(LEFT,RIGHT),ALL); // Each A (weight matrix) is copied in each B (X matrix) node
	//ab_prod_ := JOIN(B, A, LEFT.block_row = RIGHT.block_col , mul2(LEFT,RIGHT),ALL); //this is not correct, there might be more than one block row in one node

	 // Sum terms
  Layout_Part sumTerms(Layout_Part cumm, Layout_Part term) := TRANSFORM
    N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
    SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term.mat_part, 1);
    SELF := cumm;
  END;
	sorted_ab_prod := SORT(ab_prod_, partition_id, LOCAL);
  ab_prod := ROLLUP(sorted_ab_prod, sumTerms(LEFT, RIGHT), partition_id, LOCAL);
//	 ab_prod := ROLLUP(sorted_ab_prod, LEFT.partition_id = RIGHT.partition_id, sumTerms(LEFT, RIGHT), LOCAL);

// add bias vector to each columns of X
// each bias vector (b) is copied (ALL JOIN) to each node of X. The bias vector is then normalized to repeat itself, number of columns of X time in that node and then the two are added
  Layout_Part addbias(Layout_Part cumm, Layout_Part term) := TRANSFORM
		cumm_part_cols := cumm.part_cols;// number of columns in this partition
    N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
		Elem := {PBblas.Types.value_t v};
		Elem_r := {PBblas.Types.value_t v, UNSIGNED r};
		elems := DATASET(term.mat_part, Elem);
		Elem_r rep (Elem l, UNSIGNED c) := TRANSFORM
			SELF.r := c;
			SELF := l;
		END;
		elems_rep := NORMALIZE(elems, cumm_part_cols, rep(LEFT, COUNTER));// each value in bias vector is repeated cumm_part_cols number of times (as the numebr of cumm columns in this node)
		elems_rep_sort := SORT(elems_rep, r, LOCAL);
		term_rep_set := SET (elems_rep_sort, v);
    SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term_rep_set, 1);
		SELF.partition_id := cumm.partition_id;
    SELF := cumm;
  END;
	
	Layout_Part addbias_(Layout_Part cumm, Layout_Part term) := TRANSFORM
		cumm_part_cols := cumm.part_cols;// number of columns in this partition
		term_part_rows := term.part_rows;//number of elements in this partition of the bias vector
    N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
		Elem := {PBblas.Types.value_t v};
		Elem_r := {PBblas.Types.value_t v, UNSIGNED r};
		elems := DATASET(term.mat_part, Elem);
		Elem_r rep (Elem l, UNSIGNED c) := TRANSFORM
			SELF.r := c;
			SELF := l;
		END;
		elems_rep := NORMALIZE(elems, cumm_part_cols, rep(LEFT, COUNTER));// each value in bias vector is repeated cumm_part_cols number of times (as the numebr of cumm columns in this node)
		elems_rep_sort := SORT(elems_rep, r, LOCAL);
		term_rep_set := repeatbias(term_part_rows, cumm_part_cols, term.mat_part);
    //SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term_rep_set, 1);
		SELF.mat_part := mat_vec_sum_sigmoid(N, term_part_rows, cumm.mat_part, term.mat_part);
		SELF.partition_id := cumm.partition_id;
    SELF := cumm;
  END;
	
	
	 ab_bb := JOIN(ab_prod, bb_in,TRUE , addbias_(LEFT,RIGHT),ALL);
	 
	 
	 	Layout_Part rep_b1(Layout_Part one_part, Layout_Part bb_part):=TRANSFORM
    real_part_id := one_part.partition_id;
		part_id := (real_part_id-1)*C_row_part + bb_part.block_row;
    part_a_cols := bb_part.part_cols;
    part_a_rows := bb_part.part_rows;
    part_b_rows := one_part.part_rows;
    part_c_rows := map_c.part_rows(part_id);
    part_c_cols := map_c.part_cols(part_id);
    part_c_first_row  := map_c.first_row(part_id);
    part_c_first_col  := map_c.first_col(part_id);
    k := part_a_cols;
    SELF.partition_id := part_id;
    SELF.node_id      := one_part.node_id;
    SELF.block_row    := bb_part.block_row;
    SELF.block_col    := one_part.block_col;
    SELF.first_row    := map_c.first_row(part_id);
    SELF.part_rows    := part_c_rows;
    SELF.first_col    := part_c_first_col;
    SELF.part_cols    := part_c_cols;
    SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, TRUE,
                                    part_c_rows, part_c_cols, k,
                                    1.0, bb_part.mat_part, one_part.mat_part,
                                    0.0, empty_array);
  END;

	bb_repeated := JOIN (Ones_Vecdist, bb_in, TRUE , rep_b1(LEFT,RIGHT),ALL);
		 //Ones_VecMap := PBblas.Matrix_Map(m, 1, Ones_Vecdist := TrainLabel;
	ab_bb_ := PBblas.PB_daxpy(1.0, ab_prod, bb_repeated);
	RETURN ab_bb;
END; // END W1X_repmatb1




W2td3_repmatsparsity(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_bb, DATASET(Layout_Part) bb_in, PBblas.IMatrix_Map map_c, REAL8 beta_in=0) := FUNCTION

SET OF PBblas.Types.value_t empty_array := [];
	// First check maps for compatability.  Normalize for transpose operations.
		a_matrix_rows := map_a.matrix_cols;
		a_matrix_cols := map_a.matrix_rows;
		a_row_blocks  := map_a.col_blocks;
		a_col_blocks  := map_a.row_blocks;
		b_matrix_rows := map_b.matrix_rows;
		b_matrix_cols := map_b.matrix_cols;
		b_row_blocks  := map_b.row_blocks;
		b_col_blocks  := map_b.col_blocks;
		c_matrix_rows := map_c.matrix_rows;
		c_matrix_cols := map_c.matrix_cols;
		c_row_blocks  := map_c.row_blocks;
		c_col_blocks  := map_c.col_blocks;
	A := ASSERT(A_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'YES' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'NO' + ' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(a_matrix_rows=c_matrix_rows AND a_row_blocks=c_row_blocks,
                    'A-C: ' + 'Arows: ' + a_matrix_rows +' Crows '+c_matrix_rows+' '+ 
                              'Ablocks: ' + a_row_blocks +' Cblocks '+c_row_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
  B := ASSERT(B_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'YES' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'NO' +' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(b_matrix_cols=c_matrix_cols AND b_col_blocks=c_col_blocks,
                    'B-C: ' + 'Brows: ' + b_matrix_cols +' Ccols '+c_matrix_cols+' '+ 
                              'Bblocks: ' + b_row_blocks +' Cblocks '+c_col_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	
  //multiply
	B_row_part := map_b.row_blocks;
	Layout_Part mul2(Layout_Part b_part, Layout_Part a_part):=TRANSFORM
    part_id := b_part.block_col;
    part_a_cols := a_part.part_cols;
    part_a_rows := a_part.part_rows;
    part_b_rows := b_part.part_rows;
    part_c_rows := map_c.part_rows(part_id);
    part_c_cols := map_c.part_cols(part_id);
    part_c_first_row  := map_c.first_row(part_id);
    part_c_first_col  := map_c.first_col(part_id);
    k := part_a_rows;
    SELF.partition_id := part_id;
    SELF.node_id      := b_part.node_id;
    SELF.block_row    := a_part.block_col;
    SELF.block_col    := b_part.block_col;
    SELF.first_row    := map_c.first_row(part_id);
    SELF.part_rows    := part_c_rows;
    SELF.first_col    := part_c_first_col;
    SELF.part_cols    := part_c_cols;
    SELF.mat_part     := PBblas.BLAS.dgemm(TRUE, FALSE,
                                    part_c_rows, part_c_cols, k,
                                    1.0, a_part.mat_part, b_part.mat_part,
                                    0.0, empty_array);

  END;
  //ab_prod_ := JOIN(B, A, row_block_bigmat (LEFT.partition_id, B_row_part) = RIGHT.partition_id , mul2(LEFT,RIGHT),ALL); // Each A (weight matrix) is copied in each B (X matrix) node
	//ab_prod_ := JOIN(B, A, row_block_bigmat (LEFT.partition_id, B_row_part) = (RIGHT.partition_id-A_offset) , mul2(LEFT,RIGHT),ALL); // Each A (weight matrix) is copied in each B (X matrix) node
	ab_prod_ := JOIN(B, A, LEFT.block_row = RIGHT.block_row, mul2(LEFT,RIGHT),ALL);

	 // Sum terms
  Layout_Part sumTerms(Layout_Part cumm, Layout_Part term) := TRANSFORM
    N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
    SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term.mat_part, 1);
    SELF := cumm;
  END;
	sorted_ab_prod := SORT(ab_prod_, partition_id, LOCAL);
  ab_prod := ROLLUP(sorted_ab_prod, sumTerms(LEFT, RIGHT), partition_id, LOCAL);

Layout_Part addbias(Layout_Part cumm, Layout_Part term) := TRANSFORM
		cumm_part_cols := cumm.part_cols;// number of columns in this partition
		term_part_rows := term.part_rows;//number of elements in this partition of the bias vector
    N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
		SELF.mat_part := mat_vec_sum(N, term_part_rows, cumm.mat_part, term.mat_part, beta_in);
		SELF.partition_id := cumm.partition_id;
    SELF := cumm;
  END;
	
	
	 ab_bb := JOIN(ab_prod, bb_in,TRUE , addbias(LEFT,RIGHT),ALL);
	RETURN ab_bb;
END; // END W2td3_repmatsparsity







//returns the sigmoid of the result
W2a2_repmatb2(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_bb, DATASET(Layout_Part) bb_in, PBblas.IMatrix_Map map_c, REAL8 beta=0.0) := FUNCTION
SET OF PBblas.Types.value_t empty_array := [];
// First check maps for compatability.  Normalize for transpose operations.
  a_matrix_rows := map_a.matrix_rows;
  a_matrix_cols := map_a.matrix_cols;
  a_row_blocks  := map_a.row_blocks;
  a_col_blocks  := map_a.col_blocks;
  b_matrix_rows := map_b.matrix_rows;
  b_matrix_cols := map_b.matrix_cols;
  b_row_blocks  := map_b.row_blocks;
  b_col_blocks  := map_b.col_blocks;
  c_matrix_rows := map_c.matrix_rows;
  c_matrix_cols := map_c.matrix_cols;
  c_row_blocks  := map_c.row_blocks;
  c_col_blocks  := map_c.col_blocks;
  A := ASSERT(A_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' + ' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(a_matrix_rows=c_matrix_rows AND a_row_blocks=c_row_blocks,
                    'A-C: ' + 'Arows: ' + a_matrix_rows +' Crows '+c_matrix_rows+' '+ 
                              'Ablocks: ' + a_row_blocks +' Cblocks '+c_row_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	B := ASSERT(B_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' +' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(b_matrix_cols=c_matrix_cols AND b_col_blocks=c_col_blocks,
                    'B-C: ' + 'Brows: ' + b_matrix_cols +' Ccols '+c_matrix_cols+' '+ 
                              'Bblocks: ' + b_row_blocks +' Cblocks '+c_col_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	
  //multiply
	B_row_part := map_b.row_blocks;
	C_row_part := map_c.row_blocks;
	Layout_Part mul2(Layout_Part b_part, Layout_Part a_part):=TRANSFORM
    real_part_id := b_part.partition_id;  //arbitrary choice
		part_id := (real_part_id-1)*C_row_part + a_part.block_row;
    part_a_cols := a_part.part_cols;
    part_a_rows := a_part.part_rows;
    part_b_rows := b_part.part_rows;
    part_c_rows := map_c.part_rows(part_id);
    part_c_cols := map_c.part_cols(part_id);
    part_c_first_row  := map_c.first_row(part_id);
    part_c_first_col  := map_c.first_col(part_id);
    k := part_a_cols;
    SELF.partition_id := part_id;
    SELF.node_id      := b_part.node_id;
    SELF.block_row    := a_part.block_row;
    SELF.block_col    := b_part.block_col;
    SELF.first_row    := map_c.first_row(part_id);
    SELF.part_rows    := part_c_rows;
    SELF.first_col    := part_c_first_col;
    SELF.part_cols    := part_c_cols;
    SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
                                    part_c_rows, part_c_cols, k,
                                    1.0, a_part.mat_part, b_part.mat_part,
                                    0.0, empty_array);
  END;
  //ab_prod := JOIN(B, A, TRUE , mul2(LEFT,RIGHT),ALL); // Each A (weight matrix) is copied in each B (X matrix) node
	ab_prod := JOIN(B, A, TRUE , mul2(LEFT,RIGHT), ALL); // Each A (weight matrix) is copied in each B (X matrix) node


// add bias vector to each columns of X
// each bias vector (b) is copied (ALL JOIN) to each node of X. The bias vector is then normalized to repeat it to the number of columns of X in that node and then the two are added
  Layout_Part addbias(Layout_Part cumm, Layout_Part term) := TRANSFORM
		cumm_part_cols := cumm.part_cols;// number of columns in this partition
    N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
		Elem := {PBblas.Types.value_t v};
		Elem_r := {PBblas.Types.value_t v, UNSIGNED r};
		elems := DATASET(term.mat_part, Elem);
		Elem_r rep (Elem l, UNSIGNED c) := TRANSFORM
			SELF.r := c;
			SELF := l;
		END;
		elems_rep := NORMALIZE(elems, cumm_part_cols, rep(LEFT, COUNTER));// each value in bias vector is repeated cumm_part_cols number of times (as the numebr of cumm columns in this node)
		elems_rep_sort := SORT(elems_rep, r, LOCAL);
		term_rep_set := SET (elems_rep_sort, v);
    SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term_rep_set, 1);
		SELF.partition_id := cumm.partition_id;
    SELF := cumm;
  END;
	
	Layout_Part addbias_(Layout_Part cumm, Layout_Part term) := TRANSFORM
		cumm_part_cols := cumm.part_cols;// number of columns in this partition
		term_part_rows := term.part_rows;
    N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
		Elem := {PBblas.Types.value_t v};
		Elem_r := {PBblas.Types.value_t v, UNSIGNED r};
		elems := DATASET(term.mat_part, Elem);
		Elem_r rep (Elem l, UNSIGNED c) := TRANSFORM
			SELF.r := c;
			SELF := l;
		END;
		elems_rep := NORMALIZE(elems, cumm_part_cols, rep(LEFT, COUNTER));// each value in bias vector is repeated cumm_part_cols number of times (as the numebr of cumm columns in this node)
		elems_rep_sort := SORT(elems_rep, r, LOCAL);
		term_rep_set := repeatbias(term_part_rows, cumm_part_cols, term.mat_part);
    //SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term_rep_set, 1);
		SELF.mat_part := mat_vec_sum_sigmoid(N, term_part_rows, cumm.mat_part, term.mat_part);
		SELF.partition_id := cumm.partition_id;
    SELF := cumm;
  END;
	
	 ab_bb := JOIN(ab_prod, bb_in,LEFT.block_row = RIGHT.block_row , addbias_(LEFT,RIGHT),LOOKUP);
	 
	 	Layout_Part rep_b2(Layout_Part one_part, Layout_Part bb_part):=TRANSFORM
    real_part_id := one_part.partition_id;
		part_id := (real_part_id-1)*C_row_part + bb_part.block_row;
    part_a_cols := bb_part.part_cols;
    part_a_rows := bb_part.part_rows;
    part_b_rows := one_part.part_rows;
    part_c_rows := map_c.part_rows(part_id);
    part_c_cols := map_c.part_cols(part_id);
    part_c_first_row  := map_c.first_row(part_id);
    part_c_first_col  := map_c.first_col(part_id);
    k := part_a_cols;
    SELF.partition_id := part_id;
    SELF.node_id      := one_part.node_id;
    SELF.block_row    := bb_part.block_row;
    SELF.block_col    := one_part.block_col;
    SELF.first_row    := map_c.first_row(part_id);
    SELF.part_rows    := part_c_rows;
    SELF.first_col    := part_c_first_col;
    SELF.part_cols    := part_c_cols;
    SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, TRUE,
                                    part_c_rows, part_c_cols, k,
                                    1.0, bb_part.mat_part, one_part.mat_part,
                                    0.0, empty_array);
  END;

	bb_repeated := JOIN (Ones_Vecdist, bb_in, TRUE , rep_b2(LEFT,RIGHT),ALL);
		 //Ones_VecMap := PBblas.Matrix_Map(m, 1, Ones_Vecdist := TrainLabel;
	ab_bb_ := PBblas.PB_daxpy(1.0, ab_prod, bb_repeated);
	//rslt := A;
  //RETURN PROJECT (B, TRANSFORM (layout_part, SELF.partition_id := row_block_bigmat (LEFT.partition_id, B_row_part), SELF:= LEFT));
	RETURN ab_bb;
END; // END W2a2_repmatb2

  //retunrs the sigmoid(WX+b)  
WX_repmatb_sig(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_bb, DATASET(Layout_Part) bb_in, PBblas.IMatrix_Map map_c, REAL8 beta=0.0) := FUNCTION
SET OF PBblas.Types.value_t empty_array := [];
// First check maps for compatability.  Normalize for transpose operations.
  a_matrix_rows := map_a.matrix_rows;
  a_matrix_cols := map_a.matrix_cols;
  a_row_blocks  := map_a.row_blocks;
  a_col_blocks  := map_a.col_blocks;
  b_matrix_rows := map_b.matrix_rows;
  b_matrix_cols := map_b.matrix_cols;
  b_row_blocks  := map_b.row_blocks;
  b_col_blocks  := map_b.col_blocks;
  c_matrix_rows := map_c.matrix_rows;
  c_matrix_cols := map_c.matrix_cols;
  c_row_blocks  := map_c.row_blocks;
  c_col_blocks  := map_c.col_blocks;
  A := ASSERT(A_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' + ' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(a_matrix_rows=c_matrix_rows AND a_row_blocks=c_row_blocks,
                    'A-C: ' + 'Arows: ' + a_matrix_rows +' Crows '+c_matrix_rows+' '+ 
                              'Ablocks: ' + a_row_blocks +' Cblocks '+c_row_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	B := ASSERT(B_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' +' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(b_matrix_cols=c_matrix_cols AND b_col_blocks=c_col_blocks,
                    'B-C: ' + 'Brows: ' + b_matrix_cols +' Ccols '+c_matrix_cols+' '+ 
                              'Bblocks: ' + b_row_blocks +' Cblocks '+c_col_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	
  //multiply
	
	Layout_Part mul2(Layout_Part b_part, Layout_Part a_part):=TRANSFORM
    part_id     := b_part.partition_id;    //arbitrary choice
    part_a_cols := a_part.part_cols;
    part_a_rows := a_part.part_rows;
    part_b_rows := b_part.part_rows;
    part_c_rows := map_c.part_rows(part_id);
    part_c_cols := map_c.part_cols(part_id);
    part_c_first_row  := map_c.first_row(part_id);
    part_c_first_col  := map_c.first_col(part_id);
    k := part_a_cols;
    SELF.partition_id := b_part.partition_id;
    SELF.node_id      := b_part.node_id;
    SELF.block_row    := b_part.block_row;
    SELF.block_col    := b_part.block_col;
    SELF.first_row    := map_c.first_row(part_id);
    SELF.part_rows    := part_c_rows;
    SELF.first_col    := part_c_first_col;
    SELF.part_cols    := part_c_cols;
    SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
                                    part_c_rows, part_c_cols, k,
                                    1.0, a_part.mat_part, b_part.mat_part,
                                    0.0, empty_array);
  END;
  ab_prod := JOIN(B, A,TRUE , mul2(LEFT,RIGHT),ALL); // Each A (weight matrix) is copied in each B (X matrix) node

// add bias vector to each columns of X
// each bias vector (b) is copied (ALL JOIN) to each node of X. The bias vector is then normalized to repeat it to the number of columns of X in that node and then the two are added
  Layout_Part addbias(Layout_Part cumm, Layout_Part term) := TRANSFORM
		cumm_part_cols := cumm.part_cols;
    N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
		Elem := {PBblas.Types.value_t v};
		Elem_r := {PBblas.Types.value_t v, UNSIGNED r};
		elems := DATASET(term.mat_part, Elem);
		Elem_r rep (Elem l, UNSIGNED c) := TRANSFORM
			SELF.r := c;
			SELF := l;
		END;
		elems_rep := NORMALIZE(elems, cumm_part_cols, rep(LEFT, COUNTER));// each value in bias vector is repeated cumm_part_cols number of times (as the numebr of cumm columns in this node)
		elems_rep_sort := SORT(elems_rep, r);
		term_rep_set := SET (elems_rep_sort, v);
    SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term_rep_set, 1);
		SELF.partition_id := cumm.partition_id;
    SELF := cumm;
  END;
	
	 ab_bb := JOIN(ab_prod, bb_in,TRUE , addbias(LEFT,RIGHT),ALL);
	//rslt := A;
  RETURN ab_bb;
END; // END WX_repmatb_sig
		
		

		
		
		
		

WtX_repmatb(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_bb, DATASET(Layout_Part) bb_in, PBblas.IMatrix_Map map_c, REAL8 beta=0.0) := FUNCTION
	SET OF PBblas.Types.value_t empty_array := [];
	// First check maps for compatability.  Normalize for transpose operations.
		a_matrix_rows := map_a.matrix_cols;
		a_matrix_cols := map_a.matrix_rows;
		a_row_blocks  := map_a.col_blocks;
		a_col_blocks  := map_a.row_blocks;
		b_matrix_rows := map_b.matrix_rows;
		b_matrix_cols := map_b.matrix_cols;
		b_row_blocks  := map_b.row_blocks;
		b_col_blocks  := map_b.col_blocks;
		c_matrix_rows := map_c.matrix_rows;
		c_matrix_cols := map_c.matrix_cols;
		c_row_blocks  := map_c.row_blocks;
		c_col_blocks  := map_c.col_blocks;
		A := ASSERT(A_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'YES' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'NO' + ' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(a_matrix_rows=c_matrix_rows AND a_row_blocks=c_row_blocks,
                    'A-C: ' + 'Arows: ' + a_matrix_rows +' Crows '+c_matrix_rows+' '+ 
                              'Ablocks: ' + a_row_blocks +' Cblocks '+c_row_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
  B := ASSERT(B_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'YES' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'NO' +' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(b_matrix_cols=c_matrix_cols AND b_col_blocks=c_col_blocks,
                    'B-C: ' + 'Brows: ' + b_matrix_cols +' Ccols '+c_matrix_cols+' '+ 
                              'Bblocks: ' + b_row_blocks +' Cblocks '+c_col_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
		
		//multiply
		
		Layout_Part mul2(Layout_Part b_part, Layout_Part a_part):=TRANSFORM
			part_id     := b_part.partition_id;    //arbitrary choice
			part_a_cols := a_part.part_cols;
			part_a_rows := a_part.part_rows;
			part_b_rows := b_part.part_rows;
			part_c_rows := map_c.part_rows(part_id);
			part_c_cols := map_c.part_cols(part_id);
			part_c_first_row  := map_c.first_row(part_id);
			part_c_first_col  := map_c.first_col(part_id);
			k := part_a_rows;
			SELF.partition_id := b_part.partition_id;
			SELF.node_id      := b_part.node_id;
			SELF.block_row    := b_part.block_row;
			SELF.block_col    := b_part.block_col;
			SELF.first_row    := map_c.first_row(part_id);
			SELF.part_rows    := part_c_rows;
			SELF.first_col    := part_c_first_col;
			SELF.part_cols    := part_c_cols;
			SELF.mat_part     := PBblas.BLAS.dgemm(TRUE, FALSE,
																			part_c_rows, part_c_cols, k,
																			1.0, a_part.mat_part, b_part.mat_part,
																			0.0, empty_array);
		END;
		ab_prod := JOIN(B, A,TRUE , mul2(LEFT,RIGHT),ALL); // Each A (weight matrix) is copied in each B (X matrix) node



// Apply beta
  Layout_Part applyBeta(Layout_Part part) := TRANSFORM
    SELF.mat_part := PBblas.BLAS.dscal(map_bb.matrix_rows*map_bb.matrix_cols,
                                beta, part.mat_part, 1);
    SELF:= part;
  END;
  bb_beta := PROJECT(bb_in, applyBeta(LEFT), LOCAL);
	// add the vector to each columns of X
	// each vector is copied (ALL JOIN) to each node of X. The vector is then normalized to repeat it to the number of columns of X in that node and then the two are added
		Layout_Part addvec(Layout_Part cumm, Layout_Part term) := TRANSFORM
			cumm_part_cols := cumm.part_cols;
			N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
			Elem := {PBblas.Types.value_t v};
			Elem_r := {PBblas.Types.value_t v, UNSIGNED r};
			elems := DATASET(term.mat_part, Elem);
			Elem_r rep (Elem l, UNSIGNED c) := TRANSFORM
				SELF.r := c;
				SELF := l;
			END;
			elems_rep := NORMALIZE(elems, cumm_part_cols, rep(LEFT, COUNTER));// each value in bias vector is repeated cumm_part_cols number of times (as the numebr of cumm columns in this node)
			elems_rep_sort := SORT(elems_rep, r);
			term_rep_set := SET (elems_rep_sort, v);
			SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term_rep_set, 1);
			SELF.partition_id := cumm.partition_id;
			SELF := cumm;
		END;
		
		 ab_bb := JOIN(ab_prod, bb_beta,TRUE , addvec(LEFT,RIGHT),ALL);

		//rslt := A;
		RETURN ab_bb;
END; // END WtX_repmatb
		
		
		
		// the input is a matrix in PBblas format where only columns are partitions
		//B_in : ones vector which is partitioned among nodes
		// map_c is the result's map
		col_sum(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_c) := FUNCTION
			SET OF PBblas.Types.value_t empty_array := [];
				Layout_Part mul(Layout_Part a_part, Layout_Part b_part):=TRANSFORM
					part_id     := 1;    //arbitrary choice
					part_a_cols := a_part.part_cols;
					part_a_rows := a_part.part_rows;
					part_b_rows := b_part.part_rows;
					part_c_rows := map_c.part_rows(part_id);
					part_c_cols := map_c.part_cols(part_id);
					part_c_first_row  := map_c.first_row(part_id);
					part_c_first_col  := map_c.first_col(part_id);
					k := part_a_cols;
					SELF.partition_id := 1;
					SELF.node_id      := a_part.node_id;
					SELF.block_row    := 1;
					SELF.block_col    := 1;
					SELF.first_row    := map_c.first_row(part_id);
					SELF.part_rows    := part_c_rows;
					SELF.first_col    := part_c_first_col;
					SELF.part_cols    := part_c_cols;
					SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
																					part_c_rows, part_c_cols, k,
																					1.0, a_part.mat_part, b_part.mat_part,
																					0.0, empty_array);
			END;
			col_sum_part := JOIN (A_in, B_in, LEFT.partition_id = RIGHT.partition_id, mul(LEFT, RIGHT), LOCAL );
			
			Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := le.part_rows ;
				SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1);
				SELF := le;
			END;
			//rslt := ROLLUP(col_sum_part, addup(LEFT, RIGHT), partition_id); // overload becasue of grouping
			rslt := ROLLUP(col_sum_part,TRUE, addup(LEFT, RIGHT)); // no groupijng, reduces overload
			//distribute to node one
			RETURN rslt; 
		END;//END Col_Sum
		
		
		col_mean(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_c, REAL8 mean_coef) := FUNCTION
		
			Num := map_a.matrix_cols;
			Num_1 := 1/Num;
			SET OF PBblas.Types.value_t empty_array := [];
				Layout_Part mul(Layout_Part a_part, Layout_Part b_part):=TRANSFORM
					part_id     := 1;    //arbitrary choice
					part_a_cols := a_part.part_cols;
					part_a_rows := a_part.part_rows;
					part_b_rows := b_part.part_rows;
					part_c_rows := map_c.part_rows(part_id);
					part_c_cols := map_c.part_cols(part_id);
					part_c_first_row  := map_c.first_row(part_id);
					part_c_first_col  := map_c.first_col(part_id);
					k := part_a_cols;
					SELF.partition_id := 1;
					SELF.node_id      := 0;
					SELF.block_row    := 1;
					SELF.block_col    := 1;
					SELF.first_row    := map_c.first_row(part_id);
					SELF.part_rows    := part_c_rows;
					SELF.first_col    := part_c_first_col;
					SELF.part_cols    := part_c_cols;
					SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
																					part_c_rows, part_c_cols, k,
																					Num_1, a_part.mat_part, b_part.mat_part,
																					0.0, empty_array);
			END;
			col_sum_part_ := JOIN (A_in, B_in, LEFT.partition_id = RIGHT.partition_id, mul(LEFT, RIGHT), LOCAL );
			
			
			Layout_Part col_sum_tran(Layout_Part the_part):= TRANSFORM
				SELF.mat_part := sum_col_alpha (the_part.part_rows, the_part.part_cols, the_part.mat_part, Num_1);
				SELF := the_part;
			END;
			col_sum_part := PROJECT (A_in, col_sum_tran(LEFT),LOCAL);
			Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := le.part_rows ;
				SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1);
				SELF := le;
			END;
			rslt := ROLLUP(col_sum_part,TRUE, addup(LEFT, RIGHT)); // this form of rollup avoid grouping, ROLLUP(col_sum_part,addup(LEFT, RIGHT, partiion_id)) cause all the records with the same partiion_id
			//get grouped to one node which cause a lot of overload. The current rollup form avoidds grouping and improves performance
			final_rslt := DISTRIBUTE (rslt, node_id); 
			//distribute to node one
			RETURN rslt;
		END;//END colmean
		//sum (A_in,2)
		colmean_bias_grad (PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, UNSIGNED b_offset) := FUNCTION
		
			Num := map_a.matrix_cols;
			Num_1 := 1/Num;
			
			Layout_Part col_sum_tran(Layout_Part the_part):= TRANSFORM
				SELF.mat_part := sum_col_alpha (the_part.part_rows, the_part.part_cols, the_part.mat_part, Num_1);
				part_id := the_part.block_row + b_offset;
				real_part :=  the_part.block_row;
				SELF. partition_id := part_id;
				SELF.node_id := part_id-1;
				SELF.block_row    := real_part;
				SELF.block_col    := 1;
				SELF.first_row    := map_b.first_row(real_part);
				SELF.part_rows    := map_b.part_rows(real_part);
				SELF.first_col    := map_b.first_col(real_part);
				SELF.part_cols    := map_b.part_cols(real_part);
				SELF := the_part;
			END;
			col_sum_part := PROJECT (A_in, col_sum_tran(LEFT),LOCAL);
			col_sum_part_dist := DISTRIBUTE (col_sum_part, node_id);
			Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := le.part_rows ;
				SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1);
				SELF := le;
			END;
			rslt := ROLLUP(col_sum_part_dist,LEFT.partition_id = RIGHT.partition_id, addup(LEFT, RIGHT), LOCAL); // this form of rollup avoid grouping, ROLLUP(col_sum_part,addup(LEFT, RIGHT, partiion_id)) cause all the records with the same partiion_id
			//get grouped to one node which cause a lot of overload. The current rollup form avoidds grouping and improves performance

			RETURN rslt;
		END;//END colmean_bias_grad
		
		
		// This function gets two big matrices which are distributed over all nodes and generate a final relatively smaller matrix which is on one node
		// this is used for weight gradient calculation where for example a h*m matrix is multiplied by a m*f matrix. PBblas will distribute all partitions in the first and second matrix to only one node which final matrix is in
		// this causes overhead, to avoid that we multiply each col partition of first matrix with a row partition of the second matrix in each node, the final generated matrices are added up to generate the final matrix
		// this way, we don't change the distribution of the first and second matrices
		big_big_small(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_c, PBblas.Types.value_t alph=1.0) := FUNCTION
			SET OF PBblas.Types.value_t empty_array := [];
				Layout_Part mul(Layout_Part a_part, Layout_Part b_part):=TRANSFORM
					part_id     := 1;    //arbitrary choice
					part_a_cols := a_part.part_cols;
					part_a_rows := a_part.part_rows;
					part_b_rows := b_part.part_rows;
					part_b_cols := b_part.part_cols;
					k := part_a_cols;
					SELF.partition_id := part_id;
					SELF.node_id      := 0;
					SELF.block_row    := 1;
					SELF.block_col    := 1;
					SELF.first_row    := map_c.first_row(part_id);
					SELF.part_rows    := map_c.part_rows(part_id);
					SELF.first_col    := map_c.first_col(part_id);
					SELF.part_cols    := map_c.part_cols(part_id);
					SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, TRUE,
																					part_a_rows, part_b_rows, k,
																					alph, a_part.mat_part, b_part.mat_part,
																					0.0, empty_array);
			END;
			mul_part := JOIN (A_in, B_in, LEFT.partition_id = RIGHT.partition_id, mul(LEFT, RIGHT), LOCAL );
			
			Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := le.part_rows * le.part_cols ;
				SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1);
				SELF := le;
			END;
			
			Layout_Part addup_it(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := ri.part_rows * ri.part_cols ;
				SELF.mat_part := IF (le.partition_id=0, ri.mat_part, PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1));
				SELF := ri;
			END;
			//rslt := ROLLUP(mul_part, addup(LEFT, RIGHT), partition_id); // since the results of rohat is used in a ALL join, no need to distribute this to node one to be consistent with PBblas
			//rslt := ITERATE(mul_part, addup_it(LEFT, RIGHT));// using rollup cause the graph to Group all the records which are distributed between all node to only one record and then do the operation, It takes a long time to GROUP all thoese partitions in one node and we avoid it by using ITERATE instead of ROLLUP

      rslt := ROLLUP(mul_part, TRUE, addup(LEFT, RIGHT));
			final_rslt := DISTRIBUTE (rslt, node_id); 

// a_part := A_in[1];
// b_part := B_in[1];
		 // RETURN PBblas.BLAS.dgemm(FALSE, TRUE,
																					// a_part.part_rows, b_part.part_rows, a_part.part_cols,
																					// 1.0, a_part.mat_part, b_part.mat_part,
																					// 0.0, empty_array);
																					
			RETURN final_rslt;
		END;// END big_big_small
		//calculates coef * (d2*x')
		d2Xt(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_c, REAL8 coef) := FUNCTION
			SET OF PBblas.Types.value_t empty_array := [];
				Layout_Part mul(Layout_Part a_part, Layout_Part b_part):=TRANSFORM
					part_id     := b_part.block_row;
					part_a_cols := a_part.part_cols;
					part_a_rows := a_part.part_rows;
					part_b_rows := b_part.part_rows;
					part_b_cols := b_part.part_cols;
					k := part_a_cols;
					SELF.partition_id := part_id;
					SELF.node_id      := part_id-1;
					SELF.block_row    := a_part.block_row;
					SELF.block_col    := b_part.block_row;
					SELF.first_row    := map_c.first_row(part_id);
					SELF.part_rows    := map_c.part_rows(part_id);
					SELF.first_col    := map_c.first_col(part_id);
					SELF.part_cols    := map_c.part_cols(part_id);
					SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, TRUE,
																					part_a_rows, part_b_rows, k,
																					coef, a_part.mat_part, b_part.mat_part,
																					0.0, empty_array);
			END;
			B_row_part := map_b.row_blocks;
			mul_part := JOIN (A_in, B_in, LEFT.block_col = RIGHT.block_col, mul(LEFT, RIGHT), LOCAL );
			
			Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := le.part_rows * le.part_cols ;
				SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1);
				SELF := le;
			END;
			mul_part_dist := DISTRIBUTE (mul_part, node_id);
			mul_part_dist_sorted := SORT (mul_part_dist, partition_id, LOCAL);
      rslt := ROLLUP(mul_part_dist_sorted, LEFT.partition_id = RIGHT.partition_id, addup(LEFT, RIGHT), LOCAL);
			// mul_part_sort := SORT (mul_part, partition_id);
			//rslt := ROLLUP(mul_part_sort, addup(LEFT, RIGHT), partition_id);
																					
			RETURN rslt;
		END;// END d2Xt
		
		d3a2t(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_c, UNSIGNED B_offset, REAL8 coef) := FUNCTION
			SET OF PBblas.Types.value_t empty_array := [];
				Layout_Part mul(Layout_Part a_part, Layout_Part b_part):=TRANSFORM
					
					part_id := a_part.block_row;
					new_part_id     := part_id + B_offset;
					part_a_cols := a_part.part_cols;
					part_a_rows := a_part.part_rows;
					part_b_rows := b_part.part_rows;
					part_b_cols := b_part.part_cols;
					k := part_a_cols;
					SELF.partition_id := new_part_id;
					SELF.node_id      := new_part_id-1;
					SELF.block_row    := a_part.block_row;
					SELF.block_col    := b_part.block_row;
					SELF.first_row    := map_c.first_row(part_id);
					SELF.part_rows    := map_c.part_rows(part_id);
					SELF.first_col    := map_c.first_col(part_id);
					SELF.part_cols    := map_c.part_cols(part_id);
					SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, TRUE,
																					part_a_rows, part_b_rows, k,
																					coef, a_part.mat_part, b_part.mat_part,
																					0.0, empty_array);
			END;
			A_row_part := map_a.row_blocks;
			mul_part := JOIN (A_in, B_in, LEFT.block_col = RIGHT.block_col, mul(LEFT, RIGHT), LOCAL );
			
			Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := le.part_rows * le.part_cols ;
				SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1);
				SELF := le;
			END;
			

			mul_part_dist := DISTRIBUTE (mul_part, node_id);
			mul_part_dist_sorted := SORT (mul_part_dist, partition_id, LOCAL);
      rslt := ROLLUP(mul_part_dist_sorted, LEFT.partition_id = RIGHT.partition_id, addup(LEFT, RIGHT), LOCAL);
																					
			RETURN rslt;
		END;// END d3a2t
		
				d3a2t_test(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_c, UNSIGNED B_offset, REAL8 coef) := FUNCTION
			SET OF PBblas.Types.value_t empty_array := [];
				Layout_Part mul(Layout_Part a_part, Layout_Part b_part):=TRANSFORM
					
					part_id := a_part.block_row;
					new_part_id     := part_id + B_offset;
					part_a_cols := a_part.part_cols;
					part_a_rows := a_part.part_rows;
					part_b_rows := b_part.part_rows;
					part_b_cols := b_part.part_cols;
					k := part_a_cols;
					SELF.partition_id := new_part_id;
					SELF.node_id      := new_part_id-1;
					SELF.block_row    := a_part.block_row;
					SELF.block_col    := b_part.block_row;
					SELF.first_row    := map_c.first_row(part_id);
					SELF.part_rows    := map_c.part_rows(part_id);
					SELF.first_col    := map_c.first_col(part_id);
					SELF.part_cols    := map_c.part_cols(part_id);
					SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, TRUE,
																					part_a_rows, part_b_rows, k,
																					coef, a_part.mat_part, b_part.mat_part,
																					0.0, empty_array);
			END;
			A_row_part := map_a.row_blocks;
			mul_part := JOIN (A_in, B_in, LEFT.block_col = RIGHT.block_col, mul(LEFT, RIGHT), LOCAL );
			
			Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := le.part_rows * le.part_cols ;
				SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1);
				SELF := le;
			END;
			

			mul_part_dist := DISTRIBUTE (mul_part, node_id);
			mul_part_dist_sorted := SORT (mul_part_dist, partition_id, LOCAL);
      rslt := ROLLUP(mul_part_dist_sorted, LEFT.partition_id = RIGHT.partition_id, addup(LEFT, RIGHT), LOCAL);
																					
			RETURN mul_part_dist;
		END;// END d3a2t_test
		
		
		//extract sparse autoencoder parameters
   

    //FF2 returns a2
    FF2_(DATASET(Layout_Part) w1, DATASET(Layout_Part) b1v):= FUNCTION
      //b1m = repmat(b1v,1,m)
      b1m := PBblas.PB_dgemm(FALSE, TRUE, 1.0,b1vecmap, b1v, Ones_VecMap, Ones_Vecdist, b1map);
      //z2 = w1*X+b1;
      //z2 := PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1, dmap, ddist, b1map, b1m, 1.0); // gives MP closed error
      z2_ := PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1, dmap, ddist, b1map);
      z2 := PBblas.PB_daxpy(1.0, z2_, b1m);
      //a2 = sigmoid (z2);
      a2 := PBblas.Apply2Elements(b1map, z2, sigmoid);
      RETURN a2;
    END;//END FF2_
		
		
		
		//returns a2
		 // FF2(DATASET(Layout_Part) w1, DATASET(Layout_Part) b1v):= FUNCTION
      // z2=w1*x+repmat(b1,1,m)
			// z2 := WX_repmatb(w1map, w1, dmap, ddist, b1vecmap, b1v, b1map, 0.0);
      // a2 = sigmoid (z2);
      // a2 := PBblas.Apply2Elements(b1map, z2, sigmoid);
      // RETURN a2;
     // END;//END FF2
		 
		 
		
		FF2(DATASET(Layout_Part) w1, DATASET(Layout_Part) b1v):= FUNCTION
      //z2=w1*x+repmat(b1,1,m)
			//z2 := WX_repmatb(w1map, w1, dmap, ddist, b1vecmap, b1v, b1map, 0.0);
			z2 := W1X_repmatb1(w1map, w1, dmap, ddist, b1vecmap, b1v, b1map, 0.0, 0);
      //a2 = sigmoid (z2);
      a2 := PBblas.Apply2Elements(b1map, z2, sigmoid);
      RETURN z2;
     END;//END FF2
		 
    //FF3 returns a3
    FF3_(DATASET(Layout_Part) w2,DATASET(Layout_Part) b2v, DATASET(Layout_Part) a2 ):= FUNCTION
      //b2m = repmat(b2v,1,m)
      b2m := PBblas.PB_dgemm(FALSE, TRUE, 1.0,b2vecmap, b2v, Ones_VecMap, Ones_Vecdist, b2map);
      //z3 = w2*a2+b2;
      //z3 := PBblas.PB_dgemm(FALSE, FALSE,1.0,w2map, w2, a2map, a2, b2map,b2m, 1.0); //gives MP closed error
      z3_ := PBblas.PB_dgemm(FALSE, FALSE,1.0,w2map, w2, a2map, a2, b2map);
      z3 := PBblas.PB_daxpy(1.0, z3_, b2m);
      //a3 = sigmoid (z3)
      a3 := PBblas.Apply2Elements(b2map, z3, sigmoid);
      RETURN a3;
    END;//END FF3_
		
		 //FF3 returns a3
    // FF3(DATASET(Layout_Part) w2,DATASET(Layout_Part) b2v, DATASET(Layout_Part) a2 ):= FUNCTION
		  // z3 = w2*a2 + repmat(b2,1,m)
			// z3 := WX_repmatb(w2map, w2, a2map, a2, b2vecmap, b2v, b2map, 0.0);
      // a3 = sigmoid (z3)
      // a3 := PBblas.Apply2Elements(b2map, z3, sigmoid);
      // RETURN a3;
    // END;//END FF3
		
	 FF3(DATASET(Layout_Part) w2,DATASET(Layout_Part) b2v, DATASET(Layout_Part) a2 ):= FUNCTION
		  //z3 = w2*a2 + repmat(b2,1,m)
			//z3 := WX_repmatb(w2map, w2, a2map, a2, b2vecmap, b2v, b2map, 0.0);
			z3 := W2a2_repmatb2(w2map, w2, a2map, a2, b2vecmap, b2v, b2map, 0.0);
      //a3 = sigmoid (z3)
      a3 := PBblas.Apply2Elements(b2map, z3, sigmoid);
      RETURN z3;
    END;//END FF3
		

    //DELTA3 returns d3
    DELTA3 (DATASET(Layout_Part) a3 ) := FUNCTION
      //calculate delta for the last layer (3rd layer)
      //y=X;
      //d3=-(y-a3).*(a3.*(1-a3));
      // siggrad_a3 := PBblas.Apply2Elements(a3map, a3, siggrad);
      // a3_y := PBblas.PB_daxpy(-1, ddist, a3);
      // d3 := PBblas.HadamardProduct(a3map, a3_y, siggrad_a3);
			Layout_part d3_tran (Layout_part a3_part, Layout_part y_part) := TRANSFORM
				SELF.mat_part := d3_cal(a3_part.part_rows * a3_part.part_cols, a3_part.mat_part, y_part.mat_part);
				SELF := a3_part;
			END;
			d3 := JOIN (a3, ddist,LEFT.partition_id = RIGHT.partition_id, d3_tran(LEFT, RIGHT), LOCAL);


      RETURN d3 ;
    END;//END DELTA3
		
		DELTA3_ (DATASET(Layout_Part) a3 ) := FUNCTION
      //calculate delta for the last layer (3rd layer)
      //y=X;
      //d3=-(y-a3).*(a3.*(1-a3));
      siggrad_a3 := PBblas.Apply2Elements(a3map, a3, siggrad);
      a3_y := PBblas.PB_daxpy(-1, ddist, a3);
      d3 := PBblas.HadamardProduct(a3map, a3_y, siggrad_a3);
      RETURN d3 ;
    END;//END DELTA3_
    //DELTA2 retunrs d2
    rohat_ (DATASET(Layout_Part) a2) := FUNCTION
      rh := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a2map, a2, Ones_VecMap, Ones_Vecdist, Hiddmap);
      RETURN rh;
    END;
		rohat (DATASET(Layout_Part) a2) := FUNCTION
			//rh := col_mean(a2map, a2, Ones_VecMap, Ones_Vecdist, Hiddmap, 1.0);
			rh := colmean_bias_grad (a2map, a2, Hiddmap, 0);
      RETURN rh;
    END;
    DELTA2_ (DATASET(Layout_Part) w2, DATASET(Layout_Part) a2, DATASET(Layout_Part) d3, DATASET(Layout_Part) rhat) := FUNCTION
      //calculate delta for 2nd layer
      //rhohat=mean(a2,2);
      //sparsity_delta=((-sparsityParam./rhohat)+((1-sparsityParam)./(1.-rhohat)));
      //d2=((W2'*d3)+beta*repmat(sparsity_delta,1,m)) .* (a2.*(1-a2));
      //rhohat := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a2map, a2, Ones_VecMap, Ones_Vecdist, Hiddmap);
      rhohat := rhat;
      sparsity_delta := PBblas.Apply2Elements(Hiddmap, rhohat, sp_delta);
      siggrad_a2 := PBblas.Apply2Elements(a2map, a2, siggrad);
      repmat_sparsity_delta := PBblas.PB_dgemm(FALSE, TRUE, 1.0,  Hiddmap, sparsity_delta, Ones_VecMap, Ones_Vecdist, a2map);
      //d2_firstterm = (W2'*d3)+beta*repmat(sparsity_delta,1,m);
      //d2_firstterm := PBblas.PB_dgemm(TRUE, FALSE, 1.0, w2map, w2, a3map, d3, a2map, repmat_sparsity_delta, BETA); // MP close error
      d2_firstterm_ := PBblas.PB_dgemm(TRUE, FALSE, 1.0, w2map, w2, a3map, d3, a2map);
      d2_firstterm := PBblas.PB_daxpy(BETA, repmat_sparsity_delta, d2_firstterm_);
      d2 := PBblas.HadamardProduct(a2map, d2_firstterm, siggrad_a2);
      RETURN d2 ;
    END;
		
		
		DELTA2 (DATASET(Layout_Part) w2, DATASET(Layout_Part) a2, DATASET(Layout_Part) d3, DATASET(Layout_Part) rhat) := FUNCTION
      //calculate delta for 2nd layer
      //rhohat=mean(a2,2);
      //sparsity_delta=((-sparsityParam./rhohat)+((1-sparsityParam)./(1.-rhohat)));
      //d2=((W2'*d3)+beta*repmat(sparsity_delta,1,m)) .* (a2.*(1-a2));
      //rhohat := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a2map, a2, Ones_VecMap, Ones_Vecdist, Hiddmap);
      rhohat := rhat;
      sparsity_delta := PBblas.Apply2Elements(Hiddmap, rhohat, sp_delta);
      // siggrad_a2 := PBblas.Apply2Elements(a2map, a2, siggrad);
      //repmat_sparsity_delta := PBblas.PB_dgemm(FALSE, TRUE, 1.0,  Hiddmap, sparsity_delta, Ones_VecMap, Ones_Vecdist, a2map);
      //d2_firstterm = (W2'*d3)+beta*repmat(sparsity_delta,1,m);
			//d2_firstterm := WtX_repmatb(w2map, w2, a3map, d3, Hiddmap, sparsity_delta, a2map, BETA);
			d2_firstterm := W2td3_repmatsparsity(w2map, w2, a3map, d3, Hiddmap, sparsity_delta, a2map, BETA);
      //d2 := PBblas.HadamardProduct(a2map, d2_firstterm, siggrad_a2);
			Layout_part d2_tran (Layout_part a2_part, Layout_part d2_part) := TRANSFORM
				SELF.mat_part := d2_cal(a2_part.part_rows * a2_part.part_cols, a2_part.mat_part, d2_part.mat_part);
				SELF := a2_part;
			END;
			d2 := JOIN (a2, d2_firstterm, LEFT.partition_id = RIGHT.partition_id, d2_tran(LEFT, RIGHT), LOCAL);
      RETURN d2;
    END;
    //WeightGrad1 returns gradient for w1
    WeightGrad1_ (DATASET(Layout_Part) w1, DATASET(Layout_Part) d2) := FUNCTION
      w1_g := PBblas.PB_dgemm(FALSE, TRUE, m_1, a2map, d2, dmap, ddist, w1map, w1 ,LAMBDA );
      RETURN w1_g;
    END;
		WeightGrad1 (DATASET(Layout_Part) w1, DATASET(Layout_Part) d2) := FUNCTION
			//w1_g_ := big_big_small(a2map, d2, dmap, ddist, w1map, m_1);
			w1_g_ := d2Xt(a2map, d2, dmap, ddist, w1map, m_1);
			w1_g  := PBblas.PB_daxpy(LAMBDA, w1, w1_g_);
      RETURN w1_g;
    END;
		
    //WeightGrad2 returns gradient for w2
    WeightGrad2_ (DATASET(Layout_Part) w2, DATASET(Layout_Part) d3, DATASET(Layout_Part) a2) := FUNCTION
      w2_g := PBblas.PB_dgemm(FALSE, TRUE, m_1, a3map, d3, a2map, a2, w2map, w2 ,LAMBDA );
      RETURN w2_g;
    END;
		
		WeightGrad2 (DATASET(Layout_Part) w2, DATASET(Layout_Part) d3, DATASET(Layout_Part) a2) := FUNCTION
			//w2_g_ := big_big_small(a3map, d3, a2map, a2, w2map, m_1);
			w2_g_ := d3a2t(a3map, d3, a2map, a2, w2map, w1_partitions, m_1);
			w2_g  := PBblas.PB_daxpy(LAMBDA, w2, w2_g_);
      RETURN w2_g;
    END;
    //BiasGrad1 calculates the bias gradients for b1
    BiasGrad1 (DATASET(Layout_Part) d2) := FUNCTION
			//b1_g := col_mean(a2map, d2, Ones_VecMap, Ones_Vecdist, b1vecmap, m_1);
			b1_g := colmean_bias_grad (a2map, d2, b1vecmap, w1_partitions + w2_partitions);
      RETURN b1_g;
    END;
		
		BiasGrad1_ (DATASET(Layout_Part) d2) := FUNCTION
      b1_g := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a2map, d2, Ones_VecMap, Ones_Vecdist, b1vecmap);
      RETURN b1_g;
    END;
    //BiasGrad2 calculates the bias gradients for b2
    BiasGrad2_ (DATASET(Layout_Part) d3) := FUNCTION
      b2_g := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a3map, d3, Ones_VecMap, Ones_Vecdist, b2vecmap);
      RETURN b2_g;
    END;
		
		BiasGrad2 (DATASET(Layout_Part) d3) := FUNCTION
      //b2_g := col_mean(a3map, d3, Ones_VecMap, Ones_Vecdist, b2vecmap, m_1);
			b2_g := colmean_bias_grad (a3map, d3, b2vecmap, w1_partitions + w2_partitions + b1_partitions);
      RETURN b2_g;
    END;
		
		
    emptyL := DATASET([], Layout_Part);
    //theta is the input weight and bias parameters for the sparseautoencoder
    //parameters are the parameters for the sparseautoencoder function
    //train_d is the train data to learn sparse autoencoder and calculate the gradient and the cost based on taht
    // train_l is empty in the sparse autoencoder (because it is an unsupervised learning)
    SparseParam_CostGradients4 :=  FUNCTION
      PBblas.Types.MUElement minuspart(Layout_Part l, UNSIGNED8 c ) := TRANSFORM
        SELF.partition_id := l.partition_id - c;
        SElF.no := 1;
        SELF := l;
      END;
      w1m := w1dist;
      w2m := w2dist;
      b1v := b1vecdist;
      b2v := b2vecdist;
      a2 := FF2 (w1m, b1v);
			a2_ := FF2_ (w1m, b1v);
      a3 := FF3 (w2m, b2v, a2);
			a3_ := FF3_ (w2m, b2v, a2_);
      d3 := DELTA3 (a3);
			d3_ := DELTA3_ (a3_);
      rohat_a2 := rohat(a2);
			rohat_a2_ := rohat_(a2_);
      d2 := DELTA2 (w2m, a2, d3,rohat_a2);
			d2_ := DELTA2_ (w2m, a2_, d3_,rohat_a2_);
      wg1 := WeightGrad1 (w1m, d2);
      wg2 := WeightGrad2 (w2m, d3, a2);
      bg1 := BiasGrad1 (d2);
      bg2 := BiasGrad2 (d3);
			
			
			wg1_ := WeightGrad1_ (w1m, d2_);
      wg2_ := WeightGrad2_ (w2m, d3_, a2_);
      bg1_ := BiasGrad1_ (d2_);
      bg2_ := BiasGrad2_ (d3_);
      //calculate cost
      //PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1, dmap, ddist, b1map, b1m, 1.0);
      // squared_error_cost= 0.5*sum(sum((x-a3).^2));
      // cost=(1/m)*squared_error_cost+(lambda/2)*(sum(W2(:).^2)+sum(W1(:).^2))+beta*sum(KL(sparsityParam,rhohat));
			
			simple := {REAL8 v};
			simple SE_tran (Layout_Part le, Layout_Part ri) := TRANSFORM
				SELF.v := sum_pow2(le.part_cols * le.part_rows, le.mat_part, ri.mat_part);
			END;
			d_a3_sum_part := JOIN (a3, ddist, LEFT.partition_id = RIGHT.partition_id,SE_tran(LEFT,RIGHT), LOCAL );
      //squared_error_cost := 0.5*PBblas.SumElements(PBblas.Apply2Elements(dmap, PBblas.PB_daxpy(-1.0, a3, ddist), pow2));
			squared_error_cost := SUM (d_a3_sum_part, d_a3_sum_part.v);
      cost_term1 := (1/m)*squared_error_cost;
			
			simple term2_3_tran (Layout_Part le) := TRANSFORM
				SELF.v := sum_sq(le.part_cols * le.part_rows, le.mat_part);
			END;
			w1_term3 := PROJECT (w1m, term2_3_tran (LEFT), LOCAL);
			w2_term2 := PROJECT (w2m, term2_3_tran (LEFT), LOCAL);

      // cost_term2 := (lambda/2)* PBblas.SumElements(PBblas.Apply2Elements(w2map, w2m, pow2));
      //cost_term3 := (lambda/2)* PBblas.SumElements(PBblas.Apply2Elements(w1map, w1m, pow2));
			cost_term2 := (lambda/2)* SUM (w2_term2, w2_term2.v); 
			cost_term3 := (lambda/2)* SUM (w1_term3, w1_term3.v);
			
			simple kl_tran (Layout_Part le) := TRANSFORM
				SELF.v := sum_kl(le.part_cols * le.part_rows, le.mat_part, sparsityParam);
			END;
			
      PBblas.Types.value_t klterm(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := sparsityParam * LN(sparsityParam/v) + (1-sparsityParam) * LN ((1-sparsityParam)/(1-v));
      //KL := PBblas.Apply2Elements (Hiddmap,rohat_a2,klterm);
		  KL := PROJECT (rohat_a2, kl_tran(LEFT), LOCAL);
      //cost_term4 := beta * PBblas.SumElements(KL);
			cost_term4 := beta * SUM (KL, KL.v);
      cost := cost_term1 + cost_term2 + cost_term3 +cost_term4;    
      costField := DATASET ([{1,1,cost}],ML.Types.NumericField);
      one_map := PBblas.Matrix_Map(1,1,1,1);
      Cost_part_no := PBblas.MU.TO(ML.DMat.Converted.FromNumericFieldDS(costField,one_map),2);
      //convert w and b gradients to a datasets of layoutparts where partition_id differentiate them
      //w1_grad has partition_id from 1 to w1_partitions
      //w2_grad had partition_ds from w1_partitions+1 to w1_partitions + w2_partitions
      //b1_grad has partition_ids from w1_partitions + w2_partitions+1 to w1_partitions + w2_partitions+b1_partitions
      //b2_grad has partition_ids from w1_partitions + w2_partitions+b1_partitions+1 to w1_partitions + w2_partitions+b1_partitions+b2_partitions
      PBblas.Types.MUElement addpart(Layout_Part l, UNSIGNED8 c ) := TRANSFORM
        SELF.partition_id := l.partition_id + c;
        SElF.no := 1;
        SELF := l;
      END;
      wg1_reshape_no := Pbblas.MU.TO(wg1,1);
      wg2_reshape_no := PROJECT (wg2, addpart(LEFT, w1_partitions ), LOCAL);
			wg2_reshape_no_ := PROJECT (wg2_, addpart(LEFT, w1_partitions ), LOCAL);
      bg1_reshape_no := PROJECT (bg1, addpart (LEFT, w1_partitions + w2_partitions),LOCAL);
      bg2_reshape_no := PROJECT (bg2, addpart (LEFT, w1_partitions + w2_partitions + b1_partitions),LOCAL);
     // theta_Part_no := wg1_reshape_no + wg2_reshape_no + bg1_reshape_no + bg2_reshape_no;
      theta_Part_no := PROJECT (wg1 + wg2 + bg1 + bg2 ,TRANSFORM (PBblas.Types.MUElement, SELF.no :=1 ; SELF := LEFT) , LOCAL);
			//RETURN PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1dist, dmap, ddist, b1map);
			//RETURN WX(w1map, w1dist, dmap, ddist, b1map, w1dist,  1);
			//RETURN WX_repmatb(w1map, w1dist, dmap, ddist, b1vecmap, b1vecdist, b1map, 1.0) ;
			//RETURN col_sum(dmap, ddist, Ones_VecMap, Ones_Vecdist, b2vecmap);
			w1x_b1 := WX_repmatb(w1map, w1dist, dmap, ddist, b1vecmap, b1vecdist, b1map, 1.0);
			thisis := big_big_small(b1map, w1x_b1, dmap, ddist, PBblas.Matrix_Map(num_hid, num_feat, num_hid, num_feat));
			thisone := WtX_repmatb(w2map,w2dist, dmap, ddist, b1vecmap, b1vecdist, b1map, 5);
			
			
			
			//mymy2 := DELTA2 (w2m, a2, d3,rohat_a2) + DELTA2_ (w2m, a2, d3,rohat_a2);
			//mymy2 := big_big_small(a2map, d2, dmap, ddist, w1map, 8);
			//mymy2 := col_mean(a3map, d3, Ones_VecMap, Ones_Vecdist, b2vecmap);
			mymy2 := wg1 + wg2  + bg1 + bg2;
myformat2 := RECORD
    mymy2.node_id;
    mymy2.partition_id;
    mymy2.block_row;
    mymy2.block_col;
    mymy2.first_row;
    mymy2.part_rows;
    mymy2.first_col;
    mymy2.part_cols;
		//mymy2.no;
		//mymy2.mat_part;
		INTEGER real_node := STD.System.Thorlib.Node();
END;
	thisR := TABLE(mymy2,myformat2,LOCAL); 
	theta_part_no_check :=  ASSERT(theta_Part_no, node_id=Thorlib.node() and node_id=(partition_id-1), 'sparse autoencoder gradient is not distributed correctly', FAIL);
	return_record := RECORD (Layout_Part)
		REAL8 cost_value;
	END;
	ToReturn := PROJECT (theta_part_no_check, TRANSFORM (return_record, SELF.cost_value := cost; SELF:=LEFT), LOCAL);
	//RETURN  theta_part_no_check + Cost_part_no;
	RETURN d3a2t_test(a3map, d3, a2map, a2, w2map, w1_partitions, 1.0);
	// a2_part :=   W1X_repmatb1(w1map, w1dist, dmap, ddist, b1vecmap, b1v_, b1map, 0.0, 0);
	// a3_part :=  W2a2_repmatb2(w2map, w2dist, b1map, a2_part, b2vecmap, b2v_, b2map, 0.0);
	// rohat_part := col_mean(a2map, a2_part, Ones_VecMap, Ones_Vecdist, Hiddmap);
	//RETURN W2td3_repmatsparsity(w2map, w2dist, dmap, ddist, b1vecmap, b1v_, b1map, 0.0, 0);
	//RETURN W2td3_repmatsparsity(w2map, w2dist, dmap, ddist, Hiddmap, b1v, a2map, BETA);
	//RETURN cost_term4;

//RETURN d2;
			//RETURN thisR;
      //RETURN theta_Part_no;
    END;//END SparseParam_CostGradients4  
    RETURN SparseParam_CostGradients4;
   END;//END SA_lbfgs_Compatible2_param_part_test


EXPORT SA_lbfgs_Compatible2_param_part ( DATASET(Layout_Part) theta, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Part) TrainData , DATASET(Layout_Part) TrainLabel) := FUNCTION
   

// This function repeats the bias vector of size r*1 , s times in a columnwise format, so the final results will be a r*s matrix
//D = [1,2,3], r=3, s=2 => output=[1,2,3,1,2,3]
  SET OF REAL8 repeatbias(PBblas.Types.dimension_t r, PBblas.Types.dimension_t s, PBblas.Types.matrix_t D) := BEGINC++

    #body
    __lenResult = r * s * sizeof(double);
    __isAllResult = false;
    double * result = new double[r*s];
    __result = (void*) result;
    double *cell = (double*) d;
    uint32_t cells =  r * s;
    uint32_t i;
    uint32_t pos;
    for (i=0; i<cells; i++) {
      pos = i % r;
      result[i] = cell[pos];
    }
  ENDC++;
	
//this function calculates d3=-(y-a3).*(a3.*(1-a3));
//N is the total number of elements in each matrix
//A is the a3 matrix in SET format
//Y is the y matrix in SET format
	SET OF REAL8 d3_cal(PBblas.Types.dimension_t N, PBblas.Types.matrix_t A, PBblas.Types.matrix_t Y) := BEGINC++

    #body
    __lenResult = n * sizeof(double);
    __isAllResult = false;
    double * result = new double[n];
    __result = (void*) result;
    double *cella = (double*) a;
		double *celly = (double*) y;
    uint32_t cells =  n;
    uint32_t i;
    for (i=0; i<cells; i++) {
      result[i] = (cella[i]-celly[i])*(cella[i]*(1-cella[i]));
    }
  ENDC++;
	SET OF REAL8 d2_cal(PBblas.Types.dimension_t N, PBblas.Types.matrix_t A, PBblas.Types.matrix_t Y) := BEGINC++

    #body
    __lenResult = n * sizeof(double);
    __isAllResult = false;
    double * result = new double[n];
    __result = (void*) result;
    double *cella = (double*) a;
		double *celly = (double*) y;
    uint32_t cells =  n;
    uint32_t i;
    for (i=0; i<cells; i++) {
      result[i] = (celly[i])*(cella[i]*(1-cella[i]));
    }
  ENDC++;
//result = M + bet * repmat (V, 1, r)
	SET OF REAL8 mat_vec_sum(PBblas.Types.dimension_t N, PBblas.Types.dimension_t r, PBblas.Types.matrix_t M, PBblas.Types.matrix_t V, PBblas.Types.value_t bet) := BEGINC++

    #body
    __lenResult = n * sizeof(double);
    __isAllResult = false;
    double * result = new double[n];
    __result = (void*) result;
    double *cellm = (double*) m;
		double *cellv = (double*) v;
    uint32_t cells =  n;
    uint32_t i;
		uint32_t pos;
    for (i=0; i<cells; i++) {
		  pos = i % r;
      result[i] = cellm[i] + (bet * cellv[pos]);
    }
  ENDC++;
	// //result = sigmoid (M + repmat (V, 1, r))
	SET OF REAL8 mat_vec_sum_sigmoid(PBblas.Types.dimension_t N, PBblas.Types.dimension_t r, PBblas.Types.matrix_t M, PBblas.Types.matrix_t V) := BEGINC++

    #body
    __lenResult = n * sizeof(double);
    __isAllResult = false;
    double * result = new double[n];
    __result = (void*) result;
    double *cellm = (double*) m;
		double *cellv = (double*) v;
    uint32_t cells =  n;
    uint32_t i;
		uint32_t pos;
    for (i=0; i<cells; i++) {
		  pos = i % r;
      result[i] = 1/(1 + exp(-1*(cellm[i] + cellv[pos])));
    }
  ENDC++;
	
	
	SET OF REAL8 sum_col (PBblas.Types.dimension_t r, PBblas.Types.dimension_t s, PBblas.Types.matrix_t D) := BEGINC++

    #body
    __lenResult = r * sizeof(double);
    __isAllResult = false;
    double * result = new double[r];
    __result = (void*) result;
    double *cell = (double*) d;
    uint32_t cells =  r * s;
    uint32_t i;
    uint32_t pos;
		for (i=0; i<r; i++) {
      result[i] = 0;
    }
    for (i=0; i<cells; i++) {
      pos = i % r;
      result[pos] = result[pos] + cell[i];
    }

  ENDC++;
	
		SET OF REAL8 sum_col_alpha (PBblas.Types.dimension_t r, PBblas.Types.dimension_t s, PBblas.Types.matrix_t D, REAL8 thisalpha) := BEGINC++

    #body
    __lenResult = r * sizeof(double);
    __isAllResult = false;
    double * result = new double[r];
    __result = (void*) result;
    double *cell = (double*) d;
    uint32_t cells =  r * s;
    uint32_t i;
    uint32_t pos;
		for (i=0; i<r; i++) {
      result[i] = 0;
    }
    for (i=0; i<cells; i++) {
      pos = i % r;
      result[pos] = result[pos] + cell[i];
    }
		for (i=0; i<r; i++) {
      result[i] = result[i] * thisalpha;
    }

  ENDC++;
	//0.5 * sum ((M-V).^2)
	REAL8 sum_pow2(PBblas.Types.dimension_t N, PBblas.Types.matrix_t M, PBblas.Types.matrix_t V) := BEGINC++

    #body
    double result = 0;
		double tmpp ;
    double *cellm = (double*) m;
		double *cellv = (double*) v;
    uint32_t i;
		for (i=0; i<n; i++) {
		  tmpp =(cellm[i] - cellv [i]);
      result = result + (tmpp*tmpp);
    }
		return(0.5*result);

  ENDC++;
	//sum(M.^2)
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
	// sum (kl(rho, M))
	REAL8 sum_kl(PBblas.Types.dimension_t N, PBblas.Types.matrix_t M, PBblas.Types.value_t rho) := BEGINC++

    #body
    double result = 0;
		double tmpp ;
    double *cellm = (double*) m;
    uint32_t i;
		for (i=0; i<n; i++) {
			result = result + (rho*log(rho/cellm[i])) + ((1-rho)*log((1-rho)/(1-cellm[i])));
    }
		return(result);

  ENDC++;
    ddist := TrainData;
    m := CostFunc_params(id=1)[1].value;
    num_feat := CostFunc_params(id=2)[1].value;
    num_hid := CostFunc_params(id=3)[1].value;
    part_rows := CostFunc_params(id=4)[1].value; // partition size for the features (number of rows)
    part_cols := CostFunc_params(id=5)[1].value; // partition size for the number of columns (samples) in the input data
    BETA := CostFunc_params(id=6)[1].value;
    sparsityParam := CostFunc_params(id=7)[1].value;
    LAMBDA := CostFunc_params(id=8)[1].value;
    
    m_1 := 1/m;
    sparsityParam_ := -1*sparsityParam;
    sparsityParam_1 := 1-sparsityParam;
    
    //Create map for block matrix d
    dmap := PBblas.Matrix_Map(num_feat,m,part_rows,part_cols);
    
    //Create block matrix Ytmp
    Ymap := dmap;
    Ydist := ddist;
    //Creat maps for block matrices for weights

    w1map := PBblas.Matrix_Map(num_hid, num_feat, num_hid, part_rows);
    w2map := PBblas.Matrix_Map(num_feat, num_hid, part_rows, num_hid);
     //each bias vector is converted to block format
    
    b1vecmap := PBblas.Matrix_Map(num_hid, 1, num_hid, 1);
    b2vecmap := PBblas.Matrix_Map(num_feat, 1, part_rows, 1);
    
    w1_partitions := w1map.partitions_used;
    w2_partitions := w2map.partitions_used;
    b1_partitions := b1vecmap.partitions_used;
    b2_partitions := b2vecmap.partitions_used;
    Layout_Part minuspart(Layout_Part l, UNSIGNED8 c ) := TRANSFORM
      SELF.partition_id := l.partition_id - c;
      SELF := l;
    END;
    //w1m := w1dist;
    w1dist := theta (partition_id <= w1_partitions);
    //w2m := w2dist;
    w2dist := theta (partition_id > w1_partitions AND partition_id <= w1_partitions + w2_partitions);
    //w2dist  := PROJECT (w2m_, minuspart(LEFT, w1_partitions ),LOCAL);
    //b1v := b1vecdist;
    b1vecdist := theta (partition_id > w1_partitions + w2_partitions AND partition_id <= w1_partitions + w2_partitions + b1_partitions);
    //b1vecdist  := PROJECT (b1v_, minuspart (LEFT, w1_partitions + w2_partitions),LOCAL);
    
    //b2v := b2vecdist;
    b2vecdist := theta (partition_id > w1_partitions + w2_partitions + b1_partitions AND partition_id <= w1_partitions + w2_partitions + b1_partitions + b2_partitions);
    //b2vecdist  := PROJECT (b2v_, minuspart (LEFT, w1_partitions + w2_partitions + b1_partitions),LOCAL);
    
     //functions used
    PBblas.Types.value_t sp_reci(PBblas.Types.value_t v,PBblas.Types.dimension_t r,PBblas.Types.dimension_t c) := sparsityParam_/v;
    PBblas.Types.value_t sp_1_reci(PBblas.Types.value_t v,PBblas.Types.dimension_t r,PBblas.Types.dimension_t c) := sparsityParam_1/(1-v);
    //sparsity_delta=((-sparsityParam./rhohat)+((1-sparsityParam)./(1.-rhohat)));
    PBblas.Types.value_t sp_delta(PBblas.Types.value_t v,PBblas.Types.dimension_t r,PBblas.Types.dimension_t c) := (sparsityParam_/v)+(sparsityParam_1/(1-v));
    PBblas.Types.value_t siggrad(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := v*(1.0-v);
    PBblas.Types.value_t sigmoid(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := 1/(1+exp(-1*v));
    PBblas.Types.value_t pow2(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := v*v;
    //maps used
    b1map := PBblas.Matrix_Map(num_hid, m, num_hid, part_cols);
    b2map := PBblas.Matrix_Map(num_feat, m, part_rows, part_cols);
    a2map := b1map;
    a3map := b2map;
    HL_nodes := num_hid;//number of nodes in the hidden layer
    Hiddmap := b1vecmap;
    //onevec for calculating rhohat	 
	 
	 Ones_VecMap := PBblas.Matrix_Map(m, 1, part_cols, 1);
    //New Vector Generator
    // Layout_Cell gen(UNSIGNED4 c, UNSIGNED4 NumRows) := TRANSFORM
      // SELF.x := ((c-1) % NumRows) + 1;
      // SELF.y := ((c-1) DIV NumRows) + 1;
      // SELF.v := 1;
    // END;
    //Create Ones Vector for the calculations in the step fucntion
    // Ones_Vec := DATASET(m, gen(COUNTER, m));
    Ones_Vecdist := TrainLabel;
Layout_Target := PBblas.Types.Layout_Target;




row_block_bigmat (UNSIGNED8 p, UNSIGNED8 nn) := FUNCTION // p is the partition number, nn is the number of rows
	RETURN ((p-1) % nn )+1;
END;

col_block_bigmat (UNSIGNED8 p, UNSIGNED8 nn) := FUNCTION // p is the partition number, nn is the number of rows
	RETURN ((p-1) DIV nn )+1;
END;

block_smallmat (UNSIGNED8 p, UNSIGNED8 offset) := FUNCTION 
	RETURN p-offset;
END;


//A_in := w1 is h*f where f is divided to partitions of size prow
//B_in := data : ddist
// bb_in := bias1
//returns the sigmoid of the result
W1X_repmatb1(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_bb, DATASET(Layout_Part) bb_in, PBblas.IMatrix_Map map_c, REAL8 beta=0.0, UNSIGNED8 A_offset) := FUNCTION
SET OF PBblas.Types.value_t empty_array := [];
// First check maps for compatability.  Normalize for transpose operations.
  a_matrix_rows := map_a.matrix_rows;
  a_matrix_cols := map_a.matrix_cols;
  a_row_blocks  := map_a.row_blocks;
  a_col_blocks  := map_a.col_blocks;
  b_matrix_rows := map_b.matrix_rows;
  b_matrix_cols := map_b.matrix_cols;
  b_row_blocks  := map_b.row_blocks;
  b_col_blocks  := map_b.col_blocks;
  c_matrix_rows := map_c.matrix_rows;
  c_matrix_cols := map_c.matrix_cols;
  c_row_blocks  := map_c.row_blocks;
  c_col_blocks  := map_c.col_blocks;
  A := ASSERT(A_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' + ' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(a_matrix_rows=c_matrix_rows AND a_row_blocks=c_row_blocks,
                    'A-C: ' + 'Arows: ' + a_matrix_rows +' Crows '+c_matrix_rows+' '+ 
                              'Ablocks: ' + a_row_blocks +' Cblocks '+c_row_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	B := ASSERT(B_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' +' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(b_matrix_cols=c_matrix_cols AND b_col_blocks=c_col_blocks,
                    'B-C: ' + 'Brows: ' + b_matrix_cols +' Ccols '+c_matrix_cols+' '+ 
                              'Bblocks: ' + b_row_blocks +' Cblocks '+c_col_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	
  //multiply
	B_row_part := map_b.row_blocks;
	C_row_part := map_c.row_blocks;
	Layout_Part mul2(Layout_Part b_part, Layout_Part a_part):=TRANSFORM
    //real_part_id := col_block_bigmat (b_part.partition_id, B_row_part);  //arbitrary choice
		REAL_part_id := b_part.block_col;
		part_id := real_part_id;
    part_a_cols := a_part.part_cols;
    part_a_rows := a_part.part_rows;
    part_b_rows := b_part.part_rows;
    part_c_rows := map_c.part_rows(part_id);
    part_c_cols := map_c.part_cols(part_id);
    part_c_first_row  := map_c.first_row(part_id);
    part_c_first_col  := map_c.first_col(part_id);
    k := part_a_cols;
    SELF.partition_id := part_id;
    SELF.node_id      := b_part.node_id;
    SELF.block_row    := a_part.block_row;
    SELF.block_col    := b_part.block_col;
    SELF.first_row    := map_c.first_row(part_id);
    SELF.part_rows    := part_c_rows;
    SELF.first_col    := part_c_first_col;
    SELF.part_cols    := part_c_cols;
    SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
                                    part_c_rows, part_c_cols, k,
                                    1.0, a_part.mat_part, b_part.mat_part,
                                    0.0, empty_array);
  END;
  //ab_prod_ := JOIN(B, A, row_block_bigmat (LEFT.partition_id, B_row_part) = RIGHT.partition_id , mul2(LEFT,RIGHT),ALL); // Each A (weight matrix) is copied in each B (X matrix) node
	ab_prod_ := JOIN(B, A, LEFT.block_row = RIGHT.block_col , mul2(LEFT,RIGHT),ALL); // Each A (weight matrix) is copied in each B (X matrix) node
	//ab_prod_ := JOIN(B, A, LEFT.block_row = RIGHT.block_col , mul2(LEFT,RIGHT),ALL); //this is not correct, there might be more than one block row in one node

	 // Sum terms
  Layout_Part sumTerms(Layout_Part cumm, Layout_Part term) := TRANSFORM
    N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
    SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term.mat_part, 1);
    SELF := cumm;
  END;
	sorted_ab_prod := SORT(ab_prod_, partition_id, LOCAL);
  ab_prod := ROLLUP(sorted_ab_prod, sumTerms(LEFT, RIGHT), partition_id, LOCAL);
//	 ab_prod := ROLLUP(sorted_ab_prod, LEFT.partition_id = RIGHT.partition_id, sumTerms(LEFT, RIGHT), LOCAL);

// add bias vector to each columns of X
// each bias vector (b) is copied (ALL JOIN) to each node of X. The bias vector is then normalized to repeat itself, number of columns of X time in that node and then the two are added
  Layout_Part addbias(Layout_Part cumm, Layout_Part term) := TRANSFORM
		cumm_part_cols := cumm.part_cols;// number of columns in this partition
    N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
		Elem := {PBblas.Types.value_t v};
		Elem_r := {PBblas.Types.value_t v, UNSIGNED r};
		elems := DATASET(term.mat_part, Elem);
		Elem_r rep (Elem l, UNSIGNED c) := TRANSFORM
			SELF.r := c;
			SELF := l;
		END;
		elems_rep := NORMALIZE(elems, cumm_part_cols, rep(LEFT, COUNTER));// each value in bias vector is repeated cumm_part_cols number of times (as the numebr of cumm columns in this node)
		elems_rep_sort := SORT(elems_rep, r, LOCAL);
		term_rep_set := SET (elems_rep_sort, v);
    SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term_rep_set, 1);
		SELF.partition_id := cumm.partition_id;
    SELF := cumm;
  END;
	
	Layout_Part addbias_(Layout_Part cumm, Layout_Part term) := TRANSFORM
		cumm_part_cols := cumm.part_cols;// number of columns in this partition
		term_part_rows := term.part_rows;//number of elements in this partition of the bias vector
    N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
		Elem := {PBblas.Types.value_t v};
		Elem_r := {PBblas.Types.value_t v, UNSIGNED r};
		elems := DATASET(term.mat_part, Elem);
		Elem_r rep (Elem l, UNSIGNED c) := TRANSFORM
			SELF.r := c;
			SELF := l;
		END;
		elems_rep := NORMALIZE(elems, cumm_part_cols, rep(LEFT, COUNTER));// each value in bias vector is repeated cumm_part_cols number of times (as the numebr of cumm columns in this node)
		elems_rep_sort := SORT(elems_rep, r, LOCAL);
		term_rep_set := repeatbias(term_part_rows, cumm_part_cols, term.mat_part);
    //SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term_rep_set, 1);
		SELF.mat_part := mat_vec_sum_sigmoid(N, term_part_rows, cumm.mat_part, term.mat_part);
		SELF.partition_id := cumm.partition_id;
    SELF := cumm;
  END;
	
	
	 ab_bb := JOIN(ab_prod, bb_in,TRUE , addbias_(LEFT,RIGHT),ALL);
	 
	 
	 	Layout_Part rep_b1(Layout_Part one_part, Layout_Part bb_part):=TRANSFORM
    real_part_id := one_part.partition_id;
		part_id := (real_part_id-1)*C_row_part + bb_part.block_row;
    part_a_cols := bb_part.part_cols;
    part_a_rows := bb_part.part_rows;
    part_b_rows := one_part.part_rows;
    part_c_rows := map_c.part_rows(part_id);
    part_c_cols := map_c.part_cols(part_id);
    part_c_first_row  := map_c.first_row(part_id);
    part_c_first_col  := map_c.first_col(part_id);
    k := part_a_cols;
    SELF.partition_id := part_id;
    SELF.node_id      := one_part.node_id;
    SELF.block_row    := bb_part.block_row;
    SELF.block_col    := one_part.block_col;
    SELF.first_row    := map_c.first_row(part_id);
    SELF.part_rows    := part_c_rows;
    SELF.first_col    := part_c_first_col;
    SELF.part_cols    := part_c_cols;
    SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, TRUE,
                                    part_c_rows, part_c_cols, k,
                                    1.0, bb_part.mat_part, one_part.mat_part,
                                    0.0, empty_array);
  END;

	bb_repeated := JOIN (Ones_Vecdist, bb_in, TRUE , rep_b1(LEFT,RIGHT),ALL);
		 //Ones_VecMap := PBblas.Matrix_Map(m, 1, Ones_Vecdist := TrainLabel;
	ab_bb_ := PBblas.PB_daxpy(1.0, ab_prod, bb_repeated);
	RETURN ab_bb;
END; // END W1X_repmatb1




W2td3_repmatsparsity(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_bb, DATASET(Layout_Part) bb_in, PBblas.IMatrix_Map map_c, REAL8 beta_in=0) := FUNCTION

SET OF PBblas.Types.value_t empty_array := [];
	// First check maps for compatability.  Normalize for transpose operations.
		a_matrix_rows := map_a.matrix_cols;
		a_matrix_cols := map_a.matrix_rows;
		a_row_blocks  := map_a.col_blocks;
		a_col_blocks  := map_a.row_blocks;
		b_matrix_rows := map_b.matrix_rows;
		b_matrix_cols := map_b.matrix_cols;
		b_row_blocks  := map_b.row_blocks;
		b_col_blocks  := map_b.col_blocks;
		c_matrix_rows := map_c.matrix_rows;
		c_matrix_cols := map_c.matrix_cols;
		c_row_blocks  := map_c.row_blocks;
		c_col_blocks  := map_c.col_blocks;
	A := ASSERT(A_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'YES' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'NO' + ' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(a_matrix_rows=c_matrix_rows AND a_row_blocks=c_row_blocks,
                    'A-C: ' + 'Arows: ' + a_matrix_rows +' Crows '+c_matrix_rows+' '+ 
                              'Ablocks: ' + a_row_blocks +' Cblocks '+c_row_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
  B := ASSERT(B_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'YES' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'NO' +' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(b_matrix_cols=c_matrix_cols AND b_col_blocks=c_col_blocks,
                    'B-C: ' + 'Brows: ' + b_matrix_cols +' Ccols '+c_matrix_cols+' '+ 
                              'Bblocks: ' + b_row_blocks +' Cblocks '+c_col_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	
  //multiply
	B_row_part := map_b.row_blocks;
	Layout_Part mul2(Layout_Part b_part, Layout_Part a_part):=TRANSFORM
    part_id := b_part.block_col;
    part_a_cols := a_part.part_cols;
    part_a_rows := a_part.part_rows;
    part_b_rows := b_part.part_rows;
    part_c_rows := map_c.part_rows(part_id);
    part_c_cols := map_c.part_cols(part_id);
    part_c_first_row  := map_c.first_row(part_id);
    part_c_first_col  := map_c.first_col(part_id);
    k := part_a_rows;
    SELF.partition_id := part_id;
    SELF.node_id      := b_part.node_id;
    SELF.block_row    := a_part.block_col;
    SELF.block_col    := b_part.block_col;
    SELF.first_row    := map_c.first_row(part_id);
    SELF.part_rows    := part_c_rows;
    SELF.first_col    := part_c_first_col;
    SELF.part_cols    := part_c_cols;
    SELF.mat_part     := PBblas.BLAS.dgemm(TRUE, FALSE,
                                    part_c_rows, part_c_cols, k,
                                    1.0, a_part.mat_part, b_part.mat_part,
                                    0.0, empty_array);

  END;
  //ab_prod_ := JOIN(B, A, row_block_bigmat (LEFT.partition_id, B_row_part) = RIGHT.partition_id , mul2(LEFT,RIGHT),ALL); // Each A (weight matrix) is copied in each B (X matrix) node
	//ab_prod_ := JOIN(B, A, row_block_bigmat (LEFT.partition_id, B_row_part) = (RIGHT.partition_id-A_offset) , mul2(LEFT,RIGHT),ALL); // Each A (weight matrix) is copied in each B (X matrix) node
	ab_prod_ := JOIN(B, A, LEFT.block_row = RIGHT.block_row, mul2(LEFT,RIGHT),ALL);

	 // Sum terms
  Layout_Part sumTerms(Layout_Part cumm, Layout_Part term) := TRANSFORM
    N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
    SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term.mat_part, 1);
    SELF := cumm;
  END;
	sorted_ab_prod := SORT(ab_prod_, partition_id, LOCAL);
  ab_prod := ROLLUP(sorted_ab_prod, sumTerms(LEFT, RIGHT), partition_id, LOCAL);

Layout_Part addbias(Layout_Part cumm, Layout_Part term) := TRANSFORM
		cumm_part_cols := cumm.part_cols;// number of columns in this partition
		term_part_rows := term.part_rows;//number of elements in this partition of the bias vector
    N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
		SELF.mat_part := mat_vec_sum(N, term_part_rows, cumm.mat_part, term.mat_part, beta_in);
		SELF.partition_id := cumm.partition_id;
    SELF := cumm;
  END;
	
	
	 ab_bb := JOIN(ab_prod, bb_in,TRUE , addbias(LEFT,RIGHT),ALL);
	RETURN ab_bb;
END; // END W2td3_repmatsparsity







//returns the sigmoid of the result
W2a2_repmatb2(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_bb, DATASET(Layout_Part) bb_in, PBblas.IMatrix_Map map_c, REAL8 beta=0.0) := FUNCTION
SET OF PBblas.Types.value_t empty_array := [];
// First check maps for compatability.  Normalize for transpose operations.
  a_matrix_rows := map_a.matrix_rows;
  a_matrix_cols := map_a.matrix_cols;
  a_row_blocks  := map_a.row_blocks;
  a_col_blocks  := map_a.col_blocks;
  b_matrix_rows := map_b.matrix_rows;
  b_matrix_cols := map_b.matrix_cols;
  b_row_blocks  := map_b.row_blocks;
  b_col_blocks  := map_b.col_blocks;
  c_matrix_rows := map_c.matrix_rows;
  c_matrix_cols := map_c.matrix_cols;
  c_row_blocks  := map_c.row_blocks;
  c_col_blocks  := map_c.col_blocks;
  A := ASSERT(A_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' + ' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(a_matrix_rows=c_matrix_rows AND a_row_blocks=c_row_blocks,
                    'A-C: ' + 'Arows: ' + a_matrix_rows +' Crows '+c_matrix_rows+' '+ 
                              'Ablocks: ' + a_row_blocks +' Cblocks '+c_row_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	B := ASSERT(B_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' +' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(b_matrix_cols=c_matrix_cols AND b_col_blocks=c_col_blocks,
                    'B-C: ' + 'Brows: ' + b_matrix_cols +' Ccols '+c_matrix_cols+' '+ 
                              'Bblocks: ' + b_row_blocks +' Cblocks '+c_col_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	
  //multiply
	B_row_part := map_b.row_blocks;
	C_row_part := map_c.row_blocks;
	Layout_Part mul2(Layout_Part b_part, Layout_Part a_part):=TRANSFORM
    real_part_id := b_part.partition_id;  //arbitrary choice
		part_id := (real_part_id-1)*C_row_part + a_part.block_row;
    part_a_cols := a_part.part_cols;
    part_a_rows := a_part.part_rows;
    part_b_rows := b_part.part_rows;
    part_c_rows := map_c.part_rows(part_id);
    part_c_cols := map_c.part_cols(part_id);
    part_c_first_row  := map_c.first_row(part_id);
    part_c_first_col  := map_c.first_col(part_id);
    k := part_a_cols;
    SELF.partition_id := part_id;
    SELF.node_id      := b_part.node_id;
    SELF.block_row    := a_part.block_row;
    SELF.block_col    := b_part.block_col;
    SELF.first_row    := map_c.first_row(part_id);
    SELF.part_rows    := part_c_rows;
    SELF.first_col    := part_c_first_col;
    SELF.part_cols    := part_c_cols;
    SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
                                    part_c_rows, part_c_cols, k,
                                    1.0, a_part.mat_part, b_part.mat_part,
                                    0.0, empty_array);
  END;
  //ab_prod := JOIN(B, A, TRUE , mul2(LEFT,RIGHT),ALL); // Each A (weight matrix) is copied in each B (X matrix) node
	ab_prod := JOIN(B, A, TRUE , mul2(LEFT,RIGHT), ALL); // Each A (weight matrix) is copied in each B (X matrix) node


// add bias vector to each columns of X
// each bias vector (b) is copied (ALL JOIN) to each node of X. The bias vector is then normalized to repeat it to the number of columns of X in that node and then the two are added
  Layout_Part addbias(Layout_Part cumm, Layout_Part term) := TRANSFORM
		cumm_part_cols := cumm.part_cols;// number of columns in this partition
    N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
		Elem := {PBblas.Types.value_t v};
		Elem_r := {PBblas.Types.value_t v, UNSIGNED r};
		elems := DATASET(term.mat_part, Elem);
		Elem_r rep (Elem l, UNSIGNED c) := TRANSFORM
			SELF.r := c;
			SELF := l;
		END;
		elems_rep := NORMALIZE(elems, cumm_part_cols, rep(LEFT, COUNTER));// each value in bias vector is repeated cumm_part_cols number of times (as the numebr of cumm columns in this node)
		elems_rep_sort := SORT(elems_rep, r, LOCAL);
		term_rep_set := SET (elems_rep_sort, v);
    SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term_rep_set, 1);
		SELF.partition_id := cumm.partition_id;
    SELF := cumm;
  END;
	
	Layout_Part addbias_(Layout_Part cumm, Layout_Part term) := TRANSFORM
		cumm_part_cols := cumm.part_cols;// number of columns in this partition
		term_part_rows := term.part_rows;
    N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
		Elem := {PBblas.Types.value_t v};
		Elem_r := {PBblas.Types.value_t v, UNSIGNED r};
		elems := DATASET(term.mat_part, Elem);
		Elem_r rep (Elem l, UNSIGNED c) := TRANSFORM
			SELF.r := c;
			SELF := l;
		END;
		elems_rep := NORMALIZE(elems, cumm_part_cols, rep(LEFT, COUNTER));// each value in bias vector is repeated cumm_part_cols number of times (as the numebr of cumm columns in this node)
		elems_rep_sort := SORT(elems_rep, r, LOCAL);
		term_rep_set := repeatbias(term_part_rows, cumm_part_cols, term.mat_part);
    //SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term_rep_set, 1);
		SELF.mat_part := mat_vec_sum_sigmoid(N, term_part_rows, cumm.mat_part, term.mat_part);
		SELF.partition_id := cumm.partition_id;
    SELF := cumm;
  END;
	
	 ab_bb := JOIN(ab_prod, bb_in,LEFT.block_row = RIGHT.block_row , addbias_(LEFT,RIGHT),LOOKUP);
	 
	 	Layout_Part rep_b2(Layout_Part one_part, Layout_Part bb_part):=TRANSFORM
    real_part_id := one_part.partition_id;
		part_id := (real_part_id-1)*C_row_part + bb_part.block_row;
    part_a_cols := bb_part.part_cols;
    part_a_rows := bb_part.part_rows;
    part_b_rows := one_part.part_rows;
    part_c_rows := map_c.part_rows(part_id);
    part_c_cols := map_c.part_cols(part_id);
    part_c_first_row  := map_c.first_row(part_id);
    part_c_first_col  := map_c.first_col(part_id);
    k := part_a_cols;
    SELF.partition_id := part_id;
    SELF.node_id      := one_part.node_id;
    SELF.block_row    := bb_part.block_row;
    SELF.block_col    := one_part.block_col;
    SELF.first_row    := map_c.first_row(part_id);
    SELF.part_rows    := part_c_rows;
    SELF.first_col    := part_c_first_col;
    SELF.part_cols    := part_c_cols;
    SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, TRUE,
                                    part_c_rows, part_c_cols, k,
                                    1.0, bb_part.mat_part, one_part.mat_part,
                                    0.0, empty_array);
  END;

	bb_repeated := JOIN (Ones_Vecdist, bb_in, TRUE , rep_b2(LEFT,RIGHT),ALL);
		 //Ones_VecMap := PBblas.Matrix_Map(m, 1, Ones_Vecdist := TrainLabel;
	ab_bb_ := PBblas.PB_daxpy(1.0, ab_prod, bb_repeated);
	//rslt := A;
  //RETURN PROJECT (B, TRANSFORM (layout_part, SELF.partition_id := row_block_bigmat (LEFT.partition_id, B_row_part), SELF:= LEFT));
	RETURN ab_bb;
END; // END W2a2_repmatb2

  //retunrs the sigmoid(WX+b)  
WX_repmatb_sig(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_bb, DATASET(Layout_Part) bb_in, PBblas.IMatrix_Map map_c, REAL8 beta=0.0) := FUNCTION
SET OF PBblas.Types.value_t empty_array := [];
// First check maps for compatability.  Normalize for transpose operations.
  a_matrix_rows := map_a.matrix_rows;
  a_matrix_cols := map_a.matrix_cols;
  a_row_blocks  := map_a.row_blocks;
  a_col_blocks  := map_a.col_blocks;
  b_matrix_rows := map_b.matrix_rows;
  b_matrix_cols := map_b.matrix_cols;
  b_row_blocks  := map_b.row_blocks;
  b_col_blocks  := map_b.col_blocks;
  c_matrix_rows := map_c.matrix_rows;
  c_matrix_cols := map_c.matrix_cols;
  c_row_blocks  := map_c.row_blocks;
  c_col_blocks  := map_c.col_blocks;
  A := ASSERT(A_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' + ' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(a_matrix_rows=c_matrix_rows AND a_row_blocks=c_row_blocks,
                    'A-C: ' + 'Arows: ' + a_matrix_rows +' Crows '+c_matrix_rows+' '+ 
                              'Ablocks: ' + a_row_blocks +' Cblocks '+c_row_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	B := ASSERT(B_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' +' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(b_matrix_cols=c_matrix_cols AND b_col_blocks=c_col_blocks,
                    'B-C: ' + 'Brows: ' + b_matrix_cols +' Ccols '+c_matrix_cols+' '+ 
                              'Bblocks: ' + b_row_blocks +' Cblocks '+c_col_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	
  //multiply
	
	Layout_Part mul2(Layout_Part b_part, Layout_Part a_part):=TRANSFORM
    part_id     := b_part.partition_id;    //arbitrary choice
    part_a_cols := a_part.part_cols;
    part_a_rows := a_part.part_rows;
    part_b_rows := b_part.part_rows;
    part_c_rows := map_c.part_rows(part_id);
    part_c_cols := map_c.part_cols(part_id);
    part_c_first_row  := map_c.first_row(part_id);
    part_c_first_col  := map_c.first_col(part_id);
    k := part_a_cols;
    SELF.partition_id := b_part.partition_id;
    SELF.node_id      := b_part.node_id;
    SELF.block_row    := b_part.block_row;
    SELF.block_col    := b_part.block_col;
    SELF.first_row    := map_c.first_row(part_id);
    SELF.part_rows    := part_c_rows;
    SELF.first_col    := part_c_first_col;
    SELF.part_cols    := part_c_cols;
    SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
                                    part_c_rows, part_c_cols, k,
                                    1.0, a_part.mat_part, b_part.mat_part,
                                    0.0, empty_array);
  END;
  ab_prod := JOIN(B, A,TRUE , mul2(LEFT,RIGHT),ALL); // Each A (weight matrix) is copied in each B (X matrix) node

// add bias vector to each columns of X
// each bias vector (b) is copied (ALL JOIN) to each node of X. The bias vector is then normalized to repeat it to the number of columns of X in that node and then the two are added
  Layout_Part addbias(Layout_Part cumm, Layout_Part term) := TRANSFORM
		cumm_part_cols := cumm.part_cols;
    N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
		Elem := {PBblas.Types.value_t v};
		Elem_r := {PBblas.Types.value_t v, UNSIGNED r};
		elems := DATASET(term.mat_part, Elem);
		Elem_r rep (Elem l, UNSIGNED c) := TRANSFORM
			SELF.r := c;
			SELF := l;
		END;
		elems_rep := NORMALIZE(elems, cumm_part_cols, rep(LEFT, COUNTER));// each value in bias vector is repeated cumm_part_cols number of times (as the numebr of cumm columns in this node)
		elems_rep_sort := SORT(elems_rep, r);
		term_rep_set := SET (elems_rep_sort, v);
    SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term_rep_set, 1);
		SELF.partition_id := cumm.partition_id;
    SELF := cumm;
  END;
	
	 ab_bb := JOIN(ab_prod, bb_in,TRUE , addbias(LEFT,RIGHT),ALL);
	//rslt := A;
  RETURN ab_bb;
END; // END WX_repmatb_sig
		
		

		
		
		
		

WtX_repmatb(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_bb, DATASET(Layout_Part) bb_in, PBblas.IMatrix_Map map_c, REAL8 beta=0.0) := FUNCTION
	SET OF PBblas.Types.value_t empty_array := [];
	// First check maps for compatability.  Normalize for transpose operations.
		a_matrix_rows := map_a.matrix_cols;
		a_matrix_cols := map_a.matrix_rows;
		a_row_blocks  := map_a.col_blocks;
		a_col_blocks  := map_a.row_blocks;
		b_matrix_rows := map_b.matrix_rows;
		b_matrix_cols := map_b.matrix_cols;
		b_row_blocks  := map_b.row_blocks;
		b_col_blocks  := map_b.col_blocks;
		c_matrix_rows := map_c.matrix_rows;
		c_matrix_cols := map_c.matrix_cols;
		c_row_blocks  := map_c.row_blocks;
		c_col_blocks  := map_c.col_blocks;
		A := ASSERT(A_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'YES' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'NO' + ' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(a_matrix_rows=c_matrix_rows AND a_row_blocks=c_row_blocks,
                    'A-C: ' + 'Arows: ' + a_matrix_rows +' Crows '+c_matrix_rows+' '+ 
                              'Ablocks: ' + a_row_blocks +' Cblocks '+c_row_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
  B := ASSERT(B_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'YES' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'NO' +' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(b_matrix_cols=c_matrix_cols AND b_col_blocks=c_col_blocks,
                    'B-C: ' + 'Brows: ' + b_matrix_cols +' Ccols '+c_matrix_cols+' '+ 
                              'Bblocks: ' + b_row_blocks +' Cblocks '+c_col_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
		
		//multiply
		
		Layout_Part mul2(Layout_Part b_part, Layout_Part a_part):=TRANSFORM
			part_id     := b_part.partition_id;    //arbitrary choice
			part_a_cols := a_part.part_cols;
			part_a_rows := a_part.part_rows;
			part_b_rows := b_part.part_rows;
			part_c_rows := map_c.part_rows(part_id);
			part_c_cols := map_c.part_cols(part_id);
			part_c_first_row  := map_c.first_row(part_id);
			part_c_first_col  := map_c.first_col(part_id);
			k := part_a_rows;
			SELF.partition_id := b_part.partition_id;
			SELF.node_id      := b_part.node_id;
			SELF.block_row    := b_part.block_row;
			SELF.block_col    := b_part.block_col;
			SELF.first_row    := map_c.first_row(part_id);
			SELF.part_rows    := part_c_rows;
			SELF.first_col    := part_c_first_col;
			SELF.part_cols    := part_c_cols;
			SELF.mat_part     := PBblas.BLAS.dgemm(TRUE, FALSE,
																			part_c_rows, part_c_cols, k,
																			1.0, a_part.mat_part, b_part.mat_part,
																			0.0, empty_array);
		END;
		ab_prod := JOIN(B, A,TRUE , mul2(LEFT,RIGHT),ALL); // Each A (weight matrix) is copied in each B (X matrix) node



// Apply beta
  Layout_Part applyBeta(Layout_Part part) := TRANSFORM
    SELF.mat_part := PBblas.BLAS.dscal(map_bb.matrix_rows*map_bb.matrix_cols,
                                beta, part.mat_part, 1);
    SELF:= part;
  END;
  bb_beta := PROJECT(bb_in, applyBeta(LEFT), LOCAL);
	// add the vector to each columns of X
	// each vector is copied (ALL JOIN) to each node of X. The vector is then normalized to repeat it to the number of columns of X in that node and then the two are added
		Layout_Part addvec(Layout_Part cumm, Layout_Part term) := TRANSFORM
			cumm_part_cols := cumm.part_cols;
			N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
			Elem := {PBblas.Types.value_t v};
			Elem_r := {PBblas.Types.value_t v, UNSIGNED r};
			elems := DATASET(term.mat_part, Elem);
			Elem_r rep (Elem l, UNSIGNED c) := TRANSFORM
				SELF.r := c;
				SELF := l;
			END;
			elems_rep := NORMALIZE(elems, cumm_part_cols, rep(LEFT, COUNTER));// each value in bias vector is repeated cumm_part_cols number of times (as the numebr of cumm columns in this node)
			elems_rep_sort := SORT(elems_rep, r);
			term_rep_set := SET (elems_rep_sort, v);
			SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term_rep_set, 1);
			SELF.partition_id := cumm.partition_id;
			SELF := cumm;
		END;
		
		 ab_bb := JOIN(ab_prod, bb_beta,TRUE , addvec(LEFT,RIGHT),ALL);

		//rslt := A;
		RETURN ab_bb;
END; // END WtX_repmatb
		
		
		
		// the input is a matrix in PBblas format where only columns are partitions
		//B_in : ones vector which is partitioned among nodes
		// map_c is the result's map
		col_sum(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_c) := FUNCTION
			SET OF PBblas.Types.value_t empty_array := [];
				Layout_Part mul(Layout_Part a_part, Layout_Part b_part):=TRANSFORM
					part_id     := 1;    //arbitrary choice
					part_a_cols := a_part.part_cols;
					part_a_rows := a_part.part_rows;
					part_b_rows := b_part.part_rows;
					part_c_rows := map_c.part_rows(part_id);
					part_c_cols := map_c.part_cols(part_id);
					part_c_first_row  := map_c.first_row(part_id);
					part_c_first_col  := map_c.first_col(part_id);
					k := part_a_cols;
					SELF.partition_id := 1;
					SELF.node_id      := a_part.node_id;
					SELF.block_row    := 1;
					SELF.block_col    := 1;
					SELF.first_row    := map_c.first_row(part_id);
					SELF.part_rows    := part_c_rows;
					SELF.first_col    := part_c_first_col;
					SELF.part_cols    := part_c_cols;
					SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
																					part_c_rows, part_c_cols, k,
																					1.0, a_part.mat_part, b_part.mat_part,
																					0.0, empty_array);
			END;
			col_sum_part := JOIN (A_in, B_in, LEFT.partition_id = RIGHT.partition_id, mul(LEFT, RIGHT), LOCAL );
			
			Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := le.part_rows ;
				SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1);
				SELF := le;
			END;
			//rslt := ROLLUP(col_sum_part, addup(LEFT, RIGHT), partition_id); // overload becasue of grouping
			rslt := ROLLUP(col_sum_part,TRUE, addup(LEFT, RIGHT)); // no groupijng, reduces overload
			//distribute to node one
			RETURN rslt; 
		END;//END Col_Sum
		
		
		col_mean(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_c, REAL8 mean_coef) := FUNCTION
		
			Num := map_a.matrix_cols;
			Num_1 := 1/Num;
			SET OF PBblas.Types.value_t empty_array := [];
				Layout_Part mul(Layout_Part a_part, Layout_Part b_part):=TRANSFORM
					part_id     := 1;    //arbitrary choice
					part_a_cols := a_part.part_cols;
					part_a_rows := a_part.part_rows;
					part_b_rows := b_part.part_rows;
					part_c_rows := map_c.part_rows(part_id);
					part_c_cols := map_c.part_cols(part_id);
					part_c_first_row  := map_c.first_row(part_id);
					part_c_first_col  := map_c.first_col(part_id);
					k := part_a_cols;
					SELF.partition_id := 1;
					SELF.node_id      := 0;
					SELF.block_row    := 1;
					SELF.block_col    := 1;
					SELF.first_row    := map_c.first_row(part_id);
					SELF.part_rows    := part_c_rows;
					SELF.first_col    := part_c_first_col;
					SELF.part_cols    := part_c_cols;
					SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
																					part_c_rows, part_c_cols, k,
																					Num_1, a_part.mat_part, b_part.mat_part,
																					0.0, empty_array);
			END;
			col_sum_part_ := JOIN (A_in, B_in, LEFT.partition_id = RIGHT.partition_id, mul(LEFT, RIGHT), LOCAL );
			
			
			Layout_Part col_sum_tran(Layout_Part the_part):= TRANSFORM
				SELF.mat_part := sum_col_alpha (the_part.part_rows, the_part.part_cols, the_part.mat_part, Num_1);
				SELF := the_part;
			END;
			col_sum_part := PROJECT (A_in, col_sum_tran(LEFT),LOCAL);
			Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := le.part_rows ;
				SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1);
				SELF := le;
			END;
			rslt := ROLLUP(col_sum_part,TRUE, addup(LEFT, RIGHT)); // this form of rollup avoid grouping, ROLLUP(col_sum_part,addup(LEFT, RIGHT, partiion_id)) cause all the records with the same partiion_id
			//get grouped to one node which cause a lot of overload. The current rollup form avoidds grouping and improves performance
			final_rslt := DISTRIBUTE (rslt, node_id); 
			//distribute to node one
			RETURN rslt;
		END;//END colmean
		//sum (A_in,2)
		colmean_bias_grad (PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, UNSIGNED b_offset) := FUNCTION
		
			Num := map_a.matrix_cols;
			Num_1 := 1/Num;
			
			Layout_Part col_sum_tran(Layout_Part the_part):= TRANSFORM
				SELF.mat_part := sum_col_alpha (the_part.part_rows, the_part.part_cols, the_part.mat_part, Num_1);
				part_id := the_part.block_row + b_offset;
				real_part :=  the_part.block_row;
				SELF. partition_id := part_id;
				SELF.node_id := part_id-1;
				SELF.block_row    := real_part;
				SELF.block_col    := 1;
				SELF.first_row    := map_b.first_row(real_part);
				SELF.part_rows    := map_b.part_rows(real_part);
				SELF.first_col    := map_b.first_col(real_part);
				SELF.part_cols    := map_b.part_cols(real_part);
				SELF := the_part;
			END;
			col_sum_part := PROJECT (A_in, col_sum_tran(LEFT),LOCAL);
			col_sum_part_dist := DISTRIBUTE (col_sum_part, node_id);
			Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := le.part_rows ;
				SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1);
				SELF := le;
			END;
			rslt := ROLLUP(col_sum_part_dist,LEFT.partition_id = RIGHT.partition_id, addup(LEFT, RIGHT), LOCAL); // this form of rollup avoid grouping, ROLLUP(col_sum_part,addup(LEFT, RIGHT, partiion_id)) cause all the records with the same partiion_id
			//get grouped to one node which cause a lot of overload. The current rollup form avoidds grouping and improves performance

			RETURN rslt;
		END;//END colmean_bias_grad
		
		
		// This function gets two big matrices which are distributed over all nodes and generate a final relatively smaller matrix which is on one node
		// this is used for weight gradient calculation where for example a h*m matrix is multiplied by a m*f matrix. PBblas will distribute all partitions in the first and second matrix to only one node which final matrix is in
		// this causes overhead, to avoid that we multiply each col partition of first matrix with a row partition of the second matrix in each node, the final generated matrices are added up to generate the final matrix
		// this way, we don't change the distribution of the first and second matrices
		big_big_small(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_c, PBblas.Types.value_t alph=1.0) := FUNCTION
			SET OF PBblas.Types.value_t empty_array := [];
				Layout_Part mul(Layout_Part a_part, Layout_Part b_part):=TRANSFORM
					part_id     := 1;    //arbitrary choice
					part_a_cols := a_part.part_cols;
					part_a_rows := a_part.part_rows;
					part_b_rows := b_part.part_rows;
					part_b_cols := b_part.part_cols;
					k := part_a_cols;
					SELF.partition_id := part_id;
					SELF.node_id      := 0;
					SELF.block_row    := 1;
					SELF.block_col    := 1;
					SELF.first_row    := map_c.first_row(part_id);
					SELF.part_rows    := map_c.part_rows(part_id);
					SELF.first_col    := map_c.first_col(part_id);
					SELF.part_cols    := map_c.part_cols(part_id);
					SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, TRUE,
																					part_a_rows, part_b_rows, k,
																					alph, a_part.mat_part, b_part.mat_part,
																					0.0, empty_array);
			END;
			mul_part := JOIN (A_in, B_in, LEFT.partition_id = RIGHT.partition_id, mul(LEFT, RIGHT), LOCAL );
			
			Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := le.part_rows * le.part_cols ;
				SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1);
				SELF := le;
			END;
			
			Layout_Part addup_it(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := ri.part_rows * ri.part_cols ;
				SELF.mat_part := IF (le.partition_id=0, ri.mat_part, PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1));
				SELF := ri;
			END;
			//rslt := ROLLUP(mul_part, addup(LEFT, RIGHT), partition_id); // since the results of rohat is used in a ALL join, no need to distribute this to node one to be consistent with PBblas
			//rslt := ITERATE(mul_part, addup_it(LEFT, RIGHT));// using rollup cause the graph to Group all the records which are distributed between all node to only one record and then do the operation, It takes a long time to GROUP all thoese partitions in one node and we avoid it by using ITERATE instead of ROLLUP

      rslt := ROLLUP(mul_part, TRUE, addup(LEFT, RIGHT));
			final_rslt := DISTRIBUTE (rslt, node_id); 

// a_part := A_in[1];
// b_part := B_in[1];
		 // RETURN PBblas.BLAS.dgemm(FALSE, TRUE,
																					// a_part.part_rows, b_part.part_rows, a_part.part_cols,
																					// 1.0, a_part.mat_part, b_part.mat_part,
																					// 0.0, empty_array);
																					
			RETURN final_rslt;
		END;// END big_big_small
		//calculates coef * (d2*x')
		d2Xt(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_c, REAL8 coef) := FUNCTION
			SET OF PBblas.Types.value_t empty_array := [];
				Layout_Part mul(Layout_Part a_part, Layout_Part b_part):=TRANSFORM
					part_id     := b_part.block_row;
					part_a_cols := a_part.part_cols;
					part_a_rows := a_part.part_rows;
					part_b_rows := b_part.part_rows;
					part_b_cols := b_part.part_cols;
					k := part_a_cols;
					SELF.partition_id := part_id;
					SELF.node_id      := part_id-1;
					SELF.block_row    := a_part.block_row;
					SELF.block_col    := b_part.block_row;
					SELF.first_row    := map_c.first_row(part_id);
					SELF.part_rows    := map_c.part_rows(part_id);
					SELF.first_col    := map_c.first_col(part_id);
					SELF.part_cols    := map_c.part_cols(part_id);
					SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, TRUE,
																					part_a_rows, part_b_rows, k,
																					coef, a_part.mat_part, b_part.mat_part,
																					0.0, empty_array);
			END;
			B_row_part := map_b.row_blocks;
			mul_part := JOIN (A_in, B_in, LEFT.block_col = RIGHT.block_col, mul(LEFT, RIGHT), LOCAL );
			
			Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := le.part_rows * le.part_cols ;
				SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1);
				SELF := le;
			END;
			mul_part_dist := DISTRIBUTE (mul_part, node_id);
			mul_part_dist_sorted := SORT (mul_part_dist, partition_id, LOCAL);
      rslt := ROLLUP(mul_part_dist_sorted, LEFT.partition_id = RIGHT.partition_id, addup(LEFT, RIGHT), LOCAL);
			// mul_part_sort := SORT (mul_part, partition_id);
			//rslt := ROLLUP(mul_part_sort, addup(LEFT, RIGHT), partition_id);
																					
			RETURN rslt;
		END;// END d2Xt
		
		d3a2t(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_c, UNSIGNED B_offset, REAL8 coef) := FUNCTION
			SET OF PBblas.Types.value_t empty_array := [];
				Layout_Part mul(Layout_Part a_part, Layout_Part b_part):=TRANSFORM
					
					part_id := a_part.block_row;
					new_part_id     := part_id + B_offset;
					part_a_cols := a_part.part_cols;
					part_a_rows := a_part.part_rows;
					part_b_rows := b_part.part_rows;
					part_b_cols := b_part.part_cols;
					k := part_a_cols;
					SELF.partition_id := new_part_id;
					SELF.node_id      := new_part_id-1;
					SELF.block_row    := a_part.block_row;
					SELF.block_col    := b_part.block_row;
					SELF.first_row    := map_c.first_row(part_id);
					SELF.part_rows    := map_c.part_rows(part_id);
					SELF.first_col    := map_c.first_col(part_id);
					SELF.part_cols    := map_c.part_cols(part_id);
					SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, TRUE,
																					part_a_rows, part_b_rows, k,
																					coef, a_part.mat_part, b_part.mat_part,
																					0.0, empty_array);
			END;
			A_row_part := map_a.row_blocks;
			mul_part := JOIN (A_in, B_in, LEFT.block_col = RIGHT.block_col, mul(LEFT, RIGHT), LOCAL );
			
			Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := le.part_rows * le.part_cols ;
				SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1);
				SELF := le;
			END;
			

			mul_part_dist := DISTRIBUTE (mul_part, node_id);
			mul_part_dist_sorted := SORT (mul_part_dist, partition_id, LOCAL);
      rslt := ROLLUP(mul_part_dist_sorted, LEFT.partition_id = RIGHT.partition_id, addup(LEFT, RIGHT), LOCAL);
																					
			RETURN rslt;
		END;// END d3a2t
		
		
		
		
		//extract sparse autoencoder parameters
   

    //FF2 returns a2
    FF2_(DATASET(Layout_Part) w1, DATASET(Layout_Part) b1v):= FUNCTION
      //b1m = repmat(b1v,1,m)
      b1m := PBblas.PB_dgemm(FALSE, TRUE, 1.0,b1vecmap, b1v, Ones_VecMap, Ones_Vecdist, b1map);
      //z2 = w1*X+b1;
      //z2 := PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1, dmap, ddist, b1map, b1m, 1.0); // gives MP closed error
      z2_ := PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1, dmap, ddist, b1map);
      z2 := PBblas.PB_daxpy(1.0, z2_, b1m);
      //a2 = sigmoid (z2);
      a2 := PBblas.Apply2Elements(b1map, z2, sigmoid);
      RETURN a2;
    END;//END FF2_
		
		
		
		//returns a2
		 // FF2(DATASET(Layout_Part) w1, DATASET(Layout_Part) b1v):= FUNCTION
      // z2=w1*x+repmat(b1,1,m)
			// z2 := WX_repmatb(w1map, w1, dmap, ddist, b1vecmap, b1v, b1map, 0.0);
      // a2 = sigmoid (z2);
      // a2 := PBblas.Apply2Elements(b1map, z2, sigmoid);
      // RETURN a2;
     // END;//END FF2
		 
		 
		
		FF2(DATASET(Layout_Part) w1, DATASET(Layout_Part) b1v):= FUNCTION
      //z2=w1*x+repmat(b1,1,m)
			//z2 := WX_repmatb(w1map, w1, dmap, ddist, b1vecmap, b1v, b1map, 0.0);
			z2 := W1X_repmatb1(w1map, w1, dmap, ddist, b1vecmap, b1v, b1map, 0.0, 0);
      //a2 = sigmoid (z2);
      a2 := PBblas.Apply2Elements(b1map, z2, sigmoid);
      RETURN z2;
     END;//END FF2
		 
    //FF3 returns a3
    FF3_(DATASET(Layout_Part) w2,DATASET(Layout_Part) b2v, DATASET(Layout_Part) a2 ):= FUNCTION
      //b2m = repmat(b2v,1,m)
      b2m := PBblas.PB_dgemm(FALSE, TRUE, 1.0,b2vecmap, b2v, Ones_VecMap, Ones_Vecdist, b2map);
      //z3 = w2*a2+b2;
      //z3 := PBblas.PB_dgemm(FALSE, FALSE,1.0,w2map, w2, a2map, a2, b2map,b2m, 1.0); //gives MP closed error
      z3_ := PBblas.PB_dgemm(FALSE, FALSE,1.0,w2map, w2, a2map, a2, b2map);
      z3 := PBblas.PB_daxpy(1.0, z3_, b2m);
      //a3 = sigmoid (z3)
      a3 := PBblas.Apply2Elements(b2map, z3, sigmoid);
      RETURN a3;
    END;//END FF3_
		
		 //FF3 returns a3
    // FF3(DATASET(Layout_Part) w2,DATASET(Layout_Part) b2v, DATASET(Layout_Part) a2 ):= FUNCTION
		  // z3 = w2*a2 + repmat(b2,1,m)
			// z3 := WX_repmatb(w2map, w2, a2map, a2, b2vecmap, b2v, b2map, 0.0);
      // a3 = sigmoid (z3)
      // a3 := PBblas.Apply2Elements(b2map, z3, sigmoid);
      // RETURN a3;
    // END;//END FF3
		
	 FF3(DATASET(Layout_Part) w2,DATASET(Layout_Part) b2v, DATASET(Layout_Part) a2 ):= FUNCTION
		  //z3 = w2*a2 + repmat(b2,1,m)
			//z3 := WX_repmatb(w2map, w2, a2map, a2, b2vecmap, b2v, b2map, 0.0);
			z3 := W2a2_repmatb2(w2map, w2, a2map, a2, b2vecmap, b2v, b2map, 0.0);
      //a3 = sigmoid (z3)
      a3 := PBblas.Apply2Elements(b2map, z3, sigmoid);
      RETURN z3;
    END;//END FF3
		

    //DELTA3 returns d3
    DELTA3 (DATASET(Layout_Part) a3 ) := FUNCTION
      //calculate delta for the last layer (3rd layer)
      //y=X;
      //d3=-(y-a3).*(a3.*(1-a3));
      // siggrad_a3 := PBblas.Apply2Elements(a3map, a3, siggrad);
      // a3_y := PBblas.PB_daxpy(-1, ddist, a3);
      // d3 := PBblas.HadamardProduct(a3map, a3_y, siggrad_a3);
			Layout_part d3_tran (Layout_part a3_part, Layout_part y_part) := TRANSFORM
				SELF.mat_part := d3_cal(a3_part.part_rows * a3_part.part_cols, a3_part.mat_part, y_part.mat_part);
				SELF := a3_part;
			END;
			d3 := JOIN (a3, ddist,LEFT.partition_id = RIGHT.partition_id, d3_tran(LEFT, RIGHT), LOCAL);


      RETURN d3 ;
    END;//END DELTA3
		
		DELTA3_ (DATASET(Layout_Part) a3 ) := FUNCTION
      //calculate delta for the last layer (3rd layer)
      //y=X;
      //d3=-(y-a3).*(a3.*(1-a3));
      siggrad_a3 := PBblas.Apply2Elements(a3map, a3, siggrad);
      a3_y := PBblas.PB_daxpy(-1, ddist, a3);
      d3 := PBblas.HadamardProduct(a3map, a3_y, siggrad_a3);
      RETURN d3 ;
    END;//END DELTA3_
    //DELTA2 retunrs d2
    rohat_ (DATASET(Layout_Part) a2) := FUNCTION
      rh := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a2map, a2, Ones_VecMap, Ones_Vecdist, Hiddmap);
      RETURN rh;
    END;
		rohat (DATASET(Layout_Part) a2) := FUNCTION
			//rh := col_mean(a2map, a2, Ones_VecMap, Ones_Vecdist, Hiddmap, 1.0);
			rh := colmean_bias_grad (a2map, a2, Hiddmap, 0);
      RETURN rh;
    END;
    DELTA2_ (DATASET(Layout_Part) w2, DATASET(Layout_Part) a2, DATASET(Layout_Part) d3, DATASET(Layout_Part) rhat) := FUNCTION
      //calculate delta for 2nd layer
      //rhohat=mean(a2,2);
      //sparsity_delta=((-sparsityParam./rhohat)+((1-sparsityParam)./(1.-rhohat)));
      //d2=((W2'*d3)+beta*repmat(sparsity_delta,1,m)) .* (a2.*(1-a2));
      //rhohat := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a2map, a2, Ones_VecMap, Ones_Vecdist, Hiddmap);
      rhohat := rhat;
      sparsity_delta := PBblas.Apply2Elements(Hiddmap, rhohat, sp_delta);
      siggrad_a2 := PBblas.Apply2Elements(a2map, a2, siggrad);
      repmat_sparsity_delta := PBblas.PB_dgemm(FALSE, TRUE, 1.0,  Hiddmap, sparsity_delta, Ones_VecMap, Ones_Vecdist, a2map);
      //d2_firstterm = (W2'*d3)+beta*repmat(sparsity_delta,1,m);
      //d2_firstterm := PBblas.PB_dgemm(TRUE, FALSE, 1.0, w2map, w2, a3map, d3, a2map, repmat_sparsity_delta, BETA); // MP close error
      d2_firstterm_ := PBblas.PB_dgemm(TRUE, FALSE, 1.0, w2map, w2, a3map, d3, a2map);
      d2_firstterm := PBblas.PB_daxpy(BETA, repmat_sparsity_delta, d2_firstterm_);
      d2 := PBblas.HadamardProduct(a2map, d2_firstterm, siggrad_a2);
      RETURN d2 ;
    END;
		
		
		DELTA2 (DATASET(Layout_Part) w2, DATASET(Layout_Part) a2, DATASET(Layout_Part) d3, DATASET(Layout_Part) rhat) := FUNCTION
      //calculate delta for 2nd layer
      //rhohat=mean(a2,2);
      //sparsity_delta=((-sparsityParam./rhohat)+((1-sparsityParam)./(1.-rhohat)));
      //d2=((W2'*d3)+beta*repmat(sparsity_delta,1,m)) .* (a2.*(1-a2));
      //rhohat := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a2map, a2, Ones_VecMap, Ones_Vecdist, Hiddmap);
      rhohat := rhat;
      sparsity_delta := PBblas.Apply2Elements(Hiddmap, rhohat, sp_delta);
      // siggrad_a2 := PBblas.Apply2Elements(a2map, a2, siggrad);
      //repmat_sparsity_delta := PBblas.PB_dgemm(FALSE, TRUE, 1.0,  Hiddmap, sparsity_delta, Ones_VecMap, Ones_Vecdist, a2map);
      //d2_firstterm = (W2'*d3)+beta*repmat(sparsity_delta,1,m);
			//d2_firstterm := WtX_repmatb(w2map, w2, a3map, d3, Hiddmap, sparsity_delta, a2map, BETA);
			d2_firstterm := W2td3_repmatsparsity(w2map, w2, a3map, d3, Hiddmap, sparsity_delta, a2map, BETA);
      //d2 := PBblas.HadamardProduct(a2map, d2_firstterm, siggrad_a2);
			Layout_part d2_tran (Layout_part a2_part, Layout_part d2_part) := TRANSFORM
				SELF.mat_part := d2_cal(a2_part.part_rows * a2_part.part_cols, a2_part.mat_part, d2_part.mat_part);
				SELF := a2_part;
			END;
			d2 := JOIN (a2, d2_firstterm, LEFT.partition_id = RIGHT.partition_id, d2_tran(LEFT, RIGHT), LOCAL);
      RETURN d2;
    END;
    //WeightGrad1 returns gradient for w1
    WeightGrad1_ (DATASET(Layout_Part) w1, DATASET(Layout_Part) d2) := FUNCTION
      w1_g := PBblas.PB_dgemm(FALSE, TRUE, m_1, a2map, d2, dmap, ddist, w1map, w1 ,LAMBDA );
      RETURN w1_g;
    END;
		WeightGrad1 (DATASET(Layout_Part) w1, DATASET(Layout_Part) d2) := FUNCTION
			//w1_g_ := big_big_small(a2map, d2, dmap, ddist, w1map, m_1);
			w1_g_ := d2Xt(a2map, d2, dmap, ddist, w1map, m_1);
			w1_g  := PBblas.PB_daxpy(LAMBDA, w1, w1_g_);
      RETURN w1_g;
    END;
		
    //WeightGrad2 returns gradient for w2
    WeightGrad2_ (DATASET(Layout_Part) w2, DATASET(Layout_Part) d3, DATASET(Layout_Part) a2) := FUNCTION
      w2_g := PBblas.PB_dgemm(FALSE, TRUE, m_1, a3map, d3, a2map, a2, w2map, w2 ,LAMBDA );
      RETURN w2_g;
    END;
		
		WeightGrad2 (DATASET(Layout_Part) w2, DATASET(Layout_Part) d3, DATASET(Layout_Part) a2) := FUNCTION
			//w2_g_ := big_big_small(a3map, d3, a2map, a2, w2map, m_1);
			w2_g_ := d3a2t(a3map, d3, a2map, a2, w2map, w1_partitions, m_1);
			w2_g  := PBblas.PB_daxpy(LAMBDA, w2, w2_g_);
      RETURN w2_g;
    END;
    //BiasGrad1 calculates the bias gradients for b1
    BiasGrad1 (DATASET(Layout_Part) d2) := FUNCTION
			//b1_g := col_mean(a2map, d2, Ones_VecMap, Ones_Vecdist, b1vecmap, m_1);
			b1_g := colmean_bias_grad (a2map, d2, b1vecmap, w1_partitions + w2_partitions);
      RETURN b1_g;
    END;
		
		BiasGrad1_ (DATASET(Layout_Part) d2) := FUNCTION
      b1_g := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a2map, d2, Ones_VecMap, Ones_Vecdist, b1vecmap);
      RETURN b1_g;
    END;
    //BiasGrad2 calculates the bias gradients for b2
    BiasGrad2_ (DATASET(Layout_Part) d3) := FUNCTION
      b2_g := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a3map, d3, Ones_VecMap, Ones_Vecdist, b2vecmap);
      RETURN b2_g;
    END;
		
		BiasGrad2 (DATASET(Layout_Part) d3) := FUNCTION
      //b2_g := col_mean(a3map, d3, Ones_VecMap, Ones_Vecdist, b2vecmap, m_1);
			b2_g := colmean_bias_grad (a3map, d3, b2vecmap, w1_partitions + w2_partitions + b1_partitions);
      RETURN b2_g;
    END;
		
		
    emptyL := DATASET([], Layout_Part);
    //theta is the input weight and bias parameters for the sparseautoencoder
    //parameters are the parameters for the sparseautoencoder function
    //train_d is the train data to learn sparse autoencoder and calculate the gradient and the cost based on taht
    // train_l is empty in the sparse autoencoder (because it is an unsupervised learning)
    SparseParam_CostGradients4 :=  FUNCTION
      PBblas.Types.MUElement minuspart(Layout_Part l, UNSIGNED8 c ) := TRANSFORM
        SELF.partition_id := l.partition_id - c;
        SElF.no := 1;
        SELF := l;
      END;
      w1m := w1dist;
      w2m := w2dist;
      b1v := b1vecdist;
      b2v := b2vecdist;
      a2 := FF2 (w1m, b1v);
			a2_ := FF2_ (w1m, b1v);
      a3 := FF3 (w2m, b2v, a2);
			a3_ := FF3_ (w2m, b2v, a2_);
      d3 := DELTA3 (a3);
			d3_ := DELTA3_ (a3_);
      rohat_a2 := rohat(a2);
			rohat_a2_ := rohat_(a2_);
      d2 := DELTA2 (w2m, a2, d3,rohat_a2);
			d2_ := DELTA2_ (w2m, a2_, d3_,rohat_a2_);
      wg1 := WeightGrad1 (w1m, d2);
      wg2 := WeightGrad2 (w2m, d3, a2);
      bg1 := BiasGrad1 (d2);
      bg2 := BiasGrad2 (d3);
			
			
			wg1_ := WeightGrad1_ (w1m, d2_);
      wg2_ := WeightGrad2_ (w2m, d3_, a2_);
      bg1_ := BiasGrad1_ (d2_);
      bg2_ := BiasGrad2_ (d3_);
      //calculate cost
      //PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1, dmap, ddist, b1map, b1m, 1.0);
      // squared_error_cost= 0.5*sum(sum((x-a3).^2));
      // cost=(1/m)*squared_error_cost+(lambda/2)*(sum(W2(:).^2)+sum(W1(:).^2))+beta*sum(KL(sparsityParam,rhohat));
			
			simple := {REAL8 v};
			simple SE_tran (Layout_Part le, Layout_Part ri) := TRANSFORM
				SELF.v := sum_pow2(le.part_cols * le.part_rows, le.mat_part, ri.mat_part);
			END;
			d_a3_sum_part := JOIN (a3, ddist, LEFT.partition_id = RIGHT.partition_id,SE_tran(LEFT,RIGHT), LOCAL );
      //squared_error_cost := 0.5*PBblas.SumElements(PBblas.Apply2Elements(dmap, PBblas.PB_daxpy(-1.0, a3, ddist), pow2));
			squared_error_cost := SUM (d_a3_sum_part, d_a3_sum_part.v);
      cost_term1 := (1/m)*squared_error_cost;
			
			simple term2_3_tran (Layout_Part le) := TRANSFORM
				SELF.v := sum_sq(le.part_cols * le.part_rows, le.mat_part);
			END;
			w1_term3 := PROJECT (w1m, term2_3_tran (LEFT), LOCAL);
			w2_term2 := PROJECT (w2m, term2_3_tran (LEFT), LOCAL);

      // cost_term2 := (lambda/2)* PBblas.SumElements(PBblas.Apply2Elements(w2map, w2m, pow2));
      //cost_term3 := (lambda/2)* PBblas.SumElements(PBblas.Apply2Elements(w1map, w1m, pow2));
			cost_term2 := (lambda/2)* SUM (w2_term2, w2_term2.v); 
			cost_term3 := (lambda/2)* SUM (w1_term3, w1_term3.v);
			
			simple kl_tran (Layout_Part le) := TRANSFORM
				SELF.v := sum_kl(le.part_cols * le.part_rows, le.mat_part, sparsityParam);
			END;
			
      PBblas.Types.value_t klterm(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := sparsityParam * LN(sparsityParam/v) + (1-sparsityParam) * LN ((1-sparsityParam)/(1-v));
      //KL := PBblas.Apply2Elements (Hiddmap,rohat_a2,klterm);
		  KL := PROJECT (rohat_a2, kl_tran(LEFT), LOCAL);
      //cost_term4 := beta * PBblas.SumElements(KL);
			cost_term4 := beta * SUM (KL, KL.v);
      cost := cost_term1 + cost_term2 + cost_term3 +cost_term4;    
      costField := DATASET ([{1,1,cost}],ML.Types.NumericField);
      one_map := PBblas.Matrix_Map(1,1,1,1);
      Cost_part_no := PBblas.MU.TO(ML.DMat.Converted.FromNumericFieldDS(costField,one_map),2);
      //convert w and b gradients to a datasets of layoutparts where partition_id differentiate them
      //w1_grad has partition_id from 1 to w1_partitions
      //w2_grad had partition_ds from w1_partitions+1 to w1_partitions + w2_partitions
      //b1_grad has partition_ids from w1_partitions + w2_partitions+1 to w1_partitions + w2_partitions+b1_partitions
      //b2_grad has partition_ids from w1_partitions + w2_partitions+b1_partitions+1 to w1_partitions + w2_partitions+b1_partitions+b2_partitions
      PBblas.Types.MUElement addpart(Layout_Part l, UNSIGNED8 c ) := TRANSFORM
        SELF.partition_id := l.partition_id + c;
        SElF.no := 1;
        SELF := l;
      END;
      wg1_reshape_no := Pbblas.MU.TO(wg1,1);
      wg2_reshape_no := PROJECT (wg2, addpart(LEFT, w1_partitions ), LOCAL);
			wg2_reshape_no_ := PROJECT (wg2_, addpart(LEFT, w1_partitions ), LOCAL);
      bg1_reshape_no := PROJECT (bg1, addpart (LEFT, w1_partitions + w2_partitions),LOCAL);
      bg2_reshape_no := PROJECT (bg2, addpart (LEFT, w1_partitions + w2_partitions + b1_partitions),LOCAL);
     // theta_Part_no := wg1_reshape_no + wg2_reshape_no + bg1_reshape_no + bg2_reshape_no;
      theta_Part_no := PROJECT (wg1 + wg2 + bg1 + bg2 ,TRANSFORM (PBblas.Types.MUElement, SELF.no :=1 ; SELF := LEFT) , LOCAL);
			//RETURN PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1dist, dmap, ddist, b1map);
			//RETURN WX(w1map, w1dist, dmap, ddist, b1map, w1dist,  1);
			//RETURN WX_repmatb(w1map, w1dist, dmap, ddist, b1vecmap, b1vecdist, b1map, 1.0) ;
			//RETURN col_sum(dmap, ddist, Ones_VecMap, Ones_Vecdist, b2vecmap);
		
			
			
			
			//mymy2 := DELTA2 (w2m, a2, d3,rohat_a2) + DELTA2_ (w2m, a2, d3,rohat_a2);
			//mymy2 := big_big_small(a2map, d2, dmap, ddist, w1map, 8);
			//mymy2 := col_mean(a3map, d3, Ones_VecMap, Ones_Vecdist, b2vecmap);
			mymy2 := wg1 + wg2  + bg1 + bg2;
myformat2 := RECORD
    mymy2.node_id;
    mymy2.partition_id;
    mymy2.block_row;
    mymy2.block_col;
    mymy2.first_row;
    mymy2.part_rows;
    mymy2.first_col;
    mymy2.part_cols;
		//mymy2.no;
		//mymy2.mat_part;
		INTEGER real_node := STD.System.Thorlib.Node();
END;
	thisR := TABLE(mymy2,myformat2,LOCAL); 
	theta_part_no_check :=  ASSERT(theta_Part_no, node_id=Thorlib.node() and node_id=(partition_id-1), 'sparse autoencoder gradient is not distributed correctly', FAIL);
	return_record := RECORD (Layout_Part)
		REAL8 cost_value;
	END;
	ToReturn := PROJECT (theta_part_no_check, TRANSFORM (return_record, SELF.cost_value := cost; SELF:=LEFT), LOCAL);
	//RETURN  theta_part_no_check + Cost_part_no;
	RETURN ToReturn;
	// a2_part :=   W1X_repmatb1(w1map, w1dist, dmap, ddist, b1vecmap, b1v_, b1map, 0.0, 0);
	// a3_part :=  W2a2_repmatb2(w2map, w2dist, b1map, a2_part, b2vecmap, b2v_, b2map, 0.0);
	// rohat_part := col_mean(a2map, a2_part, Ones_VecMap, Ones_Vecdist, Hiddmap);
	//RETURN W2td3_repmatsparsity(w2map, w2dist, dmap, ddist, b1vecmap, b1v_, b1map, 0.0, 0);
	//RETURN W2td3_repmatsparsity(w2map, w2dist, dmap, ddist, Hiddmap, b1v, a2map, BETA);
	//RETURN cost_term4;

//RETURN d2;
			//RETURN thisR;
      //RETURN theta_Part_no;
    END;//END SparseParam_CostGradients4  
    RETURN SparseParam_CostGradients4;
   END;//END SA_lbfgs_Compatible2_param_part


//in this implementation the assumption is that in the recived theta, both bias vectors are distributed on the same node, same partition
// first we distributed parameters in all the nodes are data is distributed to, then we apply the analysis

EXPORT SA_lbfgs_Compatible2_param_part_onebias_distparam ( DATASET(Layout_Part) theta, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Part) TrainData , DATASET(Layout_Part) TrainLabel) := FUNCTION
   

// This function repeats the bias vector of size r*1 , s times in a columnwise format, so the final results will be a r*s matrix
//D = [1,2,3], r=3, s=2 => output=[1,2,3,1,2,3]
  SET OF REAL8 repeatbias(PBblas.Types.dimension_t r, PBblas.Types.dimension_t s, PBblas.Types.matrix_t D) := BEGINC++

    #body
    __lenResult = r * s * sizeof(double);
    __isAllResult = false;
    double * result = new double[r*s];
    __result = (void*) result;
    double *cell = (double*) d;
    uint32_t cells =  r * s;
    uint32_t i;
    uint32_t pos;
    for (i=0; i<cells; i++) {
      pos = i % r;
      result[i] = cell[pos];
    }
  ENDC++;
	
//this function calculates d3=-(y-a3).*(a3.*(1-a3));
//N is the total number of elements in each matrix
//A is the a3 matrix in SET format
//Y is the y matrix in SET format
	SET OF REAL8 d3_cal(PBblas.Types.dimension_t N, PBblas.Types.matrix_t A, PBblas.Types.matrix_t Y) := BEGINC++

    #body
    __lenResult = n * sizeof(double);
    __isAllResult = false;
    double * result = new double[n];
    __result = (void*) result;
    double *cella = (double*) a;
		double *celly = (double*) y;
    uint32_t cells =  n;
    uint32_t i;
    for (i=0; i<cells; i++) {
      result[i] = (cella[i]-celly[i])*(cella[i]*(1-cella[i]));
    }
  ENDC++;
	SET OF REAL8 d2_cal(PBblas.Types.dimension_t N, PBblas.Types.matrix_t A, PBblas.Types.matrix_t Y) := BEGINC++

    #body
    __lenResult = n * sizeof(double);
    __isAllResult = false;
    double * result = new double[n];
    __result = (void*) result;
    double *cella = (double*) a;
		double *celly = (double*) y;
    uint32_t cells =  n;
    uint32_t i;
    for (i=0; i<cells; i++) {
      result[i] = (celly[i])*(cella[i]*(1-cella[i]));
    }
  ENDC++;
//result = M + bet * repmat (V, 1, r)
	SET OF REAL8 mat_vec_sum(PBblas.Types.dimension_t N, PBblas.Types.dimension_t r, PBblas.Types.matrix_t M, PBblas.Types.matrix_t V, PBblas.Types.value_t bet) := BEGINC++

    #body
    __lenResult = n * sizeof(double);
    __isAllResult = false;
    double * result = new double[n];
    __result = (void*) result;
    double *cellm = (double*) m;
		double *cellv = (double*) v;
    uint32_t cells =  n;
    uint32_t i;
		uint32_t pos;
    for (i=0; i<cells; i++) {
		  pos = i % r;
      result[i] = cellm[i] + (bet * cellv[pos]);
    }
  ENDC++;

//result = M + bet * repmat (-sp./V+((1-sp)./(1-V)), 1, r)
	SET OF REAL8 mat_vec_sparsity_sum(PBblas.Types.dimension_t N, PBblas.Types.dimension_t r, PBblas.Types.matrix_t M, PBblas.Types.matrix_t V, PBblas.Types.value_t bet, PBblas.Types.value_t sp) := BEGINC++

    #body
    __lenResult = n * sizeof(double);
    __isAllResult = false;
    double * result = new double[n];
    __result = (void*) result;
    double *cellm = (double*) m;
		double *cellv = (double*) v;
		double *sv = new double[r];
		double sp_ = -1*sp;
		double sp_1 = 1-sp;
    uint32_t cells =  n;
    uint32_t i;
		uint32_t pos;
		for (pos=0; pos<r; pos++) {
		//(sparsityParam_/v)+(sparsityParam_1/(1-v));
      sv[pos] = ((sp_/cellv[pos])+(sp_1/(1-cellv[pos])));
    }
    for (i=0; i<cells; i++) {
		  pos = i % r;
      result[i] = cellm[i] + (bet * sv[pos]);
    }
  ENDC++;
	// //result = sigmoid (M + repmat (V, 1, r))
	SET OF REAL8 mat_vec_sum_sigmoid(PBblas.Types.dimension_t N, PBblas.Types.dimension_t r, PBblas.Types.matrix_t M, PBblas.Types.matrix_t V) := BEGINC++

    #body
    __lenResult = n * sizeof(double);
    __isAllResult = false;
    double * result = new double[n];
    __result = (void*) result;
    double *cellm = (double*) m;
		double *cellv = (double*) v;
    uint32_t cells =  n;
    uint32_t i;
		uint32_t pos;
    for (i=0; i<cells; i++) {
		  pos = i % r;
      result[i] = 1/(1 + exp(-1*(cellm[i] + cellv[pos])));
    }
  ENDC++;
	
		// //result = sigmoid (M + repmat (V(offset:offset+r), 1, r))
	SET OF REAL8 mat_vec_part_sum_sigmoid(PBblas.Types.dimension_t N, PBblas.Types.dimension_t offset, PBblas.Types.dimension_t r, PBblas.Types.matrix_t M, PBblas.Types.matrix_t V) := BEGINC++

    #body
    __lenResult = n * sizeof(double);
    __isAllResult = false;
    double * result = new double[n];
    __result = (void*) result;
    double *cellm = (double*) m;
		double *cellv = (double*) v;
    uint32_t cells =  n;
    uint32_t i;
		uint32_t pos;
    for (i=0; i<cells; i++) {
		  pos = (i % r) + offset;
      result[i] = 1/(1 + exp(-1*(cellm[i] + cellv[pos])));
    }
  ENDC++;
	
	
	SET OF REAL8 sum_col (PBblas.Types.dimension_t r, PBblas.Types.dimension_t s, PBblas.Types.matrix_t D) := BEGINC++

    #body
    __lenResult = r * sizeof(double);
    __isAllResult = false;
    double * result = new double[r];
    __result = (void*) result;
    double *cell = (double*) d;
    uint32_t cells =  r * s;
    uint32_t i;
    uint32_t pos;
		for (i=0; i<r; i++) {
      result[i] = 0;
    }
    for (i=0; i<cells; i++) {
      pos = i % r;
      result[pos] = result[pos] + cell[i];
    }

  ENDC++;
	
		SET OF REAL8 sum_col_alpha (PBblas.Types.dimension_t r, PBblas.Types.dimension_t s, PBblas.Types.matrix_t D, REAL8 thisalpha) := BEGINC++

    #body
    __lenResult = r * sizeof(double);
    __isAllResult = false;
    double * result = new double[r];
    __result = (void*) result;
    double *cell = (double*) d;
    uint32_t cells =  r * s;
    uint32_t i;
    uint32_t pos;
		for (i=0; i<r; i++) {
      result[i] = 0;
    }
    for (i=0; i<cells; i++) {
      pos = i % r;
      result[pos] = result[pos] + cell[i];
    }
		for (i=0; i<r; i++) {
      result[i] = result[i] * thisalpha;
    }

  ENDC++;
	//0.5 * sum ((M-V).^2)
	REAL8 sum_pow2(PBblas.Types.dimension_t N, PBblas.Types.matrix_t M, PBblas.Types.matrix_t V) := BEGINC++

    #body
    double result = 0;
		double tmpp ;
    double *cellm = (double*) m;
		double *cellv = (double*) v;
    uint32_t i;
		for (i=0; i<n; i++) {
		  tmpp =(cellm[i] - cellv [i]);
      result = result + (tmpp*tmpp);
    }
		return(0.5*result);

  ENDC++;
	//sum(M.^2)
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
	// sum (kl(rho, M))
	REAL8 sum_kl(PBblas.Types.dimension_t N, PBblas.Types.matrix_t M, PBblas.Types.value_t rho) := BEGINC++

    #body
    double result = 0;
		double tmpp ;
    double *cellm = (double*) m;
    uint32_t i;
		for (i=0; i<n; i++) {
			result = result + (rho*log(rho/cellm[i])) + ((1-rho)*log((1-rho)/(1-cellm[i])));
    }
		return(result);

  ENDC++;
    ddist := TrainData;
    m := CostFunc_params(id=1)[1].value;
    num_feat := CostFunc_params(id=2)[1].value;
    num_hid := CostFunc_params(id=3)[1].value;
    part_rows := CostFunc_params(id=4)[1].value; // partition size for the features (number of rows)
    part_cols := CostFunc_params(id=5)[1].value; // partition size for the number of columns (samples) in the input data
    BETA := CostFunc_params(id=6)[1].value;
    sparsityParam := CostFunc_params(id=7)[1].value;
    LAMBDA := CostFunc_params(id=8)[1].value;
    
    m_1 := 1/m;
    sparsityParam_ := -1*sparsityParam;
    sparsityParam_1 := 1-sparsityParam;
    
    //Create map for block matrix d
    dmap := PBblas.Matrix_Map(num_feat,m,part_rows,part_cols);
    
    //Create block matrix Ytmp
    Ymap := dmap;
    Ydist := ddist;
    //Creat maps for block matrices for weights

    w1map := PBblas.Matrix_Map(num_hid, num_feat, num_hid, part_rows);
    w2map := PBblas.Matrix_Map(num_feat, num_hid, part_rows, num_hid);
     //each bias vector is converted to block format
    
    b1vecmap := PBblas.Matrix_Map(num_hid, 1, num_hid, 1);
    b2vecmap := PBblas.Matrix_Map(num_feat, 1, part_rows, 1);
    
    w1_partitions := w1map.partitions_used;
    w2_partitions := w2map.partitions_used;
    b1_partitions := b1vecmap.partitions_used;
    b2_partitions := b2vecmap.partitions_used;
    Layout_Part minuspart(Layout_Part l, UNSIGNED8 c ) := TRANSFORM
      SELF.partition_id := l.partition_id - c;
      SELF := l;
    END;
    //w1m := w1dist;
    w1dist := theta (partition_id <= w1_partitions);
    //w2m := w2dist;
    w2dist := theta (partition_id > w1_partitions AND partition_id <= w1_partitions + w2_partitions);
    //w2dist  := PROJECT (w2m_, minuspart(LEFT, w1_partitions ),LOCAL);
    //b1v := b1vecdist;
    b1vecdist := theta (partition_id > w1_partitions + w2_partitions AND partition_id <= w1_partitions + w2_partitions + b1_partitions);
    //b1vecdist  := PROJECT (b1v_, minuspart (LEFT, w1_partitions + w2_partitions),LOCAL);
    
    //b2v := b2vecdist;
    b2vecdist := theta (partition_id > w1_partitions + w2_partitions + b1_partitions AND partition_id <= w1_partitions + w2_partitions + b1_partitions + b2_partitions);
    //b2vecdist  := PROJECT (b2v_, minuspart (LEFT, w1_partitions + w2_partitions + b1_partitions),LOCAL);
    
     //functions used
    PBblas.Types.value_t sp_reci(PBblas.Types.value_t v,PBblas.Types.dimension_t r,PBblas.Types.dimension_t c) := sparsityParam_/v;
    PBblas.Types.value_t sp_1_reci(PBblas.Types.value_t v,PBblas.Types.dimension_t r,PBblas.Types.dimension_t c) := sparsityParam_1/(1-v);
    //sparsity_delta=((-sparsityParam./rhohat)+((1-sparsityParam)./(1.-rhohat)));
    PBblas.Types.value_t sp_delta(PBblas.Types.value_t v,PBblas.Types.dimension_t r,PBblas.Types.dimension_t c) := (sparsityParam_/v)+(sparsityParam_1/(1-v));
    PBblas.Types.value_t siggrad(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := v*(1.0-v);
    PBblas.Types.value_t sigmoid(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := 1/(1+exp(-1*v));
    PBblas.Types.value_t pow2(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := v*v;
    //maps used
    b1map := PBblas.Matrix_Map(num_hid, m, num_hid, part_cols);
    b2map := PBblas.Matrix_Map(num_feat, m, part_rows, part_cols);
    a2map := b1map;
    a3map := b2map;
    HL_nodes := num_hid;//number of nodes in the hidden layer
    Hiddmap := b1vecmap;
    //onevec for calculating rhohat	 
	 
	 Ones_VecMap := PBblas.Matrix_Map(m, 1, part_cols, 1);
    //New Vector Generator
    // Layout_Cell gen(UNSIGNED4 c, UNSIGNED4 NumRows) := TRANSFORM
      // SELF.x := ((c-1) % NumRows) + 1;
      // SELF.y := ((c-1) DIV NumRows) + 1;
      // SELF.v := 1;
    // END;
    //Create Ones Vector for the calculations in the step fucntion
    // Ones_Vec := DATASET(m, gen(COUNTER, m));
    Ones_Vecdist := TrainLabel;
Layout_Target := PBblas.Types.Layout_Target;




row_block_bigmat (UNSIGNED8 p, UNSIGNED8 nn) := FUNCTION // p is the partition number, nn is the number of rows
	RETURN ((p-1) % nn )+1;
END;

col_block_bigmat (UNSIGNED8 p, UNSIGNED8 nn) := FUNCTION // p is the partition number, nn is the number of rows
	RETURN ((p-1) DIV nn )+1;
END;

block_smallmat (UNSIGNED8 p, UNSIGNED8 offset) := FUNCTION 
	RETURN p-offset;
END;


// first distribute all the parameters to the same nodes data is distributed on
data_nodesused := dmap.col_blocks;
Layout_Part_newnode := RECORD (Layout_Part)
	PBblas.Types.node_t new_node_id;
END;
Layout_Part_newnode norm_theta (Layout_Part te, INTEGER co) := TRANSFORM
	SELF.new_node_id := co % data_nodesused  ;
	SELF:= te;
END;
theta_norm := NORMALIZE(theta, data_nodesused, norm_theta(LEFT, COUNTER) );
theta_dist := DISTRIBUTE (theta_norm, new_node_id);
// calculate a2
//A_in := w1 is h*f where f is divided to partitions of size prow
//B_in := data : ddist
// bb_in := bias1
//returns the sigmoid of the result
//W1X_repmatb1(w1map, w1, dmap, ddist, b1vecmap, b1v, b1map, 0.0, 0);
W1X_repmatb1(PBblas.IMatrix_Map map_c) := FUNCTION
	SET OF PBblas.Types.value_t empty_array := [];

  A := theta_dist;
	B := ddist;
	
	Layout_Part mul(Layout_Part b_part, Layout_Part_newnode a_part):=TRANSFORM
		part_id := b_part.block_col;
    part_c_rows := map_c.part_rows(part_id);
    part_c_cols := map_c.part_cols(part_id);
    k := a_part.part_cols;
    SELF.partition_id := part_id;
    SELF.node_id      := b_part.node_id;
    SELF.block_row    := a_part.block_row;
    SELF.block_col    := b_part.block_col;
    SELF.first_row    := map_c.first_row(part_id);
    SELF.part_rows    := part_c_rows;
    SELF.first_col    := map_c.first_col(part_id);
    SELF.part_cols    := part_c_cols;
    SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
                                    part_c_rows, part_c_cols, k,
                                    1.0, a_part.mat_part, b_part.mat_part,
                                    0.0, empty_array);
  END;
	ab_prod_ := JOIN(B, A(partition_id<=w1_partitions), LEFT.block_row = RIGHT.block_col , mul(LEFT,RIGHT), LOCAL);
  // Sum terms
  Layout_Part sumTerms(Layout_Part cumm, Layout_Part term) := TRANSFORM
    N := cumm.part_rows * cumm.part_cols;
    SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term.mat_part, 1);
    SELF := cumm;
  END;
	sorted_ab_prod := SORT(ab_prod_, partition_id, LOCAL);
  ab_prod := ROLLUP(sorted_ab_prod, sumTerms(LEFT, RIGHT), partition_id, LOCAL);
	// add bias vector to the result
	Layout_Part addbias (Layout_Part cumm, Layout_Part_newnode term) := TRANSFORM
    N := cumm.part_rows * cumm.part_cols;
		SELF.mat_part := mat_vec_sum_sigmoid(N, term.part_rows, cumm.mat_part, term.mat_part);
    SELF := cumm;
  END;
	 ab_bb := JOIN(ab_prod, A(partition_id = W1_partitions + W2_partitions + 1), LEFT.block_row = RIGHT.block_row , addbias(LEFT,RIGHT), LOCAL);
	RETURN ab_bb;
END; // END W1X_repmatb1




W2td3_repmatsparsity_(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_bb, DATASET(Layout_Part) bb_in, PBblas.IMatrix_Map map_c, REAL8 beta_in=0) := FUNCTION

SET OF PBblas.Types.value_t empty_array := [];
	// First check maps for compatability.  Normalize for transpose operations.
		a_matrix_rows := map_a.matrix_cols;
		a_matrix_cols := map_a.matrix_rows;
		a_row_blocks  := map_a.col_blocks;
		a_col_blocks  := map_a.row_blocks;
		b_matrix_rows := map_b.matrix_rows;
		b_matrix_cols := map_b.matrix_cols;
		b_row_blocks  := map_b.row_blocks;
		b_col_blocks  := map_b.col_blocks;
		c_matrix_rows := map_c.matrix_rows;
		c_matrix_cols := map_c.matrix_cols;
		c_row_blocks  := map_c.row_blocks;
		c_col_blocks  := map_c.col_blocks;
	A := theta_dist;
  B := B_in;
	
  //multiply
	B_row_part := map_b.row_blocks;
	Layout_Part mul(Layout_Part b_part, Layout_Part a_part):=TRANSFORM
    part_id := b_part.block_col;
    part_c_rows := map_c.part_rows(part_id);
    part_c_cols := map_c.part_cols(part_id);
    k := a_part.part_rows;
    SELF.partition_id := part_id;
    SELF.node_id      := b_part.node_id;
    SELF.block_row    := a_part.block_col;
    SELF.block_col    := b_part.block_col;
    SELF.first_row    := map_c.first_row(part_id);
    SELF.part_rows    := part_c_rows;
    SELF.first_col    := map_c.first_col(part_id);
    SELF.part_cols    := part_c_cols;
    SELF.mat_part     := PBblas.BLAS.dgemm(TRUE, FALSE,
                                    part_c_rows, part_c_cols, k,
                                    1.0, a_part.mat_part, b_part.mat_part,
                                    0.0, empty_array);

  END;
	ab_prod_ := JOIN(B, A(partition_id > W1_partitions AND partition_id <= (W1_partitions + W2_partitions)), LEFT.block_row = RIGHT.block_row, mul(LEFT,RIGHT),LOCAL);

	 // Sum terms
  Layout_Part sumTerms(Layout_Part cumm, Layout_Part term) := TRANSFORM
    N := cumm.part_rows * cumm.part_cols;
    SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term.mat_part, 1);
    SELF := cumm;
  END;
	sorted_ab_prod := SORT(ab_prod_, partition_id, LOCAL);
  ab_prod := ROLLUP(sorted_ab_prod, sumTerms(LEFT, RIGHT), partition_id, LOCAL);

	Layout_Part add_sparsity(Layout_Part cumm, Layout_Part term) := TRANSFORM
		cumm_part_cols := cumm.part_cols;// number of columns in this partition
		term_part_rows := term.part_rows;//number of elements in this partition of the bias vector
		N := cumm.part_rows * cumm.part_cols;
		SELF.mat_part := mat_vec_sparsity_sum(N, term_part_rows, cumm.mat_part, term.mat_part, beta_in, sparsityParam);
		SELF := cumm;
	END;

	ab_bb := JOIN(ab_prod, bb_in,LEFT.block_row = RIGHT.block_row , add_sparsity(LEFT,RIGHT), LOCAL);
	RETURN ab_bb;
END; // END W2td3_repmatsparsity_


W2td3_repmatsparsity(DATASET(Layout_Part) B_in, DATASET(Layout_Part) bb_in, PBblas.IMatrix_Map map_c, REAL8 beta_in=0) := FUNCTION

SET OF PBblas.Types.value_t empty_array := [];
	// First check maps for compatability.  Normalize for transpose operations.

	A := theta_dist;
  B := B_in;
	
  //multiply
	Layout_Part mul(Layout_Part b_part, Layout_Part a_part):=TRANSFORM
    part_id := b_part.block_col;
    part_c_rows := map_c.part_rows(part_id);
    part_c_cols := map_c.part_cols(part_id);
    k := a_part.part_rows;
    SELF.partition_id := part_id;
    SELF.node_id      := b_part.node_id;
    SELF.block_row    := a_part.block_col;
    SELF.block_col    := b_part.block_col;
    SELF.first_row    := map_c.first_row(part_id);
    SELF.part_rows    := part_c_rows;
    SELF.first_col    := map_c.first_col(part_id);
    SELF.part_cols    := part_c_cols;
    SELF.mat_part     := PBblas.BLAS.dgemm(TRUE, FALSE,
                                    part_c_rows, part_c_cols, k,
                                    1.0, a_part.mat_part, b_part.mat_part,
                                    0.0, empty_array);

  END;
	ab_prod_ := JOIN(B, A(partition_id > W1_partitions AND partition_id <= (W1_partitions + W2_partitions)), LEFT.block_row = RIGHT.block_row, mul(LEFT,RIGHT),LOCAL);

	 // Sum terms
  Layout_Part sumTerms(Layout_Part cumm, Layout_Part term) := TRANSFORM
    N := cumm.part_rows * cumm.part_cols;
    SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term.mat_part, 1);
    SELF := cumm;
  END;
	sorted_ab_prod := SORT(ab_prod_, partition_id, LOCAL);
  ab_prod := ROLLUP(sorted_ab_prod, sumTerms(LEFT, RIGHT), partition_id, LOCAL);

	Layout_Part add_sparsity(Layout_Part cumm, Layout_Part term) := TRANSFORM
		cumm_part_cols := cumm.part_cols;// number of columns in this partition
		term_part_rows := term.part_rows;//number of elements in this partition of the bias vector
		N := cumm.part_rows * cumm.part_cols;
		SELF.mat_part := mat_vec_sparsity_sum(N, term_part_rows, cumm.mat_part, term.mat_part, beta_in, sparsityParam);
		SELF := cumm;
	END;

	ab_bb := JOIN(ab_prod, bb_in,LEFT.block_row = RIGHT.block_row , add_sparsity(LEFT,RIGHT), LOCAL);
	RETURN ab_bb;
END; // END W2td3_repmatsparsity




//returns the sigmoid of the result


W2a2_repmatb2(DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_c) := FUNCTION
SET OF PBblas.Types.value_t empty_array := [];

  A := theta_dist;
	B := B_in;

	C_row_part := map_c.row_blocks;
	Layout_Part mul2(Layout_Part b_part, Layout_Part_newnode a_part):=TRANSFORM
    real_part_id := b_part.partition_id;
		part_id := (real_part_id-1)*C_row_part + a_part.block_row;
    part_c_rows := map_c.part_rows(part_id);
    part_c_cols := map_c.part_cols(part_id);
    k := a_part.part_cols;
    SELF.partition_id := part_id;
    SELF.node_id      := b_part.node_id;
    SELF.block_row    := a_part.block_row;
    SELF.block_col    := b_part.block_col;
    SELF.first_row    := map_c.first_row(part_id);
    SELF.part_rows    := part_c_rows;
    SELF.first_col    := map_c.first_col(part_id);
    SELF.part_cols    := part_c_cols;
    SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
                                    part_c_rows, part_c_cols, k,
                                    1.0, a_part.mat_part, b_part.mat_part,
                                    0.0, empty_array);
  END;
  //ab_prod := JOIN(B, A, TRUE , mul2(LEFT,RIGHT),ALL); // Each A (weight matrix) is copied in each B (X matrix) node
	ab_prod := JOIN(B, A(partition_id > W1_partitions AND partition_id <= (W1_partitions + W2_partitions)), LEFT.block_row = RIGHT.block_col , mul2(LEFT,RIGHT), LOCAL); // Each A (weight matrix) is copied in each B (X matrix) node

	//add bias
	Layout_Part addbias_(Layout_Part cumm, Layout_Part_newnode term) := TRANSFORM
		cumm_part_rows := cumm.part_rows;
    N := cumm.part_rows * cumm.part_cols;
		off := cumm.first_row - 1;
		SELF.mat_part := mat_vec_part_sum_sigmoid(N, off, cumm_part_rows, cumm.mat_part, term.mat_part);
		SELF.partition_id := cumm.partition_id;
    SELF := cumm;
  END;
	
	 ab_bb := JOIN(ab_prod,  A(partition_id = W1_partitions + W2_partitions + 2), LEFT.node_id = RIGHT.new_node_id , addbias_(LEFT,RIGHT), LOCAL);

	RETURN ab_bb;
END; // END W2a2_repmatb2
  //retunrs the sigmoid(WX+b)  
WX_repmatb_sig(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_bb, DATASET(Layout_Part) bb_in, PBblas.IMatrix_Map map_c, REAL8 beta=0.0) := FUNCTION
SET OF PBblas.Types.value_t empty_array := [];
// First check maps for compatability.  Normalize for transpose operations.
  a_matrix_rows := map_a.matrix_rows;
  a_matrix_cols := map_a.matrix_cols;
  a_row_blocks  := map_a.row_blocks;
  a_col_blocks  := map_a.col_blocks;
  b_matrix_rows := map_b.matrix_rows;
  b_matrix_cols := map_b.matrix_cols;
  b_row_blocks  := map_b.row_blocks;
  b_col_blocks  := map_b.col_blocks;
  c_matrix_rows := map_c.matrix_rows;
  c_matrix_cols := map_c.matrix_cols;
  c_row_blocks  := map_c.row_blocks;
  c_col_blocks  := map_c.col_blocks;
  A := ASSERT(A_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' + ' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(a_matrix_rows=c_matrix_rows AND a_row_blocks=c_row_blocks,
                    'A-C: ' + 'Arows: ' + a_matrix_rows +' Crows '+c_matrix_rows+' '+ 
                              'Ablocks: ' + a_row_blocks +' Cblocks '+c_row_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	B := ASSERT(B_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' +' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(b_matrix_cols=c_matrix_cols AND b_col_blocks=c_col_blocks,
                    'B-C: ' + 'Brows: ' + b_matrix_cols +' Ccols '+c_matrix_cols+' '+ 
                              'Bblocks: ' + b_row_blocks +' Cblocks '+c_col_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	
  //multiply
	
	Layout_Part mul2(Layout_Part b_part, Layout_Part a_part):=TRANSFORM
    part_id     := b_part.partition_id;    //arbitrary choice
    part_a_cols := a_part.part_cols;
    part_a_rows := a_part.part_rows;
    part_b_rows := b_part.part_rows;
    part_c_rows := map_c.part_rows(part_id);
    part_c_cols := map_c.part_cols(part_id);
    part_c_first_row  := map_c.first_row(part_id);
    part_c_first_col  := map_c.first_col(part_id);
    k := part_a_cols;
    SELF.partition_id := b_part.partition_id;
    SELF.node_id      := b_part.node_id;
    SELF.block_row    := b_part.block_row;
    SELF.block_col    := b_part.block_col;
    SELF.first_row    := map_c.first_row(part_id);
    SELF.part_rows    := part_c_rows;
    SELF.first_col    := part_c_first_col;
    SELF.part_cols    := part_c_cols;
    SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
                                    part_c_rows, part_c_cols, k,
                                    1.0, a_part.mat_part, b_part.mat_part,
                                    0.0, empty_array);
  END;
  ab_prod := JOIN(B, A,TRUE , mul2(LEFT,RIGHT),ALL); // Each A (weight matrix) is copied in each B (X matrix) node

// add bias vector to each columns of X
// each bias vector (b) is copied (ALL JOIN) to each node of X. The bias vector is then normalized to repeat it to the number of columns of X in that node and then the two are added
  Layout_Part addbias(Layout_Part cumm, Layout_Part term) := TRANSFORM
		cumm_part_cols := cumm.part_cols;
    N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
		Elem := {PBblas.Types.value_t v};
		Elem_r := {PBblas.Types.value_t v, UNSIGNED r};
		elems := DATASET(term.mat_part, Elem);
		Elem_r rep (Elem l, UNSIGNED c) := TRANSFORM
			SELF.r := c;
			SELF := l;
		END;
		elems_rep := NORMALIZE(elems, cumm_part_cols, rep(LEFT, COUNTER));// each value in bias vector is repeated cumm_part_cols number of times (as the numebr of cumm columns in this node)
		elems_rep_sort := SORT(elems_rep, r);
		term_rep_set := SET (elems_rep_sort, v);
    SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term_rep_set, 1);
		SELF.partition_id := cumm.partition_id;
    SELF := cumm;
  END;
	
	 ab_bb := JOIN(ab_prod, bb_in,TRUE , addbias(LEFT,RIGHT),ALL);
	//rslt := A;
  RETURN ab_bb;
END; // END WX_repmatb_sig
		
		

		
		
		
		

WtX_repmatb(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_bb, DATASET(Layout_Part) bb_in, PBblas.IMatrix_Map map_c, REAL8 beta=0.0) := FUNCTION
	SET OF PBblas.Types.value_t empty_array := [];
	// First check maps for compatability.  Normalize for transpose operations.
		a_matrix_rows := map_a.matrix_cols;
		a_matrix_cols := map_a.matrix_rows;
		a_row_blocks  := map_a.col_blocks;
		a_col_blocks  := map_a.row_blocks;
		b_matrix_rows := map_b.matrix_rows;
		b_matrix_cols := map_b.matrix_cols;
		b_row_blocks  := map_b.row_blocks;
		b_col_blocks  := map_b.col_blocks;
		c_matrix_rows := map_c.matrix_rows;
		c_matrix_cols := map_c.matrix_cols;
		c_row_blocks  := map_c.row_blocks;
		c_col_blocks  := map_c.col_blocks;
		A := ASSERT(A_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'YES' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'NO' + ' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(a_matrix_rows=c_matrix_rows AND a_row_blocks=c_row_blocks,
                    'A-C: ' + 'Arows: ' + a_matrix_rows +' Crows '+c_matrix_rows+' '+ 
                              'Ablocks: ' + a_row_blocks +' Cblocks '+c_row_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
  B := ASSERT(B_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'YES' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'NO' +' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(b_matrix_cols=c_matrix_cols AND b_col_blocks=c_col_blocks,
                    'B-C: ' + 'Brows: ' + b_matrix_cols +' Ccols '+c_matrix_cols+' '+ 
                              'Bblocks: ' + b_row_blocks +' Cblocks '+c_col_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
		
		//multiply
		
		Layout_Part mul2(Layout_Part b_part, Layout_Part a_part):=TRANSFORM
			part_id     := b_part.partition_id;    //arbitrary choice
			part_a_cols := a_part.part_cols;
			part_a_rows := a_part.part_rows;
			part_b_rows := b_part.part_rows;
			part_c_rows := map_c.part_rows(part_id);
			part_c_cols := map_c.part_cols(part_id);
			part_c_first_row  := map_c.first_row(part_id);
			part_c_first_col  := map_c.first_col(part_id);
			k := part_a_rows;
			SELF.partition_id := b_part.partition_id;
			SELF.node_id      := b_part.node_id;
			SELF.block_row    := b_part.block_row;
			SELF.block_col    := b_part.block_col;
			SELF.first_row    := map_c.first_row(part_id);
			SELF.part_rows    := part_c_rows;
			SELF.first_col    := part_c_first_col;
			SELF.part_cols    := part_c_cols;
			SELF.mat_part     := PBblas.BLAS.dgemm(TRUE, FALSE,
																			part_c_rows, part_c_cols, k,
																			1.0, a_part.mat_part, b_part.mat_part,
																			0.0, empty_array);
		END;
		ab_prod := JOIN(B, A,TRUE , mul2(LEFT,RIGHT),ALL); // Each A (weight matrix) is copied in each B (X matrix) node



// Apply beta
  Layout_Part applyBeta(Layout_Part part) := TRANSFORM
    SELF.mat_part := PBblas.BLAS.dscal(map_bb.matrix_rows*map_bb.matrix_cols,
                                beta, part.mat_part, 1);
    SELF:= part;
  END;
  bb_beta := PROJECT(bb_in, applyBeta(LEFT), LOCAL);
	// add the vector to each columns of X
	// each vector is copied (ALL JOIN) to each node of X. The vector is then normalized to repeat it to the number of columns of X in that node and then the two are added
		Layout_Part addvec(Layout_Part cumm, Layout_Part term) := TRANSFORM
			cumm_part_cols := cumm.part_cols;
			N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
			Elem := {PBblas.Types.value_t v};
			Elem_r := {PBblas.Types.value_t v, UNSIGNED r};
			elems := DATASET(term.mat_part, Elem);
			Elem_r rep (Elem l, UNSIGNED c) := TRANSFORM
				SELF.r := c;
				SELF := l;
			END;
			elems_rep := NORMALIZE(elems, cumm_part_cols, rep(LEFT, COUNTER));// each value in bias vector is repeated cumm_part_cols number of times (as the numebr of cumm columns in this node)
			elems_rep_sort := SORT(elems_rep, r);
			term_rep_set := SET (elems_rep_sort, v);
			SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term_rep_set, 1);
			SELF.partition_id := cumm.partition_id;
			SELF := cumm;
		END;
		
		 ab_bb := JOIN(ab_prod, bb_beta,TRUE , addvec(LEFT,RIGHT),ALL);

		//rslt := A;
		RETURN ab_bb;
END; // END WtX_repmatb
		
		
		
		// the input is a matrix in PBblas format where only columns are partitions
		//B_in : ones vector which is partitioned among nodes
		// map_c is the result's map
		col_sum(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_c) := FUNCTION
			SET OF PBblas.Types.value_t empty_array := [];
				Layout_Part mul(Layout_Part a_part, Layout_Part b_part):=TRANSFORM
					part_id     := 1;    //arbitrary choice
					part_a_cols := a_part.part_cols;
					part_a_rows := a_part.part_rows;
					part_b_rows := b_part.part_rows;
					part_c_rows := map_c.part_rows(part_id);
					part_c_cols := map_c.part_cols(part_id);
					part_c_first_row  := map_c.first_row(part_id);
					part_c_first_col  := map_c.first_col(part_id);
					k := part_a_cols;
					SELF.partition_id := 1;
					SELF.node_id      := a_part.node_id;
					SELF.block_row    := 1;
					SELF.block_col    := 1;
					SELF.first_row    := map_c.first_row(part_id);
					SELF.part_rows    := part_c_rows;
					SELF.first_col    := part_c_first_col;
					SELF.part_cols    := part_c_cols;
					SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
																					part_c_rows, part_c_cols, k,
																					1.0, a_part.mat_part, b_part.mat_part,
																					0.0, empty_array);
			END;
			col_sum_part := JOIN (A_in, B_in, LEFT.partition_id = RIGHT.partition_id, mul(LEFT, RIGHT), LOCAL );
			
			Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := le.part_rows ;
				SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1);
				SELF := le;
			END;
			//rslt := ROLLUP(col_sum_part, addup(LEFT, RIGHT), partition_id); // overload becasue of grouping
			rslt := ROLLUP(col_sum_part,TRUE, addup(LEFT, RIGHT)); // no groupijng, reduces overload
			//distribute to node one
			RETURN rslt; 
		END;//END Col_Sum
		
		
		col_mean(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_c, REAL8 mean_coef) := FUNCTION
		
			Num := map_a.matrix_cols;
			Num_1 := 1/Num;
			SET OF PBblas.Types.value_t empty_array := [];
				Layout_Part mul(Layout_Part a_part, Layout_Part b_part):=TRANSFORM
					part_id     := 1;    //arbitrary choice
					part_a_cols := a_part.part_cols;
					part_a_rows := a_part.part_rows;
					part_b_rows := b_part.part_rows;
					part_c_rows := map_c.part_rows(part_id);
					part_c_cols := map_c.part_cols(part_id);
					part_c_first_row  := map_c.first_row(part_id);
					part_c_first_col  := map_c.first_col(part_id);
					k := part_a_cols;
					SELF.partition_id := 1;
					SELF.node_id      := 0;
					SELF.block_row    := 1;
					SELF.block_col    := 1;
					SELF.first_row    := map_c.first_row(part_id);
					SELF.part_rows    := part_c_rows;
					SELF.first_col    := part_c_first_col;
					SELF.part_cols    := part_c_cols;
					SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
																					part_c_rows, part_c_cols, k,
																					Num_1, a_part.mat_part, b_part.mat_part,
																					0.0, empty_array);
			END;
			col_sum_part_ := JOIN (A_in, B_in, LEFT.partition_id = RIGHT.partition_id, mul(LEFT, RIGHT), LOCAL );
			
			
			Layout_Part col_sum_tran(Layout_Part the_part):= TRANSFORM
				SELF.mat_part := sum_col_alpha (the_part.part_rows, the_part.part_cols, the_part.mat_part, Num_1);
				SELF := the_part;
			END;
			col_sum_part := PROJECT (A_in, col_sum_tran(LEFT),LOCAL);
			Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := le.part_rows ;
				SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1);
				SELF := le;
			END;
			rslt := ROLLUP(col_sum_part,TRUE, addup(LEFT, RIGHT)); // this form of rollup avoid grouping, ROLLUP(col_sum_part,addup(LEFT, RIGHT, partiion_id)) cause all the records with the same partiion_id
			//get grouped to one node which cause a lot of overload. The current rollup form avoidds grouping and improves performance
			final_rslt := DISTRIBUTE (rslt, node_id); 
			//distribute to node one
			RETURN rslt;
		END;//END colmean
		//sum (A_in,2)
		colmean_bias_grad (PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, UNSIGNED b_offset) := FUNCTION
		
			Num := map_a.matrix_cols;
			Num_1 := 1/Num;
			
			Layout_Part col_sum_tran(Layout_Part the_part):= TRANSFORM
				SELF.mat_part := sum_col_alpha (the_part.part_rows, the_part.part_cols, the_part.mat_part, Num_1);
				part_id := the_part.block_row + b_offset;
				real_part :=  the_part.block_row;
				SELF. partition_id := part_id;
				SELF.node_id := part_id-1;
				SELF.block_row    := real_part;
				SELF.block_col    := 1;
				SELF.first_row    := map_b.first_row(real_part);
				SELF.part_rows    := map_b.part_rows(real_part);
				SELF.first_col    := map_b.first_col(real_part);
				SELF.part_cols    := map_b.part_cols(real_part);
				SELF := the_part;
			END;
			col_sum_part := PROJECT (A_in, col_sum_tran(LEFT),LOCAL);
			col_sum_part_dist := DISTRIBUTE (col_sum_part, node_id);
			Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := le.part_rows ;
				SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1);
				SELF := le;
			END;
			rslt := ROLLUP(col_sum_part_dist,LEFT.partition_id = RIGHT.partition_id, addup(LEFT, RIGHT), LOCAL); // this form of rollup avoid grouping, ROLLUP(col_sum_part,addup(LEFT, RIGHT, partiion_id)) cause all the records with the same partiion_id
			//get grouped to one node which cause a lot of overload. The current rollup form avoidds grouping and improves performance

			RETURN rslt;
		END;//END colmean_bias_grad

		colmean_bias2_grad (PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, UNSIGNED b_offset) := FUNCTION
		
			Num := map_a.matrix_cols;
			Num_1 := 1/Num;
			
			Layout_Part col_sum_tran(Layout_Part the_part):= TRANSFORM
				SELF.mat_part := sum_col_alpha (the_part.part_rows, the_part.part_cols, the_part.mat_part, Num_1);
				part_id := the_part.block_row + b_offset;
				real_part :=  the_part.block_row;
				SELF. partition_id := part_id;
				SELF.node_id := b_offset;
				SELF.block_row    := real_part;
				SELF.block_col    := 1;
				SELF.first_row    := map_b.first_row(real_part);
				SELF.part_rows    := map_b.part_rows(real_part);
				SELF.first_col    := map_b.first_col(real_part);
				SELF.part_cols    := map_b.part_cols(real_part);
				SELF := the_part;
			END;
			col_sum_part := PROJECT (A_in, col_sum_tran(LEFT),LOCAL);
			col_sum_part_dist := DISTRIBUTE (col_sum_part, node_id);
			Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := le.part_rows ;
				SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1);
				SELF := le;
			END;
			col_sum_part_dist_sort := SORT (col_sum_part_dist, partition_id, LOCAL);
			rslt := ROLLUP(col_sum_part_dist_sort,LEFT.partition_id = RIGHT.partition_id, addup(LEFT, RIGHT), LOCAL);
			//append all the partitions to one partition
			Layout_Part com(Layout_Part le, Layout_Part ri) := TRANSFORM
				SELF.mat_part := (le.mat_part + ri.mat_part);
				SELF.partition_id := b_offset + 1;
				SELF.block_row := 1;
				SELF.first_row := 1;
				SELF.part_rows := le.part_rows + ri.part_rows;
				SELF := le;
			END;
			rslt2 := ROLLUP(rslt,LEFT.node_id = RIGHT.node_id, com(LEFT, RIGHT), LOCAL, ORDERED);//make sure the order of appendance is correct
			RETURN rslt2;
		END;//END colmean_bias2_grad

		colmean_a2 (PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b) := FUNCTION
		
			Num := map_a.matrix_cols;
			Num_1 := 1/Num;
			
			Layout_Part col_sum_tran(Layout_Part the_part):= TRANSFORM
				SELF.mat_part := sum_col_alpha (the_part.part_rows, the_part.part_cols, the_part.mat_part, Num_1);
				part_id := the_part.block_row;
				real_part :=  part_id;
				SELF. partition_id := part_id;
				SELF.node_id := part_id-1;
				SELF.block_row    := real_part;
				SELF.block_col    := 1;
				SELF.first_row    := map_b.first_row(real_part);
				SELF.part_rows    := map_b.part_rows(real_part);
				SELF.first_col    := map_b.first_col(real_part);
				SELF.part_cols    := map_b.part_cols(real_part);
				SELF := the_part;
			END;
			col_sum_part := PROJECT (A_in, col_sum_tran(LEFT),LOCAL);
			col_sum_part_dist := DISTRIBUTE (col_sum_part, node_id);
			Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := le.part_rows ;
				SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1);
				SELF := le;
			END;
			rslt := ROLLUP(col_sum_part,TRUE, addup(LEFT, RIGHT)); // this form of rollup avoid grouping,ROLLUP(col_sum_part,LEFT.partition_id = RIGHT.partition_id, addup(LEFT, RIGHT)); and  ROLLUP(col_sum_part,addup(LEFT, RIGHT, partiion_id)) cause all the records with the same partiion_id
			//get grouped to one node which cause a lot of overload. The current rollup form avoidds grouping and improves performance

			RETURN rslt;
		END;//END colmean_a2
		
		
		// This function gets two big matrices which are distributed over all nodes and generate a final relatively smaller matrix which is on one node
		// this is used for weight gradient calculation where for example a h*m matrix is multiplied by a m*f matrix. PBblas will distribute all partitions in the first and second matrix to only one node which final matrix is in
		// this causes overhead, to avoid that we multiply each col partition of first matrix with a row partition of the second matrix in each node, the final generated matrices are added up to generate the final matrix
		// this way, we don't change the distribution of the first and second matrices
		big_big_small(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_c, PBblas.Types.value_t alph=1.0) := FUNCTION
			SET OF PBblas.Types.value_t empty_array := [];
				Layout_Part mul(Layout_Part a_part, Layout_Part b_part):=TRANSFORM
					part_id     := 1;    //arbitrary choice
					part_a_cols := a_part.part_cols;
					part_a_rows := a_part.part_rows;
					part_b_rows := b_part.part_rows;
					part_b_cols := b_part.part_cols;
					k := part_a_cols;
					SELF.partition_id := part_id;
					SELF.node_id      := 0;
					SELF.block_row    := 1;
					SELF.block_col    := 1;
					SELF.first_row    := map_c.first_row(part_id);
					SELF.part_rows    := map_c.part_rows(part_id);
					SELF.first_col    := map_c.first_col(part_id);
					SELF.part_cols    := map_c.part_cols(part_id);
					SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, TRUE,
																					part_a_rows, part_b_rows, k,
																					alph, a_part.mat_part, b_part.mat_part,
																					0.0, empty_array);
			END;
			mul_part := JOIN (A_in, B_in, LEFT.partition_id = RIGHT.partition_id, mul(LEFT, RIGHT), LOCAL );
			
			Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := le.part_rows * le.part_cols ;
				SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1);
				SELF := le;
			END;
			
			Layout_Part addup_it(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := ri.part_rows * ri.part_cols ;
				SELF.mat_part := IF (le.partition_id=0, ri.mat_part, PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1));
				SELF := ri;
			END;
			//rslt := ROLLUP(mul_part, addup(LEFT, RIGHT), partition_id); // since the results of rohat is used in a ALL join, no need to distribute this to node one to be consistent with PBblas
			//rslt := ITERATE(mul_part, addup_it(LEFT, RIGHT));// using rollup cause the graph to Group all the records which are distributed between all node to only one record and then do the operation, It takes a long time to GROUP all thoese partitions in one node and we avoid it by using ITERATE instead of ROLLUP

      rslt := ROLLUP(mul_part, TRUE, addup(LEFT, RIGHT));
			final_rslt := DISTRIBUTE (rslt, node_id); 

// a_part := A_in[1];
// b_part := B_in[1];
		 // RETURN PBblas.BLAS.dgemm(FALSE, TRUE,
																					// a_part.part_rows, b_part.part_rows, a_part.part_cols,
																					// 1.0, a_part.mat_part, b_part.mat_part,
																					// 0.0, empty_array);
																					
			RETURN final_rslt;
		END;// END big_big_small
		//calculates coef * (d2*x')

		d2Xt( DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_c, REAL8 coef) := FUNCTION
			SET OF PBblas.Types.value_t empty_array := [];
			B_in := ddist;
				Layout_Part mul(Layout_Part a_part, Layout_Part b_part):=TRANSFORM
					part_id     := b_part.block_row;
					part_a_cols := a_part.part_cols;
					part_a_rows := a_part.part_rows;
					part_b_rows := b_part.part_rows;
					part_b_cols := b_part.part_cols;
					k := part_a_cols;
					SELF.partition_id := part_id;
					SELF.node_id      := part_id-1;
					SELF.block_row    := a_part.block_row;
					SELF.block_col    := b_part.block_row;
					SELF.first_row    := map_c.first_row(part_id);
					SELF.part_rows    := map_c.part_rows(part_id);
					SELF.first_col    := map_c.first_col(part_id);
					SELF.part_cols    := map_c.part_cols(part_id);
					SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, TRUE,
																					part_a_rows, part_b_rows, k,
																					coef, a_part.mat_part, b_part.mat_part,
																					0.0, empty_array);
			END;

			mul_part := JOIN (A_in, B_in, LEFT.block_col = RIGHT.block_col, mul(LEFT, RIGHT), LOCAL );
			
			Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := le.part_rows * le.part_cols ;
				SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1);
				SELF := le;
			END;
			mul_part_dist := DISTRIBUTE (mul_part, node_id);
			mul_part_dist_sorted := SORT (mul_part_dist, partition_id, LOCAL);
      rslt := ROLLUP(mul_part_dist_sorted, LEFT.partition_id = RIGHT.partition_id, addup(LEFT, RIGHT), LOCAL);																			
			RETURN rslt;
		END;// END d2Xt
		
		d3a2t(DATASET(Layout_Part) A_in, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_c, UNSIGNED B_offset, REAL8 coef) := FUNCTION
			SET OF PBblas.Types.value_t empty_array := [];
				Layout_Part mul(Layout_Part a_part, Layout_Part b_part):=TRANSFORM
					
					part_id := a_part.block_row;
					new_part_id     := part_id + B_offset;
					part_a_cols := a_part.part_cols;
					part_a_rows := a_part.part_rows;
					part_b_rows := b_part.part_rows;
					part_b_cols := b_part.part_cols;
					k := part_a_cols;
					SELF.partition_id := new_part_id;
					SELF.node_id      := new_part_id-1;
					SELF.block_row    := a_part.block_row;
					SELF.block_col    := b_part.block_row;
					SELF.first_row    := map_c.first_row(part_id);
					SELF.part_rows    := map_c.part_rows(part_id);
					SELF.first_col    := map_c.first_col(part_id);
					SELF.part_cols    := map_c.part_cols(part_id);
					SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, TRUE,
																					part_a_rows, part_b_rows, k,
																					coef, a_part.mat_part, b_part.mat_part,
																					0.0, empty_array);
			END;
			mul_part := JOIN (A_in, B_in, LEFT.block_col = RIGHT.block_col, mul(LEFT, RIGHT), LOCAL );
			
			Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := le.part_rows * le.part_cols ;
				SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1);
				SELF := le;
			END;
			

			mul_part_dist := DISTRIBUTE (mul_part, node_id);
			mul_part_dist_sorted := SORT (mul_part_dist, partition_id, LOCAL);
      rslt := ROLLUP(mul_part_dist_sorted, LEFT.partition_id = RIGHT.partition_id, addup(LEFT, RIGHT), LOCAL);
																					
			RETURN rslt;
		END;// END d3a2t
		
		
		
		
		//extract sparse autoencoder parameters
   

    //FF2 returns a2
    FF2_(DATASET(Layout_Part) w1, DATASET(Layout_Part) b1v):= FUNCTION
      //b1m = repmat(b1v,1,m)
      b1m := PBblas.PB_dgemm(FALSE, TRUE, 1.0,b1vecmap, b1v, Ones_VecMap, Ones_Vecdist, b1map);
      //z2 = w1*X+b1;
      //z2 := PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1, dmap, ddist, b1map, b1m, 1.0); // gives MP closed error
      z2_ := PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1, dmap, ddist, b1map);
      z2 := PBblas.PB_daxpy(1.0, z2_, b1m);
      //a2 = sigmoid (z2);
      a2 := PBblas.Apply2Elements(b1map, z2, sigmoid);
      RETURN a2;
    END;//END FF2_
		
		
		
		//returns a2
		 // FF2(DATASET(Layout_Part) w1, DATASET(Layout_Part) b1v):= FUNCTION
      // z2=w1*x+repmat(b1,1,m)
			// z2 := WX_repmatb(w1map, w1, dmap, ddist, b1vecmap, b1v, b1map, 0.0);
      // a2 = sigmoid (z2);
      // a2 := PBblas.Apply2Elements(b1map, z2, sigmoid);
      // RETURN a2;
     // END;//END FF2
		 
		 
		
		
		 
    //FF3 returns a3
    FF3_(DATASET(Layout_Part) w2,DATASET(Layout_Part) b2v, DATASET(Layout_Part) a2 ):= FUNCTION
      //b2m = repmat(b2v,1,m)
      b2m := PBblas.PB_dgemm(FALSE, TRUE, 1.0,b2vecmap, b2v, Ones_VecMap, Ones_Vecdist, b2map);
      //z3 = w2*a2+b2;
      //z3 := PBblas.PB_dgemm(FALSE, FALSE,1.0,w2map, w2, a2map, a2, b2map,b2m, 1.0); //gives MP closed error
      z3_ := PBblas.PB_dgemm(FALSE, FALSE,1.0,w2map, w2, a2map, a2, b2map);
      z3 := PBblas.PB_daxpy(1.0, z3_, b2m);
      //a3 = sigmoid (z3)
      a3 := PBblas.Apply2Elements(b2map, z3, sigmoid);
      RETURN a3;
    END;//END FF3_
		
		 //FF3 returns a3
    // FF3(DATASET(Layout_Part) w2,DATASET(Layout_Part) b2v, DATASET(Layout_Part) a2 ):= FUNCTION
		  // z3 = w2*a2 + repmat(b2,1,m)
			// z3 := WX_repmatb(w2map, w2, a2map, a2, b2vecmap, b2v, b2map, 0.0);
      // a3 = sigmoid (z3)
      // a3 := PBblas.Apply2Elements(b2map, z3, sigmoid);
      // RETURN a3;
    // END;//END FF3
		
	 FF3(DATASET(Layout_Part) w2,DATASET(Layout_Part) b2v, DATASET(Layout_Part) a2 ):= FUNCTION
		  //z3 = w2*a2 + repmat(b2,1,m)
			//z3 := WX_repmatb(w2map, w2, a2map, a2, b2vecmap, b2v, b2map, 0.0);
			//z3 := W2a2_repmatb2(w2map, w2, a2map, a2, b2vecmap, b2v, b2map, 0.0);
			z3 := W2a2_repmatb2(  a2,  b2map);
      //a3 = sigmoid (z3)
      a3 := PBblas.Apply2Elements(b2map, z3, sigmoid);
      RETURN z3;
    END;//END FF3
		

    //DELTA3 returns d3
    DELTA3 (DATASET(Layout_Part) a3 ) := FUNCTION
      //calculate delta for the last layer (3rd layer)
      //y=X;
      //d3=-(y-a3).*(a3.*(1-a3));
      // siggrad_a3 := PBblas.Apply2Elements(a3map, a3, siggrad);
      // a3_y := PBblas.PB_daxpy(-1, ddist, a3);
      // d3 := PBblas.HadamardProduct(a3map, a3_y, siggrad_a3);
			Layout_part d3_tran (Layout_part a3_part, Layout_part y_part) := TRANSFORM
				SELF.mat_part := d3_cal(a3_part.part_rows * a3_part.part_cols, a3_part.mat_part, y_part.mat_part);
				SELF := a3_part;
			END;
			d3 := JOIN (a3, ddist,LEFT.partition_id = RIGHT.partition_id, d3_tran(LEFT, RIGHT), LOCAL);


      RETURN d3 ;
    END;//END DELTA3
		
		DELTA3_ (DATASET(Layout_Part) a3 ) := FUNCTION
      //calculate delta for the last layer (3rd layer)
      //y=X;
      //d3=-(y-a3).*(a3.*(1-a3));
      siggrad_a3 := PBblas.Apply2Elements(a3map, a3, siggrad);
      a3_y := PBblas.PB_daxpy(-1, ddist, a3);
      d3 := PBblas.HadamardProduct(a3map, a3_y, siggrad_a3);
      RETURN d3 ;
    END;//END DELTA3_
    //DELTA2 retunrs d2
    rohat_ (DATASET(Layout_Part) a2) := FUNCTION
      rh := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a2map, a2, Ones_VecMap, Ones_Vecdist, Hiddmap);
      RETURN rh;
    END;
		rohat (DATASET(Layout_Part) a2) := FUNCTION
			//rh := col_mean(a2map, a2, Ones_VecMap, Ones_Vecdist, Hiddmap, 1.0);
			rh := colmean_a2 (a2map, a2, Hiddmap);
      RETURN rh;
    END;
    DELTA2_ (DATASET(Layout_Part) w2, DATASET(Layout_Part) a2, DATASET(Layout_Part) d3, DATASET(Layout_Part) rhat) := FUNCTION
      //calculate delta for 2nd layer
      //rhohat=mean(a2,2);
      //sparsity_delta=((-sparsityParam./rhohat)+((1-sparsityParam)./(1.-rhohat)));
      //d2=((W2'*d3)+beta*repmat(sparsity_delta,1,m)) .* (a2.*(1-a2));
      //rhohat := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a2map, a2, Ones_VecMap, Ones_Vecdist, Hiddmap);
      rhohat := rhat;
      sparsity_delta := PBblas.Apply2Elements(Hiddmap, rhohat, sp_delta);
      siggrad_a2 := PBblas.Apply2Elements(a2map, a2, siggrad);
      repmat_sparsity_delta := PBblas.PB_dgemm(FALSE, TRUE, 1.0,  Hiddmap, sparsity_delta, Ones_VecMap, Ones_Vecdist, a2map);
      //d2_firstterm = (W2'*d3)+beta*repmat(sparsity_delta,1,m);
      //d2_firstterm := PBblas.PB_dgemm(TRUE, FALSE, 1.0, w2map, w2, a3map, d3, a2map, repmat_sparsity_delta, BETA); // MP close error
      d2_firstterm_ := PBblas.PB_dgemm(TRUE, FALSE, 1.0, w2map, w2, a3map, d3, a2map);
      d2_firstterm := PBblas.PB_daxpy(BETA, repmat_sparsity_delta, d2_firstterm_);
      d2 := PBblas.HadamardProduct(a2map, d2_firstterm, siggrad_a2);
      RETURN d2 ;
    END;
		
		
		DELTA2 (DATASET(Layout_Part) w2, DATASET(Layout_Part) a2, DATASET(Layout_Part) d3, DATASET(Layout_Part) rhat) := FUNCTION
      //calculate delta for 2nd layer
      //rhohat=mean(a2,2);
      //sparsity_delta=((-sparsityParam./rhohat)+((1-sparsityParam)./(1.-rhohat)));
      //d2=((W2'*d3)+beta*repmat(sparsity_delta,1,m)) .* (a2.*(1-a2));
      //rhohat := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a2map, a2, Ones_VecMap, Ones_Vecdist, Hiddmap);
      rhohat := rhat;
      sparsity_delta := PBblas.Apply2Elements(Hiddmap, rhohat, sp_delta);
      // siggrad_a2 := PBblas.Apply2Elements(a2map, a2, siggrad);
      //repmat_sparsity_delta := PBblas.PB_dgemm(FALSE, TRUE, 1.0,  Hiddmap, sparsity_delta, Ones_VecMap, Ones_Vecdist, a2map);
      //d2_firstterm = (W2'*d3)+beta*repmat(sparsity_delta,1,m);
			//d2_firstterm := WtX_repmatb(w2map, w2, a3map, d3, Hiddmap, sparsity_delta, a2map, BETA);

		//first distribute rohat on all the nodes that ab_prod is distributed on
		a2map_nodesused := a2map.nodes_used;
		Layout_Part_newnode norm_rohat (Layout_Part te, INTEGER co) := TRANSFORM
			SELF.new_node_id := co % a2map_nodesused  ;
			SELF:= te;
		END;
		rohat_norm := NORMALIZE(rhohat, a2map_nodesused, norm_rohat(LEFT, COUNTER) );
		rohat_dist := DISTRIBUTE (rohat_norm, new_node_id);


			d2_firstterm := W2td3_repmatsparsity( d3, rohat_dist, a2map, BETA);
			Layout_part d2_tran (Layout_part a2_part, Layout_part d2_part) := TRANSFORM
				SELF.mat_part := d2_cal(a2_part.part_rows * a2_part.part_cols, a2_part.mat_part, d2_part.mat_part);
				SELF := a2_part;
			END;
			d2 := JOIN (a2, d2_firstterm, LEFT.partition_id = RIGHT.partition_id, d2_tran(LEFT, RIGHT), LOCAL);
      RETURN d2;
    END;
    //WeightGrad1 returns gradient for w1
    WeightGrad1_ (DATASET(Layout_Part) w1, DATASET(Layout_Part) d2) := FUNCTION
      w1_g := PBblas.PB_dgemm(FALSE, TRUE, m_1, a2map, d2, dmap, ddist, w1map, w1 ,LAMBDA );
      RETURN w1_g;
    END;
		WeightGrad1 (DATASET(Layout_Part) w1, DATASET(Layout_Part) d2) := FUNCTION
			//w1_g_ := big_big_small(a2map, d2, dmap, ddist, w1map, m_1);
			w1_g_ := d2Xt( d2, w1map, m_1);
			w1_g  := PBblas.PB_daxpy(LAMBDA, w1, w1_g_);
      RETURN w1_g;
    END;
		
    //WeightGrad2 returns gradient for w2
    WeightGrad2_ (DATASET(Layout_Part) w2, DATASET(Layout_Part) d3, DATASET(Layout_Part) a2) := FUNCTION
      w2_g := PBblas.PB_dgemm(FALSE, TRUE, m_1, a3map, d3, a2map, a2, w2map, w2 ,LAMBDA );
      RETURN w2_g;
    END;
		
		WeightGrad2 (DATASET(Layout_Part) w2, DATASET(Layout_Part) d3, DATASET(Layout_Part) a2) := FUNCTION
			//w2_g_ := big_big_small(a3map, d3, a2map, a2, w2map, m_1);
			w2_g_ := d3a2t(d3, a2, w2map, w1_partitions, m_1);
			w2_g  := PBblas.PB_daxpy(LAMBDA, w2, w2_g_);
      RETURN w2_g;
    END;
    //BiasGrad1 calculates the bias gradients for b1
    BiasGrad1 (DATASET(Layout_Part) d2) := FUNCTION
			//b1_g := col_mean(a2map, d2, Ones_VecMap, Ones_Vecdist, b1vecmap, m_1);
			b1_g := colmean_bias_grad (a2map, d2, b1vecmap, w1_partitions + w2_partitions);
      RETURN b1_g;
    END;
		
		BiasGrad1_ (DATASET(Layout_Part) d2) := FUNCTION
      b1_g := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a2map, d2, Ones_VecMap, Ones_Vecdist, b1vecmap);
      RETURN b1_g;
    END;
    //BiasGrad2 calculates the bias gradients for b2
    BiasGrad2_ (DATASET(Layout_Part) d3) := FUNCTION
      b2_g := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a3map, d3, Ones_VecMap, Ones_Vecdist, b2vecmap);
      RETURN b2_g;
    END;
		
		BiasGrad2 (DATASET(Layout_Part) d3) := FUNCTION
      //b2_g := col_mean(a3map, d3, Ones_VecMap, Ones_Vecdist, b2vecmap, m_1);
			b2_g := colmean_bias2_grad (a3map, d3, b2vecmap, w1_partitions + w2_partitions + b1_partitions);
      RETURN b2_g;
    END;
		
		
    emptyL := DATASET([], Layout_Part);
    //theta is the input weight and bias parameters for the sparseautoencoder
    //parameters are the parameters for the sparseautoencoder function
    //train_d is the train data to learn sparse autoencoder and calculate the gradient and the cost based on taht
    // train_l is empty in the sparse autoencoder (because it is an unsupervised learning)
    SparseParam_CostGradients4 :=  FUNCTION
      PBblas.Types.MUElement minuspart(Layout_Part l, UNSIGNED8 c ) := TRANSFORM
        SELF.partition_id := l.partition_id - c;
        SElF.no := 1;
        SELF := l;
      END;
      w1m := w1dist;
      w2m := w2dist;
      b1v := b1vecdist;
      b2v := b2vecdist;
      a2 := W1X_repmatb1(b1map);
			a2_ := FF2_ (w1m, b1v);
      a3 := W2a2_repmatb2(a2,  b2map);
			a3_ := FF3_ (w2m, b2v, a2_);
			Layout_part d3_tran (Layout_part a3_part, Layout_part y_part) := TRANSFORM
				SELF.mat_part := d3_cal(a3_part.part_rows * a3_part.part_cols, a3_part.mat_part, y_part.mat_part);
				SELF := a3_part;
			END;
			d3 := JOIN (a3, ddist,LEFT.partition_id = RIGHT.partition_id, d3_tran(LEFT, RIGHT), LOCAL);
			d3_ := DELTA3_ (a3_);
      rohat_a2 := colmean_a2 (a2map, a2, Hiddmap);
			rohat_a2_ := rohat_(a2_);
			rhohat := rohat_a2;
      a2map_nodesused := a2map.nodes_used;
			Layout_Part_newnode norm_rohat (Layout_Part te, INTEGER co) := TRANSFORM
				SELF.new_node_id := co % a2map_nodesused  ;
				SELF:= te;
			END;
			rohat_norm := NORMALIZE(rhohat, a2map_nodesused, norm_rohat(LEFT, COUNTER) );
			rohat_dist := DISTRIBUTE (rohat_norm, new_node_id);
			d2_firstterm := W2td3_repmatsparsity( d3, rohat_dist, a2map, BETA);
			Layout_part d2_tran (Layout_part a2_part, Layout_part d2_part) := TRANSFORM
				SELF.mat_part := d2_cal(a2_part.part_rows * a2_part.part_cols, a2_part.mat_part, d2_part.mat_part);
				SELF := a2_part;
			END;
			d2 := JOIN (a2, d2_firstterm, LEFT.partition_id = RIGHT.partition_id, d2_tran(LEFT, RIGHT), LOCAL);
			d2_ := DELTA2_ (w2m, a2_, d3_,rohat_a2_);
			w1_g_ := d2Xt( d2, w1map, m_1);
			wg1  := PBblas.PB_daxpy(LAMBDA, theta(partition_id <= W1_partitions), w1_g_);
      // wg1 := WeightGrad1 (w1m, d2);
			w2_g_ := d3a2t(d3, a2, w2map, w1_partitions, m_1);
			wg2  := PBblas.PB_daxpy(LAMBDA, theta(partition_id > W1_partitions AND partition_id <= W1_partitions + W2_partitions), w2_g_);
      // wg2 := WeightGrad2 (w2m, d3, a2);
      bg1 := BiasGrad1 (d2);
      bg2 := BiasGrad2 (d3);
			
			
			wg1_ := WeightGrad1_ (w1m, d2_);
      wg2_ := WeightGrad2_ (w2m, d3_, a2_);
      bg1_ := BiasGrad1_ (d2_);
      bg2_ := BiasGrad2_ (d3_);
      //calculate cost
      //PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1, dmap, ddist, b1map, b1m, 1.0);
      // squared_error_cost= 0.5*sum(sum((x-a3).^2));
      // cost=(1/m)*squared_error_cost+(lambda/2)*(sum(W2(:).^2)+sum(W1(:).^2))+beta*sum(KL(sparsityParam,rhohat));
			
			simple := {REAL8 v};
			simple SE_tran (Layout_Part le, Layout_Part ri) := TRANSFORM
				SELF.v := sum_pow2(le.part_cols * le.part_rows, le.mat_part, ri.mat_part);
			END;
			d_a3_sum_part := JOIN (a3, ddist, LEFT.partition_id = RIGHT.partition_id,SE_tran(LEFT,RIGHT), LOCAL );
      //squared_error_cost := 0.5*PBblas.SumElements(PBblas.Apply2Elements(dmap, PBblas.PB_daxpy(-1.0, a3, ddist), pow2));
			squared_error_cost := SUM (d_a3_sum_part, d_a3_sum_part.v);
      cost_term1 := (1/m)*squared_error_cost;
			
			simple term2_3_tran (Layout_Part le) := TRANSFORM
				SELF.v := sum_sq(le.part_cols * le.part_rows, le.mat_part);
			END;
			w1_term3 := PROJECT (w1m, term2_3_tran (LEFT), LOCAL);
			w2_term2 := PROJECT (w2m, term2_3_tran (LEFT), LOCAL);

      // cost_term2 := (lambda/2)* PBblas.SumElements(PBblas.Apply2Elements(w2map, w2m, pow2));
      //cost_term3 := (lambda/2)* PBblas.SumElements(PBblas.Apply2Elements(w1map, w1m, pow2));
			cost_term2 := (lambda/2)* SUM (w2_term2, w2_term2.v); 
			cost_term3 := (lambda/2)* SUM (w1_term3, w1_term3.v);
			
			simple kl_tran (Layout_Part le) := TRANSFORM
				SELF.v := sum_kl(le.part_cols * le.part_rows, le.mat_part, sparsityParam);
			END;
			
      PBblas.Types.value_t klterm(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := sparsityParam * LN(sparsityParam/v) + (1-sparsityParam) * LN ((1-sparsityParam)/(1-v));
      //KL := PBblas.Apply2Elements (Hiddmap,rohat_a2,klterm);
		  KL := PROJECT (rohat_a2, kl_tran(LEFT), LOCAL);
      //cost_term4 := beta * PBblas.SumElements(KL);
			cost_term4 := beta * SUM (KL, KL.v);
      cost := cost_term1 + cost_term2 + cost_term3 +cost_term4;    
      costField := DATASET ([{1,1,cost}],ML.Types.NumericField);
      one_map := PBblas.Matrix_Map(1,1,1,1);
      Cost_part_no := PBblas.MU.TO(ML.DMat.Converted.FromNumericFieldDS(costField,one_map),2);
      //convert w and b gradients to a datasets of layoutparts where partition_id differentiate them
      //w1_grad has partition_id from 1 to w1_partitions
      //w2_grad had partition_ds from w1_partitions+1 to w1_partitions + w2_partitions
      //b1_grad has partition_ids from w1_partitions + w2_partitions+1 to w1_partitions + w2_partitions+b1_partitions
      //b2_grad has partition_ids from w1_partitions + w2_partitions+b1_partitions+1 to w1_partitions + w2_partitions+b1_partitions+b2_partitions
      PBblas.Types.MUElement addpart(Layout_Part l, UNSIGNED8 c ) := TRANSFORM
        SELF.partition_id := l.partition_id + c;
        SElF.no := 1;
        SELF := l;
      END;
      wg1_reshape_no := Pbblas.MU.TO(wg1,1);
      wg2_reshape_no := PROJECT (wg2, addpart(LEFT, w1_partitions ), LOCAL);
			wg2_reshape_no_ := PROJECT (wg2_, addpart(LEFT, w1_partitions ), LOCAL);
      bg1_reshape_no := PROJECT (bg1, addpart (LEFT, w1_partitions + w2_partitions),LOCAL);
      bg2_reshape_no := PROJECT (bg2, addpart (LEFT, w1_partitions + w2_partitions + b1_partitions),LOCAL);
     // theta_Part_no := wg1_reshape_no + wg2_reshape_no + bg1_reshape_no + bg2_reshape_no;
      theta_Part_no := PROJECT (wg1 + wg2 + bg1 + bg2 ,TRANSFORM (PBblas.Types.MUElement, SELF.no :=1 ; SELF := LEFT) , LOCAL);
			//RETURN PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1dist, dmap, ddist, b1map);
			//RETURN WX(w1map, w1dist, dmap, ddist, b1map, w1dist,  1);
			//RETURN WX_repmatb(w1map, w1dist, dmap, ddist, b1vecmap, b1vecdist, b1map, 1.0) ;
			//RETURN col_sum(dmap, ddist, Ones_VecMap, Ones_Vecdist, b2vecmap);
		
			
			
			
			//mymy2 := DELTA2 (w2m, a2, d3,rohat_a2) + DELTA2_ (w2m, a2, d3,rohat_a2);
			//mymy2 := big_big_small(a2map, d2, dmap, ddist, w1map, 8);
			//mymy2 := col_mean(a3map, d3, Ones_VecMap, Ones_Vecdist, b2vecmap);
			mymy2 := wg1 + wg2  + bg1 + bg2;
myformat2 := RECORD
    mymy2.node_id;
    mymy2.partition_id;
    mymy2.block_row;
    mymy2.block_col;
    mymy2.first_row;
    mymy2.part_rows;
    mymy2.first_col;
    mymy2.part_cols;
		//mymy2.no;
		//mymy2.mat_part;
		INTEGER real_node := STD.System.Thorlib.Node();
END;
	thisR := TABLE(mymy2,myformat2,LOCAL); 
	theta_part_no_check :=  ASSERT(theta_Part_no, node_id=Thorlib.node() and node_id=(partition_id-1), 'sparse autoencoder gradient is not distributed correctly', FAIL);
	return_record := RECORD (Layout_Part)
		REAL8 cost_value;
	END;
	ToReturn := PROJECT (theta_part_no_check, TRANSFORM (return_record, SELF.cost_value := cost; SELF:=LEFT), LOCAL);
	//RETURN  theta_part_no_check + Cost_part_no;
	RETURN ToReturn;
	// RETURN PROJECT (theta_dist, TRANSFORM (return_record, SELF.cost_value := 2; SELF:=LEFT), LOCAL);
	// a2_part :=   W1X_repmatb1(w1map, w1dist, dmap, ddist, b1vecmap, b1v_, b1map, 0.0, 0);
	// a3_part :=  W2a2_repmatb2(w2map, w2dist, b1map, a2_part, b2vecmap, b2v_, b2map, 0.0);
	// rohat_part := col_mean(a2map, a2_part, Ones_VecMap, Ones_Vecdist, Hiddmap);
	//RETURN W2td3_repmatsparsity(w2map, w2dist, dmap, ddist, b1vecmap, b1v_, b1map, 0.0, 0);
	//RETURN W2td3_repmatsparsity(w2map, w2dist, dmap, ddist, Hiddmap, b1v, a2map, BETA);
	//RETURN cost_term4;

//RETURN d2;
			//RETURN thisR;
      //RETURN theta_Part_no;
    END;//END SparseParam_CostGradients4  
    RETURN SparseParam_CostGradients4;
   END;//END SA_lbfgs_Compatible2_param_part_onebias_distparam


EXPORT SA_lbfgs_Compatible2_param_part_biasonenode ( DATASET(Layout_Part) theta, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Part) TrainData , DATASET(Layout_Part) TrainLabel) := FUNCTION
   

// This function repeats the bias vector of size r*1 , s times in a columnwise format, so the final results will be a r*s matrix
//D = [1,2,3], r=3, s=2 => output=[1,2,3,1,2,3]
  SET OF REAL8 repeatbias(PBblas.Types.dimension_t r, PBblas.Types.dimension_t s, PBblas.Types.matrix_t D) := BEGINC++

    #body
    __lenResult = r * s * sizeof(double);
    __isAllResult = false;
    double * result = new double[r*s];
    __result = (void*) result;
    double *cell = (double*) d;
    uint32_t cells =  r * s;
    uint32_t i;
    uint32_t pos;
    for (i=0; i<cells; i++) {
      pos = i % r;
      result[i] = cell[pos];
    }
  ENDC++;
	
//this function calculates d3=-(y-a3).*(a3.*(1-a3));
//N is the total number of elements in each matrix
//A is the a3 matrix in SET format
//Y is the y matrix in SET format
	SET OF REAL8 d3_cal(PBblas.Types.dimension_t N, PBblas.Types.matrix_t A, PBblas.Types.matrix_t Y) := BEGINC++

    #body
    __lenResult = n * sizeof(double);
    __isAllResult = false;
    double * result = new double[n];
    __result = (void*) result;
    double *cella = (double*) a;
		double *celly = (double*) y;
    uint32_t cells =  n;
    uint32_t i;
    for (i=0; i<cells; i++) {
      result[i] = (cella[i]-celly[i])*(cella[i]*(1-cella[i]));
    }
  ENDC++;
	SET OF REAL8 d2_cal(PBblas.Types.dimension_t N, PBblas.Types.matrix_t A, PBblas.Types.matrix_t Y) := BEGINC++

    #body
    __lenResult = n * sizeof(double);
    __isAllResult = false;
    double * result = new double[n];
    __result = (void*) result;
    double *cella = (double*) a;
		double *celly = (double*) y;
    uint32_t cells =  n;
    uint32_t i;
    for (i=0; i<cells; i++) {
      result[i] = (celly[i])*(cella[i]*(1-cella[i]));
    }
  ENDC++;
//result = M + bet * repmat (V, 1, r)
	SET OF REAL8 mat_vec_sum(PBblas.Types.dimension_t N, PBblas.Types.dimension_t r, PBblas.Types.matrix_t M, PBblas.Types.matrix_t V, PBblas.Types.value_t bet) := BEGINC++

    #body
    __lenResult = n * sizeof(double);
    __isAllResult = false;
    double * result = new double[n];
    __result = (void*) result;
    double *cellm = (double*) m;
		double *cellv = (double*) v;
    uint32_t cells =  n;
    uint32_t i;
		uint32_t pos;
    for (i=0; i<cells; i++) {
		  pos = i % r;
      result[i] = cellm[i] + (bet * cellv[pos]);
    }
  ENDC++;
	// //result = sigmoid (M + repmat (V, 1, r))
	SET OF REAL8 mat_vec_sum_sigmoid(PBblas.Types.dimension_t N, PBblas.Types.dimension_t r, PBblas.Types.matrix_t M, PBblas.Types.matrix_t V) := BEGINC++

    #body
    __lenResult = n * sizeof(double);
    __isAllResult = false;
    double * result = new double[n];
    __result = (void*) result;
    double *cellm = (double*) m;
		double *cellv = (double*) v;
    uint32_t cells =  n;
    uint32_t i;
		uint32_t pos;
    for (i=0; i<cells; i++) {
		  pos = i % r;
      result[i] = 1/(1 + exp(-1*(cellm[i] + cellv[pos])));
    }
  ENDC++;
	
	
	SET OF REAL8 sum_col (PBblas.Types.dimension_t r, PBblas.Types.dimension_t s, PBblas.Types.matrix_t D) := BEGINC++

    #body
    __lenResult = r * sizeof(double);
    __isAllResult = false;
    double * result = new double[r];
    __result = (void*) result;
    double *cell = (double*) d;
    uint32_t cells =  r * s;
    uint32_t i;
    uint32_t pos;
		for (i=0; i<r; i++) {
      result[i] = 0;
    }
    for (i=0; i<cells; i++) {
      pos = i % r;
      result[pos] = result[pos] + cell[i];
    }

  ENDC++;
	
		SET OF REAL8 sum_col_alpha (PBblas.Types.dimension_t r, PBblas.Types.dimension_t s, PBblas.Types.matrix_t D, REAL8 thisalpha) := BEGINC++

    #body
    __lenResult = r * sizeof(double);
    __isAllResult = false;
    double * result = new double[r];
    __result = (void*) result;
    double *cell = (double*) d;
    uint32_t cells =  r * s;
    uint32_t i;
    uint32_t pos;
		for (i=0; i<r; i++) {
      result[i] = 0;
    }
    for (i=0; i<cells; i++) {
      pos = i % r;
      result[pos] = result[pos] + cell[i];
    }
		for (i=0; i<r; i++) {
      result[i] = result[i] * thisalpha;
    }

  ENDC++;
	//0.5 * sum ((M-V).^2)
	REAL8 sum_pow2(PBblas.Types.dimension_t N, PBblas.Types.matrix_t M, PBblas.Types.matrix_t V) := BEGINC++

    #body
    double result = 0;
		double tmpp ;
    double *cellm = (double*) m;
		double *cellv = (double*) v;
    uint32_t i;
		for (i=0; i<n; i++) {
		  tmpp =(cellm[i] - cellv [i]);
      result = result + (tmpp*tmpp);
    }
		return(0.5*result);

  ENDC++;
	//sum(M.^2)
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
	// sum (kl(rho, M))
	REAL8 sum_kl(PBblas.Types.dimension_t N, PBblas.Types.matrix_t M, PBblas.Types.value_t rho) := BEGINC++

    #body
    double result = 0;
		double tmpp ;
    double *cellm = (double*) m;
    uint32_t i;
		for (i=0; i<n; i++) {
			result = result + (rho*log(rho/cellm[i])) + ((1-rho)*log((1-rho)/(1-cellm[i])));
    }
		return(result);

  ENDC++;
    ddist := TrainData;
    m := CostFunc_params(id=1)[1].value;
    num_feat := CostFunc_params(id=2)[1].value;
    num_hid := CostFunc_params(id=3)[1].value;
    part_rows := CostFunc_params(id=4)[1].value; // partition size for the features (number of rows)
    part_cols := CostFunc_params(id=5)[1].value; // partition size for the number of columns (samples) in the input data
    BETA := CostFunc_params(id=6)[1].value;
    sparsityParam := CostFunc_params(id=7)[1].value;
    LAMBDA := CostFunc_params(id=8)[1].value;
    
    m_1 := 1/m;
    sparsityParam_ := -1*sparsityParam;
    sparsityParam_1 := 1-sparsityParam;
    
    //Create map for block matrix d
    dmap := PBblas.Matrix_Map(num_feat,m,part_rows,part_cols);
    
    //Create block matrix Ytmp
    Ymap := dmap;
    Ydist := ddist;
    //Creat maps for block matrices for weights

    w1map := PBblas.Matrix_Map(num_hid, num_feat, num_hid, part_rows);
    w2map := PBblas.Matrix_Map(num_feat, num_hid, part_rows, num_hid);
     //each bias vector is converted to block format
    
    b1vecmap := PBblas.Matrix_Map(num_hid, 1, num_hid, 1);
    b2vecmap := PBblas.Matrix_Map(num_feat, 1, part_rows, 1);
    
    w1_partitions := w1map.partitions_used;
    w2_partitions := w2map.partitions_used;
    b1_partitions := b1vecmap.partitions_used;
    b2_partitions := b2vecmap.partitions_used;
    Layout_Part minuspart(Layout_Part l, UNSIGNED8 c ) := TRANSFORM
      SELF.partition_id := l.partition_id - c;
      SELF := l;
    END;
    //w1m := w1dist;
    w1dist := theta (partition_id <= w1_partitions);
    //w2m := w2dist;
    w2dist := theta (partition_id > w1_partitions AND partition_id <= w1_partitions + w2_partitions);
    //w2dist  := PROJECT (w2m_, minuspart(LEFT, w1_partitions ),LOCAL);
    //b1v := b1vecdist;
    b1vecdist := theta (partition_id > w1_partitions + w2_partitions AND partition_id <= w1_partitions + w2_partitions + b1_partitions);
    //b1vecdist  := PROJECT (b1v_, minuspart (LEFT, w1_partitions + w2_partitions),LOCAL);
    
    //b2v := b2vecdist;
    b2vecdist := theta (partition_id > w1_partitions + w2_partitions + b1_partitions AND partition_id <= w1_partitions + w2_partitions + b1_partitions + b2_partitions);
    //b2vecdist  := PROJECT (b2v_, minuspart (LEFT, w1_partitions + w2_partitions + b1_partitions),LOCAL);
    
     //functions used
    PBblas.Types.value_t sp_reci(PBblas.Types.value_t v,PBblas.Types.dimension_t r,PBblas.Types.dimension_t c) := sparsityParam_/v;
    PBblas.Types.value_t sp_1_reci(PBblas.Types.value_t v,PBblas.Types.dimension_t r,PBblas.Types.dimension_t c) := sparsityParam_1/(1-v);
    //sparsity_delta=((-sparsityParam./rhohat)+((1-sparsityParam)./(1.-rhohat)));
    PBblas.Types.value_t sp_delta(PBblas.Types.value_t v,PBblas.Types.dimension_t r,PBblas.Types.dimension_t c) := (sparsityParam_/v)+(sparsityParam_1/(1-v));
    PBblas.Types.value_t siggrad(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := v*(1.0-v);
    PBblas.Types.value_t sigmoid(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := 1/(1+exp(-1*v));
    PBblas.Types.value_t pow2(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := v*v;
    //maps used
    b1map := PBblas.Matrix_Map(num_hid, m, num_hid, part_cols);
    b2map := PBblas.Matrix_Map(num_feat, m, part_rows, part_cols);
    a2map := b1map;
    a3map := b2map;
    HL_nodes := num_hid;//number of nodes in the hidden layer
    Hiddmap := b1vecmap;
    //onevec for calculating rhohat	 
	 
	 Ones_VecMap := PBblas.Matrix_Map(m, 1, part_cols, 1);
    //New Vector Generator
    // Layout_Cell gen(UNSIGNED4 c, UNSIGNED4 NumRows) := TRANSFORM
      // SELF.x := ((c-1) % NumRows) + 1;
      // SELF.y := ((c-1) DIV NumRows) + 1;
      // SELF.v := 1;
    // END;
    //Create Ones Vector for the calculations in the step fucntion
    // Ones_Vec := DATASET(m, gen(COUNTER, m));
    Ones_Vecdist := TrainLabel;
Layout_Target := PBblas.Types.Layout_Target;




row_block_bigmat (UNSIGNED8 p, UNSIGNED8 nn) := FUNCTION // p is the partition number, nn is the number of rows
	RETURN ((p-1) % nn )+1;
END;

col_block_bigmat (UNSIGNED8 p, UNSIGNED8 nn) := FUNCTION // p is the partition number, nn is the number of rows
	RETURN ((p-1) DIV nn )+1;
END;

block_smallmat (UNSIGNED8 p, UNSIGNED8 offset) := FUNCTION 
	RETURN p-offset;
END;


//A_in := w1 is h*f where f is divided to partitions of size prow
//B_in := data : ddist
// bb_in := bias1
//returns the sigmoid of the result
W1X_repmatb1(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_bb, DATASET(Layout_Part) bb_in, PBblas.IMatrix_Map map_c, REAL8 beta=0.0, UNSIGNED8 A_offset) := FUNCTION
SET OF PBblas.Types.value_t empty_array := [];
// First check maps for compatability.  Normalize for transpose operations.
  a_matrix_rows := map_a.matrix_rows;
  a_matrix_cols := map_a.matrix_cols;
  a_row_blocks  := map_a.row_blocks;
  a_col_blocks  := map_a.col_blocks;
  b_matrix_rows := map_b.matrix_rows;
  b_matrix_cols := map_b.matrix_cols;
  b_row_blocks  := map_b.row_blocks;
  b_col_blocks  := map_b.col_blocks;
  c_matrix_rows := map_c.matrix_rows;
  c_matrix_cols := map_c.matrix_cols;
  c_row_blocks  := map_c.row_blocks;
  c_col_blocks  := map_c.col_blocks;
  A := ASSERT(A_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' + ' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(a_matrix_rows=c_matrix_rows AND a_row_blocks=c_row_blocks,
                    'A-C: ' + 'Arows: ' + a_matrix_rows +' Crows '+c_matrix_rows+' '+ 
                              'Ablocks: ' + a_row_blocks +' Cblocks '+c_row_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	B := ASSERT(B_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' +' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(b_matrix_cols=c_matrix_cols AND b_col_blocks=c_col_blocks,
                    'B-C: ' + 'Brows: ' + b_matrix_cols +' Ccols '+c_matrix_cols+' '+ 
                              'Bblocks: ' + b_row_blocks +' Cblocks '+c_col_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	
  //multiply
	B_row_part := map_b.row_blocks;
	C_row_part := map_c.row_blocks;
	Layout_Part mul2(Layout_Part b_part, Layout_Part a_part):=TRANSFORM
    //real_part_id := col_block_bigmat (b_part.partition_id, B_row_part);  //arbitrary choice
		REAL_part_id := b_part.block_col;
		part_id := real_part_id;
    part_a_cols := a_part.part_cols;
    part_a_rows := a_part.part_rows;
    part_b_rows := b_part.part_rows;
    part_c_rows := map_c.part_rows(part_id);
    part_c_cols := map_c.part_cols(part_id);
    part_c_first_row  := map_c.first_row(part_id);
    part_c_first_col  := map_c.first_col(part_id);
    k := part_a_cols;
    SELF.partition_id := part_id;
    SELF.node_id      := b_part.node_id;
    SELF.block_row    := a_part.block_row;
    SELF.block_col    := b_part.block_col;
    SELF.first_row    := map_c.first_row(part_id);
    SELF.part_rows    := part_c_rows;
    SELF.first_col    := part_c_first_col;
    SELF.part_cols    := part_c_cols;
    SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
                                    part_c_rows, part_c_cols, k,
                                    1.0, a_part.mat_part, b_part.mat_part,
                                    0.0, empty_array);
  END;
  //ab_prod_ := JOIN(B, A, row_block_bigmat (LEFT.partition_id, B_row_part) = RIGHT.partition_id , mul2(LEFT,RIGHT),ALL); // Each A (weight matrix) is copied in each B (X matrix) node
	ab_prod_ := JOIN(B, A, LEFT.block_row = RIGHT.block_col , mul2(LEFT,RIGHT),ALL); // Each A (weight matrix) is copied in each B (X matrix) node
	//ab_prod_ := JOIN(B, A, LEFT.block_row = RIGHT.block_col , mul2(LEFT,RIGHT),ALL); //this is not correct, there might be more than one block row in one node

	 // Sum terms
  Layout_Part sumTerms(Layout_Part cumm, Layout_Part term) := TRANSFORM
    N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
    SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term.mat_part, 1);
    SELF := cumm;
  END;
	sorted_ab_prod := SORT(ab_prod_, partition_id, LOCAL);
  ab_prod := ROLLUP(sorted_ab_prod, sumTerms(LEFT, RIGHT), partition_id, LOCAL);
//	 ab_prod := ROLLUP(sorted_ab_prod, LEFT.partition_id = RIGHT.partition_id, sumTerms(LEFT, RIGHT), LOCAL);

// add bias vector to each columns of X
// each bias vector (b) is copied (ALL JOIN) to each node of X. The bias vector is then normalized to repeat itself, number of columns of X time in that node and then the two are added
  Layout_Part addbias(Layout_Part cumm, Layout_Part term) := TRANSFORM
		cumm_part_cols := cumm.part_cols;// number of columns in this partition
    N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
		Elem := {PBblas.Types.value_t v};
		Elem_r := {PBblas.Types.value_t v, UNSIGNED r};
		elems := DATASET(term.mat_part, Elem);
		Elem_r rep (Elem l, UNSIGNED c) := TRANSFORM
			SELF.r := c;
			SELF := l;
		END;
		elems_rep := NORMALIZE(elems, cumm_part_cols, rep(LEFT, COUNTER));// each value in bias vector is repeated cumm_part_cols number of times (as the numebr of cumm columns in this node)
		elems_rep_sort := SORT(elems_rep, r, LOCAL);
		term_rep_set := SET (elems_rep_sort, v);
    SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term_rep_set, 1);
		SELF.partition_id := cumm.partition_id;
    SELF := cumm;
  END;
	
	Layout_Part addbias_(Layout_Part cumm, Layout_Part term) := TRANSFORM
		cumm_part_cols := cumm.part_cols;// number of columns in this partition
		term_part_rows := term.part_rows;//number of elements in this partition of the bias vector
    N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
		Elem := {PBblas.Types.value_t v};
		Elem_r := {PBblas.Types.value_t v, UNSIGNED r};
		elems := DATASET(term.mat_part, Elem);
		Elem_r rep (Elem l, UNSIGNED c) := TRANSFORM
			SELF.r := c;
			SELF := l;
		END;
		elems_rep := NORMALIZE(elems, cumm_part_cols, rep(LEFT, COUNTER));// each value in bias vector is repeated cumm_part_cols number of times (as the numebr of cumm columns in this node)
		elems_rep_sort := SORT(elems_rep, r, LOCAL);
		term_rep_set := repeatbias(term_part_rows, cumm_part_cols, term.mat_part);
    //SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term_rep_set, 1);
		SELF.mat_part := mat_vec_sum_sigmoid(N, term_part_rows, cumm.mat_part, term.mat_part);
		SELF.partition_id := cumm.partition_id;
    SELF := cumm;
  END;
	
	
	 ab_bb := JOIN(ab_prod, bb_in,TRUE , addbias_(LEFT,RIGHT),ALL);
	 
	 
	 	Layout_Part rep_b1(Layout_Part one_part, Layout_Part bb_part):=TRANSFORM
    real_part_id := one_part.partition_id;
		part_id := (real_part_id-1)*C_row_part + bb_part.block_row;
    part_a_cols := bb_part.part_cols;
    part_a_rows := bb_part.part_rows;
    part_b_rows := one_part.part_rows;
    part_c_rows := map_c.part_rows(part_id);
    part_c_cols := map_c.part_cols(part_id);
    part_c_first_row  := map_c.first_row(part_id);
    part_c_first_col  := map_c.first_col(part_id);
    k := part_a_cols;
    SELF.partition_id := part_id;
    SELF.node_id      := one_part.node_id;
    SELF.block_row    := bb_part.block_row;
    SELF.block_col    := one_part.block_col;
    SELF.first_row    := map_c.first_row(part_id);
    SELF.part_rows    := part_c_rows;
    SELF.first_col    := part_c_first_col;
    SELF.part_cols    := part_c_cols;
    SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, TRUE,
                                    part_c_rows, part_c_cols, k,
                                    1.0, bb_part.mat_part, one_part.mat_part,
                                    0.0, empty_array);
  END;

	bb_repeated := JOIN (Ones_Vecdist, bb_in, TRUE , rep_b1(LEFT,RIGHT),ALL);
		 //Ones_VecMap := PBblas.Matrix_Map(m, 1, Ones_Vecdist := TrainLabel;
	ab_bb_ := PBblas.PB_daxpy(1.0, ab_prod, bb_repeated);
	RETURN ab_bb;
END; // END W1X_repmatb1




W2td3_repmatsparsity(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_bb, DATASET(Layout_Part) bb_in, PBblas.IMatrix_Map map_c, REAL8 beta_in=0) := FUNCTION

SET OF PBblas.Types.value_t empty_array := [];
	// First check maps for compatability.  Normalize for transpose operations.
		a_matrix_rows := map_a.matrix_cols;
		a_matrix_cols := map_a.matrix_rows;
		a_row_blocks  := map_a.col_blocks;
		a_col_blocks  := map_a.row_blocks;
		b_matrix_rows := map_b.matrix_rows;
		b_matrix_cols := map_b.matrix_cols;
		b_row_blocks  := map_b.row_blocks;
		b_col_blocks  := map_b.col_blocks;
		c_matrix_rows := map_c.matrix_rows;
		c_matrix_cols := map_c.matrix_cols;
		c_row_blocks  := map_c.row_blocks;
		c_col_blocks  := map_c.col_blocks;
	A := ASSERT(A_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'YES' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'NO' + ' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(a_matrix_rows=c_matrix_rows AND a_row_blocks=c_row_blocks,
                    'A-C: ' + 'Arows: ' + a_matrix_rows +' Crows '+c_matrix_rows+' '+ 
                              'Ablocks: ' + a_row_blocks +' Cblocks '+c_row_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
  B := ASSERT(B_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'YES' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'NO' +' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(b_matrix_cols=c_matrix_cols AND b_col_blocks=c_col_blocks,
                    'B-C: ' + 'Brows: ' + b_matrix_cols +' Ccols '+c_matrix_cols+' '+ 
                              'Bblocks: ' + b_row_blocks +' Cblocks '+c_col_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	
  //multiply
	B_row_part := map_b.row_blocks;
	Layout_Part mul2(Layout_Part b_part, Layout_Part a_part):=TRANSFORM
    part_id := b_part.block_col;
    part_a_cols := a_part.part_cols;
    part_a_rows := a_part.part_rows;
    part_b_rows := b_part.part_rows;
    part_c_rows := map_c.part_rows(part_id);
    part_c_cols := map_c.part_cols(part_id);
    part_c_first_row  := map_c.first_row(part_id);
    part_c_first_col  := map_c.first_col(part_id);
    k := part_a_rows;
    SELF.partition_id := part_id;
    SELF.node_id      := b_part.node_id;
    SELF.block_row    := a_part.block_col;
    SELF.block_col    := b_part.block_col;
    SELF.first_row    := map_c.first_row(part_id);
    SELF.part_rows    := part_c_rows;
    SELF.first_col    := part_c_first_col;
    SELF.part_cols    := part_c_cols;
    SELF.mat_part     := PBblas.BLAS.dgemm(TRUE, FALSE,
                                    part_c_rows, part_c_cols, k,
                                    1.0, a_part.mat_part, b_part.mat_part,
                                    0.0, empty_array);

  END;
  //ab_prod_ := JOIN(B, A, row_block_bigmat (LEFT.partition_id, B_row_part) = RIGHT.partition_id , mul2(LEFT,RIGHT),ALL); // Each A (weight matrix) is copied in each B (X matrix) node
	//ab_prod_ := JOIN(B, A, row_block_bigmat (LEFT.partition_id, B_row_part) = (RIGHT.partition_id-A_offset) , mul2(LEFT,RIGHT),ALL); // Each A (weight matrix) is copied in each B (X matrix) node
	ab_prod_ := JOIN(B, A, LEFT.block_row = RIGHT.block_row, mul2(LEFT,RIGHT),ALL);

	 // Sum terms
  Layout_Part sumTerms(Layout_Part cumm, Layout_Part term) := TRANSFORM
    N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
    SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term.mat_part, 1);
    SELF := cumm;
  END;
	sorted_ab_prod := SORT(ab_prod_, partition_id, LOCAL);
  ab_prod := ROLLUP(sorted_ab_prod, sumTerms(LEFT, RIGHT), partition_id, LOCAL);

Layout_Part addbias(Layout_Part cumm, Layout_Part term) := TRANSFORM
		cumm_part_cols := cumm.part_cols;// number of columns in this partition
		term_part_rows := term.part_rows;//number of elements in this partition of the bias vector
    N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
		SELF.mat_part := mat_vec_sum(N, term_part_rows, cumm.mat_part, term.mat_part, beta_in);
		SELF.partition_id := cumm.partition_id;
    SELF := cumm;
  END;
	
	
	 ab_bb := JOIN(ab_prod, bb_in,TRUE , addbias(LEFT,RIGHT),ALL);
	RETURN ab_bb;
END; // END W2td3_repmatsparsity







//returns the sigmoid of the result
W2a2_repmatb2(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_bb, DATASET(Layout_Part) bb_in, PBblas.IMatrix_Map map_c, REAL8 beta=0.0) := FUNCTION
SET OF PBblas.Types.value_t empty_array := [];
// First check maps for compatability.  Normalize for transpose operations.
  a_matrix_rows := map_a.matrix_rows;
  a_matrix_cols := map_a.matrix_cols;
  a_row_blocks  := map_a.row_blocks;
  a_col_blocks  := map_a.col_blocks;
  b_matrix_rows := map_b.matrix_rows;
  b_matrix_cols := map_b.matrix_cols;
  b_row_blocks  := map_b.row_blocks;
  b_col_blocks  := map_b.col_blocks;
  c_matrix_rows := map_c.matrix_rows;
  c_matrix_cols := map_c.matrix_cols;
  c_row_blocks  := map_c.row_blocks;
  c_col_blocks  := map_c.col_blocks;
  A := ASSERT(A_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' + ' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(a_matrix_rows=c_matrix_rows AND a_row_blocks=c_row_blocks,
                    'A-C: ' + 'Arows: ' + a_matrix_rows +' Crows '+c_matrix_rows+' '+ 
                              'Ablocks: ' + a_row_blocks +' Cblocks '+c_row_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	B := ASSERT(B_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' +' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(b_matrix_cols=c_matrix_cols AND b_col_blocks=c_col_blocks,
                    'B-C: ' + 'Brows: ' + b_matrix_cols +' Ccols '+c_matrix_cols+' '+ 
                              'Bblocks: ' + b_row_blocks +' Cblocks '+c_col_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	
  //multiply
	B_row_part := map_b.row_blocks;
	C_row_part := map_c.row_blocks;
	Layout_Part mul2(Layout_Part b_part, Layout_Part a_part):=TRANSFORM
    real_part_id := b_part.partition_id;  //arbitrary choice
		part_id := (real_part_id-1)*C_row_part + a_part.block_row;
    part_a_cols := a_part.part_cols;
    part_a_rows := a_part.part_rows;
    part_b_rows := b_part.part_rows;
    part_c_rows := map_c.part_rows(part_id);
    part_c_cols := map_c.part_cols(part_id);
    part_c_first_row  := map_c.first_row(part_id);
    part_c_first_col  := map_c.first_col(part_id);
    k := part_a_cols;
    SELF.partition_id := part_id;
    SELF.node_id      := b_part.node_id;
    SELF.block_row    := a_part.block_row;
    SELF.block_col    := b_part.block_col;
    SELF.first_row    := map_c.first_row(part_id);
    SELF.part_rows    := part_c_rows;
    SELF.first_col    := part_c_first_col;
    SELF.part_cols    := part_c_cols;
    SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
                                    part_c_rows, part_c_cols, k,
                                    1.0, a_part.mat_part, b_part.mat_part,
                                    0.0, empty_array);
  END;
  //ab_prod := JOIN(B, A, TRUE , mul2(LEFT,RIGHT),ALL); // Each A (weight matrix) is copied in each B (X matrix) node
	ab_prod := JOIN(B, A, TRUE , mul2(LEFT,RIGHT), ALL); // Each A (weight matrix) is copied in each B (X matrix) node


// add bias vector to each columns of X
// each bias vector (b) is copied (ALL JOIN) to each node of X. The bias vector is then normalized to repeat it to the number of columns of X in that node and then the two are added
  Layout_Part addbias(Layout_Part cumm, Layout_Part term) := TRANSFORM
		cumm_part_cols := cumm.part_cols;// number of columns in this partition
    N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
		Elem := {PBblas.Types.value_t v};
		Elem_r := {PBblas.Types.value_t v, UNSIGNED r};
		elems := DATASET(term.mat_part, Elem);
		Elem_r rep (Elem l, UNSIGNED c) := TRANSFORM
			SELF.r := c;
			SELF := l;
		END;
		elems_rep := NORMALIZE(elems, cumm_part_cols, rep(LEFT, COUNTER));// each value in bias vector is repeated cumm_part_cols number of times (as the numebr of cumm columns in this node)
		elems_rep_sort := SORT(elems_rep, r, LOCAL);
		term_rep_set := SET (elems_rep_sort, v);
    SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term_rep_set, 1);
		SELF.partition_id := cumm.partition_id;
    SELF := cumm;
  END;
	
	Layout_Part addbias_(Layout_Part cumm, Layout_Part term) := TRANSFORM
		cumm_part_cols := cumm.part_cols;// number of columns in this partition
		term_part_rows := term.part_rows;
    N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
		Elem := {PBblas.Types.value_t v};
		Elem_r := {PBblas.Types.value_t v, UNSIGNED r};
		elems := DATASET(term.mat_part, Elem);
		Elem_r rep (Elem l, UNSIGNED c) := TRANSFORM
			SELF.r := c;
			SELF := l;
		END;
		elems_rep := NORMALIZE(elems, cumm_part_cols, rep(LEFT, COUNTER));// each value in bias vector is repeated cumm_part_cols number of times (as the numebr of cumm columns in this node)
		elems_rep_sort := SORT(elems_rep, r, LOCAL);
		term_rep_set := repeatbias(term_part_rows, cumm_part_cols, term.mat_part);
    //SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term_rep_set, 1);
		SELF.mat_part := mat_vec_sum_sigmoid(N, term_part_rows, cumm.mat_part, term.mat_part);
		SELF.partition_id := cumm.partition_id;
    SELF := cumm;
  END;
	
	 ab_bb := JOIN(ab_prod, bb_in,LEFT.block_row = RIGHT.block_row , addbias_(LEFT,RIGHT),LOOKUP);
	 
	 	Layout_Part rep_b2(Layout_Part one_part, Layout_Part bb_part):=TRANSFORM
    real_part_id := one_part.partition_id;
		part_id := (real_part_id-1)*C_row_part + bb_part.block_row;
    part_a_cols := bb_part.part_cols;
    part_a_rows := bb_part.part_rows;
    part_b_rows := one_part.part_rows;
    part_c_rows := map_c.part_rows(part_id);
    part_c_cols := map_c.part_cols(part_id);
    part_c_first_row  := map_c.first_row(part_id);
    part_c_first_col  := map_c.first_col(part_id);
    k := part_a_cols;
    SELF.partition_id := part_id;
    SELF.node_id      := one_part.node_id;
    SELF.block_row    := bb_part.block_row;
    SELF.block_col    := one_part.block_col;
    SELF.first_row    := map_c.first_row(part_id);
    SELF.part_rows    := part_c_rows;
    SELF.first_col    := part_c_first_col;
    SELF.part_cols    := part_c_cols;
    SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, TRUE,
                                    part_c_rows, part_c_cols, k,
                                    1.0, bb_part.mat_part, one_part.mat_part,
                                    0.0, empty_array);
  END;

	bb_repeated := JOIN (Ones_Vecdist, bb_in, TRUE , rep_b2(LEFT,RIGHT),ALL);
		 //Ones_VecMap := PBblas.Matrix_Map(m, 1, Ones_Vecdist := TrainLabel;
	ab_bb_ := PBblas.PB_daxpy(1.0, ab_prod, bb_repeated);
	//rslt := A;
  //RETURN PROJECT (B, TRANSFORM (layout_part, SELF.partition_id := row_block_bigmat (LEFT.partition_id, B_row_part), SELF:= LEFT));
	RETURN ab_bb;
END; // END W2a2_repmatb2

  //retunrs the sigmoid(WX+b)  
WX_repmatb_sig(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_bb, DATASET(Layout_Part) bb_in, PBblas.IMatrix_Map map_c, REAL8 beta=0.0) := FUNCTION
SET OF PBblas.Types.value_t empty_array := [];
// First check maps for compatability.  Normalize for transpose operations.
  a_matrix_rows := map_a.matrix_rows;
  a_matrix_cols := map_a.matrix_cols;
  a_row_blocks  := map_a.row_blocks;
  a_col_blocks  := map_a.col_blocks;
  b_matrix_rows := map_b.matrix_rows;
  b_matrix_cols := map_b.matrix_cols;
  b_row_blocks  := map_b.row_blocks;
  b_col_blocks  := map_b.col_blocks;
  c_matrix_rows := map_c.matrix_rows;
  c_matrix_cols := map_c.matrix_cols;
  c_row_blocks  := map_c.row_blocks;
  c_col_blocks  := map_c.col_blocks;
  A := ASSERT(A_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' + ' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(a_matrix_rows=c_matrix_rows AND a_row_blocks=c_row_blocks,
                    'A-C: ' + 'Arows: ' + a_matrix_rows +' Crows '+c_matrix_rows+' '+ 
                              'Ablocks: ' + a_row_blocks +' Cblocks '+c_row_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	B := ASSERT(B_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' +' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(b_matrix_cols=c_matrix_cols AND b_col_blocks=c_col_blocks,
                    'B-C: ' + 'Brows: ' + b_matrix_cols +' Ccols '+c_matrix_cols+' '+ 
                              'Bblocks: ' + b_row_blocks +' Cblocks '+c_col_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	
  //multiply
	
	Layout_Part mul2(Layout_Part b_part, Layout_Part a_part):=TRANSFORM
    part_id     := b_part.partition_id;    //arbitrary choice
    part_a_cols := a_part.part_cols;
    part_a_rows := a_part.part_rows;
    part_b_rows := b_part.part_rows;
    part_c_rows := map_c.part_rows(part_id);
    part_c_cols := map_c.part_cols(part_id);
    part_c_first_row  := map_c.first_row(part_id);
    part_c_first_col  := map_c.first_col(part_id);
    k := part_a_cols;
    SELF.partition_id := b_part.partition_id;
    SELF.node_id      := b_part.node_id;
    SELF.block_row    := b_part.block_row;
    SELF.block_col    := b_part.block_col;
    SELF.first_row    := map_c.first_row(part_id);
    SELF.part_rows    := part_c_rows;
    SELF.first_col    := part_c_first_col;
    SELF.part_cols    := part_c_cols;
    SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
                                    part_c_rows, part_c_cols, k,
                                    1.0, a_part.mat_part, b_part.mat_part,
                                    0.0, empty_array);
  END;
  ab_prod := JOIN(B, A,TRUE , mul2(LEFT,RIGHT),ALL); // Each A (weight matrix) is copied in each B (X matrix) node

// add bias vector to each columns of X
// each bias vector (b) is copied (ALL JOIN) to each node of X. The bias vector is then normalized to repeat it to the number of columns of X in that node and then the two are added
  Layout_Part addbias(Layout_Part cumm, Layout_Part term) := TRANSFORM
		cumm_part_cols := cumm.part_cols;
    N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
		Elem := {PBblas.Types.value_t v};
		Elem_r := {PBblas.Types.value_t v, UNSIGNED r};
		elems := DATASET(term.mat_part, Elem);
		Elem_r rep (Elem l, UNSIGNED c) := TRANSFORM
			SELF.r := c;
			SELF := l;
		END;
		elems_rep := NORMALIZE(elems, cumm_part_cols, rep(LEFT, COUNTER));// each value in bias vector is repeated cumm_part_cols number of times (as the numebr of cumm columns in this node)
		elems_rep_sort := SORT(elems_rep, r);
		term_rep_set := SET (elems_rep_sort, v);
    SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term_rep_set, 1);
		SELF.partition_id := cumm.partition_id;
    SELF := cumm;
  END;
	
	 ab_bb := JOIN(ab_prod, bb_in,TRUE , addbias(LEFT,RIGHT),ALL);
	//rslt := A;
  RETURN ab_bb;
END; // END WX_repmatb_sig
		
		

		
		
		
		

WtX_repmatb(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_bb, DATASET(Layout_Part) bb_in, PBblas.IMatrix_Map map_c, REAL8 beta=0.0) := FUNCTION
	SET OF PBblas.Types.value_t empty_array := [];
	// First check maps for compatability.  Normalize for transpose operations.
		a_matrix_rows := map_a.matrix_cols;
		a_matrix_cols := map_a.matrix_rows;
		a_row_blocks  := map_a.col_blocks;
		a_col_blocks  := map_a.row_blocks;
		b_matrix_rows := map_b.matrix_rows;
		b_matrix_cols := map_b.matrix_cols;
		b_row_blocks  := map_b.row_blocks;
		b_col_blocks  := map_b.col_blocks;
		c_matrix_rows := map_c.matrix_rows;
		c_matrix_cols := map_c.matrix_cols;
		c_row_blocks  := map_c.row_blocks;
		c_col_blocks  := map_c.col_blocks;
		A := ASSERT(A_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'YES' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'NO' + ' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(a_matrix_rows=c_matrix_rows AND a_row_blocks=c_row_blocks,
                    'A-C: ' + 'Arows: ' + a_matrix_rows +' Crows '+c_matrix_rows+' '+ 
                              'Ablocks: ' + a_row_blocks +' Cblocks '+c_row_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
  B := ASSERT(B_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'YES' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'NO' +' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(b_matrix_cols=c_matrix_cols AND b_col_blocks=c_col_blocks,
                    'B-C: ' + 'Brows: ' + b_matrix_cols +' Ccols '+c_matrix_cols+' '+ 
                              'Bblocks: ' + b_row_blocks +' Cblocks '+c_col_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
		
		//multiply
		
		Layout_Part mul2(Layout_Part b_part, Layout_Part a_part):=TRANSFORM
			part_id     := b_part.partition_id;    //arbitrary choice
			part_a_cols := a_part.part_cols;
			part_a_rows := a_part.part_rows;
			part_b_rows := b_part.part_rows;
			part_c_rows := map_c.part_rows(part_id);
			part_c_cols := map_c.part_cols(part_id);
			part_c_first_row  := map_c.first_row(part_id);
			part_c_first_col  := map_c.first_col(part_id);
			k := part_a_rows;
			SELF.partition_id := b_part.partition_id;
			SELF.node_id      := b_part.node_id;
			SELF.block_row    := b_part.block_row;
			SELF.block_col    := b_part.block_col;
			SELF.first_row    := map_c.first_row(part_id);
			SELF.part_rows    := part_c_rows;
			SELF.first_col    := part_c_first_col;
			SELF.part_cols    := part_c_cols;
			SELF.mat_part     := PBblas.BLAS.dgemm(TRUE, FALSE,
																			part_c_rows, part_c_cols, k,
																			1.0, a_part.mat_part, b_part.mat_part,
																			0.0, empty_array);
		END;
		ab_prod := JOIN(B, A,TRUE , mul2(LEFT,RIGHT),ALL); // Each A (weight matrix) is copied in each B (X matrix) node



// Apply beta
  Layout_Part applyBeta(Layout_Part part) := TRANSFORM
    SELF.mat_part := PBblas.BLAS.dscal(map_bb.matrix_rows*map_bb.matrix_cols,
                                beta, part.mat_part, 1);
    SELF:= part;
  END;
  bb_beta := PROJECT(bb_in, applyBeta(LEFT), LOCAL);
	// add the vector to each columns of X
	// each vector is copied (ALL JOIN) to each node of X. The vector is then normalized to repeat it to the number of columns of X in that node and then the two are added
		Layout_Part addvec(Layout_Part cumm, Layout_Part term) := TRANSFORM
			cumm_part_cols := cumm.part_cols;
			N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
			Elem := {PBblas.Types.value_t v};
			Elem_r := {PBblas.Types.value_t v, UNSIGNED r};
			elems := DATASET(term.mat_part, Elem);
			Elem_r rep (Elem l, UNSIGNED c) := TRANSFORM
				SELF.r := c;
				SELF := l;
			END;
			elems_rep := NORMALIZE(elems, cumm_part_cols, rep(LEFT, COUNTER));// each value in bias vector is repeated cumm_part_cols number of times (as the numebr of cumm columns in this node)
			elems_rep_sort := SORT(elems_rep, r);
			term_rep_set := SET (elems_rep_sort, v);
			SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term_rep_set, 1);
			SELF.partition_id := cumm.partition_id;
			SELF := cumm;
		END;
		
		 ab_bb := JOIN(ab_prod, bb_beta,TRUE , addvec(LEFT,RIGHT),ALL);

		//rslt := A;
		RETURN ab_bb;
END; // END WtX_repmatb
		
		
		
		// the input is a matrix in PBblas format where only columns are partitions
		//B_in : ones vector which is partitioned among nodes
		// map_c is the result's map
		col_sum(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_c) := FUNCTION
			SET OF PBblas.Types.value_t empty_array := [];
				Layout_Part mul(Layout_Part a_part, Layout_Part b_part):=TRANSFORM
					part_id     := 1;    //arbitrary choice
					part_a_cols := a_part.part_cols;
					part_a_rows := a_part.part_rows;
					part_b_rows := b_part.part_rows;
					part_c_rows := map_c.part_rows(part_id);
					part_c_cols := map_c.part_cols(part_id);
					part_c_first_row  := map_c.first_row(part_id);
					part_c_first_col  := map_c.first_col(part_id);
					k := part_a_cols;
					SELF.partition_id := 1;
					SELF.node_id      := a_part.node_id;
					SELF.block_row    := 1;
					SELF.block_col    := 1;
					SELF.first_row    := map_c.first_row(part_id);
					SELF.part_rows    := part_c_rows;
					SELF.first_col    := part_c_first_col;
					SELF.part_cols    := part_c_cols;
					SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
																					part_c_rows, part_c_cols, k,
																					1.0, a_part.mat_part, b_part.mat_part,
																					0.0, empty_array);
			END;
			col_sum_part := JOIN (A_in, B_in, LEFT.partition_id = RIGHT.partition_id, mul(LEFT, RIGHT), LOCAL );
			
			Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := le.part_rows ;
				SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1);
				SELF := le;
			END;
			//rslt := ROLLUP(col_sum_part, addup(LEFT, RIGHT), partition_id); // overload becasue of grouping
			rslt := ROLLUP(col_sum_part,TRUE, addup(LEFT, RIGHT)); // no groupijng, reduces overload
			//distribute to node one
			RETURN rslt; 
		END;//END Col_Sum
		
		
		col_mean(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_c, REAL8 mean_coef) := FUNCTION
		
			Num := map_a.matrix_cols;
			Num_1 := 1/Num;
			SET OF PBblas.Types.value_t empty_array := [];
				Layout_Part mul(Layout_Part a_part, Layout_Part b_part):=TRANSFORM
					part_id     := 1;    //arbitrary choice
					part_a_cols := a_part.part_cols;
					part_a_rows := a_part.part_rows;
					part_b_rows := b_part.part_rows;
					part_c_rows := map_c.part_rows(part_id);
					part_c_cols := map_c.part_cols(part_id);
					part_c_first_row  := map_c.first_row(part_id);
					part_c_first_col  := map_c.first_col(part_id);
					k := part_a_cols;
					SELF.partition_id := 1;
					SELF.node_id      := 0;
					SELF.block_row    := 1;
					SELF.block_col    := 1;
					SELF.first_row    := map_c.first_row(part_id);
					SELF.part_rows    := part_c_rows;
					SELF.first_col    := part_c_first_col;
					SELF.part_cols    := part_c_cols;
					SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
																					part_c_rows, part_c_cols, k,
																					Num_1, a_part.mat_part, b_part.mat_part,
																					0.0, empty_array);
			END;
			col_sum_part_ := JOIN (A_in, B_in, LEFT.partition_id = RIGHT.partition_id, mul(LEFT, RIGHT), LOCAL );
			
			
			Layout_Part col_sum_tran(Layout_Part the_part):= TRANSFORM
				SELF.mat_part := sum_col_alpha (the_part.part_rows, the_part.part_cols, the_part.mat_part, Num_1);
				SELF := the_part;
			END;
			col_sum_part := PROJECT (A_in, col_sum_tran(LEFT),LOCAL);
			Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := le.part_rows ;
				SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1);
				SELF := le;
			END;
			rslt := ROLLUP(col_sum_part,TRUE, addup(LEFT, RIGHT)); // this form of rollup avoid grouping, ROLLUP(col_sum_part,addup(LEFT, RIGHT, partiion_id)) cause all the records with the same partiion_id
			//get grouped to one node which cause a lot of overload. The current rollup form avoidds grouping and improves performance
			final_rslt := DISTRIBUTE (rslt, node_id); 
			//distribute to node one
			RETURN rslt;
		END;//END colmean
		//sum (A_in,2)
		colmean_bias_grad (PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, UNSIGNED b_offset) := FUNCTION
		
			Num := map_a.matrix_cols;
			Num_1 := 1/Num;
			
			Layout_Part col_sum_tran(Layout_Part the_part):= TRANSFORM
				SELF.mat_part := sum_col_alpha (the_part.part_rows, the_part.part_cols, the_part.mat_part, Num_1);
				part_id := the_part.block_row + b_offset;
				real_part :=  the_part.block_row;
				SELF. partition_id := part_id;
				SELF.node_id := part_id-1;
				SELF.block_row    := real_part;
				SELF.block_col    := 1;
				SELF.first_row    := map_b.first_row(real_part);
				SELF.part_rows    := map_b.part_rows(real_part);
				SELF.first_col    := map_b.first_col(real_part);
				SELF.part_cols    := map_b.part_cols(real_part);
				SELF := the_part;
			END;
			col_sum_part := PROJECT (A_in, col_sum_tran(LEFT),LOCAL);
			col_sum_part_dist := DISTRIBUTE (col_sum_part, node_id);
			Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := le.part_rows ;
				SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1);
				SELF := le;
			END;
			rslt := ROLLUP(col_sum_part_dist,LEFT.partition_id = RIGHT.partition_id, addup(LEFT, RIGHT), LOCAL); // this form of rollup avoid grouping, ROLLUP(col_sum_part,addup(LEFT, RIGHT, partiion_id)) cause all the records with the same partiion_id
			//get grouped to one node which cause a lot of overload. The current rollup form avoidds grouping and improves performance

			RETURN rslt;
		END;//END colmean_bias_grad

		colmean_bias_grad2 (PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, UNSIGNED b_offset) := FUNCTION
		
			Num := map_a.matrix_cols;
			Num_1 := 1/Num;
			
			Layout_Part col_sum_tran(Layout_Part the_part):= TRANSFORM
				SELF.mat_part := sum_col_alpha (the_part.part_rows, the_part.part_cols, the_part.mat_part, Num_1);
				part_id := the_part.block_row + b_offset;
				real_part :=  the_part.block_row;
				SELF. partition_id := part_id;
				SELF.node_id := w1_partitions+w2_partitions;
				SELF.block_row    := real_part;
				SELF.block_col    := 1;
				SELF.first_row    := map_b.first_row(real_part);
				SELF.part_rows    := map_b.part_rows(real_part);
				SELF.first_col    := map_b.first_col(real_part);
				SELF.part_cols    := map_b.part_cols(real_part);
				SELF := the_part;
			END;
			col_sum_part := PROJECT (A_in, col_sum_tran(LEFT),LOCAL);
			col_sum_part_dist := DISTRIBUTE (col_sum_part, node_id);
			Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := le.part_rows ;
				SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1);
				SELF := le;
			END;
			col_sum_part_dist_sorted := SORT(col_sum_part_dist, partition_id, LOCAL);
			rslt := ROLLUP(col_sum_part_dist_sorted, LEFT.partition_id = RIGHT.partition_id, addup(LEFT, RIGHT), LOCAL); // this form of rollup avoid grouping, ROLLUP(col_sum_part,addup(LEFT, RIGHT, partiion_id)) cause all the records with the same partiion_id
			//get grouped to one node which cause a lot of overload. The current rollup form avoidds grouping and improves performance

			RETURN rslt;
		END;//END colmean_bias_grad2
		
		
		// This function gets two big matrices which are distributed over all nodes and generate a final relatively smaller matrix which is on one node
		// this is used for weight gradient calculation where for example a h*m matrix is multiplied by a m*f matrix. PBblas will distribute all partitions in the first and second matrix to only one node which final matrix is in
		// this causes overhead, to avoid that we multiply each col partition of first matrix with a row partition of the second matrix in each node, the final generated matrices are added up to generate the final matrix
		// this way, we don't change the distribution of the first and second matrices
		big_big_small(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_c, PBblas.Types.value_t alph=1.0) := FUNCTION
			SET OF PBblas.Types.value_t empty_array := [];
				Layout_Part mul(Layout_Part a_part, Layout_Part b_part):=TRANSFORM
					part_id     := 1;    //arbitrary choice
					part_a_cols := a_part.part_cols;
					part_a_rows := a_part.part_rows;
					part_b_rows := b_part.part_rows;
					part_b_cols := b_part.part_cols;
					k := part_a_cols;
					SELF.partition_id := part_id;
					SELF.node_id      := 0;
					SELF.block_row    := 1;
					SELF.block_col    := 1;
					SELF.first_row    := map_c.first_row(part_id);
					SELF.part_rows    := map_c.part_rows(part_id);
					SELF.first_col    := map_c.first_col(part_id);
					SELF.part_cols    := map_c.part_cols(part_id);
					SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, TRUE,
																					part_a_rows, part_b_rows, k,
																					alph, a_part.mat_part, b_part.mat_part,
																					0.0, empty_array);
			END;
			mul_part := JOIN (A_in, B_in, LEFT.partition_id = RIGHT.partition_id, mul(LEFT, RIGHT), LOCAL );
			
			Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := le.part_rows * le.part_cols ;
				SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1);
				SELF := le;
			END;
			
			Layout_Part addup_it(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := ri.part_rows * ri.part_cols ;
				SELF.mat_part := IF (le.partition_id=0, ri.mat_part, PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1));
				SELF := ri;
			END;
			//rslt := ROLLUP(mul_part, addup(LEFT, RIGHT), partition_id); // since the results of rohat is used in a ALL join, no need to distribute this to node one to be consistent with PBblas
			//rslt := ITERATE(mul_part, addup_it(LEFT, RIGHT));// using rollup cause the graph to Group all the records which are distributed between all node to only one record and then do the operation, It takes a long time to GROUP all thoese partitions in one node and we avoid it by using ITERATE instead of ROLLUP

      rslt := ROLLUP(mul_part, TRUE, addup(LEFT, RIGHT));
			final_rslt := DISTRIBUTE (rslt, node_id); 

// a_part := A_in[1];
// b_part := B_in[1];
		 // RETURN PBblas.BLAS.dgemm(FALSE, TRUE,
																					// a_part.part_rows, b_part.part_rows, a_part.part_cols,
																					// 1.0, a_part.mat_part, b_part.mat_part,
																					// 0.0, empty_array);
																					
			RETURN final_rslt;
		END;// END big_big_small
		//calculates coef * (d2*x')
		d2Xt(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_c, REAL8 coef) := FUNCTION
			SET OF PBblas.Types.value_t empty_array := [];
				Layout_Part mul(Layout_Part a_part, Layout_Part b_part):=TRANSFORM
					part_id     := b_part.block_row;
					part_a_cols := a_part.part_cols;
					part_a_rows := a_part.part_rows;
					part_b_rows := b_part.part_rows;
					part_b_cols := b_part.part_cols;
					k := part_a_cols;
					SELF.partition_id := part_id;
					SELF.node_id      := part_id-1;
					SELF.block_row    := a_part.block_row;
					SELF.block_col    := b_part.block_row;
					SELF.first_row    := map_c.first_row(part_id);
					SELF.part_rows    := map_c.part_rows(part_id);
					SELF.first_col    := map_c.first_col(part_id);
					SELF.part_cols    := map_c.part_cols(part_id);
					SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, TRUE,
																					part_a_rows, part_b_rows, k,
																					coef, a_part.mat_part, b_part.mat_part,
																					0.0, empty_array);
			END;
			B_row_part := map_b.row_blocks;
			mul_part := JOIN (A_in, B_in, LEFT.block_col = RIGHT.block_col, mul(LEFT, RIGHT), LOCAL );
			
			Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := le.part_rows * le.part_cols ;
				SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1);
				SELF := le;
			END;
			mul_part_dist := DISTRIBUTE (mul_part, node_id);
			mul_part_dist_sorted := SORT (mul_part_dist, partition_id, LOCAL);
      rslt := ROLLUP(mul_part_dist_sorted, LEFT.partition_id = RIGHT.partition_id, addup(LEFT, RIGHT), LOCAL);
			// mul_part_sort := SORT (mul_part, partition_id);
			//rslt := ROLLUP(mul_part_sort, addup(LEFT, RIGHT), partition_id);
																					
			RETURN rslt;
		END;// END d2Xt
		
		d3a2t(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_c, UNSIGNED B_offset, REAL8 coef) := FUNCTION
			SET OF PBblas.Types.value_t empty_array := [];
				Layout_Part mul(Layout_Part a_part, Layout_Part b_part):=TRANSFORM
					
					part_id := a_part.block_row;
					new_part_id     := part_id + B_offset;
					part_a_cols := a_part.part_cols;
					part_a_rows := a_part.part_rows;
					part_b_rows := b_part.part_rows;
					part_b_cols := b_part.part_cols;
					k := part_a_cols;
					SELF.partition_id := new_part_id;
					SELF.node_id      := new_part_id-1;
					SELF.block_row    := a_part.block_row;
					SELF.block_col    := b_part.block_row;
					SELF.first_row    := map_c.first_row(part_id);
					SELF.part_rows    := map_c.part_rows(part_id);
					SELF.first_col    := map_c.first_col(part_id);
					SELF.part_cols    := map_c.part_cols(part_id);
					SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, TRUE,
																					part_a_rows, part_b_rows, k,
																					coef, a_part.mat_part, b_part.mat_part,
																					0.0, empty_array);
			END;
			A_row_part := map_a.row_blocks;
			mul_part := JOIN (A_in, B_in, LEFT.block_col = RIGHT.block_col, mul(LEFT, RIGHT), LOCAL );
			
			Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := le.part_rows * le.part_cols ;
				SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1);
				SELF := le;
			END;
			

			mul_part_dist := DISTRIBUTE (mul_part, node_id);
			mul_part_dist_sorted := SORT (mul_part_dist, partition_id, LOCAL);
      rslt := ROLLUP(mul_part_dist_sorted, LEFT.partition_id = RIGHT.partition_id, addup(LEFT, RIGHT), LOCAL);
																					
			RETURN rslt;
		END;// END d3a2t
		
		
		
		
		//extract sparse autoencoder parameters
   

    //FF2 returns a2
    FF2_(DATASET(Layout_Part) w1, DATASET(Layout_Part) b1v):= FUNCTION
      //b1m = repmat(b1v,1,m)
      b1m := PBblas.PB_dgemm(FALSE, TRUE, 1.0,b1vecmap, b1v, Ones_VecMap, Ones_Vecdist, b1map);
      //z2 = w1*X+b1;
      //z2 := PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1, dmap, ddist, b1map, b1m, 1.0); // gives MP closed error
      z2_ := PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1, dmap, ddist, b1map);
      z2 := PBblas.PB_daxpy(1.0, z2_, b1m);
      //a2 = sigmoid (z2);
      a2 := PBblas.Apply2Elements(b1map, z2, sigmoid);
      RETURN a2;
    END;//END FF2_
		
		
		
		//returns a2
		 // FF2(DATASET(Layout_Part) w1, DATASET(Layout_Part) b1v):= FUNCTION
      // z2=w1*x+repmat(b1,1,m)
			// z2 := WX_repmatb(w1map, w1, dmap, ddist, b1vecmap, b1v, b1map, 0.0);
      // a2 = sigmoid (z2);
      // a2 := PBblas.Apply2Elements(b1map, z2, sigmoid);
      // RETURN a2;
     // END;//END FF2
		 
		 
		
		FF2(DATASET(Layout_Part) w1, DATASET(Layout_Part) b1v):= FUNCTION
      //z2=w1*x+repmat(b1,1,m)
			//z2 := WX_repmatb(w1map, w1, dmap, ddist, b1vecmap, b1v, b1map, 0.0);
			a2 := W1X_repmatb1(w1map, w1, dmap, ddist, b1vecmap, b1v, b1map, 0.0, 0);
      //a2 = sigmoid (z2);
      //a2 := PBblas.Apply2Elements(b1map, z2, sigmoid);
      RETURN a2;
     END;//END FF2
		 
    //FF3 returns a3
    FF3_(DATASET(Layout_Part) w2,DATASET(Layout_Part) b2v, DATASET(Layout_Part) a2 ):= FUNCTION
      //b2m = repmat(b2v,1,m)
      b2m := PBblas.PB_dgemm(FALSE, TRUE, 1.0,b2vecmap, b2v, Ones_VecMap, Ones_Vecdist, b2map);
      //z3 = w2*a2+b2;
      //z3 := PBblas.PB_dgemm(FALSE, FALSE,1.0,w2map, w2, a2map, a2, b2map,b2m, 1.0); //gives MP closed error
      z3_ := PBblas.PB_dgemm(FALSE, FALSE,1.0,w2map, w2, a2map, a2, b2map);
      z3 := PBblas.PB_daxpy(1.0, z3_, b2m);
      //a3 = sigmoid (z3)
      a3 := PBblas.Apply2Elements(b2map, z3, sigmoid);
      RETURN a3;
    END;//END FF3_
		
		 //FF3 returns a3
    // FF3(DATASET(Layout_Part) w2,DATASET(Layout_Part) b2v, DATASET(Layout_Part) a2 ):= FUNCTION
		  // z3 = w2*a2 + repmat(b2,1,m)
			// z3 := WX_repmatb(w2map, w2, a2map, a2, b2vecmap, b2v, b2map, 0.0);
      // a3 = sigmoid (z3)
      // a3 := PBblas.Apply2Elements(b2map, z3, sigmoid);
      // RETURN a3;
    // END;//END FF3
		
	 FF3(DATASET(Layout_Part) w2,DATASET(Layout_Part) b2v, DATASET(Layout_Part) a2 ):= FUNCTION
		  //z3 = w2*a2 + repmat(b2,1,m)
			//z3 := WX_repmatb(w2map, w2, a2map, a2, b2vecmap, b2v, b2map, 0.0);
			z3 := W2a2_repmatb2(w2map, w2, a2map, a2, b2vecmap, b2v, b2map, 0.0);
      //a3 = sigmoid (z3)
      a3 := PBblas.Apply2Elements(b2map, z3, sigmoid);
      RETURN z3;
    END;//END FF3
		

    //DELTA3 returns d3
    DELTA3 (DATASET(Layout_Part) a3 ) := FUNCTION
      //calculate delta for the last layer (3rd layer)
      //y=X;
      //d3=-(y-a3).*(a3.*(1-a3));
      // siggrad_a3 := PBblas.Apply2Elements(a3map, a3, siggrad);
      // a3_y := PBblas.PB_daxpy(-1, ddist, a3);
      // d3 := PBblas.HadamardProduct(a3map, a3_y, siggrad_a3);
			Layout_part d3_tran (Layout_part a3_part, Layout_part y_part) := TRANSFORM
				SELF.mat_part := d3_cal(a3_part.part_rows * a3_part.part_cols, a3_part.mat_part, y_part.mat_part);
				SELF := a3_part;
			END;
			d3 := JOIN (a3, ddist,LEFT.partition_id = RIGHT.partition_id, d3_tran(LEFT, RIGHT), LOCAL);


      RETURN d3 ;
    END;//END DELTA3
		
		DELTA3_ (DATASET(Layout_Part) a3 ) := FUNCTION
      //calculate delta for the last layer (3rd layer)
      //y=X;
      //d3=-(y-a3).*(a3.*(1-a3));
      siggrad_a3 := PBblas.Apply2Elements(a3map, a3, siggrad);
      a3_y := PBblas.PB_daxpy(-1, ddist, a3);
      d3 := PBblas.HadamardProduct(a3map, a3_y, siggrad_a3);
      RETURN d3 ;
    END;//END DELTA3_
    //DELTA2 retunrs d2
    rohat_ (DATASET(Layout_Part) a2) := FUNCTION
      rh := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a2map, a2, Ones_VecMap, Ones_Vecdist, Hiddmap);
      RETURN rh;
    END;
		rohat (DATASET(Layout_Part) a2) := FUNCTION
			//rh := col_mean(a2map, a2, Ones_VecMap, Ones_Vecdist, Hiddmap, 1.0);
			rh := colmean_bias_grad (a2map, a2, Hiddmap, 0);
      RETURN rh;
    END;
    DELTA2_ (DATASET(Layout_Part) w2, DATASET(Layout_Part) a2, DATASET(Layout_Part) d3, DATASET(Layout_Part) rhat) := FUNCTION
      //calculate delta for 2nd layer
      //rhohat=mean(a2,2);
      //sparsity_delta=((-sparsityParam./rhohat)+((1-sparsityParam)./(1.-rhohat)));
      //d2=((W2'*d3)+beta*repmat(sparsity_delta,1,m)) .* (a2.*(1-a2));
      //rhohat := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a2map, a2, Ones_VecMap, Ones_Vecdist, Hiddmap);
      rhohat := rhat;
      sparsity_delta := PBblas.Apply2Elements(Hiddmap, rhohat, sp_delta);
      siggrad_a2 := PBblas.Apply2Elements(a2map, a2, siggrad);
      repmat_sparsity_delta := PBblas.PB_dgemm(FALSE, TRUE, 1.0,  Hiddmap, sparsity_delta, Ones_VecMap, Ones_Vecdist, a2map);
      //d2_firstterm = (W2'*d3)+beta*repmat(sparsity_delta,1,m);
      //d2_firstterm := PBblas.PB_dgemm(TRUE, FALSE, 1.0, w2map, w2, a3map, d3, a2map, repmat_sparsity_delta, BETA); // MP close error
      d2_firstterm_ := PBblas.PB_dgemm(TRUE, FALSE, 1.0, w2map, w2, a3map, d3, a2map);
      d2_firstterm := PBblas.PB_daxpy(BETA, repmat_sparsity_delta, d2_firstterm_);
      d2 := PBblas.HadamardProduct(a2map, d2_firstterm, siggrad_a2);
      RETURN d2 ;
    END;
		
		
		DELTA2 (DATASET(Layout_Part) w2, DATASET(Layout_Part) a2, DATASET(Layout_Part) d3, DATASET(Layout_Part) rhat) := FUNCTION
      //calculate delta for 2nd layer
      //rhohat=mean(a2,2);
      //sparsity_delta=((-sparsityParam./rhohat)+((1-sparsityParam)./(1.-rhohat)));
      //d2=((W2'*d3)+beta*repmat(sparsity_delta,1,m)) .* (a2.*(1-a2));
      //rhohat := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a2map, a2, Ones_VecMap, Ones_Vecdist, Hiddmap);
      rhohat := rhat;
      sparsity_delta := PBblas.Apply2Elements(Hiddmap, rhohat, sp_delta);
      // siggrad_a2 := PBblas.Apply2Elements(a2map, a2, siggrad);
      //repmat_sparsity_delta := PBblas.PB_dgemm(FALSE, TRUE, 1.0,  Hiddmap, sparsity_delta, Ones_VecMap, Ones_Vecdist, a2map);
      //d2_firstterm = (W2'*d3)+beta*repmat(sparsity_delta,1,m);
			//d2_firstterm := WtX_repmatb(w2map, w2, a3map, d3, Hiddmap, sparsity_delta, a2map, BETA);
			d2_firstterm := W2td3_repmatsparsity(w2map, w2, a3map, d3, Hiddmap, sparsity_delta, a2map, BETA);
      //d2 := PBblas.HadamardProduct(a2map, d2_firstterm, siggrad_a2);
			Layout_part d2_tran (Layout_part a2_part, Layout_part d2_part) := TRANSFORM
				SELF.mat_part := d2_cal(a2_part.part_rows * a2_part.part_cols, a2_part.mat_part, d2_part.mat_part);
				SELF := a2_part;
			END;
			d2 := JOIN (a2, d2_firstterm, LEFT.partition_id = RIGHT.partition_id, d2_tran(LEFT, RIGHT), LOCAL);
      RETURN d2;
    END;
    //WeightGrad1 returns gradient for w1
    WeightGrad1_ (DATASET(Layout_Part) w1, DATASET(Layout_Part) d2) := FUNCTION
      w1_g := PBblas.PB_dgemm(FALSE, TRUE, m_1, a2map, d2, dmap, ddist, w1map, w1 ,LAMBDA );
      RETURN w1_g;
    END;
		WeightGrad1 (DATASET(Layout_Part) w1, DATASET(Layout_Part) d2) := FUNCTION
			//w1_g_ := big_big_small(a2map, d2, dmap, ddist, w1map, m_1);
			w1_g_ := d2Xt(a2map, d2, dmap, ddist, w1map, m_1);
			w1_g  := PBblas.PB_daxpy(LAMBDA, w1, w1_g_);
      RETURN w1_g;
    END;
		
    //WeightGrad2 returns gradient for w2
    WeightGrad2_ (DATASET(Layout_Part) w2, DATASET(Layout_Part) d3, DATASET(Layout_Part) a2) := FUNCTION
      w2_g := PBblas.PB_dgemm(FALSE, TRUE, m_1, a3map, d3, a2map, a2, w2map, w2 ,LAMBDA );
      RETURN w2_g;
    END;
		
		WeightGrad2 (DATASET(Layout_Part) w2, DATASET(Layout_Part) d3, DATASET(Layout_Part) a2) := FUNCTION
			//w2_g_ := big_big_small(a3map, d3, a2map, a2, w2map, m_1);
			w2_g_ := d3a2t(a3map, d3, a2map, a2, w2map, w1_partitions, m_1);
			w2_g  := PBblas.PB_daxpy(LAMBDA, w2, w2_g_);
      RETURN w2_g;
    END;
    //BiasGrad1 calculates the bias gradients for b1
    BiasGrad1 (DATASET(Layout_Part) d2) := FUNCTION
			//b1_g := col_mean(a2map, d2, Ones_VecMap, Ones_Vecdist, b1vecmap, m_1);
			b1_g := colmean_bias_grad2 (a2map, d2, b1vecmap, w1_partitions + w2_partitions);
      RETURN b1_g;
    END;
		
		BiasGrad1_ (DATASET(Layout_Part) d2) := FUNCTION
      b1_g := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a2map, d2, Ones_VecMap, Ones_Vecdist, b1vecmap);
      RETURN b1_g;
    END;
    //BiasGrad2 calculates the bias gradients for b2
    BiasGrad2_ (DATASET(Layout_Part) d3) := FUNCTION
      b2_g := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a3map, d3, Ones_VecMap, Ones_Vecdist, b2vecmap);
      RETURN b2_g;
    END;
		
		BiasGrad2 (DATASET(Layout_Part) d3) := FUNCTION
      //b2_g := col_mean(a3map, d3, Ones_VecMap, Ones_Vecdist, b2vecmap, m_1);
			b2_g := colmean_bias_grad2 (a3map, d3, b2vecmap, w1_partitions + w2_partitions + b1_partitions);
      RETURN b2_g;
    END;
		
		
    emptyL := DATASET([], Layout_Part);
    //theta is the input weight and bias parameters for the sparseautoencoder
    //parameters are the parameters for the sparseautoencoder function
    //train_d is the train data to learn sparse autoencoder and calculate the gradient and the cost based on taht
    // train_l is empty in the sparse autoencoder (because it is an unsupervised learning)
    SparseParam_CostGradients4 :=  FUNCTION
      PBblas.Types.MUElement minuspart(Layout_Part l, UNSIGNED8 c ) := TRANSFORM
        SELF.partition_id := l.partition_id - c;
        SElF.no := 1;
        SELF := l;
      END;
      w1m := w1dist;
      w2m := w2dist;
      b1v := b1vecdist;
      b2v := b2vecdist;
      a2 := FF2 (w1m, b1v);
			a2_ := FF2_ (w1m, b1v);
      a3 := FF3 (w2m, b2v, a2);
			a3_ := FF3_ (w2m, b2v, a2_);
      d3 := DELTA3 (a3);
			d3_ := DELTA3_ (a3_);
      rohat_a2 := rohat(a2);
			rohat_a2_ := rohat_(a2_);
      d2 := DELTA2 (w2m, a2, d3,rohat_a2);
			d2_ := DELTA2_ (w2m, a2_, d3_,rohat_a2_);
      wg1 := WeightGrad1 (w1m, d2);
      wg2 := WeightGrad2 (w2m, d3, a2);
      bg1 := BiasGrad1 (d2);
      bg2 := BiasGrad2 (d3);
			
			
			wg1_ := WeightGrad1_ (w1m, d2_);
      wg2_ := WeightGrad2_ (w2m, d3_, a2_);
      bg1_ := BiasGrad1_ (d2_);
      bg2_ := BiasGrad2_ (d3_);
      //calculate cost
      //PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1, dmap, ddist, b1map, b1m, 1.0);
      // squared_error_cost= 0.5*sum(sum((x-a3).^2));
      // cost=(1/m)*squared_error_cost+(lambda/2)*(sum(W2(:).^2)+sum(W1(:).^2))+beta*sum(KL(sparsityParam,rhohat));
			
			simple := {REAL8 v};
			simple SE_tran (Layout_Part le, Layout_Part ri) := TRANSFORM
				SELF.v := sum_pow2(le.part_cols * le.part_rows, le.mat_part, ri.mat_part);
			END;
			d_a3_sum_part := JOIN (a3, ddist, LEFT.partition_id = RIGHT.partition_id,SE_tran(LEFT,RIGHT), LOCAL );
      //squared_error_cost := 0.5*PBblas.SumElements(PBblas.Apply2Elements(dmap, PBblas.PB_daxpy(-1.0, a3, ddist), pow2));
			squared_error_cost := SUM (d_a3_sum_part, d_a3_sum_part.v);
      cost_term1 := (1/m)*squared_error_cost;
			
			simple term2_3_tran (Layout_Part le) := TRANSFORM
				SELF.v := sum_sq(le.part_cols * le.part_rows, le.mat_part);
			END;
			w1_term3 := PROJECT (w1m, term2_3_tran (LEFT), LOCAL);
			w2_term2 := PROJECT (w2m, term2_3_tran (LEFT), LOCAL);

      // cost_term2 := (lambda/2)* PBblas.SumElements(PBblas.Apply2Elements(w2map, w2m, pow2));
      //cost_term3 := (lambda/2)* PBblas.SumElements(PBblas.Apply2Elements(w1map, w1m, pow2));
			cost_term2 := (lambda/2)* SUM (w2_term2, w2_term2.v); 
			cost_term3 := (lambda/2)* SUM (w1_term3, w1_term3.v);
			
			simple kl_tran (Layout_Part le) := TRANSFORM
				SELF.v := sum_kl(le.part_cols * le.part_rows, le.mat_part, sparsityParam);
			END;
			
      PBblas.Types.value_t klterm(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := sparsityParam * LN(sparsityParam/v) + (1-sparsityParam) * LN ((1-sparsityParam)/(1-v));
      //KL := PBblas.Apply2Elements (Hiddmap,rohat_a2,klterm);
		  KL := PROJECT (rohat_a2, kl_tran(LEFT), LOCAL);
      //cost_term4 := beta * PBblas.SumElements(KL);
			cost_term4 := beta * SUM (KL, KL.v);
      cost := cost_term1 + cost_term2 + cost_term3 +cost_term4;    
      costField := DATASET ([{1,1,cost}],ML.Types.NumericField);
      one_map := PBblas.Matrix_Map(1,1,1,1);
      Cost_part_no := PBblas.MU.TO(ML.DMat.Converted.FromNumericFieldDS(costField,one_map),2);
      //convert w and b gradients to a datasets of layoutparts where partition_id differentiate them
      //w1_grad has partition_id from 1 to w1_partitions
      //w2_grad had partition_ds from w1_partitions+1 to w1_partitions + w2_partitions
      //b1_grad has partition_ids from w1_partitions + w2_partitions+1 to w1_partitions + w2_partitions+b1_partitions
      //b2_grad has partition_ids from w1_partitions + w2_partitions+b1_partitions+1 to w1_partitions + w2_partitions+b1_partitions+b2_partitions
      PBblas.Types.MUElement addpart(Layout_Part l, UNSIGNED8 c ) := TRANSFORM
        SELF.partition_id := l.partition_id + c;
        SElF.no := 1;
        SELF := l;
      END;
      wg1_reshape_no := Pbblas.MU.TO(wg1,1);
      wg2_reshape_no := PROJECT (wg2, addpart(LEFT, w1_partitions ), LOCAL);
			wg2_reshape_no_ := PROJECT (wg2_, addpart(LEFT, w1_partitions ), LOCAL);
      bg1_reshape_no := PROJECT (bg1, addpart (LEFT, w1_partitions + w2_partitions),LOCAL);
      bg2_reshape_no := PROJECT (bg2, addpart (LEFT, w1_partitions + w2_partitions + b1_partitions),LOCAL);
     // theta_Part_no := wg1_reshape_no + wg2_reshape_no + bg1_reshape_no + bg2_reshape_no;
      theta_Part_no := PROJECT (wg1 + wg2 + bg1 + bg2 ,TRANSFORM (PBblas.Types.MUElement, SELF.no :=1 ; SELF := LEFT) , LOCAL);
			//RETURN PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1dist, dmap, ddist, b1map);
			//RETURN WX(w1map, w1dist, dmap, ddist, b1map, w1dist,  1);
			//RETURN WX_repmatb(w1map, w1dist, dmap, ddist, b1vecmap, b1vecdist, b1map, 1.0) ;
			//RETURN col_sum(dmap, ddist, Ones_VecMap, Ones_Vecdist, b2vecmap);
		
			
			
			
			//mymy2 := DELTA2 (w2m, a2, d3,rohat_a2) + DELTA2_ (w2m, a2, d3,rohat_a2);
			//mymy2 := big_big_small(a2map, d2, dmap, ddist, w1map, 8);
			//mymy2 := col_mean(a3map, d3, Ones_VecMap, Ones_Vecdist, b2vecmap);
			mymy2 := wg1 + wg2  + bg1 + bg2;
myformat2 := RECORD
    mymy2.node_id;
    mymy2.partition_id;
    mymy2.block_row;
    mymy2.block_col;
    mymy2.first_row;
    mymy2.part_rows;
    mymy2.first_col;
    mymy2.part_cols;
		//mymy2.no;
		//mymy2.mat_part;
		INTEGER real_node := STD.System.Thorlib.Node();
END;
	thisR := TABLE(mymy2,myformat2,LOCAL); 
	theta_part_no_check :=  ASSERT(theta_Part_no,(partition_id<=w1_partitions+w2_partitions and node_id=Thorlib.node() and node_id=(partition_id-1)) or (partition_id>w1_partitions+w2_partitions and node_id=Thorlib.node() and node_id = (w1_partitions+w2_partitions) ), 'sparse autoencoder gradient is not distributed correctly', FAIL);
	return_record := RECORD (Layout_Part)
		REAL8 cost_value;
	END;
	ToReturn := PROJECT (theta_part_no_check, TRANSFORM (return_record, SELF.cost_value := cost; SELF:=LEFT), LOCAL);
	//RETURN  theta_part_no_check + Cost_part_no;
	RETURN ToReturn;

    END;//END SparseParam_CostGradients4  
    RETURN SparseParam_CostGradients4;
   END;//END SA_lbfgs_Compatible2_param_part_biasonenode


//compute the cost and gradient on a minibatch of data which includes a mini batch of data distributed on one node
EXPORT SA_lbfgs_Compatible2_param_part_minibatch ( DATASET(Layout_Part) theta, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Part) TrainData , DATASET(Layout_Part) TrainLabel) := FUNCTION
   

// This function repeats the bias vector of size r*1 , s times in a columnwise format, so the final results will be a r*s matrix
//D = [1,2,3], r=3, s=2 => output=[1,2,3,1,2,3]
  SET OF REAL8 repeatbias(PBblas.Types.dimension_t r, PBblas.Types.dimension_t s, PBblas.Types.matrix_t D) := BEGINC++

    #body
    __lenResult = r * s * sizeof(double);
    __isAllResult = false;
    double * result = new double[r*s];
    __result = (void*) result;
    double *cell = (double*) d;
    uint32_t cells =  r * s;
    uint32_t i;
    uint32_t pos;
    for (i=0; i<cells; i++) {
      pos = i % r;
      result[i] = cell[pos];
    }
  ENDC++;
	
//this function calculates d3=-(y-a3).*(a3.*(1-a3));
//N is the total number of elements in each matrix
//A is the a3 matrix in SET format
//Y is the y matrix in SET format
	SET OF REAL8 d3_cal(PBblas.Types.dimension_t N, PBblas.Types.matrix_t A, PBblas.Types.matrix_t Y) := BEGINC++

    #body
    __lenResult = n * sizeof(double);
    __isAllResult = false;
    double * result = new double[n];
    __result = (void*) result;
    double *cella = (double*) a;
		double *celly = (double*) y;
    uint32_t cells =  n;
    uint32_t i;
    for (i=0; i<cells; i++) {
      result[i] = (cella[i]-celly[i])*(cella[i]*(1-cella[i]));
    }
  ENDC++;
	SET OF REAL8 d2_cal(PBblas.Types.dimension_t N, PBblas.Types.matrix_t A, PBblas.Types.matrix_t Y) := BEGINC++

    #body
    __lenResult = n * sizeof(double);
    __isAllResult = false;
    double * result = new double[n];
    __result = (void*) result;
    double *cella = (double*) a;
		double *celly = (double*) y;
    uint32_t cells =  n;
    uint32_t i;
    for (i=0; i<cells; i++) {
      result[i] = (celly[i])*(cella[i]*(1-cella[i]));
    }
  ENDC++;
//result = M + bet * repmat (V, 1, r)
	SET OF REAL8 mat_vec_sum(PBblas.Types.dimension_t N, PBblas.Types.dimension_t r, PBblas.Types.matrix_t M, PBblas.Types.matrix_t V, PBblas.Types.value_t bet) := BEGINC++

    #body
    __lenResult = n * sizeof(double);
    __isAllResult = false;
    double * result = new double[n];
    __result = (void*) result;
    double *cellm = (double*) m;
		double *cellv = (double*) v;
    uint32_t cells =  n;
    uint32_t i;
		uint32_t pos;
    for (i=0; i<cells; i++) {
		  pos = i % r;
      result[i] = cellm[i] + (bet * cellv[pos]);
    }
  ENDC++;
	
	SET OF REAL8 mat_vec_sum_sparsity(PBblas.Types.dimension_t N, PBblas.Types.dimension_t r, PBblas.Types.matrix_t M, PBblas.Types.matrix_t V, PBblas.Types.value_t bet, PBblas.Types.value_t sp) := BEGINC++

    #body
    __lenResult = n * sizeof(double);
    __isAllResult = false;
    double * result = new double[n];
    __result = (void*) result;
    double *cellm = (double*) m;
		double *cellv = (double*) v;
		double * v2 = new double[n];;
    uint32_t cells =  n;
    uint32_t i;
		uint32_t pos;
		double sp_ = -1*sp;
		double sp_1 = 1-sp;
		for (i=0; i<r; i++) {
      v2[i] = (sp_/cellv[i])+(sp_1/(1-cellv[i]));
    }
    for (i=0; i<cells; i++) {
		  pos = i % r;
      result[i] = cellm[i] + (bet * v2[pos]);
    }
  ENDC++;
	// //result = sigmoid (M + repmat (V, 1, r))
	SET OF REAL8 mat_vec_sum_sigmoid(PBblas.Types.dimension_t N, PBblas.Types.dimension_t r, PBblas.Types.matrix_t M, PBblas.Types.matrix_t V) := BEGINC++

    #body
    __lenResult = n * sizeof(double);
    __isAllResult = false;
    double * result = new double[n];
    __result = (void*) result;
    double *cellm = (double*) m;
		double *cellv = (double*) v;
    uint32_t cells =  n;
    uint32_t i;
		uint32_t pos;
    for (i=0; i<cells; i++) {
		  pos = i % r;
      result[i] = 1/(1 + exp(-1*(cellm[i] + cellv[pos])));
    }
  ENDC++;
	
	
	
	SET OF REAL8 sum_col (PBblas.Types.dimension_t r, PBblas.Types.dimension_t s, PBblas.Types.matrix_t D) := BEGINC++

    #body
    __lenResult = r * sizeof(double);
    __isAllResult = false;
    double * result = new double[r];
    __result = (void*) result;
    double *cell = (double*) d;
    uint32_t cells =  r * s;
    uint32_t i;
    uint32_t pos;
		for (i=0; i<r; i++) {
      result[i] = 0;
    }
    for (i=0; i<cells; i++) {
      pos = i % r;
      result[pos] = result[pos] + cell[i];
    }

  ENDC++;
	
		SET OF REAL8 sum_col_alpha (PBblas.Types.dimension_t r, PBblas.Types.dimension_t s, PBblas.Types.matrix_t D, REAL8 thisalpha) := BEGINC++

    #body
    __lenResult = r * sizeof(double);
    __isAllResult = false;
    double * result = new double[r];
    __result = (void*) result;
    double *cell = (double*) d;
    uint32_t cells =  r * s;
    uint32_t i;
    uint32_t pos;
		for (i=0; i<r; i++) {
      result[i] = 0;
    }
    for (i=0; i<cells; i++) {
      pos = i % r;
      result[pos] = result[pos] + cell[i];
    }
		for (i=0; i<r; i++) {
      result[i] = result[i] * thisalpha;
    }

  ENDC++;
	//0.5 * sum ((M-V).^2)
	REAL8 sum_pow2(PBblas.Types.dimension_t N, PBblas.Types.matrix_t M, PBblas.Types.matrix_t V) := BEGINC++

    #body
    double result = 0;
		double tmpp ;
    double *cellm = (double*) m;
		double *cellv = (double*) v;
    uint32_t i;
		for (i=0; i<n; i++) {
		  tmpp =(cellm[i] - cellv [i]);
      result = result + (tmpp*tmpp);
    }
		return(0.5*result);

  ENDC++;
	//sum(M.^2)
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
	// sum (kl(rho, M))
	REAL8 sum_kl(PBblas.Types.dimension_t N, PBblas.Types.matrix_t M, PBblas.Types.value_t rho) := BEGINC++

    #body
    double result = 0;
		double tmpp ;
    double *cellm = (double*) m;
    uint32_t i;
		for (i=0; i<n; i++) {
			result = result + (rho*log(rho/cellm[i])) + ((1-rho)*log((1-rho)/(1-cellm[i])));
    }
		return(result);

  ENDC++;
    ddist := TrainData;
    m := CostFunc_params(id=1)[1].value;
    num_feat := CostFunc_params(id=2)[1].value;
    num_hid := CostFunc_params(id=3)[1].value;
    part_rows := CostFunc_params(id=4)[1].value; // partition size for the features (number of rows)
    part_cols := CostFunc_params(id=5)[1].value; // partition size for the number of columns (samples) in the input data
    BETA := CostFunc_params(id=6)[1].value;
    sparsityParam := CostFunc_params(id=7)[1].value;
    LAMBDA := CostFunc_params(id=8)[1].value;
    row_blocks_num := IF(part_rows>0, ((num_feat-1) DIV part_rows) + 1, 1);
    m_1 := 1/m;
    sparsityParam_ := -1*sparsityParam;
    sparsityParam_1 := 1-sparsityParam;
    
    //Create map for block matrix d
    dmap := PBblas.Matrix_Map(num_feat,m,part_rows,part_cols);
    
    //Create block matrix Ytmp
    Ymap := dmap;
    Ydist := ddist;
    //Creat maps for block matrices for weights

    w1map := PBblas.Matrix_Map(num_hid, num_feat, num_hid, part_rows);
    w2map := PBblas.Matrix_Map(num_feat, num_hid, part_rows, num_hid);
     //each bias vector is converted to block format
    
    b1vecmap := PBblas.Matrix_Map(num_hid, 1, num_hid, 1);
    b2vecmap := PBblas.Matrix_Map(num_feat, 1, part_rows, 1);
    
    w1_partitions := w1map.partitions_used;
    w2_partitions := w2map.partitions_used;
    b1_partitions := b1vecmap.partitions_used;
    b2_partitions := b2vecmap.partitions_used;
    Layout_Part minuspart(Layout_Part l, UNSIGNED8 c ) := TRANSFORM
      SELF.partition_id := l.partition_id - c;
      SELF := l;
    END;
    //w1m := w1dist;
    w1dist := theta (partition_id <= w1_partitions);
    //w2m := w2dist;
    w2dist := theta (partition_id > w1_partitions AND partition_id <= w1_partitions + w2_partitions);
    //w2dist  := PROJECT (w2m_, minuspart(LEFT, w1_partitions ),LOCAL);
    //b1v := b1vecdist;
    b1vecdist := theta (partition_id > w1_partitions + w2_partitions AND partition_id <= w1_partitions + w2_partitions + b1_partitions);
    //b1vecdist  := PROJECT (b1v_, minuspart (LEFT, w1_partitions + w2_partitions),LOCAL);
    
    //b2v := b2vecdist;
    b2vecdist := theta (partition_id > w1_partitions + w2_partitions + b1_partitions AND partition_id <= w1_partitions + w2_partitions + b1_partitions + b2_partitions);
    //b2vecdist  := PROJECT (b2v_, minuspart (LEFT, w1_partitions + w2_partitions + b1_partitions),LOCAL);
    
		//distribute all the parameters to the same node as the mini batch node
		node_rec := {ddist.node_id};
		mini_batch_nodeid := TABLE (ddist, node_rec,[node_id], LOCAL);
		theta_ := JOIN (theta, mini_batch_nodeid,TRUE, TRANSFORM(Layout_Part, SELF.node_id := RIGHT.node_id, SELF:= LEFT), ALL);
		theta_dist := DISTRIBUTE (theta_, node_id);// theta is distributed to the same node as the data (mini batch) so it is available for the computations
     //functions used
    PBblas.Types.value_t sp_reci(PBblas.Types.value_t v,PBblas.Types.dimension_t r,PBblas.Types.dimension_t c) := sparsityParam_/v;
    PBblas.Types.value_t sp_1_reci(PBblas.Types.value_t v,PBblas.Types.dimension_t r,PBblas.Types.dimension_t c) := sparsityParam_1/(1-v);
    //sparsity_delta=((-sparsityParam./rhohat)+((1-sparsityParam)./(1.-rhohat)));
    PBblas.Types.value_t sp_delta(PBblas.Types.value_t v,PBblas.Types.dimension_t r,PBblas.Types.dimension_t c) := (sparsityParam_/v)+(sparsityParam_1/(1-v));
    PBblas.Types.value_t siggrad(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := v*(1.0-v);
    PBblas.Types.value_t sigmoid(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := 1/(1+exp(-1*v));
    PBblas.Types.value_t pow2(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := v*v;
    //maps used
    b1map := PBblas.Matrix_Map(num_hid, m, num_hid, part_cols);
    b2map := PBblas.Matrix_Map(num_feat, m, part_rows, part_cols);
    a2map := b1map;
    a3map := b2map;
    HL_nodes := num_hid;//number of nodes in the hidden layer
    Hiddmap := b1vecmap;
    //onevec for calculating rhohat	 
	 
	 Ones_VecMap := PBblas.Matrix_Map(m, 1, part_cols, 1);
    //New Vector Generator
    // Layout_Cell gen(UNSIGNED4 c, UNSIGNED4 NumRows) := TRANSFORM
      // SELF.x := ((c-1) % NumRows) + 1;
      // SELF.y := ((c-1) DIV NumRows) + 1;
      // SELF.v := 1;
    // END;
    //Create Ones Vector for the calculations in the step fucntion
    // Ones_Vec := DATASET(m, gen(COUNTER, m));
    Ones_Vecdist := TrainLabel;
Layout_Target := PBblas.Types.Layout_Target;



row_block_bigmat (UNSIGNED8 p, UNSIGNED8 nn) := FUNCTION // p is the partition number, nn is the number of rows
	RETURN ((p-1) % nn )+1;
END;

col_block_bigmat (UNSIGNED8 p, UNSIGNED8 nn) := FUNCTION // p is the partition number, nn is the number of rows
	RETURN ((p-1) DIV nn )+1;
END;

block_smallmat (UNSIGNED8 p, UNSIGNED8 offset) := FUNCTION 
	RETURN p-offset;
END;

d_or_p := 1; // whether to distributed data or parameters
//A_in := w1 is h*f where f is divided to partitions of size prow
//B_in := mini batch data : ddist
// bb_in := bias1
//returns the sigmoid of the result
W1X_repmatb1:= FUNCTION
SET OF PBblas.Types.value_t empty_array := [];

  A := theta_dist(partition_id <= w1_partitions);
	B := ddist;
	bb_in := theta_dist(partition_id > w1_partitions + w2_partitions AND partition_id <= w1_partitions + w2_partitions + b1_partitions);
  //multiply
	Layout_Part mul2(Layout_Part b_part, Layout_Part a_part):=TRANSFORM
    //real_part_id := col_block_bigmat (b_part.partition_id, B_row_part);  //arbitrary choice
		part_id := 1;
    k := a_part.part_cols;
    SELF.partition_id := part_id;
    SELF.node_id      := b_part.node_id;
    SELF.block_row    := a_part.block_row;
    SELF.block_col    := 1;
    SELF.first_row    := 1;
    SELF.part_rows    := a_part.part_rows;
    SELF.first_col    := 1;
    SELF.part_cols    := b_part.part_cols;
    SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
                                    a_part.part_rows, b_part.part_cols, k,
                                    1.0, a_part.mat_part, b_part.mat_part,
                                    0.0, empty_array);
  END;
	//distribute weights to the same node as mini batch data

	ab_prod_ := JOIN(B, A, LEFT.block_row = RIGHT.block_col , mul2(LEFT,RIGHT), LOCAL); // A is already distributed to the same node as B
	 // Sum terms
  Layout_Part sumTerms(Layout_Part cumm, Layout_Part term) := TRANSFORM
    N := cumm.part_rows * cumm.part_cols;
    SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term.mat_part, 1);
    SELF := cumm;
  END;
  ab_prod := ROLLUP(ab_prod_, sumTerms(LEFT, RIGHT), partition_id, LOCAL);
	// add bias vector to each columns of X
	Layout_Part addbias_(Layout_Part cumm, Layout_Part term) := TRANSFORM
		term_part_rows := term.part_rows;//number of elements in this partition of the bias vector
    N := cumm.part_rows * cumm.part_cols;
		SELF.mat_part := mat_vec_sum_sigmoid(N, term_part_rows, cumm.mat_part, term.mat_part);
    SELF := cumm;
  END;
	ab_bb := JOIN(ab_prod, bb_in,LEFT.block_row = RIGHT.block_row , addbias_(LEFT,RIGHT),LOCAL);
	RETURN ab_bb;
END; // END W1X_repmatb1


W2td3_repmatsparsity( DATASET(Layout_Part) B_in,  DATASET(Layout_Part) bb_in,  REAL8 beta_in=0) := FUNCTION

SET OF PBblas.Types.value_t empty_array := [];
	// First check maps for compatability.  Normalize for transpose operations.

	A := theta_dist (partition_id > w1_partitions AND partition_id <= w1_partitions + w2_partitions);
  B := B_in;
	
  //multiply

	Layout_Part mul2(Layout_Part b_part, Layout_Part a_part):=TRANSFORM
    k := a_part.part_rows;
    SELF.partition_id := 1;
    SELF.node_id      := b_part.node_id;
    SELF.block_row    := a_part.block_col;
    SELF.block_col    := b_part.block_col;
    SELF.first_row    := 1;
    SELF.part_rows    := a_part.part_cols;
    SELF.first_col    := 1;
    SELF.part_cols    := b_part.part_cols;
    SELF.mat_part     := PBblas.BLAS.dgemm(TRUE, FALSE,
                                    a_part.part_cols, b_part.part_cols, k,
                                    1.0, a_part.mat_part, b_part.mat_part,
                                    0.0, empty_array);

  END;

	ab_prod_ := JOIN(B, A, LEFT.block_row = RIGHT.block_row, mul2(LEFT,RIGHT),LOCAL);

	 // Sum terms
  Layout_Part sumTerms(Layout_Part cumm, Layout_Part term) := TRANSFORM
    N := cumm.part_rows * cumm.part_cols;
    SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term.mat_part, 1);
    SELF := cumm;
  END;
	sorted_ab_prod := SORT(ab_prod_, partition_id, LOCAL);
  ab_prod := ROLLUP(sorted_ab_prod, sumTerms(LEFT, RIGHT), partition_id, LOCAL);

Layout_Part addbias(Layout_Part cumm, Layout_Part term) := TRANSFORM
		term_part_rows := term.part_rows;//number of elements in this partition of the bias vector
    N := cumm.part_rows * cumm.part_cols;
		SELF.mat_part := mat_vec_sum(N, term_part_rows, cumm.mat_part, term.mat_part, beta_in);
    SELF := cumm;
  END;
	
	
	 ab_bb := JOIN(ab_prod, bb_in,LEFT.block_row = RIGHT.block_row , addbias(LEFT,RIGHT),LOCAL);
	RETURN ab_bb;
END; // END W2td3_repmatsparsity







//returns the sigmoid of the result
W2a2_repmatb2(DATASET(Layout_Part) B_in) := FUNCTION
SET OF PBblas.Types.value_t empty_array := [];
// First check maps for compatability.  Normalize for transpose operations.
  
  A := theta_dist (partition_id > w1_partitions AND partition_id <= w1_partitions + w2_partitions);
  B := B_in;
	bb_in :=  theta_dist(partition_id > w1_partitions + w2_partitions + b1_partitions AND partition_id <= w1_partitions + w2_partitions + b1_partitions + b2_partitions);
  //multiply

	Layout_Part mul2(Layout_Part b_part, Layout_Part a_part):=TRANSFORM

    k := a_part.part_cols;
    SELF.partition_id := a_part.block_row;
    SELF.node_id      := b_part.node_id;
    SELF.block_row    := a_part.block_row;
    SELF.block_col    := b_part.block_col;
		//(((p-1)  %  row_blocks) * block_rows) + 1;
    SELF.first_row    := (((a_part.block_row-1)  %  row_blocks_num) * part_rows) + 1;;
    SELF.part_rows    := a_part.part_rows;
    SELF.first_col    := 1;
    SELF.part_cols    := b_part.part_cols;
    SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
                                    a_part.part_rows, b_part.part_cols, k,
                                    1.0, a_part.mat_part, b_part.mat_part,
                                    0.0, empty_array);
  END;
 
	ab_prod := JOIN(B, A, LEFT.block_row = RIGHT.block_col , mul2(LEFT,RIGHT), LOCAL);

	Layout_Part addbias_(Layout_Part cumm, Layout_Part term) := TRANSFORM
		term_part_rows := term.part_rows;
    N := cumm.part_rows * cumm.part_cols;
		SELF.mat_part := mat_vec_sum_sigmoid(N, term_part_rows, cumm.mat_part, term.mat_part);
    SELF := cumm;
  END;
	
	 ab_bb := JOIN(ab_prod, bb_in,LEFT.block_row = RIGHT.block_row , addbias_(LEFT,RIGHT),LOOKUP);

	RETURN ab_bb;
END; // END W2a2_repmatb2


	
		
		col_mean(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_c, REAL8 mean_coef) := FUNCTION
		
			Num := map_a.matrix_cols;
			Num_1 := 1/Num;
			SET OF PBblas.Types.value_t empty_array := [];
				Layout_Part mul(Layout_Part a_part, Layout_Part b_part):=TRANSFORM
					part_id     := 1;    //arbitrary choice
					part_a_cols := a_part.part_cols;
					part_a_rows := a_part.part_rows;
					part_b_rows := b_part.part_rows;
					part_c_rows := map_c.part_rows(part_id);
					part_c_cols := map_c.part_cols(part_id);
					part_c_first_row  := map_c.first_row(part_id);
					part_c_first_col  := map_c.first_col(part_id);
					k := part_a_cols;
					SELF.partition_id := 1;
					SELF.node_id      := 0;
					SELF.block_row    := 1;
					SELF.block_col    := 1;
					SELF.first_row    := map_c.first_row(part_id);
					SELF.part_rows    := part_c_rows;
					SELF.first_col    := part_c_first_col;
					SELF.part_cols    := part_c_cols;
					SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
																					part_c_rows, part_c_cols, k,
																					Num_1, a_part.mat_part, b_part.mat_part,
																					0.0, empty_array);
			END;
			col_sum_part_ := JOIN (A_in, B_in, LEFT.partition_id = RIGHT.partition_id, mul(LEFT, RIGHT), LOCAL );
			
			
			Layout_Part col_sum_tran(Layout_Part the_part):= TRANSFORM
				SELF.mat_part := sum_col_alpha (the_part.part_rows, the_part.part_cols, the_part.mat_part, Num_1);
				SELF := the_part;
			END;
			col_sum_part := PROJECT (A_in, col_sum_tran(LEFT),LOCAL);
			Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := le.part_rows ;
				SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1);
				SELF := le;
			END;
			rslt := ROLLUP(col_sum_part,TRUE, addup(LEFT, RIGHT)); // this form of rollup avoid grouping, ROLLUP(col_sum_part,addup(LEFT, RIGHT, partiion_id)) cause all the records with the same partiion_id
			//get grouped to one node which cause a lot of overload. The current rollup form avoidds grouping and improves performance
			final_rslt := DISTRIBUTE (rslt, node_id); 
			//distribute to node one
			RETURN rslt;
		END;//END colmean
		//sum (A_in,2)
		colmean_bias_grad (PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, UNSIGNED b_offset) := FUNCTION

			
			Layout_Part col_sum_tran(Layout_Part the_part):= TRANSFORM
			  Num_1 := 1/ the_part.part_cols;
				SELF.mat_part := sum_col_alpha (the_part.part_rows, the_part.part_cols, the_part.mat_part, Num_1);
				part_id := the_part.block_row + b_offset;
				real_part :=  the_part.block_row;
				SELF. partition_id := part_id;
				SELF.node_id := the_part.node_id;
				SELF.block_row    := the_part.block_row;
				SELF.block_col    := 1;
				SELF.first_row    := map_b.first_row(real_part);
				SELF.part_rows    := map_b.part_rows(real_part);
				SELF.first_col    := 1;
				SELF.part_cols    := 1;
				SELF := the_part;
			END;
			col_sum_part := PROJECT (A_in, col_sum_tran(LEFT),LOCAL);
			RETURN col_sum_part;
		END;//END colmean_bias_grad
		
		colmean_bias_grad2 (PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, UNSIGNED b_offset) := FUNCTION

			
			Layout_Part col_sum_tran(Layout_Part the_part):= TRANSFORM
			  Num_1 := 1/ the_part.part_cols;
				SELF.mat_part := sum_col_alpha (the_part.part_rows, the_part.part_cols, the_part.mat_part, Num_1);
				part_id := the_part.block_row + b_offset;
				real_part :=  the_part.block_row;
				SELF. partition_id := part_id;
				SELF.node_id := part_id - 1;
				SELF.block_row    := the_part.block_row;
				SELF.block_col    := 1;
				SELF.first_row    := map_b.first_row(real_part);
				SELF.part_rows    := map_b.part_rows(real_part);
				SELF.first_col    := 1;
				SELF.part_cols    := 1;
				SELF := the_part;
			END;
			col_sum_part := PROJECT (A_in, col_sum_tran(LEFT),LOCAL);
			RETURN DISTRIBUTE(col_sum_part,node_id);
		END;//END colmean_bias_grad2
		
		
		// This function gets two big matrices which are distributed over all nodes and generate a final relatively smaller matrix which is on one node
		// this is used for weight gradient calculation where for example a h*m matrix is multiplied by a m*f matrix. PBblas will distribute all partitions in the first and second matrix to only one node which final matrix is in
		// this causes overhead, to avoid that we multiply each col partition of first matrix with a row partition of the second matrix in each node, the final generated matrices are added up to generate the final matrix
		// this way, we don't change the distribution of the first and second matrices
		big_big_small(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_c, PBblas.Types.value_t alph=1.0) := FUNCTION
			SET OF PBblas.Types.value_t empty_array := [];
				Layout_Part mul(Layout_Part a_part, Layout_Part b_part):=TRANSFORM
					part_id     := 1;    //arbitrary choice
					part_a_cols := a_part.part_cols;
					part_a_rows := a_part.part_rows;
					part_b_rows := b_part.part_rows;
					part_b_cols := b_part.part_cols;
					k := part_a_cols;
					SELF.partition_id := part_id;
					SELF.node_id      := 0;
					SELF.block_row    := 1;
					SELF.block_col    := 1;
					SELF.first_row    := map_c.first_row(part_id);
					SELF.part_rows    := map_c.part_rows(part_id);
					SELF.first_col    := map_c.first_col(part_id);
					SELF.part_cols    := map_c.part_cols(part_id);
					SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, TRUE,
																					part_a_rows, part_b_rows, k,
																					alph, a_part.mat_part, b_part.mat_part,
																					0.0, empty_array);
			END;
			mul_part := JOIN (A_in, B_in, LEFT.partition_id = RIGHT.partition_id, mul(LEFT, RIGHT), LOCAL );
			
			Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := le.part_rows * le.part_cols ;
				SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1);
				SELF := le;
			END;
			
			Layout_Part addup_it(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := ri.part_rows * ri.part_cols ;
				SELF.mat_part := IF (le.partition_id=0, ri.mat_part, PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1));
				SELF := ri;
			END;
			//rslt := ROLLUP(mul_part, addup(LEFT, RIGHT), partition_id); // since the results of rohat is used in a ALL join, no need to distribute this to node one to be consistent with PBblas
			//rslt := ITERATE(mul_part, addup_it(LEFT, RIGHT));// using rollup cause the graph to Group all the records which are distributed between all node to only one record and then do the operation, It takes a long time to GROUP all thoese partitions in one node and we avoid it by using ITERATE instead of ROLLUP

      rslt := ROLLUP(mul_part, TRUE, addup(LEFT, RIGHT));
			final_rslt := DISTRIBUTE (rslt, node_id); 

// a_part := A_in[1];
// b_part := B_in[1];
		 // RETURN PBblas.BLAS.dgemm(FALSE, TRUE,
																					// a_part.part_rows, b_part.part_rows, a_part.part_cols,
																					// 1.0, a_part.mat_part, b_part.mat_part,
																					// 0.0, empty_array);
																					
			RETURN final_rslt;
		END;// END big_big_small
		//calculates coef * (d2*x')
		d2Xt( DATASET(Layout_Part) A_in, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_c) := FUNCTION
			SET OF PBblas.Types.value_t empty_array := [];
				Layout_Part mul(Layout_Part a_part, Layout_Part b_part):=TRANSFORM
					part_id     := b_part.block_row;
					part_a_cols := a_part.part_cols;
					part_a_rows := a_part.part_rows;
					part_b_rows := b_part.part_rows;
					part_b_cols := b_part.part_cols;
					k := a_part.part_cols;
					SELF.partition_id := part_id;
					SELF.node_id      := part_id-1;
					SELF.block_row    := a_part.block_row;
					SELF.block_col    := b_part.block_row;
					SELF.first_row    := map_c.first_row(part_id);
					SELF.part_rows    := a_part.part_rows;
					SELF.first_col    := map_c.first_col(part_id);
					SELF.part_cols    := b_part.part_rows;
					SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, TRUE,
																					a_part.part_rows, b_part.part_rows, k,
																					(1/b_part.part_cols), a_part.mat_part, b_part.mat_part,
																					0.0, empty_array);
			END;
			
			mul_part := JOIN (A_in, B_in, LEFT.block_col = RIGHT.block_col-RIGHT.block_col+1, mul(LEFT, RIGHT), LOCAL );
			mul_part_dist := DISTRIBUTE (mul_part, node_id);
			RETURN mul_part_dist;
		END;// END d2Xt
		
		d3a2t( DATASET(Layout_Part) A_in, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_c, UNSIGNED B_offset) := FUNCTION
			SET OF PBblas.Types.value_t empty_array := [];
				Layout_Part mul(Layout_Part a_part, Layout_Part b_part):=TRANSFORM
					
					part_id := a_part.block_row;
					new_part_id     := part_id + B_offset;
					k := a_part.part_cols;
					SELF.partition_id := new_part_id;
					SELF.node_id      := new_part_id-1;
					SELF.block_row    := a_part.block_row;
					SELF.block_col    := b_part.block_row;
					SELF.first_row    := map_c.first_row(part_id);
					SELF.part_rows    := a_part.part_rows;
					SELF.first_col    := 1;
					SELF.part_cols    := b_part.part_rows;
					SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, TRUE,
																					a_part.part_rows, b_part.part_rows, k,
																					(1/b_part.part_cols), a_part.mat_part, b_part.mat_part,
																					0.0, empty_array);
			END;
			mul_part := JOIN (A_in, B_in, LEFT.block_col = RIGHT.block_col, mul(LEFT, RIGHT), LOCAL );
			mul_part_dist := DISTRIBUTE (mul_part, node_id);
																					
			RETURN mul_part_dist;
		END;// END d3a2t
		
		
		
		
		//extract sparse autoencoder parameters
   

    //FF2 returns a2
    FF2_(DATASET(Layout_Part) w1, DATASET(Layout_Part) b1v):= FUNCTION
      //b1m = repmat(b1v,1,m)
      b1m := PBblas.PB_dgemm(FALSE, TRUE, 1.0,b1vecmap, b1v, Ones_VecMap, Ones_Vecdist, b1map);
      //z2 = w1*X+b1;
      //z2 := PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1, dmap, ddist, b1map, b1m, 1.0); // gives MP closed error
      z2_ := PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1, dmap, ddist, b1map);
      z2 := PBblas.PB_daxpy(1.0, z2_, b1m);
      //a2 = sigmoid (z2);
      a2 := PBblas.Apply2Elements(b1map, z2, sigmoid);
      RETURN a2;
    END;//END FF2_
		
		
		
		//returns a2
		 // FF2(DATASET(Layout_Part) w1, DATASET(Layout_Part) b1v):= FUNCTION
      // z2=w1*x+repmat(b1,1,m)
			// z2 := WX_repmatb(w1map, w1, dmap, ddist, b1vecmap, b1v, b1map, 0.0);
      // a2 = sigmoid (z2);
      // a2 := PBblas.Apply2Elements(b1map, z2, sigmoid);
      // RETURN a2;
     // END;//END FF2
		 
		 
		
		FF2 := FUNCTION
      //z2=w1*x+repmat(b1,1,m)
			//z2 := WX_repmatb(w1map, w1, dmap, ddist, b1vecmap, b1v, b1map, 0.0);
			a2 := W1X_repmatb1;
      //a2 = sigmoid (z2);
      
      RETURN a2;
     END;//END FF2
		 
    //FF3 returns a3
    FF3_(DATASET(Layout_Part) w2,DATASET(Layout_Part) b2v, DATASET(Layout_Part) a2 ):= FUNCTION
      //b2m = repmat(b2v,1,m)
      b2m := PBblas.PB_dgemm(FALSE, TRUE, 1.0,b2vecmap, b2v, Ones_VecMap, Ones_Vecdist, b2map);
      //z3 = w2*a2+b2;
      //z3 := PBblas.PB_dgemm(FALSE, FALSE,1.0,w2map, w2, a2map, a2, b2map,b2m, 1.0); //gives MP closed error
      z3_ := PBblas.PB_dgemm(FALSE, FALSE,1.0,w2map, w2, a2map, a2, b2map);
      z3 := PBblas.PB_daxpy(1.0, z3_, b2m);
      //a3 = sigmoid (z3)
      a3 := PBblas.Apply2Elements(b2map, z3, sigmoid);
      RETURN a3;
    END;//END FF3_
		
		 //FF3 returns a3
    // FF3(DATASET(Layout_Part) w2,DATASET(Layout_Part) b2v, DATASET(Layout_Part) a2 ):= FUNCTION
		  // z3 = w2*a2 + repmat(b2,1,m)
			// z3 := WX_repmatb(w2map, w2, a2map, a2, b2vecmap, b2v, b2map, 0.0);
      // a3 = sigmoid (z3)
      // a3 := PBblas.Apply2Elements(b2map, z3, sigmoid);
      // RETURN a3;
    // END;//END FF3
		
	 FF3( DATASET(Layout_Part) a2 ):= FUNCTION
		  //z3 = w2*a2 + repmat(b2,1,m)
			//z3 := WX_repmatb(w2map, w2, a2map, a2, b2vecmap, b2v, b2map, 0.0);
			a3 := W2a2_repmatb2( a2);
      //a3 = sigmoid (z3)
      
      RETURN a3;
    END;//END FF3
		

    //DELTA3 returns d3
    DELTA3 (DATASET(Layout_Part) a3 ) := FUNCTION
      //calculate delta for the last layer (3rd layer)
      //y=X;
      //d3=-(y-a3).*(a3.*(1-a3));
      // siggrad_a3 := PBblas.Apply2Elements(a3map, a3, siggrad);
      // a3_y := PBblas.PB_daxpy(-1, ddist, a3);
      // d3 := PBblas.HadamardProduct(a3map, a3_y, siggrad_a3);
			Layout_part d3_tran (Layout_part a3_part, Layout_part y_part) := TRANSFORM
				SELF.mat_part := d3_cal(a3_part.part_rows * a3_part.part_cols, a3_part.mat_part, y_part.mat_part);
				SELF := a3_part;
			END;
			d3 := JOIN (a3, ddist,LEFT.block_row = RIGHT.block_row, d3_tran(LEFT, RIGHT), LOCAL);
      RETURN d3 ;
    END;//END DELTA3
		
		DELTA3_ (DATASET(Layout_Part) a3 ) := FUNCTION
      //calculate delta for the last layer (3rd layer)
      //y=X;
      //d3=-(y-a3).*(a3.*(1-a3));
      siggrad_a3 := PBblas.Apply2Elements(a3map, a3, siggrad);
      a3_y := PBblas.PB_daxpy(-1, ddist, a3);
      d3 := PBblas.HadamardProduct(a3map, a3_y, siggrad_a3);
      RETURN d3 ;
    END;//END DELTA3_
    //DELTA2 retunrs d2
    rohat_ (DATASET(Layout_Part) a2) := FUNCTION
      rh := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a2map, a2, Ones_VecMap, Ones_Vecdist, Hiddmap);
      RETURN rh;
    END;
		rohat (DATASET(Layout_Part) a2) := FUNCTION
			//rh := col_mean(a2map, a2, Ones_VecMap, Ones_Vecdist, Hiddmap, 1.0);
			rh := colmean_bias_grad (a2map, a2, Hiddmap, 0);
      RETURN rh;
    END;
    DELTA2_ (DATASET(Layout_Part) w2, DATASET(Layout_Part) a2, DATASET(Layout_Part) d3, DATASET(Layout_Part) rhat) := FUNCTION
      //calculate delta for 2nd layer
      //rhohat=mean(a2,2);
      //sparsity_delta=((-sparsityParam./rhohat)+((1-sparsityParam)./(1.-rhohat)));
      //d2=((W2'*d3)+beta*repmat(sparsity_delta,1,m)) .* (a2.*(1-a2));
      //rhohat := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a2map, a2, Ones_VecMap, Ones_Vecdist, Hiddmap);
      rhohat := rhat;
      sparsity_delta := PBblas.Apply2Elements(Hiddmap, rhohat, sp_delta);
      siggrad_a2 := PBblas.Apply2Elements(a2map, a2, siggrad);
      repmat_sparsity_delta := PBblas.PB_dgemm(FALSE, TRUE, 1.0,  Hiddmap, sparsity_delta, Ones_VecMap, Ones_Vecdist, a2map);
      //d2_firstterm = (W2'*d3)+beta*repmat(sparsity_delta,1,m);
      //d2_firstterm := PBblas.PB_dgemm(TRUE, FALSE, 1.0, w2map, w2, a3map, d3, a2map, repmat_sparsity_delta, BETA); // MP close error
      d2_firstterm_ := PBblas.PB_dgemm(TRUE, FALSE, 1.0, w2map, w2, a3map, d3, a2map);
      d2_firstterm := PBblas.PB_daxpy(BETA, repmat_sparsity_delta, d2_firstterm_);
      d2 := PBblas.HadamardProduct(a2map, d2_firstterm, siggrad_a2);
      RETURN d2 ;
    END;
		
		
		DELTA2 (DATASET(Layout_Part) w2, DATASET(Layout_Part) a2, DATASET(Layout_Part) d3, DATASET(Layout_Part) rhat) := FUNCTION
      //calculate delta for 2nd layer
      //rhohat=mean(a2,2);
      //sparsity_delta=((-sparsityParam./rhohat)+((1-sparsityParam)./(1.-rhohat)));
      //d2=((W2'*d3)+beta*repmat(sparsity_delta,1,m)) .* (a2.*(1-a2));
      //rhohat := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a2map, a2, Ones_VecMap, Ones_Vecdist, Hiddmap);
      rhohat := rhat;
		
			
      sparsity_delta := PBblas.Apply2Elements(Hiddmap, rhohat, sp_delta);
      // siggrad_a2 := PBblas.Apply2Elements(a2map, a2, siggrad);
      //repmat_sparsity_delta := PBblas.PB_dgemm(FALSE, TRUE, 1.0,  Hiddmap, sparsity_delta, Ones_VecMap, Ones_Vecdist, a2map);
      //d2_firstterm = (W2'*d3)+beta*repmat(sparsity_delta,1,m);
			//d2_firstterm := WtX_repmatb(w2map, w2, a3map, d3, Hiddmap, sparsity_delta, a2map, BETA);
			d2_firstterm := W2td3_repmatsparsity( d3, sparsity_delta, BETA);
      //d2 := PBblas.HadamardProduct(a2map, d2_firstterm, siggrad_a2);
			Layout_part d2_tran (Layout_part a2_part, Layout_part d2_part) := TRANSFORM
				SELF.mat_part := d2_cal(a2_part.part_rows * a2_part.part_cols, a2_part.mat_part, d2_part.mat_part);
				SELF := a2_part;
			END;
			d2 := JOIN (a2, d2_firstterm, LEFT.block_row = RIGHT.block_row, d2_tran(LEFT, RIGHT), LOCAL);
      RETURN d2;
    END;
    //WeightGrad1 returns gradient for w1
    WeightGrad1_ (DATASET(Layout_Part) w1, DATASET(Layout_Part) d2) := FUNCTION
      w1_g := PBblas.PB_dgemm(FALSE, TRUE, m_1, a2map, d2, dmap, ddist, w1map, w1 ,LAMBDA );
      RETURN w1_g;
    END;
		WeightGrad1 (DATASET(Layout_Part) w1, DATASET(Layout_Part) d2) := FUNCTION
			//w1_g_ := big_big_small(a2map, d2, dmap, ddist, w1map, m_1);
			w1_g_ := d2Xt(d2, ddist, w1map);
			w1_g  := PBblas.PB_daxpy(LAMBDA, w1, w1_g_);
      RETURN w1_g;
    END;
		
    //WeightGrad2 returns gradient for w2
    WeightGrad2_ (DATASET(Layout_Part) w2, DATASET(Layout_Part) d3, DATASET(Layout_Part) a2) := FUNCTION
      w2_g := PBblas.PB_dgemm(FALSE, TRUE, m_1, a3map, d3, a2map, a2, w2map, w2 ,LAMBDA );
      RETURN w2_g;
    END;
		
		WeightGrad2 (DATASET(Layout_Part) w2, DATASET(Layout_Part) d3, DATASET(Layout_Part) a2) := FUNCTION
			//w2_g_ := big_big_small(a3map, d3, a2map, a2, w2map, m_1);
			w2_g_ := d3a2t( d3, a2, w2map, w1_partitions);
			w2_g  := PBblas.PB_daxpy(LAMBDA, w2, w2_g_);
      RETURN w2_g;
    END;
    //BiasGrad1 calculates the bias gradients for b1
    BiasGrad1 (DATASET(Layout_Part) d2) := FUNCTION
			//b1_g := col_mean(a2map, d2, Ones_VecMap, Ones_Vecdist, b1vecmap, m_1);
			b1_g := colmean_bias_grad2 (a2map, d2, b1vecmap, w1_partitions + w2_partitions);
      RETURN b1_g;
    END;
		
		BiasGrad1_ (DATASET(Layout_Part) d2) := FUNCTION
      b1_g := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a2map, d2, Ones_VecMap, Ones_Vecdist, b1vecmap);
      RETURN b1_g;
    END;
    //BiasGrad2 calculates the bias gradients for b2
    BiasGrad2_ (DATASET(Layout_Part) d3) := FUNCTION
      b2_g := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a3map, d3, Ones_VecMap, Ones_Vecdist, b2vecmap);
      RETURN b2_g;
    END;
		
		BiasGrad2 (DATASET(Layout_Part) d3) := FUNCTION
      //b2_g := col_mean(a3map, d3, Ones_VecMap, Ones_Vecdist, b2vecmap, m_1);
			b2_g := colmean_bias_grad2 (a3map, d3, b2vecmap, w1_partitions + w2_partitions + b1_partitions);
      RETURN b2_g;
    END;
		
		
    emptyL := DATASET([], Layout_Part);
    //theta is the input weight and bias parameters for the sparseautoencoder
    //parameters are the parameters for the sparseautoencoder function
    //train_d is the train data to learn sparse autoencoder and calculate the gradient and the cost based on taht
    // train_l is empty in the sparse autoencoder (because it is an unsupervised learning)
    SparseParam_CostGradients4 :=  FUNCTION
      PBblas.Types.MUElement minuspart(Layout_Part l, UNSIGNED8 c ) := TRANSFORM
        SELF.partition_id := l.partition_id - c;
        SElF.no := 1;
        SELF := l;
      END;
      w1m := w1dist;
      w2m := w2dist;
      b1v := b1vecdist;
      b2v := b2vecdist;
      a2 := FF2 ;
			a2_ := FF2_ (w1m, b1v);
      a3 := FF3 (a2);
			a3_ := FF3_ (w2m, b2v, a2_);
      d3 := DELTA3 (a3);
			d3_ := DELTA3_ (a3_);
      rohat_a2 := rohat(a2);
			rohat_a2_ := rohat_(a2_);
      d2 := DELTA2 (w2m, a2, d3,rohat_a2);
			d2_ := DELTA2_ (w2m, a2_, d3_,rohat_a2_);
      wg1 := WeightGrad1 (w1m, d2);
      wg2 := WeightGrad2 (w2m, d3, a2);
      bg1 := BiasGrad1 (d2);
      bg2 := BiasGrad2 (d3);
			
			
			wg1_ := WeightGrad1_ (w1m, d2_);
      wg2_ := WeightGrad2_ (w2m, d3_, a2_);
      bg1_ := BiasGrad1_ (d2_);
      bg2_ := BiasGrad2_ (d3_);
      //calculate cost
      //PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1, dmap, ddist, b1map, b1m, 1.0);
      // squared_error_cost= 0.5*sum(sum((x-a3).^2));
      // cost=(1/m)*squared_error_cost+(lambda/2)*(sum(W2(:).^2)+sum(W1(:).^2))+beta*sum(KL(sparsityParam,rhohat));
			
			simple := {REAL8 v};
			simple SE_tran (Layout_Part le, Layout_Part ri) := TRANSFORM
				SELF.v := 1/(le.part_cols)*sum_pow2(le.part_cols * le.part_rows, le.mat_part, ri.mat_part);
			END;
			d_a3_sum_part := JOIN (a3, ddist, LEFT.block_row = RIGHT.block_row,SE_tran(LEFT,RIGHT), LOCAL );
      //squared_error_cost := 0.5*PBblas.SumElements(PBblas.Apply2Elements(dmap, PBblas.PB_daxpy(-1.0, a3, ddist), pow2));
			squared_error_cost := SUM (d_a3_sum_part, d_a3_sum_part.v);
      cost_term1 := squared_error_cost;
			
			simple term2_3_tran (Layout_Part le) := TRANSFORM
				SELF.v := sum_sq(le.part_cols * le.part_rows, le.mat_part);
			END;
			w1_term3 := PROJECT (w1m, term2_3_tran (LEFT), LOCAL);
			w2_term2 := PROJECT (w2m, term2_3_tran (LEFT), LOCAL);

      // cost_term2 := (lambda/2)* PBblas.SumElements(PBblas.Apply2Elements(w2map, w2m, pow2));
      //cost_term3 := (lambda/2)* PBblas.SumElements(PBblas.Apply2Elements(w1map, w1m, pow2));
			cost_term2 := (lambda/2)* SUM (w2_term2, w2_term2.v); 
			cost_term3 := (lambda/2)* SUM (w1_term3, w1_term3.v);
			
			simple kl_tran (Layout_Part le) := TRANSFORM
				SELF.v := sum_kl(le.part_cols * le.part_rows, le.mat_part, sparsityParam);
			END;
			
      PBblas.Types.value_t klterm(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := sparsityParam * LN(sparsityParam/v) + (1-sparsityParam) * LN ((1-sparsityParam)/(1-v));
      //KL := PBblas.Apply2Elements (Hiddmap,rohat_a2,klterm);
		  KL := PROJECT (rohat_a2, kl_tran(LEFT), LOCAL);
      //cost_term4 := beta * PBblas.SumElements(KL);
			cost_term4 := beta * SUM (KL, KL.v);
      cost := cost_term1 + cost_term2 + cost_term3 +cost_term4;    
      costField := DATASET ([{1,1,cost}],ML.Types.NumericField);
      one_map := PBblas.Matrix_Map(1,1,1,1);
      Cost_part_no := PBblas.MU.TO(ML.DMat.Converted.FromNumericFieldDS(costField,one_map),2);
      //convert w and b gradients to a datasets of layoutparts where partition_id differentiate them
      //w1_grad has partition_id from 1 to w1_partitions
      //w2_grad had partition_ds from w1_partitions+1 to w1_partitions + w2_partitions
      //b1_grad has partition_ids from w1_partitions + w2_partitions+1 to w1_partitions + w2_partitions+b1_partitions
      //b2_grad has partition_ids from w1_partitions + w2_partitions+b1_partitions+1 to w1_partitions + w2_partitions+b1_partitions+b2_partitions
      PBblas.Types.MUElement addpart(Layout_Part l, UNSIGNED8 c ) := TRANSFORM
        SELF.partition_id := l.partition_id + c;
        SElF.no := 1;
        SELF := l;
      END;
      wg1_reshape_no := Pbblas.MU.TO(wg1,1);
      wg2_reshape_no := PROJECT (wg2, addpart(LEFT, w1_partitions ), LOCAL);
			wg2_reshape_no_ := PROJECT (wg2_, addpart(LEFT, w1_partitions ), LOCAL);
      bg1_reshape_no := PROJECT (bg1, addpart (LEFT, w1_partitions + w2_partitions),LOCAL);
      bg2_reshape_no := PROJECT (bg2, addpart (LEFT, w1_partitions + w2_partitions + b1_partitions),LOCAL);
     // theta_Part_no := wg1_reshape_no + wg2_reshape_no + bg1_reshape_no + bg2_reshape_no;
      theta_Part_no := PROJECT (wg1 + wg2 + bg1 + bg2 ,TRANSFORM (PBblas.Types.MUElement, SELF.no :=1 ; SELF := LEFT) , LOCAL);
			//RETURN PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1dist, dmap, ddist, b1map);
			//RETURN WX(w1map, w1dist, dmap, ddist, b1map, w1dist,  1);
			//RETURN WX_repmatb(w1map, w1dist, dmap, ddist, b1vecmap, b1vecdist, b1map, 1.0) ;
			//RETURN col_sum(dmap, ddist, Ones_VecMap, Ones_Vecdist, b2vecmap);
//			w1x_b1 := WX_repmatb(w1map, w1dist, dmap, ddist, b1vecmap, b1vecdist, b1map, 1.0);
			// thisis := big_big_small(b1map, w1x_b1, dmap, ddist, PBblas.Matrix_Map(num_hid, num_feat, num_hid, num_feat));
			// thisone := WtX_repmatb(w2map,w2dist, dmap, ddist, b1vecmap, b1vecdist, b1map, 5);
			
			
			
			//mymy2 := DELTA2 (w2m, a2, d3,rohat_a2) + DELTA2_ (w2m, a2, d3,rohat_a2);
			//mymy2 := big_big_small(a2map, d2, dmap, ddist, w1map, 8);
			//mymy2 := col_mean(a3map, d3, Ones_VecMap, Ones_Vecdist, b2vecmap);
			mymy2 := wg1 + wg2  + bg1 + bg2;
myformat2 := RECORD
    mymy2.node_id;
    mymy2.partition_id;
    mymy2.block_row;
    mymy2.block_col;
    mymy2.first_row;
    mymy2.part_rows;
    mymy2.first_col;
    mymy2.part_cols;
		//mymy2.no;
		//mymy2.mat_part;
		INTEGER real_node := STD.System.Thorlib.Node();
END;
	thisR := TABLE(mymy2,myformat2,LOCAL); 
	theta_part_no_check :=  ASSERT(theta_Part_no, node_id=Thorlib.node() and node_id=(partition_id-1), 'sparse autoencoder gradient is not distributed correctly', FAIL);
	return_record := RECORD (Layout_Part)
		REAL8 cost_value;
	END;
	ToReturn := PROJECT (theta_part_no_check, TRANSFORM (return_record, SELF.cost_value := cost; SELF:=LEFT), LOCAL);
	//RETURN  theta_part_no_check + Cost_part_no;
	RETURN ToReturn;

    END;//END SparseParam_CostGradients4  
    RETURN SparseParam_CostGradients4;
   END;//END SA_lbfgs_Compatible2_param_part_minibatch
	 
















EXPORT SA_lbfgs_Compatible2_2 ( DATASET(Layout_Part) theta, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Part) TrainData , DATASET(Layout_Part) Ones_Vecdist) := FUNCTION

Layout_Target := PBblas.Types.Layout_Target;
WX(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_bb, DATASET(Layout_Part) bb_in, PBblas.IMatrix_Map map_c, REAL8 beta=0.0) := FUNCTION
SET OF PBblas.Types.value_t empty_array := [];
// First check maps for compatability.  Normalize for transpose operations.
  a_matrix_rows := map_a.matrix_rows;
  a_matrix_cols := map_a.matrix_cols;
  a_row_blocks  := map_a.row_blocks;
  a_col_blocks  := map_a.col_blocks;
  b_matrix_rows := map_b.matrix_rows;
  b_matrix_cols := map_b.matrix_cols;
  b_row_blocks  := map_b.row_blocks;
  b_col_blocks  := map_b.col_blocks;
  c_matrix_rows := map_c.matrix_rows;
  c_matrix_cols := map_c.matrix_cols;
  c_row_blocks  := map_c.row_blocks;
  c_col_blocks  := map_c.col_blocks;
  A := ASSERT(A_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' + ' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(a_matrix_rows=c_matrix_rows AND a_row_blocks=c_row_blocks,
                    'A-C: ' + 'Arows: ' + a_matrix_rows +' Crows '+c_matrix_rows+' '+ 
                              'Ablocks: ' + a_row_blocks +' Cblocks '+c_row_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	B := ASSERT(B_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' +' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(b_matrix_cols=c_matrix_cols AND b_col_blocks=c_col_blocks,
                    'B-C: ' + 'Brows: ' + b_matrix_cols +' Ccols '+c_matrix_cols+' '+ 
                              'Bblocks: ' + b_row_blocks +' Cblocks '+c_col_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));

//
  Layout_Target cvt(Layout_Part par, INTEGER c, BOOLEAN keepRow) := TRANSFORM
    s_block_row       := par.block_row;
    s_block_col       := par.block_col;
    part_id_new_row   := map_c.assigned_part(c, s_block_col);
    part_id_new_col   := map_c.assigned_part(s_block_row, c);
    partition_id      := IF(keepRow, part_id_new_col, part_id_new_row);
    SELF.t_node_id    := map_c.assigned_node(partition_id);
    SELF.t_part_id    := partition_id;
    SELF.t_block_row  := IF(keepRow, s_block_row, c);
    SELF.t_block_col  := IF(keepRow, c, s_block_col);
    SELF.t_term       := IF(keepRow, s_block_col, s_block_row);
    SELF              := par;
  END;

  // A: copy of weight matrix goes to each column of X
  a_fact := map_b.col_blocks; // The number of time weight matrix (A) has to be distributed is the number of columns on matrix X (B)
  a_work := NORMALIZE(A, a_fact, cvt(LEFT, COUNTER, TRUE));
  a_dist := DISTRIBUTE(a_work, t_node_id);
  a_sort := a_dist;// only one partition in each node, so no need to sort
  // B: copy of each cell in a column goes to a row
  b_fact := map_a.row_blocks;
  b_work := PROJECT(B, cvt(LEFT, COUNTER, FALSE), LOCAL);
  b_dist := b_work; // no need to distribute as it is already distributed
  b_sort := b_dist; // only one partition in each node, so no need to sort
	
	
	
	// Elem := {PBblas.Types.value_t v};
	// Elem_col := {PBblas.Types.value_t v, UNSIGNED8 v_col:=1};
	// Layout_Target rep_bb (Layout_Target x) := TRANSFORM
		// elemsX_ := DATASET(x.mat_part, Elem);
		// elemsX := PROJECT (elemsX_, TRANSFORM(Elem_col, SELF := LEFT));
		// Elem_col cvt2(Elem_col par, INTEGER c) := TRANSFORM
			// SELF := par;
		// END;
		// repeatedelemsX := NORMALIZE(elemsX, bb_fact, cvt2(LEFT));
		// self.mat_part := SET(repeatedelemsX, v);
		// SELF := x;
	
	// END;
	
	
  // Multiply
  Layout_Part mul(Layout_Target a_part, Layout_Target b_part):=TRANSFORM
    part_id     := a_part.t_part_id;    //arbitrary choice
    part_a_cols := a_part.part_cols;
    part_a_rows := a_part.part_rows;
    part_b_rows := b_part.part_rows;
    part_c_rows := map_c.part_rows(part_id);
    part_c_cols := map_c.part_cols(part_id);
    part_c_first_row  := map_c.first_row(part_id);
    part_c_first_col  := map_c.first_col(part_id);
    k := part_a_cols;
    SELF.partition_id := part_id;
    SELF.node_id      := a_part.t_node_id;
    SELF.block_row    := a_part.t_block_row;
    SELF.block_col    := a_part.t_block_col;
    SELF.first_row    := map_c.first_row(part_id);
    SELF.part_rows    := part_c_rows;
    SELF.first_col    := part_c_first_col;
    SELF.part_cols    := part_c_cols;
    SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
                                    part_c_rows, part_c_cols, k,
                                    1.0, a_part.mat_part, b_part.mat_part,
                                    0.0, empty_array);
  END;
  ab_prod := JOIN(a_sort, b_sort,
                  LEFT.t_part_id=RIGHT.t_part_id AND LEFT.t_term=RIGHT.t_term,
                  mul(LEFT,RIGHT), LOCAL);




   // Apply beta


	
	
	// N := 32*32*3*1000;
	// Layout_Target sumTerms(Layout_Target cumm, Layout_Target term) := TRANSFORM
    // SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term.mat_part, 1);
    // SELF := cumm;
  // END;
	// sumres := ROLLUP(a_sort, sumTerms(LEFT, RIGHT), partition_id);
	
	
	mymy := a_sort;
	myformat := RECORD
    mymy.node_id;
    mymy.partition_id;
    mymy.block_row;
    mymy.block_col;
    mymy.first_row;
    mymy.part_rows;
    mymy.first_col;
    mymy.part_cols;
		mymy.t_part_id;
    mymy.t_node_id;
    mymy.t_block_row;
    mymy.t_block_col;
    mymy.t_term;
		INTEGER real_node := STD.System.Thorlib.Node();
END;
mymy2 := ab_prod;
myformat2 := RECORD
    mymy2.node_id;
    mymy2.partition_id;
    mymy2.block_row;
    mymy2.block_col;
    mymy2.first_row;
    mymy2.part_rows;
    mymy2.first_col;
    mymy2.part_cols;
		INTEGER real_node := STD.System.Thorlib.Node();
END;
	rslt := TABLE(mymy2,myformat2,LOCAL); 
  RETURN rslt;
END; // END WX




WX_2(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_bb, DATASET(Layout_Part) bb_in, PBblas.IMatrix_Map map_c, REAL8 beta=0.0) := FUNCTION
SET OF PBblas.Types.value_t empty_array := [];
// First check maps for compatability.  Normalize for transpose operations.
  a_matrix_rows := map_a.matrix_rows;
  a_matrix_cols := map_a.matrix_cols;
  a_row_blocks  := map_a.row_blocks;
  a_col_blocks  := map_a.col_blocks;
  b_matrix_rows := map_b.matrix_rows;
  b_matrix_cols := map_b.matrix_cols;
  b_row_blocks  := map_b.row_blocks;
  b_col_blocks  := map_b.col_blocks;
  c_matrix_rows := map_c.matrix_rows;
  c_matrix_cols := map_c.matrix_cols;
  c_row_blocks  := map_c.row_blocks;
  c_col_blocks  := map_c.col_blocks;
  A := ASSERT(A_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' + ' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(a_matrix_rows=c_matrix_rows AND a_row_blocks=c_row_blocks,
                    'A-C: ' + 'Arows: ' + a_matrix_rows +' Crows '+c_matrix_rows+' '+ 
                              'Ablocks: ' + a_row_blocks +' Cblocks '+c_row_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	B := ASSERT(B_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' +' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(b_matrix_cols=c_matrix_cols AND b_col_blocks=c_col_blocks,
                    'B-C: ' + 'Brows: ' + b_matrix_cols +' Ccols '+c_matrix_cols+' '+ 
                              'Bblocks: ' + b_row_blocks +' Cblocks '+c_col_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));

//
  
	
	
	
	// Elem := {PBblas.Types.value_t v};
	// Elem_col := {PBblas.Types.value_t v, UNSIGNED8 v_col:=1};
	// Layout_Target rep_bb (Layout_Target x) := TRANSFORM
		// elemsX_ := DATASET(x.mat_part, Elem);
		// elemsX := PROJECT (elemsX_, TRANSFORM(Elem_col, SELF := LEFT));
		// Elem_col cvt2(Elem_col par, INTEGER c) := TRANSFORM
			// SELF := par;
		// END;
		// repeatedelemsX := NORMALIZE(elemsX, bb_fact, cvt2(LEFT));
		// self.mat_part := SET(repeatedelemsX, v);
		// SELF := x;
	
	// END;
	
	
  //multiply
	
	Layout_Part mul2(Layout_Part b_part, Layout_Part a_part):=TRANSFORM
    part_id     := b_part.partition_id;    //arbitrary choice
    part_a_cols := a_part.part_cols;
    part_a_rows := a_part.part_rows;
    part_b_rows := b_part.part_rows;
    part_c_rows := map_c.part_rows(part_id);
    part_c_cols := map_c.part_cols(part_id);
    part_c_first_row  := map_c.first_row(part_id);
    part_c_first_col  := map_c.first_col(part_id);
    k := part_a_cols;
    SELF.partition_id := b_part.partition_id;
    SELF.node_id      := b_part.node_id;
    SELF.block_row    := b_part.block_row;
    SELF.block_col    := b_part.block_col;
    SELF.first_row    := map_c.first_row(part_id);
    SELF.part_rows    := part_c_rows;
    SELF.first_col    := part_c_first_col;
    SELF.part_cols    := part_c_cols;
    SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
                                    part_c_rows, part_c_cols, k,
                                    1.0, a_part.mat_part, b_part.mat_part,
                                    0.0, empty_array);
  END;
  ab_prod := JOIN(B, A,TRUE , mul2(LEFT,RIGHT),ALL);




   // Apply beta

// Sum terms
  Layout_Part sumTerms(Layout_Part cumm, Layout_Part term) := TRANSFORM
		cumm_part_cols := cumm.part_cols;
    N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
		Elem := {PBblas.Types.value_t v};
		Elem_r := {PBblas.Types.value_t v, UNSIGNED r};
		elems := DATASET(term.mat_part, Elem);
		Elem_r rep (Elem l, UNSIGNED c) := TRANSFORM
			SELF.r := c;
			SELF := l;
		END;
		elems_rep := NORMALIZE(elems, cumm_part_cols, rep(LEFT, COUNTER));
		elems_rep_sort := SORT(elems_rep, r);
		term_rep_set := SET (elems_rep_sort, v);
    SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term_rep_set, 1);
		SELF.partition_id := cumm.partition_id;
    SELF := cumm;
  END;
	
	 ab_bb := JOIN(ab_prod, bb_in,TRUE , sumTerms(LEFT,RIGHT),ALL);

cumm := B[1];
cumm_part_cols := cumm.part_cols;
term := bb_in[1];
Elem := {PBblas.Types.value_t v};
		Elem_r := {PBblas.Types.value_t v, UNSIGNED r:=1};
		elems := DATASET(term.mat_part, Elem);
		Elem_r rep (Elem l, UNSIGNED c) := TRANSFORM
			SELF.r := c;
			SELF := l;
		END;
		elems_rep := NORMALIZE(elems, cumm_part_cols, rep(LEFT, COUNTER));
		elems_rep_sort := SORT(elems_rep, r);
		term_rep_set := SET (elems_rep_sort, v);
	
	// N := 32*32*3*1000;
	// Layout_Target sumTerms(Layout_Target cumm, Layout_Target term) := TRANSFORM
    // SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term.mat_part, 1);
    // SELF := cumm;
  // END;
	// sumres := ROLLUP(a_sort, sumTerms(LEFT, RIGHT), partition_id);
	
	
	
mymy2 := ab_bb;
myformat2 := RECORD
    mymy2.node_id;
    mymy2.partition_id;
    mymy2.block_row;
    mymy2.block_col;
    mymy2.first_row;
    mymy2.part_rows;
    mymy2.first_col;
    mymy2.part_cols;
		//mymy2.mat_part;
		INTEGER real_node := STD.System.Thorlib.Node();
END;
	rslt := TABLE(mymy2,myformat2,LOCAL);
	//rslt := A;
  RETURN rslt;
END; // END WX_2

// w*x + repmat (b, 1,m)
//A_in :w
//B_in :x
//bb_in : bias vector (b)
WX_repmatb(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_bb, DATASET(Layout_Part) bb_in, PBblas.IMatrix_Map map_c, REAL8 beta=0.0) := FUNCTION
SET OF PBblas.Types.value_t empty_array := [];
// First check maps for compatability.  Normalize for transpose operations.
  a_matrix_rows := map_a.matrix_rows;
  a_matrix_cols := map_a.matrix_cols;
  a_row_blocks  := map_a.row_blocks;
  a_col_blocks  := map_a.col_blocks;
  b_matrix_rows := map_b.matrix_rows;
  b_matrix_cols := map_b.matrix_cols;
  b_row_blocks  := map_b.row_blocks;
  b_col_blocks  := map_b.col_blocks;
  c_matrix_rows := map_c.matrix_rows;
  c_matrix_cols := map_c.matrix_cols;
  c_row_blocks  := map_c.row_blocks;
  c_col_blocks  := map_c.col_blocks;
  A := ASSERT(A_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' + ' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(a_matrix_rows=c_matrix_rows AND a_row_blocks=c_row_blocks,
                    'A-C: ' + 'Arows: ' + a_matrix_rows +' Crows '+c_matrix_rows+' '+ 
                              'Ablocks: ' + a_row_blocks +' Cblocks '+c_row_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	B := ASSERT(B_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' +' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(b_matrix_cols=c_matrix_cols AND b_col_blocks=c_col_blocks,
                    'B-C: ' + 'Brows: ' + b_matrix_cols +' Ccols '+c_matrix_cols+' '+ 
                              'Bblocks: ' + b_row_blocks +' Cblocks '+c_col_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	
  //multiply
	
	Layout_Part mul2(Layout_Part b_part, Layout_Part a_part):=TRANSFORM
    part_id     := b_part.partition_id;    //arbitrary choice
    part_a_cols := a_part.part_cols;
    part_a_rows := a_part.part_rows;
    part_b_rows := b_part.part_rows;
    part_c_rows := map_c.part_rows(part_id);
    part_c_cols := map_c.part_cols(part_id);
    part_c_first_row  := map_c.first_row(part_id);
    part_c_first_col  := map_c.first_col(part_id);
    k := part_a_cols;
    SELF.partition_id := b_part.partition_id;
    SELF.node_id      := b_part.node_id;
    SELF.block_row    := b_part.block_row;
    SELF.block_col    := b_part.block_col;
    SELF.first_row    := map_c.first_row(part_id);
    SELF.part_rows    := part_c_rows;
    SELF.first_col    := part_c_first_col;
    SELF.part_cols    := part_c_cols;
    SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
                                    part_c_rows, part_c_cols, k,
                                    1.0, a_part.mat_part, b_part.mat_part,
                                    0.0, empty_array);
  END;
  ab_prod := JOIN(B, A,TRUE , mul2(LEFT,RIGHT),ALL); // Each A (weight matrix) is copied in each B (X matrix) node

// add bias vector to each columns of X
// each bias vector (b) is copied (ALL JOIN) to each node of X. The bias vector is then normalized to repeat it to the number of columns of X in that node and then the two are added
  Layout_Part addbias(Layout_Part cumm, Layout_Part term) := TRANSFORM
		cumm_part_cols := cumm.part_cols;
    N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
		Elem := {PBblas.Types.value_t v};
		Elem_r := {PBblas.Types.value_t v, UNSIGNED r};
		elems := DATASET(term.mat_part, Elem);
		Elem_r rep (Elem l, UNSIGNED c) := TRANSFORM
			SELF.r := c;
			SELF := l;
		END;
		elems_rep := NORMALIZE(elems, cumm_part_cols, rep(LEFT, COUNTER));// each value in bias vector is repeated cumm_part_cols number of times (as the numebr of cumm columns in this node)
		elems_rep_sort := SORT(elems_rep, r);
		term_rep_set := SET (elems_rep_sort, v);
    SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term_rep_set, 1);
		SELF.partition_id := cumm.partition_id;
    SELF := cumm;
  END;
	
	 ab_bb := JOIN(ab_prod, bb_in,TRUE , addbias(LEFT,RIGHT),ALL);
	//rslt := A;
  RETURN ab_bb;
END; // END WX_repmatb
  //retunrs the sigmoid(WX+b)  
WX_repmatb_sig(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_bb, DATASET(Layout_Part) bb_in, PBblas.IMatrix_Map map_c, REAL8 beta=0.0) := FUNCTION
SET OF PBblas.Types.value_t empty_array := [];
// First check maps for compatability.  Normalize for transpose operations.
  a_matrix_rows := map_a.matrix_rows;
  a_matrix_cols := map_a.matrix_cols;
  a_row_blocks  := map_a.row_blocks;
  a_col_blocks  := map_a.col_blocks;
  b_matrix_rows := map_b.matrix_rows;
  b_matrix_cols := map_b.matrix_cols;
  b_row_blocks  := map_b.row_blocks;
  b_col_blocks  := map_b.col_blocks;
  c_matrix_rows := map_c.matrix_rows;
  c_matrix_cols := map_c.matrix_cols;
  c_row_blocks  := map_c.row_blocks;
  c_col_blocks  := map_c.col_blocks;
  A := ASSERT(A_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' + ' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(a_matrix_rows=c_matrix_rows AND a_row_blocks=c_row_blocks,
                    'A-C: ' + 'Arows: ' + a_matrix_rows +' Crows '+c_matrix_rows+' '+ 
                              'Ablocks: ' + a_row_blocks +' Cblocks '+c_row_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	B := ASSERT(B_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'No' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'No' +' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(b_matrix_cols=c_matrix_cols AND b_col_blocks=c_col_blocks,
                    'B-C: ' + 'Brows: ' + b_matrix_cols +' Ccols '+c_matrix_cols+' '+ 
                              'Bblocks: ' + b_row_blocks +' Cblocks '+c_col_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
	
  //multiply
	
	Layout_Part mul2(Layout_Part b_part, Layout_Part a_part):=TRANSFORM
    part_id     := b_part.partition_id;    //arbitrary choice
    part_a_cols := a_part.part_cols;
    part_a_rows := a_part.part_rows;
    part_b_rows := b_part.part_rows;
    part_c_rows := map_c.part_rows(part_id);
    part_c_cols := map_c.part_cols(part_id);
    part_c_first_row  := map_c.first_row(part_id);
    part_c_first_col  := map_c.first_col(part_id);
    k := part_a_cols;
    SELF.partition_id := b_part.partition_id;
    SELF.node_id      := b_part.node_id;
    SELF.block_row    := b_part.block_row;
    SELF.block_col    := b_part.block_col;
    SELF.first_row    := map_c.first_row(part_id);
    SELF.part_rows    := part_c_rows;
    SELF.first_col    := part_c_first_col;
    SELF.part_cols    := part_c_cols;
    SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
                                    part_c_rows, part_c_cols, k,
                                    1.0, a_part.mat_part, b_part.mat_part,
                                    0.0, empty_array);
  END;
  ab_prod := JOIN(B, A,TRUE , mul2(LEFT,RIGHT),ALL); // Each A (weight matrix) is copied in each B (X matrix) node

// add bias vector to each columns of X
// each bias vector (b) is copied (ALL JOIN) to each node of X. The bias vector is then normalized to repeat it to the number of columns of X in that node and then the two are added
  Layout_Part addbias(Layout_Part cumm, Layout_Part term) := TRANSFORM
		cumm_part_cols := cumm.part_cols;
    N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
		Elem := {PBblas.Types.value_t v};
		Elem_r := {PBblas.Types.value_t v, UNSIGNED r};
		elems := DATASET(term.mat_part, Elem);
		Elem_r rep (Elem l, UNSIGNED c) := TRANSFORM
			SELF.r := c;
			SELF := l;
		END;
		elems_rep := NORMALIZE(elems, cumm_part_cols, rep(LEFT, COUNTER));// each value in bias vector is repeated cumm_part_cols number of times (as the numebr of cumm columns in this node)
		elems_rep_sort := SORT(elems_rep, r);
		term_rep_set := SET (elems_rep_sort, v);
    SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term_rep_set, 1);
		SELF.partition_id := cumm.partition_id;
    SELF := cumm;
  END;
	
	 ab_bb := JOIN(ab_prod, bb_in,TRUE , addbias(LEFT,RIGHT),ALL);
	//rslt := A;
  RETURN ab_bb;
END; // END WX_repmatb_sig
		
		

		
		
		
		
		//((W2'*d3)+beta*repmat(sparsity_delta,1,m))
		// w'*x + beta * repmat (b, 1,m)
		//A_in :w
		//B_in :x
		//bb_in : bias vector (b)
		
WtX_repmatb(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_bb, DATASET(Layout_Part) bb_in, PBblas.IMatrix_Map map_c, REAL8 beta=0.0) := FUNCTION
	SET OF PBblas.Types.value_t empty_array := [];
	// First check maps for compatability.  Normalize for transpose operations.
		a_matrix_rows := map_a.matrix_cols;
		a_matrix_cols := map_a.matrix_rows;
		a_row_blocks  := map_a.col_blocks;
		a_col_blocks  := map_a.row_blocks;
		b_matrix_rows := map_b.matrix_rows;
		b_matrix_cols := map_b.matrix_cols;
		b_row_blocks  := map_b.row_blocks;
		b_col_blocks  := map_b.col_blocks;
		c_matrix_rows := map_c.matrix_rows;
		c_matrix_cols := map_c.matrix_cols;
		c_row_blocks  := map_c.row_blocks;
		c_col_blocks  := map_c.col_blocks;
		A := ASSERT(A_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'YES' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'NO' + ' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(a_matrix_rows=c_matrix_rows AND a_row_blocks=c_row_blocks,
                    'A-C: ' + 'Arows: ' + a_matrix_rows +' Crows '+c_matrix_rows+' '+ 
                              'Ablocks: ' + a_row_blocks +' Cblocks '+c_row_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
  B := ASSERT(B_in,
              ASSERT(a_matrix_cols=b_matrix_rows AND a_col_blocks=b_row_blocks,
                    'A-B: ' + 'A is ' + map_a.matrix_rows+'x'+map_a.matrix_cols +
                    ' Trans: '+ 'YES' + ' and B is ' + map_b.matrix_rows+'x'+map_b.matrix_cols +
                    ' Trans: '+ 'NO' +' ' +PBblas.Constants.Dimension_Incompat, FAIL),
              ASSERT(b_matrix_cols=c_matrix_cols AND b_col_blocks=c_col_blocks,
                    'B-C: ' + 'Brows: ' + b_matrix_cols +' Ccols '+c_matrix_cols+' '+ 
                              'Bblocks: ' + b_row_blocks +' Cblocks '+c_col_blocks+' '+
                              PBblas.Constants.Dimension_Incompat, FAIL));
		
		//multiply
		
		Layout_Part mul2(Layout_Part b_part, Layout_Part a_part):=TRANSFORM
			part_id     := b_part.partition_id;    //arbitrary choice
			part_a_cols := a_part.part_cols;
			part_a_rows := a_part.part_rows;
			part_b_rows := b_part.part_rows;
			part_c_rows := map_c.part_rows(part_id);
			part_c_cols := map_c.part_cols(part_id);
			part_c_first_row  := map_c.first_row(part_id);
			part_c_first_col  := map_c.first_col(part_id);
			k := part_a_rows;
			SELF.partition_id := b_part.partition_id;
			SELF.node_id      := b_part.node_id;
			SELF.block_row    := b_part.block_row;
			SELF.block_col    := b_part.block_col;
			SELF.first_row    := map_c.first_row(part_id);
			SELF.part_rows    := part_c_rows;
			SELF.first_col    := part_c_first_col;
			SELF.part_cols    := part_c_cols;
			SELF.mat_part     := PBblas.BLAS.dgemm(TRUE, FALSE,
																			part_c_rows, part_c_cols, k,
																			1.0, a_part.mat_part, b_part.mat_part,
																			0.0, empty_array);
		END;
		ab_prod := JOIN(B, A,TRUE , mul2(LEFT,RIGHT),ALL); // Each A (weight matrix) is copied in each B (X matrix) node



// Apply beta
  Layout_Part applyBeta(Layout_Part part) := TRANSFORM
    SELF.mat_part := PBblas.BLAS.dscal(map_bb.matrix_rows*map_bb.matrix_cols,
                                beta, part.mat_part, 1);
    SELF:= part;
  END;
  bb_beta := PROJECT(bb_in, applyBeta(LEFT), LOCAL);
	// add the vector to each columns of X
	// each vector is copied (ALL JOIN) to each node of X. The vector is then normalized to repeat it to the number of columns of X in that node and then the two are added
		Layout_Part addvec(Layout_Part cumm, Layout_Part term) := TRANSFORM
			cumm_part_cols := cumm.part_cols;
			N := map_c.part_rows(cumm.partition_id) * map_c.part_cols(cumm.partition_id);
			Elem := {PBblas.Types.value_t v};
			Elem_r := {PBblas.Types.value_t v, UNSIGNED r};
			elems := DATASET(term.mat_part, Elem);
			Elem_r rep (Elem l, UNSIGNED c) := TRANSFORM
				SELF.r := c;
				SELF := l;
			END;
			elems_rep := NORMALIZE(elems, cumm_part_cols, rep(LEFT, COUNTER));// each value in bias vector is repeated cumm_part_cols number of times (as the numebr of cumm columns in this node)
			elems_rep_sort := SORT(elems_rep, r);
			term_rep_set := SET (elems_rep_sort, v);
			SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term_rep_set, 1);
			SELF.partition_id := cumm.partition_id;
			SELF := cumm;
		END;
		
		 ab_bb := JOIN(ab_prod, bb_beta,TRUE , addvec(LEFT,RIGHT),ALL);

		//rslt := A;
		RETURN ab_bb;
END; // END WtX_repmatb
		
		
		
		// the input is a matrix in PBblas format where only columns are partitions
		//B_in : ones vector which is partitioned among nodes
		// map_c is the result's map
		col_sum(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_c) := FUNCTION
			SET OF PBblas.Types.value_t empty_array := [];
				Layout_Part mul(Layout_Part a_part, Layout_Part b_part):=TRANSFORM
					part_id     := 1;    //arbitrary choice
					part_a_cols := a_part.part_cols;
					part_a_rows := a_part.part_rows;
					part_b_rows := b_part.part_rows;
					part_c_rows := map_c.part_rows(part_id);
					part_c_cols := map_c.part_cols(part_id);
					part_c_first_row  := map_c.first_row(part_id);
					part_c_first_col  := map_c.first_col(part_id);
					k := part_a_cols;
					SELF.partition_id := 1;
					SELF.node_id      := a_part.node_id;
					SELF.block_row    := 1;
					SELF.block_col    := 1;
					SELF.first_row    := map_c.first_row(part_id);
					SELF.part_rows    := part_c_rows;
					SELF.first_col    := part_c_first_col;
					SELF.part_cols    := part_c_cols;
					SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
																					part_c_rows, part_c_cols, k,
																					1.0, a_part.mat_part, b_part.mat_part,
																					0.0, empty_array);
			END;
			col_sum_part := JOIN (A_in, B_in, LEFT.partition_id = RIGHT.partition_id, mul(LEFT, RIGHT), LOCAL );
			
			Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := le.part_rows ;
				SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1);
				SELF := le;
			END;
			//rslt := ROLLUP(col_sum_part, addup(LEFT, RIGHT), partition_id); // overload becasue of grouping
			rslt := ROLLUP(col_sum_part,TRUE, addup(LEFT, RIGHT)); // no groupijng, reduces overload
			//distribute to node one
			RETURN rslt; 
		END;//END Col_Sum
		
		
		col_mean(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_c) := FUNCTION
		
			Num := map_a.matrix_cols;
			Num_1 := 1/Num;
			SET OF PBblas.Types.value_t empty_array := [];
				Layout_Part mul(Layout_Part a_part, Layout_Part b_part):=TRANSFORM
					part_id     := 1;    //arbitrary choice
					part_a_cols := a_part.part_cols;
					part_a_rows := a_part.part_rows;
					part_b_rows := b_part.part_rows;
					part_c_rows := map_c.part_rows(part_id);
					part_c_cols := map_c.part_cols(part_id);
					part_c_first_row  := map_c.first_row(part_id);
					part_c_first_col  := map_c.first_col(part_id);
					k := part_a_cols;
					SELF.partition_id := 1;
					SELF.node_id      := 0;
					SELF.block_row    := 1;
					SELF.block_col    := 1;
					SELF.first_row    := map_c.first_row(part_id);
					SELF.part_rows    := part_c_rows;
					SELF.first_col    := part_c_first_col;
					SELF.part_cols    := part_c_cols;
					SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
																					part_c_rows, part_c_cols, k,
																					Num_1, a_part.mat_part, b_part.mat_part,
																					0.0, empty_array);
			END;
			col_sum_part := JOIN (A_in, B_in, LEFT.partition_id = RIGHT.partition_id, mul(LEFT, RIGHT), LOCAL );
			
			Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := le.part_rows ;
				SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1);
				SELF := le;
			END;
			rslt := ROLLUP(col_sum_part,TRUE, addup(LEFT, RIGHT)); // this form of rollup avoid grouping, ROLLUP(col_sum_part,addup(LEFT, RIGHT, partiion_id)) cause all the records with the same partiion_id
			//get grouped to one node which cause a lot of overload. The current rollup form avoidds grouping and improves performance
			final_rslt := DISTRIBUTE (rslt, node_id); 
			//distribute to node one
			RETURN final_rslt; 
		END;//END colmean
		
		
		// This function gets two big matrices which are distributed over all nodes and generate a final relatively smaller matrix which is on one node
		// this is used for weight gradient calculation where for example a h*m matrix is multiplied by a m*f matrix. PBblas will distribute all partitions in the first and second matrix to only one node which final matrix is in
		// this causes overhead, to avoid that we multiply each col partition of first matrix with a row partition of the second matrix in each node, the final generated matrices are added up to generate the final matrix
		// this way, we don't change the distribution of the first and second matrices
		big_big_small(PBblas.IMatrix_Map map_a, DATASET(Layout_Part) A_in, PBblas.IMatrix_Map map_b, DATASET(Layout_Part) B_in, PBblas.IMatrix_Map map_c, PBblas.Types.value_t alph=1.0) := FUNCTION
			SET OF PBblas.Types.value_t empty_array := [];
				Layout_Part mul(Layout_Part a_part, Layout_Part b_part):=TRANSFORM
					part_id     := 1;    //arbitrary choice
					part_a_cols := a_part.part_cols;
					part_a_rows := a_part.part_rows;
					part_b_rows := b_part.part_rows;
					part_b_cols := b_part.part_cols;
					k := part_a_cols;
					SELF.partition_id := part_id;
					SELF.node_id      := 0;
					SELF.block_row    := 1;
					SELF.block_col    := 1;
					SELF.first_row    := map_c.first_row(part_id);
					SELF.part_rows    := map_c.part_rows(part_id);
					SELF.first_col    := map_c.first_col(part_id);
					SELF.part_cols    := map_c.part_cols(part_id);
					SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, TRUE,
																					part_a_rows, part_b_rows, k,
																					alph, a_part.mat_part, b_part.mat_part,
																					0.0, empty_array);
			END;
			mul_part := JOIN (A_in, B_in, LEFT.partition_id = RIGHT.partition_id, mul(LEFT, RIGHT), LOCAL );
			
			Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := le.part_rows * le.part_cols ;
				SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1);
				SELF := le;
			END;
			
			Layout_Part addup_it(Layout_Part le, Layout_Part ri) := TRANSFORM
				N := ri.part_rows * ri.part_cols ;
				SELF.mat_part := IF (le.partition_id=0, ri.mat_part, PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1));
				SELF := ri;
			END;
			//rslt := ROLLUP(mul_part, addup(LEFT, RIGHT), partition_id); // since the results of rohat is used in a ALL join, no need to distribute this to node one to be consistent with PBblas
			//rslt := ITERATE(mul_part, addup_it(LEFT, RIGHT));// using rollup cause the graph to Group all the records which are distributed between all node to only one record and then do the operation, It takes a long time to GROUP all thoese partitions in one node and we avoid it by using ITERATE instead of ROLLUP

      rslt := ROLLUP(mul_part, TRUE, addup(LEFT, RIGHT));
			final_rslt := DISTRIBUTE (rslt, node_id); 

// a_part := A_in[1];
// b_part := B_in[1];
		 // RETURN PBblas.BLAS.dgemm(FALSE, TRUE,
																					// a_part.part_rows, b_part.part_rows, a_part.part_cols,
																					// 1.0, a_part.mat_part, b_part.mat_part,
																					// 0.0, empty_array);
																					
			RETURN final_rslt;
		END;// END big_big_small
		
		
		
		
		//extract sparse autoencoder parameters
    ddist := TrainData;
    m := CostFunc_params(id=1)[1].value;
    num_feat := CostFunc_params(id=2)[1].value;
    num_hid := CostFunc_params(id=3)[1].value;
    part_rows := CostFunc_params(id=4)[1].value;
    part_cols := CostFunc_params(id=5)[1].value;
    BETA := CostFunc_params(id=6)[1].value;
    sparsityParam := CostFunc_params(id=7)[1].value;
    LAMBDA := CostFunc_params(id=8)[1].value;
    
    m_1 := 1/m;
    sparsityParam_ := -1*sparsityParam;
    sparsityParam_1 := 1-sparsityParam;
    
     sizeRec := RECORD
      PBblas.Types.dimension_t m_rows;
      PBblas.Types.dimension_t m_cols;
      PBblas.Types.dimension_t f_b_rows;
      PBblas.Types.dimension_t f_b_cols;
    END;
    sizeTable := DATASET([{num_feat,m,num_feat,part_cols}], sizeRec);
    
    //Create block matrix d
    dmap := PBblas.Matrix_Map(num_feat,m,num_feat,part_cols);
    
    //Create block matrix Ytmp
    Ymap := dmap;
    Ydist := ddist;
    //Creat block matrices for weights

    w1map := PBblas.Matrix_Map(num_hid, num_feat, num_hid, num_feat);
    w2map := PBblas.Matrix_Map(num_feat, num_hid, num_feat, num_hid);
     //each bias vector is converted to block format
    
    b1vecmap := PBblas.Matrix_Map(num_hid, 1, num_hid, 1);
    b2vecmap := PBblas.Matrix_Map(num_feat, 1, num_feat, 1);
    
    w1_partitions := w1map.partitions_used;
    w2_partitions := w2map.partitions_used;
    b1_partitions := b1vecmap.partitions_used;
    b2_partitions := b2vecmap.partitions_used;
    PBblas.Types.MUElement minuspart(Layout_Part l, UNSIGNED8 c ) := TRANSFORM
      SELF.partition_id := l.partition_id - c;
      SElF.no := 1;
      SELF := l;
    END;
    //w1m := w1dist;
    w1dist := theta (partition_id <= w1_partitions);
    //w2m := w2dist;
    w2m_ := theta (partition_id > w1_partitions AND partition_id <= w1_partitions + w2_partitions);
    w2dist  := PROJECT (w2m_, minuspart(LEFT, w1_partitions ),LOCAL);
    //b1v := b1vecdist;
    b1v_ := theta (partition_id > w1_partitions + w2_partitions AND partition_id <= w1_partitions + w2_partitions + b1_partitions);
    b1vecdist  := PROJECT (b1v_, minuspart (LEFT, w1_partitions + w2_partitions),LOCAL);
    
    //b2v := b2vecdist;
    b2v_ := theta (partition_id > w1_partitions + w2_partitions + b1_partitions AND partition_id <= w1_partitions + w2_partitions + b1_partitions + b2_partitions);
    b2vecdist  := PROJECT (b2v_, minuspart (LEFT, w1_partitions + w2_partitions + b1_partitions),LOCAL);
    
     //functions used
    PBblas.Types.value_t sp_reci(PBblas.Types.value_t v,PBblas.Types.dimension_t r,PBblas.Types.dimension_t c) := sparsityParam_/v;
    PBblas.Types.value_t sp_1_reci(PBblas.Types.value_t v,PBblas.Types.dimension_t r,PBblas.Types.dimension_t c) := sparsityParam_1/(1-v);
    //sparsity_delta=((-sparsityParam./rhohat)+((1-sparsityParam)./(1.-rhohat)));
    PBblas.Types.value_t sp_delta(PBblas.Types.value_t v,PBblas.Types.dimension_t r,PBblas.Types.dimension_t c) := (sparsityParam_/v)+(sparsityParam_1/(1-v));
    PBblas.Types.value_t siggrad(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := v*(1.0-v);
    PBblas.Types.value_t sigmoid(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := 1/(1+exp(-1*v));
    PBblas.Types.value_t pow2(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := v*v;
    //maps used
    b1map := PBblas.Matrix_Map(num_hid, m, num_hid, part_cols);
    b2map := PBblas.Matrix_Map(num_feat, m, num_feat, part_cols);
    a2map := b1map;
    a3map := b2map;
    HL_nodes := num_hid;//number of nodes in the hidden layer
    Hiddmap := b1vecmap;
    //onevec for calculating rhohat
    Ones_VecMap := PBblas.Matrix_Map(m, 1, part_cols, 1);
    //New Vector Generator
    Layout_Cell gen(UNSIGNED4 c, UNSIGNED4 NumRows) := TRANSFORM
      SELF.x := ((c-1) % NumRows) + 1;
      SELF.y := ((c-1) DIV NumRows) + 1;
      SELF.v := 1;
    END;
    //Create Ones Vector for the calculations in the step fucntion
    Ones_Vec := DATASET(m, gen(COUNTER, m));
    //Ones_Vecdist := DMAT.Converted.FromCells(Ones_VecMap, Ones_Vec);
    //FF2 returns a2
    FF2_(DATASET(Layout_Part) w1, DATASET(Layout_Part) b1v):= FUNCTION
      //b1m = repmat(b1v,1,m)
      b1m := PBblas.PB_dgemm(FALSE, TRUE, 1.0,b1vecmap, b1v, Ones_VecMap, Ones_Vecdist, b1map);
      //z2 = w1*X+b1;
      //z2 := PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1, dmap, ddist, b1map, b1m, 1.0); // gives MP closed error
      z2_ := PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1, dmap, ddist, b1map);
      z2 := PBblas.PB_daxpy(1.0, z2_, b1m);
      //a2 = sigmoid (z2);
      a2 := PBblas.Apply2Elements(b1map, z2, sigmoid);
      RETURN a2;
    END;//END FF2_
		
		
		
		//returns a2
		 FF2(DATASET(Layout_Part) w1, DATASET(Layout_Part) b1v):= FUNCTION
      //z2=w1*x+repmat(b1,1,m)
			z2 := WX_repmatb(w1map, w1, dmap, ddist, b1vecmap, b1v, b1map, 0.0);
      //a2 = sigmoid (z2);
      a2 := PBblas.Apply2Elements(b1map, z2, sigmoid);
      RETURN a2;
     END;//END FF2
		
		
    //FF3 returns a3
    FF3_(DATASET(Layout_Part) w2,DATASET(Layout_Part) b2v, DATASET(Layout_Part) a2 ):= FUNCTION
      //b2m = repmat(b2v,1,m)
      b2m := PBblas.PB_dgemm(FALSE, TRUE, 1.0,b2vecmap, b2v, Ones_VecMap, Ones_Vecdist, b2map);
      //z3 = w2*a2+b2;
      //z3 := PBblas.PB_dgemm(FALSE, FALSE,1.0,w2map, w2, a2map, a2, b2map,b2m, 1.0); //gives MP closed error
      z3_ := PBblas.PB_dgemm(FALSE, FALSE,1.0,w2map, w2, a2map, a2, b2map);
      z3 := PBblas.PB_daxpy(1.0, z3_, b2m);
      //a3 = sigmoid (z3)
      a3 := PBblas.Apply2Elements(b2map, z3, sigmoid);
      RETURN a3;
    END;//END FF3_
		
		 //FF3 returns a3
    FF3(DATASET(Layout_Part) w2,DATASET(Layout_Part) b2v, DATASET(Layout_Part) a2 ):= FUNCTION
		  //z3 = w2*a2 + repmat(b2,1,m)
			z3 := WX_repmatb(w2map, w2, a2map, a2, b2vecmap, b2v, b2map, 0.0);
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
		
		DELTA3_ (DATASET(Layout_Part) a3 ) := FUNCTION
      //calculate delta for the last layer (3rd layer)
      //y=X;
      //d3=-(y-a3).*(a3.*(1-a3));
      siggrad_a3 := PBblas.Apply2Elements(a3map, a3, siggrad);
      a3_y := PBblas.PB_daxpy(-1, ddist, a3);
      d3 := PBblas.HadamardProduct(a3map, a3_y, siggrad_a3);
      RETURN d3 ;
    END;//END DELTA3_
    //DELTA2 retunrs d2
    rohat_ (DATASET(Layout_Part) a2) := FUNCTION
      rh := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a2map, a2, Ones_VecMap, Ones_Vecdist, Hiddmap);
      RETURN rh;
    END;
		rohat (DATASET(Layout_Part) a2) := FUNCTION
			rh := col_mean(a2map, a2, Ones_VecMap, Ones_Vecdist, Hiddmap);
      RETURN rh;
    END;
    DELTA2_ (DATASET(Layout_Part) w2, DATASET(Layout_Part) a2, DATASET(Layout_Part) d3, DATASET(Layout_Part) rhat) := FUNCTION
      //calculate delta for 2nd layer
      //rhohat=mean(a2,2);
      //sparsity_delta=((-sparsityParam./rhohat)+((1-sparsityParam)./(1.-rhohat)));
      //d2=((W2'*d3)+beta*repmat(sparsity_delta,1,m)) .* (a2.*(1-a2));
      //rhohat := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a2map, a2, Ones_VecMap, Ones_Vecdist, Hiddmap);
      rhohat := rhat;
      sparsity_delta := PBblas.Apply2Elements(Hiddmap, rhohat, sp_delta);
      siggrad_a2 := PBblas.Apply2Elements(a2map, a2, siggrad);
      repmat_sparsity_delta := PBblas.PB_dgemm(FALSE, TRUE, 1.0,  Hiddmap, sparsity_delta, Ones_VecMap, Ones_Vecdist, a2map);
      //d2_firstterm = (W2'*d3)+beta*repmat(sparsity_delta,1,m);
      //d2_firstterm := PBblas.PB_dgemm(TRUE, FALSE, 1.0, w2map, w2, a3map, d3, a2map, repmat_sparsity_delta, BETA); // MP close error
      d2_firstterm_ := PBblas.PB_dgemm(TRUE, FALSE, 1.0, w2map, w2, a3map, d3, a2map);
      d2_firstterm := PBblas.PB_daxpy(BETA, repmat_sparsity_delta, d2_firstterm_);
      d2 := PBblas.HadamardProduct(a2map, d2_firstterm, siggrad_a2);
      RETURN d2 ;
    END;
		
		
		DELTA2 (DATASET(Layout_Part) w2, DATASET(Layout_Part) a2, DATASET(Layout_Part) d3, DATASET(Layout_Part) rhat) := FUNCTION
      //calculate delta for 2nd layer
      //rhohat=mean(a2,2);
      //sparsity_delta=((-sparsityParam./rhohat)+((1-sparsityParam)./(1.-rhohat)));
      //d2=((W2'*d3)+beta*repmat(sparsity_delta,1,m)) .* (a2.*(1-a2));
      //rhohat := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a2map, a2, Ones_VecMap, Ones_Vecdist, Hiddmap);
      rhohat := rhat;
      sparsity_delta := PBblas.Apply2Elements(Hiddmap, rhohat, sp_delta);
      siggrad_a2 := PBblas.Apply2Elements(a2map, a2, siggrad);
      //repmat_sparsity_delta := PBblas.PB_dgemm(FALSE, TRUE, 1.0,  Hiddmap, sparsity_delta, Ones_VecMap, Ones_Vecdist, a2map);
      //d2_firstterm = (W2'*d3)+beta*repmat(sparsity_delta,1,m);
			d2_firstterm := WtX_repmatb(w2map, w2, a3map, d3, Hiddmap, sparsity_delta, a2map, BETA);
      d2 := PBblas.HadamardProduct(a2map, d2_firstterm, siggrad_a2);
      RETURN d2 ;
    END;
    //WeightGrad1 returns gradient for w1
    WeightGrad1_ (DATASET(Layout_Part) w1, DATASET(Layout_Part) d2) := FUNCTION
      w1_g := PBblas.PB_dgemm(FALSE, TRUE, m_1, a2map, d2, dmap, ddist, w1map, w1 ,LAMBDA );
      RETURN w1_g;
    END;
		WeightGrad1 (DATASET(Layout_Part) w1, DATASET(Layout_Part) d2) := FUNCTION
			w1_g_ := big_big_small(a2map, d2, dmap, ddist, w1map, m_1);
			w1_g  := PBblas.PB_daxpy(LAMBDA, w1, w1_g_);
      RETURN w1_g;
    END;
		
    //WeightGrad2 returns gradient for w2
    WeightGrad2_ (DATASET(Layout_Part) w2, DATASET(Layout_Part) d3, DATASET(Layout_Part) a2) := FUNCTION
      w2_g := PBblas.PB_dgemm(FALSE, TRUE, m_1, a3map, d3, a2map, a2, w2map, w2 ,LAMBDA );
      RETURN w2_g;
    END;
		
		WeightGrad2 (DATASET(Layout_Part) w2, DATASET(Layout_Part) d3, DATASET(Layout_Part) a2) := FUNCTION
			w2_g_ := big_big_small(a3map, d3, a2map, a2, w2map, m_1);
			w2_g  := PBblas.PB_daxpy(LAMBDA, w2, w2_g_);
      RETURN w2_g;
    END;
    //BiasGrad1 calculates the bias gradients for b1
    BiasGrad1 (DATASET(Layout_Part) d2) := FUNCTION
			b1_g := col_mean(a2map, d2, Ones_VecMap, Ones_Vecdist, b1vecmap);
      RETURN b1_g;
    END;
		
		BiasGrad1_ (DATASET(Layout_Part) d2) := FUNCTION
      b1_g := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a2map, d2, Ones_VecMap, Ones_Vecdist, b1vecmap);
      RETURN b1_g;
    END;
    //BiasGrad2 calculates the bias gradients for b2
    BiasGrad2_ (DATASET(Layout_Part) d3) := FUNCTION
      b2_g := PBblas.PB_dgemm(FALSE, FALSE, m_1,  a3map, d3, Ones_VecMap, Ones_Vecdist, b2vecmap);
      RETURN b2_g;
    END;
		
		BiasGrad2 (DATASET(Layout_Part) d3) := FUNCTION
      b2_g := col_mean(a3map, d3, Ones_VecMap, Ones_Vecdist, b2vecmap);
      RETURN b2_g;
    END;
		
		
    emptyL := DATASET([], Layout_Part);
    //theta is the input weight and bias parameters for the sparseautoencoder
    //parameters are the parameters for the sparseautoencoder function
    //train_d is the train data to learn sparse autoencoder and calculate the gradient and the cost based on taht
    // train_l is empty in the sparse autoencoder (because it is an unsupervised learning)
    SparseParam_CostGradients4 :=  FUNCTION
      PBblas.Types.MUElement minuspart(Layout_Part l, UNSIGNED8 c ) := TRANSFORM
        SELF.partition_id := l.partition_id - c;
        SElF.no := 1;
        SELF := l;
      END;
      w1m := w1dist;
      w2m := w2dist;
      b1v := b1vecdist;
      b2v := b2vecdist;
      a2 := FF2 (w1m, b1v);
			a2_ := FF2_ (w1m, b1v);
      a3 := FF3 (w2m, b2v, a2);
			a3_ := FF3_ (w2m, b2v, a2_);
      d3 := DELTA3 (a3);
			d3_ := DELTA3_ (a3_);
      rohat_a2 := rohat(a2);
			rohat_a2_ := rohat(a2_);
      d2 := DELTA2 (w2m, a2, d3,rohat_a2);
			d2_ := DELTA2_ (w2m, a2_, d3_,rohat_a2_);
      wg1 := WeightGrad1 (w1m, d2);
      wg2 := WeightGrad2 (w2m, d3, a2);
      bg1 := BiasGrad1 (d2);
      bg2 := BiasGrad2 (d3);
			
			
			wg1_ := WeightGrad1 (w1m, d2_);
      wg2_ := WeightGrad2 (w2m, d3_, a2_);
      bg1_ := BiasGrad1 (d2_);
      bg2_ := BiasGrad2 (d3_);
      //calculate cost
      //PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1, dmap, ddist, b1map, b1m, 1.0);
      // squared_error_cost= 0.5*sum(sum((x-a3).^2));
      // cost=(1/m)*squared_error_cost+(lambda/2)*(sum(W2(:).^2)+sum(W1(:).^2))+beta*sum(KL(sparsityParam,rhohat));
      squared_error_cost := 0.5*PBblas.SumElements(PBblas.Apply2Elements(dmap, PBblas.PB_daxpy(-1.0, a3, ddist), pow2));
      cost_term1 := (1/m)*squared_error_cost;
      cost_term2 := (lambda/2)* PBblas.SumElements(PBblas.Apply2Elements(w2map, w2m, pow2));
      cost_term3 := (lambda/2)* PBblas.SumElements(PBblas.Apply2Elements(w1map, w1m, pow2));
      PBblas.Types.value_t klterm(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := sparsityParam * LN(sparsityParam/v) + (1-sparsityParam) * LN ((1-sparsityParam)/(1-v));
      KL := PBblas.Apply2Elements (Hiddmap,rohat_a2,klterm);
      cost_term4 := beta * PBblas.SumElements(KL);
      cost := cost_term1 + cost_term2 + cost_term3 +cost_term4;    
      costField := DATASET ([{1,1,cost}],ML.Types.NumericField);
      one_map := PBblas.Matrix_Map(1,1,1,1);
      Cost_part_no := PBblas.MU.TO(ML.DMat.Converted.FromNumericFieldDS(costField,one_map),2);
      //convert w and b gradients to a datasets of layoutparts where partition_id differentiate them
      //w1_grad has partition_id from 1 to w1_partitions
      //w2_grad had partition_ds from w1_partitions+1 to w1_partitions + w2_partitions
      //b1_grad has partition_ids from w1_partitions + w2_partitions+1 to w1_partitions + w2_partitions+b1_partitions
      //b2_grad has partition_ids from w1_partitions + w2_partitions+b1_partitions+1 to w1_partitions + w2_partitions+b1_partitions+b2_partitions
      PBblas.Types.MUElement addpart(Layout_Part l, UNSIGNED8 c ) := TRANSFORM
        SELF.partition_id := l.partition_id + c;
        SElF.no := 1;
        SELF := l;
      END;
      wg1_reshape_no := Pbblas.MU.TO(wg1,1);
      wg2_reshape_no := PROJECT (wg2, addpart(LEFT, w1_partitions ), LOCAL);
      bg1_reshape_no := PROJECT (bg1, addpart (LEFT, w1_partitions + w2_partitions),LOCAL);
      bg2_reshape_no := PROJECT (bg2, addpart (LEFT, w1_partitions + w2_partitions + b1_partitions),LOCAL);
      theta_Part_no := wg1_reshape_no + wg2_reshape_no + bg1_reshape_no + bg2_reshape_no;
      
			//RETURN PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1dist, dmap, ddist, b1map);
			//RETURN WX(w1map, w1dist, dmap, ddist, b1map, w1dist,  1);
			//RETURN WX_repmatb(w1map, w1dist, dmap, ddist, b1vecmap, b1vecdist, b1map, 1.0) ;
			//RETURN col_sum(dmap, ddist, Ones_VecMap, Ones_Vecdist, b2vecmap);
			w1x_b1 := WX_repmatb(w1map, w1dist, dmap, ddist, b1vecmap, b1vecdist, b1map, 1.0);
			thisis := big_big_small(b1map, w1x_b1, dmap, ddist, PBblas.Matrix_Map(num_hid, num_feat, num_hid, num_feat));
			thisone := WtX_repmatb(w2map,w2dist, dmap, ddist, b1vecmap, b1vecdist, b1map, 5);
			
			
			
			//mymy2 := DELTA2 (w2m, a2, d3,rohat_a2) + DELTA2_ (w2m, a2, d3,rohat_a2);
			//mymy2 := big_big_small(a2map, d2, dmap, ddist, w1map, 8);
			//mymy2 := col_mean(a3map, d3, Ones_VecMap, Ones_Vecdist, b2vecmap);
			mymy2 := wg1 + wg2  + bg1 + bg2;
myformat2 := RECORD
    mymy2.node_id;
    mymy2.partition_id;
    mymy2.block_row;
    mymy2.block_col;
    mymy2.first_row;
    mymy2.part_rows;
    mymy2.first_col;
    mymy2.part_cols;
		//mymy2.no;
		//mymy2.mat_part;
		INTEGER real_node := STD.System.Thorlib.Node();
END;
	thisR := TABLE(mymy2,myformat2,LOCAL); 
	RETURN  theta_Part_no + Cost_part_no;
	//RETURN  theta_Part_no;

			//RETURN thisR;
      //RETURN theta_Part_no;
    END;//END SparseParam_CostGradients4  
    RETURN SparseParam_CostGradients4;
   END;//END SA_lbfgs_Compatible2_2









EXPORT Sparse_Autoencoder_lbfgs (INTEGER4 NumberofFeatures, INTEGER4 NumberofHiddenLayerNodes, UNSIGNED4 prows=0, UNSIGNED4 pcols=0, UNSIGNED4 Maxrows=0, UNSIGNED4 Maxcols=0) := MODULE
   //this is a un-supervised learning algorithm, no need for the labled data
   //m : number of input samples
   //NumberofFeatures : number of input features = number of input layer nodes in sparse autoencoder = number of output layer nodes in sparse autoencoder
   //NumberofHiddenLayerNodes : number of hidden layer nodes in Sparse Autoencoder
   
   SHARED SA_4(DATASET(Types.NumericField) X, DATASET(Mat.Types.Element) IntW1, DATASET(Mat.Types.Element) IntW2, DATASET(Mat.Types.Element) Intb1,  DATASET(Mat.Types.Element) Intb2, REAL8 BETA=0.1, REAL8 sparsityParam=0.1 , REAL8 LAMBDA=0.001, UNSIGNED2 MaxIter=100) := MODULE
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
    output_num := d_n;
    derivemap := IF(havemaxrowcol, PBblas.AutoBVMap(d_n, d_m,prows,pcols,maxrows, maxcols),
                   IF(havemaxrow, PBblas.AutoBVMap(d_n, d_m,prows,pcols,maxrows),
                      IF(havemaxcol, PBblas.AutoBVMap(d_n, d_m,prows,pcols,,maxcols),
                      PBblas.AutoBVMap(d_n, d_m,prows,pcols))));
    sizeTable := DATASET([{derivemap.matrix_rows,derivemap.matrix_cols,derivemap.part_rows(1),derivemap.part_cols(1)}], sizeRec);
    
    //Create block matrix d
    dmap := PBblas.Matrix_Map(sizeTable[1].m_rows,sizeTable[1].m_cols,sizeTable[1].f_b_rows,sizeTable[1].f_b_cols);
    ddist := DMAT.Converted.FromElement(d,dmap);
    //Creat block matrices for weights
    w1_mat := IntW1;
    //w1_mat_x := Mat.Has(w1_mat).Stats.Xmax;
		w1_mat_x := NumberofHiddenLayerNodes;
    // w1_mat_y := Mat.Has(w1_mat).Stats.Ymax;
		w1_mat_y := NumberofFeatures;
    w1map := PBblas.Matrix_Map(w1_mat_x, w1_mat_y, w1_mat_x, sizeTable[1].f_b_rows);
    w1dist := DMAT.Converted.FromElement(w1_mat,w1map);
    w2_mat := IntW2;
    w2_mat_x := w1_mat_y;
    w2_mat_y := w1_mat_x;
    w2map := PBblas.Matrix_Map(w2_mat_x, w2_mat_y, sizeTable[1].f_b_rows, w2_mat_y);
    w2dist := DMAT.Converted.FromElement(w2_mat,w2map);
    //each bias vector is converted to block format
    b1vec := Intb1;
    // b1vec_x := Mat.Has(b1vec).Stats.Xmax;
		b1vec_x := NumberofHiddenLayerNodes;
    b1vecmap := PBblas.Matrix_Map(b1vec_x, 1, b1vec_x, 1);
    b1vecdist := DMAT.Converted.FromElement(b1vec,b1vecmap);
    b2vec := Intb2;
    // b2vec_x := Mat.Has(b2vec).Stats.Xmax;
		b2vec_x := NumberofFeatures;
    b2vecmap := PBblas.Matrix_Map(b2vec_x, 1, sizeTable[1].f_b_rows, 1);
    b2vecdist := DMAT.Converted.FromElement(b2vec,b2vecmap);
    
    SA_param := DATASET([
    {1,1,m},
    {2,1,NumberofFeatures},
    {3,1,NumberofHiddenLayerNodes},
    {4,1,sizeTable[1].f_b_rows},
    {5,1,sizeTable[1].f_b_cols},
    {6,1,BETA},
    {7,1,sparsityParam},
    {8,1,LAMBDA}
    ], Types.NumericField);

    //Intialize Theta
    Layout_Part addpart_(Layout_Part l, INTEGER8 c ) := TRANSFORM
        SELF.partition_id := l.partition_id + c;
        SELF := l;
    END;
    w1_partitions := w1map.partitions_used;
    w2_partitions := w2map.partitions_used;
    b1_partitions := b1vecmap.partitions_used;
    b2_partitions := b2vecmap.partitions_used;
    w1_reshape := w1dist;
    w2_reshape := PROJECT (w2dist, addpart_(LEFT, w1_partitions ),LOCAL);
    b1_reshape := PROJECT (b1vecdist, addpart_ (LEFT, w1_partitions + w2_partitions), LOCAL);
    b2_reshape := PROJECT (b2vecdist, addpart_ (LEFT, w1_partitions + w2_partitions + b1_partitions),LOCAL);
    Inttheta := w1_reshape + w2_reshape + b1_reshape + b2_reshape;
		
		
		
		Ones_VecMap := PBblas.Matrix_Map(m, 1, sizeTable[1].f_b_cols, 1);
    //New Vector Generator
    Layout_Cell gen(UNSIGNED4 c, UNSIGNED4 NumRows) := TRANSFORM
      SELF.x := ((c-1) % NumRows) + 1;
      SELF.y := ((c-1) DIV NumRows) + 1;
      SELF.v := 1;
    END;
    //Create Ones Vector for the calculations in the step fucntion
    Ones_Vec := DATASET(m, gen(COUNTER, m));
    Ones_Vecdist := DMAT.Converted.FromCells(Ones_VecMap, Ones_Vec);
    
    emptyL := DATASET([], Layout_Part);
    
    CG := SA_lbfgs_Compatible ( Inttheta, SA_param, ddist , emptyL);
    CG2 := SA_lbfgs_Compatible ( Pbblas.MU.FROM(CG,1), SA_param, ddist , emptyL);
    
    paramnumber := 2*NumberofFeatures*NumberofHiddenLayerNodes + NumberofFeatures + NumberofHiddenLayerNodes;
    LBFGS_MAXitr := MaxIter;
    LBFGS_corrections := 100;
    lbfgs_results := Optimization (0, 0, 0, 0).MinFUNC(Inttheta,SA_param,ddist,emptyL,SA_lbfgs_Compatible, paramnumber,LBFGS_MAXitr, 0.00001, 0.000000001,  1000, LBFGS_corrections, 0, 0, 0,0) ;
		lbfgs_results2 := Optimization (0, 0, 0, 0).MinFUNC(Inttheta,SA_param,ddist,emptyL,SA_lbfgs_Compatible2, paramnumber,LBFGS_MAXitr, 0.00001, 0.000000001,  1000, LBFGS_corrections, 0, 0, 0,0) ;
		lbfgs_results2_2 := Optimization (0, 0, 0, 0).MinFUNC(Inttheta,SA_param,ddist,Ones_Vecdist,SA_lbfgs_Compatible2_2, paramnumber,LBFGS_MAXitr, 0.00001, 0.000000001,  1000, LBFGS_corrections, 0, 0, 0,0) ;
    //convert lbfgs_results to numericfield
    maxno := MAX(lbfgs_results, lbfgs_results.no);
    optTHETA_ := lbfgs_results(no=maxno);
		optTHETA_2 := lbfgs_results2(no=maxno);
		optTHETA_2_2 := lbfgs_results2_2(no=maxno);
    optTHETA_part := PROJECT(optTHETA_, TRANSFORM(Layout_Part, SELF := LEFT));

    Layout_Part minuspart(Layout_Part l, UNSIGNED8 c ) := TRANSFORM
      SELF.partition_id := l.partition_id - c;
      SELF := l;
    END;
    
    optW1 := optTHETA_part (partition_id <= w1_partitions);
    
    optW2_ := optTHETA_part (partition_id > w1_partitions AND partition_id <= w1_partitions + w2_partitions);
    optW2  := PROJECT (optW2_, minuspart(LEFT, w1_partitions ));
    
    optb1_ := optTHETA_part (partition_id > w1_partitions + w2_partitions AND partition_id <= w1_partitions + w2_partitions + b1_partitions);
    optb1  := PROJECT (optb1_, minuspart (LEFT, w1_partitions + w2_partitions));
    
    
    optb2_ := optTHETA_part (partition_id > w1_partitions + w2_partitions + b1_partitions AND partition_id <= w1_partitions + w2_partitions + b1_partitions + b2_partitions);
    optb2  := PROJECT (optb2_, minuspart (LEFT, w1_partitions + w2_partitions + b1_partitions));
    
    
    
    SAprm1 := optW1;
    SAprm2 := optW2;
    SAprm3 := optb1;
    SAprm4 := optb2;
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
    //EXPORT mod := ddist;
		EXPORT mod := SAprm_MUE_out;
		//EXPORT mod := CG;
		//EXPORT mod := SA_lbfgs_Compatible2 ( Inttheta, SA_param, ddist , emptyL);
		//EXPORT mod := Inttheta;
		//EXPORT mod := lbfgs_results;
		
		//EXPORT mod := PROJECT (ddist, TRANSFORM ({UNSIGNED node_id}, SELF := LEFT));
		//EXPORT mod := lbfgs_results;
		//EXPORT mod := SA_lbfgs_Compatible2 ( Inttheta, SA_param, ddist , emptyL);
		//EXPORT mod := SA_lbfgs_Compatible ( Inttheta, SA_param, ddist , emptyL);
		//EXPORT mod := Optimization (0, 0, 0, 0).MinFUNC(Inttheta,SA_param,ddist,emptyL,myfunc5, paramnumber,LBFGS_MAXitr, 0.00001, 0.000000001,  1000, LBFGS_corrections, 0, 0, 0,0) ;
		//EXPORT mod := Optimization (0, 0, 0, 0).MinFUNC(Inttheta,SA_param,ddist,emptyL,SA_lbfgs_Compatible, paramnumber,LBFGS_MAXitr, 0.00001, 0.000000001,  1000, LBFGS_corrections, 0, 0, 0,0) ;
   END;//END SA_4
   
  EXPORT LearnC (DATASET(Types.NumericField) indep, DATASET(Mat.Types.Element) IntW1, DATASET(Mat.Types.Element) IntW2, DATASET(Mat.Types.Element) Intb1,  DATASET(Mat.Types.Element) Intb2, REAL8 BETA=0.1, REAL8 sparsityParam=0.1 , REAL8 LAMBDA=0.001, UNSIGNED2 MaxIter=100) := SA_4(Indep,IntW1,IntW2,Intb1,Intb2, BETA,sparsityParam,LAMBDA, MaxIter).mod;
	EXPORT test (DATASET(Types.NumericField) indep, DATASET(Mat.Types.Element) IntW1, DATASET(Mat.Types.Element) IntW2, DATASET(Mat.Types.Element) Intb1,  DATASET(Mat.Types.Element) Intb2, REAL8 BETA=0.1, REAL8 sparsityParam=0.1 , REAL8 LAMBDA=0.001, UNSIGNED2 MaxIter=100) := SA_4(Indep,IntW1,IntW2,Intb1,Intb2, BETA,sparsityParam,LAMBDA, MaxIter).mod;
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
                          //derivemap :=  PBblas.AutoBVMap(d_n, d_m,d_n,d_m);
    sizeTable := DATASET([{derivemap.matrix_rows,derivemap.matrix_cols,derivemap.part_rows(1),derivemap.part_cols(1)}], sizeRec);
    //Create block matrix d
    dmap := PBblas.Matrix_Map(sizeTable[1].m_rows,sizeTable[1].m_cols,sizeTable[1].f_b_rows,sizeTable[1].f_b_cols);
    ddist := DMAT.Converted.FromElement(d,dmap);
    //Creat block matrices for weights
    w1_mat := Mat.MU.From(Inputmod,1);
    w1_mat_x := Mat.Has(w1_mat).Stats.Xmax;
    w1_mat_y := Mat.Has(w1_mat).Stats.Ymax;
    w1map := PBblas.Matrix_Map(w1_mat_x, w1_mat_y, MIN([sizeTable[1].f_b_rows, w1_mat_x]) , sizeTable[1].f_b_rows);
    w1dist := DMAT.Converted.FromElement(w1_mat,w1map);
    //each bias vector is converted to block format
    b1vec := Mat.MU.From(Inputmod,3);
    b1vec_x := Mat.Has(b1vec).Stats.Xmax;
    b1vecmap := PBblas.Matrix_Map(b1vec_x, 1, MIN([b1vec_x,sizeTable[1].f_b_rows]) , 1);
    b1vecdist := DMAT.Converted.FromElement(b1vec,b1vecmap);
    //functions used
    PBblas.Types.value_t sigmoid(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := 1/(1+exp(-1*v));
    //maps used
    b1map := PBblas.Matrix_Map(b1vec_x, m, MIN([b1vec_x,sizeTable[1].f_b_rows]), sizeTable[1].f_b_cols);
    a2map := b1map;
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
    Ones_Vec := DATASET(m, gen(COUNTER, m));
    Ones_Vecdist := DMAT.Converted.FromCells(Ones_VecMap, Ones_Vec);
    //b1m = repmat(b1v,1,m)
    b1m := PBblas.PB_dgemm(FALSE, TRUE, 1.0,b1vecmap, b1vecdist, Ones_VecMap, Ones_Vecdist, b1map);
    //z2 = w1*X+b1;
    //z2 := PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1dist, dmap, ddist, b1map, b1m, 1.0); // MP closed error
    z2_ := PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1dist, dmap, ddist, b1map);
    z2 := PBblas.PB_daxpy(1.0, b1m, z2_);
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
END;//END Sparse_Autoencoder_lbfgs





//in order to run the optimization algorithm with whole data we should have mini_num =1, this way data is distributed among all nodes and no extrat partitioning is done in each node

EXPORT Sparse_Autoencoder_lbfgs_part (INTEGER4 NumberofFeatures, INTEGER4 NumberofHiddenLayerNodes, UNSIGNED4 prows=0, UNSIGNED4 pcols=0, UNSIGNED4 mini_num=1) := MODULE
   //this is a un-supervised learning algorithm, no need for the labled data
   //m : number of input samples
   //NumberofFeatures : number of input features = number of input layer nodes in sparse autoencoder = number of output layer nodes in sparse autoencoder
   //NumberofHiddenLayerNodes : number of hidden layer nodes in Sparse Autoencoder
   
   SHARED SA_4(DATASET(Types.NumericField) X, DATASET(Mat.Types.Element) IntW1, DATASET(Mat.Types.Element) IntW2, DATASET(Mat.Types.Element) Intb1,  DATASET(Mat.Types.Element) Intb2, REAL8 BETA=0.1, REAL8 sparsityParam=0.1 , REAL8 LAMBDA=0.001, UNSIGNED2 MaxIter=100, UNSIGNED LBFGS_corrections = 100) := MODULE
    dt := Types.ToMatrix (X);
    dTmp := dt;
    //d := dt; //in the entire of the calculations we work with the d matrix that each sample is presented in one column
    d := Mat.Trans(dTmp);

   //Map for Matrix d.
    // havemaxrow := maxrows > 0;
    // havemaxcol := maxcols > 0;
    //havemaxrowcol := havemaxrow and havemaxcol;
    dstats := Mat.Has(d).Stats;
    d_n := dstats.XMax;
    d_m := dstats.YMax;
		m := d_m;// number of samples
		f := NumberofFeatures; // number of features
    output_num := d_n;

    
    //Create block matrix d
    dmap_mini := PBblas.Matrix_Map(f,m,prows,pcols);
		dmap := PBblas.Matrix_Map(f,m,prows,pcols);// we use this map only to partiion the data, however the distribution is done in a fashion where each big column partition of the data which includes smaller row partitions ends up in one node
    ddist_tmp := DMAT.Converted.FromElement(d,dmap);
		//number of mini batches in each bigger batch : bigger batches have pcols columns, mini batches have minibatch column. all the mini batches inside a bigger match are distributed in the same node becasue we want to make sure all the big batches end up in the same node
		//redistribute ddist in a way that the main partitioning is based on matrix columns, all partitions in the same column will end up in the same node
		d_row_b := dmap.row_blocks;
		layout_part new_node_id (layout_part dd) := TRANSFORM
			// part_id := dd.partition_id;
			// node_part_id := ((part_id-1) DIV d_row_b )+1;// this i sthe partition id if we partition only the matrix columns
			// new_node_id := dmap.assigned_node(node_part_id);
			new_node_id := dmap.assigned_node(((dd.block_col-1) DIV mini_num)+1);
			SELF.node_id := new_node_id;
			//SELF.partition_id := new_part_id;
			SELF := dd;
		END;
		ddist_ := PROJECT (ddist_tmp, new_node_id (LEFT), LOCAL );
		ddist := DISTRIBUTE(ddist_, node_id);
    //Creat block matrices for weights
    w1_mat := IntW1;
    //w1_mat_x := Mat.Has(w1_mat).Stats.Xmax;
		w1_mat_x := NumberofHiddenLayerNodes;
    // w1_mat_y := Mat.Has(w1_mat).Stats.Ymax;
		w1_mat_y := NumberofFeatures;
    w1map := PBblas.Matrix_Map(w1_mat_x, w1_mat_y, w1_mat_x, prows);
    w1dist := DMAT.Converted.FromElement(w1_mat,w1map);
    w2_mat := IntW2;
    w2_mat_x := w1_mat_y;
    w2_mat_y := w1_mat_x;
    w2map := PBblas.Matrix_Map(w2_mat_x, w2_mat_y, prows, w2_mat_y);
    w2dist := DMAT.Converted.FromElement(w2_mat,w2map);
    //each bias vector is converted to block format
    b1vec := Intb1;
    // b1vec_x := Mat.Has(b1vec).Stats.Xmax;
		b1vec_x := NumberofHiddenLayerNodes;
    b1vecmap := PBblas.Matrix_Map(b1vec_x, 1, b1vec_x, 1);
    b1vecdist := DMAT.Converted.FromElement(b1vec,b1vecmap);
    b2vec := Intb2;
    // b2vec_x := Mat.Has(b2vec).Stats.Xmax;
		b2vec_x := NumberofFeatures;
    b2vecmap := PBblas.Matrix_Map(b2vec_x, 1, prows, 1);
    b2vecdist := DMAT.Converted.FromElement(b2vec,b2vecmap);
    
    SA_param := DATASET([
    {1,1,m},
    {2,1,NumberofFeatures},
    {3,1,NumberofHiddenLayerNodes},
    {4,1,prows},
    {5,1,pcols},
    {6,1,BETA},
    {7,1,sparsityParam},
    {8,1,LAMBDA}
    ], Types.NumericField);

    //Intialize Theta
    Layout_Part addpart_(Layout_Part l, INTEGER8 c ) := TRANSFORM
        SELF.partition_id := l.partition_id + c;
        SELF := l;
    END;
    w1_partitions := w1map.partitions_used;
    w2_partitions := w2map.partitions_used;
    b1_partitions := b1vecmap.partitions_used;
    b2_partitions := b2vecmap.partitions_used;
    w1_reshape := w1dist;
    w2_reshape := PROJECT (w2dist, addpart_(LEFT, w1_partitions ),LOCAL);
    b1_reshape := PROJECT (b1vecdist, addpart_ (LEFT, w1_partitions + w2_partitions), LOCAL);
    b2_reshape := PROJECT (b2vecdist, addpart_ (LEFT, w1_partitions + w2_partitions + b1_partitions),LOCAL);
    Inttheta_ := w1_reshape + w2_reshape + b1_reshape + b2_reshape;
		Inttheta_newnode_id := PROJECT(Inttheta_, TRANSFORM (layout_part, SELF.node_id := LEFT.partition_id-1; SELF := LEFT));
		Inttheta := DISTRIBUTE (Inttheta_newnode_id, node_id);// no matter what map we use to assigned the node, the formula is the same as  ((partition_id-1) % nodes_used);
		
		
		
		Ones_VecMap := PBblas.Matrix_Map(m, 1, pcols, 1);
    //New Vector Generator
    Layout_Cell gen(UNSIGNED4 c, UNSIGNED4 NumRows) := TRANSFORM
      SELF.x := ((c-1) % NumRows) + 1;
      SELF.y := ((c-1) DIV NumRows) + 1;
      SELF.v := 1;
    END;
    //Create Ones Vector for the calculations in the step fucntion
    Ones_Vec := DATASET(m, gen(COUNTER, m));
    Ones_Vecdist := DMAT.Converted.FromCells(Ones_VecMap, Ones_Vec);
    
    emptyL := DATASET([], Layout_Part);
    
    CG := SA_lbfgs_Compatible ( Inttheta, SA_param, ddist , emptyL);
    CG2 := SA_lbfgs_Compatible ( Pbblas.MU.FROM(CG,1), SA_param, ddist , emptyL);
    
    paramnumber := 2*NumberofFeatures*NumberofHiddenLayerNodes + NumberofFeatures + NumberofHiddenLayerNodes;
    LBFGS_MAXitr := MaxIter;
    lbfgs_results := Optimization (0, 0, 0, 0).MinFUNC(Inttheta,SA_param,ddist,emptyL,SA_lbfgs_Compatible, paramnumber,LBFGS_MAXitr, 0.00001, 0.000000001,  1000, LBFGS_corrections, 0, 0, 0,0) ;
		lbfgs_results2 := Optimization (0, 0, 0, 0).MinFUNC(Inttheta,SA_param,ddist,emptyL,SA_lbfgs_Compatible2, paramnumber,LBFGS_MAXitr, 0.00001, 0.000000001,  1000, LBFGS_corrections, 0, 0, 0,0) ;
		lbfgs_results2_2 := Optimization (0, 0, 0, 0).MinFUNC(Inttheta,SA_param,ddist,Ones_Vecdist,SA_lbfgs_Compatible2_2, paramnumber,LBFGS_MAXitr, 0.00001, 0.000000001,  1000, LBFGS_corrections, 0, 0, 0,0) ;
		//Optimization_new (0, 0, 0, 0).MinFUNC(Inttheta,SA_param,ddist,emptyL,SA_lbfgs_Compatible2_param_part, paramnumber,LBFGS_MAXitr, 0.00001, 0.000000001,  1000, LBFGS_corrections, 0, 0, 0,0);
		


lbfgs_result := Optimization_new_new_2_2 (0, 0, 0, 0).MinFUNC(Inttheta,SA_param,ddist,emptyL,SA_lbfgs_Compatible2_param_part, paramnumber,LBFGS_MAXitr, 0.00001, 0.000000001,  1000, LBFGS_corrections, 0, 0, 0,0);
		//Extract the layout_part
		optTHETA_part := PROJECT (lbfgs_result (no=2), TRANSFORM(Layout_Part , SELF := LEFT), LOCAL);


    Layout_Part minuspart(Layout_Part l, UNSIGNED8 c ) := TRANSFORM
      SELF.partition_id := l.partition_id - c;
      SELF := l;
    END;
    
    optW1 := optTHETA_part (partition_id <= w1_partitions);
    
    optW2_ := optTHETA_part (partition_id > w1_partitions AND partition_id <= w1_partitions + w2_partitions);
    optW2  := PROJECT (optW2_, minuspart(LEFT, w1_partitions ));
    
    optb1_ := optTHETA_part (partition_id > w1_partitions + w2_partitions AND partition_id <= w1_partitions + w2_partitions + b1_partitions);
    optb1  := PROJECT (optb1_, minuspart (LEFT, w1_partitions + w2_partitions));
    
    
    optb2_ := optTHETA_part (partition_id > w1_partitions + w2_partitions + b1_partitions AND partition_id <= w1_partitions + w2_partitions + b1_partitions + b2_partitions);
    optb2  := PROJECT (optb2_, minuspart (LEFT, w1_partitions + w2_partitions + b1_partitions));
    
    
    
    SAprm1 := optW1;
    SAprm2 := optW2;
    SAprm3 := optb1;
    SAprm4 := optb2;
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
		lbfgs_rec := RECORD (Layout_Part)
		lbfgs_result.cost_value;
		lbfgs_result.min_funEval;
		END;
		EXPORT mod := PROJECT (lbfgs_result (no=2), TRANSFORM(lbfgs_rec , SELF := LEFT), LOCAL);
		// EXPORT mod := lbfgs_result;

	//EXPORT mod := Optimization_new_new_2_2 (0, 0, 0, 0).MinFUNC(Inttheta,SA_param,ddist,emptyL,SA_lbfgs_Compatible2_param_part, paramnumber,LBFGS_MAXitr, 0.00001, 0.000000001,  1000, LBFGS_corrections, 0, 0, 0,0);
	
	//export mod := SAprm_MUE_out;
	
	// EXPORT mod := SA_lbfgs_Compatible2_param_part ( Inttheta, SA_param, ddist , emptyL);
	//EXPORT mod := SA_lbfgs_Compatible2_param_part_minibatch ( Inttheta, SA_param, ddist(block_col=50) , emptyL);
	//EXPORT mod := ddist;
		//EXPORT mod := Inttheta;
		//EXPORT mod := ddist;
   END;//END SA_4
   
  EXPORT LearnC (DATASET(Types.NumericField) indep, DATASET(Mat.Types.Element) IntW1, DATASET(Mat.Types.Element) IntW2, DATASET(Mat.Types.Element) Intb1,  DATASET(Mat.Types.Element) Intb2, REAL8 BETA=0.1, REAL8 sparsityParam=0.1 , REAL8 LAMBDA=0.001, UNSIGNED2 MaxIter=100,UNSIGNED LBFGS_corrections = 100) := SA_4(Indep,IntW1,IntW2,Intb1,Intb2, BETA,sparsityParam,LAMBDA, MaxIter,LBFGS_corrections).mod;
	SHARED lbfgs_rec := RECORD (Layout_Part)
		REAL8 cost_value;
		INTEGER8 min_funEval;
		END;
		EXPORT extractcost_funeval(DATASET(lbfgs_rec) mod) := FUNCTION
			my_rec := RECORD 
			REAL8 cost_value;
			INTEGER8 min_funEval;
			END;
		 RETURN PROJECT (mod, TRANSFORM(my_rec , SELF := LEFT), LOCAL);
		END;
		
   EXPORT Model(DATASET(lbfgs_rec) mod) := FUNCTION
	 optTHETA_part := PROJECT (mod, TRANSFORM(Layout_Part , SELF := LEFT), LOCAL);

		max_part := MAX(optTHETA_part, optTHETA_part.partition_id);
		pp := (max_part-1) DIV 3;
		w1_partitions := pp;
		w2_partitions := pp;
		b1_partitions := 1;
		b2_partitions := pp;
    Layout_Part minuspart(Layout_Part l, UNSIGNED8 c ) := TRANSFORM
      SELF.partition_id := l.partition_id - c;
      SELF := l;
    END;
    
    optW1 := optTHETA_part (partition_id <= w1_partitions);
    
    optW2_ := optTHETA_part (partition_id > w1_partitions AND partition_id <= w1_partitions + w2_partitions);
    optW2  := PROJECT (optW2_, minuspart(LEFT, w1_partitions ));
    
    optb1_ := optTHETA_part (partition_id > w1_partitions + w2_partitions AND partition_id <= w1_partitions + w2_partitions + b1_partitions);
    optb1  := PROJECT (optb1_, minuspart (LEFT, w1_partitions + w2_partitions));
    
    
    optb2_ := optTHETA_part (partition_id > w1_partitions + w2_partitions + b1_partitions AND partition_id <= w1_partitions + w2_partitions + b1_partitions + b2_partitions);
    optb2  := PROJECT (optb2_, minuspart (LEFT, w1_partitions + w2_partitions + b1_partitions));
    
    
    
    SAprm1 := optW1;
    SAprm2 := optW2;
    SAprm3 := optb1;
    SAprm4 := optb2;
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
    modelD_Map :=	DATASET([{'id','ID'},{'x','1'},{'y','2'},{'value','3'},{'no','4'}], {STRING orig_name; STRING assigned_name;});
    FromField(SAprm_MUE_out,Mat.Types.MUElement,dOut,modelD_Map);
    RETURN dOut;
  END;//END Model



 EXPORT ExtractWeights (DATASET(lbfgs_rec) mod) := FUNCTION
    SAmod := Model (mod);
    RETURN SAmod (no<3);
  END;//END ExtractWeights
  EXPORT ExtractBias (DATASET(lbfgs_rec) mod) := FUNCTION
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
END;//END Sparse_Autoencoder_lbfgs_part




EXPORT Sparse_Autoencoder_lbfgs_part_onebias_paramdist (INTEGER4 NumberofFeatures, INTEGER4 NumberofHiddenLayerNodes, UNSIGNED4 prows=0, UNSIGNED4 pcols=0) := MODULE
   //this is a un-supervised learning algorithm, no need for the labled data
   //m : number of input samples
   //NumberofFeatures : number of input features = number of input layer nodes in sparse autoencoder = number of output layer nodes in sparse autoencoder
   //NumberofHiddenLayerNodes : number of hidden layer nodes in Sparse Autoencoder
   
   SHARED SA_4(DATASET(Types.NumericField) X, DATASET(Mat.Types.Element) IntW1, DATASET(Mat.Types.Element) IntW2, DATASET(Mat.Types.Element) Intb1,  DATASET(Mat.Types.Element) Intb2, REAL8 BETA=0.1, REAL8 sparsityParam=0.1 , REAL8 LAMBDA=0.001, UNSIGNED2 MaxIter=100, UNSIGNED LBFGS_corrections = 100) := MODULE
    dt := Types.ToMatrix (X);
    dTmp := dt;
    //d := dt; //in the entire of the calculations we work with the d matrix that each sample is presented in one column
    d := Mat.Trans(dTmp);

   //Map for Matrix d.
    // havemaxrow := maxrows > 0;
    // havemaxcol := maxcols > 0;
    //havemaxrowcol := havemaxrow and havemaxcol;
    dstats := Mat.Has(d).Stats;
    d_n := dstats.XMax;
    d_m := dstats.YMax;
		m := d_m;// number of samples
		f := NumberofFeatures; // number of features
    output_num := d_n;

    
    //Create block matrix d
    dmap_mini := PBblas.Matrix_Map(f,m,prows,pcols);
		dmap := PBblas.Matrix_Map(f,m,prows,pcols);// we use this map only to partiion the data, however the distribution is done in a fashion where each big column partition of the data which includes smaller row partitions ends up in one node
    ddist_tmp := DMAT.Converted.FromElement(d,dmap);
		//number of mini batches in each bigger batch : bigger batches have pcols columns, mini batches have minibatch column. all the mini batches inside a bigger match are distributed in the same node becasue we want to make sure all the big batches end up in the same node
		//redistribute ddist in a way that the main partitioning is based on matrix columns, all partitions in the same column will end up in the same node
		d_row_b := dmap.row_blocks;
		layout_part new_node_id (layout_part dd) := TRANSFORM
			// part_id := dd.partition_id;
			// node_part_id := ((part_id-1) DIV d_row_b )+1;// this i sthe partition id if we partition only the matrix columns
			// new_node_id := dmap.assigned_node(node_part_id);
			new_node_id := dmap.assigned_node(dd.block_col);
			SELF.node_id := new_node_id;
			//SELF.partition_id := new_part_id;
			SELF := dd;
		END;
		ddist_ := PROJECT (ddist_tmp, new_node_id (LEFT), LOCAL );
		ddist := DISTRIBUTE(ddist_, node_id);
    //Creat block matrices for weights
    w1_mat := IntW1;
    //w1_mat_x := Mat.Has(w1_mat).Stats.Xmax;
		w1_mat_x := NumberofHiddenLayerNodes;
    // w1_mat_y := Mat.Has(w1_mat).Stats.Ymax;
		w1_mat_y := NumberofFeatures;
    w1map := PBblas.Matrix_Map(w1_mat_x, w1_mat_y, w1_mat_x, prows);
    w1dist := DMAT.Converted.FromElement(w1_mat,w1map);
    w2_mat := IntW2;
    w2_mat_x := w1_mat_y;
    w2_mat_y := w1_mat_x;
    w2map := PBblas.Matrix_Map(w2_mat_x, w2_mat_y, prows, w2_mat_y);
    w2dist := DMAT.Converted.FromElement(w2_mat,w2map);
    //each bias vector is converted to block format
    b1vec := Intb1;
    // b1vec_x := Mat.Has(b1vec).Stats.Xmax;
		b1vec_x := NumberofHiddenLayerNodes;
    b1vecmap := PBblas.Matrix_Map(b1vec_x, 1, b1vec_x, 1);
    b1vecdist := DMAT.Converted.FromElement(b1vec,b1vecmap);
    b2vec := Intb2;
    // b2vec_x := Mat.Has(b2vec).Stats.Xmax;
		b2vec_x := NumberofFeatures;
    b2vecmap := PBblas.Matrix_Map(b2vec_x, 1, b2vec_x, 1);
    b2vecdist := DMAT.Converted.FromElement(b2vec,b2vecmap);
    
    SA_param := DATASET([
    {1,1,m},
    {2,1,NumberofFeatures},
    {3,1,NumberofHiddenLayerNodes},
    {4,1,prows},
    {5,1,pcols},
    {6,1,BETA},
    {7,1,sparsityParam},
    {8,1,LAMBDA}
    ], Types.NumericField);

    //Intialize Theta
    Layout_Part addpart_(Layout_Part l, INTEGER8 c ) := TRANSFORM
        SELF.partition_id := l.partition_id + c;
        SELF := l;
    END;
    w1_partitions := w1map.partitions_used;
    w2_partitions := w2map.partitions_used;
    b1_partitions := b1vecmap.partitions_used;
    b2_partitions := b2vecmap.partitions_used;
    w1_reshape := w1dist;
    w2_reshape := PROJECT (w2dist, addpart_(LEFT, w1_partitions ),LOCAL);
    b1_reshape := PROJECT (b1vecdist, addpart_ (LEFT, w1_partitions + w2_partitions), LOCAL);
    b2_reshape := PROJECT (b2vecdist, addpart_ (LEFT, w1_partitions + w2_partitions + b1_partitions),LOCAL);
    Inttheta_ := w1_reshape + w2_reshape + b1_reshape + b2_reshape;
		Inttheta_newnode_id := PROJECT(Inttheta_, TRANSFORM (layout_part, SELF.node_id := LEFT.partition_id-1; SELF := LEFT));
		Inttheta := DISTRIBUTE (Inttheta_newnode_id, node_id);// no matter what map we use to assigned the node, the formula is the same as  ((partition_id-1) % nodes_used);

		Ones_VecMap := PBblas.Matrix_Map(m, 1, pcols, 1);
    //New Vector Generator
    Layout_Cell gen(UNSIGNED4 c, UNSIGNED4 NumRows) := TRANSFORM
      SELF.x := ((c-1) % NumRows) + 1;
      SELF.y := ((c-1) DIV NumRows) + 1;
      SELF.v := 1;
    END;
    //Create Ones Vector for the calculations in the step fucntion
    Ones_Vec := DATASET(m, gen(COUNTER, m));
    Ones_Vecdist := DMAT.Converted.FromCells(Ones_VecMap, Ones_Vec);
    
    emptyL := DATASET([], Layout_Part);
    
    CG := SA_lbfgs_Compatible ( Inttheta, SA_param, ddist , emptyL);
    CG2 := SA_lbfgs_Compatible ( Pbblas.MU.FROM(CG,1), SA_param, ddist , emptyL);
    
    paramnumber := 2*NumberofFeatures*NumberofHiddenLayerNodes + NumberofFeatures + NumberofHiddenLayerNodes;
    LBFGS_MAXitr := MaxIter;
    lbfgs_results := Optimization (0, 0, 0, 0).MinFUNC(Inttheta,SA_param,ddist,emptyL,SA_lbfgs_Compatible, paramnumber,LBFGS_MAXitr, 0.00001, 0.000000001,  1000, LBFGS_corrections, 0, 0, 0,0) ;
		lbfgs_results2 := Optimization (0, 0, 0, 0).MinFUNC(Inttheta,SA_param,ddist,emptyL,SA_lbfgs_Compatible2, paramnumber,LBFGS_MAXitr, 0.00001, 0.000000001,  1000, LBFGS_corrections, 0, 0, 0,0) ;
		lbfgs_results2_2 := Optimization (0, 0, 0, 0).MinFUNC(Inttheta,SA_param,ddist,Ones_Vecdist,SA_lbfgs_Compatible2_2, paramnumber,LBFGS_MAXitr, 0.00001, 0.000000001,  1000, LBFGS_corrections, 0, 0, 0,0) ;
		//Optimization_new (0, 0, 0, 0).MinFUNC(Inttheta,SA_param,ddist,emptyL,SA_lbfgs_Compatible2_param_part, paramnumber,LBFGS_MAXitr, 0.00001, 0.000000001,  1000, LBFGS_corrections, 0, 0, 0,0);
		


lbfgs_result := Optimization_new_new_2_2 (0, 0, 0, 0).MinFUNC(Inttheta,SA_param,ddist,emptyL,SA_lbfgs_Compatible2_param_part_onebias_distparam, paramnumber,LBFGS_MAXitr, 0.00001, 0.000000001,  1000, LBFGS_corrections, 0, 0, 0,0);
		
		
		lbfgs_rec := RECORD (Layout_Part)
		lbfgs_result.cost_value;
		lbfgs_result.min_funEval;
		END;
		EXPORT mod := PROJECT (lbfgs_result (no=2), TRANSFORM(lbfgs_rec , SELF := LEFT), LOCAL);
		//EXPORT mod := lbfgs_result;

	//EXPORT mod := Optimization_new_new_2_2 (0, 0, 0, 0).MinFUNC(Inttheta,SA_param,ddist,emptyL,SA_lbfgs_Compatible2_param_part, paramnumber,LBFGS_MAXitr, 0.00001, 0.000000001,  1000, LBFGS_corrections, 0, 0, 0,0);
	
	//export mod := SAprm_MUE_out;
	
	// EXPORT mod := SA_lbfgs_Compatible2_param_part_onebias_distparam ( Inttheta, SA_param, ddist , emptyL);
	//EXPORT mod := SA_lbfgs_Compatible2_param_part_minibatch ( Inttheta, SA_param, ddist(block_col=50) , emptyL);
	//EXPORT mod := ddist;
		// EXPORT mod := Inttheta;
		//EXPORT mod := ddist;
   END;//END SA_4
   
  EXPORT LearnC (DATASET(Types.NumericField) indep, DATASET(Mat.Types.Element) IntW1, DATASET(Mat.Types.Element) IntW2, DATASET(Mat.Types.Element) Intb1,  DATASET(Mat.Types.Element) Intb2, REAL8 BETA=0.1, REAL8 sparsityParam=0.1 , REAL8 LAMBDA=0.001, UNSIGNED2 MaxIter=100,UNSIGNED LBFGS_corrections = 100) := SA_4(Indep,IntW1,IntW2,Intb1,Intb2, BETA,sparsityParam,LAMBDA, MaxIter,LBFGS_corrections).mod;
	SHARED lbfgs_rec := RECORD (Layout_Part)
		REAL8 cost_value;
		INTEGER8 min_funEval;
		END;
		EXPORT extractcost_funeval(DATASET(lbfgs_rec) mod) := FUNCTION
			my_rec := RECORD 
			REAL8 cost_value;
			INTEGER8 min_funEval;
			END;
		 RETURN PROJECT (mod, TRANSFORM(my_rec , SELF := LEFT), LOCAL);
		END;
		
   EXPORT Model(DATASET(lbfgs_rec) mod) := FUNCTION
	 optTHETA_part := PROJECT (mod, TRANSFORM(Layout_Part , SELF := LEFT), LOCAL);

		max_part := MAX(optTHETA_part, optTHETA_part.partition_id);
		pp := (max_part-2) DIV 2;
		w1_partitions := pp;
		w2_partitions := pp;
		b1_partitions := 1;
		b2_partitions := 1;
    Layout_Part minuspart(Layout_Part l, UNSIGNED8 c ) := TRANSFORM
      SELF.partition_id := l.partition_id - c;
      SELF := l;
    END;
    
    optW1 := optTHETA_part (partition_id <= w1_partitions);
    
    optW2_ := optTHETA_part (partition_id > w1_partitions AND partition_id <= w1_partitions + w2_partitions);
    optW2  := PROJECT (optW2_, minuspart(LEFT, w1_partitions ));
    
    optb1_ := optTHETA_part (partition_id > w1_partitions + w2_partitions AND partition_id <= w1_partitions + w2_partitions + b1_partitions);
    optb1  := PROJECT (optb1_, minuspart (LEFT, w1_partitions + w2_partitions));
    
    
    optb2_ := optTHETA_part (partition_id > w1_partitions + w2_partitions + b1_partitions AND partition_id <= w1_partitions + w2_partitions + b1_partitions + b2_partitions);
    optb2  := PROJECT (optb2_, minuspart (LEFT, w1_partitions + w2_partitions + b1_partitions));
    
    
    
    SAprm1 := optW1;
    SAprm2 := optW2;
    SAprm3 := optb1;
    SAprm4 := optb2;
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
    modelD_Map :=	DATASET([{'id','ID'},{'x','1'},{'y','2'},{'value','3'},{'no','4'}], {STRING orig_name; STRING assigned_name;});
    FromField(SAprm_MUE_out,Mat.Types.MUElement,dOut,modelD_Map);
    RETURN dOut;
  END;//END Model



 EXPORT ExtractWeights (DATASET(lbfgs_rec) mod) := FUNCTION
    SAmod := Model (mod);
    RETURN SAmod (no<3);
  END;//END ExtractWeights
  EXPORT ExtractBias (DATASET(lbfgs_rec) mod) := FUNCTION
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
END;//END Sparse_Autoencoder_lbfgs_part_onebias_paramdist


//this is an implementation of Sparse_Autoencoder_lbfgs_part where b1 and b2 are distributed on only one node, in Sparse_Autoencoder_lbfgs_part b1 is distributed on one node and  b2 is distributed in f/prows node
EXPORT Sparse_Autoencoder_lbfgs_part_eachbiasinonenode (INTEGER4 NumberofFeatures, INTEGER4 NumberofHiddenLayerNodes, UNSIGNED4 prows=0, UNSIGNED4 pcols=0, UNSIGNED4 mini_num=1) := MODULE
   //this is a un-supervised learning algorithm, no need for the labled data
   //m : number of input samples
   //NumberofFeatures : number of input features = number of input layer nodes in sparse autoencoder = number of output layer nodes in sparse autoencoder
   //NumberofHiddenLayerNodes : number of hidden layer nodes in Sparse Autoencoder
   
   SHARED SA_4(DATASET(Types.NumericField) X, DATASET(Mat.Types.Element) IntW1, DATASET(Mat.Types.Element) IntW2, DATASET(Mat.Types.Element) Intb1,  DATASET(Mat.Types.Element) Intb2, REAL8 BETA=0.1, REAL8 sparsityParam=0.1 , REAL8 LAMBDA=0.001, UNSIGNED2 MaxIter=100, UNSIGNED LBFGS_corrections = 100) := MODULE
    dt := Types.ToMatrix (X);
    dTmp := dt;
    //d := dt; //in the entire of the calculations we work with the d matrix that each sample is presented in one column
    d := Mat.Trans(dTmp);

   //Map for Matrix d.
    // havemaxrow := maxrows > 0;
    // havemaxcol := maxcols > 0;
    //havemaxrowcol := havemaxrow and havemaxcol;
    dstats := Mat.Has(d).Stats;
    d_n := dstats.XMax;
    d_m := dstats.YMax;
		m := d_m;// number of samples
		f := NumberofFeatures; // number of features
    output_num := d_n;

    
    //Create block matrix d
    dmap_mini := PBblas.Matrix_Map(f,m,prows,pcols);
		dmap := PBblas.Matrix_Map(f,m,prows,pcols);// we use this map only to partiion the data, however the distribution is done in a fashion where each big column partition of the data which includes smaller row partitions ends up in one node
    ddist_tmp := DMAT.Converted.FromElement(d,dmap);
		//number of mini batches in each bigger batch : bigger batches have pcols columns, mini batches have minibatch column. all the mini batches inside a bigger match are distributed in the same node becasue we want to make sure all the big batches end up in the same node
		//redistribute ddist in a way that the main partitioning is based on matrix columns, all partitions in the same column will end up in the same node
		d_row_b := dmap.row_blocks;
		layout_part new_node_id (layout_part dd) := TRANSFORM
			// part_id := dd.partition_id;
			// node_part_id := ((part_id-1) DIV d_row_b )+1;// this i sthe partition id if we partition only the matrix columns
			// new_node_id := dmap.assigned_node(node_part_id);
			new_node_id := dmap.assigned_node(((dd.block_col-1) DIV mini_num)+1);
			SELF.node_id := new_node_id;
			//SELF.partition_id := new_part_id;
			SELF := dd;
		END;
		ddist_ := PROJECT (ddist_tmp, new_node_id (LEFT), LOCAL );
		ddist := DISTRIBUTE(ddist_, node_id);
    //Creat block matrices for weights
    w1_mat := IntW1;
    //w1_mat_x := Mat.Has(w1_mat).Stats.Xmax;
		w1_mat_x := NumberofHiddenLayerNodes;
    // w1_mat_y := Mat.Has(w1_mat).Stats.Ymax;
		w1_mat_y := NumberofFeatures;
    w1map := PBblas.Matrix_Map(w1_mat_x, w1_mat_y, w1_mat_x, prows);
    w1dist := DMAT.Converted.FromElement(w1_mat,w1map);
    w2_mat := IntW2;
    w2_mat_x := w1_mat_y;
    w2_mat_y := w1_mat_x;
    w2map := PBblas.Matrix_Map(w2_mat_x, w2_mat_y, prows, w2_mat_y);
    w2dist := DMAT.Converted.FromElement(w2_mat,w2map);
    //each bias vector is converted to block format
    b1vec := Intb1;
    // b1vec_x := Mat.Has(b1vec).Stats.Xmax;
		b1vec_x := NumberofHiddenLayerNodes;
    b1vecmap := PBblas.Matrix_Map(b1vec_x, 1, b1vec_x, 1);
    b1vecdist := DMAT.Converted.FromElement(b1vec,b1vecmap);
    b2vec := Intb2;
    // b2vec_x := Mat.Has(b2vec).Stats.Xmax;
		b2vec_x := NumberofFeatures;
    b2vecmap := PBblas.Matrix_Map(b2vec_x, 1, prows, 1);
    b2vecdist := DMAT.Converted.FromElement(b2vec,b2vecmap);
    SA_param := DATASET([
    {1,1,m},
    {2,1,NumberofFeatures},
    {3,1,NumberofHiddenLayerNodes},
    {4,1,prows},
    {5,1,pcols},
    {6,1,BETA},
    {7,1,sparsityParam},
    {8,1,LAMBDA}
    ], Types.NumericField);

    //Intialize Theta
    Layout_Part addpart_(Layout_Part l, INTEGER8 c ) := TRANSFORM
        SELF.partition_id := l.partition_id + c;
        SELF := l;
    END;
    w1_partitions := w1map.partitions_used;
    w2_partitions := w2map.partitions_used;
    b1_partitions := b1vecmap.partitions_used;
    b2_partitions := b2vecmap.partitions_used;
    w1_reshape := w1dist;
    w2_reshape := PROJECT (w2dist, addpart_(LEFT, w1_partitions ),LOCAL);
    b1_reshape := PROJECT (b1vecdist, addpart_ (LEFT, w1_partitions + w2_partitions), LOCAL);
    b2_reshape := PROJECT (b2vecdist, addpart_ (LEFT, w1_partitions + w2_partitions + b1_partitions),LOCAL);
    Inttheta_ := w1_reshape + w2_reshape + b1_reshape + b2_reshape;
		Inttheta_newnode_id := PROJECT(Inttheta_, TRANSFORM (layout_part, SELF.node_id := IF (LEFT.partition_id <= w1_partitions+w2_partitions, LEFT.partition_id-1, w1_partitions+w2_partitions); SELF := LEFT));
		Inttheta := DISTRIBUTE (Inttheta_newnode_id, node_id);// no matter what map we use to assigned the node, the formula is the same as  ((partition_id-1) % nodes_used);
		
		
		
		Ones_VecMap := PBblas.Matrix_Map(m, 1, pcols, 1);
    //New Vector Generator
    Layout_Cell gen(UNSIGNED4 c, UNSIGNED4 NumRows) := TRANSFORM
      SELF.x := ((c-1) % NumRows) + 1;
      SELF.y := ((c-1) DIV NumRows) + 1;
      SELF.v := 1;
    END;
    //Create Ones Vector for the calculations in the step fucntion
    Ones_Vec := DATASET(m, gen(COUNTER, m));
    Ones_Vecdist := DMAT.Converted.FromCells(Ones_VecMap, Ones_Vec);
    
    emptyL := DATASET([], Layout_Part);
    
    CG := SA_lbfgs_Compatible ( Inttheta, SA_param, ddist , emptyL);
    CG2 := SA_lbfgs_Compatible ( Pbblas.MU.FROM(CG,1), SA_param, ddist , emptyL);
    
    paramnumber := 2*NumberofFeatures*NumberofHiddenLayerNodes + NumberofFeatures + NumberofHiddenLayerNodes;
    LBFGS_MAXitr := MaxIter;
    lbfgs_results := Optimization (0, 0, 0, 0).MinFUNC(Inttheta,SA_param,ddist,emptyL,SA_lbfgs_Compatible, paramnumber,LBFGS_MAXitr, 0.00001, 0.000000001,  1000, LBFGS_corrections, 0, 0, 0,0) ;
		lbfgs_results2 := Optimization (0, 0, 0, 0).MinFUNC(Inttheta,SA_param,ddist,emptyL,SA_lbfgs_Compatible2, paramnumber,LBFGS_MAXitr, 0.00001, 0.000000001,  1000, LBFGS_corrections, 0, 0, 0,0) ;
		lbfgs_results2_2 := Optimization (0, 0, 0, 0).MinFUNC(Inttheta,SA_param,ddist,Ones_Vecdist,SA_lbfgs_Compatible2_2, paramnumber,LBFGS_MAXitr, 0.00001, 0.000000001,  1000, LBFGS_corrections, 0, 0, 0,0) ;
		//Optimization_new (0, 0, 0, 0).MinFUNC(Inttheta,SA_param,ddist,emptyL,SA_lbfgs_Compatible2_param_part, paramnumber,LBFGS_MAXitr, 0.00001, 0.000000001,  1000, LBFGS_corrections, 0, 0, 0,0);
		


lbfgs_result := Optimization_new_new_2_2 (0, 0, 0, 0).MinFUNC(Inttheta,SA_param,ddist,emptyL,SA_lbfgs_Compatible2_param_part_biasonenode, paramnumber,LBFGS_MAXitr, 0.00001, 0.000000001,  1000, LBFGS_corrections, 0, 0, 0,0);
		//Extract the layout_part
		// optTHETA_part := PROJECT (lbfgs_result (no=2), TRANSFORM(Layout_Part , SELF := LEFT), LOCAL);


    // Layout_Part minuspart(Layout_Part l, UNSIGNED8 c ) := TRANSFORM
      // SELF.partition_id := l.partition_id - c;
      // SELF := l;
    // END;
    
    // optW1 := optTHETA_part (partition_id <= w1_partitions);
    
    // optW2_ := optTHETA_part (partition_id > w1_partitions AND partition_id <= w1_partitions + w2_partitions);
    // optW2  := PROJECT (optW2_, minuspart(LEFT, w1_partitions ));
    
    // optb1_ := optTHETA_part (partition_id > w1_partitions + w2_partitions AND partition_id <= w1_partitions + w2_partitions + b1_partitions);
    // optb1  := PROJECT (optb1_, minuspart (LEFT, w1_partitions + w2_partitions));
    
    
    // optb2_ := optTHETA_part (partition_id > w1_partitions + w2_partitions + b1_partitions AND partition_id <= w1_partitions + w2_partitions + b1_partitions + b2_partitions);
    // optb2  := PROJECT (optb2_, minuspart (LEFT, w1_partitions + w2_partitions + b1_partitions));
    
    
    
    // SAprm1 := optW1;
    // SAprm2 := optW2;
    // SAprm3 := optb1;
    // SAprm4 := optb2;
    // SAprm1_mat := DMat.Converted.FromPart2Elm (SAprm1);
    // SAprm2_mat := DMat.Converted.FromPart2Elm (SAprm2);
    // SAprm3_mat := DMat.Converted.FromPart2Elm (SAprm3);
    // SAprm4_mat := DMat.Converted.FromPart2Elm (SAprm4);
    // SAprm1_mat_no := Mat.MU.TO(SAprm1_mat,1);
    // SAprm2_mat_no := Mat.MU.TO(SAprm2_mat,2);
    // SAprm3_mat_no := Mat.MU.TO(SAprm3_mat,3);
    // SAprm4_mat_no := Mat.MU.TO(SAprm4_mat,4);
    // SAprm_MUE := SAprm1_mat_no + SAprm2_mat_no + SAprm3_mat_no + SAprm4_mat_no;
    // AppendID(SAprm_MUE, id, SAprm_MUE_id);
    // ToField (SAprm_MUE_id, SAprm_MUE_out, id, 'x,y,value,no');
		// lbfgs_rec := RECORD (Layout_Part)
		// lbfgs_result.cost_value;
		// lbfgs_result.min_funEval;
		// END;
		// EXPORT mod := PROJECT (lbfgs_result (no=2), TRANSFORM(lbfgs_rec , SELF := LEFT), LOCAL); orig
		EXPORT mod := lbfgs_result;
		//EXPORT mod := Inttheta;

	//EXPORT mod := Optimization_new_new_2_2 (0, 0, 0, 0).MinFUNC(Inttheta,SA_param,ddist,emptyL,SA_lbfgs_Compatible2_param_part, paramnumber,LBFGS_MAXitr, 0.00001, 0.000000001,  1000, LBFGS_corrections, 0, 0, 0,0);
	
	//export mod := SAprm_MUE_out;
	
	//EXPORT mod := SA_lbfgs_Compatible2_param_part ( Inttheta, SA_param, ddist , emptyL);
	// EXPORT mod := SA_lbfgs_Compatible2_param_part_biasonenode ( Inttheta, SA_param, ddist , emptyL);
	//EXPORT mod := SA_lbfgs_Compatible2_param_part_minibatch ( Inttheta, SA_param, ddist(block_col=50) , emptyL);
	//EXPORT mod := ddist;
		//EXPORT mod := Inttheta;
		//EXPORT mod := ddist;
   END;//END SA_4
   
  EXPORT LearnC (DATASET(Types.NumericField) indep, DATASET(Mat.Types.Element) IntW1, DATASET(Mat.Types.Element) IntW2, DATASET(Mat.Types.Element) Intb1,  DATASET(Mat.Types.Element) Intb2, REAL8 BETA=0.1, REAL8 sparsityParam=0.1 , REAL8 LAMBDA=0.001, UNSIGNED2 MaxIter=100,UNSIGNED LBFGS_corrections = 100) := SA_4(Indep,IntW1,IntW2,Intb1,Intb2, BETA,sparsityParam,LAMBDA, MaxIter,LBFGS_corrections).mod;
	SHARED lbfgs_rec := RECORD (Layout_Part)
		REAL8 cost_value;
		INTEGER8 min_funEval;
		END;
		EXPORT extractcost_funeval(DATASET(lbfgs_rec) mod) := FUNCTION
			my_rec := RECORD 
			REAL8 cost_value;
			INTEGER8 min_funEval;
			END;
		 RETURN PROJECT (mod, TRANSFORM(my_rec , SELF := LEFT), LOCAL);
		END;
		
   EXPORT Model(DATASET(lbfgs_rec) mod) := FUNCTION
	 optTHETA_part := PROJECT (mod, TRANSFORM(Layout_Part , SELF := LEFT), LOCAL);

		max_part := MAX(optTHETA_part, optTHETA_part.partition_id);
		pp := (max_part-1) DIV 3;
		w1_partitions := pp;
		w2_partitions := pp;
		b1_partitions := 1;
		b2_partitions := pp;
    Layout_Part minuspart(Layout_Part l, UNSIGNED8 c ) := TRANSFORM
      SELF.partition_id := l.partition_id - c;
      SELF := l;
    END;
    
    optW1 := optTHETA_part (partition_id <= w1_partitions);
    
    optW2_ := optTHETA_part (partition_id > w1_partitions AND partition_id <= w1_partitions + w2_partitions);
    optW2  := PROJECT (optW2_, minuspart(LEFT, w1_partitions ));
    
    optb1_ := optTHETA_part (partition_id > w1_partitions + w2_partitions AND partition_id <= w1_partitions + w2_partitions + b1_partitions);
    optb1  := PROJECT (optb1_, minuspart (LEFT, w1_partitions + w2_partitions));
    
    
    optb2_ := optTHETA_part (partition_id > w1_partitions + w2_partitions + b1_partitions AND partition_id <= w1_partitions + w2_partitions + b1_partitions + b2_partitions);
    optb2  := PROJECT (optb2_, minuspart (LEFT, w1_partitions + w2_partitions + b1_partitions));
    
    
    
    SAprm1 := optW1;
    SAprm2 := optW2;
    SAprm3 := optb1;
    SAprm4 := optb2;
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
    modelD_Map :=	DATASET([{'id','ID'},{'x','1'},{'y','2'},{'value','3'},{'no','4'}], {STRING orig_name; STRING assigned_name;});
    FromField(SAprm_MUE_out,Mat.Types.MUElement,dOut,modelD_Map);
    RETURN dOut;
  END;//END Model



 EXPORT ExtractWeights (DATASET(lbfgs_rec) mod) := FUNCTION
    SAmod := Model (mod);
    RETURN SAmod (no<3);
  END;//END ExtractWeights
  EXPORT ExtractBias (DATASET(lbfgs_rec) mod) := FUNCTION
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
END;//END Sparse_Autoencoder_lbfgs_part_eachbiasinonenode


//theta is parameter matrix of size k by f, where k is number of classes and f is number of features. f is partitioned using part_rows
// Train data is of size f by m, where f paritioned by part_rows and m partitioned by part_cols. The paritions are distributed based on columns though
// row partitions in the same column end up on one node
EXPORT  SoftMax_compatible_lbfgs( DATASET(Layout_Part) theta, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Part) TrainData , DATASET(Layout_Part) TrainLabel) := function
  Numfeat:= CostFunc_params(id=2)[1].value;
  NumClass := CostFunc_params(id=3)[1].value;// number of features
  m := CostFunc_params(id=1)[1].value;// number of samples
	m_ := -1/m;
  part_rows := CostFunc_params(id=4)[1].value;
  part_cols := CostFunc_params(id=5)[1].value;
  LAMBDA:= CostFunc_params(id=6)[1].value;
	//maps used
	SET OF PBblas.Types.value_t empty_array := [];
	map_c := PBblas.Matrix_Map(NumClass, m, NumClass, part_cols);
	dmap := PBblas.Matrix_Map(Numfeat, m, part_rows, part_cols);
	theta_map := PBblas.Matrix_Map(NumClass, Numfeat, NumClass, part_rows);
	//distribute theta to all the nodes that train data is on
	data_nodesused := dmap.col_blocks;
	Layout_Part_newnode := RECORD (Layout_Part)
		PBblas.Types.node_t new_node_id;
	END;
	Layout_Part_newnode norm_theta (Layout_Part te, INTEGER co) := TRANSFORM
		SELF.new_node_id := co % data_nodesused  ;
		SELF:= te;
	END;
	theta_norm := NORMALIZE(theta, data_nodesused, norm_theta(LEFT, COUNTER) );
	theta_dist := DISTRIBUTE (theta_norm, new_node_id);
	//calculated theta*TrainData
	//M=(theta*x);
	Layout_Part multx(Layout_Part_newnode t_part, Layout_Part x_part):=TRANSFORM
		part_id := x_part.block_col;
		part_c_rows := map_c.part_rows(part_id);
		part_c_cols := map_c.part_cols(part_id);
		k := t_part.part_cols;
		SELF.partition_id := part_id;
		SELF.node_id      := x_part.node_id;
		SELF.block_row    := t_part.block_row;
		SELF.block_col    := x_part.block_col;
		SELF.first_row    := map_c.first_row(part_id);
		SELF.part_rows    := part_c_rows;
		SELF.first_col    := map_c.first_col(part_id);
		SELF.part_cols    := part_c_cols;
		SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
																		part_c_rows, part_c_cols, k,
																		1.0, t_part.mat_part, x_part.mat_part,
																		0.0, empty_array);
  END;
	tx_ := JOIN (theta_dist, TrainData, LEFT.block_col = RIGHT.block_row, multx(LEFT, RIGHT), LOCAL );
	// Sum terms
  Layout_Part sumTerms(Layout_Part cumm, Layout_Part term) := TRANSFORM
    N := cumm.part_rows * cumm.part_cols;
    SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term.mat_part, 1);
    SELF := cumm;
  END;
	tx_sorted := SORT(tx_, partition_id, LOCAL);
  tx := ROLLUP(tx_sorted, sumTerms(LEFT, RIGHT), partition_id, LOCAL);
	// define a c++ function who does the following operations on each partition
	// M=tx;
	// M = bsxfun(@minus, M, max(M, [], 1));
	// M=exp(M);
	// M = bsxfun(@rdivide, M, sum(M));
	//r : rows
	//s : cols
	// D : input matrix
	SET OF REAL8 soft_fun (PBblas.Types.dimension_t r, PBblas.Types.dimension_t s, PBblas.Types.matrix_t D) := BEGINC++

    #body
    __lenResult = r * s* sizeof(double);
    __isAllResult = false;
    double * result = new double[r*s];
    __result = (void*) result;
    double *cell = (double*) d;
		double *max_arr = new double[s];
		double *sum_arr = new double[s];
		double max_tmp;
		double sum_tmp;
    uint32_t cells =  r * s;
    uint32_t i;
		uint32_t j;
    uint32_t pos;
		uint32_t posj;
		for (i=0; i<s; i++) {
			pos = i * r;
			max_tmp = cell[pos];
			for (j=1; j<r; j++)
			{
				posj = pos +j;
				if(cell[posj]>max_tmp)
					max_tmp = cell[posj];
			}
			max_arr[i]=max_tmp;
    }
		for (i=0; i<cells; i++) {
			pos = (int) i / r;
			result[i]=exp(cell[i]-max_arr[pos]);
		}
    for (i=0; i<s; i++) {
			sum_tmp = 0;
			pos = i * r;
			for (j=0; j<r; j++)
			{
					sum_tmp = sum_tmp + result[pos+j];
			}
			sum_arr[i]=sum_tmp;
    }
		for (i=0; i<cells; i++) {
			pos = (int) i / r;
			result[i]=result[i] / sum_arr[pos];
		}

  ENDC++;
	Layout_Part soft_tran(Layout_Part le) := TRANSFORM
    SELF.mat_part := soft_fun(le.part_rows, le.part_cols, le.mat_part);
    SELF := le;
  END;
	tx_soft := PROJECT (tx, soft_tran (LEFT), LOCAL);//same as M in MATLAB
	//groundTruth - tx_soft : (groundTruth-M)
	groundTruth := TrainLabel;
	Layout_Part grnd_tran(Layout_Part le, Layout_Part ri) := TRANSFORM
		N := le.part_rows * le.part_cols;
    SELF.mat_part := PBblas.BLAS.daxpy(N, -1.0, ri.mat_part, 1, le.mat_part, 1);
    SELF := le;
  END;
	grnd_tx := JOIN (groundTruth, tx_soft, LEFT.partition_id = RIGHT.partition_id, grnd_tran(LEFT, RIGHT),LOCAL);
	//grnd_tx * x' : (groundTruth-M)*x')
	Layout_Part grndtx_xt_mul(Layout_Part g_part, Layout_Part x_part):=TRANSFORM
		part_id     := x_part.block_row;
		part_g_cols := g_part.part_cols;
		part_g_rows := g_part.part_rows;
		part_x_rows := x_part.part_rows;
		part_x_cols := x_part.part_cols;
		k := part_g_cols;
		SELF.partition_id := part_id;
		SELF.node_id      := theta_map.assigned_node(part_id);
		SELF.block_row    := g_part.block_row;
		SELF.block_col    := x_part.block_row;
		SELF.first_row    := theta_map.first_row(part_id);
		SELF.part_rows    := theta_map.part_rows(part_id);
		SELF.first_col    := theta_map.first_col(part_id);
		SELF.part_cols    := theta_map.part_cols(part_id);
		SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, TRUE,
																		part_g_rows, part_x_rows, k,
																		m_, g_part.mat_part, x_part.mat_part,
																		0.0, empty_array);
	END;
	//-1/m*(groundTruth-M)*x')
	grndtx_xt := JOIN (grnd_tx, TrainData, LEFT.block_col = RIGHT.block_col, grndtx_xt_mul(LEFT, RIGHT), LOCAL );
	Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
		N := le.part_rows * le.part_cols ;
		SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1);
		SELF := le;
	END;
	grndtx_xt_dist := DISTRIBUTE (grndtx_xt, node_id);
	grndtx_xt_dist_sorted := SORT (grndtx_xt_dist, partition_id, LOCAL);
	grndtx_xt_m := ROLLUP(grndtx_xt_dist_sorted, LEFT.partition_id = RIGHT.partition_id, addup(LEFT, RIGHT), LOCAL);
	//calculate theta_grad
	Layout_Part theta_grad_tran(Layout_Part le, Layout_Part ri) := TRANSFORM
		N := le.part_rows * le.part_cols ;
		SELF.mat_part := PBblas.BLAS.daxpy(N, LAMBDA, le.mat_part, 1, ri.mat_part, 1);
		SELF := le;
	END;
	theta_grad := JOIN (theta, grndtx_xt_m,LEFT.partition_id = RIGHT.partition_id, theta_grad_tran(LEFT,RIGHT), LOCAL);
	//calculate cost
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
	simple := {Pbblas.types.value_t v};
	simple wd_tran (Layout_Part le) := TRANSFORM
		SELF.v := sum_sq(le.part_cols * le.part_rows, le.mat_part);
	END;
	weightdecay_term_ := PROJECT (theta, wd_tran (LEFT), LOCAL);
	weightdecay_term := SUM (weightdecay_term_, weightdecay_term_.v);
	REAL8 log_cost_c (PBblas.Types.dimension_t N, PBblas.Types.matrix_t M, PBblas.Types.matrix_t D) := BEGINC++

    #body
    double result = 0;
		double tmpp ;
    double *cellm = (double*) m;
		double *celld = (double*) d;
    uint32_t i;
		for (i=0; i<n; i++) {
      result = result + (log(cellm[i])*celld[i]);
    }
		return(result);
  ENDC++;
	simple cost_log_tran (Layout_Part le, Layout_Part ri) := TRANSFORM
		SELF.v := log_cost_c(le.part_cols * le.part_rows, le.mat_part, ri.mat_part);
	END;
	log_cost_ := JOIN (tx_soft, groundTruth, LEFT.partition_id = RIGHT.partition_id, cost_log_tran(LEFT,RIGHT), LOCAL);
	log_cost := SUM (log_cost_, log_cost_.v);
	//cost=((-1/m)*log_cost)+((lambda/2)*weightdecay_term);
	cost := m_*log_cost + ((LAMBDA/2)*weightdecay_term);
	//return results
	return_record := RECORD (Layout_Part)
		REAL8 cost_value;
	END;
	theta_cost := PROJECT (theta_grad, TRANSFORM (return_record, SELF.cost_value := cost; SELF:=LEFT), LOCAL);
	theta_cost_check :=  ASSERT(theta_cost, node_id=Thorlib.node() and node_id=(partition_id-1), 'softmax gradient is not distributed correctly', FAIL);
	ToReturn := theta_cost_check;

  RETURN ToReturn;

  END; //END SoftMax_compatible_lbfgs
	EXPORT  SoftMax_compatible_lbfgs_sparse_test( DATASET(Layout_Part) theta, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Part) TrainData , DATASET(Layout_Part) TrainLabel) := function
  Numfeat:= CostFunc_params(id=2)[1].value;
  NumClass := CostFunc_params(id=3)[1].value;// number of features
  m := CostFunc_params(id=1)[1].value;// number of samples
	m_ := -1/m;
  part_rows := CostFunc_params(id=4)[1].value;
  part_cols := CostFunc_params(id=5)[1].value;
  LAMBDA:= CostFunc_params(id=6)[1].value;
	//maps used
	SET OF PBblas.Types.value_t empty_array := [];
	map_c := PBblas.Matrix_Map(NumClass, m, NumClass, part_cols);
	dmap := PBblas.Matrix_Map(Numfeat, m, part_rows, part_cols);
	theta_map := PBblas.Matrix_Map(NumClass, Numfeat, NumClass, part_rows);
	//distribute theta to all the nodes that train data is on
	data_nodesused := dmap.col_blocks;
	Layout_Part_newnode := RECORD (Layout_Part)
		PBblas.Types.node_t new_node_id;
	END;
	Layout_Part_newnode norm_theta (Layout_Part te, INTEGER co) := TRANSFORM
		SELF.new_node_id := co % data_nodesused  ;
		SELF:= te;
	END;
	theta_norm := NORMALIZE(theta, data_nodesused, norm_theta(LEFT, COUNTER) );
	theta_dist := DISTRIBUTE (theta_norm, new_node_id);
	//calculated theta*TrainData
	//M=(theta*x);
	Layout_Part multx(Layout_Part_newnode t_part, Layout_Part x_part):=TRANSFORM
		part_id := x_part.block_col;
		part_c_rows := map_c.part_rows(part_id);
		part_c_cols := map_c.part_cols(part_id);
		k := t_part.part_cols;
		SELF.partition_id := part_id;
		SELF.node_id      := x_part.node_id;
		SELF.block_row    := t_part.block_row;
		SELF.block_col    := x_part.block_col;
		SELF.first_row    := map_c.first_row(part_id);
		SELF.part_rows    := part_c_rows;
		SELF.first_col    := map_c.first_col(part_id);
		SELF.part_cols    := part_c_cols;
		SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
																		part_c_rows, part_c_cols, k,
																		1.0, t_part.mat_part, x_part.mat_part,
																		0.0, empty_array);
  END;
	tx_ := JOIN (theta_dist, TrainData, LEFT.block_col = RIGHT.block_row, multx(LEFT, RIGHT), LOCAL );
	// Sum terms
	//In the transform function check whether the left side is NULL, it can be possible that there is only one partition on one node that needs to rollup
  Layout_Part sumTerms(Layout_Part cumm, Layout_Part term) := TRANSFORM
		HaveTerm := IF(term.part_cols=0, FALSE, TRUE);
    N := cumm.part_rows * cumm.part_cols;
    SELF.mat_part := IF (HaveTerm, PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term.mat_part, 1), cumm.mat_part);
    SELF := cumm;
  END;
	tx_sorted := SORT(tx_, partition_id, LOCAL);
  tx := ROLLUP(tx_sorted, sumTerms(LEFT, RIGHT), partition_id, LOCAL);
	// define a c++ function who does the following operations on each partition
	// M=tx;
	// M = bsxfun(@minus, M, max(M, [], 1));
	// M=exp(M);
	// M = bsxfun(@rdivide, M, sum(M));
	//r : rows
	//s : cols
	// D : input matrix
	SET OF REAL8 soft_fun (PBblas.Types.dimension_t r, PBblas.Types.dimension_t s, PBblas.Types.matrix_t D) := BEGINC++

    #body
    __lenResult = r * s* sizeof(double);
    __isAllResult = false;
    double * result = new double[r*s];
    __result = (void*) result;
    double *cell = (double*) d;
		double *max_arr = new double[s];
		double *sum_arr = new double[s];
		double max_tmp;
		double sum_tmp;
    uint32_t cells =  r * s;
    uint32_t i;
		uint32_t j;
    uint32_t pos;
		uint32_t posj;
		for (i=0; i<s; i++) {
			pos = i * r;
			max_tmp = cell[pos];
			for (j=1; j<r; j++)
			{
				posj = pos +j;
				if(cell[posj]>max_tmp)
					max_tmp = cell[posj];
			}
			max_arr[i]=max_tmp;
    }
		for (i=0; i<cells; i++) {
			pos = (int) i / r;
			result[i]=exp(cell[i]-max_arr[pos]);
		}
    for (i=0; i<s; i++) {
			sum_tmp = 0;
			pos = i * r;
			for (j=0; j<r; j++)
			{
					sum_tmp = sum_tmp + result[pos+j];
			}
			sum_arr[i]=sum_tmp;
    }
		for (i=0; i<cells; i++) {
			pos = (int) i / r;
			result[i]=result[i] / sum_arr[pos];
		}

  ENDC++;
	Layout_Part soft_tran(Layout_Part le) := TRANSFORM
    SELF.mat_part := soft_fun(le.part_rows, le.part_cols, le.mat_part);
    SELF := le;
  END;
	tx_soft := PROJECT (tx, soft_tran (LEFT), LOCAL);//same as M in MATLAB
	//groundTruth - tx_soft : (groundTruth-M)
	groundTruth := TrainLabel;
	Layout_Part grnd_tran(Layout_Part le, Layout_Part ri) := TRANSFORM
		N := le.part_rows * le.part_cols;
    SELF.mat_part := PBblas.BLAS.daxpy(N, -1.0, ri.mat_part, 1, le.mat_part, 1);
    SELF := le;
  END;
	grnd_tx := JOIN (groundTruth, tx_soft, LEFT.partition_id = RIGHT.partition_id, grnd_tran(LEFT, RIGHT),FULL OUTER, LOCAL );
	//grnd_tx * x' : (groundTruth-M)*x')
	Layout_Part grndtx_xt_mul(Layout_Part g_part, Layout_Part x_part):=TRANSFORM
		part_id     := x_part.block_row;
		part_g_cols := g_part.part_cols;
		part_g_rows := g_part.part_rows;
		part_x_rows := x_part.part_rows;
		part_x_cols := x_part.part_cols;
		k := part_g_cols;
		SELF.partition_id := part_id;
		SELF.node_id      := theta_map.assigned_node(part_id);
		SELF.block_row    := g_part.block_row;
		SELF.block_col    := x_part.block_row;
		SELF.first_row    := theta_map.first_row(part_id);
		SELF.part_rows    := theta_map.part_rows(part_id);
		SELF.first_col    := theta_map.first_col(part_id);
		SELF.part_cols    := theta_map.part_cols(part_id);
		SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, TRUE,
																		part_g_rows, part_x_rows, k,
																		m_, g_part.mat_part, x_part.mat_part,
																		0.0, empty_array);
	END;
	//-1/m*(groundTruth-M)*x')
	grndtx_xt := JOIN (grnd_tx, TrainData, LEFT.block_col = RIGHT.block_col, grndtx_xt_mul(LEFT, RIGHT), LOCAL );
	
	Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
	  HaveRight := IF(ri.part_cols=0, FALSE, TRUE);
		N := le.part_rows * le.part_cols ;
		SELF.mat_part := IF (HaveRight, PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1), le.mat_part);
		SELF := le;
	END;
	grndtx_xt_dist := DISTRIBUTE (grndtx_xt, node_id);
	grndtx_xt_dist_sorted := SORT (grndtx_xt_dist, partition_id, LOCAL);
	grndtx_xt_m := ROLLUP(grndtx_xt_dist_sorted, LEFT.partition_id = RIGHT.partition_id, addup(LEFT, RIGHT), LOCAL);
	//calculate theta_grad
	Layout_Part theta_grad_tran(Layout_Part le, Layout_Part ri) := TRANSFORM
		N := le.part_rows * le.part_cols ;
		SELF.mat_part := PBblas.BLAS.daxpy(N, LAMBDA, le.mat_part, 1, ri.mat_part, 1);
		SELF := le;
	END;
	theta_grad := JOIN (theta, grndtx_xt_m,LEFT.partition_id = RIGHT.partition_id, theta_grad_tran(LEFT,RIGHT), FULL OUTER, LOCAL);
	//calculate cost
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
	simple := {Pbblas.types.value_t v};
	simple wd_tran (Layout_Part le) := TRANSFORM
		SELF.v := sum_sq(le.part_cols * le.part_rows, le.mat_part);
	END;
	weightdecay_term_ := PROJECT (theta, wd_tran (LEFT), LOCAL);
	weightdecay_term := SUM (weightdecay_term_, weightdecay_term_.v);
	// tmp=log(M).*groundTruth;
  // log_cost=sum(tmp(:));
	REAL8 log_cost_c (PBblas.Types.dimension_t N, PBblas.Types.matrix_t M, PBblas.Types.matrix_t D) := BEGINC++

    #body
    double result = 0;
		double tmpp ;
    double *cellm = (double*) m;
		double *celld = (double*) d;
    uint32_t i;
		for (i=0; i<n; i++) {
      result = result + (log(cellm[i])*celld[i]);
    }
		return(result);
  ENDC++;
	simple cost_log_tran (Layout_Part le, Layout_Part ri) := TRANSFORM
		SELF.v := log_cost_c(le.part_cols * le.part_rows, le.mat_part, ri.mat_part);
	END;
	log_cost_ := JOIN (tx_soft, groundTruth, LEFT.partition_id = RIGHT.partition_id, cost_log_tran(LEFT,RIGHT), LOCAL);
	log_cost := SUM (log_cost_, log_cost_.v);
	//cost=((-1/m)*log_cost)+((lambda/2)*weightdecay_term);
	cost := m_*log_cost + ((LAMBDA/2)*weightdecay_term);
	//return results
	return_record := RECORD (Layout_Part)
		REAL8 cost_value;
	END;
	theta_cost := PROJECT (theta_grad, TRANSFORM (return_record, SELF.cost_value := cost; SELF:=LEFT), LOCAL);
	theta_cost_check :=  ASSERT(theta_cost, node_id=Thorlib.node() and node_id=(partition_id-1), 'softmax gradient is not distributed correctly', FAIL);
	ToReturn := theta_cost_check;

  RETURN  PROJECT (grndtx_xt, TRANSFORM (return_record, SELF.cost_value := 10; SELF:=LEFT), LOCAL);
	// RETURN  ToReturn;

  END; //END SoftMax_compatible_lbfgs_sparse_test
	// implementation of SoftMax_compatible_lbfgs where it works for sparse matrices (where some partitions are missing ude to being all zero). basically i just added full outer to daxpy kind of joins
	EXPORT  SoftMax_compatible_lbfgs_sparse( DATASET(Layout_Part) theta, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Part) TrainData , DATASET(Layout_Part) TrainLabel) := function
  Numfeat:= CostFunc_params(id=2)[1].value;
  NumClass := CostFunc_params(id=3)[1].value;// number of features
  m := CostFunc_params(id=1)[1].value;// number of samples
	m_ := -1/m;
  part_rows := CostFunc_params(id=4)[1].value;
  part_cols := CostFunc_params(id=5)[1].value;
  LAMBDA:= CostFunc_params(id=6)[1].value;
	//maps used
	SET OF PBblas.Types.value_t empty_array := [];
	map_c := PBblas.Matrix_Map(NumClass, m, NumClass, part_cols);
	dmap := PBblas.Matrix_Map(Numfeat, m, part_rows, part_cols);
	theta_map := PBblas.Matrix_Map(NumClass, Numfeat, NumClass, part_rows);
	//distribute theta to all the nodes that train data is on
	data_nodesused := dmap.col_blocks;
	Layout_Part_newnode := RECORD (Layout_Part)
		PBblas.Types.node_t new_node_id;
	END;
	Layout_Part_newnode norm_theta (Layout_Part te, INTEGER co) := TRANSFORM
		SELF.new_node_id := co % data_nodesused  ;
		SELF:= te;
	END;
	theta_norm := NORMALIZE(theta, data_nodesused, norm_theta(LEFT, COUNTER) );
	theta_dist := DISTRIBUTE (theta_norm, new_node_id);
	//calculated theta*TrainData
	//M=(theta*x);
	Layout_Part multx(Layout_Part_newnode t_part, Layout_Part x_part):=TRANSFORM
		part_id := x_part.block_col;
		part_c_rows := map_c.part_rows(part_id);
		part_c_cols := map_c.part_cols(part_id);
		k := t_part.part_cols;
		SELF.partition_id := part_id;
		SELF.node_id      := x_part.node_id;
		SELF.block_row    := t_part.block_row;
		SELF.block_col    := x_part.block_col;
		SELF.first_row    := map_c.first_row(part_id);
		SELF.part_rows    := part_c_rows;
		SELF.first_col    := map_c.first_col(part_id);
		SELF.part_cols    := part_c_cols;
		SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
																		part_c_rows, part_c_cols, k,
																		1.0, t_part.mat_part, x_part.mat_part,
																		0.0, empty_array);
  END;
	tx_ := JOIN (theta_dist, TrainData, LEFT.block_col = RIGHT.block_row, multx(LEFT, RIGHT), LOCAL );
	// Sum terms
	//In the transform function check whether the left side is NULL, it can be possible that there is only one partition on one node that needs to rollup
  Layout_Part sumTerms(Layout_Part cumm, Layout_Part term) := TRANSFORM
		HaveTerm := IF(term.part_cols=0, FALSE, TRUE);
    N := cumm.part_rows * cumm.part_cols;
    SELF.mat_part := IF (HaveTerm, PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term.mat_part, 1), cumm.mat_part);
    SELF := cumm;
  END;
	tx_sorted := SORT(tx_, partition_id, LOCAL);
  tx := ROLLUP(tx_sorted, sumTerms(LEFT, RIGHT), partition_id, LOCAL);
	// define a c++ function who does the following operations on each partition
	// M=tx;
	// M = bsxfun(@minus, M, max(M, [], 1));
	// M=exp(M);
	// M = bsxfun(@rdivide, M, sum(M));
	//r : rows
	//s : cols
	// D : input matrix
	SET OF REAL8 soft_fun (PBblas.Types.dimension_t r, PBblas.Types.dimension_t s, PBblas.Types.matrix_t D) := BEGINC++

    #body
    __lenResult = r * s* sizeof(double);
    __isAllResult = false;
    double * result = new double[r*s];
    __result = (void*) result;
    double *cell = (double*) d;
		double *max_arr = new double[s];
		double *sum_arr = new double[s];
		double max_tmp;
		double sum_tmp;
    uint32_t cells =  r * s;
    uint32_t i;
		uint32_t j;
    uint32_t pos;
		uint32_t posj;
		for (i=0; i<s; i++) {
			pos = i * r;
			max_tmp = cell[pos];
			for (j=1; j<r; j++)
			{
				posj = pos +j;
				if(cell[posj]>max_tmp)
					max_tmp = cell[posj];
			}
			max_arr[i]=max_tmp;
    }
		for (i=0; i<cells; i++) {
			pos = (int) i / r;
			result[i]=exp(cell[i]-max_arr[pos]);
		}
    for (i=0; i<s; i++) {
			sum_tmp = 0;
			pos = i * r;
			for (j=0; j<r; j++)
			{
					sum_tmp = sum_tmp + result[pos+j];
			}
			sum_arr[i]=sum_tmp;
    }
		for (i=0; i<cells; i++) {
			pos = (int) i / r;
			result[i]=result[i] / sum_arr[pos];
		}

  ENDC++;
	Layout_Part soft_tran(Layout_Part le) := TRANSFORM
    SELF.mat_part := soft_fun(le.part_rows, le.part_cols, le.mat_part);
    SELF := le;
  END;
	tx_soft := PROJECT (tx, soft_tran (LEFT), LOCAL);//same as M in MATLAB
	//groundTruth - tx_soft : (groundTruth-M)
	groundTruth := TrainLabel;
	Layout_Part grnd_tran(Layout_Part le, Layout_Part ri) := TRANSFORM
		N := le.part_rows * le.part_cols;
    SELF.mat_part := PBblas.BLAS.daxpy(N, -1.0, ri.mat_part, 1, le.mat_part, 1);
    SELF := le;
  END;
	grnd_tx := JOIN (groundTruth, tx_soft, LEFT.partition_id = RIGHT.partition_id, grnd_tran(LEFT, RIGHT),FULL OUTER, LOCAL );
	//grnd_tx * x' : (groundTruth-M)*x')
	Layout_Part grndtx_xt_mul(Layout_Part g_part, Layout_Part x_part):=TRANSFORM
		part_id     := x_part.block_row;
		part_g_cols := g_part.part_cols;
		part_g_rows := g_part.part_rows;
		part_x_rows := x_part.part_rows;
		part_x_cols := x_part.part_cols;
		k := part_g_cols;
		SELF.partition_id := part_id;
		SELF.node_id      := theta_map.assigned_node(part_id);
		SELF.block_row    := g_part.block_row;
		SELF.block_col    := x_part.block_row;
		SELF.first_row    := theta_map.first_row(part_id);
		SELF.part_rows    := theta_map.part_rows(part_id);
		SELF.first_col    := theta_map.first_col(part_id);
		SELF.part_cols    := theta_map.part_cols(part_id);
		SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, TRUE,
																		part_g_rows, part_x_rows, k,
																		m_, g_part.mat_part, x_part.mat_part,
																		0.0, empty_array);
	END;
	//-1/m*(groundTruth-M)*x')
	grndtx_xt := JOIN (grnd_tx, TrainData, LEFT.block_col = RIGHT.block_col, grndtx_xt_mul(LEFT, RIGHT), LOCAL );
	
	Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
	  HaveRight := IF(ri.part_cols=0, FALSE, TRUE);
		N := le.part_rows * le.part_cols ;
		SELF.mat_part := IF (HaveRight, PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1), le.mat_part);
		SELF := le;
	END;
	grndtx_xt_dist := DISTRIBUTE (grndtx_xt, node_id);
	grndtx_xt_dist_sorted := SORT (grndtx_xt_dist, partition_id, LOCAL);
	grndtx_xt_m := ROLLUP(grndtx_xt_dist_sorted, LEFT.partition_id = RIGHT.partition_id, addup(LEFT, RIGHT), LOCAL);
	//calculate theta_grad
	Layout_Part theta_grad_tran(Layout_Part le, Layout_Part ri) := TRANSFORM
		N := le.part_rows * le.part_cols ;
		SELF.mat_part := PBblas.BLAS.daxpy(N, LAMBDA, le.mat_part, 1, ri.mat_part, 1);
		SELF := le;
	END;
	theta_grad := JOIN (theta, grndtx_xt_m,LEFT.partition_id = RIGHT.partition_id, theta_grad_tran(LEFT,RIGHT), FULL OUTER, LOCAL);
	//calculate cost
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
	simple := {Pbblas.types.value_t v};
	simple wd_tran (Layout_Part le) := TRANSFORM
		SELF.v := sum_sq(le.part_cols * le.part_rows, le.mat_part);
	END;
	weightdecay_term_ := PROJECT (theta, wd_tran (LEFT), LOCAL);
	weightdecay_term := SUM (weightdecay_term_, weightdecay_term_.v);
	// tmp=log(M).*groundTruth;
  // log_cost=sum(tmp(:));
	REAL8 log_cost_c (PBblas.Types.dimension_t N, PBblas.Types.matrix_t M, PBblas.Types.matrix_t D) := BEGINC++

    #body
    double result = 0;
		double tmpp ;
    double *cellm = (double*) m;
		double *celld = (double*) d;
    uint32_t i;
		for (i=0; i<n; i++) {
      result = result + (log(cellm[i])*celld[i]);
    }
		return(result);
  ENDC++;
	simple cost_log_tran (Layout_Part le, Layout_Part ri) := TRANSFORM
		SELF.v := log_cost_c(le.part_cols * le.part_rows, le.mat_part, ri.mat_part);
	END;
	log_cost_ := JOIN (tx_soft, groundTruth, LEFT.partition_id = RIGHT.partition_id, cost_log_tran(LEFT,RIGHT), LOCAL);
	log_cost := SUM (log_cost_, log_cost_.v);
	//cost=((-1/m)*log_cost)+((lambda/2)*weightdecay_term);
	cost := m_*log_cost + ((LAMBDA/2)*weightdecay_term);
	//return results
	return_record := RECORD (Layout_Part)
		REAL8 cost_value;
	END;
	theta_cost := PROJECT (theta_grad, TRANSFORM (return_record, SELF.cost_value := cost; SELF:=LEFT), LOCAL);
	theta_cost_check :=  ASSERT(theta_cost, node_id=Thorlib.node() and node_id=(partition_id-1), 'softmax gradient is not distributed correctly', FAIL);
	ToReturn := theta_cost_check;

  // RETURN  PROJECT (tx_, TRANSFORM (return_record, SELF.cost_value := 10; SELF:=LEFT), LOCAL);
	RETURN  ToReturn;

  END; //END SoftMax_compatible_lbfgs_sparse

// implementation of SoftMax_compatible_lbfgs where it works for sparse matrices (where some partitions are missing ude to being all zero). basically i just added full outer to daxpy kind of joins
	EXPORT  SoftMax_compatible_lbfgs_sparse_partitions( DATASET(Layout_Part) theta, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Part) TrainData , DATASET(Layout_Part) TrainLabel) := function
  Numfeat:= CostFunc_params(id=2)[1].value;
  NumClass := CostFunc_params(id=3)[1].value;// number of features
  m := CostFunc_params(id=1)[1].value;// number of samples
	m_ := -1/m;
  part_rows := CostFunc_params(id=4)[1].value;
  part_cols := CostFunc_params(id=5)[1].value;
  LAMBDA:= CostFunc_params(id=6)[1].value;
	//maps used
	SET OF PBblas.Types.value_t empty_array := [];
	map_c := PBblas.Matrix_Map(NumClass, m, NumClass, part_cols);
	dmap := PBblas.Matrix_Map(Numfeat, m, part_rows, part_cols);
	theta_map := PBblas.Matrix_Map(NumClass, Numfeat, NumClass, part_rows);
	//distribute theta to all the nodes that train data is on
	nodes_available := Thorlib.nodes();
	data_nodesused := MIN(nodes_available, dmap.col_blocks);
	Layout_Part_newnode := RECORD (Layout_Part)
		PBblas.Types.node_t new_node_id;
	END;
	Layout_Part_newnode norm_theta (Layout_Part te, INTEGER co) := TRANSFORM
		SELF.new_node_id := co % data_nodesused;
		SELF:= te;
	END;
	theta_norm := NORMALIZE(theta, data_nodesused, norm_theta(LEFT, COUNTER) );
	theta_dist := DISTRIBUTE (theta_norm, new_node_id);//with this dirtsibution the whole theta matrix is available on all nodes
	//calculated theta*TrainData
	//M=(theta*x);
	Layout_Part multx(Layout_Part_newnode t_part, Layout_Part x_part):=TRANSFORM
		part_id := x_part.block_col;
		part_c_rows := map_c.part_rows(part_id);
		part_c_cols := map_c.part_cols(part_id);
		k := t_part.part_cols;
		SELF.partition_id := part_id;
		SELF.node_id      := x_part.node_id;
		SELF.block_row    := t_part.block_row;
		SELF.block_col    := x_part.block_col;
		SELF.first_row    := map_c.first_row(part_id);
		SELF.part_rows    := part_c_rows;
		SELF.first_col    := map_c.first_col(part_id);
		SELF.part_cols    := part_c_cols;
		SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
																		part_c_rows, part_c_cols, k,
																		1.0, t_part.mat_part, x_part.mat_part,
																		0.0, empty_array);
  END;
	tx_ := JOIN (theta_dist, TrainData, LEFT.block_col = RIGHT.block_row, multx(LEFT, RIGHT), LOCAL );
	// Sum terms
	//In the transform function check whether the left side is NULL, it can be possible that there is only one partition on one node that needs to rollup
  Layout_Part sumTerms(Layout_Part cumm, Layout_Part term) := TRANSFORM
		HaveTerm := IF(term.part_cols=0, FALSE, TRUE);
    N := cumm.part_rows * cumm.part_cols;
    SELF.mat_part := IF (HaveTerm, PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term.mat_part, 1), cumm.mat_part);
    SELF := cumm;
  END;
	tx_sorted := SORT(tx_, partition_id, LOCAL);
  tx := ROLLUP(tx_sorted, sumTerms(LEFT, RIGHT), partition_id, LOCAL);
	// define a c++ function who does the following operations on each partition
	// M=tx;
	// M = bsxfun(@minus, M, max(M, [], 1));
	// M=exp(M);
	// M = bsxfun(@rdivide, M, sum(M));
	//r : rows
	//s : cols
	// D : input matrix
	SET OF REAL8 soft_fun (PBblas.Types.dimension_t r, PBblas.Types.dimension_t s, PBblas.Types.matrix_t D) := BEGINC++

    #body
    __lenResult = r * s* sizeof(double);
    __isAllResult = false;
    double * result = new double[r*s];
    __result = (void*) result;
    double *cell = (double*) d;
		double *max_arr = new double[s];
		double *sum_arr = new double[s];
		double max_tmp;
		double sum_tmp;
    uint32_t cells =  r * s;
    uint32_t i;
		uint32_t j;
    uint32_t pos;
		uint32_t posj;
		for (i=0; i<s; i++) {
			pos = i * r;
			max_tmp = cell[pos];
			for (j=1; j<r; j++)
			{
				posj = pos +j;
				if(cell[posj]>max_tmp)
					max_tmp = cell[posj];
			}
			max_arr[i]=max_tmp;
    }
		for (i=0; i<cells; i++) {
			pos = (int) i / r;
			result[i]=exp(cell[i]-max_arr[pos]);
		}
    for (i=0; i<s; i++) {
			sum_tmp = 0;
			pos = i * r;
			for (j=0; j<r; j++)
			{
					sum_tmp = sum_tmp + result[pos+j];
			}
			sum_arr[i]=sum_tmp;
    }
		for (i=0; i<cells; i++) {
			pos = (int) i / r;
			result[i]=result[i] / sum_arr[pos];
		}

  ENDC++;
	Layout_Part soft_tran(Layout_Part le) := TRANSFORM
    SELF.mat_part := soft_fun(le.part_rows, le.part_cols, le.mat_part);
    SELF := le;
  END;
	tx_soft := PROJECT (tx, soft_tran (LEFT), LOCAL);//same as M in MATLAB
	//groundTruth - tx_soft : (groundTruth-M)
	groundTruth := TrainLabel;
	Layout_Part grnd_tran(Layout_Part le, Layout_Part ri) := TRANSFORM
		N := le.part_rows * le.part_cols;
    SELF.mat_part := PBblas.BLAS.daxpy(N, -1.0, ri.mat_part, 1, le.mat_part, 1);
    SELF := le;
  END;
	// grnd_tx := JOIN (groundTruth, tx_soft, LEFT.partition_id = RIGHT.partition_id, grnd_tran(LEFT, RIGHT),FULL OUTER, LOCAL );
	grnd_tx := Pbblas.PB_daxpy(-1, tx_soft, groundTruth);
	//grnd_tx * x' : (groundTruth-M)*x')
	Layout_Part grndtx_xt_mul(Layout_Part g_part, Layout_Part x_part):=TRANSFORM
		part_id     := x_part.block_row;
		part_g_cols := g_part.part_cols;
		part_g_rows := g_part.part_rows;
		part_x_rows := x_part.part_rows;
		part_x_cols := x_part.part_cols;
		k := part_g_cols;
		SELF.partition_id := part_id;
		SELF.node_id      := theta_map.assigned_node(part_id);
		SELF.block_row    := g_part.block_row;
		SELF.block_col    := x_part.block_row;
		SELF.first_row    := theta_map.first_row(part_id);
		SELF.part_rows    := theta_map.part_rows(part_id);
		SELF.first_col    := theta_map.first_col(part_id);
		SELF.part_cols    := theta_map.part_cols(part_id);
		SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, TRUE,
																		part_g_rows, part_x_rows, k,
																		m_, g_part.mat_part, x_part.mat_part,
																		0.0, empty_array);
	END;
	//-1/m*(groundTruth-M)*x')
	grndtx_xt := JOIN (grnd_tx, TrainData, LEFT.block_col = RIGHT.block_col, grndtx_xt_mul(LEFT, RIGHT), LOCAL );
	
	Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
	  HaveRight := IF(ri.part_cols=0, FALSE, TRUE);
		N := le.part_rows * le.part_cols ;
		SELF.mat_part := IF (HaveRight, PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1), le.mat_part);
		SELF := le;
	END;
	grndtx_xt_dist := DISTRIBUTE (grndtx_xt, node_id);
	grndtx_xt_dist_sorted := SORT (grndtx_xt_dist, partition_id, LOCAL);
	grndtx_xt_m := ROLLUP(grndtx_xt_dist_sorted, LEFT.partition_id = RIGHT.partition_id, addup(LEFT, RIGHT), LOCAL);
	//calculate theta_grad
	Layout_Part theta_grad_tran(Layout_Part le, Layout_Part ri) := TRANSFORM
		N := le.part_rows * le.part_cols ;
		SELF.mat_part := PBblas.BLAS.daxpy(N, LAMBDA, le.mat_part, 1, ri.mat_part, 1);
		SELF := le;
	END;
	theta_grad := JOIN (theta, grndtx_xt_m,LEFT.partition_id = RIGHT.partition_id, theta_grad_tran(LEFT,RIGHT), FULL OUTER, LOCAL);
	//calculate cost
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
	simple := {Pbblas.types.value_t v};
	simple wd_tran (Layout_Part le) := TRANSFORM
		SELF.v := sum_sq(le.part_cols * le.part_rows, le.mat_part);
	END;
	weightdecay_term_ := PROJECT (theta, wd_tran (LEFT), LOCAL);
	weightdecay_term := SUM (weightdecay_term_, weightdecay_term_.v);
	// tmp=log(M).*groundTruth;
  // log_cost=sum(tmp(:));
	REAL8 log_cost_c (PBblas.Types.dimension_t N, PBblas.Types.matrix_t M, PBblas.Types.matrix_t D) := BEGINC++

    #body
    double result = 0;
		double tmpp ;
    double *cellm = (double*) m;
		double *celld = (double*) d;
    uint32_t i;
		for (i=0; i<n; i++) {
      result = result + (log(cellm[i])*celld[i]);
    }
		return(result);
  ENDC++;
	simple cost_log_tran (Layout_Part le, Layout_Part ri) := TRANSFORM
		SELF.v := log_cost_c(le.part_cols * le.part_rows, le.mat_part, ri.mat_part);
	END;
	log_cost_ := JOIN (tx_soft, groundTruth, LEFT.partition_id = RIGHT.partition_id, cost_log_tran(LEFT,RIGHT), LOCAL);
	log_cost := SUM (log_cost_, log_cost_.v);
	//cost=((-1/m)*log_cost)+((lambda/2)*weightdecay_term);
	cost := m_*log_cost + ((LAMBDA/2)*weightdecay_term);
	//return results
	return_record := RECORD (Layout_Part)
		REAL8 cost_value;
	END;
	theta_cost := PROJECT (theta_grad, TRANSFORM (return_record, SELF.cost_value := cost; SELF:=LEFT), LOCAL);
	theta_cost_check :=  ASSERT(theta_cost, node_id=Thorlib.node() and node_id=theta_map.assigned_node (partition_id), 'softmax gradient is not distributed correctly', FAIL);
	ToReturn := theta_cost_check;

  // RETURN  PROJECT (tx_, TRANSFORM (return_record, SELF.cost_value := 10; SELF:=LEFT), LOCAL);
	RETURN  ToReturn;

  END; //END SoftMax_compatible_lbfgs_sparse_partitions


//the implementation of SoftMax_compatible_lbfgs_sparse_partitions where the whole data is distributed on all the nodes to calculate t_x instead of theta being distribited on all the nodes
EXPORT  SoftMax_compatible_lbfgs_sparse_partitions_datadist( DATASET(Layout_Part) theta, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Part) TrainData , DATASET(Layout_Part) TrainLabel) := function
  Numfeat:= CostFunc_params(id=2)[1].value;
  NumClass := CostFunc_params(id=3)[1].value;// number of features
  m := CostFunc_params(id=1)[1].value;// number of samples
	m_ := -1/m;
  part_rows := CostFunc_params(id=4)[1].value;
  part_cols := CostFunc_params(id=5)[1].value;
  LAMBDA:= CostFunc_params(id=6)[1].value;
	//maps used
	SET OF PBblas.Types.value_t empty_array := [];
	map_c := PBblas.Matrix_Map(NumClass, m, NumClass, part_cols);// tx map (tx is of size numebr of classe by number of samples)
	dmap := PBblas.Matrix_Map(Numfeat, m, part_rows, part_cols);
	theta_map := PBblas.Matrix_Map(NumClass, Numfeat, NumClass, part_rows);
	//distribute theta to all the nodes that train data is on
	nodes_available := Thorlib.nodes();
	data_nodesused := MIN(nodes_available, dmap.row_blocks);
	Layout_Part_newnode := RECORD (Layout_Part)
		PBblas.Types.node_t new_node_id;
	END;
	Layout_Part_newnode norm_theta (Layout_Part te, INTEGER co) := TRANSFORM
		SELF.new_node_id := co % data_nodesused;
		SELF:= te;
	END;
	theta_norm := NORMALIZE(theta, data_nodesused, norm_theta(LEFT, COUNTER) );
	theta_dist := DISTRIBUTE (theta_norm, new_node_id);//with this dirtsibution the whole theta matrix is available on all nodes
	//calculated theta*TrainData
	//M=(theta*x);
	Layout_Part multx(Layout_Part t_part, Layout_Part x_part):=TRANSFORM
		part_id := x_part.block_col;
		part_c_rows := map_c.part_rows(part_id);
		part_c_cols := map_c.part_cols(part_id);
		k := t_part.part_cols;
		SELF.partition_id := part_id;
		SELF.node_id      := map_c.assigned_node(part_id);
		SELF.block_row    := t_part.block_row;
		SELF.block_col    := x_part.block_col;
		SELF.first_row    := map_c.first_row(part_id);
		SELF.part_rows    := part_c_rows;
		SELF.first_col    := map_c.first_col(part_id);
		SELF.part_cols    := part_c_cols;
		SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
																		part_c_rows, part_c_cols, k,
																		1.0, t_part.mat_part, x_part.mat_part,
																		0.0, empty_array);
  END;
	tx_ := JOIN (theta, TrainData, LEFT.block_col = RIGHT.block_row, multx(LEFT, RIGHT), LOCAL );
	tx_dist := DISTRIBUTE (tx_, node_id);
	// Sum terms
	//In the transform function check whether the left side is NULL, it can be possible that there is only one partition on one node that needs to rollup
  Layout_Part sumTerms(Layout_Part cumm, Layout_Part term) := TRANSFORM
		HaveTerm := IF(term.part_cols=0, FALSE, TRUE);
    N := cumm.part_rows * cumm.part_cols;
    SELF.mat_part := IF (HaveTerm, PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term.mat_part, 1), cumm.mat_part);
    SELF := cumm;
  END;
	tx_sorted := SORT(tx_dist, partition_id, LOCAL);
  tx := ROLLUP(tx_sorted, sumTerms(LEFT, RIGHT), partition_id, LOCAL);
	// define a c++ function who does the following operations on each partition
	// M=tx;
	// M = bsxfun(@minus, M, max(M, [], 1));
	// M=exp(M);
	// M = bsxfun(@rdivide, M, sum(M));
	//r : rows
	//s : cols
	// D : input matrix
	SET OF REAL8 soft_fun (PBblas.Types.dimension_t r, PBblas.Types.dimension_t s, PBblas.Types.matrix_t D) := BEGINC++

    #body
    __lenResult = r * s* sizeof(double);
    __isAllResult = false;
    double * result = new double[r*s];
    __result = (void*) result;
    double *cell = (double*) d;
		double *max_arr = new double[s];
		double *sum_arr = new double[s];
		double max_tmp;
		double sum_tmp;
    uint32_t cells =  r * s;
    uint32_t i;
		uint32_t j;
    uint32_t pos;
		uint32_t posj;
		for (i=0; i<s; i++) {
			pos = i * r;
			max_tmp = cell[pos];
			for (j=1; j<r; j++)
			{
				posj = pos +j;
				if(cell[posj]>max_tmp)
					max_tmp = cell[posj];
			}
			max_arr[i]=max_tmp;
    }
		for (i=0; i<cells; i++) {
			pos = (int) i / r;
			result[i]=exp(cell[i]-max_arr[pos]);
		}
    for (i=0; i<s; i++) {
			sum_tmp = 0;
			pos = i * r;
			for (j=0; j<r; j++)
			{
					sum_tmp = sum_tmp + result[pos+j];
			}
			sum_arr[i]=sum_tmp;
    }
		for (i=0; i<cells; i++) {
			pos = (int) i / r;
			result[i]=result[i] / sum_arr[pos];
		}

  ENDC++; // END soft_fun
	Layout_Part soft_tran(Layout_Part le) := TRANSFORM
    SELF.mat_part := soft_fun(le.part_rows, le.part_cols, le.mat_part);
    SELF := le;
  END;
	tx_soft := PROJECT (tx, soft_tran (LEFT), LOCAL);//same as M in MATLAB
	//groundTruth - tx_soft : (groundTruth-M)
	groundTruth := TrainLabel;
	Layout_Part grnd_tran(Layout_Part le, Layout_Part ri) := TRANSFORM
		N := le.part_rows * le.part_cols;
    SELF.mat_part := PBblas.BLAS.daxpy(N, -1.0, ri.mat_part, 1, le.mat_part, 1);
    SELF := le;
  END;
	// grnd_tx := JOIN (groundTruth, tx_soft, LEFT.partition_id = RIGHT.partition_id, grnd_tran(LEFT, RIGHT),FULL OUTER, LOCAL );
	grnd_tx := Pbblas.PB_daxpy(-1, tx_soft, groundTruth);

	Layout_Part_newnode norm_grndtx (Layout_Part te, INTEGER co) := TRANSFORM
		SELF.new_node_id := co % data_nodesused;
		SELF:= te;
	END;
	grnd_tx_norm := NORMALIZE(grnd_tx, data_nodesused, norm_grndtx(LEFT, COUNTER) );
	grnd_tx_dist := DISTRIBUTE (grnd_tx_norm, new_node_id);//with this dirtsibution the whole theta matrix is available on all nodes

	//grnd_tx * x' : (groundTruth-M)*x')
	Layout_Part grndtx_xt_mul(Layout_Part_newnode g_part, Layout_Part x_part):=TRANSFORM
		part_id     := x_part.block_row;
		part_g_cols := g_part.part_cols;
		part_g_rows := g_part.part_rows;
		part_x_rows := x_part.part_rows;
		part_x_cols := x_part.part_cols;
		k := part_g_cols;
		SELF.partition_id := part_id;
		SELF.node_id      := theta_map.assigned_node(part_id);
		SELF.block_row    := g_part.block_row;
		SELF.block_col    := x_part.block_row;
		SELF.first_row    := theta_map.first_row(part_id);
		SELF.part_rows    := theta_map.part_rows(part_id);
		SELF.first_col    := theta_map.first_col(part_id);
		SELF.part_cols    := theta_map.part_cols(part_id);
		SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, TRUE,
																		part_g_rows, part_x_rows, k,
																		m_, g_part.mat_part, x_part.mat_part,
																		0.0, empty_array);
	END;
	//-1/m*(groundTruth-M)*x')
	grndtx_xt := JOIN (grnd_tx_dist, TrainData, LEFT.block_col = RIGHT.block_col, grndtx_xt_mul(LEFT, RIGHT), LOCAL );
	
	Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
	  HaveRight := IF(ri.part_cols=0, FALSE, TRUE);
		N := le.part_rows * le.part_cols ;
		SELF.mat_part := IF (HaveRight, PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1), le.mat_part);
		SELF := le;
	END;
	grndtx_xt_dist := grndtx_xt;// already distributed based on node_id
	grndtx_xt_dist_sorted := SORT (grndtx_xt_dist, partition_id, LOCAL);
	grndtx_xt_m := ROLLUP(grndtx_xt_dist_sorted, LEFT.partition_id = RIGHT.partition_id, addup(LEFT, RIGHT), LOCAL);
	//calculate theta_grad
	Layout_Part theta_grad_tran(Layout_Part le, Layout_Part ri) := TRANSFORM
		N := le.part_rows * le.part_cols ;
		SELF.mat_part := PBblas.BLAS.daxpy(N, LAMBDA, le.mat_part, 1, ri.mat_part, 1);
		SELF := le;
	END;
	theta_grad := JOIN (theta, grndtx_xt_m,LEFT.partition_id = RIGHT.partition_id, theta_grad_tran(LEFT,RIGHT), FULL OUTER, LOCAL);
	//calculate cost
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
	simple := {Pbblas.types.value_t v};
	simple wd_tran (Layout_Part le) := TRANSFORM
		SELF.v := sum_sq(le.part_cols * le.part_rows, le.mat_part);
	END;
	weightdecay_term_ := PROJECT (theta, wd_tran (LEFT), LOCAL);
	weightdecay_term := SUM (weightdecay_term_, weightdecay_term_.v);
	// tmp=log(M).*groundTruth;
  // log_cost=sum(tmp(:));
	REAL8 log_cost_c (PBblas.Types.dimension_t N, PBblas.Types.matrix_t M, PBblas.Types.matrix_t D) := BEGINC++

    #body
    double result = 0;
		double tmpp ;
    double *cellm = (double*) m;
		double *celld = (double*) d;
    uint32_t i;
		for (i=0; i<n; i++) {
      result = result + (log(cellm[i])*celld[i]);
    }
		return(result);
  ENDC++;
	simple cost_log_tran (Layout_Part le, Layout_Part ri) := TRANSFORM
		SELF.v := log_cost_c(le.part_cols * le.part_rows, le.mat_part, ri.mat_part);
	END;
	log_cost_ := JOIN (tx_soft, groundTruth, LEFT.partition_id = RIGHT.partition_id, cost_log_tran(LEFT,RIGHT), LOCAL);
	log_cost := SUM (log_cost_, log_cost_.v);
	//cost=((-1/m)*log_cost)+((lambda/2)*weightdecay_term);
	cost := m_*log_cost + ((LAMBDA/2)*weightdecay_term);
	//return results
	return_record := RECORD (Layout_Part)
		REAL8 cost_value;
	END;
	theta_cost := PROJECT (theta_grad, TRANSFORM (return_record, SELF.cost_value := cost; SELF:=LEFT), LOCAL);
	theta_cost_check :=  ASSERT(theta_cost, node_id=Thorlib.node() and node_id=theta_map.assigned_node (partition_id), 'softmax gradient is not distributed correctly', FAIL);
	ToReturn := theta_cost_check;

  // RETURN  PROJECT (tx, TRANSFORM (return_record, SELF.cost_value := 10; SELF:=LEFT), LOCAL);
	RETURN  ToReturn;
	

  END; //END SoftMax_compatible_lbfgs_sparse_partitions_datadist

// The implementation of SoftMax_compatible_lbfgs where train data is provided in numericfield format and theta is in Pbblas format where it is partitioned row-wise
// if theta is l*f matrix , l is partitioned to part_rows parts and the partitions are of side part_rows * f
// I decided on this implementation because of the lhtcs large dataset which would cause memory issues with SoftMax_compatible_lbfgs_sparse_partitions_datadist implementation
//train data is distributed on all the nodes theta is distributed on. since train data is in numerifcfield and sparse format, it does not take a lot of memory and can be distributed on all the nodes/nodes theta is distributed on
EXPORT  SoftMax_compatible_lbfgs_sparse_label( DATASET(Layout_Part) theta, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Cell_nid)  TrainData , DATASET(Layout_Part) TrainLabel) := function
  Numfeat:= CostFunc_params(id=2)[1].value;
  NumClass := CostFunc_params(id=3)[1].value;// number of features
  m := CostFunc_params(id=1)[1].value;// number of samples
	m_ := -1/m;
  part_rows := CostFunc_params(id=4)[1].value;// this is used to partition the rows in theta matrix
  part_cols := CostFunc_params(id=5)[1].value;// this is not used in this version of the softmax implementation
  LAMBDA:= CostFunc_params(id=6)[1].value;
	//maps used
	SET OF PBblas.Types.value_t empty_array := [];
	theta_map := PBblas.Matrix_Map(NumClass, Numfeat, part_rows, Numfeat);
	map_c := PBblas.Matrix_Map(NumClass, m, part_rows, m);
	//calculated theta*TrainData
	//M=(theta*x);
	//part_sparse_mul multiplies matrix M in Pbblas format of size r by s to the matrix D which is in numericfield format and its size is s by samp and the size od matrix D is num. Natrix D is a sparse matrix so its size is not necessarily s*samp
	SET OF REAL8 part_sparse_mul(PBblas.types.dimension_t r, PBblas.types.dimension_t s, PBblas.types.dimension_t samp, PBblas.types.dimension_t num, PBblas.types.matrix_t M, DATASET(PBblas.types.Layout_Cell) D) := BEGINC++
    typedef struct work3 {      // copy of numericfield translated to C
      uint32_t x;
      uint32_t y;
      double v;
    };
    #body
    __lenResult = r * samp * sizeof(double);
    __isAllResult = false;
    double *result = new double[r * samp];
    __result = (void*) result;
    work3 *celld = (work3*) d;
		double *cellm = (double*) m;
		uint32_t cells = num;
    uint32_t i, j;
    uint32_t pos;
    for (i=0; i< r * samp; i++) {
      result[i] =  0.0;
    }
		uint32_t x, y;
		for (i=0; i < cells; i++){
			x = celld[i].x - 1;   // input co-ordinates are one based,
      y = celld[i].y - 1;  //x and y are zero based.
			for (j=0; j<r; j++){
				pos = y * r + j;
				result[pos] = result[pos] + cellm[r * x + j] * celld[i].v;
			}		
		}
  ENDC++;
	PBblas.Types.Layout_Part txtran (PBblas.Types.Layout_Part le, DATASET(PBblas.Types.Layout_Cell) cells) := TRANSFORM
		part_id := le.partition_id;;
		SELF.partition_id := part_id;
		SELF.node_id      := map_c.assigned_node(part_id);
		SELF.block_row    := le.block_row;
		SELF.block_col    := le.block_col;
		SELF.first_row    := map_c.first_row(part_id);
		SELF.part_rows    := map_c.part_rows(part_id);
		SELF.first_col    := map_c.first_col(part_id);
		SELF.part_cols    := map_c.part_cols(part_id);
		SELF.mat_part := part_sparse_mul(le.part_rows, le.part_cols, m, COUNT (cells), le.mat_part, PROJECT(cells, TRANSFORM (PBblas.Types.layout_cell, SELF:= LEFT)));
		SELF := le;
	END;
	
	tx := DENORMALIZE(theta, TrainData,
                            LEFT.node_id = RIGHT.node_id,
                            GROUP,
                            txtran(LEFT,ROWS(RIGHT)), LOCAL);
	// M=tx;
	// M = bsxfun(@minus, M, max(M, [], 1));
	// first calculate  max(M, [], 1)
	SET OF REAL8 max_col (PBblas.Types.dimension_t r, PBblas.Types.dimension_t s, PBblas.Types.matrix_t D) := BEGINC++

    #body
    __lenResult = s * sizeof(double);
    __isAllResult = false;
    double * result = new double[s];
    __result = (void*) result;
    double *cell = (double*) d;
		double max_tmp;
    uint32_t i;
		uint32_t j;
    uint32_t pos;
		uint32_t posj;
		for (i=0; i<s; i++) {
			pos = i * r;
			max_tmp = cell[pos];
			for (j=1; j<r; j++)
			{
				posj = pos +j;
				if(cell[posj]>max_tmp)
					max_tmp = cell[posj];
			}
			result[i]=max_tmp;
    }

  ENDC++;
	//calculates the max between two arrays elemenwise
	SET OF REAL8 arr_max (PBblas.Types.dimension_t s, PBblas.Types.matrix_t C, PBblas.Types.matrix_t D) := BEGINC++
	#body
    __lenResult = s * sizeof(double);
    __isAllResult = false;
    double * result = new double[s];
    __result = (void*) result;
    double *celld = (double*) d;
		double *cellc = (double*) c;
		double max_tmp;
    uint32_t i;
		uint32_t j;
    uint32_t pos;
		uint32_t posj;
		for (i=0; i<s; i++) {
			if (celld[i]>cellc[i])
			{
				result[i]= celld[i];
			}
			else
			{
				result[i]= cellc[i];
			}
    }

  ENDC++;
	Layout_Part max_tran(Layout_Part le) := TRANSFORM
    SELF.mat_part := max_col(le.part_rows, le.part_cols, le.mat_part);
		SELF.partition_id := 1;
    SELF := le;
  END;
	M_max_col_ := PROJECT ( tx, max_tran (LEFT), LOCAL);
	PBblas.Types.Layout_Part maxtran (PBblas.Types.Layout_Part le, PBblas.Types.Layout_Part ri) := TRANSFORM
		part_id := 1;
		SELF.partition_id := part_id;
		SELF.node_id      := 0;
		SELF.block_row    := 1;
		SELF.block_col    := 1;
		SELF.first_row    := 1;
		SELF.part_rows    := 1;
		SELF.first_col    := 1;
		SELF.part_cols    := le.part_cols;
		SELF.mat_part :=  arr_max (le.part_cols, le.mat_part, ri.mat_part);
		SELF := le;
	END;
	M_max_col := ROLLUP (M_max_col_, LEFT.partition_id = RIGHT.partition_id, maxtran(LEFT,RIGHT));
	//M = bsxfun(@minus, M, max(M, [], 1));
	//M=exp(M);
	//result = exp(M - repmat (V, r, 1))
	SET OF REAL8 exp_mat_vec_minus(PBblas.Types.dimension_t N, PBblas.Types.dimension_t r, PBblas.Types.matrix_t M, PBblas.Types.matrix_t V) := BEGINC++

    #body
    __lenResult = n * sizeof(double);
    __isAllResult = false;
    double * result = new double[n];
    __result = (void*) result;
    double *cellm = (double*) m;
		double *cellv = (double*) v;
    uint32_t cells =  n;
    uint32_t i;
		uint32_t pos;
    for (i=0; i<cells; i++) {
		  pos = i / r;
      result[i] = exp(cellm[i] - cellv[pos]);
    }
  ENDC++;
	PBblas.Types.Layout_Part expmax_tran (PBblas.Types.Layout_Part le, PBblas.Types.Layout_Part ri) := TRANSFORM
		SELF.mat_part := exp_mat_vec_minus(le.part_cols * le.part_rows, le.part_rows, le.mat_part, ri.mat_part) ;
		SELF := le;
	END;
	tx_max_exp := JOIN (tx, M_max_col, TRUE, expmax_tran(LEFT,RIGHT), ALL);
	// M = bsxfun(@rdivide, M, sum(M));
	//returns the summation of elements in each coumn
	SET OF REAL8 sum_col (PBblas.Types.dimension_t r, PBblas.Types.dimension_t s, PBblas.Types.matrix_t D) := BEGINC++

    #body
    __lenResult = s * sizeof(double);
    __isAllResult = false;
    double * result = new double[s];
    __result = (void*) result;
    double *cell = (double*) d;
    uint32_t cells =  r * s;
    uint32_t i,j;
    uint32_t pos;
		double sum_tmp;
	 for (i=0; i<s; i++) {
		sum_tmp = 0;
		pos = i * r;
		for (j=0; j<r; j++)
		{
				sum_tmp = sum_tmp + cell[pos+j];
		}
		result[i]=sum_tmp;
    }

  ENDC++;

	SET OF REAL8 mat_vec_div(PBblas.Types.dimension_t N, PBblas.Types.dimension_t r, PBblas.Types.matrix_t M, PBblas.Types.matrix_t V) := BEGINC++

    #body
    __lenResult = n * sizeof(double);
    __isAllResult = false;
    double * result = new double[n];
    __result = (void*) result;
    double *cellm = (double*) m;
		double *cellv = (double*) v;
    uint32_t cells =  n;
    uint32_t i;
		uint32_t pos;
    for (i=0; i<cells; i++) {
		  pos = i / r;
      result[i] = cellm[i] / cellv[pos];

    }
  ENDC++;
	
	Layout_Part sum_tran(Layout_Part le) := TRANSFORM
    SELF.mat_part := sum_col(le.part_rows, le.part_cols, le.mat_part);
		SELF.partition_id := 1;
    SELF := le;
  END;
	tx_max_exp_sum_col_ := PROJECT (tx_max_exp, sum_tran (LEFT), LOCAL);
	//summation of two arrays element-wise
	SET OF REAL8 arr_sum (PBblas.Types.dimension_t s, PBblas.Types.matrix_t C, PBblas.Types.matrix_t D) := BEGINC++
	#body
    __lenResult = s * sizeof(double);
    __isAllResult = false;
    double * result = new double[s];
    __result = (void*) result;
    double *celld = (double*) d;
		double *cellc = (double*) c;
		double max_tmp;
    uint32_t i;
		uint32_t j;
    uint32_t pos;
		uint32_t posj;
		for (i=0; i<s; i++) {
			result[i] = celld[i] + cellc[i];
    }

  ENDC++;
	PBblas.Types.Layout_Part sumtran (PBblas.Types.Layout_Part le, PBblas.Types.Layout_Part ri) := TRANSFORM
		part_id := 1;
		SELF.partition_id := part_id;
		SELF.node_id      := 0;
		SELF.block_row    := 1;
		SELF.block_col    := 1;
		SELF.first_row    := 1;
		SELF.part_rows    := 1;
		SELF.first_col    := 1;
		SELF.part_cols    := le.part_cols;
		SELF.mat_part :=  arr_sum (le.part_cols, le.mat_part, ri.mat_part);
		SELF := le;
	END;
	tx_max_exp_sum_col := ROLLUP (tx_max_exp_sum_col_, LEFT.partition_id = RIGHT.partition_id, sumtran(LEFT,RIGHT));
	PBblas.Types.Layout_Part divsumtran (PBblas.Types.Layout_Part le, PBblas.Types.Layout_Part ri) := TRANSFORM
		SELF.mat_part := mat_vec_div(le.part_cols * le.part_rows, le.part_rows, le.mat_part, ri.mat_part) ;
		SELF := le;
	END;
	tx_soft := JOIN (tx_max_exp, tx_max_exp_sum_col, TRUE, divsumtran(LEFT,RIGHT), ALL);//same as M in MATLAB


	//groundTruth - tx_soft : (groundTruth-M)
	groundTruth := TrainLabel;
	grnd_tx := Pbblas.PB_daxpy(-1, tx_soft, groundTruth);
	//-1/m*(groundTruth-M)*x'
	SET OF REAL8 part_sparse_tran_mul(PBblas.types.dimension_t r, PBblas.types.dimension_t s, PBblas.types.dimension_t samp, PBblas.types.dimension_t num, PBblas.types.matrix_t M, DATASET(PBblas.types.Layout_Cell) D) := BEGINC++
    typedef struct work2 {      // copy of numericfield translated to C
      uint32_t y;
      uint32_t x;
      double v;
    };
    #body
    __lenResult = r * samp * sizeof(double);
    __isAllResult = false;
    double *result = new double[r * samp];
    __result = (void*) result;
    work2 *celld = (work2*) d;
		double *cellm = (double*) m;
		uint32_t cells = num;
    uint32_t i, j;
    uint32_t pos;
    for (i=0; i< r * samp; i++) {
      result[i] =  0.0;
    }
		uint32_t x, y;
		for (i=0; i < cells; i++){
			x = celld[i].x - 1;   // input co-ordinates are one based,
      y = celld[i].y - 1;  //x and y are zero based.
			for (j=0; j<r; j++){
				pos = y * r + j;
				result[pos] = result[pos] + cellm[r * x + j] * celld[i].v;
			}		
		}
  ENDC++; // END part_sparse_tran_mul
	PBblas.Types.Layout_Part gradtran (PBblas.Types.Layout_Part le, DATASET(PBblas.Types.Layout_Cell) cells) := TRANSFORM 
		part_id := le.partition_id;;
		SELF.partition_id := part_id;
		SELF.node_id      := map_c.assigned_node(part_id);
		SELF.block_row    := le.block_row;
		SELF.block_col    := le.block_col;
		SELF.first_row    := theta_map.first_row(part_id);
		SELF.part_rows    := theta_map.part_rows(part_id);
		SELF.first_col    := theta_map.first_col(part_id);
		SELF.part_cols    := theta_map.part_cols(part_id);
		SELF.mat_part := part_sparse_tran_mul(le.part_rows, le.part_cols, Numfeat, COUNT (cells), le.mat_part, PROJECT(cells, TRANSFORM (PBblas.Types.layout_cell, SELF:= LEFT)));
		SELF := le;
	END;
	
	grnd_tx_xt := DENORMALIZE(grnd_tx, TrainData,
                            LEFT.node_id = RIGHT.node_id,
                            GROUP,
	                         gradtran(LEFT,ROWS(RIGHT)), LOCAL);
	grndtx_xt_m :=  PBblas.PB_dscal(m_, grnd_tx_xt);
	//calculate theta_grad
	Layout_Part theta_grad_tran(Layout_Part le, Layout_Part ri) := TRANSFORM
		N := le.part_rows * le.part_cols ;
		SELF.mat_part := PBblas.BLAS.daxpy(N, LAMBDA, le.mat_part, 1, ri.mat_part, 1);
		SELF := le;
	END;
	theta_grad := JOIN (theta, grndtx_xt_m,LEFT.partition_id = RIGHT.partition_id, theta_grad_tran(LEFT,RIGHT), FULL OUTER, LOCAL);
	//calculate cost
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
	simple := {Pbblas.types.value_t v};
	simple wd_tran (Layout_Part le) := TRANSFORM
		SELF.v := sum_sq(le.part_cols * le.part_rows, le.mat_part);
	END;
	weightdecay_term_ := PROJECT (theta, wd_tran (LEFT), LOCAL);
	weightdecay_term := SUM (weightdecay_term_, weightdecay_term_.v);
	// tmp=log(M).*groundTruth;
  // log_cost=sum(tmp(:));
	REAL8 log_cost_c (PBblas.Types.dimension_t N, PBblas.Types.matrix_t M, PBblas.Types.matrix_t D) := BEGINC++

    #body
    double result = 0;
		double tmpp ;
    double *cellm = (double*) m;
		double *celld = (double*) d;
    uint32_t i;
		for (i=0; i<n; i++) {
      result = result + (log(cellm[i])*celld[i]);
    }
		return(result);
  ENDC++;
	simple cost_log_tran (Layout_Part le, Layout_Part ri) := TRANSFORM
		SELF.v := log_cost_c(le.part_cols * le.part_rows, le.mat_part, ri.mat_part);
	END;
	log_cost_ := JOIN (tx_soft, groundTruth, LEFT.partition_id = RIGHT.partition_id, cost_log_tran(LEFT,RIGHT), LOCAL);
	log_cost := SUM (log_cost_, log_cost_.v);
	//cost=((-1/m)*log_cost)+((lambda/2)*weightdecay_term);
	cost := m_*log_cost + ((LAMBDA/2)*weightdecay_term);
	//return results
	return_record := RECORD (Layout_Part)
		REAL8 cost_value;
	END;
	theta_cost := PROJECT (theta_grad, TRANSFORM (return_record, SELF.cost_value := cost; SELF:=LEFT), LOCAL);
	theta_cost_check :=  ASSERT(theta_cost, node_id=Thorlib.node() and node_id=theta_map.assigned_node (partition_id), 'softmax gradient is not distributed correctly', FAIL);
	ToReturn := theta_cost_check;
//soft function should change to work with the new paritiones
	RETURN   ToReturn;
	

  END; //END SoftMax_compatible_lbfgs_sparse_label

// the implementation of SoftMax_compatible_lbfgs_sparse_label_sparse where there is not groundtruth matrix and labels are provided on each node as a single partition. for exampel if the label vector is [{1,1,1},{2,1,2},{3,1,3},{4,1,4}] (we have overal 4 instances) then in each node we will have a PBblas partition of which the mat_part inclues [1,2,3,4]
EXPORT  SoftMax_compatible_lbfgs_sparse_label_sparse( DATASET(Layout_Part) theta, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Cell_nid)  TrainData , DATASET(Layout_Part) TrainLabel) := function
  PBblas.types.dimension_t Numfeat:= CostFunc_params(id=2)[1].value;
  PBblas.types.dimension_t NumClass := CostFunc_params(id=3)[1].value;// number of features
  PBblas.types.dimension_t m := CostFunc_params(id=1)[1].value;// number of samples
	m_ := -1.0/(CostFunc_params(id=1)[1].value);
  PBblas.types.dimension_t part_rows := CostFunc_params(id=4)[1].value;// this is used to partition the rows in theta matrix
  PBblas.types.dimension_t part_cols := CostFunc_params(id=5)[1].value;// this is not used in this version of the softmax implementation
  LAMBDA:= CostFunc_params(id=6)[1].value;
	//maps used
	SET OF PBblas.Types.value_t empty_array := [];
	theta_map := PBblas.Matrix_Map(NumClass, Numfeat, part_rows, Numfeat);
	map_c := PBblas.Matrix_Map(NumClass, m, part_rows, m);
	//calculated theta*TrainData
	//M=(theta*x);
	//part_sparse_mul multiplies matrix M in Pbblas format of size r by s to the matrix D which is in numericfield format and its size is s by samp and the size od matrix D is num. Natrix D is a sparse matrix so its size is not necessarily s*samp
	SET OF REAL8 part_sparse_mul(PBblas.types.dimension_t r, PBblas.types.dimension_t s, PBblas.types.dimension_t samp, PBblas.types.dimension_t num, PBblas.types.matrix_t M, DATASET(PBblas.types.Layout_Cell) D) := BEGINC++
    typedef struct work3 {      // copy of numericfield translated to C
      uint32_t x;
      uint32_t y;
      double v;
    };
    #body
    __lenResult = r * samp * sizeof(double);
    __isAllResult = false;
    double *result = new double[r * samp];
    __result = (void*) result;
    work3 *celld = (work3*) d;
		double *cellm = (double*) m;
		uint32_t cells = num;
    uint32_t i, j;
    uint32_t pos;
    for (i=0; i< r * samp; i++) {
      result[i] =  0.0;
    }
		uint32_t x, y;
		for (i=0; i < cells; i++){
			x = celld[i].x - 1;   // input co-ordinates are one based,
      y = celld[i].y - 1;  //x and y are zero based.
			for (j=0; j<r; j++){
				pos = y * r + j;
				result[pos] = result[pos] + cellm[r * x + j] * celld[i].v;
			}		
		}
  ENDC++;
	PBblas.Types.Layout_Part txtran (PBblas.Types.Layout_Part le, DATASET(PBblas.Types.Layout_Cell) cells) := TRANSFORM
		part_id := le.partition_id;;
		SELF.partition_id := part_id;
		SELF.node_id      := map_c.assigned_node(part_id);
		SELF.block_row    := le.block_row;
		SELF.block_col    := le.block_col;
		SELF.first_row    := map_c.first_row(part_id);
		SELF.part_rows    := map_c.part_rows(part_id);
		SELF.first_col    := map_c.first_col(part_id);
		SELF.part_cols    := map_c.part_cols(part_id);
		SELF.mat_part := part_sparse_mul(le.part_rows, le.part_cols, m, COUNT (cells), le.mat_part, PROJECT(cells, TRANSFORM (PBblas.Types.layout_cell, SELF:= LEFT)));
		SELF := le;
	END;
	
	tx := DENORMALIZE(theta, TrainData,
                            LEFT.node_id = RIGHT.node_id,
                            GROUP,
                            txtran(LEFT,ROWS(RIGHT)), LOCAL);
	// M=tx;
	// M = bsxfun(@minus, M, max(M, [], 1));
	// first calculate  max(M, [], 1)
	SET OF REAL8 max_col (PBblas.Types.dimension_t r, PBblas.Types.dimension_t s, PBblas.Types.matrix_t D) := BEGINC++

    #body
    __lenResult = s * sizeof(double);
    __isAllResult = false;
    double * result = new double[s];
    __result = (void*) result;
    double *cell = (double*) d;
		double max_tmp;
    uint32_t i;
		uint32_t j;
    uint32_t pos;
		uint32_t posj;
		for (i=0; i<s; i++) {
			pos = i * r;
			max_tmp = cell[pos];
			for (j=1; j<r; j++)
			{
				posj = pos +j;
				if(cell[posj]>max_tmp)
					max_tmp = cell[posj];
			}
			result[i]=max_tmp;
    }

  ENDC++;
	//calculates the max between two arrays elemenwise
	SET OF REAL8 arr_max (PBblas.Types.dimension_t s, PBblas.Types.matrix_t C, PBblas.Types.matrix_t D) := BEGINC++
	#body
    __lenResult = s * sizeof(double);
    __isAllResult = false;
    double * result = new double[s];
    __result = (void*) result;
    double *celld = (double*) d;
		double *cellc = (double*) c;
		// double max_tmp;
    uint32_t i;
		// uint32_t j;
    // uint32_t pos;
		// uint32_t posj;
		for (i=0; i<s; i++) {
			if (celld[i]>cellc[i])
			{
				result[i]= celld[i];
			}
			else
			{
				result[i]= cellc[i];
			}
    }

  ENDC++;
	Layout_Part max_tran(Layout_Part le) := TRANSFORM
    SELF.mat_part := max_col(le.part_rows, le.part_cols, le.mat_part);
		SELF.partition_id := 1;
    SELF := le;
  END;
	M_max_col_ := PROJECT ( tx, max_tran (LEFT), LOCAL);
	PBblas.Types.Layout_Part maxtran (PBblas.Types.Layout_Part le, PBblas.Types.Layout_Part ri) := TRANSFORM
		part_id := 1;
		SELF.partition_id := part_id;
		SELF.node_id      := 0;
		SELF.block_row    := 1;
		SELF.block_col    := 1;
		SELF.first_row    := 1;
		SELF.part_rows    := 1;
		SELF.first_col    := 1;
		SELF.part_cols    := le.part_cols;
		SELF.mat_part :=  arr_max (le.part_cols, le.mat_part, ri.mat_part);
		SELF := le;
	END;
	M_max_col := ROLLUP (M_max_col_, LEFT.partition_id = RIGHT.partition_id, maxtran(LEFT,RIGHT));
	//M = bsxfun(@minus, M, max(M, [], 1));
	//M=exp(M);
	//result = exp(M - repmat (V, r, 1))
	SET OF REAL8 exp_mat_vec_minus(PBblas.Types.dimension_t N, PBblas.Types.dimension_t r, PBblas.Types.matrix_t M, PBblas.Types.matrix_t V) := BEGINC++

    #body
    __lenResult = n * sizeof(double);
    __isAllResult = false;
    double * result = new double[n];
    __result = (void*) result;
    double *cellm = (double*) m;
		double *cellv = (double*) v;
    uint32_t cells =  n;
    uint32_t i;
		uint32_t pos;
    for (i=0; i<cells; i++) {
		  pos = i / r;
      result[i] = exp(cellm[i] - cellv[pos]);
    }
  ENDC++;
	PBblas.Types.Layout_Part expmax_tran (PBblas.Types.Layout_Part le, PBblas.Types.Layout_Part ri) := TRANSFORM
		SELF.mat_part := exp_mat_vec_minus(le.part_cols * le.part_rows, le.part_rows, le.mat_part, ri.mat_part) ;
		SELF := le;
	END;
	tx_max_exp := JOIN (tx, M_max_col, TRUE, expmax_tran(LEFT,RIGHT), ALL);
	// M = bsxfun(@rdivide, M, sum(M));
	//returns the summation of elements in each coumn
	SET OF REAL8 sum_col (PBblas.Types.dimension_t r, PBblas.Types.dimension_t s, PBblas.Types.matrix_t D) := BEGINC++

    #body
    __lenResult = s * sizeof(double);
    __isAllResult = false;
    double * result = new double[s];
    __result = (void*) result;
    double *cell = (double*) d;
    // uint32_t cells =  r * s;
    uint32_t i,j;
    uint32_t pos;
		double sum_tmp;
	 for (i=0; i<s; i++) {
		sum_tmp = 0;
		pos = i * r;
		for (j=0; j<r; j++)
		{
				sum_tmp = sum_tmp + cell[pos+j];
		}
		result[i]=sum_tmp;
    }

  ENDC++;

	SET OF REAL8 mat_vec_div(PBblas.Types.dimension_t N, PBblas.Types.dimension_t r, PBblas.Types.matrix_t M, PBblas.Types.matrix_t V) := BEGINC++

    #body
    __lenResult = n * sizeof(double);
    __isAllResult = false;
    double * result = new double[n];
    __result = (void*) result;
    double *cellm = (double*) m;
		double *cellv = (double*) v;
    uint32_t cells =  n;
    uint32_t i;
		uint32_t pos;
    for (i=0; i<cells; i++) {
		  pos = i / r;
      result[i] = cellm[i] / cellv[pos];

    }
  ENDC++;
	
	Layout_Part sum_tran(Layout_Part le) := TRANSFORM
    SELF.mat_part := sum_col(le.part_rows, le.part_cols, le.mat_part);
		SELF.partition_id := 1;
    SELF := le;
  END;
	tx_max_exp_sum_col_ := PROJECT (tx_max_exp, sum_tran (LEFT), LOCAL);
	//summation of two arrays element-wise
	SET OF REAL8 arr_sum (PBblas.Types.dimension_t s, PBblas.Types.matrix_t C, PBblas.Types.matrix_t D) := BEGINC++
	#body
    __lenResult = s * sizeof(double);
    __isAllResult = false;
    double * result = new double[s];
    __result = (void*) result;
    double *celld = (double*) d;
		double *cellc = (double*) c;
		// double max_tmp;
    uint32_t i;
		// uint32_t j;
    // uint32_t pos;
		// uint32_t posj;
		for (i=0; i<s; i++) {
			result[i] = celld[i] + cellc[i];
    }

  ENDC++;
	PBblas.Types.Layout_Part sumtran (PBblas.Types.Layout_Part le, PBblas.Types.Layout_Part ri) := TRANSFORM
		part_id := 1;
		SELF.partition_id := part_id;
		SELF.node_id      := 0;
		SELF.block_row    := 1;
		SELF.block_col    := 1;
		SELF.first_row    := 1;
		SELF.part_rows    := 1;
		SELF.first_col    := 1;
		SELF.part_cols    := le.part_cols;
		SELF.mat_part :=  arr_sum (le.part_cols, le.mat_part, ri.mat_part);
		SELF := le;
	END;
	tx_max_exp_sum_col := ROLLUP (tx_max_exp_sum_col_, LEFT.partition_id = RIGHT.partition_id, sumtran(LEFT,RIGHT));
	PBblas.Types.Layout_Part divsumtran (PBblas.Types.Layout_Part le, PBblas.Types.Layout_Part ri) := TRANSFORM
		SELF.mat_part := mat_vec_div(le.part_cols * le.part_rows, le.part_rows, le.mat_part, ri.mat_part) ;
		SELF := le;
	END;
	tx_soft := JOIN (tx_max_exp, tx_max_exp_sum_col, TRUE, divsumtran(LEFT,RIGHT), ALL);//same as M in MATLAB


	//groundTruth - tx_soft : (groundTruth-M)
SET OF Pbblas.Types.value_t m_label_minus(PBblas.types.dimension_t r, PBblas.types.dimension_t s, PBblas.types.dimension_t f, PBblas.types.matrix_t M, PBblas.types.matrix_t D) := BEGINC++

    #body
    __lenResult = r * s * sizeof(double);
    __isAllResult = false;
    double * result = new double[r*s];
    __result = (void*) result;
		double *cellm = (double*) m;
    double *celld = (double*) d;
		uint32_t tmp;
    uint32_t i;
		// uint32_t j;
    uint32_t pos;
		// uint32_t f_ = f-1;
		uint32_t posj = r+f;
		for (i=0; i<s*r; i++) {
			result[i] = -1.0 * cellm[i];
    }
		for (i=0; i<s; i++) {
			tmp = (uint32_t)celld[i];
			if (tmp >= f && tmp <posj) {
				pos = (i*r) + tmp - f;
				result [pos] = result [pos] + 1;
			}
		}

  ENDC++;
	
	grnd_tx := JOIN (tx_soft, TrainLabel, LEFT.node_id=RIGHT.node_id, TRANSFORM (Layout_PART, SELF.mat_part:= m_label_minus(LEFT.part_rows, LEFT.part_cols, LEFT.first_row, LEFT.mat_part, RIGHT.mat_part)  ; SELF:=LEFT), LOCAL);

	//-1/m*(groundTruth-M)*x'
	SET OF REAL8 part_sparse_tran_mul(PBblas.types.dimension_t r, PBblas.types.dimension_t s, PBblas.types.dimension_t samp, PBblas.types.dimension_t num, PBblas.types.matrix_t M, DATASET(PBblas.types.Layout_Cell) D) := BEGINC++
    typedef struct work2 {      // copy of numericfield translated to C
      uint32_t y;
      uint32_t x;
      double v;
    };
    #body
    __lenResult = r * samp * sizeof(double);
    __isAllResult = false;
    double *result = new double[r * samp];
    __result = (void*) result;
    work2 *celld = (work2*) d;
		double *cellm = (double*) m;
		uint32_t cells = num;
    uint32_t i, j;
    uint32_t pos;
    for (i=0; i< r * samp; i++) {
      result[i] =  0.0;
    }
		uint32_t x, y;
		for (i=0; i < cells; i++){
			x = celld[i].x - 1;   // input co-ordinates are one based,
      y = celld[i].y - 1;  //x and y are zero based.
			for (j=0; j<r; j++){
				pos = y * r + j;
				result[pos] = result[pos] + cellm[r * x + j] * celld[i].v;
			}		
		}
  ENDC++; // END part_sparse_tran_mul
	PBblas.Types.Layout_Part gradtran (PBblas.Types.Layout_Part le, DATASET(PBblas.Types.Layout_Cell) cells) := TRANSFORM 
		part_id := le.partition_id;
		SELF.partition_id := part_id;
		SELF.node_id      := map_c.assigned_node(part_id);
		SELF.block_row    := le.block_row;
		SELF.block_col    := le.block_col;
		SELF.first_row    := theta_map.first_row(part_id);
		SELF.part_rows    := theta_map.part_rows(part_id);
		SELF.first_col    := theta_map.first_col(part_id);
		SELF.part_cols    := theta_map.part_cols(part_id);
		SELF.mat_part := part_sparse_tran_mul(le.part_rows, le.part_cols, Numfeat, COUNT (cells), le.mat_part, PROJECT(cells, TRANSFORM (PBblas.Types.layout_cell, SELF:= LEFT)));
		SELF := le;
	END;
	
	grnd_tx_xt := DENORMALIZE(grnd_tx, TrainData,
                            LEFT.node_id = RIGHT.node_id,
                            GROUP,
	                         gradtran(LEFT,ROWS(RIGHT)), LOCAL);
	grndtx_xt_m :=  PBblas.PB_dscal(m_, grnd_tx_xt);
	//calculate theta_grad
	Layout_Part theta_grad_tran(Layout_Part le, Layout_Part ri) := TRANSFORM
		N := le.part_rows * le.part_cols ;
		SELF.mat_part := PBblas.BLAS.daxpy(N, LAMBDA, le.mat_part, 1, ri.mat_part, 1);
		SELF := le;
	END;
	theta_grad := JOIN (theta, grndtx_xt_m,LEFT.partition_id = RIGHT.partition_id, theta_grad_tran(LEFT,RIGHT), FULL OUTER, LOCAL);
	//calculate cost
	REAL8 sum_sq(PBblas.Types.dimension_t N, PBblas.Types.matrix_t M) := BEGINC++

    #body
    double result = 0;
		// double tmpp ;
    double *cellm = (double*) m;
    uint32_t i;
		for (i=0; i<n; i++) {
      result = result + (cellm[i]*cellm[i]);
    }
		return(result);

  ENDC++;
	simple := {Pbblas.types.value_t v};
	simple wd_tran (Layout_Part le) := TRANSFORM
		SELF.v := sum_sq(le.part_cols * le.part_rows, le.mat_part);
	END;
	weightdecay_term_ := PROJECT (theta, wd_tran (LEFT), LOCAL);
	weightdecay_term := SUM (weightdecay_term_, weightdecay_term_.v);
	// tmp=log(M).*groundTruth;
  // log_cost=sum(tmp(:));
	Pbblas.Types.value_t log_cost_c(PBblas.types.dimension_t r, PBblas.types.dimension_t s, PBblas.types.dimension_t f, PBblas.types.matrix_t M, PBblas.types.matrix_t D) := BEGINC++

    #body
    double result = 0.0;
		double *cellm = (double*) m;
    double *celld = (double*) d;
		uint32_t tmp;
    uint32_t i;
		// uint32_t j;
    uint32_t pos;
		// uint32_t f_ = f-1;
		uint32_t posj = r+f;
		for (i=0; i<s; i++) {
			tmp = (uint32_t)celld[i];
			if (tmp >= f && tmp <posj) {
				pos = (i*r) + tmp - f;
				result  = result + log(cellm [pos]);
			}
		}
		return (result);
  ENDC++;
	simple cost_log_tran (Layout_Part le, Layout_Part ri) := TRANSFORM
		SELF.v := log_cost_c(le.part_rows , le.part_cols, le.first_row, le.mat_part, ri.mat_part);
	END;
	log_cost_ := JOIN (tx_soft, TrainLabel, LEFT.node_id = RIGHT.node_id, cost_log_tran(LEFT,RIGHT), LOCAL);
	log_cost := SUM (log_cost_, log_cost_.v);
	//cost=((-1/m)*log_cost)+((lambda/2)*weightdecay_term);
	cost := m_*log_cost + ((LAMBDA/2)*weightdecay_term);
	//return results
	return_record := RECORD (Layout_Part)
		REAL8 cost_value;
	END;
	theta_cost := PROJECT (theta_grad, TRANSFORM (return_record, SELF.cost_value := cost; SELF:=LEFT), LOCAL);
	theta_cost_check :=  ASSERT(theta_cost, node_id=Thorlib.node() and node_id=theta_map.assigned_node (partition_id) , 'softmax gradient is not distributed correctly', FAIL);
	ToReturn := theta_cost_check;
//soft function should change to work with the new paritiones
	RETURN   ToReturn;
	

  END; //END SoftMax_compatible_lbfgs_sparse_label_sparse



EXPORT  Sphere_compatible_lbfgs( DATASET(Layout_Part) theta, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Cell_nid)  TrainData , DATASET(Layout_Part) TrainLabel) := function
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
	
	SET OF Pbblas.Types.value_t x2_der(PBblas.types.dimension_t N, PBblas.types.matrix_t M) := BEGINC++

    #body
    __lenResult = n * sizeof(double);
    __isAllResult = false;
    double * result = new double[n];
    __result = (void*) result;
		double *cellm = (double*) m;
    uint32_t i;
		for (i=0; i<n; i++) {
			result[i] = 2 * cellm[i];
    }
		
  ENDC++;
	simple := {Pbblas.types.value_t v};
	simple sq_tran (Layout_Part le) := TRANSFORM
		SELF.v := sum_sq(le.part_cols * le.part_rows, le.mat_part);
	END;
	theta_2_ := PROJECT (theta, sq_tran (LEFT), LOCAL);
	theta_2 := SUM (theta_2_, theta_2_.v);
	cost := theta_2;

	Layout_Part grad_tran (Layout_part le) := TRANSFORM
		SELF.mat_part := x2_der (le.part_rows*le.part_cols, le.mat_part);
		SELF := le;
	END;
	theta_grad := PROJECT (theta, grad_tran (LEFT), LOCAL);
	return_record := RECORD (Layout_Part)
		REAL8 cost_value;
	END;
	theta_cost := PROJECT (theta_grad, TRANSFORM (return_record, SELF.cost_value := cost; SELF:=LEFT), LOCAL);
	ToReturn := theta_cost;
	RETURN ToReturn;
END;



//// the implementation of SoftMax_compatible_lbfgs_sparse_partitions_datadist where the last join is done in a loop in order to avoid memory problem
EXPORT  SoftMax_compatible_lbfgs_sparse_partitions_datadist_loop( DATASET(Layout_Part) theta, DATASET(Types.NumericField) CostFunc_params, DATASET(Layout_Part) TrainData , DATASET(Layout_Part) TrainLabel) := function
  Numfeat:= CostFunc_params(id=2)[1].value;
  NumClass := CostFunc_params(id=3)[1].value;// number of features
  m := CostFunc_params(id=1)[1].value;// number of samples
	m_ := -1/m;
  part_rows := CostFunc_params(id=4)[1].value;
  part_cols := CostFunc_params(id=5)[1].value;
  LAMBDA:= CostFunc_params(id=6)[1].value;
	//maps used
	SET OF PBblas.Types.value_t empty_array := [];
	map_c := PBblas.Matrix_Map(NumClass, m, NumClass, part_cols);// tx map (tx is of size numebr of classe by number of samples)
	dmap := PBblas.Matrix_Map(Numfeat, m, part_rows, part_cols);
	theta_map := PBblas.Matrix_Map(NumClass, Numfeat, NumClass, part_rows);
	//distribute theta to all the nodes that train data is on
	nodes_available := Thorlib.nodes();
	data_nodesused := MIN(nodes_available, dmap.row_blocks);
	Layout_Part_newnode := RECORD (Layout_Part)
		PBblas.Types.node_t new_node_id;
	END;
	Layout_Part_newnode norm_theta (Layout_Part te, INTEGER co) := TRANSFORM
		SELF.new_node_id := co % data_nodesused;
		SELF:= te;
	END;
	theta_norm := NORMALIZE(theta, data_nodesused, norm_theta(LEFT, COUNTER) );
	//theta_dist := DISTRIBUTE (theta_norm, new_node_id);//with this dirtsibution the whole theta matrix is available on all nodes
	//calculated theta*TrainData
	//M=(theta*x);
	Layout_Part multx(Layout_Part t_part, Layout_Part x_part):=TRANSFORM
		part_id := x_part.block_col;
		part_c_rows := map_c.part_rows(part_id);
		part_c_cols := map_c.part_cols(part_id);
		k := t_part.part_cols;
		SELF.partition_id := part_id;
		SELF.node_id      := map_c.assigned_node(part_id);
		SELF.block_row    := t_part.block_row;
		SELF.block_col    := x_part.block_col;
		SELF.first_row    := map_c.first_row(part_id);
		SELF.part_rows    := part_c_rows;
		SELF.first_col    := map_c.first_col(part_id);
		SELF.part_cols    := part_c_cols;
		SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
																		part_c_rows, part_c_cols, k,
																		1.0, t_part.mat_part, x_part.mat_part,
																		0.0, empty_array);
  END;
	tx_ := JOIN (theta, TrainData, LEFT.block_col = RIGHT.block_row, multx(LEFT, RIGHT), LOCAL );
	tx_dist := DISTRIBUTE (tx_, node_id);
	// Sum terms
	//In the transform function check whether the left side is NULL, it can be possible that there is only one partition on one node that needs to rollup
  Layout_Part sumTerms(Layout_Part cumm, Layout_Part term) := TRANSFORM
		HaveTerm := IF(term.part_cols=0, FALSE, TRUE);
    N := cumm.part_rows * cumm.part_cols;
    SELF.mat_part := IF (HaveTerm, PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term.mat_part, 1), cumm.mat_part);
    SELF := cumm;
  END;
	tx_sorted := SORT(tx_dist, partition_id, LOCAL);
  tx := ROLLUP(tx_sorted, sumTerms(LEFT, RIGHT), partition_id, LOCAL);
	// define a c++ function who does the following operations on each partition
	// M=tx;
	// M = bsxfun(@minus, M, max(M, [], 1));
	// M=exp(M);
	// M = bsxfun(@rdivide, M, sum(M));
	//r : rows
	//s : cols
	// D : input matrix
	SET OF REAL8 soft_fun (PBblas.Types.dimension_t r, PBblas.Types.dimension_t s, PBblas.Types.matrix_t D) := BEGINC++

    #body
    __lenResult = r * s* sizeof(double);
    __isAllResult = false;
    double * result = new double[r*s];
    __result = (void*) result;
    double *cell = (double*) d;
		double *max_arr = new double[s];
		double *sum_arr = new double[s];
		double max_tmp;
		double sum_tmp;
    uint32_t cells =  r * s;
    uint32_t i;
		uint32_t j;
    uint32_t pos;
		uint32_t posj;
		for (i=0; i<s; i++) {
			pos = i * r;
			max_tmp = cell[pos];
			for (j=1; j<r; j++)
			{
				posj = pos +j;
				if(cell[posj]>max_tmp)
					max_tmp = cell[posj];
			}
			max_arr[i]=max_tmp;
    }
		for (i=0; i<cells; i++) {
			pos = (int) i / r;
			result[i]=exp(cell[i]-max_arr[pos]);
		}
    for (i=0; i<s; i++) {
			sum_tmp = 0;
			pos = i * r;
			for (j=0; j<r; j++)
			{
					sum_tmp = sum_tmp + result[pos+j];
			}
			sum_arr[i]=sum_tmp;
    }
		for (i=0; i<cells; i++) {
			pos = (int) i / r;
			result[i]=result[i] / sum_arr[pos];
		}

  ENDC++; // END soft_fun
	Layout_Part soft_tran(Layout_Part le) := TRANSFORM
    SELF.mat_part := soft_fun(le.part_rows, le.part_cols, le.mat_part);
    SELF := le;
  END;
	tx_soft := PROJECT (tx, soft_tran (LEFT), LOCAL);//same as M in MATLAB
	//groundTruth - tx_soft : (groundTruth-M)
	groundTruth := TrainLabel;
	Layout_Part grnd_tran(Layout_Part le, Layout_Part ri) := TRANSFORM
		N := le.part_rows * le.part_cols;
    SELF.mat_part := PBblas.BLAS.daxpy(N, -1.0, ri.mat_part, 1, le.mat_part, 1);
    SELF := le;
  END;
	// grnd_tx := JOIN (groundTruth, tx_soft, LEFT.partition_id = RIGHT.partition_id, grnd_tran(LEFT, RIGHT),FULL OUTER, LOCAL );
	grnd_tx := Pbblas.PB_daxpy(-1, tx_soft, groundTruth);

	Layout_Part_newnode norm_grndtx (Layout_Part te, INTEGER co) := TRANSFORM
		SELF.new_node_id := co % data_nodesused;
		SELF:= te;
	END;
	grnd_tx_norm := NORMALIZE(grnd_tx, data_nodesused, norm_grndtx(LEFT, COUNTER) );// this normalization cause all the partitions to be repeated the same number of nodes, It actually means that in each node we should be able to fit a data os aproximatly size labels*samples
//in the memory. The large lshtc dataset needs aproximatly 9 gig for that which is not avaiable
	grnd_tx_dist := DISTRIBUTE (grnd_tx_norm, new_node_id);//with this dirtsibution the whole theta matrix is available on all nodes. not possible for big datasets

	//grnd_tx * x' : (groundTruth-M)*x')
	Layout_Part grndtx_xt_mul(Layout_Part_newnode g_part, Layout_Part x_part):=TRANSFORM
		part_id     := x_part.block_row;
		part_g_cols := g_part.part_cols;
		part_g_rows := g_part.part_rows;
		part_x_rows := x_part.part_rows;
		part_x_cols := x_part.part_cols;
		k := part_g_cols;
		SELF.partition_id := part_id;
		SELF.node_id      := theta_map.assigned_node(part_id);
		SELF.block_row    := g_part.block_row;
		SELF.block_col    := x_part.block_row;
		SELF.first_row    := theta_map.first_row(part_id);
		SELF.part_rows    := theta_map.part_rows(part_id);
		SELF.first_col    := theta_map.first_col(part_id);
		SELF.part_cols    := theta_map.part_cols(part_id);
		SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, TRUE,
																		part_g_rows, part_x_rows, k,
																		m_, g_part.mat_part, x_part.mat_part,
																		0.0, empty_array);
	END;
	//-1/m*(groundTruth-M)*x')
	grndtx_xt := JOIN (grnd_tx_dist, TrainData, LEFT.block_col = RIGHT.block_col, grndtx_xt_mul(LEFT, RIGHT), LOCAL );
	
	// use loop to calculate 	//-1/m*(groundTruth-M)*x')
	Layout_Part addup(Layout_Part le, Layout_Part ri) := TRANSFORM
	  HaveRight := IF(ri.part_cols=0, FALSE, TRUE);
		N := le.part_rows * le.part_cols ;
		SELF.mat_part := IF (HaveRight, PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1), le.mat_part);
		SELF := le;
	END;
	grndtx_xt_dist := grndtx_xt;// already distributed based on node_id
	grndtx_xt_dist_sorted := SORT (grndtx_xt_dist, partition_id, LOCAL);
	grndtx_xt_m_2 := ROLLUP(grndtx_xt_dist_sorted, LEFT.partition_id = RIGHT.partition_id, addup(LEFT, RIGHT), LOCAL);
	


	//-1/m*(groundTruth-M)*x')
	//the loop continutes for data_nodesused times. In each iteration each partition of grnd_tx is distributed on a different node
	// grnd_tx_dist is calculated and rollup is done. By end of the loop grnd_tx_dist is calculated
	//multiplication loop
	multi_loop (DATASET (Layout_Part) res_in, UNSIGNED coun) := FUNCTION
		//first distributed grnd_tx on different nodes compraed to the preceding iteration using coun
		Layout_Part_newnode norm_grndtx_coun (Layout_Part te) := TRANSFORM
			SELF.new_node_id :=  (te.node_id + coun-1 )% data_nodesused ;
			SELF:= te;
		END;
		grnd_tx_norm_coun := PROJECT (grnd_tx, norm_grndtx_coun(LEFT), LOCAL);
		grnd_tx_norm_coun_dist := DISTRIBUTE (grnd_tx_norm_coun, new_node_id);
		//grnd_tx * x' : (groundTruth-M)*x')
	Layout_Part mul_tran(Layout_Part_newnode g_part, Layout_Part x_part):=TRANSFORM
		part_id     := x_part.block_row;
		part_g_cols := g_part.part_cols;
		part_g_rows := g_part.part_rows;
		part_x_rows := x_part.part_rows;
		part_x_cols := x_part.part_cols;
		k := part_g_cols;
		SELF.partition_id := part_id;
		SELF.node_id      := theta_map.assigned_node(part_id);
		SELF.block_row    := g_part.block_row;
		SELF.block_col    := x_part.block_row;
		SELF.first_row    := theta_map.first_row(part_id);
		SELF.part_rows    := theta_map.part_rows(part_id);
		SELF.first_col    := theta_map.first_col(part_id);
		SELF.part_cols    := theta_map.part_cols(part_id);
		SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, TRUE,
																		part_g_rows, part_x_rows, k,
																		m_, g_part.mat_part, x_part.mat_part,
																		0.0, empty_array);
	END;
	//-1/m*(groundTruth-M)*x')
	grndtx_xt_ := JOIN (grnd_tx_norm_coun_dist, TrainData, LEFT.block_col = RIGHT.block_col, mul_tran(LEFT, RIGHT), LOCAL );
	Layout_Part addup_tran(Layout_Part le, Layout_Part ri) := TRANSFORM
	  HaveRight := IF(ri.part_cols=0, FALSE, TRUE);
		N := le.part_rows * le.part_cols ;
		SELF.mat_part := IF (HaveRight, PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1), le.mat_part);
		SELF := le;
	END;
	grndtx_xt_sorted := SORT (res_in + grndtx_xt_, partition_id, LOCAL);
	res_out := ROLLUP(grndtx_xt_sorted, LEFT.partition_id = RIGHT.partition_id, addup_tran(LEFT, RIGHT), LOCAL);
		RETURN res_out;
	END;// END multi_loop

grndtx_xt_m := LOOP (DATASET ([], Layout_Part), data_nodesused, multi_loop (ROWS(LEFT), COUNTER) );
	//calculate theta_grad
	Layout_Part theta_grad_tran(Layout_Part le, Layout_Part ri) := TRANSFORM
		N := le.part_rows * le.part_cols ;
		SELF.mat_part := PBblas.BLAS.daxpy(N, LAMBDA, le.mat_part, 1, ri.mat_part, 1);
		SELF := le;
	END;
	theta_grad := JOIN (theta, grndtx_xt_m,LEFT.partition_id = RIGHT.partition_id, theta_grad_tran(LEFT,RIGHT), FULL OUTER, LOCAL);
	//calculate cost
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
	simple := {Pbblas.types.value_t v};
	simple wd_tran (Layout_Part le) := TRANSFORM
		SELF.v := sum_sq(le.part_cols * le.part_rows, le.mat_part);
	END;
	weightdecay_term_ := PROJECT (theta, wd_tran (LEFT), LOCAL);
	weightdecay_term := SUM (weightdecay_term_, weightdecay_term_.v);
	// tmp=log(M).*groundTruth;
  // log_cost=sum(tmp(:));
	REAL8 log_cost_c (PBblas.Types.dimension_t N, PBblas.Types.matrix_t M, PBblas.Types.matrix_t D) := BEGINC++

    #body
    double result = 0;
		double tmpp ;
    double *cellm = (double*) m;
		double *celld = (double*) d;
    uint32_t i;
		for (i=0; i<n; i++) {
      result = result + (log(cellm[i])*celld[i]);
    }
		return(result);
  ENDC++;
	simple cost_log_tran (Layout_Part le, Layout_Part ri) := TRANSFORM
		SELF.v := log_cost_c(le.part_cols * le.part_rows, le.mat_part, ri.mat_part);
	END;
	log_cost_ := JOIN (tx_soft, groundTruth, LEFT.partition_id = RIGHT.partition_id, cost_log_tran(LEFT,RIGHT), LOCAL);
	log_cost := SUM (log_cost_, log_cost_.v);
	//cost=((-1/m)*log_cost)+((lambda/2)*weightdecay_term);
	cost := m_*log_cost + ((LAMBDA/2)*weightdecay_term);
	//return results
	return_record := RECORD (Layout_Part)
		REAL8 cost_value;
	END;
	theta_cost := PROJECT (theta_grad, TRANSFORM (return_record, SELF.cost_value := cost; SELF:=LEFT), LOCAL);
	theta_cost_check :=  ASSERT(theta_cost, node_id=Thorlib.node() and node_id=theta_map.assigned_node (partition_id), 'softmax gradient is not distributed correctly', FAIL);
	ToReturn := theta_cost_check;

  RETURN  PROJECT (grnd_tx, TRANSFORM (return_record, SELF.cost_value := 10; SELF:=LEFT), LOCAL);
	// RETURN  ToReturn; 
	

  END; //END SoftMax_compatible_lbfgs_sparse_partitions_datadist_loop

// if the input dataset is in a sparse form (the numericfield fromat does not cover all id number values assigned to each id = most number get values 0 for a specific id (sample))
// still we need to make sure there is not any feature for it all the values are zero. such feature should be existing in the dataset because it does not have any value for any sample
// when the is distributed in the SoftMax_compatible_lbfgs_sparse function, the assumption is that data is already distributed on all the nodes from node 0 to node = max (data.column_part)
// if there are some columns in the data which are all zero , then there might be some nodes that do not have any data and that can casue problem
// this module works for both sparse and full representation of the input data in numericfield format 
EXPORT softmax_lbfgs (INTEGER4 NumberofFeatures, INTEGER4 NumberofClasses, UNSIGNED4 prows=0, UNSIGNED4 pcols=0) := MODULE


SHARED SM(DATASET(Types.NumericField) X, DATASET(Types.NumericField) Y,DATASET(Mat.Types.Element) Inttheta, REAL8 LAMBDA=0.001, UNSIGNED2 MaxIter=100, UNSIGNED LBFGS_corrections = 100) := MODULE
	
	m := MAX (X,X.id);// number of samples
	f := NumberofFeatures; // number of features
	//Create block matrix d
	Xtran := PROJECT (X,TRANSFORM ( ML.Types.NumericField, SELF.id := LEFT.number; SELF.number := LEFT.id; SELF:=LEFT),LOCAL);//through the analysis rows represent features and columns represent samples
	
	dmap := PBblas.Matrix_Map(f,m,prows,pcols);// we use this map only to partition the data, however the distribution is done in a fashion where each big column partition of the data which includes smaller row partitions ends up in one node
	// ddist_tmp := DMAT.Converted.FromElement(Xtran,dmap); orig
	ddist_tmp := DMAT.Converted.FromNumericFieldDS(Xtran,dmap);
	layout_part new_node_id (layout_part dd) := TRANSFORM
		new_node_id := dmap.assigned_node(dd.block_col);
		SELF.node_id := new_node_id;
		SELF := dd;
	END;
	ddist_ := PROJECT (ddist_tmp, new_node_id (LEFT), LOCAL );
	ddist := DISTRIBUTE(ddist_, node_id);
	// groundTruth := Utils.ToGroundTruth (Y); orig//Instead of working with label matrix we work with groundTruth matrix 
	groundTruth := Utils.LabelToGroundTruth (Y);
  //groundTruth is a Numclass*NumSamples matrix. groundTruth(i,j)=1 if label of the jth sample is i, otherwise groundTruth(i,j)=0
	ymap := PBblas.Matrix_Map(NumberofClasses,m,NumberofClasses,pcols);
	// ydist := DMAT.Converted.FromElement(groundTruth,ymap); orig
	ydist := DMAT.Converted.FromNumericFieldDS(groundTruth,ymap);
	// partition theta
	thetamap := PBblas.Matrix_Map(NumberofClasses,f,NumberofClasses,prows);
	theta_dist := DMAT.Converted.FromElement(Inttheta,thetamap);
	// parameters for softmax
	SM_param := DATASET([
    {1,1,m},
    {2,1,NumberofFeatures},
    {3,1,NumberofClasses},
    {4,1,prows},
    {5,1,pcols},
    {6,1,LAMBDA}
    ], Types.NumericField);
		paramnumber := NumberofFeatures * NumberofClasses;
	lbfgs_result := Optimization_new_new_2_2 (0, 0, 0, 0).MinFUNC(theta_dist,SM_param,ddist,ydist,SoftMax_compatible_lbfgs_sparse, paramnumber,MaxIter, 0.00001, 0.000000001,  1000, LBFGS_corrections, 0, 0, 0,0);
	gg := SoftMax_compatible_lbfgs_sparse( theta_dist, SM_param, ddist , ydist);
	arm_t_rec := RECORD
			real8 init_arm_t;
			Layout_part.partition_id;
		END;
		
		arminitt := DATASET ([{0.01,1},{0.01,2}],arm_t_rec);
		wolfe_result := Optimization_new_new_2_2 (0, 0, 0, 0).WolfeLineSearch4_4_2(1, theta_dist, SM_param, ddist , ydist, SoftMax_compatible_lbfgs_sparse , 1.0, theta_dist, gg, 10,0.0001, 0.9, 25, 0.000000001);
	armijo_result := Optimization_new_new_2_2 (0, 0, 0, 0).ArmijoBacktrack_fromwolfe(theta_dist, SM_param, ddist , ydist,SoftMax_compatible_lbfgs_sparse, DISTRIBUTE(arminitt, partition_id-1), theta_dist, gg, 10, 0.0001, 0.9, 0.000000001);
		// lbfgs_result := Optimization_new_new_2_2 (0, 0, 0, 0).MinFUNC(theta_dist,SM_param,ddist,ydist,SoftMax_compatible_lbfgs, paramnumber,MaxIter, 0.00001, 0.000000001,  1000, LBFGS_corrections, 0, 0, 0,0);
		
	// lbfgs_rec := RECORD (Layout_Part)
		// lbfgs_result.cost_value;
		// lbfgs_result.min_funEval;
	// END;
	// EXPORT mod := PROJECT (lbfgs_result (no=2), TRANSFORM(lbfgs_rec , SELF := LEFT), LOCAL);
	// EXPORT mod := theta_dist;
	EXPORT mod := lbfgs_result;
	// EXPORt mod := wolfe_result;
	// EXPORt mod :=  ddist;
		// EXPORT mod := SoftMax_compatible_lbfgs_sparse_test( theta_dist, SM_param, ddist , ydist);
		// EXPORT mod := ddist_tmp;

	END; // END SM

	EXPORT LearnC (DATASET(Types.NumericField) X, DATASET(Types.NumericField) Y,DATASET(Mat.Types.Element) Inttheta, REAL8 LAMBDA=0.001, UNSIGNED2 MaxIter=100, UNSIGNED LBFGS_corrections = 100)  := SM( X,  Y, Inttheta, LAMBDA,  MaxIter,  LBFGS_corrections ) .mod;
	
SHARED lbfgs_rec2 := RECORD (Layout_part)
			REAL8 cost_value;
      REAL8 h ;//hdiag value
      INTEGER8 min_funEval;
      INTEGER break_cond ;
			REAL8 sty;
			PBblas.Types.t_mu_no no;
			INTEGER8 update_itr ; //This value is increased whenever a update is done and s and y vectors are added to the corrections. If no update is done due to the condition ys > 1e-10 then this value is not increased
			// we use this value to update the corrections vectors as well as in the lbfgs algorithm
			INTEGER8 itr_counter;
    END;
		
		SHARED lbfgs_rec := Optimization_new_new_2_2 (0, 0, 0, 0).minfRec;
	// SHARED lbfgs_rec := RECORD (Layout_Part)
		// REAL8 cost_value;
		// INTEGER8 min_funEval;
	// END; // END lbfgs_rec
	EXPORT extractcost_funeval(DATASET(lbfgs_rec) mod) := FUNCTION
	 my_rec := RECORD 
		REAL8 cost_value;
		INTEGER8 min_funEval;
	 END;
	 RETURN PROJECT (mod, TRANSFORM(my_rec , SELF := LEFT), LOCAL);
	END;// END extractcost_funeval
 EXPORT Model(DATASET(lbfgs_rec) mod) := FUNCTION
	optTHETA_part := PROJECT (mod(no=2), TRANSFORM(Layout_Part , SELF := LEFT), LOCAL);
	theta_matrix :=  DMat.Converted.FromPart2Elm (optTHETA_part);
  RETURN theta_matrix;
 END;//END Model
 EXPORT ClassProbDistribC(DATASET(Types.NumericField) Indep,DATASET(lbfgs_rec) mod) :=FUNCTION
	//softmax theta parameters
	mod_theta := PROJECT (mod(no=2), TRANSFORM (Layout_Part, SELF:=LEFT), LOCAL);
	//convert data to layout_part
	dt := Types.ToMatrix (Indep);
	dTmp := dt;
	//d := dt; //in the entire of the calculations we work with the d matrix that each sample is presented in one column
	d := Mat.Trans(dTmp);
	dstats := Mat.Has(d).Stats;
	d_n := dstats.XMax;
	d_m := dstats.YMax;
	m := d_m;// number of samples
	f := d_n; // number of features
	//Create block matrix d
	dmap := PBblas.Matrix_Map(f,m,prows,pcols);// we use this map only to partition the data, however the distribution is done in a fashion where each big column partition of the data which includes smaller row partitions ends up in one node
	ddist_tmp := DMAT.Converted.FromElement(d,dmap);
	layout_part new_node_id (layout_part dd) := TRANSFORM
		new_node_id := dmap.assigned_node(dd.block_col);
		SELF.node_id := new_node_id;
		SELF := dd;
	END;
	ddist_ := PROJECT (ddist_tmp, new_node_id (LEFT), LOCAL );
	ddist := DISTRIBUTE(ddist_, node_id);
	

	Numfeat:= d_n;
  NumClass := NumberofClasses;// number of features
	m_ := -1/m;
  part_rows := prows;
  part_cols := pcols;

	//maps used
	SET OF PBblas.Types.value_t empty_array := [];
	map_c := PBblas.Matrix_Map(NumClass, m, NumClass, part_cols);
	theta_map := PBblas.Matrix_Map(NumClass, Numfeat, NumClass, part_rows);
	//distribute theta to all the nodes that train data is on
	data_nodesused := dmap.col_blocks;
	Layout_Part_newnode := RECORD (Layout_Part)
		PBblas.Types.node_t new_node_id;
	END;
	Layout_Part_newnode norm_theta (Layout_Part te, INTEGER co) := TRANSFORM
		SELF.new_node_id := co % data_nodesused  ;
		SELF:= te;
	END;
	theta_norm := NORMALIZE(mod_theta, data_nodesused, norm_theta(LEFT, COUNTER) );
	theta_dist := DISTRIBUTE (theta_norm, new_node_id);
	//calculated theta*TrainData
	//M=(theta*x);
	Layout_Part multx(Layout_Part_newnode t_part, Layout_Part x_part):=TRANSFORM
		part_id := x_part.block_col;
		part_c_rows := map_c.part_rows(part_id);
		part_c_cols := map_c.part_cols(part_id);
		k := t_part.part_cols;
		SELF.partition_id := part_id;
		SELF.node_id      := x_part.node_id;
		SELF.block_row    := t_part.block_row;
		SELF.block_col    := x_part.block_col;
		SELF.first_row    := map_c.first_row(part_id);
		SELF.part_rows    := part_c_rows;
		SELF.first_col    := map_c.first_col(part_id);
		SELF.part_cols    := part_c_cols;
		SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
																		part_c_rows, part_c_cols, k,
																		1.0, t_part.mat_part, x_part.mat_part,
																		0.0, empty_array);
  END;
	tx_ := JOIN (theta_dist, ddist, LEFT.block_col = RIGHT.block_row, multx(LEFT, RIGHT), LOCAL );
	// Sum terms
  Layout_Part sumTerms(Layout_Part cumm, Layout_Part term) := TRANSFORM
    N := cumm.part_rows * cumm.part_cols;
    SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term.mat_part, 1);
    SELF := cumm;
  END;
	tx_sorted := SORT(tx_, partition_id, LOCAL);
  tx := ROLLUP(tx_sorted, sumTerms(LEFT, RIGHT), partition_id, LOCAL);
	// define a c++ function who does the following operations on each partition
	// M=tx;
	// M = bsxfun(@minus, M, max(M, [], 1));
	// M=exp(M);
	// M = bsxfun(@rdivide, M, sum(M));
	//r : rows
	//s : cols
	// D : input matrix
	SET OF REAL8 soft_fun (PBblas.Types.dimension_t r, PBblas.Types.dimension_t s, PBblas.Types.matrix_t D) := BEGINC++

    #body
    __lenResult = r * s* sizeof(double);
    __isAllResult = false;
    double * result = new double[r*s];
    __result = (void*) result;
    double *cell = (double*) d;
		double *max_arr = new double[s];
		double *sum_arr = new double[s];
		double max_tmp;
		double sum_tmp;
    uint32_t cells =  r * s;
    uint32_t i;
		uint32_t j;
    uint32_t pos;
		uint32_t posj;
		for (i=0; i<s; i++) {
			pos = i * r;
			max_tmp = cell[pos];
			for (j=1; j<r; j++)
			{
				posj = pos +j;
				if(cell[posj]>max_tmp)
					max_tmp = cell[posj];
			}
			max_arr[i]=max_tmp;
    }
		for (i=0; i<cells; i++) {
			pos = (int) i / r;
			result[i]=exp(cell[i]-max_arr[pos]);
		}
    for (i=0; i<s; i++) {
			sum_tmp = 0;
			pos = i * r;
			for (j=0; j<r; j++)
			{
					sum_tmp = sum_tmp + result[pos+j];
			}
			sum_arr[i]=sum_tmp;
    }
		for (i=0; i<cells; i++) {
			pos = (int) i / r;
			result[i]=result[i] / sum_arr[pos];
		}

  ENDC++;
	Layout_Part soft_tran(Layout_Part le) := TRANSFORM
    SELF.mat_part := soft_fun(le.part_rows, le.part_cols, le.mat_part);
    SELF := le;
  END;
	tx_soft := PROJECT (tx, soft_tran (LEFT), LOCAL);//same as M in MATLAB
	Prob_mat := DMAT.Converted.FromPart2Elm (tx_soft);
	Types.l_result tr(Mat.Types.Element le) := TRANSFORM
		SELF.value := le.x;
		SELF.id := le.y;
		SELF.number := 1;
		SELF.conf := le.value;
  END;
	RETURN PROJECT (Prob_mat, tr(LEFT));
 END;//ClassProbDistribC
 EXPORT ClassifyC(DATASET(Types.NumericField) Indep, DATASET(lbfgs_rec) mod) := FUNCTION
    Dist := ClassProbDistribC(Indep, mod);
    numrow := MAX (Dist,Dist.value);
    S:= SORT(Dist,id,conf);
    SeqRec := RECORD
			l_result;
			INTEGER8 Sequence := 0;
    END;
    //add seq field to S
    SeqRec AddS (S l, INTEGER c) := TRANSFORM
			SELF.Sequence := c%numrow;
			SELF := l;
    END;
    Sseq := PROJECT(S, AddS(LEFT,COUNTER));
    classified := Sseq (Sseq.Sequence=0);
    RETURN PROJECT(classified,l_result);
  END; // END ClassifyC Function
END;//softmax_lbfgs


EXPORT softmax_lbfgs_partitions (INTEGER4 NumberofFeatures, INTEGER4 NumberofClasses, UNSIGNED4 prows=0, UNSIGNED4 pcols=0) := MODULE


SHARED SM(DATASET(Types.NumericField) X, DATASET(Types.NumericField) Y,DATASET(Mat.Types.Element) Inttheta, REAL8 LAMBDA=0.001, UNSIGNED2 MaxIter=100, UNSIGNED LBFGS_corrections = 100) := MODULE
	
	m := MAX (X,X.id);// number of samples
	f := NumberofFeatures; // number of features
	//Create block matrix d
	Xtran := PROJECT (X,TRANSFORM ( ML.Types.NumericField, SELF.id := LEFT.number; SELF.number := LEFT.id; SELF:=LEFT),LOCAL);//through the analysis rows represent features and columns represent samples
	
	dmap := PBblas.Matrix_Map(f,m,prows,pcols);// we use this map only to partition the data, however the distribution is done in a fashion where each big column partition of the data which includes smaller row partitions ends up in one node
	// DMAT.Converted.FromNumericFieldDS distribute the data based on partition id, I want the data to be distributed based on column id so I use the coder from DMAT.Converted.FromNumericFieldDS with a little twist here
	insert_columns:=0;
	insert_value:=0.0d;
	Layout_Cell cvt_2_cell(ML.Types.NumericField lr) := TRANSFORM
		SELF.x := lr.id;     // 1 based
		SELF.y := lr.number; // 1 based
		SELF.v := lr.value;
  END;
  Xtran_cell := PROJECT(Xtran, cvt_2_cell(LEFT));
	Work1 := RECORD(Layout_Cell)
    PBblas.Types.partition_t     partition_id;
    PBblas.Types.node_t          node_id;
    PBblas.Types.dimension_t     block_row;
    PBblas.Types.dimension_t     block_col;
  END;
  FromCells(PBblas.IMatrix_Map mat_map, DATASET(Layout_Cell) cells,
                   PBblas.Types.dimension_t insert_columns=0,
                   PBblas.Types.value_t insert_value=0.0d) := FUNCTION
    Work1 cvt_2_xcell(Layout_Cell lr) := TRANSFORM
      block_row           := mat_map.row_block(lr.x);
      block_col           := mat_map.col_block(lr.y + insert_columns);
      partition_id        := mat_map.assigned_part(block_row, block_col);
      SELF.partition_id   := partition_id;
      SELF.node_id        := mat_map.assigned_node(block_col);// instead of using partition id in order to distribute the data, block column number is used 
      SELF.block_row      := block_row;
      SELF.block_col      := block_col;
      SELF := lr;
    END;
    inMatrix := cells.x BETWEEN 1 AND mat_map.matrix_rows
            AND cells.y BETWEEN 1 AND mat_map.matrix_cols - insert_columns;
    d0 := PROJECT(cells(inMatrix), cvt_2_xcell(LEFT));
    d1 := DISTRIBUTE(d0, node_id);
    d2 := SORT(d1, partition_id, y, x, LOCAL);    // prep for column major
    d3 := GROUP(d2, partition_id, LOCAL);
    Layout_Part roll_cells(Work1 parent, DATASET(Work1) cells) := TRANSFORM
      first_row     := mat_map.first_row(parent.partition_id);
      first_col     := mat_map.first_col(parent.partition_id);
      part_rows     := mat_map.part_rows(parent.partition_id);
      part_cols     := mat_map.part_cols(parent.partition_id);
      SELF.mat_part := PBblas.MakeR8Set(part_rows, part_cols, first_row, first_col,
                                        PROJECT(cells, Layout_Cell),
                                        insert_columns, insert_value);
      SELF.partition_id:= parent.partition_id;
      SELF.node_id     := parent.node_id;
      SELF.block_row   := parent.block_row;
      SELF.block_col   := parent.block_col;
      SELF.first_row   := first_row;
      SELF.part_rows   := part_rows;
      SELF.first_col   := first_col;
      SELF.part_cols   := part_cols;
      SELF := [];
    END;
    rslt := ROLLUP(d3, GROUP, roll_cells(LEFT, ROWS(LEFT)));
    RETURN rslt;
  END;
	ddist := FromCells(dmap, Xtran_cell, insert_columns, insert_value);
	// groundTruth := Utils.ToGroundTruth (Y); orig//Instead of working with label matrix we work with groundTruth matrix 
	groundTruth := Utils.LabelToGroundTruth (Y);
  //groundTruth is a Numclass*NumSamples matrix. groundTruth(i,j)=1 if label of the jth sample is i, otherwise groundTruth(i,j)=0
	ymap := PBblas.Matrix_Map(NumberofClasses,m,NumberofClasses,pcols);
	// ydist := DMAT.Converted.FromElement(groundTruth,ymap); orig
	ydist := DMAT.Converted.FromNumericFieldDS(groundTruth,ymap);// for groundtruth matrix partiion id equals the block)col so DMAT.Converted.FromNumericFieldDS distributes the data based on block_col which is what we want 
	// partition theta
	thetamap := PBblas.Matrix_Map(NumberofClasses,f,NumberofClasses,prows);
	theta_dist := DMAT.Converted.FromElement(Inttheta,thetamap);
	// parameters for softmax
	SM_param := DATASET([
    {1,1,m},
    {2,1,NumberofFeatures},
    {3,1,NumberofClasses},
    {4,1,prows},
    {5,1,pcols},
    {6,1,LAMBDA}
    ], Types.NumericField);
	paramnumber := NumberofFeatures * NumberofClasses;
	lbfgs_result := Optimization_new_new_2_2 (0, 0, 0, 0).MinFUNC(theta_dist,SM_param,ddist,ydist,SoftMax_compatible_lbfgs_sparse_partitions, paramnumber,MaxIter, 0.00001, 0.000000001,  1000, LBFGS_corrections, 0, 0, 0,0);
	gg := SoftMax_compatible_lbfgs_sparse( theta_dist, SM_param, ddist , ydist);
	arm_t_rec := RECORD
			real8 init_arm_t;
			Layout_part.partition_id;
		END;
		
		arminitt := DATASET ([{0.01,1},{0.01,2}],arm_t_rec);
		wolfe_result := Optimization_new_new_2_2 (0, 0, 0, 0).WolfeLineSearch4_4_2(1, theta_dist, SM_param, ddist , ydist, SoftMax_compatible_lbfgs_sparse , 1.0, theta_dist, gg, 10,0.0001, 0.9, 25, 0.000000001);
	armijo_result := Optimization_new_new_2_2 (0, 0, 0, 0).ArmijoBacktrack_fromwolfe(theta_dist, SM_param, ddist , ydist,SoftMax_compatible_lbfgs_sparse, DISTRIBUTE(arminitt, partition_id-1), theta_dist, gg, 10, 0.0001, 0.9, 0.000000001);
		// lbfgs_result := Optimization_new_new_2_2 (0, 0, 0, 0).MinFUNC(theta_dist,SM_param,ddist,ydist,SoftMax_compatible_lbfgs, paramnumber,MaxIter, 0.00001, 0.000000001,  1000, LBFGS_corrections, 0, 0, 0,0);
		
	// lbfgs_rec := RECORD (Layout_Part)
		// lbfgs_result.cost_value;
		// lbfgs_result.min_funEval;
	// END;
	// EXPORT mod := PROJECT (lbfgs_result (no=2), TRANSFORM(lbfgs_rec , SELF := LEFT), LOCAL);
	// EXPORT mod := theta_dist;
	EXPORT mod := lbfgs_result; //orig
	// EXPORT mod := ddist;
	// EXPORt mod := wolfe_result;
	// EXPORt mod :=  ddist;
		// EXPORT mod := SoftMax_compatible_lbfgs_sparse_partitions( theta_dist, SM_param, ddist , ydist);
		// EXPORT mod := ddist;

	END; // END SM

	EXPORT LearnC (DATASET(Types.NumericField) X, DATASET(Types.NumericField) Y,DATASET(Mat.Types.Element) Inttheta, REAL8 LAMBDA=0.001, UNSIGNED2 MaxIter=100, UNSIGNED LBFGS_corrections = 100)  := SM( X,  Y, Inttheta, LAMBDA,  MaxIter,  LBFGS_corrections ) .mod;
	
SHARED lbfgs_rec2 := RECORD (Layout_part)
			REAL8 cost_value;
      REAL8 h ;//hdiag value
      INTEGER8 min_funEval;
      INTEGER break_cond ;
			REAL8 sty;
			PBblas.Types.t_mu_no no;
			INTEGER8 update_itr ; //This value is increased whenever a update is done and s and y vectors are added to the corrections. If no update is done due to the condition ys > 1e-10 then this value is not increased
			// we use this value to update the corrections vectors as well as in the lbfgs algorithm
			INTEGER8 itr_counter;
    END;
		
		SHARED lbfgs_rec := Optimization_new_new_2_2 (0, 0, 0, 0).minfRec;
	// SHARED lbfgs_rec := RECORD (Layout_Part)
		// REAL8 cost_value;
		// INTEGER8 min_funEval;
	// END; // END lbfgs_rec
	EXPORT extractcost_funeval(DATASET(lbfgs_rec) mod) := FUNCTION
	 my_rec := RECORD 
		REAL8 cost_value;
		INTEGER8 min_funEval;
	 END;
	 RETURN PROJECT (mod, TRANSFORM(my_rec , SELF := LEFT), LOCAL);
	END;// END extractcost_funeval
 EXPORT Model(DATASET(lbfgs_rec) mod) := FUNCTION
	optTHETA_part := PROJECT (mod(no=2), TRANSFORM(Layout_Part , SELF := LEFT), LOCAL);
	theta_matrix :=  DMat.Converted.FromPart2Elm (optTHETA_part);
  RETURN theta_matrix;
 END;//END Model
 EXPORT ClassProbDistribC(DATASET(Types.NumericField) Indep,DATASET(lbfgs_rec) mod) :=FUNCTION
	//softmax theta parameters
	mod_theta := PROJECT (mod(no=2), TRANSFORM (Layout_Part, SELF:=LEFT), LOCAL);
	//convert data to layout_part
	m := MAX (Indep,Indep.id);// number of samples
	f := NumberofFeatures; // number of features
	//Create block matrix d
	Xtran := PROJECT (Indep,TRANSFORM ( ML.Types.NumericField, SELF.id := LEFT.number; SELF.number := LEFT.id; SELF:=LEFT),LOCAL);//through the analysis rows represent features and columns represent samples
	
	dmap := PBblas.Matrix_Map(f,m,prows,pcols);// we use this map only to partition the data, however the distribution is done in a fashion where each big column partition of the data which includes smaller row partitions ends up in one node
	// DMAT.Converted.FromNumericFieldDS distribute the data based on partition id, I want the data to be distributed based on column id so I use the coder from DMAT.Converted.FromNumericFieldDS with a little twist here
	insert_columns:=0;
	insert_value:=0.0d;
	Layout_Cell cvt_2_cell(ML.Types.NumericField lr) := TRANSFORM
		SELF.x := lr.id;     // 1 based
		SELF.y := lr.number; // 1 based
		SELF.v := lr.value;
  END;
  Xtran_cell := PROJECT(Xtran, cvt_2_cell(LEFT));
	Work1 := RECORD(Layout_Cell)
    PBblas.Types.partition_t     partition_id;
    PBblas.Types.node_t          node_id;
    PBblas.Types.dimension_t     block_row;
    PBblas.Types.dimension_t     block_col;
  END;
  FromCells(PBblas.IMatrix_Map mat_map, DATASET(Layout_Cell) cells,
                   PBblas.Types.dimension_t insert_columns=0,
                   PBblas.Types.value_t insert_value=0.0d) := FUNCTION
    Work1 cvt_2_xcell(Layout_Cell lr) := TRANSFORM
      block_row           := mat_map.row_block(lr.x);
      block_col           := mat_map.col_block(lr.y + insert_columns);
      partition_id        := mat_map.assigned_part(block_row, block_col);
      SELF.partition_id   := partition_id;
      SELF.node_id        := mat_map.assigned_node(block_col);// instead of using partition id in order to distribute the data, block column number is used 
      SELF.block_row      := block_row;
      SELF.block_col      := block_col;
      SELF := lr;
    END;
    inMatrix := cells.x BETWEEN 1 AND mat_map.matrix_rows
            AND cells.y BETWEEN 1 AND mat_map.matrix_cols - insert_columns;
    d0 := PROJECT(cells(inMatrix), cvt_2_xcell(LEFT));
    d1 := DISTRIBUTE(d0, node_id);
    d2 := SORT(d1, partition_id, y, x, LOCAL);    // prep for column major
    d3 := GROUP(d2, partition_id, LOCAL);
    Layout_Part roll_cells(Work1 parent, DATASET(Work1) cells) := TRANSFORM
      first_row     := mat_map.first_row(parent.partition_id);
      first_col     := mat_map.first_col(parent.partition_id);
      part_rows     := mat_map.part_rows(parent.partition_id);
      part_cols     := mat_map.part_cols(parent.partition_id);
      SELF.mat_part := PBblas.MakeR8Set(part_rows, part_cols, first_row, first_col,
                                        PROJECT(cells, Layout_Cell),
                                        insert_columns, insert_value);
      SELF.partition_id:= parent.partition_id;
      SELF.node_id     := parent.node_id;
      SELF.block_row   := parent.block_row;
      SELF.block_col   := parent.block_col;
      SELF.first_row   := first_row;
      SELF.part_rows   := part_rows;
      SELF.first_col   := first_col;
      SELF.part_cols   := part_cols;
      SELF := [];
    END;
    rslt := ROLLUP(d3, GROUP, roll_cells(LEFT, ROWS(LEFT)));
    RETURN rslt;
  END;
	ddist := FromCells(dmap, Xtran_cell, insert_columns, insert_value);
	

	Numfeat:= NumberofFeatures;
  NumClass := NumberofClasses;// number of features
	m_ := -1/m;
  part_rows := prows;
  part_cols := pcols;

	//maps used
	SET OF PBblas.Types.value_t empty_array := [];
	map_c := PBblas.Matrix_Map(NumClass, m, NumClass, part_cols);
	theta_map := PBblas.Matrix_Map(NumClass, Numfeat, NumClass, part_rows);
	//distribute theta to all the nodes that train data is on
	nodes_available := Thorlib.nodes();
	data_nodesused := MIN(nodes_available, dmap.col_blocks);
	Layout_Part_newnode := RECORD (Layout_Part)
		PBblas.Types.node_t new_node_id;
	END;
	Layout_Part_newnode norm_theta (Layout_Part te, INTEGER co) := TRANSFORM
		SELF.new_node_id := co % data_nodesused  ;
		SELF:= te;
	END;
	theta_norm := NORMALIZE(mod_theta, data_nodesused, norm_theta(LEFT, COUNTER) );
	theta_dist := DISTRIBUTE (theta_norm, new_node_id);
	//calculated theta*TrainData
	//M=(theta*x);
	Layout_Part multx(Layout_Part_newnode t_part, Layout_Part x_part):=TRANSFORM
		part_id := x_part.block_col;
		part_c_rows := map_c.part_rows(part_id);
		part_c_cols := map_c.part_cols(part_id);
		k := t_part.part_cols;
		SELF.partition_id := part_id;
		SELF.node_id      := x_part.node_id;
		SELF.block_row    := t_part.block_row;
		SELF.block_col    := x_part.block_col;
		SELF.first_row    := map_c.first_row(part_id);
		SELF.part_rows    := part_c_rows;
		SELF.first_col    := map_c.first_col(part_id);
		SELF.part_cols    := part_c_cols;
		SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
																		part_c_rows, part_c_cols, k,
																		1.0, t_part.mat_part, x_part.mat_part,
																		0.0, empty_array);
  END;
	tx_ := JOIN (theta_dist, ddist, LEFT.block_col = RIGHT.block_row, multx(LEFT, RIGHT), LOCAL );
	// Sum terms
  Layout_Part sumTerms(Layout_Part cumm, Layout_Part term) := TRANSFORM
    N := cumm.part_rows * cumm.part_cols;
    SELF.mat_part := PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term.mat_part, 1);
    SELF := cumm;
  END;
	tx_sorted := SORT(tx_, partition_id, LOCAL);
  tx := ROLLUP(tx_sorted, sumTerms(LEFT, RIGHT), partition_id, LOCAL);
	// define a c++ function who does the following operations on each partition
	// M=tx;
	// M = bsxfun(@minus, M, max(M, [], 1));
	// M=exp(M);
	// M = bsxfun(@rdivide, M, sum(M));
	//r : rows
	//s : cols
	// D : input matrix
	SET OF REAL8 soft_fun (PBblas.Types.dimension_t r, PBblas.Types.dimension_t s, PBblas.Types.matrix_t D) := BEGINC++

    #body
    __lenResult = r * s* sizeof(double);
    __isAllResult = false;
    double * result = new double[r*s];
    __result = (void*) result;
    double *cell = (double*) d;
		double *max_arr = new double[s];
		double *sum_arr = new double[s];
		double max_tmp;
		double sum_tmp;
    uint32_t cells =  r * s;
    uint32_t i;
		uint32_t j;
    uint32_t pos;
		uint32_t posj;
		for (i=0; i<s; i++) {
			pos = i * r;
			max_tmp = cell[pos];
			for (j=1; j<r; j++)
			{
				posj = pos +j;
				if(cell[posj]>max_tmp)
					max_tmp = cell[posj];
			}
			max_arr[i]=max_tmp;
    }
		for (i=0; i<cells; i++) {
			pos = (int) i / r;
			result[i]=exp(cell[i]-max_arr[pos]);
		}
    for (i=0; i<s; i++) {
			sum_tmp = 0;
			pos = i * r;
			for (j=0; j<r; j++)
			{
					sum_tmp = sum_tmp + result[pos+j];
			}
			sum_arr[i]=sum_tmp;
    }
		for (i=0; i<cells; i++) {
			pos = (int) i / r;
			result[i]=result[i] / sum_arr[pos];
		}

  ENDC++;
	Layout_Part soft_tran(Layout_Part le) := TRANSFORM
    SELF.mat_part := soft_fun(le.part_rows, le.part_cols, le.mat_part);
    SELF := le;
  END;
	tx_soft := PROJECT (tx, soft_tran (LEFT), LOCAL);//same as M in MATLAB
	Prob_mat := DMAT.Converted.FromPart2Elm (tx_soft);
	Types.l_result tr(Mat.Types.Element le) := TRANSFORM
		SELF.value := le.x;
		SELF.id := le.y;
		SELF.number := 1;
		SELF.conf := le.value;
  END;
	RETURN PROJECT (Prob_mat, tr(LEFT));
 END;//ClassProbDistribC
 EXPORT ClassifyC(DATASET(Types.NumericField) Indep, DATASET(lbfgs_rec) mod) := FUNCTION
    Dist := ClassProbDistribC(Indep, mod);
    numrow := MAX (Dist,Dist.value);
    S:= SORT(Dist,id,conf);
    SeqRec := RECORD
			l_result;
			INTEGER8 Sequence := 0;
    END;
    //add seq field to S
    SeqRec AddS (S l, INTEGER c) := TRANSFORM
			SELF.Sequence := c%numrow;
			SELF := l;
    END;
    Sseq := PROJECT(S, AddS(LEFT,COUNTER));
    classified := Sseq (Sseq.Sequence=0);
    RETURN PROJECT(classified,l_result);
  END; // END ClassifyC Function
END;//softmax_lbfgs_partitions


/*a version of softmax_lbfgs_partitions where data row blocks are distributed instead of data column blocks.
This way we do not need to distribute the whole theta on all the nodes in order to calculate tx. theta is just distributed based on partition id and data is distributed based on block row
so corresponding block rows of data end up on the same node of the coresponding theta block columns. and tx can be calculated. */
EXPORT softmax_lbfgs_partitions_datadist (INTEGER4 NumberofFeatures, INTEGER4 NumberofClasses, UNSIGNED4 prows=0, UNSIGNED4 pcols=0, BOOLEAN tonorm=FALSE) := MODULE


SHARED SM(DATASET(Types.NumericField) X, DATASET(Types.NumericField) Y,DATASET(Mat.Types.Element) Inttheta, REAL8 LAMBDA=0.001, UNSIGNED2 MaxIter=100, UNSIGNED LBFGS_corrections = 100) := MODULE

	NormalizeOne (DATASET (Layout_Part) in_d) := FUNCTION
	//this function adds up the column value in a matrix of size r by c, the result is a vector of size r
		SET OF REAL8 sum_col (PBblas.Types.dimension_t r, PBblas.Types.dimension_t s, PBblas.Types.matrix_t D) := BEGINC++

			#body
			__lenResult = r * sizeof(double);
			__isAllResult = false;
			double * result = new double[r];
			__result = (void*) result;
			double *cell = (double*) d;
			uint32_t cells =  r * s;
			uint32_t i;
			uint32_t pos;
			for (i=0; i<r; i++) {
				result[i] = 0;
			}
			for (i=0; i<cells; i++) {
				pos = i % r;
				result[pos] = result[pos] + cell[i];
			}	
	 ENDC++;
	 
	 
	 //result = M / repmat (V, 1, c)
	SET OF REAL8 mat_vec_div(PBblas.Types.dimension_t N, PBblas.Types.dimension_t r, PBblas.Types.matrix_t M, PBblas.Types.matrix_t V) := BEGINC++

    #body
    __lenResult = n * sizeof(double);
    __isAllResult = false;
    double * result = new double[n];
    __result = (void*) result;
    double *cellm = (double*) m;
		double *cellv = (double*) v;
    uint32_t cells =  n;
    uint32_t i;
		uint32_t pos;
    for (i=0; i<cells; i++) {
		  pos = i % r;
      result[i] = cellm[i] / cellv[pos];
    }
  ENDC++;
	 
	d_sum_col_ := PROJECT (in_d, TRANSFORM(Layout_Part, SELF.mat_part := sum_col (LEFT.part_rows, LEFT.part_cols, LEFT.mat_part);SELF:= LEFT), LOCAL);
		Layout_Part addparts(Layout_Part le, Layout_Part ri) := TRANSFORM
	  HaveRight := IF(ri.part_cols=0, FALSE, TRUE);
		N := le.part_rows * le.part_cols ;
		SELF.mat_part := IF (HaveRight, PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1), le.mat_part);
		SELF := le;
	END;
	d_sum_col_sorted := SORT (d_sum_col_, block_row, LOCAL);
	d_sum_col := ROLLUP(d_sum_col_sorted, LEFT.block_row = RIGHT.block_row, addparts(LEFT, RIGHT), LOCAL);

	Layout_Part norm_tran(Layout_Part cumm, Layout_Part term) := TRANSFORM
		cumm_part_cols := cumm.part_cols;// number of columns in this partition
		term_part_rows := term.part_rows;//number of elements in this partition of the bias vector
    N := cumm.part_rows * cumm.part_cols;
		Elem := {PBblas.Types.value_t v};
		SELF.mat_part := mat_vec_div(N, term_part_rows, cumm.mat_part, term.mat_part);
    SELF := cumm;
  END;
	 rs := JOIN(in_d, d_sum_col, LEFT.block_row = RIGHT.block_row , norm_tran(LEFT,RIGHT), LOCAL);
		RETURN rs;
	END; //END NormalizeOne
	m := MAX (X,X.id);// number of samples
	f := NumberofFeatures; // number of features
	labelmap := PBblas.Matrix_Map(m,1,m,1);
	//Create block matrix d
	Xtran := PROJECT (X,TRANSFORM ( ML.Types.NumericField, SELF.id := LEFT.number; SELF.number := LEFT.id; SELF:=LEFT),LOCAL);//through the analysis rows represent features and columns represent samples
	dmap := PBblas.Matrix_Map(f,m,prows,pcols);
	dmap_usednodes := MIN (dmap.row_blocks, Thorlib.nodes());
	insert_columns:=0;
	insert_value:=0.0d;
	Layout_Cell cvt_2_cell(ML.Types.NumericField lr) := TRANSFORM
		SELF.x := lr.id;     // 1 based
		SELF.y := lr.number; // 1 based
		SELF.v := lr.value;
  END;
  Xtran_cell := PROJECT(Xtran, cvt_2_cell(LEFT));
	Y_cell := PROJECT(Y, cvt_2_cell(LEFT));
	Work1 := RECORD(Layout_Cell)
    PBblas.Types.partition_t     partition_id;
    PBblas.Types.node_t          node_id;
    PBblas.Types.dimension_t     block_row;
    PBblas.Types.dimension_t     block_col;
  END;
  FromCells(PBblas.IMatrix_Map mat_map, DATASET(Layout_Cell) cells,
                   PBblas.Types.dimension_t insert_columns=0,
                   PBblas.Types.value_t insert_value=0.0d) := FUNCTION
    Work1 cvt_2_xcell(Layout_Cell lr) := TRANSFORM
      block_row           := mat_map.row_block(lr.x);
      block_col           := mat_map.col_block(lr.y + insert_columns);
      partition_id        := mat_map.assigned_part(block_row, block_col);
      SELF.partition_id   := partition_id;
      SELF.node_id        := ((block_row-1) % dmap_usednodes);// instead of using partition id in order to distribute the data, block row number is used to distribute the data
      SELF.block_row      := block_row;
      SELF.block_col      := block_col;
      SELF := lr;
    END;
    inMatrix := cells.x BETWEEN 1 AND mat_map.matrix_rows
            AND cells.y BETWEEN 1 AND mat_map.matrix_cols - insert_columns;
    d0 := PROJECT(cells(inMatrix), cvt_2_xcell(LEFT));
    d1 := DISTRIBUTE(d0, node_id);
    d2 := SORT(d1, partition_id, y, x, LOCAL);    // prep for column major
    d3 := GROUP(d2, partition_id, LOCAL);
    Layout_Part roll_cells(Work1 parent, DATASET(Work1) cells) := TRANSFORM
      first_row     := mat_map.first_row(parent.partition_id);
      first_col     := mat_map.first_col(parent.partition_id);
      part_rows     := mat_map.part_rows(parent.partition_id);
      part_cols     := mat_map.part_cols(parent.partition_id);
      SELF.mat_part := PBblas.MakeR8Set(part_rows, part_cols, first_row, first_col,
                                        PROJECT(cells, Layout_Cell),
                                        insert_columns, insert_value);
      SELF.partition_id:= parent.partition_id;
      SELF.node_id     := parent.node_id;
      SELF.block_row   := parent.block_row;
      SELF.block_col   := parent.block_col;
      SELF.first_row   := first_row;
      SELF.part_rows   := part_rows;
      SELF.first_col   := first_col;
      SELF.part_cols   := part_cols;
      SELF := [];
    END;
    rslt := ROLLUP(d3, GROUP, roll_cells(LEFT, ROWS(LEFT)));
    RETURN rslt;
  END;
	ddist_ := FromCells(dmap, Xtran_cell, insert_columns, insert_value);
	labeldist := FromCells(labelmap, Y_cell, insert_columns, insert_value);
	//this function normalize ddist_ data where each column adds up to one
		NormalizeFeaturesOne (DATASET (Layout_Part) in_d) := FUNCTION
	//this function adds up the row value in a matrix of size r by s, the result is a vector of size s
		SET OF REAL8 sum_row (PBblas.Types.dimension_t r, PBblas.Types.dimension_t s, PBblas.Types.matrix_t D) := BEGINC++

			#body
			__lenResult = s * sizeof(double);
			__isAllResult = false;
			double * result = new double[s];
			__result = (void*) result;
			double *cell = (double*) d;
			uint32_t cells =  r * s;
			uint32_t i;
			uint32_t pos;
			for (i=0; i<s; i++) {
				result[i] = 0;
			}
			for (i=0; i<cells; i++) {
				pos = i / r;
				result[pos] = result[pos] + cell[i];
			}	
	 ENDC++;
	 
	 
	 //result = M / repmat (V, r, 1)
	SET OF REAL8 mat_vec_div(PBblas.Types.dimension_t N, PBblas.Types.dimension_t r, PBblas.Types.matrix_t M, PBblas.Types.matrix_t V) := BEGINC++

    #body
    __lenResult = n * sizeof(double);
    __isAllResult = false;
    double * result = new double[n];
    __result = (void*) result;
    double *cellm = (double*) m;
		double *cellv = (double*) v;
    uint32_t cells =  n;
    uint32_t i;
		uint32_t pos;
    for (i=0; i<cells; i++) {
		  pos = i / r;
      result[i] = cellm[i] / cellv[pos];
    }
  ENDC++;
	 
	d_sum_row_ := PROJECT (in_d, TRANSFORM(Layout_Part, SELF.mat_part := sum_row (LEFT.part_rows, LEFT.part_cols, LEFT.mat_part); SELF:= LEFT), LOCAL);
	Layout_Part addparts(Layout_Part le, Layout_Part ri) := TRANSFORM
	  HaveRight := IF(ri.part_cols=0, FALSE, TRUE);
		N := le.part_cols ;
		SELF.mat_part := IF (HaveRight, PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1), le.mat_part);
		SELF := le;
	END;
	d_sum_row_dist := DISTRIBUTE (d_sum_row_, block_col);
	d_sum_row_sorted := SORT (d_sum_row_dist, block_col, LOCAL);
	d_sum_row := ROLLUP(d_sum_row_sorted, LEFT.block_col = RIGHT.block_col, addparts(LEFT, RIGHT), LOCAL);

Layout_Part norm_d_sum_row (Layout_Part te, INTEGER co) := TRANSFORM
		SELF.node_id := co % dmap_usednodes;
		SELF:= te;
	END;
	d_sum_row_norm := NORMALIZE(d_sum_row, dmap_usednodes, norm_d_sum_row(LEFT, COUNTER) );
	d_sum_row_norm_dist := DISTRIBUTE (d_sum_row_norm, node_id);//with this dirtsibution the whole theta matrix is available on all nodes



	Layout_Part norm_tran(Layout_Part cumm, Layout_Part term) := TRANSFORM
		cumm_part_cols := cumm.part_cols;// number of columns in this partition
		term_part_rows := term.part_rows;//number of elements in this partition of the bias vector
    N := cumm.part_rows * cumm.part_cols;
		Elem := {PBblas.Types.value_t v};
		SELF.mat_part := mat_vec_div(N, cumm.part_rows, cumm.mat_part, term.mat_part);
    SELF := cumm;
  END;
	 rs := JOIN(in_d, d_sum_row_norm_dist, LEFT.block_col = RIGHT.block_col , norm_tran(LEFT,RIGHT), LOCAL);
		RETURN rs;
	END; //END NormalizeFeaturesOne
	ddist := IF (tonorm, NormalizeFeaturesOne(ddist_), ddist_);
	// groundTruth := Utils.ToGroundTruth (Y); orig//Instead of working with label matrix we work with groundTruth matrix 
	groundTruth := Utils.DistinctLabeltoGroundTruth (Y);
	// groundTruth := Utils.LabelToGroundTruth (Y);
  //groundTruth is a Numclass*NumSamples matrix. groundTruth(i,j)=1 if label of the jth sample is i, otherwise groundTruth(i,j)=0
	ymap := PBblas.Matrix_Map(NumberofClasses,m,NumberofClasses,pcols);
	ymap_label := PBblas.Matrix_Map(NumberofClasses,m,prows,m);
	// ydist := DMAT.Converted.FromElement(groundTruth,ymap); orig
	ydist := DMAT.Converted.FromNumericFieldDS(groundTruth,ymap);// for groundtruth matrix partiion id equals the block)col so DMAT.Converted.FromNumericFieldDS distributes the data based on block_col which is what we want 
	ydist_label := DMAT.Converted.FromNumericFieldDS(groundTruth,ymap_label);
	// partition theta
	thetamap := PBblas.Matrix_Map(NumberofClasses,f,NumberofClasses,prows);
	theta_dist := DMAT.Converted.FromElement(Inttheta,thetamap);

	thetamap_label := PBblas.Matrix_Map(NumberofClasses,f,prows,f);
	theta_dist_label := DMAT.Converted.FromElement(Inttheta,thetamap_label);
	//NormalizeFeaturesOne for numerifcField dataset
	NormalizeFeaturesOne_nf (DATASET (Types.NumericField) d_in) := FUNCTION
		d_in_dist := DISTRIBUTE (d_in, d_in.number);
		d_in_dist_sorted := SORT (d_in_dist, d_in_dist.number, LOCAL);
		d_in_dist_sorted_grouped := GROUP (d_in_dist_sorted, d_in_dist_sorted.number);
		sumcol_red := RECORD
			d_in_dist_sorted_grouped.number;
			REAL8 sc := SUM (GROUP, d_in_dist_sorted_grouped.value);//sum col
		END;
		sumcol := TABLE (d_in_dist_sorted_grouped, sumcol_red, number, LOCAL);
		norm_d_in := JOIN (d_in_dist_sorted, sumcol, LEFT.number = RIGHT.number, TRANSFORM (Types.NumericField, SELF.value := LEFT.value/RIGHT.sc ;SELF:= LEFT),LOCAL);
		RETURN norm_d_in;
	END;
	Xtran_normone_ := NormalizeFeaturesOne_nf (Xtran);
	Xtran_normone := IF (tonorm,Xtran_normone_ , Xtran);
	theta_node_used := thetamap_label.nodes_used;
	Layout_Cell_nid x_norm_tran (Types.NumericField le, UNSIGNED coun) := TRANSFORM
		SELF.x := le.id;
		SELF.y := le.number;
		SELF.v := le.value;
		SELF.node_id := coun % theta_node_used;
	END;
	Xtran_normone_norm := NORMALIZE(Xtran_normone, theta_node_used, x_norm_tran(LEFT, COUNTER) );
	Xtran_normone_norm_dist := DISTRIBUTE (Xtran_normone_norm, node_id);

	Layout_Part label_norm_tran (Layout_Part le, UNSIGNED coun) := TRANSFORM
		SELF.node_id := coun%theta_node_used;
		SELF := le;
	END;
	labeldist_norm := NORMALIZE(labeldist, theta_node_used, label_norm_tran(LEFT, COUNTER) );
	labeldist_norm_dist := DISTRIBUTE (labeldist_norm, node_id);
	// parameters for softmax
	SM_param := DATASET([
    {1,1,m},
    {2,1,NumberofFeatures},
    {3,1,NumberofClasses},
    {4,1,prows},
    {5,1,pcols},
    {6,1,LAMBDA}
    ], Types.NumericField);
	paramnumber := NumberofFeatures * NumberofClasses;
	lbfgs_result := Optimization_new_new_2_2 (0, 0, 0, 0).MinFUNC(theta_dist,SM_param,ddist,ydist,SoftMax_compatible_lbfgs_sparse_partitions_datadist, paramnumber,MaxIter, 0.00001, 0.000000001,  1000, LBFGS_corrections, 0, 0, 0,0);
	// lbfgs_result := Optimization_new_new_2_2_test (0, 0, 0, 0).MinFUNC(theta_dist,SM_param,ddist,ydist,SoftMax_compatible_lbfgs_sparse_partitions_datadist, paramnumber,MaxIter, 0.00001, 0.000000001,  1000, LBFGS_corrections, 0, 0, 0,0); 
	
	gg := SoftMax_compatible_lbfgs_sparse( theta_dist, SM_param, ddist , ydist);
	arm_t_rec := RECORD
			real8 init_arm_t;
			Layout_part.partition_id;
		END;
		
		arminitt := DATASET ([{0.01,1},{0.01,2}],arm_t_rec);
		wolfe_result := Optimization_new_new_2_2 (0, 0, 0, 0).WolfeLineSearch4_4_2(1, theta_dist, SM_param, ddist , ydist, SoftMax_compatible_lbfgs_sparse , 1.0, theta_dist, gg, 10,0.0001, 0.9, 25, 0.000000001);
	armijo_result := Optimization_new_new_2_2 (0, 0, 0, 0).ArmijoBacktrack_fromwolfe(theta_dist, SM_param, ddist , ydist,SoftMax_compatible_lbfgs_sparse, DISTRIBUTE(arminitt, partition_id-1), theta_dist, gg, 10, 0.0001, 0.9, 0.000000001);
		// lbfgs_result := Optimization_new_new_2_2 (0, 0, 0, 0).MinFUNC(theta_dist,SM_param,ddist,ydist,SoftMax_compatible_lbfgs, paramnumber,MaxIter, 0.00001, 0.000000001,  1000, LBFGS_corrections, 0, 0, 0,0);
		
	// lbfgs_rec := RECORD (Layout_Part)
		// lbfgs_result.cost_value;
		// lbfgs_result.min_funEval;
	// END;
	// EXPORT mod := PROJECT (lbfgs_result (no=2), TRANSFORM(lbfgs_rec , SELF := LEFT), LOCAL);

SET OF REAL8 sum_row2 (PBblas.Types.dimension_t r, PBblas.Types.dimension_t s, PBblas.Types.matrix_t D) := BEGINC++

			#body
			__lenResult = s * sizeof(double);
			__isAllResult = false;
			double * result = new double[s];
			__result = (void*) result;
			double *cell = (double*) d;
			uint32_t cells =  r * s;
			uint32_t i;
			uint32_t pos;
			for (i=0; i<s; i++) {
				result[i] = 0;
			}
			for (i=0; i<cells; i++) {
				pos = i / r;
				result[pos] = result[pos] + cell[i];
			}	
	 ENDC++;
	 d_sum_row_ := PROJECT (ddist, TRANSFORM(Layout_Part, SELF.mat_part := sum_row2 (LEFT.part_rows, LEFT.part_cols, LEFT.mat_part); SELF:= LEFT), LOCAL);

Layout_Part addparts(Layout_Part le, Layout_Part ri) := TRANSFORM
	  HaveRight := IF(ri.part_cols=0, FALSE, TRUE);
		N := 1 * le.part_cols ;
		SELF.mat_part := IF (HaveRight, PBblas.BLAS.daxpy(N, 1.0, le.mat_part, 1, ri.mat_part, 1), le.mat_part);
		SELF := le;
	END;
	d_sum_row_dist := DISTRIBUTE (d_sum_row_, block_col);
	d_sum_row_sorted := SORT (d_sum_row_dist, block_col, LOCAL);
	d_sum_row := ROLLUP(d_sum_row_sorted, LEFT.block_col = RIGHT.block_col, addparts(LEFT, RIGHT), LOCAL);
	// EXPORT mod := theta_dist;
	// EXPORT mod := lbfgs_result; //orig
	// EXPORT mod := ddist;
	
	// EXPORT mod := d_sum_row;
	// EXPORt mod := wolfe_result;
	// EXPORt mod :=  theta_dist;
	/*
SET OF REAL8 sum_row (PBblas.Types.dimension_t r, PBblas.Types.dimension_t s, PBblas.Types.matrix_t D) := BEGINC++

    #body
    __lenResult = s * sizeof(double);
    __isAllResult = false;
    double * result = new double[s];
    __result = (void*) result;
    double *cell = (double*) d;
    uint32_t cells =  r * s;
    uint32_t i;
    uint32_t pos;
		for (i=0; i<s; i++) {
      result[i] = 0;
    }
    for (i=0; i<cells; i++) {
      pos = i / r;
      result[pos] = result[pos] + cell[i];
    }

  ENDC++; */
	// EXPORT mod := PROJECT (ydist, TRANSFORM (Layout_part, SELF.mat_part := sum_row(LEFT.part_rows, LEFT.part_cols, LEFT.mat_part);SELF:= LEFT),LOCAL);
		// EXPORT mod := SoftMax_compatible_lbfgs_sparse_partitions_datadist( theta_dist, SM_param, ddist , ydist);
		part_theta := ML.Utils.distrow_ranmap_part(NumberofClasses,f,prows , 0.005) ;
		// EXPORT mod := SoftMax_compatible_lbfgs_sparse_label( part_theta, SM_param, Xtran_normone_norm_dist , ydist_label);
		// EXPORT mod := SoftMax_compatible_lbfgs_sparse_partitions_datadist_loop( theta_dist, SM_param, ddist , ydist);
		// EXPORT mod := part_theta;
		EXPORT mod :=  Optimization_new_new_2_2_nf_cost (0, 0, 0, 0).MinFUNC(part_theta,SM_param,Xtran_normone_norm_dist,labeldist_norm_dist,SoftMax_compatible_lbfgs_sparse_label_sparse, paramnumber,MaxIter, 0.00001, 0.000000001,  1000, LBFGS_corrections, 0, 0, 0,0);//orig
		




//start
Xtran_normone_norm_dist2 := DATASET('~maryam::mytest::Xtrannormonenormdist', Layout_Cell_nid, THOR);
part_theta2 := DATASET('~maryam::mytest::parttheta', Layout_Part, THOR);
labeldist_norm_dist2 := DATASET('~maryam::mytest::labeldistnormdist', Layout_Part, THOR);
// EXPORT mod := SoftMax_compatible_lbfgs_sparse_label_sparse( part_theta2, SM_param, Xtran_normone_norm_dist2 , labeldist_norm_dist2);
// EXPORT mod := Optimization_new_new_2_2_nf_cost (0, 0, 0, 0).MinFUNC(part_theta2,SM_param,Xtran_normone_norm_dist2,labeldist_norm_dist2,SoftMax_compatible_lbfgs_sparse_label_sparse, paramnumber,MaxIter, 0.00001, 0.000000001,  1000, LBFGS_corrections, 0, 0, 0,0);//orig
// EXPORT mod  := part_theta2;


// EXPORT mod := SoftMax_compatible_lbfgs_sparse_label_sparse( part_theta, SM_param, Xtran_normone_norm_dist , labeldist_norm_dist);

		// OUTPUT(part_theta,Layout_Part,'~maryam::mytest::parttheta', OVERWRITE);
		// OUTPUT(Xtran_normone_norm_dist,Layout_Cell_nid,'~maryam::mytest::Xtrannormonenormdist', OVERWRITE);
		// OUTPUT(labeldist_norm_dist,Layout_Part,'~maryam::mytest::labeldistnormdist', OVERWRITE);
		// EXPORT mod := 1;
/*
		part_theta := DATASET('~maryam::mytest::parttheta', Layout_Part);
		labeldist_norm_dist := DATASET('~maryam::mytest::labeldistnormdist', Layout_Part);
		Xtran_normone_norm_dist := DATASET('~maryam::mytest::Xtrannormonenormdist', Layout_Cell_nid);
*/

		// EXPORT mod := part_theta;
// EXPORT mod := Xtran_normone_norm_dist;
// EXPORT mod := SoftMax_compatible_lbfgs_sparse_label_sparse( part_theta, SM_param, Xtran_normone_norm_dist , labeldist_norm_dist);

//sphere
//Sphere_compatible_lbfgs
// sphere_theta := ML.utils.distcol_ranmap_part(1,2000000000,40000000 , 0.005) ;
// sphere_param := DATASET ([],Types.NumericField);
// sphere_data := DATASET ([],Layout_cell_nid);
// sphere_label :=  DATASET ([],Layout_Part);
// sphereout := Sphere_compatible_lbfgs (sphere_theta, sphere_param, sphere_data, sphere_label);
// sphereopt := Optimization_new_new_2_2_nf_cost (0, 0, 0, 0).MinFUNC(sphere_theta, sphere_param, sphere_data, sphere_label,Sphere_compatible_lbfgs, paramnumber,MaxIter, 0.00001, 0.000000001,  1000, LBFGS_corrections, 0, 0, 0,0);
// EXPORT mod := sphereopt;
	END; // END SM

	EXPORT LearnC (DATASET(Types.NumericField) X, DATASET(Types.NumericField) Y,DATASET(Mat.Types.Element) Inttheta, REAL8 LAMBDA=0.001, UNSIGNED2 MaxIter=100, UNSIGNED LBFGS_corrections = 100)  := SM( X,  Y, Inttheta, LAMBDA,  MaxIter,  LBFGS_corrections ) .mod;
	
SHARED lbfgs_rec2 := RECORD (Layout_part)
			REAL8 cost_value;
      REAL8 h ;//hdiag value
      INTEGER8 min_funEval;
      INTEGER break_cond ;
			REAL8 sty;
			PBblas.Types.t_mu_no no;
			INTEGER8 update_itr ; //This value is increased whenever a update is done and s and y vectors are added to the corrections. If no update is done due to the condition ys > 1e-10 then this value is not increased
			// we use this value to update the corrections vectors as well as in the lbfgs algorithm
			INTEGER8 itr_counter;
    END;
		
		SHARED lbfgs_rec := Optimization_new_new_2_2 (0, 0, 0, 0).minfRec;
	// SHARED lbfgs_rec := RECORD (Layout_Part)
		// REAL8 cost_value;
		// INTEGER8 min_funEval;
	// END; // END lbfgs_rec
	EXPORT extractcost_funeval(DATASET(lbfgs_rec) mod) := FUNCTION
	 my_rec := RECORD 
		REAL8 cost_value;
		INTEGER8 min_funEval;
	 END;
	 RETURN PROJECT (mod, TRANSFORM(my_rec , SELF := LEFT), LOCAL);
	END;// END extractcost_funeval
 EXPORT Model(DATASET(lbfgs_rec) mod) := FUNCTION
	optTHETA_part := PROJECT (mod(no=2), TRANSFORM(Layout_Part , SELF := LEFT), LOCAL);
	theta_matrix :=  DMat.Converted.FromPart2Elm (optTHETA_part);
  RETURN theta_matrix;
 END;//END Model
 EXPORT ClassProbDistribC(DATASET(Types.NumericField) Indep,DATASET(lbfgs_rec) mod) :=FUNCTION
	//softmax theta parameters
	theta := PROJECT (mod(no=2), TRANSFORM (Layout_Part, SELF:=LEFT), LOCAL);
	//convert data to layout_part
	m := MAX (Indep,Indep.id);// number of samples
	f := NumberofFeatures; // number of features
	//Create block matrix d
	Xtran := PROJECT (Indep,TRANSFORM ( ML.Types.NumericField, SELF.id := LEFT.number; SELF.number := LEFT.id; SELF:=LEFT),LOCAL);//through the analysis rows represent features and columns represent samples
	dmap := PBblas.Matrix_Map(f,m,prows,pcols);// we use this map only to partition the data, however the distribution is done in a fashion where each big column partition of the data which includes smaller row partitions ends up in one node
	dmap_usednodes := MIN (dmap.row_blocks, Thorlib.nodes());
	insert_columns:=0;
	insert_value:=0.0d;
	Layout_Cell cvt_2_cell(ML.Types.NumericField lr) := TRANSFORM
		SELF.x := lr.id;     // 1 based
		SELF.y := lr.number; // 1 based
		SELF.v := lr.value;
  END;
  Xtran_cell := PROJECT(Xtran, cvt_2_cell(LEFT));
	Work1 := RECORD(Layout_Cell)
    PBblas.Types.partition_t     partition_id;
    PBblas.Types.node_t          node_id;
    PBblas.Types.dimension_t     block_row;
    PBblas.Types.dimension_t     block_col;
  END;
  FromCells(PBblas.IMatrix_Map mat_map, DATASET(Layout_Cell) cells,
                   PBblas.Types.dimension_t insert_columns=0,
                   PBblas.Types.value_t insert_value=0.0d) := FUNCTION
    Work1 cvt_2_xcell(Layout_Cell lr) := TRANSFORM
      block_row           := mat_map.row_block(lr.x);
      block_col           := mat_map.col_block(lr.y + insert_columns);
      partition_id        := mat_map.assigned_part(block_row, block_col);
      SELF.partition_id   := partition_id;
      SELF.node_id        := ((block_row-1) % dmap_usednodes);// instead of using partition id in order to distribute the data, block row number is used 
      SELF.block_row      := block_row;
      SELF.block_col      := block_col;
      SELF := lr;
    END;
    inMatrix := cells.x BETWEEN 1 AND mat_map.matrix_rows
            AND cells.y BETWEEN 1 AND mat_map.matrix_cols - insert_columns;
    d0 := PROJECT(cells(inMatrix), cvt_2_xcell(LEFT));
    d1 := DISTRIBUTE(d0, node_id);
    d2 := SORT(d1, partition_id, y, x, LOCAL);    // prep for column major
    d3 := GROUP(d2, partition_id, LOCAL);
    Layout_Part roll_cells(Work1 parent, DATASET(Work1) cells) := TRANSFORM
      first_row     := mat_map.first_row(parent.partition_id);
      first_col     := mat_map.first_col(parent.partition_id);
      part_rows     := mat_map.part_rows(parent.partition_id);
      part_cols     := mat_map.part_cols(parent.partition_id);
      SELF.mat_part := PBblas.MakeR8Set(part_rows, part_cols, first_row, first_col,
                                        PROJECT(cells, Layout_Cell),
                                        insert_columns, insert_value);
      SELF.partition_id:= parent.partition_id;
      SELF.node_id     := parent.node_id;
      SELF.block_row   := parent.block_row;
      SELF.block_col   := parent.block_col;
      SELF.first_row   := first_row;
      SELF.part_rows   := part_rows;
      SELF.first_col   := first_col;
      SELF.part_cols   := part_cols;
      SELF := [];
    END;
    rslt := ROLLUP(d3, GROUP, roll_cells(LEFT, ROWS(LEFT)));
    RETURN rslt;
  END;
	ddist := FromCells(dmap, Xtran_cell, insert_columns, insert_value);
	

	Numfeat:= NumberofFeatures;
  NumClass := NumberofClasses;// number of features
	m_ := -1/m;
  part_rows := prows;
  part_cols := pcols;

	//maps used
	SET OF PBblas.Types.value_t empty_array := [];
	map_c := PBblas.Matrix_Map(NumClass, m, NumClass, part_cols);
	theta_map := PBblas.Matrix_Map(NumClass, Numfeat, NumClass, part_rows);
nodes_available := Thorlib.nodes();
	data_nodesused := MIN(nodes_available, dmap.row_blocks);
	Layout_Part_newnode := RECORD (Layout_Part)
		PBblas.Types.node_t new_node_id;
	END;

	//calculated theta*TrainData
	//M=(theta*x);
	Layout_Part multx(Layout_Part t_part, Layout_Part x_part):=TRANSFORM
		part_id := x_part.block_col;
		part_c_rows := map_c.part_rows(part_id);
		part_c_cols := map_c.part_cols(part_id);
		k := t_part.part_cols;
		SELF.partition_id := part_id;
		SELF.node_id      := map_c.assigned_node(part_id);
		SELF.block_row    := t_part.block_row;
		SELF.block_col    := x_part.block_col;
		SELF.first_row    := map_c.first_row(part_id);
		SELF.part_rows    := part_c_rows;
		SELF.first_col    := map_c.first_col(part_id);
		SELF.part_cols    := part_c_cols;
		SELF.mat_part     := PBblas.BLAS.dgemm(FALSE, FALSE,
																		part_c_rows, part_c_cols, k,
																		1.0, t_part.mat_part, x_part.mat_part,
																		0.0, empty_array);
  END;
	tx_ := JOIN (theta, ddist, LEFT.block_col = RIGHT.block_row, multx(LEFT, RIGHT), LOCAL );
	tx_dist := DISTRIBUTE (tx_, node_id);
	// Sum terms
	//In the transform function check whether the left side is NULL, it can be possible that there is only one partition on one node that needs to rollup
  Layout_Part sumTerms(Layout_Part cumm, Layout_Part term) := TRANSFORM
		HaveTerm := IF(term.part_cols=0, FALSE, TRUE);
    N := cumm.part_rows * cumm.part_cols;
    SELF.mat_part := IF (HaveTerm, PBblas.BLAS.daxpy(N, 1.0, cumm.mat_part, 1, term.mat_part, 1), cumm.mat_part);
    SELF := cumm;
  END;
	tx_sorted := SORT(tx_dist, partition_id, LOCAL);
  tx := ROLLUP(tx_sorted, sumTerms(LEFT, RIGHT), partition_id, LOCAL);
	// define a c++ function who does the following operations on each partition
	// M=tx;
	// M = bsxfun(@minus, M, max(M, [], 1));
	// M=exp(M);
	// M = bsxfun(@rdivide, M, sum(M));
	//r : rows
	//s : cols
	// D : input matrix
	SET OF REAL8 soft_fun (PBblas.Types.dimension_t r, PBblas.Types.dimension_t s, PBblas.Types.matrix_t D) := BEGINC++

    #body
    __lenResult = r * s* sizeof(double);
    __isAllResult = false;
    double * result = new double[r*s];
    __result = (void*) result;
    double *cell = (double*) d;
		double *max_arr = new double[s];
		double *sum_arr = new double[s];
		double max_tmp;
		double sum_tmp;
    uint32_t cells =  r * s;
    uint32_t i;
		uint32_t j;
    uint32_t pos;
		uint32_t posj;
		for (i=0; i<s; i++) {
			pos = i * r;
			max_tmp = cell[pos];
			for (j=1; j<r; j++)
			{
				posj = pos +j;
				if(cell[posj]>max_tmp)
					max_tmp = cell[posj];
			}
			max_arr[i]=max_tmp;
    }
		for (i=0; i<cells; i++) {
			pos = (int) i / r;
			result[i]=exp(cell[i]-max_arr[pos]);
		}
    for (i=0; i<s; i++) {
			sum_tmp = 0;
			pos = i * r;
			for (j=0; j<r; j++)
			{
					sum_tmp = sum_tmp + result[pos+j];
			}
			sum_arr[i]=sum_tmp;
    }
		for (i=0; i<cells; i++) {
			pos = (int) i / r;
			result[i]=result[i] / sum_arr[pos];
		}

  ENDC++; // END soft_fun
	Layout_Part soft_tran(Layout_Part le) := TRANSFORM
    SELF.mat_part := soft_fun(le.part_rows, le.part_cols, le.mat_part);
    SELF := le;
  END;
	tx_soft := PROJECT (tx, soft_tran (LEFT), LOCAL);//same as M in MATLAB
	Prob_mat := DMAT.Converted.FromPart2Elm (tx_soft);
	Types.l_result tr(Mat.Types.Element le) := TRANSFORM
		SELF.value := le.x;
		SELF.id := le.y;
		SELF.number := 1;
		SELF.conf := le.value;
  END;
	RETURN PROJECT (Prob_mat, tr(LEFT));
 END;//ClassProbDistribC
 EXPORT ClassifyC(DATASET(Types.NumericField) Indep, DATASET(lbfgs_rec) mod) := FUNCTION
    Dist := ClassProbDistribC(Indep, mod);
    numrow := MAX (Dist,Dist.value);
    S:= SORT(Dist,id,conf);
    SeqRec := RECORD
			l_result;
			INTEGER8 Sequence := 0;
    END;
    //add seq field to S
    SeqRec AddS (S l, INTEGER c) := TRANSFORM
			SELF.Sequence := c%numrow;
			SELF := l;
    END;
    Sseq := PROJECT(S, AddS(LEFT,COUNTER));
    classified := Sseq (Sseq.Sequence=0);
    RETURN PROJECT(classified,l_result);
  END; // END ClassifyC Function
 
END;//softmax_lbfgs_partitions_datadist

// add extracting distinct labels
// extracting distinct features
// whether to normalize features
END;//END DeepLearning