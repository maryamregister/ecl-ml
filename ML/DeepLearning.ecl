IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat;
IMPORT * FROM ML.Types;
IMPORT PBblas;
Layout_Cell := PBblas.Types.Layout_Cell;
Layout_Part := PBblas.Types.Layout_Part;

EXPORT DeepLearning := MODULE
//Implementation of the Sparse Autoencoder based on the stanford Deep Learning tutorial
EXPORT Sparse_Autoencoder (DATASET(Mat.Types.MUElement) IntW, DATASET(Mat.Types.MUElement) Intb, REAL8 LAMBDA=0.001, REAL8 ALPHA=0.1, UNSIGNED2 MaxIter=100,
  UNSIGNED4 prows=0, UNSIGNED4 pcols=0, UNSIGNED4 Maxrows=0, UNSIGNED4 Maxcols=0) := MODULE
  //this is a un-supervised learning algorithm, no need for the labled data
  SA(DATASET(Types.NumericField) X) := MODULE
    dt := Types.ToMatrix (X);
    dTmp := dt;
    SHARED d := Mat.Trans(dTmp); //in the entire of the calculations we work with the d matrix that each sample is presented in one column
    SHARED m := MAX (d, d.y); //number of samples
    SHARED m_1 := 1/m;
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
    //w1no := PBblas.MU.TO(w1dist,1);
    
    w2_mat := Mat.MU.From(IntW,2);
    w2_mat_x := w1_mat_y;
    w2_mat_y := w1_mat_x;
    w2map := PBblas.Matrix_Map(w2_mat_x, w2_mat_y, sizeTable[1].f_b_rows, sizeTable[1].f_b_rows);
    w2dist := DMAT.Converted.FromElement(w2_mat,w2map);
    //w2no := PBblas.MU.TO(w1dist,1);

    //two kind of Bias blocks are calculated
    //1- each bias vector is converted to block format
    //2-each Bias vector is repeated first to m columns, then the final repreated bias matrix is converted to block format
    //the second kind of bias is calculated to make the next calculations easier, the first vector bias format is used just when we
    //want to update the bias vectors
    //Creat block vectors for Bias (above case 1)
    b1vec := Mat.MU.From(Intb,1);
    b1vec_x := Mat.Has(b1vec).Stats.Xmax;
    b1vecmap := PBblas.Matrix_Map(b1vec_x, 1, sizeTable[1].f_b_rows, 1);
    b1vecdist := DMAT.Converted.FromElement(b1vec,b1vecmap);
    //b1vecno := PBblas.MU.TO(b1vecdist,1);
   
    b2vec := Mat.MU.From(Intb,2);
    b2vec_x := Mat.Has(b2vec).Stats.Xmax;
    b2vecmap := PBblas.Matrix_Map(b2vec_x, 1, sizeTable[1].f_b_rows, 1);
    b2vecdist := DMAT.Converted.FromElement(b2vec,b2vecmap);
    //b2vecno := PBblas.MU.TO(b1vecdist,1);
    //Creat block matrices for Bias (repeat each bias vector to a matrix with m columns) (above case 2)
    b1_mat := Mat.MU.From(Intb,1);
    b1_mat_x := Mat.Has(b1_mat).Stats.Xmax;
    b1_mat_rep := Mat.Repmat(b1_mat, 1, m); // Bias vector is repeated in m columns to make the future calculations easier
    b1map := PBblas.Matrix_Map(b1_mat_x, m, sizeTable[1].f_b_rows, sizeTable[1].f_b_cols);
    b1dist := DMAT.Converted.FromElement(b1_mat_rep,b1map);
    //b1no := PBblas.MU.TO(b1dist,1);
    
    b2_mat := Mat.MU.From(Intb,2);
    b2_mat_x := Mat.Has(b2_mat).Stats.Xmax;
    b2_mat_rep := Mat.Repmat(b2_mat, 1, m); // Bias vector is repeated in m columns to make the future calculations easier
    b2map := PBblas.Matrix_Map(b2_mat_x, m, sizeTable[1].f_b_rows, sizeTable[1].f_b_cols);
    b2dist := DMAT.Converted.FromElement(b2_mat_rep,b2map);
    //b2no := PBblas.MU.TO(b2dist,1);
    //by now we have converted all our parameters to the partition format
    //weight parameters: w1dist,w2dist
    //bias vector parameters: b1vecdist, b2vecdist
    //matrix of bias vectors for making future calculations easier b1dist, b2dist
    // creat ones vector for calculating bias gradients
    Layout_Cell gen(UNSIGNED4 c, UNSIGNED4 NumRows, REAL8 v) := TRANSFORM
      SELF.x := ((c-1) % NumRows) + 1;
      SELF.y := ((c-1) DIV NumRows) + 1;
      SELF.v := v;
     END;
     onesmap := PBblas.Matrix_Map(m, 1, sizeTable[1].f_b_cols, 1);
     ones := DATASET(m, gen(COUNTER, m, 1.0),DISTRIBUTED);
     onesdist := DMAT.Converted.FromCells(onesmap, ones);
    //functions used
    PBblas.Types.value_t sigmoid(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := 1/(1+exp(-1*v));
    //maps used
    HL_nodes := w1_mat_x;//number of nodes in the hidden layer
    a2map := PBblas.Matrix_Map(HL_nodes,sizeTable[1].m_cols,HL_nodes,sizeTable[1].f_b_cols);
    FF(DATASET(Layout_Part) w1,  DATASET(Layout_Part) w2,DATASET(Layout_Part) b1, DATASET(Layout_Part) b2 ):= FUNCTION
      //z2 = w1*X+b1;
      z2 := PBblas.PB_dgemm(FALSE, FALSE,1.0,w1map, w1, dmap, ddist, b1map,b1, 1.0);
      //a2 = sigmoid (z2);
      a2 := PBblas.Apply2Elements(b1map, z2, sigmoid);
      a2_no := PBblas.MU.To(a2,2);
      //z3 = w2*a2+b2;
      z3 := PBblas.PB_dgemm(FALSE, FALSE,1.0,w2map, w2, a2map, a2, b2map,b2, 1.0);
      //a3 = sigmoid (z3)
      a3 := PBblas.Apply2Elements(b1map, z3, sigmoid);
      a3_no := PBblas.MU.To(a3,3);
      RETURN a2_no+a3_no;
    END;//END FF
    EXPORT alaki := FF(w1dist,w2dist,b1dist,b2dist);
    // w1_no := PBblas.MU.To(w1dist,1);
    // w2_no := PBblas.MU.To(w2dist,2);
    // wparam := w1_no+w2_no;
    // b1_no := PBblas.MU.To(b1dist,1);
    // b2_no := PBblas.MU.To(b2dist,2);
    // bparam := b1_no+b2_no;
 // FF2(DATASET(PBblas.Types.MUElement) w, DATASET(PBblas.Types.MUElement) b ):= FUNCTION
// net := DATASET([
// {1, 1, 3},
// {2,1,3},
// {3,1,3}],
// Types.DiscreteField);
// iterations :=1; 
      // w1 := PBblas.MU.From(W, 1); // weight matrix between layer 1 and layer 2 of the neural network
      // b1 := PBblas.MU.From(b, 1); //bias entered to the layer 2 of the neural network
      // z2 := PBblas.PB_dgemm(FALSE, FALSE,1.0,w1map, W1, dmap, ddist, b1map,b1, 1.0  );
      //a2 = sigmoid (z2);
      // a2 := PBblas.Apply2Elements(b1map, z2, sigmoid);
      // a2no := PBblas.MU.To(a2,2);

      // FF_Step(DATASET(PBblas.Types.MUElement) InputA, INTEGER coun) := FUNCTION
        // L := coun+1;
        // wL := PBblas.MU.From(w, L); // weight matrix between layer L and layer L+1 of the neural network
        // wL_x := net(id=(L+1))[1].value;
        // wL_y := net(id=(L))[1].value;;
        // bL := PBblas.MU.From(b, L); //bias entered to the layer L+1 of the neural network
        // bL_x := net(id=(L+1))[1].value;
        // aL := PBblas.MU.From(InputA, L); //output of layer L
        // aL_x := net(id=(L))[1].value;;
        // wLmap := PBblas.Matrix_Map(wL_x, wL_y, sizeTable[1].f_b_rows, sizeTable[1].f_b_rows);
        // bLmap := PBblas.Matrix_Map(bL_x, m, sizeTable[1].f_b_rows, sizeTable[1].f_b_cols);
        // aLmap := PBblas.Matrix_Map(aL_x,m,sizeTable[1].f_b_rows,sizeTable[1].f_b_cols);
        //z(L+1) = wL*aL+bL;
        // zL_1 := PBblas.PB_dgemm(FALSE, FALSE,1.0,wLmap, wL, aLmap, aL, bLmap,bL, 1.0  );
        //aL_1 = sigmoid (zL_1);
        // aL_1 := PBblas.Apply2Elements(bLmap, zL_1, sigmoid);
        // aL_1no := PBblas.MU.To(aL_1,L+1);
        // RETURN InputA+aL_1no;
      // END;//end FF_step
      // final_A := LOOP(a2no, COUNTER <= iterations, FF_Step(ROWS(LEFT),COUNTER));
      // return final_A;
    // END;//end FF2
     // alaki2 := FF2(wparam,bparam);
     // EXPORT alaki3 := alaki+alaki2;
  END;//END SA
  EXPORT testit(DATASET(Types.NumericField) Indep) := SA(Indep).alaki;
END;//END Sparse_Autoencoder
END;//END DeepLearning