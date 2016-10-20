IMPORT * FROM ML;
IMPORT * FROM $;
IMPORT PBblas;
Layout_Cell := PBblas.Types.Layout_Cell;
Layout_Part := PBblas.Types.Layout_Part;
// W20160722-150442
// test Sparse_Autoencoder_lbfgs on an image dataset, the dataset includes patches of randome imagae of size 8 by 8

INTEGER4 hl := 25;//number of nodes in the hiddenlayer
INTEGER4 f := 8*8;//number of input features

//input data

value_record := RECORD
real	f1	;
real	f2	;
real	f3	;
real	f4	;
real	f5	;
real	f6	;
real	f7	;
real	f8	;
real	f9	;
real	f10	;
real	f11	;
real	f12	;
real	f13	;
real	f14	;
real	f15	;
real	f16	;
real	f17	;
real	f18	;
real	f19	;
real	f20	;
real	f21	;
real	f22	;
real	f23	;
real	f24	;
real	f25	;
real	f26	;
real	f27	;
real	f28	;
real	f29	;
real	f30	;
real	f31	;
real	f32	;
real	f33	;
real	f34	;
real	f35	;
real	f36	;
real	f37	;
real	f38	;
real	f39	;
real	f40	;
real	f41	;
real	f42	;
real	f43	;
real	f44	;
real	f45	;
real	f46	;
real	f47	;
real	f48	;
real	f49	;
real	f50	;
real	f51	;
real	f52	;
real	f53	;
real	f54	;
real	f55	;
real	f56	;
real	f57	;
real	f58	;
real	f59	;
real	f60	;
real	f61	;
real	f62	;
real	f63	;
real	f64	;
END;

input_data_tmp := DATASET('~maryam::mytest::patches', value_record, CSV);
// OUTPUT(input_data_tmp);
ML.AppendID(input_data_tmp, id, input_data);
sample_table := input_data;
ML.ToField(sample_table, indepDataC);
//OUTPUT(indepDataC);
//define the parameters for the Sparse Autoencoder
REAL8 sparsityParam  := 0.01;
REAL8 BETA := 3;
REAL8 ALPHA := 0.1;
REAL8 LAMBDA :=0.0001;
UNSIGNED2 MaxIter :=1;
UNSIGNED4 prows:=0;
UNSIGNED4 pcols:=0;
UNSIGNED4 Maxrows:=0;
UNSIGNED4 Maxcols:=0;

//m := MAX(indepDataC, indepDataC.id);
m := 10000;

IntW1 := DeepLearning.Sparse_Autoencoder_IntWeights1(f, hl);
//OUTPUT(IntW1);
IntW2 := DeepLearning.Sparse_Autoencoder_IntWeights2(f, hl);
//OUTPUT(IntW2);
Intb1 :=  DeepLearning.Sparse_Autoencoder_IntBias1_matrix(f, hl, m);
Intb2 :=  DeepLearning.Sparse_Autoencoder_IntBias2_matrix(f, hl, m);
//OUTPUT(Intb1);

X := indepDataC;
dt := Types.ToMatrix (X);
    dTmp := dt;
    d := Mat.Trans(dTmp); //in the entire of the calculations we work with the d matrix that each sample is presented in one column
    
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
    d_n := 64;
    d_m := m;
    output_num := d_n;
    derivemap := PBblas.AutoBVMap(d_n, d_m,64,200);
    sizeTable := DATASET([{derivemap.matrix_rows,derivemap.matrix_cols,derivemap.part_rows(1),derivemap.part_cols(1)}], sizeRec);
    
    //Create block matrix d
    dmap := PBblas.Matrix_Map(sizeTable[1].m_rows,sizeTable[1].m_cols,sizeTable[1].f_b_rows,sizeTable[1].f_b_cols);
    ddist := DMAT.Converted.FromElement(d,dmap);
    //Creat block matrices for weights
    w1_mat := IntW1;
    w1_mat_x := hl;
    w1_mat_y := f;
    w1map := PBblas.Matrix_Map(w1_mat_x, w1_mat_y, w1_mat_x, sizeTable[1].f_b_rows);
    w1dist := DMAT.Converted.FromElement(w1_mat,w1map);
    w2_mat := IntW2;
    w2_mat_x := f;
    w2_mat_y := hl;
    w2map := PBblas.Matrix_Map(w2_mat_x, w2_mat_y, sizeTable[1].f_b_rows, w2_mat_y);
    w2dist := DMAT.Converted.FromElement(w2_mat,w2map);
    //each bias vector is converted to block format
    b1 := Intb1;
    b1_x := hl;
    b1map := PBblas.Matrix_Map(b1_x, m, b1_x, sizeTable[1].f_b_cols);
    b1dist := DMAT.Converted.FromElement(b1,b1map);
    b2:= Intb2;
    b2_x := f;
    b2map := PBblas.Matrix_Map(f, m, sizeTable[1].f_b_rows, sizeTable[1].f_b_cols);
    b2dist := DMAT.Converted.FromElement(b2,b2map);
    
		
		
		PBblas.Types.value_t sigmoid(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := 1/(1+exp(-1*v));
      //z2 = w1*X+b1;
      //z2 := PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1dist, dmap, ddist, b1map, b1dist, 1.0); // gives MP closed error
		  z2_ := PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1dist, dmap, ddist, b1map);
			z2 := PBblas.PB_daxpy(1.0, z2_, b1dist);
      a2 := PBblas.Apply2Elements(b1map, z2, sigmoid);
			
			//z3 = W2 * a2 + repmat(b2,1,m);
			//z3 := PBblas.PB_dgemm(FALSE, FALSE, 1.0,w2map, w2dist, b1map, a2, b2map, b2dist, 1.0); // gives MP closed error
			z3_ := PBblas.PB_dgemm(FALSE, FALSE, 1.0,w2map, w2dist, b1map, a2, b2map);
			z3 := PBblas.PB_daxpy(1.0, z3_, b2dist);
			a3 := PBblas.Apply2Elements(b2map, z3, sigmoid);
			
			PBblas.Types.value_t siggrad(PBblas.Types.value_t v, PBblas.Types.dimension_t r, PBblas.Types.dimension_t c) := v*(1.0-v);
			//y=X;
      //d3=-(y-a3).*(a3.*(1-a3));
      siggrad_a3 := PBblas.Apply2Elements(b2map, a3, siggrad);
      a3_y := PBblas.PB_daxpy(-1, ddist, a3);
      d3 := PBblas.HadamardProduct(b2map, a3_y, siggrad_a3);
			
			
			
			// rhohat=mean(a2,2);


// sparsity_delta=((-sparsityParam./rhohat)+((1-sparsityParam)./(1.-rhohat)));

// d2=((W2'*d3)+beta*repmat(sparsity_delta,1,m)).*(a2.*(1-a2));

d2 := PBblas.PB_dgemm(TRUE, FALSE, 1.0,w2map, w2dist, b2map, d3, b1map);
// OUTPUT(PBblas.PB_dgemm(FALSE, FALSE, 1.0,w1map, w1dist, dmap, ddist, b1map));
    //OUTPUT(PBblas.PB_daxpy(1.0, z3, b2dist));
		OUTPUT(a3_y);