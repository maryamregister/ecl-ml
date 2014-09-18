IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;
// This fucntion back propagate the error through the netwrok to calculate the gradients. it also calculate the 
//cost. The input is the desired outputs and the network weights and actual outputs of the feed forward pass



EXPORT SA_BPcost(DATASET($.M_Types.MatRecord) Desired,DATASET($.M_Types.CellMatRec) W,DATASET($.M_Types.CellMatRec) A, REAL8 LAMBDA, REAL sparsityParam=0.1, REAL BETA=0.1 ) := MODULE

 SHARED SigGrad(DATASET($.M_Types.MatRecord) m ) := FUNCTION // this function recieves a matrix m and return m.*(1-m)

m_1 := ML.Mat.Each.add(ML.Mat.Each.ScalarMul(m,-1),1);
Result := ML.Mat.Each.Mul(m,m_1);
RETURN Result;
END;


 SHARED LastLayerDelta (DATASET($.M_Types.MatRecord) y, DATASET($.M_Types.MatRecord) a) := FUNCTION

y_a := ML.Mat.Sub(y,a);
R := Ml.Mat.Each.Mul (y_a,SigGrad(a));
Result1 := Ml.Mat.Each.ScalarMul(R,-1);
RETURN Result1;

END;



 //hidden layer for sparse autoencoder
  SHARED HiddenLayerDelta (DATASET($.M_Types.MatRecord) w,DATASET($.M_Types.MatRecord) delta, DATASET($.M_Types.MatRecord) a) := FUNCTION

/*
rhohat=mean(a2,2);


sparsity_delta=((-sparsityParam./rhohat)+((1-sparsityParam)./(1.-rhohat)));

d2=((W2'*d3)+beta*repmat(sparsity_delta,1,m)).*(a2.*(1-a2));
*/

rhohat := ML.Mat.has(a).MeanRow;

 

sparsityParam_  := -1*sparsityParam;
sparsityParam_1 := 1-sparsityParam;

term1    := ML.Mat.Each.Reciprocal(rhohat, sparsityParam_); 
rhohat_1 := ML.Mat.Each.Add(ML.Mat.Each.ScalarMul(rhohat,-1),1);
term2    := ML.Mat.Each.Reciprocal(rhohat_1, sparsityParam_1); 

sparsity_delta := ML.Mat.Add(term1, term2);



wTd    := Ml.Mat.Mul(Ml.Mat.Trans(w),delta);

mm := Max (Desired, Desired.y);
sparsity_delta_1_m := ML.Mat.Repmat(sparsity_delta,1,mm);
beta_sparsity_delta_1_m := ML.mat.Each.ScalarMul(sparsity_delta_1_m, BETA);

term := ML.Mat.Add(wTd, beta_sparsity_delta_1_m );
Result := Ml.Mat.Each.Mul(term,SigGrad(a)); 
 
RETURN Result;
 
 END;





SHARED M := Max (Desired, Desired.y); // number of samples (each colomn of Desired is related to one sample)
SHARED M_1 := 1/M;




EXPORT Delta := FUNCTION

nw :=  Max (w, w.id);
wExtra := DATASET ([{nw+1,[]}],$.M_Types.CellMatRec);
wa := $.cell2(W+wExtra,A);


SWA := SORT(wa,-id);


//APPLY ITERATE

$.M_Types.CellMatRec2 IT($.M_Types.CellMatRec2 L, $.M_Types.CellMatRec2 R, INTEGER C) := TRANSFORM
  $.M_Types.MatRecord z := IF(C=1,LastLayerDelta(Desired,R.cellmat2),HiddenLayerDelta(R.cellmat1,L.cellmat2,R.cellmat2));//
  SELF.cellmat2 := z;
	SELF.cellmat1 := [];
  SELF := R;
END;

Myset1 := ITERATE(SWA,IT(LEFT,RIGHT,COUNTER));// this result is as //[{n+1,[],delta_n+1},..,{2,w2,delta2},{1,w1,delta1}]

$.M_types.CellMatRec TDelta(MySet1 l) := TRANSFORM
  SELF.id := l.id;
  SELF.cellMat := l.cellMat2;
END;

DD := PROJECT(MySet1,TDelta(LEFT)) ;

RETURN DD;

END;












//Seprate DELTA, W and A to calculate gradients by using JOIN
EXPORT Wgrad (DATASET ($.M_types.CellMatRec) DELTA) := FUNCTION

//calculate first term of gradients

$.M_types.CellMatRec Jgrad (DELTA l, A r) := TRANSFORM
		SELF.cellMat := Ml.Mat.Mul(l.cellMat, Ml.Mat.Trans(r.cellMat));
		SELF := r;
END;

GradW_Term1 := JOIN(DELTA,A,LEFT.id=(RIGHT.id+1),Jgrad(LEFT,RIGHT));

//calculate second term of gradients for the weights (wight decay) and add it to the first term

$.M_types.CellMatRec JfinalWgrad (GradW_Term1 l, W r) := TRANSFORM
		WD :=Ml.Mat.Each.ScalarMul(r.cellMat,LAMBDA); //weight decay term
		GM := Ml.Mat.Each.ScalarMul(l.cellMat,M_1); //Grad Mean
		SELF.cellMat := Ml.Mat.Add(GM,WD);
		SELF := r;
END;

GradW_final := JOIN(GradW_Term1,W,LEFT.id=RIGHT.id,JfinalWgrad(LEFT,RIGHT));




//calculate b gradients

// $.M_types.CellMatRec Bgrad (DELTA l) := TRANSFORM
		// SELF.cellMat := Ml.Mat.Each.ScalarMul(Ml.Mat.Has(l.cellMat).MeanRow,M_1);
		// SELF := l;
// END;

// GradB_final := PROJECT (DELTA, Bgrad(LEFT));


//calculate cost

Return GradW_final;

END;



EXPORT Bgrad (DATASET ($.M_types.CellMatRec) DELTA) := FUNCTION


//calculate b gradients

$.M_types.CellMatRec PBgrd (DELTA l) := TRANSFORM
		SELF.cellMat := Ml.Mat.Has(l.cellMat).MeanRow;
		SELF.id := l.id - 1;
END;

GradB_final := PROJECT (DELTA, PBgrd(LEFT));

RETURN GradB_final;
END;

//to be completed later
EXPORT Cost := FUNCTION



RETURN 10;
END;

END;






