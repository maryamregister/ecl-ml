IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;
// This fucntion back propagate the error through the netwrok to calculate the gradients. it also calculate the 
//cost. The input is the desired outputs and the network weights and actual outputs of the feed forward pass
 SigGrad(DATASET($.M_Types.MatRecord) m ) := FUNCTION // this function recieves a matrix m and return m.*(1-m)

m_1 := ML.Mat.Each.add(ML.Mat.Each.ScalarMul(m,-1),1);
Result := ML.Mat.Each.Mul(m,m_1);
RETURN Result;
END;


 LastLayerDelta (DATASET($.M_Types.MatRecord) y, DATASET($.M_Types.MatRecord) a) := FUNCTION

y_a := ML.Mat.Sub(y,a);
R := Ml.Mat.Each.Mul (y_a,SigGrad(a));
Result1 := Ml.Mat.Each.ScalarMul(R,-1);
RETURN Result1;

END;


 HiddenLayerDelta (DATASET($.M_Types.MatRecord) w,DATASET($.M_Types.MatRecord) delta, DATASET($.M_Types.MatRecord) a) := FUNCTION

wTd    := Ml.Mat.Mul(Ml.Mat.Trans(w),delta);
Result := Ml.Mat.Each.Mul(wTd,SigGrad(a)); 
 
RETURN Result;
 
 END;




EXPORT Grad(DATASET($.M_Types.MatRecord) y,DATASET($.M_Types.CellMatRec) W,DATASET($.M_Types.CellMatRec) A ) := FUNCTION

wExtra := DATASET ([{4,[]}],$.M_Types.CellMatRec);
wa := $.cell2(W+wExtra,A);


SWA := SORT(wa,-id);


//APPLY ITERATE

$.M_Types.CellMatRec2 IT($.M_Types.CellMatRec2 L, $.M_Types.CellMatRec2 R, INTEGER C) := TRANSFORM
  $.M_Types.MatRecord z := IF(C=1,LastLayerDelta(y,R.cellmat2),HiddenLayerDelta(R.cellmat1,L.cellmat2,R.cellmat2));//
  SELF.cellmat2 := z;
	SELF.cellmat1 := [];
  SELF := R;
END;

Myset1 := ITERATE(SWA,IT(LEFT,RIGHT,COUNTER));// this result is as //[{n+1,[],delta_n+1},..,{2,w2,delta2},{1,w1,delta1}]

$.M_types.CellMatRec TDelta(MySet1 l) := TRANSFORM
  SELF.id := l.id;
  SELF.cellMat := l.cellMat2;
END;

DELTA := PROJECT(MySet1,TDelta(LEFT)) ;



//Seprate DELTA, W and A to calculate gradients by using JOIN


//calculate first term of gradients

$.M_types.CellMatRec Jgrad (DELTA l, A r) := TRANSFORM
		SELF.cellMat := Ml.Mat.Mul(l.cellMat, Ml.Mat.Trans(r.cellMat));
		SELF := r;
END;

GradW_Term1 := JOIN(DELTA,A,LEFT.id=(RIGHT.id+1),Jgrad(LEFT,RIGHT));

//calculate second term of gradients for the weights (wight decay) and add it to the first term
lambda := 0.1; //???? define lambda seprately
M:=3;// ??? define M seprately
M_1 := 1/M;
$.M_types.CellMatRec JfinalWgrad (GradW_Term1 l, W r) := TRANSFORM
		WD :=Ml.Mat.Each.ScalarMul(r.cellMat,lambda); //weight decay term
		GM := Ml.Mat.Each.ScalarMul(l.cellMat,M_1); //Grad Mean
		SELF.cellMat := Ml.Mat.Add(GM,WD);
		SELF := r;
END;

GradW_final := JOIN(GradW_Term1,W,LEFT.id=RIGHT.id,JfinalWgrad(LEFT,RIGHT));




//calculate b gradients

$.M_types.CellMatRec Bgrad (DELTA l) := TRANSFORM
		SELF.cellMat := Ml.Mat.Each.ScalarMul(Ml.Mat.Has(l.cellMat).MeanRow,M_1);
		SELF := l;
END;

GradB_final := PROJECT (DELTA, Bgrad(LEFT));


//calculate cost

Return Myset1;



END;






