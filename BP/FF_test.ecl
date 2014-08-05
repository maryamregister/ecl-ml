IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;




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


 HiddenLayerDelta (DATASET($.M_Types.MatRecord) w,DATASET($.M_Types.MatRecord) d, DATASET($.M_Types.MatRecord) a) := FUNCTION

wTd    := Ml.Mat.Mul(Ml.Mat.Trans(w),d);
Result := Ml.Mat.Each.Mul(wTd,SigGrad(a)); 
 
RETURN Result;
 
 END;








d := DATASET([
{1,1,0.1},
{1,2,0.9},
{1,3,0.6},
{2,1,0.4},
{2,2,0.3},
{2,3,0.1},
{3,1,0.2},
{3,2,0.4},
{3,3,0.7},
{4,1,0.4},
{4,2,0.5},
{4,3,0.3}],
$.M_Types.MatRecord);



OUTPUT(d, ALL, NAMED('d'));

W1 := DATASET ([
{2,1,1,0.1},
{2,1,2,0.1},
{2,2,1,0.2},
{2,2,2,0.2},
{2,3,1,0.1},
{2,3,2,0.2},
{2,4,1,0.1},
{2,4,2,0.2},
{1,1,1,0.1},
{1,1,2,0.2},
{1,1,3,0.1},
{1,1,4,0.1},
{1,2,1,0.2},
{1,2,2,0.1},
{1,2,3,0.2},
{1,2,4,0.1},
{3,1,1,0.1},
{3,1,2,0.2},
{3,1,3,0.3},
{3,1,4,0.4},
{3,2,1,0.6},
{3,2,2,0.5},
{3,2,3,0.4},
{3,2,4,0.3},
{3,3,1,0.1},
{3,3,2,0.1},
{3,3,3,0.2},
{3,3,4,0.2}],
$.M_types.IDMatRec);
OUTPUT(w1, ALL, NAMED('w1'));

B1 := DATASET ([
{2,1,1,1},
{2,2,1,2},
{2,3,1,3},
{2,4,1,4},
{1,1,1,1},
{1,2,1,2},
{3,1,1,1},
{3,2,1,2},
{3,3,1,3}],
$.M_types.IDMatRec);
OUTPUT(b1, ALL, NAMED('b1'));




// make id numbers for the matrices which need to aggregated in a cell 
IDW := DATASET ([
{1},
{2},
{3}],
$.M_Types.IDRec);

IDB := DATASET ([
{1},
{2},
{3}],
$.M_Types.IDRec);





// make the cell for weights, paches, 
w := $.Cell(IDW,W1) ;
OUTPUT(w,NAMED('w'));

b := $.Cell(IDB,B1) ;
OUTPUT(b,NAMED('b'));







wb := $.cell2(w,b);
OUTPUT(wb,NAMED('wb'));
//ready to apply ITERATE
// make a dataset like [{ 2 z2 a2},{3 w2 b2},{4 w3 b3},...,{n+1 wn bn}], here id is actually the id of the
// next layar's z and a values which are going to be calculated based on the preceding layaer's w and b

$.M_Types.CellMatRec2 P1(wb le) := TRANSFORM 
	  $.M_Types.MatRecord z       := ML.Mat.Vec.Add_Mat_Vec (ML.Mat.Mul(le.cellmat1,d),le.cellmat2,1);
		SELF.cellmat1 := IF (le.id=1, z , le.cellmat1);
		SELF.cellmat2 := IF (le.id=1, ML.Mat.Each.Sigmoid(z) , le.cellmat2);
		SELF.id := le.id+1;
	END;
	
Temp := PROJECT(wb, P1(LEFT)); 

OUTPUT(Temp,NAMED('Temp'));
//Apply ITERATE 


$.M_Types.CellMatRec2 IT($.M_Types.CellMatRec2 L, $.M_Types.CellMatRec2 R, INTEGER C) := TRANSFORM
  $.M_Types.MatRecord z := IF(C!=1,ML.Mat.Vec.Add_Mat_Vec (ML.Mat.Mul(R.cellmat1,L.cellmat2),R.cellmat2,1));
  SELF.cellmat1 := IF (C=1,R.cellmat1,z);
	SELF.cellmat2 := IF (C=1,R.cellmat2,ML.Mat.Each.Sigmoid(z));
  SELF := R;
END;

MySet1 := ITERATE(TEMP,IT(LEFT,RIGHT,COUNTER));

OUTPUT(MySet1,NAMED('MySet1'));

//make just a

a1 := DATASET ([{1,d}],$.M_Types.CellMatRec);


$.M_Types.CellMatRec P2 (MySet1 l) := TRANSFORM
	SELF.cellMat := l.cellMat2;
	SELF.id      := l.id;
END;

Justa := a1+PROJECT(MySet1, P2(LEFT));
OUTPUT(Justa,NAMED('Justa'));


wExtra := DATASET ([{4,[]}],$.M_Types.CellMatRec);
wa := $.cell2(w+wExtra,Justa);
OUTPUT(wa,NAMED('wa'));

//calculate grad cost
S := SORT(wa,-id);
OUTPUT(S,NAMED('SSS'));



y := DATASET([
{1,1,1},
{1,2,0},
{1,3,0},
{2,1,0},
{2,2,0},
{2,3,1},
{3,1,0},
{3,2,1},
{3,3,0}],
$.M_Types.MatRecord);






SWA := SORT(wa,-id);
OUTPUT(SWA,NAMED('SWA'));

//APPLY ITERATE




$.M_Types.CellMatRec2 IT2($.M_Types.CellMatRec2 L, $.M_Types.CellMatRec2 R, INTEGER C) := TRANSFORM
  $.M_Types.MatRecord z := IF(C=1,LastLayerDelta(y,R.cellmat2),HiddenLayerDelta(R.cellmat1,L.cellmat2,R.cellmat2));//
  SELF.cellmat2 := z;
  SELF := R;
END;

MySet2 := ITERATE(SWA,IT2(LEFT,RIGHT,COUNTER));// this result is as [{2,z2,a2},{3,z3,a3},...]
OUTPUT(MySet2,NAMED('MySet2'));

$.M_types.CellMatRec TDelta(MySet2 l) := TRANSFORM
  SELF.id := l.id;
  SELF.cellMat := l.cellMat2;
END;

DELTA := PROJECT(MySet2,TDelta(LEFT)) ;

OUTPUT(DELTA,NAMED('DELTA'));


$.M_types.CellMatRec TW(wa l) := TRANSFORM
  SELF.id := l.id;
  SELF.cellMat := l.cellMat1;
END;


WW := PROJECT(wa,TW(LEFT)) ;

OUTPUT(WW,NAMED('WW'));

$.M_types.CellMatRec TA(wa l) := TRANSFORM
  SELF.id := l.id;
  SELF.cellMat := l.cellMat2;
END;

A := PROJECT(wa,TA(LEFT)) ;

OUTPUT(A,NAMED('A'));


//first w grad term
$.M_types.CellMatRec Jgrad (DELTA l, A r) := TRANSFORM
		SELF.cellMat := Ml.Mat.Mul(l.cellMat, Ml.Mat.Trans(r.cellMat));
		SELF := r;
END;

GradW_Term1 := JOIN(DELTA,A,LEFT.id=(RIGHT.id+1),Jgrad(LEFT,RIGHT));
OUTPUT(GradW_Term1,NAMED('GradW_Term1'));

//add weight decay to calculate final gradiation

lambda := 0.1; //???? define lambda seprately
M:=3;//define M
M_1 := 1/M;
$.M_types.CellMatRec JfinalWgrad (GradW_Term1 l, WW r) := TRANSFORM
		WD :=Ml.Mat.Each.ScalarMul(r.cellMat,lambda); //weight decay term
		GM := Ml.Mat.Each.ScalarMul(l.cellMat,M_1); //Grad Mean
		SELF.cellMat := Ml.Mat.Add(GM,WD);
		SELF := r;
END;

GradW_final := JOIN(GradW_Term1,W,LEFT.id=RIGHT.id,JfinalWgrad(LEFT,RIGHT));

OUTPUT(GradW_final,NAMED('GradW_final'));


//calculate b gradients
$.M_types.CellMatRec Bgrad (DELTA l) := TRANSFORM
		SELF.cellMat := Ml.Mat.Each.ScalarMul(Ml.Mat.Has(l.cellMat).MeanRow,M_1);
		SELF := l;
END;

GradB_final := PROJECT (DELTA, Bgrad(LEFT));

OUTPUT(GradB_final,NAMED('GradB_final'));
