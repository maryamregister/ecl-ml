//my original implementation of Feed Forward pass
IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;
// This function fucntion apply the Feed Forward pass by feeding d to the network
//with parameters w as weights and b as bias values
//the output is a cell of z, a values. 


EXPORT FF3(DATASET($.M_Types.MatRecord) d,DATASET($.M_Types.CellMatRec) w, DATASET($.M_Types.CellMatRec) b ) := FUNCTION


wb := $.cell2(w,b);

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


//Apply ITERATE 


$.M_Types.CellMatRec2 IT($.M_Types.CellMatRec2 L, $.M_Types.CellMatRec2 R, INTEGER C) := TRANSFORM
  $.M_Types.MatRecord z := IF(C!=1,ML.Mat.Vec.Add_Mat_Vec (ML.Mat.Mul(R.cellmat1,L.cellmat2),R.cellmat2,1));
  SELF.cellmat1 := IF (C=1,R.cellmat1,z);
	SELF.cellmat2 := IF (C=1,R.cellmat2,ML.Mat.Each.Sigmoid(z));
  SELF := R;
END;

MySet1 := ITERATE(TEMP,IT(LEFT,RIGHT,COUNTER));

RETURN MySet1;

END;

