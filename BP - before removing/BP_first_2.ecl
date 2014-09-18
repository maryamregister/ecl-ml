IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;


MatRecord := ML.Mat.Types.Element;

VecRecord := ML.Mat.Types.VecElement;
MatIDRec := RECORD
UNSIGNED8  id;
MatRecord;
END;

Patches := DATASET([
{1,1,1,0.1},
{1,1,2,0.9},
{1,1,3,0.6},
{1,2,1,0.4},
{1,2,2,0.3},
{1,2,3,0.1},
{1,3,1,0.2},
{1,3,2,0.4},
{1,3,3,0.7},
{1,4,1,0.4},
{1,4,2,0.5},
{1,4,3,0.3}],
MatIDRec);
OUTPUT(Patches, ALL, NAMED('Patches'));

W := DATASET ([
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
{1,2,4,0.1}],
MatIDRec);

//define weights and bias values , the weights must be initialized randomly
// W1 = rand (hiddensize, visiblesize)
W1 := DATASET ([
{1,1,1,0.1},
{1,1,2,0.2},
{1,1,3,0.1},
{1,1,4,0.1},
{1,2,1,0.2},
{1,2,2,0.1},
{1,2,3,0.2},
{1,2,4,0.1}],
MatIDRec);
OUTPUT(W1, ALL, NAMED('W1'));
// W2 = rand (visiblesize, hiddensize)
W2 := DATASET ([
{2,1,1,0.1},
{2,1,2,0.1},
{2,2,1,0.2},
{2,2,2,0.2},
{2,3,1,0.1},
{2,3,2,0.2},
{2,4,1,0.1},
{2,4,2,0.2}],
MatIDRec);


OUTPUT(W2, ALL, NAMED('W2'));
//b1 = zeros (hidensize);
b1 := DATASET ([
{1,1,0},
{2,1,0}],
VecRecord);
OUTPUT(b1, ALL, NAMED('b1'));
//b2 = zeros (visiblesize);
b2 := DATASET ([
{1,1,0},
{2,1,0},
{3,1,0},
{4,1,0}],
VecRecord);
OUTPUT(b2, ALL, NAMED('b2'));


CellMatRecord := RECORD 
UNSIGNED8  id;
INTEGER1 NumRows;
DATASET(MatIDRec) cellMat {MAXCOUNT(100)};
END;



ParentRec := RECORD
    UNSIGNED8  id;
END;
 
par := DATASET ([
{1},
{2}],
ParentRec);

 
 CellMatRecord ParentLoad(ParentRec L) := TRANSFORM
    SELF.NumRows := 0;
		SELF.cellMat := [];
    SELF := L;
END;
//Ptbl := TABLE(NamesTable,DenormedRec);
Ptbl := PROJECT(par,ParentLoad(LEFT));
OUTPUT(Ptbl,NAMED('Ptbl'));

OUTPUT(W,NAMED('W'));


CellMatRecord DeNormThem(CellMatRecord L, MatIDRec R, INTEGER C) := TRANSFORM
    SELF.NumRows := C;
    SELF.cellMat := L.cellMat + R;
    SELF := L;
END;

Result := DENORMALIZE(Ptbl, W,
				    LEFT.id = RIGHT.id,
				    DeNormThem(LEFT,RIGHT,COUNTER));

OUTPUT(Result,NAMED('Result'));








parP := DATASET ([
{1}],
ParentRec);

 

//Ptbl := TABLE(NamesTable,DenormedRec);
Ptbl2 := PROJECT(parP,ParentLoad(LEFT));
OUTPUT(Ptbl2,NAMED('Ptbl2'));




Result2 := DENORMALIZE(Ptbl2, Patches,
				    LEFT.id = RIGHT.id,
				    DeNormThem(LEFT,RIGHT,COUNTER));

OUTPUT(Result2,NAMED('Result2'));



CellMatRecord2 := RECORD 
UNSIGNED8  id;
INTEGER1 NumRows;
DATASET(MatRecord) cellMat {MAXCOUNT(100)};
END;
R2 := PROJECT(Result2,CellMatRecord2) ;
OUTPUT(R2,NAMED('R2'));

R1 := PROJECT(Result,CellMatRecord2) ;
OUTPUT(R1,NAMED('R1'));

CellMatRecord2 Mulc(CellMatRecord2 le,CellMatRecord2 ri) := TRANSFORM
		SELF.cellMat := ML.Mat.Mul(le.cellMat,ri.cellMat); 
		SELF := le;
	END;
	
  outputt := JOIN(R1,R2,LEFT.id=RIGHT.id,Mulc(LEFT,RIGHT)); // Form all of the intermediate computations

OUTPUT(outputt,NAMED('outputt'));



