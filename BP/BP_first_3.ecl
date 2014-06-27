IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;


MatRecord := ML.Mat.Types.Element;

VecRecord := ML.Mat.Types.VecElement;


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
$.M_types.IDMatRec);

Patches2 := DATASET([
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
MatRecord);


// Patches := DATASET([
// {1,1,1,0.1},
// {1,1,2,0.9},
// {1,1,3,0.6},
// {1,2,1,0.4},
// {1,2,2,0.3},
// {1,2,3,0.1},
// {1,3,1,0.2},
// {1,3,2,0.4},
// {1,3,3,0.7},
// {1,4,1,0.4},
// {1,4,2,0.5},
// {1,4,3,0.3},
// {2,1,1,0.1},
// {2,1,2,0.9},
// {2,2,1,0.4},
// {2,2,2,0.3}],
// $.M_types.IDMatRed);


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
$.M_types.IDMatRec);

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
$.M_types.IDMatRec);
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
$.M_types.IDMatRec);


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

// make id numbers for the matrices which need to aggregated in a cell 
IDW := DATASET ([
{1},
{2}],
$.M_Types.IDRec);



IDPA := DATASET ([
{1},
{2}],
$.M_Types.IDRec);

// make the cell for weights, paches, 
R1 := $.Cell(IDW,W) ;
OUTPUT(R1,NAMED('R1'));

R2 := $.Cell(IDPA,Patches) ;
OUTPUT(R2,NAMED('R2'));


//Feed Forward pass


//z2 = W1 * x + repmat(b1,1,m);

// z2_t := ML.Mat.Mul(W1,x);
// OUTPUT(z2_t, ALL, NAMED('z2_t'));

// z2   := $.Add_Mat_Vec (z2_t,b1,1);

// OUTPUT(z2, ALL, NAMED('z2'));
// a2   := $.sigmoid (z2);
// OUTPUT(a2, ALL, NAMED('a2'));



$.M_Types.CellMatRec Mulc($.M_Types.CellMatRec le,$.M_Types.CellMatRec ri) := TRANSFORM
		
		SELF.cellMat := ML.Mat.Mul(le.cellMat,ri.cellMat); 
		SELF := le;
		
	END;
	
  outputt := JOIN(R1,R2,LEFT.id=RIGHT.id,Mulc(LEFT,RIGHT)); // Form all of the intermediate computations

OUTPUT(outputt,NAMED('outputt'));



//first make space


IDF := DATASET ([
{1},
{2}],
$.M_Types.IDRec);




$.M_Types.CellMatRec ParentLoad1($.M_Types.IDRec L) := TRANSFORM
    SELF.NumRows := 0;
		SELF.cellMat := IF (L.id=1,Patches2);//IF (L.id=1,Patches,[]) did not work, I dont know why
    SELF := L;
END;
//Ptbl := TABLE(NamesTable,DenormedRec);
Ptbl := PROJECT(IDF,ParentLoad1(LEFT));

OUTPUT(Ptbl,NAMED('Ptbl'));







//ready to apply ITERATE
// make a dataset like [{w1*a1},{w2},{w3},...,{wn}]
$.M_Types.CellMatRec fill($.M_Types.CellMatRec le,$.M_Types.CellMatRec ri) := TRANSFORM
		
		SELF.cellMat := IF (le.id=1,ML.Mat.Mul(le.cellMat,ri.cellMat),le.cellMat); 
		SELF := le;
		
	END;
	
  o2 := JOIN(R1,R2,(LEFT.id=RIGHT.id),fill(LEFT,RIGHT)); // Form all of the intermediate computations

OUTPUT(o2,NAMED('o2'));


//apply ITERATE

$.M_Types.CellMatRec T($.M_Types.CellMatRec L, $.M_Types.CellMatRec R, INTEGER C) := TRANSFORM
  SELF.cellmat := IF (C=1,R.cellmat,ML.Mat.Mul(R.cellMat,L.cellMat));
  SELF := R;
END;

MySet1 := ITERATE(o2,T(LEFT,RIGHT,COUNTER));

OUTPUT(MySet1,NAMED('MySet1'));




