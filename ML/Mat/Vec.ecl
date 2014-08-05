IMPORT * FROM $;
// This module exists to handle a special sub-case of a matrix; the Vector
// The vector is really just a matrix with only one dimension
EXPORT Vec := MODULE

// Create a vector from 'thin air' - with N entries each of value def
EXPORT From(Types.t_Index N,Types.t_value def = 1.0) := FUNCTION
	seed := DATASET([{0,0,0}], Types.VecElement);
  PerCluster := ROUNDUP(N/CLUSTERSIZE);
	Types.VecElement addNodeNum(Types.VecElement L, UNSIGNED4 c) := transform
    SELF.x := (c-1)*PerCluster;
		SELF.y := 1;
		SELF.value := DEF;
  END;
	// Create eventual vector across all nodes (in case it is huge)
	one_per_node := DISTRIBUTE(NORMALIZE(seed, CLUSTERSIZE, addNodeNum(LEFT, COUNTER)), x DIV PerCluster);

	Types.VecElement fillRow(Types.VecElement L, UNSIGNED4 c) := TRANSFORM
		SELF.x := l.x+c;
		SELF.y := 1;
		SELF.value := def;
	END;
	// Now generate on each node; filter 'clips' of the possible extra '1' generated on some nodes
	m := NORMALIZE(one_per_node, PerCluster, fillRow(LEFT,COUNTER))(x <= N);
	RETURN m;

END;

// The 'To' routines - used to create a matrix from a vector
// Fill in the leading diagonal of a matrix starting with the vector element N
EXPORT ToDiag(DATASET(Types.VecElement) v, Types.t_index N=1) := PROJECT( v(x>=N), TRANSFORM(Types.Element,SELF.x:=LEFT.x-N+1,SELF.y:=LEFT.x-N+1,SELF := LEFT));

// Fill in the upper diagonal of a matrix starting with the vector element N
EXPORT ToLowerDiag(DATASET(Types.VecElement) v, Types.t_index N=1) := PROJECT( v(x>=N), TRANSFORM(Types.Element,SELF.x:=LEFT.x-N+2,SELF.y:=LEFT.x-N+1,SELF := LEFT));

// Fill in the Lower diagonal of a matrix starting with the vector element N
EXPORT ToUpperDiag(DATASET(Types.VecElement) v, Types.t_index N=1) := PROJECT( v(x>=N), TRANSFORM(Types.Element,SELF.x:=LEFT.x-N+1,SELF.y:=LEFT.x-N+2,SELF := LEFT));

// Fill in a column
EXPORT ToCol(DATASET(Types.VecElement) v,Types.t_index N) := PROJECT( v, TRANSFORM(Types.Element,SELF.x:=LEFT.x,SELF.y:=N,SELF := LEFT));

// Fill in a row
EXPORT ToRow(DATASET(Types.VecElement) v,Types.t_index N) := PROJECT( v, TRANSFORM(Types.Element,SELF.y:=LEFT.x,SELF.x:=N,SELF := LEFT));

// The 'Rep' routines - used to replace part of a matrix with a vector

EXPORT RepDiag(DATASET(Types.Element) M, DATASET(Types.VecElement) v, Types.t_index N=1) := M(X<>Y)+ ToDiag(v, N);

EXPORT RepCol(DATASET(Types.Element) M,DATASET(Types.VecElement) v,Types.t_index N) := M(Y<>N)+ToCol(v, N);

EXPORT RepRow(DATASET(Types.Element) M,DATASET(Types.VecElement) v,Types.t_index N) := M(X<>N)+ToRow(v, N);

// The 'From' routines - extract a vector from part of a matrix
// FromDiag returns a vector formed from the elements of the Kth diagonal of M
EXPORT FromDiag(DATASET(Types.Element) M, INTEGER4 K=0) := PROJECT( M(x=y-k), TRANSFORM(Types.VecElement,SELF.x:=IF(K<0,LEFT.y,LEFT.x),SELF.y:=1,SELF := LEFT));

EXPORT FromCol(DATASET(Types.Element) M,Types.t_index N) := PROJECT( M(Y=N), TRANSFORM(Types.VecElement,SELF.x:=LEFT.x,SELF.y:=1,SELF := LEFT));

EXPORT FromRow(DATASET(Types.Element) M,Types.t_index N) := PROJECT( M(X=N), TRANSFORM(Types.VecElement,SELF.x:=LEFT.y,SELF.y:=1,SELF := LEFT));

// Vector math
// Compute the dot product of two vectors
EXPORT Dot(DATASET(Types.VecElement) X,DATASET(Types.VecElement) Y) := FUNCTION
  J := JOIN(x,y,LEFT.x=RIGHT.x,TRANSFORM(Types.VecElement,SELF.x := LEFT.x, SELF.value := LEFT.value*RIGHT.value, SELF:=LEFT));
	RETURN SUM(J,value);
END; 

EXPORT Norm(DATASET(Types.VecElement) X) := SQRT(Dot(X,X));

EXPORT Length(DATASET(Types.VecElement) X) := Has(X).Dimension;




// This function adds elements of a matrix with elements of a vector. If R_C=1 then the summation is done per 
//row, it means that the elements of row 1 are added to the elemnet 1 of the vector, the elements of row 2 are added
//to the element 2 of the vector and so on. If R_C=2 then the summation is done colimn-wise

EXPORT Add_Mat_Vec(DATASET(Types.Element) l,DATASET(Types.VecElement) r, R_C) := FUNCTION

StatsL := Has(l).Stats;
StatsR := Has(r).Stats;
SizeMatch :=  (R_C=1 AND StatsL.XMax=StatsR.XMax) OR (R_C=2 AND StatsL.YMax=StatsR.XMax);


checkAssert := ASSERT(SizeMatch, 'Add FAILED - Size mismatch', FAIL);	

Types.Element Ad(l le,r ri) := TRANSFORM
    SELF.x := le.x;
    SELF.y := le.y;
	  SELF.value := le.value + ri.value; // Fortuitously; 0 is the null value
  END;

Result :=  IF(R_C=1,JOIN(l,r,LEFT.x=RIGHT.x,Ad(LEFT,RIGHT)),JOIN(l,r,LEFT.y=RIGHT.x,Ad(LEFT,RIGHT))); 

RETURN WHEN(Result, checkAssert);


END;





// This function multiplies elements of a matrix with elements of a vector. If R_C=1 then the multiplication is done per 
//row, it means that the elements of row 1 are multiplied by the elemnet 1 of the vector, the elements of row 2 are multiplied
//by the element 2 of the vector and so on. If R_C=2 then the multiplication is done column-wise
// test file : 
EXPORT Mul_Mat_Vec(DATASET(Types.Element) l,DATASET(Types.VecElement) r, R_C) := FUNCTION

StatsL := Has(l).Stats;
StatsR := Has(r).Stats;
SizeMatch :=  (R_C=1 AND StatsL.XMax=StatsR.XMax) OR (R_C=2 AND StatsL.YMax=StatsR.XMax);


checkAssert := ASSERT(SizeMatch, 'Add FAILED - Size mismatch', FAIL);	

Types.Element M(l le,r ri) := TRANSFORM
    SELF.x := le.x;
    SELF.y := le.y;
	  SELF.value := le.value * ri.value; 
  END;

Result :=  IF(R_C=1,JOIN(l,r,LEFT.x=RIGHT.x,M(LEFT,RIGHT)),JOIN(l,r,LEFT.y=RIGHT.x,M(LEFT,RIGHT))); 

RETURN WHEN(Result, checkAssert);


END;

END;