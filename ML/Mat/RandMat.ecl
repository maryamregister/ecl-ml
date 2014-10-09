IMPORT * FROM ML.Mat;



Produce_Random () := FUNCTION

G := 1000000;

R := (RANDOM()%G) / (REAL8)G;

RETURN R;

END;



//return a matrix with "Nrows" number of rows and "NCols" number of cols. The  matrix is initilize with random numbers
EXPORT RandMat (UNSIGNED Nrows, UNSIGNED NCols) := FUNCTION

//UniRange :=  ML.Distribution.Uniform(0,1);
ONE := DATASET ([{1,1,0}],Types.Element);

MatZero := Mat.Repmat (ONE, Nrows, NCols); // matrix of zeros of the size NRows*NCols

//replce zero values in MatZero with random generated values
Types.Element RandomizeMat(Types.Element l) := TRANSFORM


// the commented code bellow produce the same number at each round
	// b1 := ML.Distribution.GenData(1,UniRange,1);
	// r1 := MAX (b1,b1.value);
	r1 := Produce_Random();

	Self.Value := r1;
	Self := l;
END;


Result := PROJECT (MatZero,RandomizeMat(LEFT) );
RETURN Result;

END;