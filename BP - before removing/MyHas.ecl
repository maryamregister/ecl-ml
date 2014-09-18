// Matrix Properties
IMPORT * FROM ML;
IMPORT ML.Mat;
EXPORT MyHas(DATASET(ML.Mat.Types.Element) d) := MODULE

r := RECORD
  ML.Mat.Types.t_Index x := d.x ;
	ML.Mat.Types.t_Index y := 1;
	ML.Mat.Types.t_Value value := SUM(GROUP,d.value);
END;

// SumRow is a column vector containing the sum value of each row.
EXPORT SumRow := TABLE(d,r,d.x);

r := RECORD
  ML.Mat.Types.t_Index x := 1 ;
	ML.Mat.Types.t_Index y := d.y;
	ML.Mat.Types.t_Value value := SUM(GROUP,d.value);
END;

// SumCol is a row vector containing the sum value of each column.
EXPORT SumCol := TABLE(d,r,d.y);

END;