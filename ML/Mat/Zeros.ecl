IMPORT * FROM ML.Mat;

EXPORT Zeros(UNSIGNED Nrows, UNSIGNED NCols) := FUNCTION

Types.Element Z(UNSIGNED4 c, UNSIGNED4 NumRows) := TRANSFORM
      SELF.x := ((c-1) % NumRows) + 1;
      SELF.y := ((c-1) DIV NumRows) + 1;
      SELF.value := 0;
     END;
m := Nrows * NCols;
RETURN DATASET(m, Z(COUNTER, Nrows),DISTRIBUTED);
END;
