
EXPORT Produce_Random () := FUNCTION

G := 1000000;

R := (RANDOM()%G) / (REAL8)G;

RETURN R;

END;