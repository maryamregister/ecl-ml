IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;

W:= DATASET ([{1,1,2},{1,1,2},{1,1,2},{1,1,2}],$.M_types.MatRecord);
OUTPUT (W,ALL,NAMED('W'));

cal (REAL8 num) := FUNCTION

RETURN num*2;
END;


loopBody(DATASET($.M_types.MatRecord) ds, unsigned4 c) :=
 PROJECT(ds,
    TRANSFORM($.M_types.MatRecord,
    SELF.x := cal(LEFT.x);
    SELF := LEFT));
		
		
		
		OUTPUT(LOOP(W,
  COUNTER <= 5,
  loopBody(ROWS(LEFT), COUNTER)));

