IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;


//first check the steps of implementation
d1 := DATASET([
{1,1,1},
{2,1,20},
{3,1,3},
{4,1,4},
{1,2,1},
{2,2,3},
{3,2,5},
{4,2,5},
{1,3,60},
{2,3,1},
{3,3,2},
{4,3,3}



],ML.Mat.Types.Element);
OUTPUT(d1, ALL, NAMED('d1'));


numrow := MAX (d1,d1.x);
OUTPUT(numrow, ALL, NAMED('numrow'));

S := SORT (d1, y,value);

OUTPUT(S, ALL, NAMED('S'));


SequencedS := RECORD
  $.M_types.MatRecord;
  INTEGER8 Sequence := 0;
END;

SequencedS AddSequence(S l, INTEGER c) := TRANSFORM
  SELF.Sequence := c%numrow;
  SELF := l;
END;

SequencedSRecs := PROJECT(S,
          AddSequence(LEFT,COUNTER));
					
OUTPUT(SequencedSRecs, ALL, NAMED('SequencedSRecs'));

out := SequencedSRecs (SequencedSRecs.Sequence=0);
OUTPUT(out, ALL, NAMED('out'));


$.M_types.MatRecord makematrix(SequencedS l) := TRANSFORM
  SELF.x := 1;
	SELF.value :=  l.x;
  SELF.y := l.y;
END;

final_result := PROJECT (out, makematrix(LEFT));

OUTPUT(final_result, ALL, NAMED('final_result'));


//second check the function iteself, which is ML.Mat.has.MaxColIndex


fun_out := ML.Mat.Has(d1).MaxColIndex;

OUTPUT(fun_out, ALL, NAMED('fun_out'));