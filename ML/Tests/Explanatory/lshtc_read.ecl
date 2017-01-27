IMPORT * FROM ML;
IMPORT STD;
IMPORT * FROM $;
IMPORT PBblas;
IMPORT STD;
Layout_Cell := PBblas.Types.Layout_Cell;
Layout_part := PBblas.Types.Layout_part;

//http://www.dabhand.org/ECL/construct_a_simple_bible_search.htm

// fileName := '~vherrara::datasets::sparsearfffile.arff';
fileName := '~vherrara::datasets::sentiment_75pct.arff';
filename2 := '~maryam::mytest::small_lshtc_train';
   InDS    := DATASET(fileName, {STRING Line}, CSV(SEPARATOR([])));
	 InDS2    := DATASET(filename2, {STRING Line}, CSV(SEPARATOR([])));
	 // InDS := DATASET ([{'{1043 1,10627 1,12296 1,16125 1,23131 1,35975 1,41713 1,50028 1,53812 1,58235 1,67653 1,78653 1,104679 1,107737 1,109735 0}'}],{STRING Line});
	 // InDS2 := DATASET ([{'144 10:202 56319:76'}, {'144 10:202 56319:76'}],{STRING Line});
   ParseDS := PROJECT(InDS, TRANSFORM({UNSIGNED RecID, STRING Line}, SELF.RecID:= COUNTER, SELF := LEFT));
	 ParseDS2 := PROJECT(InDS2, TRANSFORM({UNSIGNED RecID, STRING Line}, SELF.Line := LEFT.Line + ' '; SELF.RecID:= COUNTER, SELF := LEFT));
   //Parse the fields and values out
	 PATTERN lb := PATTERN('[0-9]')+;
   PATTERN ws       := ' ';
	 PATTERN cl := ':';
   PATTERN RecStart := '{';
   PATTERN ValEnd   := '}' | ',';
   PATTERN FldNum   := PATTERN('[0-9]')+;
   PATTERN DataQ    := '"' PATTERN('[ a-zA-Z0-9]')+ '"';
   PATTERN DataNQ   := PATTERN('[a-zA-Z0-9]')+;
   PATTERN DataVal  := DataQ | DataNQ;
   PATTERN FldVal   := OPT(RecStart) FldNum ws DataVal ValEnd;
	 PATTERN ValEnd2 := ws | '';
	 PATTERN FldVal2   := FldNum cl DataVal ws;
   OutRec := RECORD
     UNSIGNED RecID;
     STRING   FldName;
     STRING   FldVal;
   END;
   Types.NumericField XF(ParseDS L) := TRANSFORM
     SELF.id     := L.RecID;
     SELF.number := (TYPEOF(SELF.number))MATCHTEXT(FldNum) + 1;
     SELF.value  := (TYPEOF(SELF.value))MATCHTEXT(DataVal);
   END;
TrainDS :=  PARSE(ParseDS, Line, FldVal, XF(LEFT));

TrainDS2 :=  PARSE(ParseDS2, Line, FldVal2, XF(LEFT));
OUTPUT (InDS2, named ('InDS2'));
OUTPUT (InDS, named ('InDS'));
OUTPUT (ParseDS2, named ('ParseDS2'));
// OUTPUT (ParseDS2);
OUTPUT (TrainDS2, named ('TrainDS2'));
ParseDS3 := PROJECT(InDS2, TRANSFORM({UNSIGNED RecID, STRING Line}, SELF.Line := ''; SELF.RecID:= COUNTER, SELF := LEFT));
// OUTPUT (ParseDS3);

R2 := RECORD
	INTEGER lili := (TYPEOF(INTEGER)) STD.Str.GetNthWord (InDS2.Line,1);

END;
L := TABLE (InDS2, R2);
OUTPUT (L, named ('labels'));
OUTPUT (COUNT (L), named ('label_count'));
OUTPUT (MAX (L, lili), named ('label_max'));
OUTPUT (MIN (L, lili), named ('label_min'));

L_n := PROJECT (ParseDS2, TRANSFORM (Types.NumericField, SELF.id := LEFT.recid; SELF.number :=1; SELF.value := (TYPEOF(INTEGER8)) STD.Str.GetNthWord (LEFT.Line,1)));
OUTPUT (L_n, named ('L_n'));
// YB := utils.DistinctLabeltoGroundTruth (Y);
// OUTPUT (ParseDS2);

// OUTPUT (PROJECT (ParseDS2, TRANSFORM ({UNSIGNED RecID, INTEGER Line}, SELF.Line := (TYPEOF(INTEGER)) STD.Str.GetNthWord (LEFT.Line,1), SELF := LEFT)));