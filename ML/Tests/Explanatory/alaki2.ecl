IMPORT * FROM ML;
IMPORT STD;
IMPORT * FROM $;
IMPORT PBblas;
IMPORT STD;
Layout_Cell := PBblas.Types.Layout_Cell;
Layout_part := PBblas.Types.Layout_part;
IMPORT std.system.Thorlib;

fileName := '~vherrara::datasets::sparsearfffile.arff';


InDS    := DATASET([{'{1043 1,10627 1,12296 1,16125 1,23131 1,35975 1,41713 1,50028 1,53812 1,58235 1,67653 1,78653 1,104679 1,107737 1,109735 0}'}, {'{1043 1,10627 1,12296 1,16125 1,23131 1,35975 1,41713 1,50028 1,53812 1,58235 1,67653 1,78653 1,104679 1,107737 1,109735 0}'}], {STRING Line});
   InDS2    := DATASET(fileName, {STRING Line}, CSV(SEPARATOR([])));
   ParseDS := PROJECT(InDS, TRANSFORM({UNSIGNED RecID, STRING Line}, SELF.RecID:= COUNTER, SELF := LEFT), LOCAL);
   //Parse the fields and values out
   PATTERN ws       := ' ';
   PATTERN RecStart := '{';
   PATTERN ValEnd   := '}' | ',';
   PATTERN FldNum   := PATTERN('[0-9]')+;
   PATTERN DataQ    := '"' PATTERN('[ a-zA-Z0-9]')+ '"';
   PATTERN DataNQ   := PATTERN('[a-zA-Z0-9]')+;
   PATTERN DataVal  := DataQ | DataNQ;
   PATTERN FldVal   := OPT(RecStart) FldNum ws DataVal ValEnd;
   OutRec := RECORD
     UNSIGNED RecID;
     STRING   FldName;
     STRING   FldVal;
   END;
   Types.DiscreteField XF(ParseDS L) := TRANSFORM
     SELF.id     := L.RecID;
     SELF.number := (TYPEOF(SELF.number))MATCHTEXT(FldNum) + 1;
     SELF.value  := (TYPEOF(SELF.value))MATCHTEXT(DataVal);
   END;
   kk :=  PARSE(ParseDS, Line, FldVal, XF(LEFT));

OUTPUT (InDS);
OUTPUT (ParseDS);
OUTPUT (kk);
OUTPUT (kk(Number=109736));