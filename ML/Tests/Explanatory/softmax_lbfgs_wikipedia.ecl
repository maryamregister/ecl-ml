IMPORT * FROM ML;
IMPORT STD;
IMPORT * FROM $;
IMPORT PBblas;
IMPORT STD;
IMPORT std.system.Thorlib;
Layout_Cell := PBblas.Types.Layout_Cell;
Layout_part := PBblas.Types.Layout_part;


// filename2 := '~maryam::mytest::small_lshtc_train';
 // filename2 := '~maryam::mytest::small_lshtc_train_task3.txt';
// filename2:= '~online::maryam::wikipedia_task1_train.txt'; // 347256 number of features, 12294 labels, 93805 samples
filename2:= '~online::maryam::wikipediamediumpreproclshtcv3-train.txt';

InDS2    := DATASET(filename2, {STRING Line}, CSV(SEPARATOR([])));
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
Types.NumericField XF(ParseDS2 L) := TRANSFORM
 SELF.id     := L.RecID;
 SELF.number := (TYPEOF(SELF.number))MATCHTEXT(FldNum) + 1;
 SELF.value  := (TYPEOF(SELF.value))MATCHTEXT(DataVal);
END;
TrainDS2 :=  PARSE(ParseDS2, Line, FldVal2, XF(LEFT));
OUTPUT (InDS2, named ('InDS2'));
OUTPUT (ParseDS2, named ('ParseDS2'));
L_n := PROJECT (ParseDS2, TRANSFORM (Types.DiscreteField, SELF.id := LEFT.recid; SELF.number :=1; SELF.value := (TYPEOF(INTEGER4)) STD.Str.GetNthWord (LEFT.Line,1)));
OUTPUT (L_n, named ('L_n'));
OUTPUT (MAX (L_n,value), named ('maxLN'));
indepDataC := utils.DistinctFeaturest(TrainDS2);

OUTPUT (COUNT (DEDUP (SORT (indepDataC, id),id)), named ('samplenumber'));
OUTPUT (COUNT (DEDUP (SORT (indepDataC, number),number)), named ('featureenumber'));
OUTPUT (MAX (indepDataC,indepDataC.number), named ('numberMax'));
OUTPUT (MIN (indepDataC,indepDataC.number), named ('numberMin'));
OUTPUT (MAX (indepDataC,indepDataC.id), named ('IDMax'));
OUTPUT (MIN (indepDataC,indepDataC.id), named ('IDMin'));

// OUTPUT (TrainDS2, named ('TrainDS2'));
// OUTPUT (indepDataC, named ('indepDataC'));
// OUTPUT (MIN (indepDataC, value), named ('indepDataCMin'));
// OUTPUT (MAX (indepDataC, value), named ('indepDataCMax'));
// OUTPUT (COUNT (indepDataC), named ('CountindepDataC'));
depDataC := L_n;

//initialize THETA
Numclass := COUNT (DEDUP (SORT (depDataC, value),value)); // number of distinct values represents the numebr of classes
OUTPUT  (Numclass, NAMED ('Numclass'));
InputSize := COUNT (DEDUP (SORT (indepDataC, number),number));
numsamples := MAX (indepDataC, indepDataC.id);
OUTPUT (InputSize, named ('InputSize'));
OUTPUT (MAX (indepDataC,number),named('maxnumber'));
OUTPUT (MIN (indepDataC,number),named('minnumber'));
OUTPUT (numsamples, named('numsamples'));


/*
//Set Parameters
LoopNum := 200; // Number of iterations in softmax algortihm
LAMBDA := 0.0001; // weight decay parameter in  claculation of SoftMax Cost fucntion


 UNSIGNED corr := 10;
prows := 31;
// prows := 41;
// prows := 49;
// prows := 61;
pcols := 1;
// T1 := ML.Utils.distrow_ranmap(12294, 347256, prows ) ;

 // OUTPUT  (T1,  NAMED ('T1'));
 
// IntTHETA := Mat.Scale (T1,0.005);
// IntTHETA := PROJECT (T1, TRANSFORM (Mat.Types.Element, SELF.value := LEFT.value * 0.005; SELF := LEFT), LOCAL);
IntTHETA := DATASET ([],Mat.Types.Element);
// OUTPUT  (IntTHETA,  NAMED ('IntTHETA'));
//SoftMax_Sparse Classfier

// OUTPUT (MAX (IntTHETA,IntTHETA.x));

// OUTPUT (MAX (IntTHETA, IntTHETA.y));

// OUTPUT (COUNT (IntTHETA));


depDataC_distinc := Utils.DistinctLabel( depDataC);
OUTPUT (depDataC_distinc, named ('depDataC_distinc'));
OUTPUT (MAX (depDataC_distinc, depDataC_distinc.value));

*/