IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat; 
IMPORT * FROM ML.Types;
IMPORT PBblas;
d := DATASET ([{1,2,1.175494e-10},{1,2,1.175494e-14},{1,2,1.175494e-14}],Types.NumericField4);
d2 := DATASET ([{1,2,16777216},{1,2,1},{1,2,1}],Types.NumericField4);
// OUTPUT (SUM (d2,d2.value));
 unsigned m := 2;
 output (1.0/m);
// output (d);
// REAL4 sd := SUM (d,d.value);
// OUTPUT (sd,named ('sd'));

// REAL4 sd2 := SUM (d2,d2.value);
// OUTPUT (sd2,named ('sd2'));
// f := sd;
// gtd := sd2;
// REAL4 result := f-(f+ (((REAL4)((REAL4)1.0/(real4)(939578.9))) * gtd)); 
// OUTPUT (result, named ('result'));

