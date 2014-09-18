IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;

d1 := DATASET([{1,1,1},{1,2,0},{2,1,3},{2,2,4},{3,1,4},{3,2,5}],ML.Mat.Types.Element);
OUTPUT(d1, ALL, NAMED('d1'));

d2 := DATASET([{1,1,10},{1,2,10},{2,1,13},{2,2,14},{3,1,5},{3,2,1}],ML.Mat.Types.Element);
OUTPUT(d2, ALL, NAMED('d2'));

Results := $.Mul_ElementWise (d1,d2);
OUTPUT(Results, ALL, NAMED('Results'));