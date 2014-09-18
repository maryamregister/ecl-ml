IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;

d1 := DATASET([{1,1,1},{1,2,0},{2,1,3},{2,2,4},{3,1,4},{3,2,5}],ML.Mat.Types.Element);
OUTPUT(d1, ALL, NAMED('d1'));

Num := 5;

Result := $.Add_Mat_num (d1,Num);
OUTPUT(Result, ALL, NAMED('Result'));