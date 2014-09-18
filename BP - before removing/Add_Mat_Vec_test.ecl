IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;

d1 := DATASET([{1,1,1},{1,2,0},{2,1,3},{2,2,4},{3,1,4},{3,2,5}],ML.Mat.Types.Element);
OUTPUT(d1, ALL, NAMED('d1'));


a1 := DATASET([{1,1,1},{2,1,2},{3,1,3}],ML.Mat.Types.VecElement);
OUTPUT(a1, ALL, NAMED('a1'));

a2 := DATASET([{1,1,1},{2,1,2}],ML.Mat.Types.VecElement);
OUTPUT(a2, ALL, NAMED('a2'));




Re1 := $.Add_Mat_Vec (d1,a1,1);
OUTPUT(Re1, ALL, NAMED('RE_row'));


Re2 := $.Add_Mat_Vec (d1,a2,2);
OUTPUT(Re2, ALL, NAMED('RE_col'));
