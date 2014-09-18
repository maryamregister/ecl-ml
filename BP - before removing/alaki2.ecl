IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;

Label := DATASET([
{1,1,1},
{1,2,0},
{1,3,0},
{2,1,0},
{2,2,0},
{2,3,1},
{3,1,0},
{3,2,1},
{3,3,0}],
$.M_Types.MatRecord);

P := DATASET([{1}],$.M_Types.IDRec);

OUTPUT (P,ALL,NAMED('p')); 

M := DATASET([
{1,1,1,1},
{1,1,2,0},
{1,1,3,0},
{1,2,1,0},
{1,2,2,0},
{1,2,3,1},
{1,3,1,0},
{1,3,2,1},
{1,3,3,0}],
$.M_types.IDMatRec);


A:= $.Cell( P, M);
OUTPUT (A,ALL,NAMED('A')); 
 
