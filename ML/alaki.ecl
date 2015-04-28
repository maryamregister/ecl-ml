IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat;
IMPORT * FROM ML.Types;
emptyC := DATASET([], Types.NumericField);
net := DATASET([
{1, 1, 3},
{2,1,4},
{3,1,5},
{4,1,6}],
Types.NumericField);


optn := DATASET([
{1, 1, 2},
{2,1,10},
{3,1,4},
{4,1,2}],
Types.NumericField);

a:= myfunc (net);
//OUTPUT (a);

//MinFunc(xout, net, myfunc, emptyC, emptyC, emptyC, 1,10,100, optn);
//OUTPUT (xout);
 WolfeLineSearch(wolfeout, net,3,net,2,net,5,1,2,3,0.001,myfunc,emptyC, emptyC, emptyC,0,0,0,0);

 OUTPUT (wolfeout);


// ArmijoBacktrack(ArmOut,net,1,net,7,net,5,0.001,0.00001,myfunc,emptyC, emptyC, emptyC,0,0,0,0);

 // OUTPUT (ArmOut);
// E := DATASET([], Mat.Types.Element);
// E2 := DATASET([{1,1,1,10}], Mat.Types.MUElement);
// Enno := Mat.MU.To (E,13);
// output (Enno + E2);
ak:= FALSE;
output((INTEGER)ak)