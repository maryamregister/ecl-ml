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
WolfeLineSearch(wolfeout, net,3,net,2,net,5,1,2,3,0.001,myfunc);
//OUTPUT (xout);
OUTPUT (wolfeout);

// E := DATASET([], Mat.Types.Element);
// E2 := DATASET([{1,1,1,10}], Mat.Types.MUElement);
// Enno := Mat.MU.To (E,13);
// output (Enno + E2);
