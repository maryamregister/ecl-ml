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

MinFunc(xout, net, myfunc, emptyC, emptyC, emptyC, 1, optn);

OUTPUT (xout);