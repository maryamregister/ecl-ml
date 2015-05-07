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
x := DATASET([
{1, 1, 3},
{2,1,4},
{3,1,5},
{4,1,6}],
Types.NumericField);

t := 1;

d := DATASET([
{1, 1, 0.3},
{2,1,0.4},
{3,1,0.5},
{4,1,0.6}],
Types.NumericField);
optn := DATASET([
{1, 1, 2},
{2,1,10},
{3,1,4},
{4,1,2}],
Types.NumericField);

f := 10;

g := DATASET([
{1, 1, 0.01},
{2,1,0.02},
{3,1,0.03},
{4,1,0.04}],
Types.NumericField);

gtd := 0.5;

a:= myfunc (net);
//OUTPUT (a);

//MinFunc(xout, net, myfunc, emptyC, emptyC, emptyC, 1,10,100, optn);
//OUTPUT (xout);
 WolfeLineSearch(wolfeout, x,t,d,f,g,gtd,1,2,3,0.001,myfunc,emptyC, emptyC, emptyC,0,0,0,0);

 OUTPUT (wolfeout, NAMED('Wolfe'));


// ArmijoBacktrack(ArmOut,net,1,net,7,net,5,0.001,0.00001,myfunc,emptyC, emptyC, emptyC,0,0,0,0);

 // OUTPUT (ArmOut);
// E := DATASET([], Mat.Types.Element);
// E2 := DATASET([{1,1,1,10}], Mat.Types.MUElement);
// Enno := Mat.MU.To (E,13);
// output (Enno + E2);
// ak:= FALSE;
// output((INTEGER)ak)
insuf := 1 ;
OUTPUT((BOOLEAN)insuf);