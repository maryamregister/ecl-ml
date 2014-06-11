IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;

//define parameters
VisibleSize := 4;

HiddenSize := 2;

SparsityParam := 0.01;

lambda := 0.0001;

beta := 3;


MatRecord := ML.Mat.Types.Element;

VecRecord := ML.Mat.Types.VecElement;
//define the data, all elements must be between 0 and 1

Patches := DATASET([
{1,1,0.1},
{1,2,0.9},
{1,3,0.6},
{2,1,0.4},
{2,2,0.3},
{2,3,0.1},
{3,1,0.2},
{3,2,0.4},
{3,3,0.7},
{4,1,0.4},
{4,2,0.5},
{4,3,0.3}],
MatRecord);
OUTPUT(Patches, ALL, NAMED('Patches'));
//define weights and bias values , the weights must be initialized randomly
// W1 = rand (hiddensize, visiblesize)
W1 := DATASET ([
{1,1,0.1},
{1,2,0.2},
{1,3,0.1},
{1,4,0.1},
{2,1,0.2},
{2,2,0.1},
{2,3,0.2},
{2,4,0.1}],
MatRecord);
OUTPUT(W1, ALL, NAMED('W1'));
// W2 = rand (visiblesize, hiddensize)
W2 := DATASET ([
{1,1,0.1},
{1,2,0.1},
{2,1,0.2},
{2,2,0.2},
{3,1,0.1},
{3,2,0.2},
{4,1,0.1},
{4,2,0.2}],
MatRecord);
OUTPUT(W2, ALL, NAMED('W2'));
//b1 = zeros (hidensize);
b1 := DATASET ([
{1,1,0},
{2,1,0}],
VecRecord);
OUTPUT(b1, ALL, NAMED('b1'));
//b2 = zeros (visiblesize);
b2 := DATASET ([
{1,1,0},
{2,1,0},
{3,1,0},
{4,1,0}],
VecRecord);
OUTPUT(b2, ALL, NAMED('b2'));

x := Patches;

//z2 = W1 * x + repmat(b1,1,m);

z2_t := ML.Mat.Mul(W1,x);
OUTPUT(z2_t, ALL, NAMED('z2_t'));

z2   := $.Add_Mat_Vec (z2_t,b1,1);

OUTPUT(z2, ALL, NAMED('z2'));
a2   := $.sigmoid (z2);
OUTPUT(a2, ALL, NAMED('a2'));


// z3 = W2 * a2 + repmat(b2,1,m);

z3_t := ML.Mat.Mul(W2,a2);
OUTPUT(z3_t, ALL, NAMED('z3_t'));

z3   := $.Add_Mat_Vec (z3_t,b2,1);
OUTPUT(z3, ALL, NAMED('z3'));

a3   := $.sigmoid (z3);
OUTPUT(a3, ALL, NAMED('a3'));


RhoHat := ML.Mat.Has(a2).MeanRow;
OUTPUT(RhoHat, ALL, NAMED('RhoHat'));


y := x; 

// d3=-(y-a3).*(a3.*(1-a3));

y_a3 := -1 * ML.Mat.Sub (y,a3);
a3_1 := $.Add_Mat_Num (-1 * a3,1);
a3_  := $.Mul_ElementWise (a3,a3_1);
d3   := $.Mul_ElementWise (y_a3,a3_);
