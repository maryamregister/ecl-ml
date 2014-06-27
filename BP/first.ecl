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

y_a3    := ML.Mat.Sub (y,a3);
a3_     := $.Mul_Mat_Num (a3,-1);
a3_1    := $.Add_Mat_Num (a3_,1);
a3_a3_1 := $.Mul_ElementWise (a3, a3_1);
d3_     := $.Mul_ElementWise (y_a3,a3_a3_1);
d3      := $.Mul_Mat_Num (d3_,-1);
OUTPUT(d3, ALL, NAMED('d3'));


//sparsity_delta=((-sparsityParam./rhohat)+((1-sparsityParam)./(1.-rhohat)));


term1          := $.Div_Num_Mat (RhoHat,-1*SparsityParam);
RhoHat_        := $.Mul_Mat_Num (RhoHat,-1);
RhoHat_1       := $.Add_Mat_Num (RhoHat_,1);
term2          := $.Div_Num_Mat (RhoHat_1,(1-SparsityParam));
Sparsity_Delta := ML.Mat.Add (term1,term2);
OUTPUT(Sparsity_Delta, ALL, NAMED('Sparsity_Delta'));


//d2=((W2'*d3)+beta*repmat(sparsity_delta,1,m)).*(a2.*(1-a2));  Trans

W2_T      					:= ML.Mat.Trans (W2);
OUTPUT(W2_T, ALL, NAMED('W2_T'));
W2_T_d3   					:= ML.Mat.Mul (W2_T,d3);
OUTPUT(W2_T_d3, ALL, NAMED('W2_T_d3'));
Sparsity_Delta_Beta := $.Mul_Mat_Num (Sparsity_Delta,Beta);
OUTPUT(Sparsity_Delta_Beta, ALL, NAMED('Sparsity_Delta_Beta'));
d2_term1 						:= $.Add_Mat_Vec (W2_T_d3,Sparsity_Delta_Beta,1);
OUTPUT(d2_term1, ALL, NAMED('d2_term1'));
a2_      					  := $.Mul_Mat_Num (a2,-1);
a2_1     					  := $.Add_Mat_Num (a2_,1);
a2_a2_1  					  := $.Mul_ElementWise (a2, a2_1);
OUTPUT(a2_a2_1, ALL, NAMED('a2_a2_1'));

d2       					  := $.Mul_ElementWise (d2_term1,a2_a2_1);
OUTPUT(d2, ALL, NAMED('d2'));



// W1delta=d2*x';
// W2delta=d3*a2';
// b1delta=sum(d2,2);
// b2delta=sum(d3,2);

OUTPUT(x, ALL, NAMED('x'));
x_T       := ML.Mat.Trans (x);
OUTPUT(x_T, ALL, NAMED('x_T'));

W1delta		:= ML.Mat.Mul (d2,x_T);
OUTPUT(W1delta, ALL, NAMED('W1delta'));
a2_T      := ML.Mat.Trans (a2);
W2delta		:= ML.Mat.Mul (d3,a2_T);
OUTPUT(W2delta, ALL, NAMED('W2delta'));

b1delta   := $.MyHas(d2).SumRow;
OUTPUT(b1delta, ALL, NAMED('b1delta'));
b2delta   := $.MyHas(d3).SumRow;
OUTPUT(b2delta, ALL, NAMED('b2delta'));


// squared_error_cost=sum(0.5*sum((x-a3).^2));

x_a3   := ML.Mat.Sub (x,a3);
x_a3_2 := $.Pow_Each_El (x_a3,2);

OUTPUT(x_a3_2, ALL, NAMED('x_a3_2'));

Sum_x_a3_2  := $.MyHas(x_a3_2).SumCol;


Sum_x_a3_2_5 := $.Mul_Mat_Num (Sum_x_a3_2,0.5);

OUTPUT(Sum_x_a3_2_5, ALL, NAMED('Sum_x_a3_2_5'));

Squared_Error_Cost := $.MyHas(Sum_x_a3_2_5).SumRow;
OUTPUT(Squared_Error_Cost, ALL, NAMED('Squared_Error_Cost'));



// cost=(1/m)*squared_error_cost+(lambda/2)*(sum(W2(:).^2)+sum(W1(:).^2))+beta*sum(KL(sparsityParam,rhohat));
// W1grad=(1/m)*W1delta+lambda*W1;
// W2grad=(1/m)*W2delta+lambda*W2;
// b1grad=(1/m)*b1delta;
// b2grad=(1/m)*b2delta;

