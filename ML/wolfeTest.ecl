IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat;
IMPORT * FROM ML.Types;
emptyC := DATASET([], Types.NumericField);
//x=[15, 80, 40, 39];
x := DATASET([
{1, 1, 15},
{2,1,80},
{3,1,40},
{4,1,39}],
Types.NumericField);

t:=1;

//d=[0.1, 0.4, -0.2, 0.8];
d := DATASET([
{1, 1, 0.1},
{2,1,0.4},
{3,1,-0.2},
{4,1,0.8}],
Types.NumericField);

p_um := MAX (x,id);

//[f g]= myfunc2(x);
fg := myfunc2(x, emptyC, emptyC, emptyC);
g := fg (id <= p_um);
f := fg (id = p_um+1)[1].value;


OUTPUT (x, NAMED ('xx'));
OUTPUT (g, NAMED ('gg'));
OUTPUT (f,NAMED('ff'));
OUTPUT (d,NAMED('dd'));


//gtd = d'*d;
gtdT := ML.Mat.Mul (ML.Mat.Trans(ML.Types.ToMatrix(g)),ML.Types.ToMatrix(d));
gtd := gtdT[1].value;
OUTPUT (gtd,NAMED('gtd'));

//WolfeLineSearch(wolfeout, x,t,d,f,g,gtd,0.0001,0.9,25,0.000000001,myfunc2,emptyC, emptyC, emptyC,0,0,0,0);
 //wolfeout := WolfeLineSearch(x,t,d,f,g,gtd,0.0001,0.9,3,0.000000001,emptyC, emptyC, emptyC,myfunc2,0,0,0,0);
//OUTPUT(wolfeout);

WResult := Optimization (0, 0, 0, 0).WolfeLineSearch(x,t,d,f,g,gtd,0.0001,0.9,10,0.000000001,emptyC, emptyC, emptyC,myfunc2,0,0,0,0);
OUTPUT(WResult,NAMED('WResult'));
// funresult := myfunc2 ( x, emptyC, emptyC , emptyC);
// OUTPUT(funresult, NAMED('funresults'));
