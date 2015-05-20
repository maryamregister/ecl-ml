IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat;
IMPORT * FROM ML.Types;
emptyC := DATASET([], Types.NumericField);

xx := DATASET([
{1, 1, 30},
{2,1,49},
{3,1,57},
{4,1,16}],
Types.NumericField);

dd := DATASET([
{1, 1, -1},
{2,1,4},
{3,1,5},
{4,1,3.5}],
Types.NumericField);

p_um := MAX (xx,id);
fg := myfunc2(xx, emptyC, emptyC, emptyC);
gg := fg (id <= p_um);
ff := fg (id = p_um+1)[1].value;
OUTPUT (xx, NAMED ('xx'));
OUTPUT (gg, NAMED ('gg'));
OUTPUT (ff,NAMED('ff'));
OUTPUT (dd,NAMED('dd'));
gtdT := ML.Mat.Mul (ML.Mat.Trans(ML.Types.ToMatrix(gg)),ML.Types.ToMatrix(dd));
gtdd := gtdT[1].value;
OUTPUT (gtdd,NAMED('gtdd'));
step:=1;
//WolfeLineSearch(wolfeout, x,t,d,f,g,gtd,0.0001,0.9,25,0.000000001,myfunc2,emptyC, emptyC, emptyC,0,0,0,0);
 wolfeout := WolfeLineSearch(xx,step,dd,ff,gg,gtdd,0.0001,0.9,3,0.000000001,emptyC, emptyC, emptyC,myfunc2,0,0,0,0);
//OUTPUT(wolfeout);

O := Optimization (0, 0, 0, 0).WolfeLineSearch(xx,step,dd,ff,gg,gtdd,0.0001,0.9,3,0.000000001,emptyC, emptyC, emptyC,myfunc2,0,0,0,0);
//OUTPUT(O);
OUTPUT(0)