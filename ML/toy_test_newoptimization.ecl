IMPORT * FROM ML;
IMPORT STD;
IMPORT * FROM $;
IMPORT STD;
IMPORT PBblas;
Layout_Cell := PBblas.Types.Layout_Cell;
// test Sparse_Autoencoder_lbfgs on an MNIST dataset which contains only five digits {0,1,2,3,4} : workunit W20160724-100930
param_vec := DATASET ([
{1,1,11},
{1,2,22},
{1,3,23},
{1,4,52},
{1,5,72},
{1,6,92},
{1,7,28},
{1,8,278},
{1,9,62},
{1,10,62},
{1,11,278},
{1,12,27},
{1,13,525},
{1,14,234},
{1,15,24},
{1,16,25},
{1,17,727},
{1,18,287},
{1,19,263},
{1,20,32},
{1,21,22}

],ML.Mat.Types.Element);
m:=21;
m_part := 5;
parammap := PBblas.Matrix_Map(1,m,1,m_part);
paramdist := DMAT.Converted.FromElement(param_vec,parammap);
cost_param := DATASET([
    {1,1,m},
    {2,1,m_part}
    ], Types.NumericField);
		

emptyL := DATASET([], PBblas.Types.Layout_part);
fun_re := myfunc_new ( paramdist, cost_param, emptyL , emptyL);
LBFGS_MAXitr := 1;
LBFGS_corrections := 3;
paramnumber := m;
lbfgs_result := Optimization_new (0, 0, 0, 0).MinFUNC(paramdist, cost_param, emptyL , emptyL,myfunc_new, paramnumber,LBFGS_MAXitr, 0.00001, 0.000000001,  1000, LBFGS_corrections, 0, 0, 0,0) ;
realnode_rec := RECORD (PBblas.Types.MUElement)
UNSIGNED real_node := 0;
END;
funre_ := PROJECT (fun_re, TRANSFORM(realnode_rec, SELF.real_node := STD.System.Thorlib.Node(); SELF := LEFT));
//OUTPUT(PROJECT (lbfgs_result, TRANSFORM(realnode_rec,SELF.real_node := STD.System.Thorlib.Node(); SELF := LEFT)));
OUTPUT(lbfgs_result);
// aa := [1,2,3];
// bb := repeatbias(3, 4, aa);
// OUTPUT(bb);
