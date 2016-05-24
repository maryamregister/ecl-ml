
IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat;
IMPORT * FROM ML.Types;
IMPORT PBblas;
Layout_Cell := PBblas.Types.Layout_Cell;
Layout_Part := PBblas.Types.Layout_Part;

emptyC := DATASET([], Types.NumericField);
//x=[15, 80, 40, 39];
// x := DATASET([
// {1, 1, 0.01},
// {2,1,0.01},
// {3,1,0.01},
// {4,1,0.01}],
// Types.NumericField);



   
   
   
x := DATASET(
[{1,1,1},
{2,1,2},
{3,1,3},
{4,1,4},
{5,1,5},
{1,2,6},
{2,2,7},
{3,2,8},
{4,2,9},
{5,2,10}],Types.NumericField);

x2 := DATASET(
[{1,1,1},
{2,1,2},

{1,2,3},
{2,2,4},

{1,3,5},
{2,3,6},

{1,4,7},
{2,4,8},

{1,5,9},
{2,5,10}

],Types.NumericField);


  
param_map := PBblas.Matrix_Map(5,2,2,1);
param_map2 := PBblas.Matrix_Map(2,5,2,1);
xdist := ML.DMat.Converted.FromNumericFieldDS(x, param_map);
x2dist := ML.DMat.Converted.FromNumericFieldDS(x2, param_map2);
output(x);
output(xdist);
output(x2);
output(x2dist);