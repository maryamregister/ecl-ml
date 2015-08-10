
IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat;
IMPORT * FROM ML.Types;
IMPORT PBblas;


   
   
   
   
   lbg := DATASET([
{1, 1,  -0.3604},
{2,1,-0.2707},
{3,1, 0.0872},
{4,1,-0.9317}],
Mat.Types.Element);

 
    
   
   
   
   old_dir0 := DATASET([
{1,1,0},
{2,1,0},
{3,1,0},
{4,1,0},
{1,2, 2.6474},
{2,2,0.3847},
{3,2, 2.3242},
{4,2,-0.9292}],
Mat.Types.Element);

 
    
    
    
    
     old_step0 := DATASET([
{1,1,0},
{2,1,0},
{3,1,0},
{4,1,0},
{1,2, 1.1200},
{2,2,0.3810},
{3,2, 0.5797},
{4,2,0.6651}],
Mat.Types.Element);

old_dir10 := DATASET([

{1,1, 2.6474},
{2,1,0.3847},
{3,1, 2.3242},
{4,1,-0.9292}],
Mat.Types.Element);

old_step10 := DATASET([

{1,1, 1.1200},
{2,1,0.3810},
{3,1, 0.5797},
{4,1,0.6651}],
Mat.Types.Element);



HDIG := DATASET([
{1,1, 1.7636}],
Mat.Types.Element);
lb := Optimization (0, 0, 0, 0).Limited_Memory_BFGS (4, 2).lbfgs ( lbg, old_dir0, old_step0,  HDIG);  
  
OUTPUT (lb);

