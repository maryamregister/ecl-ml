IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;


// Number of iterations in softmax algortihm
LoopNum := 2; 

// weight decay
LAMBDA := 0.1;
//Learning Rate
ALPHA := 0.1;
// for the input data each colomn corresponds to one instance (sample)
d := DATASET([
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
$.M_Types.MatRecord);
OUTPUT  (d, ALL, NAMED ('d'));
// for the desired output values (class labels for the samples) each colomns corresponds to one sample's output
// in each colomn there should be exactly one 1 and all other values zeros, 1 value would show the class for that sample
Label := DATASET([
{1,1,1},
{1,2,0},
{1,3,0},
{2,1,0},
{2,2,0},
{2,3,1},
{3,1,0},
{3,2,1},
{3,3,0}],
$.M_Types.MatRecord);
OUTPUT  (Label, ALL, NAMED ('Label'));




//initilize THETA (parameters for softmax classifier)
//THETA should be a matrix of size numclass*number of input features (input size)
Numclass := MAX (Label, Label.x);
OUTPUT  (Numclass, NAMED ('Numclass'));
InputSize := MAX (d,d.x);
OUTPUT  (InputSize, NAMED ('InputSize'));
 THETA := Ml.Mat.Each.ScalarMul ($.RandMat (Numclass,InputSize),0.005);


OUTPUT  (THETA, ALL, NAMED ('THETA'));


UpTHETA := $.SoftMax( d, Label,  LAMBDA,  ALPHA,THETA ,  LoopNum ).SoftMaxGradIterations;

OUTPUT  (UpTHETA, ALL, NAMED ('UpTHETA'));




//test phase

Test_Prob := $.SoftMaxPredict( d, UpTHETA );
OUTPUT  (Test_Prob, ALL, NAMED ('Test_Prob'));
























// groundTruth:= Label; 

// m := MAX (d, d.y); //number of samples

// OUTPUT  (m,NAMED ('m'));

// x := d;

// tx := ML.Mat.Mul (THETA, x);
// OUTPUT  (tx, ALL, NAMED ('tx'));

// MaxCol_tx := Ml.Mat.Has(tx).MaxCol;
// OUTPUT  (MaxCol_tx, ALL, NAMED ('MaxCol_tx'));



// ML.Mat.Types.Element DoMinus(tx le,MaxCol_tx ri) := TRANSFORM
    // SELF.x := le.x;
    // SELF.y := le.y;
	  // SELF.value := le.value - ri.value; 
  // END;
	
	

// tx_M :=  JOIN(tx, MaxCol_tx, LEFT.y=RIGHT.y, DoMinus(LEFT,RIGHT)); 


// OUTPUT  (tx_M, ALL, NAMED ('tx_M'));
 // exp_tx_M := ML.Mat.Each.Exponential(tx_M);

// OUTPUT  (exp_tx_M, ALL, NAMED ('exp_tx_M'));
 // SumCol_exp_tx_M := Ml.Mat.Has(exp_tx_M).SumCol;

// OUTPUT  (SumCol_exp_tx_M, ALL, NAMED ('SumCol_exp_tx_M'));


// ML.Mat.Types.Element DoDiv(exp_tx_M le,SumCol_exp_tx_M ri) := TRANSFORM
    // SELF.x := le.x;
    // SELF.y := le.y;
	  // SELF.value := le.value / ri.value; 
  // END;
	
	

// Prob :=  JOIN(exp_tx_M, SumCol_exp_tx_M, LEFT.y=RIGHT.y, DoDiv(LEFT,RIGHT)); 

// OUTPUT  (Prob, ALL, NAMED ('Prob'));

// second_term := Ml.Mat.Each.ScalarMul (THETA, LAMBDA);
// OUTPUT  (second_term, ALL, NAMED ('second_term'));
// groundTruth_Prob := ML.Mat.Minus (groundTruth, Prob);
// OUTPUT  (groundTruth_Prob, ALL, NAMED ('groundTruth_Prob'));
// groundTruth_Prob_x := Ml.Mat.Mul (groundTruth_Prob, Ml.Mat.Trans(x));
// OUTPUT  (groundTruth_Prob_x, ALL, NAMED ('groundTruth_Prob_x'));
// m_1 := -1 * (1/m);

// first_term := Ml.Mat.Each.ScalarMul (groundTruth_Prob_x, m_1);
// OUTPUT  (first_term, ALL, NAMED ('first_term'));
// THETAGrad := ML.Mat.Add (first_term, second_term);
// OUTPUT  (THETAGrad, ALL, NAMED ('THETAGrad'));
// UpdatedTHETA := $.UpdateWB_Mat(THETA, THETAGrad, ALPHA).Regular;
// OUTPUT  (UpdatedTHETA, ALL, NAMED ('UpdatedTHETA'));


