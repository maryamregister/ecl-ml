IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;



// Number of iterations in softmax algortihm
LoopNum := 2; 

// weight decay
LAMBDA := 0.0001;
//Learning Rate
ALPHA := 0.1;




//input data
value_record := RECORD
			unsigned      id;
      real      f1;
			real      f2;
			real      f3;
			real      f4;
      integer1        label; 
END;
input_data := DATASET([
{1, 0.1, 0.2, 0.5 , 0.7, 1},
{2, 0.8, 0.9, 0.4, 0.9 , 2},
{3, 0.5, 0.9, 0.1, 0.1, 3},
{4, 0.8, 0.7, 0.4, 0.6, 3},
{5, 0.9,0.1, 0.3, 0.2, 2},
{6, 0.1, 0.3, 0.6, 0.8, 1}],
 value_record);
OUTPUT  (input_data, ALL, NAMED ('input_data'));


//convert input data to two datset: samples dataset and labels dataset
Sampledata_Format := RECORD 
			input_data.id;
			input_data.f1;
			input_data.f2;
			input_data.f3;
			input_data.f4;

END;

sample_table := TABLE(input_data,Sampledata_Format);
OUTPUT  (sample_table, ALL, NAMED ('sample_table'));


labeldata_Format := RECORD 
			input_data.id;
			input_data.label;

END;

label_table := TABLE(input_data,labeldata_Format);
OUTPUT  (label_table, ALL, NAMED ('label_table'));



ML.ToField(sample_table, sample_table_in_numeric_field_format);

OUTPUT  (sample_table_in_numeric_field_format, ALL, NAMED ('sample_table_in_numeric_field_format'));

ML.ToField(label_table, label_table_in_numeric_field_format)
OUTPUT  (label_table_in_numeric_field_format, ALL, NAMED ('label_table_in_numeric_field_format'));






//initilize THETA (parameters for softmax classifier)
//THETA should be a matrix of size numclass*number of input features (input size)





Numclass := MAX (label_table_in_numeric_field_format, label_table_in_numeric_field_format.value);
OUTPUT  (Numclass, NAMED ('Numclass'));

InputSize := MAX (sample_table_in_numeric_field_format,sample_table_in_numeric_field_format.number);
OUTPUT  (InputSize, NAMED ('InputSize'));

 THETA := Ml.Mat.Each.ScalarMul ($.RandMat (Numclass,InputSize),0.005);


 OUTPUT  (THETA, ALL, NAMED ('THETA'));


// UpTHETA := $.SoftMax_NumericFieldInput( sample_table_in_numeric_field_format, label_table_in_numeric_field_format,  LAMBDA,  ALPHA,THETA ,  LoopNum ).SoftMaxGradIterations;

 UpTHETA := $.SoftMax( sample_table_in_numeric_field_format, label_table_in_numeric_field_format,  LAMBDA,  ALPHA,THETA ,  LoopNum ).SoftMaxGradIterations;

OUTPUT  (UpTHETA,  NAMED ('UpTHETA'));




//test phase

 Test_Prob := $.SoftMaxPredict( sample_table_in_numeric_field_format, UpTHETA );
 OUTPUT  (Test_Prob, ALL, NAMED ('Test_Prob'));












//convert input_data to matrix foramt
 // dTmp := ML.Types.ToMatrix (sample_table_in_numeric_field_format);
 // d := Ml.Mat.Trans(dTmp);
 

// x := d;
 // OUTPUT  (x, ALL, NAMED ('x'));
// tx := ML.Mat.Mul (UpTHETA, x);

 // OUTPUT  (tx, ALL, NAMED ('tx'));

// MaxCol_tx := Ml.Mat.Has(tx).MaxCol;
 // OUTPUT  (MaxCol_tx, ALL, NAMED ('MaxCol_tx'));

//minus each element of MaxCol_tx from it's corresponding colomn in tx

// ML.Mat.Types.Element DoMinus(tx le,MaxCol_tx ri) := TRANSFORM
    // SELF.x := le.x;
    // SELF.y := le.y;
	  // SELF.value := le.value - ri.value; 
  // END;
	
	
//tx minus max of each colomn
// tx_M :=  JOIN(tx, MaxCol_tx, LEFT.y=RIGHT.y, DoMinus(LEFT,RIGHT)); 

 // OUTPUT  (tx_M, ALL, NAMED ('tx_M'));

//compute exponential on each element of tx_M

// exp_tx_M := ML.Mat.Each.Exponential(tx_M);

 // OUTPUT  (exp_tx_M, ALL, NAMED ('exp_tx_M'));
 
// SumCol_exp_tx_M := Ml.Mat.Has(exp_tx_M).SumCol;

 // OUTPUT  (SumCol_exp_tx_M, ALL, NAMED ('SumCol_exp_tx_M'));

//divide each element of exp_tx_M on it's corresponding element in SumCol_exp_tx_M (same colomn)

// ML.Mat.Types.Element DoDiv(exp_tx_M le,SumCol_exp_tx_M ri) := TRANSFORM
    // SELF.x := le.x;
    // SELF.y := le.y;
	  // SELF.value := le.value / ri.value; 
  // END;
	
	
//tx minus max of each colomn, same as Final M in matlab code
// Prob :=  JOIN(exp_tx_M, SumCol_exp_tx_M, LEFT.y=RIGHT.y, DoDiv(LEFT,RIGHT)); 

 // OUTPUT  (Prob, ALL, NAMED ('Prob'));























// OUTPUT  (groundTruth, ALL, NAMED ('groundTruth'));

// m := MAX (d, d.y); //number of samples

// x := d;

// tx := ML.Mat.Mul (THETA, x);

// MaxCol_tx := Ml.Mat.Has(tx).MaxCol;



// ML.Mat.Types.Element DoMinus(tx le,MaxCol_tx ri) := TRANSFORM
    // SELF.x := le.x;
    // SELF.y := le.y;
	  // SELF.value := le.value - ri.value; 
  // END;
	
	

// tx_M :=  JOIN(tx, MaxCol_tx, LEFT.y=RIGHT.y, DoMinus(LEFT,RIGHT)); 



// exp_tx_M := ML.Mat.Each.Exponential(tx_M);


// SumCol_exp_tx_M := Ml.Mat.Has(exp_tx_M).SumCol;



// ML.Mat.Types.Element DoDiv(exp_tx_M le,SumCol_exp_tx_M ri) := TRANSFORM
    // SELF.x := le.x;
    // SELF.y := le.y;
	  // SELF.value := le.value / ri.value; 
  // END;
	
	

// Prob :=  JOIN(exp_tx_M, SumCol_exp_tx_M, LEFT.y=RIGHT.y, DoDiv(LEFT,RIGHT)); 


// second_term := Ml.Mat.Each.ScalarMul (THETA, LAMBDA);

// groundTruth_Prob := ML.Mat.Minus (groundTruth, Prob);

// groundTruth_Prob_x := Ml.Mat.Mul (groundTruth_Prob, Ml.Mat.Trans(x));

// m_1 := -1 * (1/m);

// first_term := Ml.Mat.Each.ScalarMul (groundTruth_Prob_x, m_1);

// THETAGrad := ML.Mat.Add (first_term, second_term);

// UpdatedTHETA := $.UpdateWB_Mat(THETA, THETAGrad, ALPHA).Regular;


