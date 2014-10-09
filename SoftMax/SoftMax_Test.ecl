IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;



// Number of iterations in softmax algortihm
LoopNum := 1; 

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


samplematrix := ML.Types.ToMatrix (sample_table_in_numeric_field_format);
OUTPUT  (samplematrix, ALL, NAMED ('samplematrix'));

labelmatrix := ML.Types.ToMatrix (label_table_in_numeric_field_format);
OUTPUT  (labelmatrix, ALL, NAMED ('labelmatrix'));

//initilize THETA (parameters for softmax classifier)
//THETA should be a matrix of size numclass*number of input features (input size)





Numclass := MAX (label_table_in_numeric_field_format, label_table_in_numeric_field_format.value);
OUTPUT  (Numclass, NAMED ('Numclass'));

InputSize := MAX (sample_table_in_numeric_field_format,sample_table_in_numeric_field_format.number);
OUTPUT  (InputSize, NAMED ('InputSize'));

 
 THETA :=  DATASET([
{1,1,0.1},
{1,2,0.9},
{1,3,0.6},
{1,4,0.6},
{2,1,0.4},
{2,2,0.3},
{2,3,0.1},
{2,4,0.6},
{3,1,0.2},
{3,2,0.4},
{3,3,0.7},
{3,4,0.6}],
$.M_Types.MatRecord);



 OUTPUT  (THETA, ALL, NAMED ('THETA'));



 UpTHETA := $.SoftMax( sample_table_in_numeric_field_format, label_table_in_numeric_field_format,  LAMBDA,  ALPHA,THETA ,  LoopNum );

OUTPUT  (UpTHETA,  NAMED ('UpTHETA'));




//test phase

Prediction := $.SoftMaxPredict(sample_table_in_numeric_field_format, UpTHETA  );


OUTPUT  (Prediction,  NAMED ('Prediction'));






