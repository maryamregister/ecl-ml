IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;



// Number of iterations in softmax algortihm
LoopNum := 100; 

OUTPUT  (LoopNum,NAMED ('LoopNum'));

// weight decay
LAMBDA := 0.0001;
//Learning Rate
ALPHA := 0.1;



//input data
value_record := RECORD
real	f1	;
real	f2	;
real	f3	;
real	f4	;
real	f5	;
real	f6	;
real	f7	;
real	f8	;
real	f9	;
real	f10	;
real	f11	;
real	f12	;
real	f13	;
real	f14	;
real	f15	;
real	f16	;
real	f17	;
real	f18	;
real	f19	;
real	f20	;
INTEGER Label  ;
END;
input_data_tmp := DATASET('~online::maryam::mytest::cancer_20at_96sa_data', value_record, CSV);
OUTPUT  (input_data_tmp,NAMED ('input_data_tmp'));

 ML.AppendID(input_data_tmp, id, input_data);
//convert input data to two datset: samples dataset and labels dataset
Sampledata_Format := RECORD 
input_data.id;
input_data.f1	;
input_data.f2	;
input_data.f3	;
input_data.f4	;
input_data.f5	;
input_data.f6	;
input_data.f7	;
input_data.f8	;
input_data.f9	;
input_data.f10	;
input_data.f11	;
input_data.f12	;
input_data.f13	;
input_data.f14	;
input_data.f15	;
input_data.f16	;
input_data.f17	;
input_data.f18	;
input_data.f19	;
input_data.f20	;
END;

sample_table := TABLE(input_data,Sampledata_Format);
OUTPUT  (sample_table, NAMED ('sample_table'));


labeldata_Format := RECORD 
			input_data.id;
			input_data.Label;

END;

label_table := TABLE(input_data,labeldata_Format);
OUTPUT  (label_table,  NAMED ('label_table'));



ML.ToField(sample_table, sample_table_in_numeric_field_format);

OUTPUT  (sample_table_in_numeric_field_format, NAMED ('sample_table_in_numeric_field_format'));

ML.ToField(label_table, label_table_in_numeric_field_format)
OUTPUT  (label_table_in_numeric_field_format, NAMED ('label_table_in_numeric_field_format'));






//initilize THETA (parameters for softmax classifier)
//THETA should be a matrix of size numclass*number of input features (input size)





Numclass := MAX (label_table_in_numeric_field_format, label_table_in_numeric_field_format.value);
OUTPUT  (Numclass, NAMED ('Numclass'));

InputSize := MAX (sample_table_in_numeric_field_format,sample_table_in_numeric_field_format.number);
OUTPUT  (InputSize, NAMED ('InputSize'));

T1 := $.RandMat (Numclass,InputSize);
 OUTPUT  (T1, NAMED ('T1'));

 THETA := Ml.Mat.Each.ScalarMul (T1,0.005);


 OUTPUT  (THETA, NAMED ('THETA'));


 UpTHETA := $.SoftMax( sample_table_in_numeric_field_format, label_table_in_numeric_field_format,  LAMBDA,  ALPHA,THETA ,  LoopNum);

 OUTPUT  (UpTHETA, NAMED ('UpTHETA'));




//test phase

 // Test_Prob := $.SoftMaxPredict( sample_table_in_numeric_field_format, UpTHETA );
 // OUTPUT  (Test_Prob,  NAMED ('Test_Prob'));
























