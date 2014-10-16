IMPORT * FROM ML;
IMPORT * FROM $;



//Set Parameters
LoopNum := 10; // Number of iterations in softmax algortihm
LAMBDA := 0.0001; // weight decay parameter in  claculation of SoftMax Cost fucntion
ALPHA := 0.1; // //Learning Rate for updating SoftMax parameters


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


ML.ToField(sample_table, indepDataC);

OUTPUT  (indepDataC, ALL, NAMED ('indepDataC'));

ML.ToField(label_table, depDataC);

OUTPUT  (depDataC, ALL, NAMED ('depDataC'));


label := PROJECT(depDataC,Types.DiscreteField);

OUTPUT  (label, ALL, NAMED ('label'));

//initialize THETA
Numclass := MAX (label, label.value);
OUTPUT  (Numclass, ALL, NAMED ('Numclass'));
InputSize := MAX (indepDataC,indepDataC.number);
OUTPUT  (InputSize, ALL, NAMED ('InputSize'));
T1 := Mat.RandMat (Numclass,InputSize);
OUTPUT  (T1, ALL, NAMED ('T1'));
IntTHETA := Mat.Scale (T1,0.005);
OUTPUT  (IntTHETA, ALL, NAMED ('IntTHETA'));
//SoftMax_Sparse Classfier
trainer:= ML.Classify.SoftMax_Sparse(IntTHETA, LAMBDA, ALPHA, LoopNum);
//Learning Phase
Model := trainer.LearnC(indepDataC, label);
OUTPUT  (Model, ALL, NAMED ('Model'));

//test phase
dist := trainer.ClassProbDistribC(indepDataC,Model );
classified := trainer.ClassifyC(indepDataC,Model);
OUTPUT  (dist, ALL, NAMED ('dist'));
OUTPUT  (classified, ALL, NAMED ('classified'));