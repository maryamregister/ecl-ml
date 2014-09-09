IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;

SparsityParam := 0.1;

LAMBDA := 0.001; // weight decay 

BETA   := 3;  // weight of sparsity penalty term 

ALPHA := 0.1 ;//earning rate

LoopNum := 3; // number of gradient descent iterations in Sparse Autoencoder


weatherRecord := RECORD
	Types.t_RecordID id;
	Types.t_FieldNumber outlook;
	Types.t_FieldNumber temperature;
	Types.t_FieldNumber humidity;
	Types.t_FieldNumber windy;
	Types.t_FieldNumber play;
END;

weather_Data := DATASET([
{1,0,0,1,0,2},
{2,0,0,1,1,2},
{3,1,0,1,0,1},
{4,2,1,1,0,1},
{5,2,2,0,0,1},
{6,2,2,0,1,2},
{7,1,2,0,1,1},
{8,0,1,1,0,2},
{9,0,2,0,0,1},
{10,2,1,0,0,1},
{11,0,1,0,1,1},
{12,1,1,1,1,1},
{13,1,0,0,0,1},
{14,2,1,1,1,2}],
weatherRecord);
OUTPUT(weather_Data, ALL, NAMED('weather_Data'));

indep_data:= TABLE(weather_Data,{id, outlook, temperature, humidity, windy});
dep_data:= TABLE(weather_Data,{id, play});
OUTPUT(indep_data, ALL, NAMED('indep_data'));
OUTPUT(dep_data, ALL, NAMED('dep_data'));



// Sparse Autoencoder is an unsupervised method, no need to use dependent attribute (class label)
ML.ToField(indep_data, pr_indep);
OUTPUT(pr_indep, ALL, NAMED('pr_indep'));


ML.ToField(dep_data, pr_dep);
OUTPUT(pr_dep, ALL, NAMED('pr_dep'));


label_mat := $.ToGroundTruth (pr_dep); 
OUTPUT(label_mat, ALL, NAMED('label_mat'));


Hidden_Nodes := 6; // number of hidden nodes

// feed the data to Sparse Autoencoder
param := $.SA(pr_indep, LAMBDA , ALPHA ,  Hidden_Nodes,  LoopNum );


OUTPUT(param, ALL, NAMED('param'));


//EXtract W1 (id =3) and B1 (id =1)



B := Param (Param.id =1 ); // >0 bcz no bias for the input layer

W :=  Param (Param.id =3 );



OUTPUT(W, ALL, NAMED('W'));
OUTPUT(B, ALL, NAMED('B'));

W1 := W.cellmat;
B1 := B.cellmat;
OUTPUT(W1, ALL, NAMED('W1'));
OUTPUT(B1, ALL, NAMED('B1'));
//extract W1 from W and B1 from B


SA1Features := $.FF_Autoencoder(pr_indep, W1,  B1 );
OUTPUT(SA1Features, ALL, NAMED('SA1Features'));


//feed the inermediate features build in the first SA to a second SA



param2 := $.SA_MatrixInput(SA1Features, LAMBDA , ALPHA ,  Hidden_Nodes,  LoopNum );


OUTPUT(param2, ALL, NAMED('param2'));


//EXtract W1 (id =3) and B1 (id =1)



//EXtract W1 (id =3) and B1 (id =1)



B2 := Param2 (Param2.id =1 ); // >0 bcz no bias for the input layer

W2:=  Param2 (Param2.id =3 );



OUTPUT(W2, ALL, NAMED('W2'));
OUTPUT(B2, ALL, NAMED('B2'));

W2_1 := W2.cellmat;
B2_1 := B2.cellmat;
OUTPUT(W2_1, ALL, NAMED('W2_1'));
OUTPUT(B2_1, ALL, NAMED('B2_1'));
//extract W1 from W and B1 from B


SA2Features := $.FF_Autoencoder_MatrixInput(SA1Features, W2_1,  B2_1 );
OUTPUT(SA2Features, ALL, NAMED('SA2Features'));


// softmax model training

//train the soft,ax model with SA2Features
// weight decay
LAMBDA_soft := 0.0001;
//Learning Rate
ALPHA_soft := 0.1;


Numclass := 2;
OUTPUT  (Numclass, NAMED ('Numclass'));

InputSize := MAX (SA2Features, SA2Features.x);
OUTPUT  (InputSize, NAMED ('InputSize'));

 THETA_soft := Ml.Mat.Each.ScalarMul ($.RandMat (Numclass,InputSize),0.005);


 OUTPUT  (THETA_soft, NAMED ('THETA_soft'));


soft_param := $.SoftMax_matrixinput(SA2Features, label_mat, LAMBDA_soft, ALPHA_soft, THETA_soft , LoopNum ).SoftMaxGradIterations; 
 OUTPUT  (soft_param, NAMED ('soft_param'));
 
 
 
 //initialize stack with the parameters learnt until now and then fine tune the stack