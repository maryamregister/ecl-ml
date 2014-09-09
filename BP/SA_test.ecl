IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;



weatherRecord := RECORD
	Types.t_RecordID id;
	Types.t_FieldNumber outlook;
	Types.t_FieldNumber temperature;
	Types.t_FieldNumber humidity;
	Types.t_FieldNumber windy;
	Types.t_FieldNumber play;
END;

weather_Data := DATASET([
{1,0,0,1,0,0},
{2,0,0,1,1,0},
{3,1,0,1,0,1},
{4,2,1,1,0,1},
{5,2,2,0,0,1},
{6,2,2,0,1,0},
{7,1,2,0,1,1},
{8,0,1,1,0,0},
{9,0,2,0,0,1},
{10,2,1,0,0,1},
{11,0,1,0,1,1},
{12,1,1,1,1,1},
{13,1,0,0,0,1},
{14,2,1,1,1,0}],
weatherRecord);
indep_data:= TABLE(weather_Data,{id, outlook, temperature, humidity, windy});
dep_data:= TABLE(weather_Data,{id, play});
OUTPUT(indep_data, ALL, NAMED('indep_data'));
OUTPUT(dep_data, ALL, NAMED('dep_data'));
OUTPUT(weather_Data, ALL, NAMED('weather_Data'));


// Sparse Autoencoder is an unsupervised method, no need to use dependent attribute (class label)
ML.ToField(indep_data, pr_indep);
OUTPUT(pr_indep, ALL, NAMED('pr_indep'));


LAMBDA := 0.001; //weight decay parameter 
ALPHA  := 0.1; // learning rate
Hidden_Nodes := 6; // number of hidden nodes
LoopNum := 3; // number of gradient descent iterations in Sparse Autoencoder

// feed the data to Sparse Autoencoder
param := $.SA(pr_indep, LAMBDA , ALPHA ,  Hidden_Nodes,  LoopNum );


OUTPUT(param, ALL, NAMED('param'));


 dTmp := ML.Types.ToMatrix (pr_indep);
 d := Ml.Mat.Trans(dTmp);
OUTPUT(d, ALL, NAMED('d'));