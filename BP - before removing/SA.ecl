IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;
//this function learns a sparse autoencoder and return the learnt parameters (training phase)
//test is done on SA_test
//the parameters are wights and bias values. two weight matrices and two bias matrices. the bias matrix of the first layer has 
//id 1, the bias matrix of second layer has id 2, the weight matrix of first layer has id 3, the weight of secod layer has id 4

EXPORT SA(DATASET(ML.Types.NumericField) input_data, REAL8 LAMBDA, REAL8 ALPHA, UNSIGNED Hidden_Nodes, UNSIGNED LoopNum ) := FUNCTION


//convert input_data to matrix foramt
 dTmp := ML.Types.ToMatrix (input_data);
 d := Ml.Mat.Trans(dTmp);
 
 
 




// for the desired output values (class labeld for the samples) each colomns corresponds to one sample's output
Y := d;


//the NodeNum is the essential input which shows the structure of the network, each element of
//the nodeNum is like {layer number, number of nodes in that layer}
//obviously the first layer has the same number of nodes of the number of input data features
// and the output layer has the same number of nodes as the number of classes (if the problem is numeric prediction
//then the output layer has just one node)
//Please note that the number of nodes in each layer should be without considering biad node (bias nodes are seprately considered in the implementation)

// sparse autoencoder has 3 layers, in  first and third layers the number of nodes in equal to number of input features
// the number of nodes in the hiddne layer (second layer) should be set by input parameters


First_Nodes := Max (d, d.x); // number of first layer nodes
Last_nodes  := First_Nodes; // number of last layerndoes which is equal to number of first layer nodes


NodeNum := DATASET ([{1,First_Nodes},{2,Hidden_Nodes},{3,Last_nodes}],$.M_Types.IDNUMRec);




//initilize the weight and bias values (weights with randome samll number, bias with zeros)
W := $.IntWeights  (NodeNum);



B := $.IntBias (NodeNum);



//Maked PARAM and pass it to Gradietn Desent Function
add_num := MAX (W, W.id);


$.M_Types.CellMatRec addone (W l) := TRANSFORM
	SELF.id := l.id+add_num;
	SELF := l;
END;

Wadd := PROJECT (W,addone(LEFT));


Parameters := Wadd+B; //Now the ids related to B matrices are from 0 to n (number of layers)
// and ids for W matrices are from 1+n to n+n
// in the GradDes W and B matrix are going to be extracted from the "Parameters" again and it is
//done based on id values (the B matrix related to id=0 is not nessecary and do not need to be extracted);


Updated_Param:= $.SA_GradDesLoop (  d, y,Parameters,  LAMBDA,  ALPHA,  LoopNum).GDIterations;
//now updated_parameters contain the updated weights and bias values. and you need to extract W and B matrices
//by considering weight ids as id+number(number of layers-1) of w matrices

RETURN Updated_Param;

END;
