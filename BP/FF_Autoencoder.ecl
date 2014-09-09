IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;
//Feed Forward  Autoencoder : Instructions: Compute the activation of the hidden layer for the Sparse Autoencoder.


EXPORT FF_Autoencoder(DATASET(ML.Types.NumericField) input_data, DATASET($.M_Types.MatRecord) W1, DATASET($.M_Types.MatRecord) B1 ) := FUNCTION


//convert input_data to matrix foramt
 dTmp := ML.Types.ToMatrix (input_data);
 d := Ml.Mat.Trans(dTmp);
 
 
/*
x=data;
m=size(data,2);
z2 = W1 * x + repmat(b1,1,m);
a2 = sigmoid(z2);
%-------------------------------------------------------------------
activation=a2;

*/


m := MAX (d, d.y);

z2 :=  ML.Mat.Vec.Add_Mat_Vec (ML.Mat.Mul(W1,d),B1,1);

a2 := ML.Mat.Each.Sigmoid(z2);

RETURN a2;
END;
