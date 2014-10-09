IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;
//change
//data is input data

//theta is the softmax parameters which it's size is num_classes * num_features(input size)
EXPORT SoftMaxPredict(DATASET(ML.Types.NumericField) input_data, DATASET($.M_Types.MatRecord) THETA  ) := FUNCTION



//convert input_data to matrix foramt
 dTmp := ML.Types.ToMatrix (input_data);
 d := Ml.Mat.Trans(dTmp);
 

x := d;

tx := ML.Mat.Mul (THETA, x);

MaxCol_tx := Ml.Mat.Has(tx).MaxCol;

//minus each element of MaxCol_tx from it's corresponding colomn in tx

ML.Mat.Types.Element DoMinus(tx le,MaxCol_tx ri) := TRANSFORM
    SELF.x := le.x;
    SELF.y := le.y;
	  SELF.value := le.value - ri.value; 
  END;
	
	
//tx minus max of each colomn
tx_M :=  JOIN(tx, MaxCol_tx, LEFT.y=RIGHT.y, DoMinus(LEFT,RIGHT)); 

//compute exponential on each element of tx_M

exp_tx_M := ML.Mat.Each.Exponential(tx_M);


SumCol_exp_tx_M := Ml.Mat.Has(exp_tx_M).SumCol;

//
//divide each element of exp_tx_M on it's corresponding element in SumCol_exp_tx_M (same colomn)

ML.Mat.Types.Element DoDiv(exp_tx_M le,SumCol_exp_tx_M ri) := TRANSFORM
    SELF.x := le.x;
    SELF.y := le.y;
	  SELF.value := le.value / ri.value; 
  END;
	
	
//tx minus max of each colomn, same as Final M in matlab code
Prob :=  JOIN(exp_tx_M, SumCol_exp_tx_M, LEFT.y=RIGHT.y, DoDiv(LEFT,RIGHT)); 

//for each sample return index of the maximum probability value in Prob Matrix
//In Prob matrix each olomn corresponds to one sample. 

prediction_results := ML.Mat.Has(Prob).MaxColIndex;


RETURN prediction_results;
END;