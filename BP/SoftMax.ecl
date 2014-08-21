IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;
//the first version of implementation of SoftMax classifier, I then changed it's name to SoftMax
//d is input data
//y is labels : c(i,j)=1 IFF the lable of the jth sample is i
//theta is the softmax parameters which it's size is num_classes * num_features(input size)
//LAMBDA is wight decay parameter
EXPORT SoftMax(DATASET($.M_Types.MatRecord) d, DATASET($.M_Types.MatRecord) y, REAL8 LAMBDA, REAL8 ALPHA,DATASET($.M_Types.MatRecord) IntTHETA , UNSIGNED LoopNum ) := MODULE



SHARED SoftMaxGrad (DATASET($.M_Types.MatRecord) THETA ):= FUNCTION

groundTruth:= y; 

m := MAX (d, d.y); //number of samples

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

//thetagrad=((-1/m)*(groundTruth-M)*x')+lambda*theta;

second_term := Ml.Mat.Each.ScalarMul (THETA, LAMBDA);

groundTruth_Prob := ML.Mat.Minus (groundTruth, Prob);

groundTruth_Prob_x := Ml.Mat.Mul (groundTruth_Prob, Ml.Mat.Trans(x));

m_1 := -1 * (1/m);

first_term := Ml.Mat.Each.ScalarMul (groundTruth_Prob_x, m_1);

THETAGrad := ML.Mat.Add (first_term, second_term);

UpdatedTHETA := $.UpdateWB_Mat(THETA, THETAGrad, ALPHA).Regular;

RETURN UpdatedTHETA;
END; 



EXPORT SoftMaxGradIterations := FUNCTION

//apply SoftmaxGrad fucntion on Param for LoopNum number of iterations to update THETA in each iteration
//then return the final updated THETA




loopBody(DATASET($.M_Types.MatRecord) ds) :=
 SoftMaxGrad (ds);
		
		

		
Final_Updated_THETA := LOOP(IntTHETA,  COUNTER <= LoopNum,  loopBody(ROWS(LEFT)));		





RETURN 	Final_Updated_THETA;

END;



END;
