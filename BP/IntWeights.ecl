IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;

EXPORT IntWeights (DATASET($.M_Types.IDNUMRec) NodeNums) := FUNCTION


one := DATASET ([{1,1,0}],$.M_Types.MatRecord);


Rec := RECORD 
	UNSIGNED id;
	UNSIGNED RowNodeNum;
	UNSIGNED ColNodeNum;
END;

Rec extractW($.M_Types.IDNUMRec l,$.M_Types.IDNUMRec r) := TRANSFORM 
	SELF.RowNodeNum := r.Num;
	SELF.ColNodeNum := l.Num;
	SELF := l;
	
END;

Wnodes := JOIN(NodeNums,NodeNums,LEFT.id=(RIGHT.id-1),extractW(LEFT,RIGHT)); //Wnodes contain the row, col number for 
//each w , for example if Wnodes = [{1 , 2, 2},{2 ,3 ,4}] it means that second weight matrix (between layer 1 and 2) has
//the size of 2* and second wight matrix (between layer 2 and 3) has the size of 3*4
//it is important to note that for the weight matrix size is the weight matrix is between layer l and l+1 
//the row number is equal to the nodes in layer l+1 and the col number if equal to the nodes in layer l

//now randomely initialize the weight matrix based on the info you have from Wnodes (the row number and col number for each weight matrix)
 $.M_Types.CellMatRec IntRnd (Rec l) := TRANSFORM 

SELF.id := l.id;
SELF. cellMat := $.RandMat (l.RowNodeNum,l.ColNodeNum);


END;

W_Rnd := NORMALIZE (Wnodes, 1, IntRnd(LEFT));

RETURN W_Rnd;

END;


