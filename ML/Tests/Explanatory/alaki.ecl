IMPORT * FROM ML;
IMPORT STD;
IMPORT * FROM $;
IMPORT PBblas;
IMPORT STD;
Layout_Cell := PBblas.Types.Layout_Cell;
Layout_part := PBblas.Types.Layout_part;

nrow := 3;
ncol := 13;
 ncol_part:= 5;
	blk := nrow * ncol_part;
	Produce_Random () := FUNCTION
		G := 1000000;
		R := (RANDOM()%G) / (REAL8)G;
		RETURN R;
	END;
	nodes_used := IF(ncol_part>0, ((ncol-1) DIV ncol_part) + 1, 1);//how many nodes are needed
	W_Rec := RECORD
			STRING1 x:= '';
  END;
	empty_init := DATASET([{' '}], W_Rec);
	nid_rec := {UNSIGNED n_id};
	nid_rec init_tran (UNSIGNED c) := TRANSFORM
		SELF.n_id := c-1;
	END;
	init_ := NORMALIZE (empty_init, nodes_used, init_tran (COUNTER) );
	init_dist := DISTRIBUTE (init_, n_id);
	ML.Mat.Types.Element gen_tran (nid_rec le, UNSIGNED c) := TRANSFORM
		col_offset := ncol_part*le.n_id;
		SELF.x := ((c-1) % nrow) + 1 ;
    SELF.y := ((c-1) DIV nrow) + 1 + col_offset;
		SELF.value := Produce_Random ();
	END;
	
	result := NORMALIZE (init_dist, blk, gen_tran (LEFT, COUNTER));
	fltr := result.x BETWEEN 1 AND nrow AND result.y BETWEEN 1 AND ncol;


// OUTPUT (init_dist);
// OUTPUT (result);
// OUTPUT (result (fltr));
filednode := RECORD (Types.NumericField)
	UNSIGNED real_node;
END;
// thiis := PROJECT (result (fltr), TRANSFORM (filednode, SELF.real_node := STD.System.Thorlib.Node(); SELF:=LEFT), LOCAL);
// OUTPUT (thiis, named ('thiis'));
/*

fileName := '~vherrara::datasets::sparsearfffile.arff';
// fileName := '~vherrara::datasets::sentiment_75pct.arff';
Srec := {STRING Line};
SnodeRec := RECORD (Srec)
	UNSIGNED real_node;
END;
   InDS    := DATASET(fileName, Srec, CSV(SEPARATOR([])));
	 OUTPUT (InDS);
	 
	 Indprj := PROJECT (InDS, TRANSFORM (SnodeRec,SELF.real_node := STD.System.Thorlib.Node(); SELF := LEFT), LOCAL);
	 OUTPUT (Indprj);
	a :=  TABLE(Indprj,{real_node});
	Lasts   := SORT(a,real_node);
	MySet   := DEDUP(Lasts,real_node);

	 OUTPUT (MySet);
	 
rec := RECORD
Indprj.real_node;
StCnt := COUNT(GROUP);
END;

recnode := RECORD (rec)
UNSIGNED real_node2;
END;
Mytable := TABLE(Indprj,rec,real_node, LOCAL);
	 
	 mytblprj := PROJECT (Mytable, TRANSFORM (recnode,SELF.real_node2 := STD.System.Thorlib.Node();SELF:=LEFT),LOCAL);
	 OUTPUT (mytblprj);
	 
	 withid :=  PROJECT(InDS, TRANSFORM({UNSIGNED RecID, STRING Line}, SELF.RecID:= COUNTER, SELF := LEFT));
	 withid_node := PROJECT (withid,TRANSFORM ({UNSIGNED RecID, STRING Line, UNSIGNED real_node},SELF.real_node := STD.System.Thorlib.Node();SELF:=LEFT),LOCAL);
	 OUTPUT (MAX (withid, RecID));
	 
	 b :=  TABLE(withid_node,{real_node});
	Lastsb   := SORT(b,real_node);
	MySetb   := DEDUP(Lastsb,real_node);

	OUTPUT (MySetb);
	OUTPUT (withid);
	 // OUTPUT (DEDUP(Indprj,LEFT.real_node=RIGHT.real_node, LOCAL));
	*/ 
	 
	 /*
   // ParseDS := PROJECT(InDS, TRANSFORM({UNSIGNED RecID, STRING Line}, SELF.RecID:= COUNTER, SELF := LEFT),LOCAL);
	  ParseDS := PROJECT(InDS, TRANSFORM({UNSIGNED RecID, STRING Line}, SELF.RecID:= ((COUNTER-1)* STD.System.Thorlib.nodes())+ STD.System.Thorlib.node() + 1, SELF := LEFT), LOCAL);
   //Parse the fields and values out
   PATTERN ws       := ' ';
   PATTERN RecStart := '{';
   PATTERN ValEnd   := '}' | ',';
   PATTERN FldNum   := PATTERN('[0-9]')+;
   PATTERN DataQ    := '"' PATTERN('[ a-zA-Z0-9]')+ '"';
   PATTERN DataNQ   := PATTERN('[a-zA-Z0-9]')+;
   PATTERN DataVal  := DataQ | DataNQ;
   PATTERN FldVal   := OPT(RecStart) FldNum ws DataVal ValEnd;
   OutRec := RECORD
     UNSIGNED RecID;
     STRING   FldName;
     STRING   FldVal;
   END;
   Types.DiscreteField XF(ParseDS L) := TRANSFORM
     SELF.id     := L.RecID;
     SELF.number := (TYPEOF(SELF.number))MATCHTEXT(FldNum) + 1;
     SELF.value  := (TYPEOF(SELF.value))MATCHTEXT(DataVal);
   END;
TrainDS :=  PARSE(ParseDS, Line, FldVal, XF(LEFT));
indepData := TrainDS(Number<109736);
depData   := TrainDS(Number=109736);
// input_data_tmp := DATASET('~maryam::mytest::mnist_5digits_traindata', value_record, CSV); // This dataset is a subset of MNIST dtaset that includes 5 digits (0 to 4), it is used for traibn
//// max(id) = 15298
indepDataC := PROJECT (indepData, TRANSFORM (ML.Types.NumericField,SELF:=LEFT), LOCAL);

OUTPUT (indepDataC);
numnode := RECORD (ML.Types.NumericField)
UNSIGNED real_node;
END;
indeprealnode := PROJECT (indepDataC, TRANSFORM (numnode, SELF.real_node := STD.System.Thorlib.Node(); SELF := LEFT), LOCAL);
OUTPUT (indeprealnode, named ('indeprealnode'));

OUTPUT (MAX (indeprealnode, indeprealnode.real_node));


OUTPUT (MAX (ParseDS, ParseDS.recID));
*/

Y := DATASET ([{1,10,1},
{2,10,10},
{3,11,1},
{4,11,189},
{5,12,12976986},
{6,1,2},
{7,1,2},
{8,13,78},
{9,13,12976986},{10,12,78},{11,12,1},{12,1,189}],Types.NumericField);

OUTPUT (Y, named ('YY'));

YB := utils.DistinctFeaturest (Y);

OUTPUT (YB, named ('YB'), ALL);