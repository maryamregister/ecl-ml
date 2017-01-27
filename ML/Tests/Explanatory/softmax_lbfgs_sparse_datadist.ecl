IMPORT * FROM ML;
IMPORT STD;
IMPORT * FROM $;
IMPORT PBblas;
IMPORT STD;
Layout_Cell := PBblas.Types.Layout_Cell;
Layout_part := PBblas.Types.Layout_part;
//matlab code stlExercise_softmax_LExisNexis_toy_sparse.m

indepDataC := DATASET ([

{ 1,1 , 32},
{ 1,2 , 43},
{ 2, 1,4 },
{ 2,2 ,3 },
{ 3,1 , 43},
{4 ,1 ,2 },
{ 5, 1,765 },
{ 6, 1, 78},
{ 7, 1, 5},
{8 , 2,43 },
{ 9,2 , 32},
{ 10,2 , 112},
{ 11,1 , 32},
{ 11,2 , 43},
{ 12, 1,4 },
{ 12,2 ,3 },
{ 13,1 , 43},
{14 ,1 ,2 },
{ 15, 1,765 },
{ 16, 1, 78},
{ 17, 1, 5},
{18 , 2,43 },
{ 19,2 , 32},
{ 20,2 , 112},

{ 21, 3, 23},
{ 21, 4,3 },
{ 22,3 , 4},
{ 23,4 ,54 },
{ 24,4 ,54 },
{ 25,1 , 34},
{ 26,1 , 45},
{ 27,2 , 45},
{28 ,1 , 23},
{ 29, 1, 78},
{ 30,2 , 6},
{ 31, 3,5 },
{ 31, 4, 4},
{ 32,3 ,23 },
{ 33,4 , 43},
{ 34,4 , 43},
{ 35,1 , 43},
{ 36,1 ,23 },
{ 37,2 ,32 },
{38 ,1 ,7 },
{ 39, 1,7 },
{ 40,2 , 6},
{ 41, 3,5 },
{41, 4,4 },
{ 42,3 , 3},
{ 43,4 , 2},
{ 44,4 , 1},
{ 45,1 , 25},
{ 46,1 , 45},
{ 47,2 , 768},
{48 ,1 , 45},
{ 49, 1, 43},
{ 50,2 , 414},

{ 51,1 , 32},
{51,2 , 43},
{ 52, 1,4 },
{ 52,2 ,3 },
{ 53,1 , 43},
{54 ,1 ,2 },
{ 55, 1,765 },
{ 56, 1, 78},
{ 57, 1, 5},
{58 , 2,43 },
{ 59,2 , 32},
{ 60,2 , 112},
{ 61,1 , 32},
{ 61,2 , 43},
{ 62, 1,4 },
{ 62,2 ,3 },
{ 63,1 , 43},
{64 ,1 ,2 },
{ 65, 1,765 },
{ 66, 1, 78},
{ 67, 1, 5},
{68 , 2,43 },
{ 69,2 , 32},
{ 70,2 , 112},
{ 71, 3, 415},
{ 71, 4,245 },
{ 72,3 ,2345 },
{ 73,4 ,432 },
{ 74,4 ,432 },
{ 75,1 , 234},
{ 76,1 ,1234 },
{ 77,2 ,3 },
{78 ,1 ,24 },
{ 79, 1,43 },
{ 80,2 , 43},
{ 81, 3, 24},
{81, 4, 53},
{ 82,3 , 2},
{ 83,4 , 52},
{ 84,4 ,24 },
{ 85,1 , 2},
{ 86,1 , 65},
{ 87,2 , 76},
{88 ,1 ,87 },
{ 89, 1, 87},
{ 90,2 , 7},
{ 91,2 , 7},
{ 92,2 , 7},
{ 93,2 , 7},
{ 94,2 , 7},
{ 95,2 , 7},
{ 96,2 , 7},
{ 97,2 , 7},
{ 98,2 , 7},
{ 99,2 , 7},
{100, 2, 1},
{ 101,2 , 7},
{ 102,2 , 7},
{ 103,2 , 7},
{ 104,2 , 7},
{105, 2, 1}], ML.Types.NumericField);



output(indepDataC, named ('indepDataC'), ALL);

labeldata_Format := RECORD
  UNSIGNED id;
  INTEGER label;
END;

label_table := DATASET ([
{	1	,	1	}	,
{	2	,	2	}	,
{	3	,	1	}	,
{	4	,	2	}	,
{	5	,	1	}	,
{	6	,	2	}	,
{	7	,	1	}	,
{	8	,	1	}	,
{	9	,	1	}	,
{	10	,	2	}	,
{	11	,	2	}	,
{	12	,	2	}	,
{	13	,	2	}	,
{	14	,	1	}	,
{	15	,	2	}	,
{	16	,	1	}	,
{	17	,	2	}	,
{	18	,	1	}	,
{	19	,	2	}	,
{	20	,	1	}	,
{	21	,	1	}	,
{	22	,	1	}	,
{	23	,	2	}	,
{	24	,	2	}	,
{	25	,	2	}	,
{	26	,	2	}	,
{	27	,	1	}	,
{	28	,	2	}	,
{	29	,	1	}	,
{	30	,	2	}	,
{	31	,	1	}	,
{	32	,	2	}	,
{	33	,	1	}	,
{	34	,	1	}	,
{	35	,	1	}	,
{	36	,	2	}	,
{	37	,	2	}	,
{	38	,	2	}	,
{	39	,	2	}	,
{	40	,	2	}	,
{	41	,	1	}	,
{	42	,	2	}	,
{	43	,	1	}	,
{	44	,	2	}	,
{	45	,	1	}	,
{	46	,	1	}	,
{	47	,	1	}	,
{	48	,	2	}	,
{	49	,	2	}	,
{	50	,	2	}	,
{	51	,	2	}	,
{	52	,	2	}	,
{	53	,	1	}	,
{	54	,	2	}	,
{	55	,	2	}	,
{	56	,	1	}	,
{	57	,	2	}	,
{	58	,	1	}	,
{	59	,	1	}	,
{	60	,	1	}	,
{	61	,	2	}	,
{	62	,	1	}	,
{	63	,	2	}	,
{	64	,	1	}	,
{	65	,	2	}	,
{	66	,	1	}	,
{	67	,	1	}	,
{	68	,	1	}	,
{	69	,	2	}	,
{	70	,	2	}	,
{	71	,	2	}	,
{	72	,	2	}	,
{	73	,	2	}	,
{	74	,	1	}	,
{	75	,	1	}	,
{	76	,	2	}	,
{	77	,	2	}	,
{	78	,	2	}	,
{	79	,	2	}	,
{	80	,	2	}	,
{	81	,	1	}	,
{	82	,	2	}	,
{	83	,	2	}	,
{	84	,	1	}	,
{	85	,	2	}	,
{	86	,	1	}	,
{	87	,	1	}	,
{	88	,	1	}	,
{	89	,	2	}	,
{	90	,	1	}	,
{	91	,	2	}	,
{	92	,	1	}	,
{	93	,	2	}	,
{	94	,	1	}	,
{	95	,	1	}	,
{	96	,	1	}	,
{	97	,	2	}	,
{	98	,	2	}	,
{	99	,	2	}	,
{	100	,	2	},
{	101	,	2	},	
{	102	,	2	},	
{	103	,	2	},	
{	104	,	2	}	,
{	105	,	2	}	], labeldata_Format);
OUTPUT  (label_table, ALL, NAMED ('label_table'));



ML.ToField(label_table, depDataC);
OUTPUT  (depDataC, ALL, NAMED ('depDataC'));
label := PROJECT(depDataC,Types.DiscreteField);
OUTPUT  (label, ALL, NAMED ('label'));

//initialize THETA
Numclass := MAX (label, label.value);
OUTPUT  (Numclass, ALL, NAMED ('Numclass'));
InputSize := 4;

T1 := DATASET ([
{1,1, 0.1},
{2,1, 0.2},
{1,2, 0.3},
{2,2,0.001},
{1,3,0.009},
{2,3, 0.098},
{1,4,0.543},
{2,4,0.876}

], Mat.Types.Element);
OUTPUT  (T1, ALL, NAMED ('T1'));
IntTHETA := Mat.Scale (T1,0.005);
OUTPUT  (IntTHETA, ALL, NAMED ('IntTHETA'));
//SoftMax_Sparse Classfier

//Set Parameters
LoopNum := 20; // Number of iterations in softmax algortihm
LAMBDA := 0.0001; // weight decay parameter in  claculation of SoftMax Cost fucntion

UNSIGNED4 prows:=0;
 UNSIGNED4 pcols:=0;

 UNSIGNED corr := 3;

trainer := DeepLearning.softmax_lbfgs_partitions_datadist (InputSize, Numclass, 2,  2); 

softresult := trainer.LearnC (indepDataC, depDataC,IntTHETA, LAMBDA, LoopNum, corr);

OUTPUT (softresult, named('softresult'), ALL);

thsirec := RECORD (Layout_Part)
UNSIGNED real_node;
END;

OUTPUT (PROJECT(softresult, TRANSFORM (thsirec,  SELF.real_node := STD.System.Thorlib.Node(); SELF := LEFT), LOCAL), ALL);


SAmod := trainer.model(softresult);

OUTPUT(SAmod, named('SAmod'));

prob := trainer.ClassProbDistribC(indepDataC,softresult);

OUTPUT(prob, named('prob'), ALL);

// classprob := trainer.ClassifyC(indepDataC,softresult);

// OUTPUT(classprob, named('classprob'), ALL);
