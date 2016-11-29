﻿IMPORT * FROM ML;
IMPORT STD;
IMPORT * FROM $;
IMPORT PBblas;
IMPORT STD;
Layout_Cell := PBblas.Types.Layout_Cell;
Layout_part := PBblas.Types.Layout_part;
//matlab code stlExercise_softmax_LExisNexis_toy.m


value_record := RECORD
REAL8	f1	;
REAL8	f2	;
REAL8	f3	;
REAL8	f4	;
END;


input_data_tmp := DATASET([

{16,2,3,4},
{11,2,3,4},
{17,22,3,4},
{18,2,3,4},
{61,23,3,4},
{19,2,3,54},
{91,26,3,4},
{91,2,3,4},
{19,52,3,4},
{1,62,53,64},
{10,26,33,24},
{1,26,73,40},
{1,25,3,44},
{1,62,39,4},
{1,28,38,40},
{1,2,93,4},
{1,72,53,4},
{10,20,33,4},
{1,2,3,4},
{1,29,3,4},
{11,2,3,4},
{21,2,328,44},
{14,27,3,45},
{13,52,36,45},
{12,62,43,4},
{11,2,3,54},
{13,2,63,4},
{15,2,33,4},
{31,26,33,45},
{13,25,33,4},
{16,2,3,4},
{11,2,3,4},
{17,22,3,4},
{18,2,3,4},
{61,23,3,4},
{19,2,3,54},
{91,26,3,4},
{91,2,3,4},
{19,52,3,4},
{1,62,53,64},
{10,26,33,24},
{1,26,73,40},
{1,25,3,44},
{1,62,39,4},
{1,28,38,40},
{1,2,93,4},
{1,72,53,4},
{10,20,33,4},
{1,2,3,4},
{1,29,3,4},
{111,2,3,4},
{21,2,328,44},
{14,27,3,45},
{13,52,36,45},
{12,62,43,4},
{121,2,3,54},
{13,2,63,4},
{15,2,33,4},
{31,26,33,45},
{13,25,33,4},
{16,2,3,4},
{11,2,3,4},
{17,22,3,4},
{18,2,3,4},
{61,23,3,4},
{19,2,3,54},
{231,26,3,4},
{9,2,3,4},
{19,52,3,4},
{12,625,53,64},
{10,26,33,24},
{1,26,73,40},
{1,25,3,44},
{18,62,39,4},
{1,28,38,40},
{11,42,93,4},
{1,72,53,4},
{10,20,33,4},
{1,2,33,4},
{1,29,3,4},
{11,2,3,4},
{21,32,328,44},
{14,27,3,45},
{13,52,36,45},
{12,62,43,4},
{11,12,343,54},
{13,32,63,4},
{15,22,33,4},
{31,26,33,45},
{13,232,33,44},
{113,25,33,73},
{13,23,3,31},
{143,25,6,2},
{23,7,33,2},
{13,21,23,54},
{13,25,33,7},
{1,35,33,32},
{2,24,33,4},
{13,25,33,47},
{113,25,32,73},
{1,62,39,4},
{1,28,38,40},
{1,2,93,4},
{1,72,53,4},
{10,20,33,4},
{1,2,3,4},
{1,29,3,4},
{111,2,3,4},
{21,2,328,44},
{14,27,3,45},
{13,52,36,45},
{12,62,43,4},
{121,2,3,54},
{13,2,63,4},
{15,2,33,4},
{31,26,33,45},
{13,25,33,4},
{16,2,3,4},
{11,2,3,4},
{17,22,3,4},
{18,2,3,4},
{61,23,3,4},
{19,2,3,54},
{231,26,3,4},
{9,2,3,4},
{19,52,3,4},
{12,625,53,64},
{10,26,33,24},
{1,26,73,40},
{1,25,3,44},
{18,62,39,4},
{1,28,38,40},
{11,42,93,4},
{1,72,53,4},
{10,20,33,4},
{1,2,33,4},
{1,29,3,4},
{11,2,3,4},
{21,32,328,44},
{14,27,3,45},
{13,52,36,45},
{12,62,43,4},
{11,12,343,54},
{13,32,63,4},
{15,22,33,4},
{31,26,33,45},
{13,232,33,44},
{113,25,33,73},
{13,23,3,31},
{143,25,6,2}

], value_record);


ML.AppendID(input_data_tmp, id, input_data);
sample_table := input_data;
ML.ToField(sample_table, indepDataC);



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
{	100	,	2	}	,
{	101	,	2	}	,
{	102	,	2	}	,
{	103	,	1	}	,
{	104	,	2	}	,
{	105	,	1	}	,
{	106	,	1	}	,
{	107	,	1	}	,
{	108	,	2	}	,
{	109	,	1	}	,
{	110	,	2	}	,
{	111	,	1	}	,
{	112	,	2	}	,
{	113	,	1	}	,
{	114	,	1	}	,
{	115	,	1	}	,
{	116	,	2	}	,
{	117	,	2	}	,
{	118	,	2	}	,
{	119	,	2	}	,
{	120	,	2	}	,
{	121	,	1	}	,
{	122	,	2	}	,
{	123	,	1	}	,
{	124	,	1	}	,
{	125	,	1	}	,
{	126	,	2	}	,
{	127	,	1	}	,
{	128	,	2	}	,
{	129	,	1	}	,
{	130	,	2	}	,
{	131	,	1	}	,
{	132	,	1	}	,
{	133	,	1	}	,
{	134	,	2	}	,
{	135	,	2	}	,
{	136	,	2	}	,
{	137	,	2	}	,
{	138	,	1	}	,
{	139	,	1	}	,
{	140	,	1	}	,
{	141	,	1	}	,
{	142	,	1	}	,
{	143	,	1	}	,
{	144	,	1	}	,
{	145	,	1	}	,
{	146	,	2	}	,
{	147	,	2	}	,
{	148	,	2	}	,
{	149	,	2	}	,
{	150	,	2	}	], labeldata_Format);
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

trainer := DeepLearning.softmax_lbfgs (InputSize, Numclass, 2,  3); 

softresult := trainer.LearnC (indepDataC, depDataC,IntTHETA, LAMBDA, LoopNum, corr);

OUTPUT (softresult, named('softresult'));

/* thsirec := RECORD (Layout_Part)
   UNSIGNED real_node;
   END;
   
   OUTPUT (PROJECT(softresult, TRANSFORM (thsirec,  SELF.real_node := STD.System.Thorlib.Node(); SELF := LEFT), LOCAL), ALL);
*/


// SAmod := trainer.model(softresult);

// OUTPUT(SAmod, named('SAmod'));

// prob := trainer.ClassProbDistribC(indepDataC,softresult);

// OUTPUT(prob, named('prob'), ALL);

// classprob := trainer.ClassifyC(indepDataC,softresult);

// OUTPUT(classprob, named('classprob'), ALL);