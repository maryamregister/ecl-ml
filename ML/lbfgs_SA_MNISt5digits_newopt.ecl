﻿IMPORT * FROM ML;
IMPORT * FROM $;
IMPORT PBblas;
IMPORT STD;
Layout_Cell := PBblas.Types.Layout_Cell;
// most recent  W20160915-232308
// test Sparse_Autoencoder_lbfgs on an MNIST dataset which contains only five digits {0,1,2,3,4} : workunit W20160724-100930
INTEGER4 hl := 25;//number of nodes in the hiddenlayer
INTEGER4 f := 28*28;//number of input features

//input data

value_record := RECORD
real8	f1	;
real8	f2	;
real8	f3	;
real8	f4	;
real8	f5	;
real8	f6	;
real8	f7	;
real8	f8	;
real8	f9	;
real8	f10	;
real8	f11	;
real8	f12	;
real8	f13	;
real8	f14	;
real8	f15	;
real8	f16	;
real8	f17	;
real8	f18	;
real8	f19	;
real8	f20	;
real8	f21	;
real8	f22	;
real8	f23	;
real8	f24	;
real8	f25	;
real8	f26	;
real8	f27	;
real8	f28	;
real8	f29	;
real8	f30	;
real8	f31	;
real8	f32	;
real8	f33	;
real8	f34	;
real8	f35	;
real8	f36	;
real8	f37	;
real8	f38	;
real8	f39	;
real8	f40	;
real8	f41	;
real8	f42	;
real8	f43	;
real8	f44	;
real8	f45	;
real8	f46	;
real8	f47	;
real8	f48	;
real8	f49	;
real8	f50	;
real8	f51	;
real8	f52	;
real8	f53	;
real8	f54	;
real8	f55	;
real8	f56	;
real8	f57	;
real8	f58	;
real8	f59	;
real8	f60	;
real8	f61	;
real8	f62	;
real8	f63	;
real8	f64	;
real8	f65	;
real8	f66	;
real8	f67	;
real8	f68	;
real8	f69	;
real8	f70	;
real8	f71	;
real8	f72	;
real8	f73	;
real8	f74	;
real8	f75	;
real8	f76	;
real8	f77	;
real8	f78	;
real8	f79	;
real8	f80	;
real8	f81	;
real8	f82	;
real8	f83	;
real8	f84	;
real8	f85	;
real8	f86	;
real8	f87	;
real8	f88	;
real8	f89	;
real8	f90	;
real8	f91	;
real8	f92	;
real8	f93	;
real8	f94	;
real8	f95	;
real8	f96	;
real8	f97	;
real8	f98	;
real8	f99	;
real8	f100	;
real8	f101	;
real8	f102	;
real8	f103	;
real8	f104	;
real8	f105	;
real8	f106	;
real8	f107	;
real8	f108	;
real8	f109	;
real8	f110	;
real8	f111	;
real8	f112	;
real8	f113	;
real8	f114	;
real8	f115	;
real8	f116	;
real8	f117	;
real8	f118	;
real8	f119	;
real8	f120	;
real8	f121	;
real8	f122	;
real8	f123	;
real8	f124	;
real8	f125	;
real8	f126	;
real8	f127	;
real8	f128	;
real8	f129	;
real8	f130	;
real8	f131	;
real8	f132	;
real8	f133	;
real8	f134	;
real8	f135	;
real8	f136	;
real8	f137	;
real8	f138	;
real8	f139	;
real8	f140	;
real8	f141	;
real8	f142	;
real8	f143	;
real8	f144	;
real8	f145	;
real8	f146	;
real8	f147	;
real8	f148	;
real8	f149	;
real8	f150	;
real8	f151	;
real8	f152	;
real8	f153	;
real8	f154	;
real8	f155	;
real8	f156	;
real8	f157	;
real8	f158	;
real8	f159	;
real8	f160	;
real8	f161	;
real8	f162	;
real8	f163	;
real8	f164	;
real8	f165	;
real8	f166	;
real8	f167	;
real8	f168	;
real8	f169	;
real8	f170	;
real8	f171	;
real8	f172	;
real8	f173	;
real8	f174	;
real8	f175	;
real8	f176	;
real8	f177	;
real8	f178	;
real8	f179	;
real8	f180	;
real8	f181	;
real8	f182	;
real8	f183	;
real8	f184	;
real8	f185	;
real8	f186	;
real8	f187	;
real8	f188	;
real8	f189	;
real8	f190	;
real8	f191	;
real8	f192	;
real8	f193	;
real8	f194	;
real8	f195	;
real8	f196	;
real8	f197	;
real8	f198	;
real8	f199	;
real8	f200	;
real8	f201	;
real8	f202	;
real8	f203	;
real8	f204	;
real8	f205	;
real8	f206	;
real8	f207	;
real8	f208	;
real8	f209	;
real8	f210	;
real8	f211	;
real8	f212	;
real8	f213	;
real8	f214	;
real8	f215	;
real8	f216	;
real8	f217	;
real8	f218	;
real8	f219	;
real8	f220	;
real8	f221	;
real8	f222	;
real8	f223	;
real8	f224	;
real8	f225	;
real8	f226	;
real8	f227	;
real8	f228	;
real8	f229	;
real8	f230	;
real8	f231	;
real8	f232	;
real8	f233	;
real8	f234	;
real8	f235	;
real8	f236	;
real8	f237	;
real8	f238	;
real8	f239	;
real8	f240	;
real8	f241	;
real8	f242	;
real8	f243	;
real8	f244	;
real8	f245	;
real8	f246	;
real8	f247	;
real8	f248	;
real8	f249	;
real8	f250	;
real8	f251	;
real8	f252	;
real8	f253	;
real8	f254	;
real8	f255	;
real8	f256	;
real8	f257	;
real8	f258	;
real8	f259	;
real8	f260	;
real8	f261	;
real8	f262	;
real8	f263	;
real8	f264	;
real8	f265	;
real8	f266	;
real8	f267	;
real8	f268	;
real8	f269	;
real8	f270	;
real8	f271	;
real8	f272	;
real8	f273	;
real8	f274	;
real8	f275	;
real8	f276	;
real8	f277	;
real8	f278	;
real8	f279	;
real8	f280	;
real8	f281	;
real8	f282	;
real8	f283	;
real8	f284	;
real8	f285	;
real8	f286	;
real8	f287	;
real8	f288	;
real8	f289	;
real8	f290	;
real8	f291	;
real8	f292	;
real8	f293	;
real8	f294	;
real8	f295	;
real8	f296	;
real8	f297	;
real8	f298	;
real8	f299	;
real8	f300	;
real8	f301	;
real8	f302	;
real8	f303	;
real8	f304	;
real8	f305	;
real8	f306	;
real8	f307	;
real8	f308	;
real8	f309	;
real8	f310	;
real8	f311	;
real8	f312	;
real8	f313	;
real8	f314	;
real8	f315	;
real8	f316	;
real8	f317	;
real8	f318	;
real8	f319	;
real8	f320	;
real8	f321	;
real8	f322	;
real8	f323	;
real8	f324	;
real8	f325	;
real8	f326	;
real8	f327	;
real8	f328	;
real8	f329	;
real8	f330	;
real8	f331	;
real8	f332	;
real8	f333	;
real8	f334	;
real8	f335	;
real8	f336	;
real8	f337	;
real8	f338	;
real8	f339	;
real8	f340	;
real8	f341	;
real8	f342	;
real8	f343	;
real8	f344	;
real8	f345	;
real8	f346	;
real8	f347	;
real8	f348	;
real8	f349	;
real8	f350	;
real8	f351	;
real8	f352	;
real8	f353	;
real8	f354	;
real8	f355	;
real8	f356	;
real8	f357	;
real8	f358	;
real8	f359	;
real8	f360	;
real8	f361	;
real8	f362	;
real8	f363	;
real8	f364	;
real8	f365	;
real8	f366	;
real8	f367	;
real8	f368	;
real8	f369	;
real8	f370	;
real8	f371	;
real8	f372	;
real8	f373	;
real8	f374	;
real8	f375	;
real8	f376	;
real8	f377	;
real8	f378	;
real8	f379	;
real8	f380	;
real8	f381	;
real8	f382	;
real8	f383	;
real8	f384	;
real8	f385	;
real8	f386	;
real8	f387	;
real8	f388	;
real8	f389	;
real8	f390	;
real8	f391	;
real8	f392	;
real8	f393	;
real8	f394	;
real8	f395	;
real8	f396	;
real8	f397	;
real8	f398	;
real8	f399	;
real8	f400	;
real8	f401	;
real8	f402	;
real8	f403	;
real8	f404	;
real8	f405	;
real8	f406	;
real8	f407	;
real8	f408	;
real8	f409	;
real8	f410	;
real8	f411	;
real8	f412	;
real8	f413	;
real8	f414	;
real8	f415	;
real8	f416	;
real8	f417	;
real8	f418	;
real8	f419	;
real8	f420	;
real8	f421	;
real8	f422	;
real8	f423	;
real8	f424	;
real8	f425	;
real8	f426	;
real8	f427	;
real8	f428	;
real8	f429	;
real8	f430	;
real8	f431	;
real8	f432	;
real8	f433	;
real8	f434	;
real8	f435	;
real8	f436	;
real8	f437	;
real8	f438	;
real8	f439	;
real8	f440	;
real8	f441	;
real8	f442	;
real8	f443	;
real8	f444	;
real8	f445	;
real8	f446	;
real8	f447	;
real8	f448	;
real8	f449	;
real8	f450	;
real8	f451	;
real8	f452	;
real8	f453	;
real8	f454	;
real8	f455	;
real8	f456	;
real8	f457	;
real8	f458	;
real8	f459	;
real8	f460	;
real8	f461	;
real8	f462	;
real8	f463	;
real8	f464	;
real8	f465	;
real8	f466	;
real8	f467	;
real8	f468	;
real8	f469	;
real8	f470	;
real8	f471	;
real8	f472	;
real8	f473	;
real8	f474	;
real8	f475	;
real8	f476	;
real8	f477	;
real8	f478	;
real8	f479	;
real8	f480	;
real8	f481	;
real8	f482	;
real8	f483	;
real8	f484	;
real8	f485	;
real8	f486	;
real8	f487	;
real8	f488	;
real8	f489	;
real8	f490	;
real8	f491	;
real8	f492	;
real8	f493	;
real8	f494	;
real8	f495	;
real8	f496	;
real8	f497	;
real8	f498	;
real8	f499	;
real8	f500	;
real8	f501	;
real8	f502	;
real8	f503	;
real8	f504	;
real8	f505	;
real8	f506	;
real8	f507	;
real8	f508	;
real8	f509	;
real8	f510	;
real8	f511	;
real8	f512	;
real8	f513	;
real8	f514	;
real8	f515	;
real8	f516	;
real8	f517	;
real8	f518	;
real8	f519	;
real8	f520	;
real8	f521	;
real8	f522	;
real8	f523	;
real8	f524	;
real8	f525	;
real8	f526	;
real8	f527	;
real8	f528	;
real8	f529	;
real8	f530	;
real8	f531	;
real8	f532	;
real8	f533	;
real8	f534	;
real8	f535	;
real8	f536	;
real8	f537	;
real8	f538	;
real8	f539	;
real8	f540	;
real8	f541	;
real8	f542	;
real8	f543	;
real8	f544	;
real8	f545	;
real8	f546	;
real8	f547	;
real8	f548	;
real8	f549	;
real8	f550	;
real8	f551	;
real8	f552	;
real8	f553	;
real8	f554	;
real8	f555	;
real8	f556	;
real8	f557	;
real8	f558	;
real8	f559	;
real8	f560	;
real8	f561	;
real8	f562	;
real8	f563	;
real8	f564	;
real8	f565	;
real8	f566	;
real8	f567	;
real8	f568	;
real8	f569	;
real8	f570	;
real8	f571	;
real8	f572	;
real8	f573	;
real8	f574	;
real8	f575	;
real8	f576	;
real8	f577	;
real8	f578	;
real8	f579	;
real8	f580	;
real8	f581	;
real8	f582	;
real8	f583	;
real8	f584	;
real8	f585	;
real8	f586	;
real8	f587	;
real8	f588	;
real8	f589	;
real8	f590	;
real8	f591	;
real8	f592	;
real8	f593	;
real8	f594	;
real8	f595	;
real8	f596	;
real8	f597	;
real8	f598	;
real8	f599	;
real8	f600	;
real8	f601	;
real8	f602	;
real8	f603	;
real8	f604	;
real8	f605	;
real8	f606	;
real8	f607	;
real8	f608	;
real8	f609	;
real8	f610	;
real8	f611	;
real8	f612	;
real8	f613	;
real8	f614	;
real8	f615	;
real8	f616	;
real8	f617	;
real8	f618	;
real8	f619	;
real8	f620	;
real8	f621	;
real8	f622	;
real8	f623	;
real8	f624	;
real8	f625	;
real8	f626	;
real8	f627	;
real8	f628	;
real8	f629	;
real8	f630	;
real8	f631	;
real8	f632	;
real8	f633	;
real8	f634	;
real8	f635	;
real8	f636	;
real8	f637	;
real8	f638	;
real8	f639	;
real8	f640	;
real8	f641	;
real8	f642	;
real8	f643	;
real8	f644	;
real8	f645	;
real8	f646	;
real8	f647	;
real8	f648	;
real8	f649	;
real8	f650	;
real8	f651	;
real8	f652	;
real8	f653	;
real8	f654	;
real8	f655	;
real8	f656	;
real8	f657	;
real8	f658	;
real8	f659	;
real8	f660	;
real8	f661	;
real8	f662	;
real8	f663	;
real8	f664	;
real8	f665	;
real8	f666	;
real8	f667	;
real8	f668	;
real8	f669	;
real8	f670	;
real8	f671	;
real8	f672	;
real8	f673	;
real8	f674	;
real8	f675	;
real8	f676	;
real8	f677	;
real8	f678	;
real8	f679	;
real8	f680	;
real8	f681	;
real8	f682	;
real8	f683	;
real8	f684	;
real8	f685	;
real8	f686	;
real8	f687	;
real8	f688	;
real8	f689	;
real8	f690	;
real8	f691	;
real8	f692	;
real8	f693	;
real8	f694	;
real8	f695	;
real8	f696	;
real8	f697	;
real8	f698	;
real8	f699	;
real8	f700	;
real8	f701	;
real8	f702	;
real8	f703	;
real8	f704	;
real8	f705	;
real8	f706	;
real8	f707	;
real8	f708	;
real8	f709	;
real8	f710	;
real8	f711	;
real8	f712	;
real8	f713	;
real8	f714	;
real8	f715	;
real8	f716	;
real8	f717	;
real8	f718	;
real8	f719	;
real8	f720	;
real8	f721	;
real8	f722	;
real8	f723	;
real8	f724	;
real8	f725	;
real8	f726	;
real8	f727	;
real8	f728	;
real8	f729	;
real8	f730	;
real8	f731	;
real8	f732	;
real8	f733	;
real8	f734	;
real8	f735	;
real8	f736	;
real8	f737	;
real8	f738	;
real8	f739	;
real8	f740	;
real8	f741	;
real8	f742	;
real8	f743	;
real8	f744	;
real8	f745	;
real8	f746	;
real8	f747	;
real8	f748	;
real8	f749	;
real8	f750	;
real8	f751	;
real8	f752	;
real8	f753	;
real8	f754	;
real8	f755	;
real8	f756	;
real8	f757	;
real8	f758	;
real8	f759	;
real8	f760	;
real8	f761	;
real8	f762	;
real8	f763	;
real8	f764	;
real8	f765	;
real8	f766	;
real8	f767	;
real8	f768	;
real8	f769	;
real8	f770	;
real8	f771	;
real8	f772	;
real8	f773	;
real8	f774	;
real8	f775	;
real8	f776	;
real8	f777	;
real8	f778	;
real8	f779	;
real8	f780	;
real8	f781	;
real8	f782	;
real8	f783	;
real8	f784	;
END;

input_data_tmp := DATASET('~maryam::mytest::mnist_5digits_traindata', value_record, CSV); // This dataset is a subset of MNIST dtaset that includes 5 digits (0 to 4), it is used for traibn
ML.AppendID(input_data_tmp, id, input_data);
sample_table := input_data;
ML.ToField(sample_table, indepDataC);
//define the parameters for the Sparse Autoencoder
REAL8 sparsityParam  := 0.1;
REAL8 BETA := 3;
REAL8 ALPHA := 0.1;
REAL8 LAMBDA :=0.003;
UNSIGNED2 MaxIter :=1;
UNSIGNED4 prows:=0;
UNSIGNED4 pcols:=0;
UNSIGNED4 Maxrows:=0;
UNSIGNED4 Maxcols:=0;
//initialize weight and bias values for the Back Propagation algorithm
IntW := DeepLearning.Sparse_Autoencoder_IntWeights(f,hl);
Intb := DeepLearning.Sparse_Autoencoder_IntBias(f,hl);
//SA_mine4_1 :=DeepLearning.Sparse_Autoencoder_lbfgs (f,hl,40,7649,40,7649);
//SA_mine4_1 :=DeepLearning.Sparse_Autoencoder_lbfgs (f,hl,16,15298,16,15298);
//SA_mine4_1 :=DeepLearning.Sparse_Autoencoder_lbfgs (f,hl,f,306,f,306);
SA_mine4_1 :=DeepLearning.Sparse_Autoencoder_lbfgs_part (f,hl,98,306);
IntW1 := Mat.MU.From(IntW,1);
IntW2 := Mat.MU.From(IntW,2);
Intb1 := Mat.MU.From(Intb,1);
Intb2 := Mat.MU.From(Intb,2);
// OUTPUT(IntW1,ALL, named ('IntW1'));
// OUTPUT(IntW2,ALL, named ('IntW2'));
// OUTPUT(IntB1,ALL, named ('IntB1'));
// OUTPUT(IntB2,ALL, named ('IntB2'));
// train the sparse autoencoer with train data
lbfgs_model_mine4_1 := SA_mine4_1.LearnC(indepDataC,IntW1, IntW2, Intb1, Intb2, BETA,sparsityParam ,LAMBDA, MaxIter);//the output includes the learnt parameters for the sparse autoencoder (W1,W2,b1,b2) in numericfield format

//OUTPUT (MAX(input_data, input_data.id));
// OUTPUT(lbfgs_model_mine4_1);
 //OUTPUT(lbfgs_model_mine4_1,,'~thor::maryam::mytest::5digist_newopt2',CSV(HEADING(SINGLE)), OVERWRITE);
 // mymy2 := lbfgs_model_mine4_1;
 // myformat2 := RECORD
    // mymy2.node_id;
    // mymy2.partition_id;
    // mymy2.block_row;
    // mymy2.block_col;
    // mymy2.first_row;
    // mymy2.part_rows;
    // mymy2.first_col;
    // mymy2.part_cols;
		// mymy2.cost_value;
	//	mymy2.mat_part;
		// INTEGER real_node := STD.System.Thorlib.Node();
// END;
	// rslt := TABLE(mymy2,myformat2,LOCAL); 
	// OUTPUT(rslt);

// OUTPUT(rslt);
// OUTPUT(mymy2);
// OUTPUT(lbfgs_model_mine4_1[1]);
// OUTPUT(lbfgs_model_mine4_1[2]);
// OUTPUT(lbfgs_model_mine4_1[3]);
// OUTPUT(lbfgs_model_mine4_1[4]);
// OUTPUT(lbfgs_model_mine4_1[5]);
// OUTPUT(lbfgs_model_mine4_1[6]);
// OUTPUT(lbfgs_model_mine4_1[7]);
// OUTPUT(lbfgs_model_mine4_1[8]);
//OUTPUT(MAX(input_data, input_data.id));

 // MatrixModel := SA_mine4_1.Model (lbfgs_model_mine4_1);//convert the model to matrix format where no=1 is W1, no=2 is W2, no=3 is b1 and no=4 is b2
// OUTPUT(MatrixModel, named ('MatrixModel'));
 Extractedweights := SA_mine4_1.ExtractWeights (lbfgs_model_mine4_1);
// OUTPUT(Extractedweights, named ('Extractedweights'));
 ExtractedBias := SA_mine4_1.ExtractBias (lbfgs_model_mine4_1);
// OUTPUT(ExtractedBias, named ('ExtractedBias'));
W1_matrix := ML.Mat.MU.FROM(Extractedweights,1);
OUTPUT(W1_matrix, NAMED('W1_matrix'));
W2_matrix := ML.Mat.MU.FROM(Extractedweights,2);
OUTPUT(W2_matrix, NAMED('W2_matrix'));
b1_matrix := ML.Mat.MU.FROM(ExtractedBias,1);
OUTPUT(b1_matrix, NAMED('b1_matrix'));
b2_matrix := ML.Mat.MU.FROM(ExtractedBias,2);
OUTPUT(b2_matrix, NAMED('b2_matrix'));
OUTPUT(SA_mine4_1.extractcost_funeval(lbfgs_model_mine4_1));
// OUTPUT(W1_matrix,,'~thor::maryam::mytest::W1_matrix_MNIST_5digits_newopt',CSV(HEADING(SINGLE)));//3_bbs means after implementing big_big_smal and all others to calculate sparseautoencoder
// OUTPUT(W2_matrix,,'~thor::maryam::mytest::W2_matrix_MNIST_5digits_newopt',CSV(HEADING(SINGLE)));
// OUTPUT(b1_matrix,,'~thor::maryam::mytest::b1_matrix_MNIST_5digits_newopt',CSV(HEADING(SINGLE)));
// OUTPUT(b2_matrix,,'~thor::maryam::mytest::b2_matrix_MNIST_5digits_newopt',CSV(HEADING(SINGLE)));