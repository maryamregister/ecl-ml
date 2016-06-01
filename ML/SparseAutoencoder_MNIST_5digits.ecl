﻿IMPORT * FROM ML;
IMPORT * FROM $;
IMPORT PBblas;
Layout_Cell := PBblas.Types.Layout_Cell;
//This is not test on MNIST, this is test on image patches

INTEGER4 hl := 25;//number of nodes in the hiddenlayer
INTEGER4 f := 28*28;//number of input features

//input data

value_record := RECORD
real	f1	;
real	f2	;
real	f3	;
real	f4	;
real	f5	;
real	f6	;
real	f7	;
real	f8	;
real	f9	;
real	f10	;
real	f11	;
real	f12	;
real	f13	;
real	f14	;
real	f15	;
real	f16	;
real	f17	;
real	f18	;
real	f19	;
real	f20	;
real	f21	;
real	f22	;
real	f23	;
real	f24	;
real	f25	;
real	f26	;
real	f27	;
real	f28	;
real	f29	;
real	f30	;
real	f31	;
real	f32	;
real	f33	;
real	f34	;
real	f35	;
real	f36	;
real	f37	;
real	f38	;
real	f39	;
real	f40	;
real	f41	;
real	f42	;
real	f43	;
real	f44	;
real	f45	;
real	f46	;
real	f47	;
real	f48	;
real	f49	;
real	f50	;
real	f51	;
real	f52	;
real	f53	;
real	f54	;
real	f55	;
real	f56	;
real	f57	;
real	f58	;
real	f59	;
real	f60	;
real	f61	;
real	f62	;
real	f63	;
real	f64	;
real	f65	;
real	f66	;
real	f67	;
real	f68	;
real	f69	;
real	f70	;
real	f71	;
real	f72	;
real	f73	;
real	f74	;
real	f75	;
real	f76	;
real	f77	;
real	f78	;
real	f79	;
real	f80	;
real	f81	;
real	f82	;
real	f83	;
real	f84	;
real	f85	;
real	f86	;
real	f87	;
real	f88	;
real	f89	;
real	f90	;
real	f91	;
real	f92	;
real	f93	;
real	f94	;
real	f95	;
real	f96	;
real	f97	;
real	f98	;
real	f99	;
real	f100	;
real	f101	;
real	f102	;
real	f103	;
real	f104	;
real	f105	;
real	f106	;
real	f107	;
real	f108	;
real	f109	;
real	f110	;
real	f111	;
real	f112	;
real	f113	;
real	f114	;
real	f115	;
real	f116	;
real	f117	;
real	f118	;
real	f119	;
real	f120	;
real	f121	;
real	f122	;
real	f123	;
real	f124	;
real	f125	;
real	f126	;
real	f127	;
real	f128	;
real	f129	;
real	f130	;
real	f131	;
real	f132	;
real	f133	;
real	f134	;
real	f135	;
real	f136	;
real	f137	;
real	f138	;
real	f139	;
real	f140	;
real	f141	;
real	f142	;
real	f143	;
real	f144	;
real	f145	;
real	f146	;
real	f147	;
real	f148	;
real	f149	;
real	f150	;
real	f151	;
real	f152	;
real	f153	;
real	f154	;
real	f155	;
real	f156	;
real	f157	;
real	f158	;
real	f159	;
real	f160	;
real	f161	;
real	f162	;
real	f163	;
real	f164	;
real	f165	;
real	f166	;
real	f167	;
real	f168	;
real	f169	;
real	f170	;
real	f171	;
real	f172	;
real	f173	;
real	f174	;
real	f175	;
real	f176	;
real	f177	;
real	f178	;
real	f179	;
real	f180	;
real	f181	;
real	f182	;
real	f183	;
real	f184	;
real	f185	;
real	f186	;
real	f187	;
real	f188	;
real	f189	;
real	f190	;
real	f191	;
real	f192	;
real	f193	;
real	f194	;
real	f195	;
real	f196	;
real	f197	;
real	f198	;
real	f199	;
real	f200	;
real	f201	;
real	f202	;
real	f203	;
real	f204	;
real	f205	;
real	f206	;
real	f207	;
real	f208	;
real	f209	;
real	f210	;
real	f211	;
real	f212	;
real	f213	;
real	f214	;
real	f215	;
real	f216	;
real	f217	;
real	f218	;
real	f219	;
real	f220	;
real	f221	;
real	f222	;
real	f223	;
real	f224	;
real	f225	;
real	f226	;
real	f227	;
real	f228	;
real	f229	;
real	f230	;
real	f231	;
real	f232	;
real	f233	;
real	f234	;
real	f235	;
real	f236	;
real	f237	;
real	f238	;
real	f239	;
real	f240	;
real	f241	;
real	f242	;
real	f243	;
real	f244	;
real	f245	;
real	f246	;
real	f247	;
real	f248	;
real	f249	;
real	f250	;
real	f251	;
real	f252	;
real	f253	;
real	f254	;
real	f255	;
real	f256	;
real	f257	;
real	f258	;
real	f259	;
real	f260	;
real	f261	;
real	f262	;
real	f263	;
real	f264	;
real	f265	;
real	f266	;
real	f267	;
real	f268	;
real	f269	;
real	f270	;
real	f271	;
real	f272	;
real	f273	;
real	f274	;
real	f275	;
real	f276	;
real	f277	;
real	f278	;
real	f279	;
real	f280	;
real	f281	;
real	f282	;
real	f283	;
real	f284	;
real	f285	;
real	f286	;
real	f287	;
real	f288	;
real	f289	;
real	f290	;
real	f291	;
real	f292	;
real	f293	;
real	f294	;
real	f295	;
real	f296	;
real	f297	;
real	f298	;
real	f299	;
real	f300	;
real	f301	;
real	f302	;
real	f303	;
real	f304	;
real	f305	;
real	f306	;
real	f307	;
real	f308	;
real	f309	;
real	f310	;
real	f311	;
real	f312	;
real	f313	;
real	f314	;
real	f315	;
real	f316	;
real	f317	;
real	f318	;
real	f319	;
real	f320	;
real	f321	;
real	f322	;
real	f323	;
real	f324	;
real	f325	;
real	f326	;
real	f327	;
real	f328	;
real	f329	;
real	f330	;
real	f331	;
real	f332	;
real	f333	;
real	f334	;
real	f335	;
real	f336	;
real	f337	;
real	f338	;
real	f339	;
real	f340	;
real	f341	;
real	f342	;
real	f343	;
real	f344	;
real	f345	;
real	f346	;
real	f347	;
real	f348	;
real	f349	;
real	f350	;
real	f351	;
real	f352	;
real	f353	;
real	f354	;
real	f355	;
real	f356	;
real	f357	;
real	f358	;
real	f359	;
real	f360	;
real	f361	;
real	f362	;
real	f363	;
real	f364	;
real	f365	;
real	f366	;
real	f367	;
real	f368	;
real	f369	;
real	f370	;
real	f371	;
real	f372	;
real	f373	;
real	f374	;
real	f375	;
real	f376	;
real	f377	;
real	f378	;
real	f379	;
real	f380	;
real	f381	;
real	f382	;
real	f383	;
real	f384	;
real	f385	;
real	f386	;
real	f387	;
real	f388	;
real	f389	;
real	f390	;
real	f391	;
real	f392	;
real	f393	;
real	f394	;
real	f395	;
real	f396	;
real	f397	;
real	f398	;
real	f399	;
real	f400	;
real	f401	;
real	f402	;
real	f403	;
real	f404	;
real	f405	;
real	f406	;
real	f407	;
real	f408	;
real	f409	;
real	f410	;
real	f411	;
real	f412	;
real	f413	;
real	f414	;
real	f415	;
real	f416	;
real	f417	;
real	f418	;
real	f419	;
real	f420	;
real	f421	;
real	f422	;
real	f423	;
real	f424	;
real	f425	;
real	f426	;
real	f427	;
real	f428	;
real	f429	;
real	f430	;
real	f431	;
real	f432	;
real	f433	;
real	f434	;
real	f435	;
real	f436	;
real	f437	;
real	f438	;
real	f439	;
real	f440	;
real	f441	;
real	f442	;
real	f443	;
real	f444	;
real	f445	;
real	f446	;
real	f447	;
real	f448	;
real	f449	;
real	f450	;
real	f451	;
real	f452	;
real	f453	;
real	f454	;
real	f455	;
real	f456	;
real	f457	;
real	f458	;
real	f459	;
real	f460	;
real	f461	;
real	f462	;
real	f463	;
real	f464	;
real	f465	;
real	f466	;
real	f467	;
real	f468	;
real	f469	;
real	f470	;
real	f471	;
real	f472	;
real	f473	;
real	f474	;
real	f475	;
real	f476	;
real	f477	;
real	f478	;
real	f479	;
real	f480	;
real	f481	;
real	f482	;
real	f483	;
real	f484	;
real	f485	;
real	f486	;
real	f487	;
real	f488	;
real	f489	;
real	f490	;
real	f491	;
real	f492	;
real	f493	;
real	f494	;
real	f495	;
real	f496	;
real	f497	;
real	f498	;
real	f499	;
real	f500	;
real	f501	;
real	f502	;
real	f503	;
real	f504	;
real	f505	;
real	f506	;
real	f507	;
real	f508	;
real	f509	;
real	f510	;
real	f511	;
real	f512	;
real	f513	;
real	f514	;
real	f515	;
real	f516	;
real	f517	;
real	f518	;
real	f519	;
real	f520	;
real	f521	;
real	f522	;
real	f523	;
real	f524	;
real	f525	;
real	f526	;
real	f527	;
real	f528	;
real	f529	;
real	f530	;
real	f531	;
real	f532	;
real	f533	;
real	f534	;
real	f535	;
real	f536	;
real	f537	;
real	f538	;
real	f539	;
real	f540	;
real	f541	;
real	f542	;
real	f543	;
real	f544	;
real	f545	;
real	f546	;
real	f547	;
real	f548	;
real	f549	;
real	f550	;
real	f551	;
real	f552	;
real	f553	;
real	f554	;
real	f555	;
real	f556	;
real	f557	;
real	f558	;
real	f559	;
real	f560	;
real	f561	;
real	f562	;
real	f563	;
real	f564	;
real	f565	;
real	f566	;
real	f567	;
real	f568	;
real	f569	;
real	f570	;
real	f571	;
real	f572	;
real	f573	;
real	f574	;
real	f575	;
real	f576	;
real	f577	;
real	f578	;
real	f579	;
real	f580	;
real	f581	;
real	f582	;
real	f583	;
real	f584	;
real	f585	;
real	f586	;
real	f587	;
real	f588	;
real	f589	;
real	f590	;
real	f591	;
real	f592	;
real	f593	;
real	f594	;
real	f595	;
real	f596	;
real	f597	;
real	f598	;
real	f599	;
real	f600	;
real	f601	;
real	f602	;
real	f603	;
real	f604	;
real	f605	;
real	f606	;
real	f607	;
real	f608	;
real	f609	;
real	f610	;
real	f611	;
real	f612	;
real	f613	;
real	f614	;
real	f615	;
real	f616	;
real	f617	;
real	f618	;
real	f619	;
real	f620	;
real	f621	;
real	f622	;
real	f623	;
real	f624	;
real	f625	;
real	f626	;
real	f627	;
real	f628	;
real	f629	;
real	f630	;
real	f631	;
real	f632	;
real	f633	;
real	f634	;
real	f635	;
real	f636	;
real	f637	;
real	f638	;
real	f639	;
real	f640	;
real	f641	;
real	f642	;
real	f643	;
real	f644	;
real	f645	;
real	f646	;
real	f647	;
real	f648	;
real	f649	;
real	f650	;
real	f651	;
real	f652	;
real	f653	;
real	f654	;
real	f655	;
real	f656	;
real	f657	;
real	f658	;
real	f659	;
real	f660	;
real	f661	;
real	f662	;
real	f663	;
real	f664	;
real	f665	;
real	f666	;
real	f667	;
real	f668	;
real	f669	;
real	f670	;
real	f671	;
real	f672	;
real	f673	;
real	f674	;
real	f675	;
real	f676	;
real	f677	;
real	f678	;
real	f679	;
real	f680	;
real	f681	;
real	f682	;
real	f683	;
real	f684	;
real	f685	;
real	f686	;
real	f687	;
real	f688	;
real	f689	;
real	f690	;
real	f691	;
real	f692	;
real	f693	;
real	f694	;
real	f695	;
real	f696	;
real	f697	;
real	f698	;
real	f699	;
real	f700	;
real	f701	;
real	f702	;
real	f703	;
real	f704	;
real	f705	;
real	f706	;
real	f707	;
real	f708	;
real	f709	;
real	f710	;
real	f711	;
real	f712	;
real	f713	;
real	f714	;
real	f715	;
real	f716	;
real	f717	;
real	f718	;
real	f719	;
real	f720	;
real	f721	;
real	f722	;
real	f723	;
real	f724	;
real	f725	;
real	f726	;
real	f727	;
real	f728	;
real	f729	;
real	f730	;
real	f731	;
real	f732	;
real	f733	;
real	f734	;
real	f735	;
real	f736	;
real	f737	;
real	f738	;
real	f739	;
real	f740	;
real	f741	;
real	f742	;
real	f743	;
real	f744	;
real	f745	;
real	f746	;
real	f747	;
real	f748	;
real	f749	;
real	f750	;
real	f751	;
real	f752	;
real	f753	;
real	f754	;
real	f755	;
real	f756	;
real	f757	;
real	f758	;
real	f759	;
real	f760	;
real	f761	;
real	f762	;
real	f763	;
real	f764	;
real	f765	;
real	f766	;
real	f767	;
real	f768	;
real	f769	;
real	f770	;
real	f771	;
real	f772	;
real	f773	;
real	f774	;
real	f775	;
real	f776	;
real	f777	;
real	f778	;
real	f779	;
real	f780	;
real	f781	;
real	f782	;
real	f783	;
real	f784	;
END;

input_data_tmp := DATASET('~maryam::mytest::mnist_5digits_traindata', value_record, CSV); // This dataset is a subset of MNISt dtaset that includes 5 digits


OUTPUT(input_data_tmp, NAMED('input_data_tmp'));

ML.AppendID(input_data_tmp, id, input_data);
//OUTPUT  (input_data, NAMED ('input_data'));



sample_table := input_data;
//OUTPUT  (sample_table, NAMED ('sample_table'));

ML.ToField(sample_table, indepDataC);
//OUTPUT  (indepDataC, NAMED ('indepDataC'));


//define the parameters for the Sparse Autoencoder
//ALPHA is learning rate
//LAMBDA is weight decay rate
REAL8 sparsityParam  := 0.1;
REAL8 BETA := 3;
REAL8 ALPHA := 0.1;
REAL8 LAMBDA :=0.003;
UNSIGNED2 MaxIter :=400;
UNSIGNED4 prows:=0;
UNSIGNED4 pcols:=0;
UNSIGNED4 Maxrows:=0;
UNSIGNED4 Maxcols:=0;
//initialize weight and bias values for the Back Propagation algorithm
IntW := DeepLearning4_1.Sparse_Autoencoder_IntWeights(f,hl);
Intb := DeepLearning4_1.Sparse_Autoencoder_IntBias(f,hl);
output(IntW, named ('IntW'));
output(IntB, named ('IntB'));
//trainer module
// SA :=DeepLearning.Sparse_Autoencoder(f,hl,prows, pcols, Maxrows,  Maxcols);

// LearntModel := SA.LearnC(indepDataC,IntW, Intb,BETA, sparsityParam, LAMBDA, ALPHA, MaxIter);
// mout := max(LearntModel,id);
//output(LearntModel(id=1));

// MatrixModel := SA.Model (LearntModel);
// output(MatrixModel, named ('MatrixModel'));

// Out := SA.SAOutput (indepDataC, LearntModel);
// output(Out, named ('Out'));


//MINE
SA_mine4_1 :=DeepLearning4_1.Sparse_Autoencoder_mine (f, hl, 0, 0,0,0);
IntW1 := Mat.MU.From(IntW,1);

IntW2 := Mat.MU.From(IntW,2);


Intb1 := Mat.MU.From(Intb,1);

Intb2 := Mat.MU.From(Intb,2);


// OUTPUT(IntW1,ALL, named ('IntW1'));
// OUTPUT(IntW2,ALL, named ('IntW2'));
// OUTPUT(IntB1,ALL, named ('IntB1'));
// OUTPUT(IntB2,ALL, named ('IntB2'));

// lbfgs_model_mine := SA_mine.LearnC_lbfgs(indepDataC,IntW1, IntW2, Intb1, Intb2, BETA,sparsityParam ,LAMBDA, 200);
// lbfgs_model_mine4 := SA_mine4.LearnC_lbfgs_4(indepDataC,IntW1, IntW2, Intb1, Intb2, BETA,sparsityParam ,LAMBDA, 200);
lbfgs_model_mine4_1 := SA_mine4_1.LearnC_lbfgs_4(indepDataC,IntW1, IntW2, Intb1, Intb2, BETA,sparsityParam ,LAMBDA, MaxIter);

OUTPUT(lbfgs_model_mine4_1);
//extract theta
maxno := MAX(lbfgs_model_mine4_1, lbfgs_model_mine4_1.no);
THETA_ := lbfgs_model_mine4_1(no=maxno);
OUTPUT(MAXno,NAMED('maxno'));
THETA_part := PROJECT(THETA_, TRANSFORM(PBblas.Types.Layout_Part, SELF := LEFT));

PBblas.Types.Layout_Part part_1(PBblas.Types.Layout_Part l ) := TRANSFORM
    SELF.partition_id := 1;
    SELF := l;
END;
W1_part := THETA_part(partition_id =1);

W2_part_ := THETA_part(partition_id =2);
w2_part := PROJECT(W2_part_, part_1(LEFT));

b1_part_ := THETA_part(partition_id =3);
b1_part := PROJECT(b1_part_, part_1(LEFT));

b2_part_ := THETA_part(partition_id =4);
b2_part := PROJECT(b2_part_, part_1(LEFT));


W1_matrix := ML.DMat.Converted.FromPart2Elm(W1_part);
OUTPUT(W1_matrix, NAMED('W1_matrix'));

W2_matrix := ML.DMat.Converted.FromPart2Elm(W2_part);
OUTPUT(W2_matrix, NAMED('W2_matrix'));

b1_matrix := ML.DMat.Converted.FromPart2Elm(b1_part);
OUTPUT(b1_matrix, NAMED('b1_matrix'));

b2_matrix := ML.DMat.Converted.FromPart2Elm(b2_part);
OUTPUT(b2_matrix, NAMED('b2_matrix'));


OUTPUT(W1_matrix,,'~thor::maryam::mytest::W1_matrix_MNIST_5digits1.csv',CSV(HEADING(SINGLE)));
OUTPUT(W2_matrix,,'~thor::maryam::mytest::W2_matrix_MNIST_5digits1.csv',CSV(HEADING(SINGLE)));
OUTPUT(b1_matrix,,'~thor::maryam::mytest::b1_matrix_MNIST_5digits1.csv',CSV(HEADING(SINGLE)));
OUTPUT(b2_matrix,,'~thor::maryam::mytest::b2_matrix_MNIST_5digits1.csv',CSV(HEADING(SINGLE)));


