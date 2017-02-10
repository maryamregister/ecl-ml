// Utilities for the implementation of ML (rather than the interface to it)
IMPORT PBblas;
IMPORT * FROM $;
IMPORT Std.Str;
EXPORT Utils := MODULE

EXPORT Pi := 3.1415926535897932384626433;

EXPORT REAL8 Fac(UNSIGNED2 i) := BEGINC++
  double accum = 1.0;
	for ( int j = 2; j <= i; j++ )
		accum *= (double)j;
	return accum;
  ENDC++;

// The 'double' factorial is defined for ODD n and is the product of all the odd numbers up to and including that number
// We are extending the meaning to even numbers to mean the product of the even numbers up to and including that number
// Thus DoubleFac(8) = 8*6*4*2
// We also defend against i < 2 (returning 1.0)
EXPORT REAL8 DoubleFac(INTEGER2 i) := BEGINC++
  if ( i < 2 )
		return 1.0;
  double accum = (double)i;
	for ( int j = i-2; j > 1; j -= 2 )
		accum *= (double)j;
	return accum;
  ENDC++;

// N Choose K - finds the number of combinations of K elements out of a possible N
// Should eventually do this in a way to avoid the intermediates (such as Fac(N)) exploding
EXPORT REAL8 NCK(INTEGER2 N, INTEGER2 K) := Fac(N)/(Fac(K)*Fac(N-k));

// Evaluate a polynomial from a set of co-effs. Co-effs 1 is assumed to be the HIGH order of the equation
// Thus for ax^2+bx+c - the set would need to be Coef := [a,b,c];
EXPORT REAL8 Poly(REAL8 x, SET OF REAL8 Coeffs) := BEGINC++
  if (isAllCoeffs)
	  return 0.0;
	int num = lenCoeffs / 8; // Note - REAL8 specified in prototype
	if ( num == 0 )
		return 0.0;
	const double * cp = (const double *)coeffs; // Will not work if sizeof(double) != 8
	double tot = *cp++;
	while ( --num )
		tot = tot * x + *cp++;
	return tot;
  ENDC++;

EXPORT stirlingFormula(real8 x) :=FUNCTION
	 stirCoefs :=[7.87311395793093628397E-4,
								-2.29549961613378126380E-4,
								-2.68132617805781232825E-3,
								3.47222221605458667310E-3,
								8.33333333333482257126E-2];
												
		REAL8 stirmax := 143.01608;
    REAL8 w := 1.0/x;
    REAL8  y := exp(x);

    v := 1.0 + w * Poly(w, stirCoefs);
    
		z := IF(x > stirmax, POWER(x,0.5 * x - 0.25), //Avoid overflow in Math.pow()
													POWER(x, x - 0.5)/y);
		u := IF(x > stirmax, z*(z/y), z);
		
    RETURN SQRT(PI)*u*v;
end;
/*
	return the value of gamma function of real number x
  The implementation references open source weka gamma function but does not strictly follow it
	12/02/2011
*/
EXPORT gamma(REAL8 x) :=FUNCTION
	P :=[
			1.60119522476751861407E-4,
      1.19135147006586384913E-3,
      1.04213797561761569935E-2,
      4.76367800457137231464E-2,
      2.07448227648435975150E-1,
      4.94214826801497100753E-1,
      9.99999999999999996796E-1];
	
	Q :=[
			-2.31581873324120129819E-5,
      5.39605580493303397842E-4,
      -4.45641913851797240494E-3,
      1.18139785222060435552E-2,
      3.58236398605498653373E-2,
      -2.34591795718243348568E-1,
      7.14304917030273074085E-2,
      1.00000000000000000320E0];
												
	 absx := abs(x);
	 intx := (INTEGER) absx;
	 isRightInt := (absx-intx)<1.0e-9;
	 isLeftInt :=ABS((ROUND(absx)-absx))<1.0e-9;
	 // x can't be zero or negative integer
	 isfail := absx<1.0e-9 OR (x<0 AND (isRightInt OR isLeftInt));
	 
	  // x is positive natural numbers 
		REAL8 g0 := IF( intx=1 OR intx=2, 1.0, fac(intx-1)); 
		
	  //x < -6
		REAL8 y := absx * SIN(PI*absx);
		REAL8 g1 := - PI/(y*stirlingFormula(absx));		
		REAL8 g2 := IF(x>6.0, stirlingFormula(x), g1);
		
		//abs(x) <6
		z0 := 1.0;
		z1 :=IF(x>3, MAP(//x>3
								    x >5 =>(x-1)*(x-2)*(x-3),
										x >4 =>(x-1)*(x-2),
										x >3 =>(x-1),
										1 
						       ), z0);
	  REAL8 x1 := IF(x>3, x-(INTEGER)x+2, x);
		
		//for x1<0
		z2 :=IF(x1<-1 AND x1 >-6,
						MAP(
						x1 <-5=>z1/(x1*(x1+1.0)*(x1+2.0)*(x1+3.0)*(x1+4.0)),
						x1 <-4 =>z1/(x1*(x1+1.0)*(x1+2.0)*(x1+3.0)),
						x1 <-3 =>z1/(x1*(x1+1.0)*(x1+2.0)),
						x1 <-2 => z1/(x1*(x1+1.0)),
						z1/x1
						), z1);
		x2 := IF(x1<-1 AND x1 >-6.0, x1+(INTEGER)ABS(x1), x1);
		REAL8 w0 := IF(x2<0 AND x2>-1.0E-9, z2/((1.0+0.5772156649015329 * x2)*x2),z2);
		z3 := IF(x2<-1.0E-9, z2/x2, z2);
		x3 := IF(x2<-1.0E-9, x2+1.0, x2);
		
		//x3>0 and x3<2
		REAL8 w1 := IF(x3<1.0E-9 AND x3>0, z3/((1.0+0.5772156649015329 * x3)*x3),z3);
	  z4 := IF(x3<2.0, IF(x3>1.0, z3/x3, z3/(x3*(x3+1.0))), z3);
	  x4 := IF(x3<2.0, IF(x3>1.0, x3+1, x3+2), x3);
	 
		x5 := x4-2.0;
		REAL8 u := Poly(x5,P);
		REAL8 v := Poly(x5,Q);
		REAL8 g3 := z4 * u / v;
		
	  REAL8 g := MAP(
						isFail => 9999,//FAIL(99, 'x should not be zero or negative integers'),
						x>1.0e-9 AND ((absx-intx)<1.0e-9 OR abs((round(absx)-absx))<1.0e-9) => g0,
						//x is big enough
						ABS(x)>=6.0 => g2,
						x2<0 AND x2>-1.0E-9 => w0,
						x3<1.0E-9 AND x3>0 => w1,
						g3);
		RETURN g;
END;

/*
	return the lower incomplete gamma value of two real numbers, x and y
*/
EXPORT REAL8 lowerGamma(REAL8 x, REAL8 y)	:= BEGINC++
	#include <math.h>
	double n,r,s,ga,t,gin;
	int k;

	if ((x < 0.0) || (y < 0)) return 0;
	n = -y+x*log(y);

	if (y == 0.0) {
		gin = 0.0;
		return gin;
	}

	if (y <= 1.0+x) {
		s = 1.0/x;
		r = s;
		for (k=1;k<=100;k++) {
			r *= y/(x+k);
			s += r;
			if (fabs(r/s) < 1e-15) break;
		}

	gin = exp(n)*s;
	}
	else {
		t = 0.0;
		for (k=100;k>=1;k--) {
			t = (k-x)/(1.0+(k/(y+t)));
		}
		ga = exp(gamma(x));
		gin = ga-(exp(n)/(y+t));
	}
	return gin;
ENDC++;

/*
	return the upper incomplete gamma value of two real numbers, x and y
*/
EXPORT REAL8 upperGamma(REAL8 x, REAL8 y)	:= BEGINC++
	#include <math.h>
	double n,r,s,ga,t,gim;
	int k;

	if ((x < 0.0) || (y < 0)) return 0;
	n = -y+x*log(y);

	if (y == 0.0) {
		gim = exp(gamma(x));
		return gim;
	}

	if (y <= 1.0+x) {
		s = 1.0/x;
		r = s;
		for (k=1;k<=100;k++) {
			r *= y/(x+k);
			s += r;
			if (fabs(r/s) < 1e-15) break;
		}

	ga = exp(gamma(x));
	gim = ga-(exp(n)*s);
	}
	else {
		t = 0.0;
		for (k=100;k>=1;k--) {
			t = (k-x)/(1.0+(k/(y+t)));
		}
		gim = exp(n)/(y+t);
	}
	return gim;
ENDC++;

/*
	return the beta value of two real numbers, x and y
*/
EXPORT Beta(REAL8 x, REAL8 y) := FUNCTION
	 absx := ABS(x);
	 intx := (INTEGER) absx;
	 isXRightInt := (absx-intx)<1.0e-9;
	 isXLeftInt :=ABS((ROUND(absx)-absx))<1.0e-9;
	 isXfail := absx<1.0e-9 OR (x<0 AND (isXRightInt OR isXLeftInt));
	 
	 absy := ABS(y);
	 inty := (INTEGER) absy;
	 isYRightInt := (absy-inty)<1.0e-9;
	 isYLeftInt :=ABS((ROUND(absy)-absy))<1.0e-9;
	 isYfail := absy<1.0e-9 OR (y<0 AND (isYRightInt OR isYLeftInt));
	 
	 bp := gamma(x)*gamma(y)/gamma(x+y);
	 bn :=(x+y)*gamma(x+1)*gamma(y+1)/(x*y*gamma(x+y+1));
	
	 b := MAP(
						x>0 AND y>0 => bp,
						isXfail OR isYfail => 9999, // failed because one of them negative integers or zero
						bn //when both x and y negative real numbers				
					 );
					
	RETURN b;
END;

// In constrast to the matrix function thin
// Will take a potentially sparse file d and fill in the blanks with value v
EXPORT Fat(DATASET(Types.NumericField) d0,Types.t_FieldReal v=0) := FUNCTION
  dn := DISTRIBUTE(d0,HASH(id)); // all the values for a given ID now on one node
  seeds := TABLE(dn,{id,m := MAX(GROUP,number)},id,LOCAL); // get the list of ids on each node (and get 'max' number for free
	mn := MAX(seeds,m); // The number of fields to fill in
	Types.NumericField bv(seeds le,UNSIGNED C) := TRANSFORM
	  SELF.value := v;
		SELF.id := le.id;
		SELF.number := c;
	END;
	// turn n into a fully 'blank' matrix - distributed along with the 'real' data
	n := NORMALIZE(seeds,mn,bv(LEFT,COUNTER),LOCAL); 
	// subtract from 'n' those values that already exist
	n1 := JOIN(n,dn,LEFT.id=RIGHT.id AND LEFT.number=RIGHT.number,TRANSFORM(LEFT),LEFT ONLY,LOCAL);
	RETURN n1+dn;
END;

// Same function for discrete fields	
EXPORT FatD(DATASET(Types.DiscreteField) d0,Types.t_Discrete v=0) := FUNCTION
  dn := DISTRIBUTE(d0,HASH(id)); // all the values for a given ID now on one node
  seeds := TABLE(dn,{id,m := MAX(GROUP,number)},id,LOCAL); // get the list of ids on each node (and get 'max' number for free
	mn := MAX(seeds,m); // The number of fields to fill in
	Types.DiscreteField bv(seeds le,UNSIGNED C) := TRANSFORM
	  SELF.value := v;
		SELF.id := le.id;
		SELF.number := c;
	END;
	// turn n into a fully 'blank' matrix - distributed along with the 'real' data
	n := NORMALIZE(seeds,mn,bv(LEFT,COUNTER),LOCAL); 
	// subtract from 'n' those values that already exist
	n1 := JOIN(n,dn,LEFT.id=RIGHT.id AND LEFT.number=RIGHT.number,TRANSFORM(LEFT),LEFT ONLY,LOCAL);
	RETURN n1+dn;
END;

// Creates a file of pivot/target pairs with a Gini impurity value
EXPORT Gini(infile,pivot,target) := FUNCTIONMACRO
	// First count up the values of each target for each pivot
		agg := TABLE(infile,{pivot,target,Cnt := COUNT(GROUP)},pivot,target,MERGE);
	// Now compute the total number for each pivot
		aggc := TABLE(agg,{pivot,TCnt := SUM(GROUP,Cnt)},pivot,MERGE);
		r := RECORD
		  agg;
			REAL4 Prop; // Proportion pertaining to this dependant value
		END;
		// Now on each row we have the proportion of the node that is that dependant value
		prop := JOIN(agg,aggc,LEFT.pivot=RIGHT.pivot,
		             TRANSFORM(r, SELF.Prop := LEFT.Cnt/RIGHT.Tcnt, SELF := LEFT),HASH);
		// Compute 1-gini coefficient for each node for each field for each value
		RETURN TABLE(prop,{pivot,TotalCnt := SUM(GROUP,Cnt),Gini := 1-SUM(GROUP,Prop*Prop)},pivot);
  ENDMACRO;


// Given a file which is sorted by INFIELD (and possibly other values), add sequence numbers within the range of each infield
// Slighly elaborate code is to avoid having to partition the data to one value of infield per node
EXPORT mac_SequenceInField(infile,infield,seq,outfile) := MACRO

#uniquename(add_rank)
TYPEOF(infile) %add_rank%(infile le,UNSIGNED c) := TRANSFORM
  SELF.seq := c;
	SELF := le;
  END;
	
#uniquename(P)
%P% := PROJECT(infile,%add_rank%(LEFT,COUNTER));

#uniquename(RS)
%RS% := RECORD
  __Seq := MIN(GROUP,%P%.seq);
  %P%.infield;
  END;

#uniquename(Splits)
%Splits% := TABLE(%P%,%RS%,infield,FEW);

#uniquename(to_1)
TYPEOF(infile) %to_1%(%P% le,%Splits% ri) := TRANSFORM
	SELF.Seq := 1+le.Seq - ri.__Seq;
	SELF := le;
  END;
	
outfile := JOIN(%P%,%Splits%,LEFT.InField=RIGHT.InField,%to_1%(LEFT,RIGHT),LOOKUP);

ENDMACRO;

// Shift the column-numbers of a file of discretefields so that the left-most column is now new_lowval
// Can move colums left or right (or not at all)
EXPORT RebaseDiscrete(DATASET(Types.DiscreteField) cl,Types.t_FieldNumber new_lowval) := FUNCTION
  CurrentBase := MIN(cl,number);
	INTEGER Delta := new_lowval-CurrentBase;
	RETURN PROJECT(cl,TRANSFORM(Types.DiscreteField,SELF.number := LEFT.number+Delta,SELF := LEFT));
  END;

EXPORT RebaseNumericField(DATASET(Types.NumericField) cl) := MODULE
  SHARED MapRec:=RECORD
		Types.t_FieldNumber old;
		Types.t_FieldNumber new;
	END;
  olds := TABLE(cl, {cl.number,COUNT(GROUP)}, number, FEW);	
	
	EXPORT Mapping(Types.t_FieldNumber new_lowval=1) := FUNCTION
	MapRec mapthem(olds le, UNSIGNED c) := TRANSFORM
		SELF.old := le.number;
		SELF.new := c-1+new_lowval;
	END;
		RETURN PROJECT(olds, mapthem(LEFT, COUNTER));
	END;
		
	EXPORT ToNew(DATASET(MapRec) MapTable) := FUNCTION
 		RETURN JOIN(cl,MapTable,LEFT.number=RIGHT.old,TRANSFORM(Types.NumericField, SELF.number := RIGHT.new, SELF:=LEFT),LOOKUP);
  END;	
	
	EXPORT ToOld(DATASET(Types.NumericField) cl, DATASET(MapRec) MapTable) := FUNCTION
 		RETURN JOIN(cl,MapTable,LEFT.number=RIGHT.new,TRANSFORM(Types.NumericField, SELF.number := RIGHT.old, SELF:=LEFT),LOOKUP);
  END;	
  
    EXPORT ToOldFromElemToPart(DATASET(Types.NumericField) cl, DATASET(MapRec) MapTable) := FUNCTION
 		RETURN JOIN(cl,MapTable,LEFT.id=RIGHT.new,TRANSFORM(Types.NumericField, SELF.id := RIGHT.old, SELF.number:=LEFT.number,SELF:=LEFT),LOOKUP);
  END;
	
  END;	

EXPORT RebaseNumericFieldID(DATASET(Types.NumericField) cl) := MODULE
  SHARED MapRec:=RECORD
		Types.t_RecordID old;
		Types.t_RecordID new;
	END;
  olds := TABLE(cl, {cl.id,COUNT(GROUP)}, id, FEW);
	
	EXPORT MappingID(Types.t_FieldNumber new_lowval=1) := FUNCTION
	MapRec mapthem(olds le, UNSIGNED c) := TRANSFORM
		SELF.old := le.id;
		SELF.new := c-1+new_lowval;
	END;
		RETURN PROJECT(olds, mapthem(LEFT, COUNTER));
	END;
	
	
		
	EXPORT ToNew(DATASET(MapRec) MapTable) := FUNCTION
 		RETURN JOIN(cl,MapTable,LEFT.id=RIGHT.old,TRANSFORM(Types.NumericField, SELF.id := RIGHT.new, SELF:=LEFT),LOOKUP);
  END;	
	
	EXPORT ToOld(DATASET(Types.NumericField) cl, DATASET(MapRec) MapTable) := FUNCTION
 		RETURN JOIN(cl,MapTable,LEFT.id=RIGHT.new,TRANSFORM(Types.NumericField, SELF.id := RIGHT.old, SELF:=LEFT),LOOKUP);
  END;	
	
  END;

// Service functions and support pattern
EXPORT	NotFirst(STRING S) := IF(Str.FindCount(S,' ')=0,'',S[Str.Find(S,' ',1)+1..]);
EXPORT	NotLast(STRING S) := IF(Str.FindCount(S,' ')=0,'',S[1..Str.Find(S,' ',Str.FindCount(S,' '))-1]);
EXPORT	NotNN(STRING S,UNSIGNED2 NN) := MAP( NN = 1 => NotFirst(S),
																						 NN = Str.WordCount(S) => NotLast(S),
                                             S[1..Str.Find(S,' ',NN-1)]+S[Str.Find(S,' ',NN)+1..] );
EXPORT  LastN(STRING S) := Str.GetNthWord(S,Str.WordCount(S));			

// Choose K (ascending element) permutations out of string of '1 2 3 ... N'  elements
// E.g. KoutofN(2,3) = '1 2', '2 3'
EXPORT  NchooseK(UNSIGNED1 N, UNSIGNED1 K) := FUNCTION
// generate string sample txt '1 2 3 ... N' to choose K elements from
rec := {UNSIGNED1 num};
seed := DATASET([{0}], rec);
txt := Str.CombineWords(SET(NORMALIZE(seed, N, TRANSFORM(rec, SELF.num := COUNTER)), (STRING2)num), ' ' );

R := RECORD
	STRING Kperm ;
	STRING From ;
END;
Init := DATASET([{'',txt}],R);
R Permutate(DATASET(R) infile) := FUNCTION
R TakeOne(R le, UNSIGNED1 c) := TRANSFORM
  SELF.Kperm := IF( (INTEGER1)Str.GetNthWord(le.from,c)> (INTEGER1)LastN(le.Kperm),le.Kperm + ' '+Str.GetNthWord(le.From, c),SKIP);
	SELF.From := NotNN(le.From,c);
END;
RETURN NORMALIZE(infile,Str.WordCount(LEFT.From),TakeOne(LEFT,COUNTER));
END;

RETURN TABLE(LOOP(Init,K,Permutate(ROWS(LEFT))), {Kperm});

END;
//generate a radnom matrix of size nrow*ncol where matrix is partitioned on the nodes column-wise. each partition is of size nrow * ncol_part;
EXPORT distcol_ranmap(UNSIGNED nrow, UNSIGNED ncol, UNSIGNED ncol_part ) := FUNCTION
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
	Mat.Types.Element gen_tran (nid_rec le, UNSIGNED c) := TRANSFORM
		col_offset := ncol_part*le.n_id;
		SELF.x := ((c-1) % nrow) + 1 ;
    SELF.y := ((c-1) DIV nrow) + 1 + col_offset;
		SELF.value := Produce_Random ();
	END;
	
	result := NORMALIZE (init_dist, blk, gen_tran (LEFT, COUNTER));
	fltr := result.x BETWEEN 1 AND nrow AND result.y BETWEEN 1 AND ncol;
	RETURN result (fltr);
END;//END distcol_ranmap

EXPORT distrow_ranmap(UNSIGNED4 nrow, UNSIGNED4 ncol, UNSIGNED4 nrow_part ) := FUNCTION
blk := nrow_part * ncol;
	Produce_Random () := FUNCTION
		G := 1000000;
		R := (RANDOM()%G) / (REAL4)G;
		RETURN R;
	END;
	nodes_used := IF(nrow_part>0, ((nrow-1) DIV nrow_part) + 1, 1);//how many nodes are needed
	W_Rec := RECORD
			STRING1 x:= '';
  END;
	empty_init := DATASET([{' '}], W_Rec);
	nid_rec := {UNSIGNED4 n_id};
	nid_rec init_tran (UNSIGNED c) := TRANSFORM
		SELF.n_id := c-1;
	END;
	init_ := NORMALIZE (empty_init, nodes_used, init_tran (COUNTER) );
	init_dist := DISTRIBUTE (init_, n_id);
	Mat.Types.Element4 gen_tran (nid_rec le, UNSIGNED c) := TRANSFORM
		row_offset := nrow_part*le.n_id;
		SELF.x := ((c-1) % nrow_part) + 1 + row_offset;
    SELF.y := ((c-1) DIV nrow_part) + 1;
		SELF.value := Produce_Random ();
	END;
	
	result := NORMALIZE (init_dist, blk, gen_tran (LEFT, COUNTER));
	fltr := result.x BETWEEN 1 AND nrow AND result.y BETWEEN 1 AND ncol;
	RETURN result (fltr);
END;//END distrow_ranmap

EXPORT distrow_ranmap_part(UNSIGNED nrow, UNSIGNED ncol, UNSIGNED nrow_part, REAL8 coef ) := FUNCTION

	SET OF REAL8 rand_vec(PBblas.Types.dimension_t N, PBblas.Types.dimension_t s, REAL8 cc) := BEGINC++

    #body
    __lenResult = n * sizeof(double);
    __isAllResult = false;
    double * result = new double[n];
    __result = (void*) result;
    uint32_t i;
		//double G = 1000000.0;
		srand (s);
    for (i=0; i<n; i++) {
      // result[i] = (rand() % G) / G;
			result[i] =  cc * ((double) rand() / (RAND_MAX));
    }
  ENDC++;
	Produce_Random () := FUNCTION
		G := 1000000;
		R := (RANDOM()%G) / (REAL8)G;
		RETURN R;
	END;
	parts_used := IF(nrow_part>0, ((nrow-1) DIV nrow_part) + 1, 1);//how many nodes are needed
	W_Rec := RECORD
			STRING1 x:= '';
  END;
	empty_init := DATASET([{' '}], W_Rec);
	nid_rec := {UNSIGNED p_id, UNSIGNED rand_seed};
	nid_rec init_tran (UNSIGNED c) := TRANSFORM
		SELF.p_id := c;
		SELF.rand_seed := RANDOM();
	END;
	init_ := NORMALIZE (empty_init, parts_used, init_tran (COUNTER) );
	mat_map := PBblas.Matrix_Map(nrow,ncol,nrow_part,ncol);
	init_dist := DISTRIBUTE (init_, mat_map.assigned_node (p_id));
	Pbblas.Types.Layout_part part_tran (nid_rec le) := TRANSFORM
			part_id := le.p_id;
			first_row     := mat_map.first_row(part_id);
      first_col     := mat_map.first_col(part_id);
      part_rows     := mat_map.part_rows(part_id);
      part_cols     := mat_map.part_cols(part_id);
      SELF.mat_part := rand_vec (part_rows * part_cols, le.rand_seed, coef);
      SELF.partition_id:= part_id;
      SELF.node_id     := mat_map.assigned_node (part_id);
      SELF.block_row   := part_id;
      SELF.block_col   := 1;
      SELF.first_row   := first_row;
      SELF.part_rows   := part_rows;
      SELF.first_col   := first_col;
      SELF.part_cols   := part_cols;
      SELF := [];
	END;
	result := PROJECT (init_dist, part_tran(LEFT), LOCAL);
	RETURN result;
END;//END distrow_ranmap_part

//generate a random matrix in PBblas format of size ncol by nrow where the matrix is partitioned column-wise, each column partition is of size ncol_part
EXPORT distcol_ranmap_part(UNSIGNED nrow, UNSIGNED ncol, UNSIGNED ncol_part, REAL8 coef ) := FUNCTION

	SET OF REAL8 rand_vec(PBblas.Types.dimension_t N, PBblas.Types.dimension_t s, REAL8 cc) := BEGINC++

    #body
    __lenResult = n * sizeof(double);
    __isAllResult = false;
    double * result = new double[n];
    __result = (void*) result;
    uint32_t i;
		double G = 1000000.0;
		srand (s);
    for (i=0; i<n; i++) {
      // result[i] = (rand() % G) / G;
			result[i] =  cc * ((double) rand() / (RAND_MAX));
    }
  ENDC++;
	Produce_Random () := FUNCTION
		G := 1000000;
		R := (RANDOM()%G) / (REAL8)G;
		RETURN R;
	END;
	parts_used := IF(ncol_part>0, ((ncol-1) DIV ncol_part) + 1, 1);//how many nodes are needed
	W_Rec := RECORD
			STRING1 x:= '';
  END;
	empty_init := DATASET([{' '}], W_Rec);
	nid_rec := {UNSIGNED p_id, UNSIGNED rand_seed};
	nid_rec init_tran (UNSIGNED c) := TRANSFORM
		SELF.p_id := c;
		SELF.rand_seed := RANDOM();
	END;
	init_ := NORMALIZE (empty_init, parts_used, init_tran (COUNTER) );
	mat_map := PBblas.Matrix_Map(nrow,ncol,nrow,ncol_part);
	init_dist := DISTRIBUTE (init_, mat_map.assigned_node (p_id));
	Pbblas.Types.Layout_part part_tran (nid_rec le) := TRANSFORM
			part_id := le.p_id;
			first_row     := mat_map.first_row(part_id);
      first_col     := mat_map.first_col(part_id);
      part_rows     := mat_map.part_rows(part_id);
      part_cols     := mat_map.part_cols(part_id);
      SELF.mat_part := rand_vec (part_rows * part_cols, le.rand_seed, coef);
      SELF.partition_id:= part_id;
      SELF.node_id     := mat_map.assigned_node (part_id);
      SELF.block_row   := part_id;
      SELF.block_col   := 1;
      SELF.first_row   := first_row;
      SELF.part_rows   := part_rows;
      SELF.first_col   := first_col;
      SELF.part_cols   := part_cols;
      SELF := [];
	END;
	result := PROJECT (init_dist, part_tran(LEFT), LOCAL);
	RETURN result;
END;//END distcol_ranmap_part

EXPORT distrow_ranmap_part4(UNSIGNED4 nrow, UNSIGNED4 ncol, UNSIGNED4 nrow_part, REAL4 coef ) := FUNCTION

	SET OF Pbblas.Types.value_t4 rand_vec(PBblas.Types.dimension_t N, PBblas.Types.dimension_t s, REAL4 cc) := BEGINC++

    #body
    __lenResult = n * sizeof(float);
    __isAllResult = false;
    float * result = new float[n];
    __result = (void*) result;
    uint32_t i;
		double G = 1000000.0;
		srand (s);
    for (i=0; i<n; i++) {
      // result[i] = (rand() % G) / G;
			result[i] =  cc * ((float) rand() / (float)(RAND_MAX));
			//float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
  ENDC++;
	Produce_Random () := FUNCTION
		G := 1000000;
		R := (RANDOM()%G) / (REAL4)G;
		RETURN R;
	END;
	parts_used := IF(nrow_part>0, ((nrow-1) DIV nrow_part) + 1, 1);//how many nodes are needed
	W_Rec := RECORD
			STRING1 x:= '';
  END;
	empty_init := DATASET([{' '}], W_Rec);
	nid_rec := {UNSIGNED4 p_id, UNSIGNED4 rand_seed};
	nid_rec init_tran (UNSIGNED c) := TRANSFORM
		SELF.p_id := c;
		SELF.rand_seed := RANDOM();
	END;
	init_ := NORMALIZE (empty_init, parts_used, init_tran (COUNTER) );
	mat_map := PBblas.Matrix_Map(nrow,ncol,nrow_part,ncol);
	init_dist := DISTRIBUTE (init_, mat_map.assigned_node (p_id));
	Pbblas.Types.Layout_part4 part_tran (nid_rec le) := TRANSFORM
			part_id := le.p_id;
			first_row     := mat_map.first_row(part_id);
      first_col     := mat_map.first_col(part_id);
      part_rows     := mat_map.part_rows(part_id);
      part_cols     := mat_map.part_cols(part_id);
      SELF.mat_part := rand_vec (part_rows * part_cols, le.rand_seed, coef);
      SELF.partition_id:= part_id;
      SELF.node_id     := mat_map.assigned_node (part_id);
      SELF.block_row   := part_id;
      SELF.block_col   := 1;
      SELF.first_row   := first_row;
      SELF.part_rows   := part_rows;
      SELF.first_col   := first_col;
      SELF.part_cols   := part_cols;
      SELF := [];
	END;
	result := PROJECT (init_dist, part_tran(LEFT), LOCAL);
	RETURN result;
END;//END distrow_ranmap_part4

// a different version of ToGroundTruth which is more optimized
EXPORT LabelToGroundTruth4(DATASET(Types.NumericField4) Y ) := FUNCTION
	sample_num  := MAX (Y,Y.id);
	class_num   := MAX (Y, Y.value);
	Types.NumericField4 grnd (Types.NumericField4 le, UNSIGNED4 c) := TRANSFORM
		SELF.number := le.id;
		SELF.id := c;
		SELF.value := IF (c=le.value,1,0);
		SELF:= le;
	END;
	grndt_result := NORMALIZE (Y,class_num,grnd(LEFT, COUNTER));
	RETURN grndt_result;
END;//END LabelToGroundTruth

EXPORT LabelToGroundTruth(DATASET(Types.NumericField) Y ) := FUNCTION
	sample_num  := MAX (Y,Y.id);
	class_num   := MAX (Y, Y.value);
	Types.NumericField grnd (Types.NumericField le, UNSIGNED4 c) := TRANSFORM
		SELF.number := le.id;
		SELF.id := c;
		SELF.value := IF (c=le.value,1,0);
		SELF:= le;
	END;
	grndt_result := NORMALIZE (Y,class_num,grnd(LEFT, COUNTER));
	RETURN grndt_result;
END;//END LabelToGroundTruth


EXPORT ToGroundTruth(DATASET(Types.NumericField) Y ) := FUNCTION


zero_mat    := DATASET ([{1,1,0}],Mat.Types.Element);
sample_num  := MAX (Y,Y.id);
class_num   := MAX (Y, Y.value);
scratch_mat := Mat.Repmat (zero_mat, class_num, sample_num);



Mat.Types.Element ToGT(scratch_mat l, Y r) := TRANSFORM
  SELF.value := IF(l.x=r.value,1,0);
  SELF := l;
END;


Result := JOIN (scratch_mat, Y, LEFT.y=RIGHT.id , ToGT(LEFT,RIGHT));

RETURN Result;

END; // END ToGroundTruth

// This version converts label to grountruth, as long as labels are distinct integer numbers it works correctly

EXPORT DistinctLabeltoGroundTruth4(DATASET(Types.NumericField4) Y ) := FUNCTION
	// find DISTINCT  labels
	Y_sorted := SORT (Y, Y.value);
	distinct_labels := DEDUP (Y_sorted, Y_sorted.value);
	// assigne 1 to N (N is the number of classes) to the distict labels
	distinct_labels_1 := PROJECT (distinct_labels, TRANSFORM (Types.NumericField4, SELF.id := COUNTER; SELF:=LEFT));
	//JOIN distinct_labels_1 and Y in order to get a Y dataset where labels start from 1 
	Y_1 := JOIN (Y, distinct_labels_1, LEFT.value = RIGHT.value, TRANSFORM (Types.NumericField4, SELF.value := RIGHT.id; SELF:=LEFT));
	// now that Y_1's labels start from 1, it can be passe to LabelToGroundTruth in order to be converted to groundtruth matrix
	RETURN LabelToGroundTruth4 (Y_1);
END;// END DistinctLabeltoGroundTruth4


//Extract distinct labels from 1 to number of classes 
// input with labels 3 , 4 ,4 10 will get lables 1,2,2,3
EXPORT DistinctLabel4(DATASET(Types.NumericField4) Y ) := FUNCTION
	// find DISTINCT  labels
	Y_sorted := SORT (Y, Y.value);
	distinct_labels := DEDUP (Y_sorted, Y_sorted.value);
	// assigne 1 to N (N is the number of classes) to the distict labels
	distinct_labels_1 := PROJECT (distinct_labels, TRANSFORM (Types.NumericField4, SELF.id := COUNTER; SELF:=LEFT));
	//JOIN distinct_labels_1 and Y in order to get a Y dataset where labels start from 1 
	Y_1 := JOIN (Y, distinct_labels_1, LEFT.value = RIGHT.value, TRANSFORM (Types.NumericField4, SELF.value :=(REAL4)RIGHT.id; SELF:=LEFT));
	// now that Y_1's labels start from 1, it can be passe to LabelToGroundTruth in order to be converted to groundtruth matrix
	RETURN Y_1;
END;// END DistinctLabel4


EXPORT DistinctLabel(DATASET(Types.NumericField) Y ) := FUNCTION
	// find DISTINCT  labels
	Y_sorted := SORT (Y, Y.value);
	distinct_labels := DEDUP (Y_sorted, Y_sorted.value);
	// assigne 1 to N (N is the number of classes) to the distict labels
	distinct_labels_1 := PROJECT (distinct_labels, TRANSFORM (Types.NumericField, SELF.id := COUNTER; SELF:=LEFT));
	//JOIN distinct_labels_1 and Y in order to get a Y dataset where labels start from 1 
	Y_1 := JOIN (Y, distinct_labels_1, LEFT.value = RIGHT.value, TRANSFORM (Types.NumericField, SELF.value := (REAL8)RIGHT.id; SELF:=LEFT));
	// now that Y_1's labels start from 1, it can be passe to LabelToGroundTruth in order to be converted to groundtruth matrix
	RETURN Y_1;
END;// END DistinctLabel

EXPORT DistinctLabel_test(DATASET(Types.NumericField) Y ) := FUNCTION
	// find DISTINCT  labels
	Y_sorted := SORT (Y, Y.value);
	distinct_labels := DEDUP (Y_sorted, Y_sorted.value);
	// assigne 1 to N (N is the number of classes) to the distict labels
	distinct_labels_1 := PROJECT (distinct_labels, TRANSFORM (Types.NumericField, SELF.id := COUNTER; SELF:=LEFT));
	//JOIN distinct_labels_1 and Y in order to get a Y dataset where labels start from 1 
	Y_1 := JOIN (Y, distinct_labels_1, LEFT.value = RIGHT.value, TRANSFORM (Types.NumericField, SELF.value := (REAL8)RIGHT.id; SELF:=LEFT));
	// now that Y_1's labels start from 1, it can be passe to LabelToGroundTruth in order to be converted to groundtruth matrix
	RETURN Y_1;
END;// END DistinctLabel_test


EXPORT DistinctLabeltoGroundTruth(DATASET(Types.NumericField) Y ) := FUNCTION
	// find DISTINCT  labels
	Y_sorted := SORT (Y, Y.value);
	distinct_labels := DEDUP (Y_sorted, Y_sorted.value);
	// assigne 1 to N (N is the number of classes) to the distict labels
	distinct_labels_1 := PROJECT (distinct_labels, TRANSFORM (Types.NumericField, SELF.id := COUNTER; SELF:=LEFT));
	//JOIN distinct_labels_1 and Y in order to get a Y dataset where labels start from 1 
	Y_1 := JOIN (Y, distinct_labels_1, LEFT.value = RIGHT.value, TRANSFORM (Types.NumericField, SELF.value := RIGHT.id; SELF:=LEFT));
	// now that Y_1's labels start from 1, it can be passe to LabelToGroundTruth in order to be converted to groundtruth matrix
	RETURN LabelToGroundTruth (Y_1);
END;// END DistinctLabeltoGroundTruth


// This function converts the number values in a way they start from 1 to the number of distinct number values
// [{1,10,9},{2,3,89}] -> [{1,2,9},{2,1,89}] : features number 10 and 3 are converted to features number 2 and 1 respectively
EXPORT DistinctFeaturest4(DATASET(Types.NumericField4) Y ) := FUNCTION
	// find DISTINCT  features
	Y_sorted := SORT (Y, Y.number);
	distinct_features := DEDUP (Y_sorted, Y_sorted.number);
	// assigne 1 to N (N is the number of features) to the distict features
	distinct_features_1 := PROJECT (distinct_features, TRANSFORM (Types.NumericField4, SELF.id := COUNTER; SELF:=LEFT));
	//JOIN distinct_features_1 and Y in order to get a Y dataset where numbers (features) start from 1 
	Y_1 := JOIN (Y, distinct_features_1, LEFT.number = RIGHT.number, TRANSFORM (Types.NumericField4, SELF.number := RIGHT.id; SELF:=LEFT));
	// now that Y_1's labels start from 1, it can be passe to LabelToGroundTruth in order to be converted to groundtruth matrix
	RETURN Y_1;
END;// END DistinctFeaturest4
// This function normalize

// This function converts the number values in a way they start from 1 to the number of distinct number values
// [{1,10,9},{2,3,89}] -> [{1,2,9},{2,1,89}] : features number 10 and 3 are converted to features number 2 and 1 respectively
EXPORT DistinctFeaturest(DATASET(Types.NumericField) Y ) := FUNCTION
	// find DISTINCT  features
	Y_sorted := SORT (Y, Y.number);
	distinct_features := DEDUP (Y_sorted, Y_sorted.number);
	// assigne 1 to N (N is the number of features) to the distict features
	distinct_features_1 := PROJECT (distinct_features, TRANSFORM (Types.NumericField, SELF.id := COUNTER; SELF:=LEFT));
	//JOIN distinct_features_1 and Y in order to get a Y dataset where numbers (features) start from 1 
	Y_1 := JOIN (Y, distinct_features_1, LEFT.number = RIGHT.number, TRANSFORM (Types.NumericField, SELF.number := RIGHT.id; SELF:=LEFT));
	// now that Y_1's labels start from 1, it can be passe to LabelToGroundTruth in order to be converted to groundtruth matrix
	RETURN Y_1;
END;// END DistinctFeaturest

//Take a dataset of cells for a partition and pack into a dense matrix.  Specify Row or Column major
//First row and first column are one based.
//Insert is used insert columns with a spacific value.  Typical use is building a matrix for a solver
//where the first column is an inserted column of 1 values for the intercept.
IMPORT PBblas.Types;
dimension_t := Types.dimension_t;
value_t := Types.value_t;
Layout_Cell := Types.Layout_Cell;
END;