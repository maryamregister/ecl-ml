IMPORT PBblas;
IMPORT PBblas.IMatrix_Map;
IMPORT PBblas.Types;
IMPORT ML.DMAT;
Part := Types.Layout_Part;
Side := Types.Side;
Triangle := Types.Triangle;
Diagonal := Types.Diagonal;

IMPORT ML;
IMPORT * FROM $;
IMPORT $.Mat;
IMPORT * FROM ML.Types;
IMPORT PBblas;
Layout_Cell := PBblas.Types.Layout_Cell;
Layout_Part := PBblas.Types.Layout_Part;
emptyC := DATASET([], Types.NumericField);

EXPORT  polyinterp_both (REAL8 t_1, REAL8 f_1, REAL8 gtd_1, REAL8 t_2, REAL8 f_2, REAL8 gtd_2, REAL8 xminBound, REAL8 xmaxBound) := FUNCTION
    poly1 := FUNCTION
    //orig
    /*
      setp1 := FUNCTION
        points := DATASET([{1,1,t_1},{2,1,t_2},{3,1,f_1},{4,1,f_2},{5,1,gtd_1},{6,2,gtd_2}], Types.NumericField);
        RETURN points;
      END;
      setp2 := FUNCTION
        points := DATASET([{2,1,t_1},{1,1,t_2},{4,1,f_1},{3,1,f_2},{6,1,gtd_1},{5,2,gtd_2}], Types.NumericField);
        RETURN points;
      END;
      orderedp := IF (t_1<t_2,setp1 , setp2);
      tmin := orderedp (id=1)[1].value;
      tmax := orderedp (id=2)[1].value;
      fmin := orderedp (id=3)[1].value;
      fmax := orderedp (id=4)[1].value;
      gtdmin := orderedp (id=5)[1].value;
      gtdmax := orderedp (id=6)[1].value;
      
      */
      

      points_t1 :=[t_1,t_2,f_1,f_2,gtd_1,gtd_2];

      points_t2 := [t_2,t_1,f_2,f_1,gtd_2,gtd_1];
      orderedp := IF (t_1<t_2,points_t1 , points_t2);
      
      tmin := orderedp [1];
      tmax := orderedp [2];
      fmin := orderedp [3];
      fmax := orderedp [4];
      gtdmin := orderedp [5];
      gtdmax := orderedp [6];
      
      
      // A= [t_1^3 t_1^2 t_1 1
      //    t_2^3 t_2^2 t_2 1
      //    3*t_1^2 2*t_1 t_1 0
      //    3*t_2^2 2*t_2 t_2 0]
      //b = [f_1 f_2 dtg_1 gtd_2]'
      // A := DATASET([
      // {1,1,POWER(t_1,3)},
      // {1,2,POWER(t_1,2)},
      // {1,3,POWER(t_1,3)},
      // {1,4,1},
      // {2,1,POWER(t_2,3)},
      // {2,2,POWER(t_2,2)},
      // {2,3,POWER(t_2,1)},
      // {2,4,1},
      // {3,1,3*POWER(t_1,2)},
      // {3,2,2*t_1},
      // {3,3,1},
      // {3,4,0},
      // {4,1,3*POWER(t_2,2)},
      // {4,2,2*t_2},
      // {4,3,1},
      // {4,4,0}],
      // Types.NumericField);
      Aset := [POWER(t_1,3),POWER(t_2,3),3*POWER(t_1,2),3*POWER(t_2,2),
      POWER(t_1,2),POWER(t_2,2), 2*t_1,2*t_2,
      POWER(t_1,3),POWER(t_2,1), 1, 1,
      1, 1, 0, 0]; // A 4*4 Matrix      
      // b := DATASET([
      // {1,1,f_1},
      // {2,1,f_2},
      // {3,1,gtd_1},
      // {4,1,gtd_2}],
      // Types.NumericField);
      Bset := [f_1, f_2, gtd_1, gtd_2]; // A 4*1 Matrix
      // Find interpolating polynomial
      
      //params = A\b;
      //orig
      /*
      A_map := PBblas.Matrix_Map(4, 4, 4, 4);
      b_map := PBblas.Matrix_Map(4, 1, 4, 1);
      A_part := ML.DMat.Converted.FromNumericFieldDS (A, A_map);
      b_part := ML.DMat.Converted.FromNumericFieldDS (b, b_map);
      params_part := DMAT.solvelinear (A_map,  A_part, FALSE, b_map, b_part);
      params := DMat.Converted.FromPart2DS (params_part);
      params1 := params(id=1)[1].value;
      params2 := params(id=2)[1].value;
      params3 := params(id=3)[1].value;
      params4 := params(id=4)[1].value;
      dParams1 := 3*params(id=1)[1].value;
      dparams2 := 2*params(id=2)[1].value;
      dparams3 := params(id=3)[1].value;*/
      params_partset := PBblas.BLAS.solvelinear (Aset, Bset, 4,1,4,4);
      //params_partset := [1,2,3,4];
      params1 := params_partset[1];
      params2 := params_partset[2];
      params3 := params_partset[3];
      params4 := params_partset[4];
      
      dParams1 := 3*params_partset[1];
      dparams2 := 2*params_partset[2];
      dparams3 := params_partset[3];

      Rvalues := roots (dParams1, dparams2, dparams3);
      // Compute Critical Points
      INANYINF := FALSE; //????for now
      cp1 := xminBound;
      cp2 := xmaxBound;
      cp3 := t_1;
      cp4 := t_2;
      cp5 := Rvalues (id=2)[1].value;
      cp6 := Rvalues (id=3)[1].value;
      ISrootsreal := (BOOLEAN) Rvalues (id=1)[1].value;
      cp_real := DATASET([
      {1,1,cp1},
      {2,1,cp2},
      {3,1,cp3},
      {4,1,cp4},
      {5,1,cp5},
      {6,1,cp6}],
      Types.NumericField);
      cp_imag := DATASET([
      {1,1,cp1},
      {2,1,cp2},
      {3,1,cp3},
      {4,1,cp4}],
      Types.NumericField);
      cp := IF (ISrootsreal, cp_real, cp_imag);
      itr := IF (ISrootsreal, 6, 4);
      // Test Critical Points
      topa :=  DATASET([{1,1,(xminBound+xmaxBound)/2},{2,1,1000000}], Types.NumericField);//send minpos and fmin value to Resultsstep
      Resultstep (DATASET(Types.NumericField) x, UNSIGNED coun) := FUNCTION
        inr := x(id=1)[1].value;
        f_min := x(id=2)[1].value;
        // if imag(xCP)==0 && xCP >= xminBound && xCP <= xmaxBound
        xCP := cp(id=coun)[1].value;
        cond := xCP >= xminBound AND xCP <= xmaxBound; //???
        // fCP = polyval(params,xCP);
        fCP := params1*POWER(xCP,3)+params2*POWER(xCP,2)+params3*xCP+params4;
        //if imag(fCP)==0 && fCP < fmin
        cond2 := (coun=1 OR fCP<f_min) AND ISrootsreal; // If the roots are imaginary so is FCP
        rr := IF (cond,IF (cond2, xCP, inr),inr);
        ff := IF (cond,IF (cond2, fCP, f_min),f_min);
        RETURN DATASET([{1,1,rr},{2,1,ff}], Types.NumericField);
      END;
      finalresult := LOOP(topa, COUNTER <= itr, Resultstep(ROWS(LEFT),COUNTER));
      //RETURN finalresult;
      //RETURN IF(t_1=0, 10, 100);
      // RETURN DATASET([
      // {1,1,dParams1},
      // {2,1,dParams2},
      // {3,1,dParams3}],
      // Types.NumericField);
     RETURN finalresult(id=1)[1].value;
    END;//END poly1
     poly2 := FUNCTION
        setp1 := FUNCTION
          points := DATASET([{1,1,t_1},{2,1,t_2},{3,1,f_1},{4,1,f_2},{5,1,gtd_1},{6,2,gtd_2}], Types.NumericField);
          RETURN points;
        END;
        setp2 := FUNCTION
          points := DATASET([{2,1,t_1},{1,1,t_2},{4,1,f_1},{3,1,f_2},{6,1,gtd_1},{5,2,gtd_2}], Types.NumericField);
          RETURN points;
        END;
        orderedp := IF (t_1<t_2,setp1 , setp2);
        tmin := orderedp (id=1)[1].value;
        tmax := orderedp (id=2)[1].value;
        fmin := orderedp (id=3)[1].value;
        fmax := orderedp (id=4)[1].value;
        gtdmin := orderedp (id=5)[1].value;
        gtdmax := orderedp (id=6)[1].value;
        // d1 = points(minPos,3) + points(notMinPos,3) - 3*(points(minPos,2)-points(notMinPos,2))/(points(minPos,1)-points(notMinPos,1));
        d1 := gtdmin + gtdmax - (3*((fmin-fmax)/(tmin-tmax)));
        //d2 = sqrt(d1^2 - points(minPos,3)*points(notMinPos,3));
        d2 := SQRT ((d1*d1)-(gtdmin*gtdmax));
        d2real := TRUE; //check it ???
        //t = points(notMinPos,1) - (points(notMinPos,1) - points(minPos,1))*((points(notMinPos,3) + d2 - d1)/(points(notMinPos,3) - points(minPos,3) + 2*d2));
        temp := tmax - ((tmax-tmin)*((gtdmax+d2-d1)/(gtdmax-gtdmin+(2*d2))));
        //min(max(t,points(minPos,1)),points(notMinPos,1));
        minpos1 := MIN([MAX([temp,tmin]),tmax]);
        minpos2 := (t_1+t_2)/2;
        pol1Result := IF (d2real,minpos1,minpos2);
        RETURN pol1Result;
        //RETURN IF(t_1=0, 10, 100);
      END;//END poly2
    polResult := poly1;
    RETURN polResult;
  END;//end polyinterp_both
   
  newt := polyinterp_both (0 ,  15.05 ,  49.9975,  1.0  , 12.2381 , -41.6795, 1.01, 10);
  sideSw := Side.Ax;
  Aset := [ 1 ,   2 ,    3,    90    , 3   , 80,    10   , 11 ,    8];
  Asett := [  1,90,10,
     2 ,3 , 11,
     3, 80 ,8];
  Bset := [ 231, 5996, 620];
  Tset := PBblas.BLAS.dtrsm (sideSw, Triangle.Upper, FALSE, Diagonal.NotUnitTri, 3,1,  3, 1.0, Asett,Bset);
  ah := PBblas.BLAS.solvelinear (Asett, Bset, 3, 1,  3, 3);
 // output(ah, named('Tset'));
 //polyinterp_noboundry (REAL8 t_1, REAL8 f_1, REAL8 gtd_1, REAL8 t_2, REAL8 f_2, REAL8 gtd_2)
 
  
    

    
hihihi := Optimization (0, 0, 0, 0).polyinterp_noboundry ( 1.0000 ,  12.2381 , -41.6795 ,1.3432 ,  18.8396   ,32.0273);
man := roots (1, 2, 4);
 //output(man);

theta := DATASET([
{1,1,1},
{2,2,2},
{3,3,3},
{4,4,4},
{5,5,5},
{6,6,6},
{7,7,7},
{8,8,8},
{9,9,9},
{10,10,10},
{11,11,11},
{12,12,12},
{13,13,13},
{14,14,14},
{15,15,15},
{16,16,16},
{17,17,17}
], Types.NumericField);
    nf := 3;
    nh := 2;
    nfh := nf*nh;
    nfh_2 := 2*nfh;
    Mat.Types.MUElement Wreshape (Types.NumericField l) := TRANSFORM
      no_temp := (l.id DIV (nfh+1))+1;
      SELF.no := no_temp;
      //SELF.x := IF (no_temp =1 ,((l.id-1)%nh)+1, ((l.id-1-nfh)%nh)+1);
      SELF.x := IF (no_temp=1, 1+((l.id-1)%nh) , 1+((l.id-1-nfh)%nf));
      SELF.y := IF (no_temp=1, ((l.id-1) DIV nh)+1, ((l.id-1-nfh) DIV nf)+1);
      SELF.value := l.value;
    END;
    SA_W := PROJECT (theta(id<=2*nfh),Wreshape(LEFT));
    Mat.Types.MUElement Breshape (Types.NumericField l) := TRANSFORM
      no_temp := IF (l.id-nfh_2<=nh,1,2);
      SELF.no := no_temp;
      SELF.x := IF (no_temp =1 ,l.id-nfh_2, l.id-nfh_2-nh);
      SELF.y := 1;
      SELF := l;
    END;
    SA_B := PROJECT (theta(id>nfh_2),Breshape(LEFT));
//OUTPUT(theta,NAMED('theta'));
//OUTPUT(SA_W,NAMED('SA_W'));
//OUTPUT(SA_B,NAMED('SA_B'));


myrecord := RECORD
real8 x;
DATASET(ML.Mat.Types.Element) m;
DATASET(ML.Mat.Types.Element) m2;
END;
hkl := DATASET([{1,1,2},{1,2,4}],ML.Mat.Types.Element);
hkl2 := DATASET([{1,1,2},{1,2,4},{3,4,10}],ML.Mat.Types.Element);
myrecdata := DATASET ([{0.1,hkl,hkl2}],myrecord);
my := DATASET ([],myrecord);
h := myrecdata.m2;

my[1].x := 10;
my[1].m := hkl2;
output(my);
myrecord CalcAges(myrecord l) := TRANSFORM
  SELF.x := 10 - l.x;
  SELF.m := ML.Mat.Scale(l.m,10);
  SELF := l;
END;
xx := PROJECT(myrecdata,CalcAges(LEFT));
output(xx.m2);

