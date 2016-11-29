
IMPORT PBblas;

REAL8 polyinterpc (Pbblas.Types.value_t p1_1, Pbblas.Types.value_t p1_2, Pbblas.Types.value_t p1_3, Pbblas.Types.value_t p2_1, Pbblas.Types.value_t p2_2,
		Pbblas.Types.value_t p2_3, Pbblas.Types.value_t xminBound, Pbblas.Types.value_t xmaxBound, PBblas.Types.dimension_t minboundprovided,PBblas.Types.dimension_t maxboundprovided,
		PBblas.Types.dimension_t f1_2, PBblas.Types.dimension_t g1_3, PBblas.Types.dimension_t f2_2 , PBblas.Types.dimension_t g2_3):= BEGINC++
			#body
				double result = 0;
				double mint;
				double minf;
				double ming;
				double maxt;
				double maxf;
				double maxg;
				double d1;
				double d2tmp;
				double d2;
				double t;
				double minPos;
				double xmin;
				double xmax;
				uint32_t order = f1_2 + g1_3 + f2_2 + g2_3 -1;
				uint32_t nPoints = 2;
				uint32_t realfg = order +1;// number of real f and g values
				double *A = new double[realfg * realfg]; // realfg * (order +1)
				double *b = new double[realfg];
				uint32_t i;
				uint32_t A_off=0;
				uint32_t B_off=0;

				if (nPoints==2 && order==3 && minboundprovided==0 && maxboundprovided)
				{
				  //[minVal minPos] = min(points(:,1));
					//notMinPos = -minPos+3;
					if (p1_1 < p2_1)
					{
						mint = p1_1;
						minf = p1_2;
						ming = p1_3;
						maxt = p2_1;
						maxf = p2_2;
						maxg = p2_3;
					}
					else 
					{
						mint = p2_1;
						minf = p2_2;
						ming = p2_3;
						maxt = p1_1;
						maxf = p1_2;
						maxg = p1_3;
					}
					// d1 = points(minPos,3) + points(notMinPos,3) - 3*(points(minPos,2)-points(notMinPos,2))/(points(minPos,1)-points(notMinPos,1));
					d1 = ming + maxg - 3*(minf-maxf)/(mint-maxt);
					//d2 = sqrt(d1^2 - points(minPos,3)*points(notMinPos,3));
					d2tmp = (d1*d1) - (ming*maxg);
					/*
					if isreal(d2)
					t = points(notMinPos,1) - (points(notMinPos,1) - points(minPos,1))*((points(notMinPos,3) + d2 - d1)/(points(notMinPos,3) - points(minPos,3) + 2*d2));
						minPos = min(max(t,points(minPos,1)),points(notMinPos,1));
					else
						minPos = mean(points(:,1));
					end*/
					if (d2tmp >= 0 ) // thi smean d2=sqrt(d2tmp) is a real number
					{
						// result = points(notMinPos,1) - (points(notMinPos,1) - points(minPos,1))*((points(notMinPos,3) + d2 - d1)/(points(notMinPos,3) - points(minPos,3) + 2*d2));
						// minPos = min(max(t,points(minPos,1)),points(notMinPos,1));
						d2 = sqrt (d2tmp);
						t = maxt - (maxt - mint)*((maxg + d2 - d1)/(maxg - ming + 2*d2));
						//minPos = min(max(t,points(minPos,1)),points(notMinPos,1));
						if (t > mint)
						{
							if (t < maxt)
							{
								minPos = t;
							}
							else
							{
								minPos = maxt;
							}
						}
						else
						{
							if (mint < maxt)
							{
								minPos = mint;
							}
							else
							{
								minPos = maxt;
							}
						}
					}
					else
					{
						//minPos = mean(points(:,1));
						minPos = mint;
					}
					result = minPos;
					return result;
				}
				// xmin = min(points(:,1));
				// xmax = max(points(:,1));
				if (p1_1 < p2_1)
				{
					xmin = p1_1;
					xmax = p2_1;
				}
				else
				{
					xmin = p2_1;
					xmax = p1_2;
				}
				//Compute Bounds of Interpolation Area
				if (minboundprovided==0)
				{
					xminbound = xmin;
				}
				if (maxboundprovided==0)
				{
					xmaxbound = xmax;
				}
				// Constraints Based on available Function Values
				if (f1_2 != 0)
				{
					for (i=order; i--; i>=0)
					{
						//points(i,1)^j;
						A[A_off+order-i] = p1_1*i;
					}
					A_off = A_off + order;
					b[B_off] = p1_2;
					B_off = B_off + 1;
				}
				if (f2_2 != 0)
				{
					for (i=order; i--; i>=0)
					{
						//points(i,1)^j;
						A[A_off+order-i] = p2_1*i;
					}
					A_off = A_off+ order;
					b[B_off] = p2_2;
					B_off = B_off + 1;
				}
				//Constraints based on available Derivatives
				if (g1_3 != 0)
				{
					for (i=1; i++; i<order)
					{
						//(order-j+1)*points(i,1)^(order-j);
						A[A_off+order-i] = (order-i+1)*p1_1*(order-i);
					}
					A[A_off+order] = 0;
					A_off = A_off + order;
					b[B_off] = p1_3;
					B_off = B_off + 1;
				}
				if (g2_3 != 0)
				{
					for (i=1; i++; i<order)
					{
						//(order-j+1)*points(i,1)^(order-j);
						A[A_off+order-i] = (order-i+1)*p2_1*(order-i);
					}
					A[A_off+order] = 0;
					b[B_off] = p2_3;
				}

				return(result);
		ENDC++;



		
		OUTPUT ( polyinterpc (0.1, 2, 4, 0.5, 6,
		5, 0, 1, 1,1,1, 1, 1 , 1));