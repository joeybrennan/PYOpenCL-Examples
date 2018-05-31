
    // kernel:  pi    
    //
    // Purpose: accumulate partial sums of pi comp
    // 
    // input: float step_size
    //        int   niters per work item
    //        local float* an array to hold sums from each work item
    //
    // output: partial_sums   float vector of partial sums
    //
    double phi(double W,double R,double wavelen);
    double Wz(double Wo,double z,double wavelen);
    double Rz(double Wo,double z,double wavelen);
    double genhermite(int m,double W, double x);
    double fact(int x);
    double h_pJB( int m, int n, double x );
    void reduce(                                          
       __local  float*,                          
       __global float*);


    __kernel void pi(
       __global  float*    x,                          
       __global float*    AN,
       __global  float*    ETR,                          
       __global float*    ETI,
       const int steps,
       const int          modes,
       const float wavelen,
       const float z)                        
    {                                                          

       int i =get_global_id(0);                   
       double Wo = 1.5*wavelen;
       double accumR,accumI;    


       double WZ = Wz(Wo,z,wavelen);
       double RZ = Rz(Wo,z,wavelen);
       double phiz = phi(WZ,RZ,wavelen); 
       double k = (2*M_PI)/wavelen;
       if(i < steps){
           accumR = 0.0;
           accumI = 0.0;
           double xn = x[i];
           for(int j = 0; j<modes; j++){
               accumR += AN[j]*genhermite(j,WZ,xn)*cos(-k*(z+(pow(xn,2.0))/(2*RZ))+((double)j+0.5)*phiz);
               accumI += AN[j]*genhermite(j,WZ,xn)*sin(-k*(z+(pow(xn,2.0))/(2*RZ))+((double)j+0.5)*phiz);
            }
           ETR[i] = accumR;
           ETI[i] = accumI;
       }

    }


    double fact(int x){
        double factorial = 1;
        for(int i=1; i<=x; i++)
            {
                factorial *= i;              // factorial = factorial*i;
            }
        return factorial;
    }

    double genhermite(int m,double W, double x){

        double norm = pow((1/(sqrt(M_PI)*pow(2.0,(double)m-0.5)*fact(m)*W)),0.5);
        double herm = h_pJB(1,m,sqrt(2.0)*(x/W));
        return norm*herm*exp(-1*pow(x/W,2.0));

    }

    double Wz(double Wo,double z,double wavelen){
        return sqrt(pow(Wo,2.0)*(1+pow((wavelen*z)/(M_PI*pow(Wo,2.0)),2.0)));

    }

    double Rz(double Wo,double z,double wavelen){
        return z*(1+pow(((M_PI*pow(Wo,2.0))/(wavelen*z)),2.0));

    }

    double phi(double W,double R,double wavelen){
        return atan((M_PI*pow(W,2.0))/(wavelen*R));
    }


    double h_pJB( int m, int n, double x ){
      int i;
      int j;
      double p[60];

      if ( n < 0 )
      {
        return NULL;
      }

      for ( i = 0; i < m; i++ )
      {
        p[i+0*m] = 1.0;
      }

      if ( n == 0 )
      {
        return p[0];
      }

      for ( i = 0; i < m; i++ )
      {
        p[i+1*m] = 2.0 * x;
      }

      for ( j = 2; j <= n; j++ )
      {
        for ( i = 0; i < m; i++ )
        {
          p[i+j*m] = 2.0 * x * p[i+(j-1)*m]
            - 2.0 * ( double ) ( j - 1 ) * p[i+(j-2)*m];
        }
      }
      return p[n];
    }
