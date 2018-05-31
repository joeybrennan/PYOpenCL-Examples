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

    double genhermite(int m,double W, double x,double a1,double b1,double a2,double b2);
    double fact(int x);
    double h_pJB( int m, int n, double x );
    void reduce(                                          
       __local  float*,                          
       __global float*);


    __kernel void pi(
       const int themodenumber,
       const int          niters,
       const float        step_size,
       const float        start,
       const float wavelen,
       __local  float*    local_sums,                          
       __global float*    partial_sums)                        
    {                                                          
       int num_wrk_items  = get_local_size(0);                 
       int local_id       = get_local_id(0);                   
       int group_id       = get_group_id(0);                   
       double W = 1.5*wavelen;
       double t = 6*wavelen;
       double d = 3*wavelen;
       double a1,b1,a2,b2;

       a1 = -d - t/2;
       b1 = -t/2;
       a2 = t/2;
       b2 = d + t/2;


       

       double x,accum = 0.0;    

       int i,istart,iend;                                      
       int modeNum = (int) themodenumber;
       istart = (group_id * num_wrk_items + local_id) * niters;
       iend   = istart+niters;      

       for(i = istart; i<iend; i++){
           x = start + i*step_size;
           accum += genhermite(modeNum,W,x,a1,b1,a2,b2);
       } 
       local_sums[local_id] = accum;
       barrier(CLK_LOCAL_MEM_FENCE);

       reduce(local_sums, partial_sums);                  
        }

    //------------------------------------------------------------------------------
    //
    // OpenCL function:  reduction    
    //
    // Purpose: reduce across all the work-items in a work-group
    // 
    // input: local float* an array to hold sums from each work item
    //
    // output: global float* partial_sums   float vector of partial sums
    //

    void reduce(                                          
       __local  float*    local_sums,                          
       __global float*    partial_sums)                        
    {                                                          
       int num_wrk_items  = get_local_size(0);                 
       int local_id       = get_local_id(0);                   
       int group_id       = get_group_id(0);                   

       float sum;                              
       int i;                                      

       if (local_id == 0) {                      
          sum = 0.0f;                            

          for (i=0; i<num_wrk_items; i++) {        
              sum += local_sums[i];             
          }                                     

          partial_sums[group_id] = sum;         
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

     double genhermite(int m,double W, double x,double a1,double b1,double a2,double b2){
        if(-18.0e-3<= x && x <= -9.0e-3){
                double norm = pow((1/(sqrt(M_PI)*pow(2.0,(double)m-0.5)*fact(m)*W)),0.5);
                double herm = h_pJB(1,m,sqrt(2.0)*(x/W));
                return norm*herm*exp(-1*pow(x/W,2.0));
               }
        else if(9.0e-3<= x && x <= 18.0e-3)
        {
            double norm = pow((1/(sqrt(M_PI)*pow(2.0,(double)m-0.5)*fact(m)*W)),0.5);
            double herm = h_pJB(1,m,sqrt(2.0)*(x/W));
            return norm*herm*exp(-1*pow(x/W,2.0));
       }

        else{
            return 0;
           }

    }

    double h_pJB( int m, int n, double x )

    {
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