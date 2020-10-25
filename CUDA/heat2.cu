#include <stdio.h>
#include <stdlib.h>
#include"/usr/local/cuda/include/cuda_runtime.h"
#define STEPS 5000

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void init(float *d_a, float *d_b, int x_dim, int y_dim) {
    int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int elements = (x_dim * y_dim);
    for (int i = thread_id; i < elements; i += stride) {
        int j_index = i % y_dim;
        int i_index = i / y_dim;
        //d_a[i] = (float) (i_index * (x_dim - i_index - 1) * j_index * (y_dim - j_index - 1));
        //d_b[i] = (float) d_a[i];
        //*(d_a + i_index * sd[1] + j_index) = (float) ((i_index + c[1]*sd[0]) * (d[0] - (i_index + c[1]*sd[0]) - 1) * (j_index + c[0]*sd[1]) * (d[1] - (j_index + c[0]*sd[1]) - 1));
        unsigned long int value = i_index * (x_dim - i_index - 1) * j_index * (y_dim - j_index - 1);
        *(d_a + i_index * y_dim + j_index) = (float)(value);
        *(d_b + i_index * y_dim + j_index) = (float)(value);
    }
    __syncthreads();
}

void prtdat(int nx, int ny, float *u1, char *fnam) {
    int ix, iy;
    FILE *fp;
    
    fp = fopen(fnam, "w");
    for (iy = ny-1; iy >= 0; iy--) {
      for (ix = 0; ix <= nx-1; ix++) {
        if (ix == nx-1)
          fprintf(fp, "%12.1f           \n", *(u1+ix*ny+iy));
        else
          fprintf(fp, "%12.1f            ", *(u1+ix*ny+iy));
        }
      }
    fclose(fp);
}

__global__ void update(float *d_a, float *d_b, int x_dim, int y_dim){

    int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int elements = (x_dim * y_dim);
    for (int i = thread_id; i < elements; i += stride) {
        int j_index = i % y_dim;
        int i_index = i / y_dim;
      
          
        if ((i_index == 0 || i_index == x_dim-1) || (j_index == 0 || j_index == y_dim-1)) continue;
    
        struct Parms {
          float cx;
          float cy;
        } parms = {0.1, 0.1};
    
        *(d_a + i_index * y_dim + j_index) = *(d_b + i_index * y_dim + j_index)  +
                          parms.cx * (*(d_b + (i_index +1) * y_dim + j_index) +
                          *(d_b + (i_index -1) * y_dim + j_index) -
                          2.0 * *(d_b + i_index * y_dim + j_index)) +
                          parms.cy * (*(d_b + i_index * y_dim + j_index + 1) +
                        *(d_b + i_index * y_dim + j_index - 1) -
                          2.0 * *(d_b + i_index * y_dim + j_index));
    }
   
  
    __syncthreads();
  
  }
    

int main(int argc, char const *argv[]) {
    cudaEvent_t start, stop, stop2;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&stop2);
    if (argc != 4) {
      printf("error give the two dimensions of the array\n");
      return -1;
    }
  
    int x_dimension, y_dimension;
    x_dimension = atoi(argv[1]);
    y_dimension = atoi(argv[2]);
  
    float *d_a, *d_b, *a, *b;
    int bytes = x_dimension*y_dimension*sizeof(float);
    gpuErrchk( cudaMalloc((void**)&d_a, bytes));
    gpuErrchk( cudaMalloc((void**)&d_b, bytes));
    a = (float*) malloc(bytes);
    b = (float*) malloc(bytes);
    float *temp;
  
    gpuErrchk( cudaMemset(d_a, 0, bytes));
    gpuErrchk( cudaMemset(d_b, 0, bytes));

    int size = atoi(argv[3]);

    dim3 block_number(size);
    dim3 thread_number(32);


    init<<<block_number, thread_number>>>(d_a, d_b, x_dimension, y_dimension);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  

    gpuErrchk( cudaMemcpy(a, d_a, bytes, cudaMemcpyDeviceToHost));
    prtdat(x_dimension, y_dimension, a, "initial.dat");
    gpuErrchk(  cudaEventRecord(start));
    for (size_t i = 0; i < STEPS; i++) {

        update<<<block_number, thread_number>>>(d_a, d_b, x_dimension, y_dimension);

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        temp = d_a;
        d_a = d_b;
        d_b = temp;

    }
    gpuErrchk(  cudaEventRecord(stop));
    gpuErrchk( cudaMemcpy(a, temp, bytes, cudaMemcpyDeviceToHost));
    gpuErrchk( cudaEventRecord(stop2));
    //cudaMemcpy(b, d_b, bytes, cudaMemcpyDeviceToHost);



    prtdat(x_dimension, y_dimension, a, "final.dat");
    //prtdat(x_dimension, y_dimension, b, "2eend_res.dat");

    gpuErrchk( cudaFree(d_a));
    gpuErrchk( cudaFree(d_b));


    gpuErrchk(  cudaEventSynchronize(stop));
    float milliseconds = 0;
    gpuErrchk( cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Time: %f\n", milliseconds);

    gpuErrchk( cudaEventElapsedTime(&milliseconds, start, stop2));
    printf("Time with transfer: %f\n", milliseconds);
    return 0;

}