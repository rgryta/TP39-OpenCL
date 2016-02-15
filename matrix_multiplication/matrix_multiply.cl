__kernel void matrix_multiply(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C)
{
	uint row = get_global_id(0);
	uint col = get_global_id(1);
	
	
	float acc = 0.0f;
	for (int k=0; k<K; k++) {
        acc += A[row*K + k] * B[k*N + col];
    }
 
    // Store the result
    C[row*N + col] = acc;
};
