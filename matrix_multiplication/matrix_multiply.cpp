#include <stdio.h>
#include <stdlib.h>
#include <iostream> // for standard I/O
#include <math.h>
#include <time.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#define STRING_BUFFER_LEN 1024
#define BILLION 1000000000L
using namespace std;




void print_clbuild_errors(cl_program program,cl_device_id device)
	{
		cout<<"Program Build failed\n";
		size_t length;
		char buffer[2048];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
		cout<<"--- Build log ---\n "<<buffer<<endl;
		exit(1);
	}

unsigned char ** read_file(const char *name) {
  size_t size;
  unsigned char **output=(unsigned char **)malloc(sizeof(unsigned char *));
  FILE* fp = fopen(name, "rb");
  if (!fp) {
    printf("no such file:%s",name);
    exit(-1);
  }

  fseek(fp, 0, SEEK_END);
  size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  *output = (unsigned char *)malloc(size);
  unsigned char **outputstr=(unsigned char **)malloc(sizeof(unsigned char *));
  *outputstr= (unsigned char *)malloc(size);
  if (!*output) {
    fclose(fp);
    printf("mem allocate failure:%s",name);
    exit(-1);
  }

  if(!fread(*output, size, 1, fp)) printf("failed to read file\n");
  fclose(fp);
  printf("file size %d\n",size);
  printf("-------------------------------------------\n");
  snprintf((char *)*outputstr,size,"%s\n",*output);
  printf("%s\n",*outputstr);
  printf("-------------------------------------------\n");
  return outputstr;
}
void callback(const char *buffer, size_t length, size_t final, void *user_data)
{
     fwrite(buffer, 1, length, stdout);
}


void checkError(int status, const char *msg) {
	if(status!=CL_SUCCESS)	
		printf("%s\n",msg);
}

// Randomly generate a floating-point number between -10 and 10.
float rand_float() {
  return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f;
}

int main()
{
     char char_buffer[STRING_BUFFER_LEN];
     cl_platform_id platform;
     cl_device_id device;
     cl_context context;
     cl_context_properties context_properties[] =
     { 
          CL_CONTEXT_PLATFORM, 0,
          CL_PRINTF_CALLBACK_ARM, (cl_context_properties)callback,
          CL_PRINTF_BUFFERSIZE_ARM, 0x1000,
          0
     };
     cl_command_queue queue;
     cl_program program;
     cl_kernel kernel;



	//--------------------------------------------------------------------
	const unsigned N = 1000;
	const unsigned M = N;
	const unsigned K = N;
	bool print = true;
	if (N>10) print = false;
	float* ref_output=(float *) malloc(sizeof(float)*N*M);
	cl_mem input_a_buf; // num_devices elements
	cl_mem input_b_buf; // num_devices elements
	cl_mem output_buf; // num_devices elements
	int status;


     clGetPlatformIDs(1, &platform, NULL);
     clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
     printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
     clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
     printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
     clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
     printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);

     context_properties[1] = (cl_context_properties)platform;
     clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
     context = clCreateContext(context_properties, 1, &device, NULL, NULL, NULL);
     queue = clCreateCommandQueue(context, device, 0, NULL);

     unsigned char **opencl_program=read_file("matrix_multiply.cl");
     program = clCreateProgramWithSource(context, 1, (const char **)opencl_program, NULL, NULL);
     if (program == NULL)
	{
         printf("Program creation failed\n");
         return 1;
	}	
     int success=clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	 if(success!=CL_SUCCESS) print_clbuild_errors(program,device);
     kernel = clCreateKernel(program, "matrix_multiply", NULL);
 // Input buffers.
    input_a_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,
       M*K* sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input A");

    input_b_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,
        K*N* sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input B");

    // Output buffer.
    output_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,
        M*N* sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for output");


    cl_event write_event[2];
	cl_event kernel_event;//,finish_event;

	cl_int errcode;
    float* input_a = (float *)clEnqueueMapBuffer(queue, input_a_buf, CL_TRUE, CL_MAP_WRITE, 0, N*sizeof(float), 0, NULL, &write_event[0], &errcode);
	checkError(errcode, "Failed to map input A");

	float* input_b = (float *)clEnqueueMapBuffer(queue, input_b_buf, CL_TRUE, CL_MAP_WRITE, 0, N*sizeof(float), 0, NULL, &write_event[1], &errcode);
	checkError(errcode, "Failed to map input B");

	
	timespec start,end;
    double diff;
	
    for (unsigned i = 0; i < M*K; i++) {
    	input_a[i] = rand_float();
    }
    for (unsigned i = 0; i < K*N; i++) {
    	input_b[i] = rand_float();
    }

	printf("\n\nA = [");
	for (unsigned m=0; m<M; m++) {
		for (unsigned k=0; k<K; k++) {
			if (print)	printf("%.2f  ", input_a[m*K+k]);
		}
	if (print)	printf("\n");
	}
	printf("]\n\n");

	printf("\n\nB = [");
	for (unsigned m=0; m<K; m++) {
		for (unsigned k=0; k<N; k++) {
	if (print)	printf("%.2f  ", input_b[m*K+k]);
		}
	if (print)	printf("\n");
	}
	printf("]\n\n");

    clock_gettime(CLOCK_REALTIME, &start);

	for (unsigned m=0; m<M; m++) {
		for (unsigned n=0; n<N; n++) {
		    float acc = 0.0f;
		    for (unsigned k=0; k<K; k++) {
		        acc += input_a[m*M + k] * input_b[k*N + n];
		    }
		    ref_output[m*M + n] = acc;
		}
	}

    clock_gettime(CLOCK_REALTIME, &end);

	printf("\n\nC = [");
	for (unsigned m=0; m<M; m++) {
		for (unsigned k=0; k<N; k++) {
	if (print)	printf("%.2f  ", ref_output[m*K+k]);
		}
	if (print)	printf("\n");
	}
	printf("]\n\n");

    diff = (double) (end.tv_sec - start.tv_sec) + (double)(end.tv_nsec-start.tv_nsec)/BILLION;
    printf ("CPU took %.4lf seconds to run.\n", diff );

    clEnqueueUnmapMemObject(queue, input_a_buf, input_a, 0, NULL, NULL);
    clEnqueueUnmapMemObject(queue, input_b_buf, input_b, 0, NULL, NULL);

    // Set kernel arguments.
    unsigned argi = 0;

	status = clSetKernelArg(kernel, argi++, sizeof(int), (void*)&M);
	status = clSetKernelArg(kernel, argi++, sizeof(int), (void*)&N);
	status = clSetKernelArg(kernel, argi++, sizeof(int), (void*)&K);

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_a_buf);
    checkError(status, "Failed to set argument 1");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_b_buf);
    checkError(status, "Failed to set argument 2");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_buf);
    checkError(status, "Failed to set argument 3");

    const size_t global_work_size[2] = {M, N};
    const size_t local_work_size[2] = {10, 10};
	clock_gettime(CLOCK_REALTIME, &start);
    status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL,
        global_work_size, local_work_size, 2, write_event, &kernel_event);
    
	clWaitForEvents(1, &kernel_event);
	
	clock_gettime(CLOCK_REALTIME, &end);
	
	diff = (double) (end.tv_sec - start.tv_sec) + (double)(end.tv_nsec-start.tv_nsec)/BILLION;
	checkError(status, "Failed to launch kernel");

    float* output = (float *)clEnqueueMapBuffer(queue, output_buf, CL_TRUE, CL_MAP_READ, 0, N*sizeof(float), 0, NULL, NULL, &errcode);
    checkError(errcode, "Failed to map output");
	
	

   printf ("GPU took %.4lf seconds to run.\n", diff );
// Verify results.
   bool pass = true;

for(unsigned j = 0; j < N && pass; ++j) {
      if(fabsf(output[j] - ref_output[j]) > 1.0e-5f) {
        printf("Failed verification @ index %d\nOutput: %f\nReference: %f\nin_a: %f, in_b: %f\n",
            j, output[j], ref_output[j], input_a[j], input_b[j]);
        pass = false;
      }
}
	if (pass) {
		printf("Passed all!");
	}

	printf("\n\nC2 = [");
	for (unsigned m=0; m<M; m++) {
		for (unsigned k=0; k<N; k++) {
	if (print)	printf("%.2f  ", output[m*K+k]);
		}
	if (print)	printf("\n");
	}
	printf("]\n\n");

	clEnqueueUnmapMemObject(queue, output_buf, output, 0, NULL, NULL);
    // Release local events.
    clReleaseEvent(write_event[0]);
    clReleaseEvent(write_event[1]);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseMemObject(input_a_buf);
	clReleaseMemObject(input_b_buf);
	clReleaseMemObject(output_buf);
	clReleaseProgram(program);
	clReleaseContext(context);


//--------------------------------------------------------------------






     clFinish(queue);

     return 0;
}
