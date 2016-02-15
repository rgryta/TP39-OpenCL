#include <stdio.h>

#include <stdlib.h>

#include <iostream> // for standard I/O

#include <fstream>

#include <time.h>

#include "opencv2/opencv.hpp"

#include <math.h>

#include <CL/cl.h>

#include <CL/cl_ext.h>

#define STRING_BUFFER_LEN 1024
#define BILLION 1000000000L

using namespace cv;
using namespace std;

void print_clbuild_errors(cl_program program, cl_device_id device) {
  cout << "Program Build failed\n";
  size_t length;
  char buffer[2048];
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, & length);
  cout << "--- Build log ---\n " << buffer << endl;
  exit(1);
}

unsigned char ** read_file(const char * name) {
  size_t size;
  unsigned char ** output = (unsigned char ** ) malloc(sizeof(unsigned char * ));
  FILE * fp = fopen(name, "rb");
  if (!fp) {
    printf("no such file:%s", name);
    exit(-1);
  }

  fseek(fp, 0, SEEK_END);
  size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  * output = (unsigned char * ) malloc(size);
  unsigned char ** outputstr = (unsigned char ** ) malloc(sizeof(unsigned char * ));
  * outputstr = (unsigned char * ) malloc(size);
  if (! * output) {
    fclose(fp);
    printf("mem allocate failure:%s", name);
    exit(-1);
  }

  if (!fread( * output, size, 1, fp)) printf("failed to read file\n");
  fclose(fp);
  printf("file size %d\n", size);
  printf("-------------------------------------------\n");
  snprintf((char * ) * outputstr, size, "%s\n", * output);
  printf("%s\n", * outputstr);
  printf("-------------------------------------------\n");
  return outputstr;
}
void callback(const char * buffer, size_t length, size_t final, void * user_data) {
  fwrite(buffer, 1, length, stdout);
}

void checkError(int status,
  const char * msg) {
  if (status != CL_SUCCESS)
    printf("%s with status: %d\n", msg, status);
}

//#define SHOW
int main(int, char ** ) {
  VideoCapture camera("./bourne.mp4");
  if (!camera.isOpened()) // check if we succeeded
    return -1;

  const string NAME = "./output.avi"; // Form the new name with container

  int ex = static_cast < int > (CV_FOURCC('M', 'J', 'P', 'G'));
  Size S = Size((int) camera.get(CV_CAP_PROP_FRAME_WIDTH), // Acquire input size
    (int) camera.get(CV_CAP_PROP_FRAME_HEIGHT));
  //Size S =Size(1280,720);
  cout << "SIZE:" << S << endl;

  VideoWriter outputVideo;
  outputVideo.open(NAME, ex, 25, S, true);

  if (!outputVideo.isOpened()) {
    cout << "Could not open the output video for write: " << NAME << endl;
    return -1;
  }
  int count = 0;
  //const char * windowName = "filter";
  #ifdef SHOW
  namedWindow(windowName);
  #endif

  //--------------------------------------------
  timespec start, end;
  double diff, tot = 0;
  //CPU
  while (true) {
    Mat cameraFrame, displayframe;
    count = count + 1;
    if (count > 299) break;
    camera >> cameraFrame;
    Mat filterframe = Mat(cameraFrame.size(), CV_8UC3);
    Mat grayframe, edge_x, edge_y, edge, edge_inv;
    cvtColor(cameraFrame, grayframe, CV_BGR2GRAY);
    //printf("CPU|Frame: %d\n", count);
    clock_gettime(CLOCK_REALTIME, & start);
    //--------------------------------------------	
    GaussianBlur(grayframe, grayframe, Size(3, 3), 0, 0);
    GaussianBlur(grayframe, grayframe, Size(3, 3), 0, 0);
    GaussianBlur(grayframe, grayframe, Size(3, 3), 0, 0);
    Scharr(grayframe, edge_x, CV_8U, 0, 1, 1, 0, BORDER_DEFAULT);
    Scharr(grayframe, edge_y, CV_8U, 1, 0, 1, 0, BORDER_DEFAULT);
    addWeighted(edge_x, 0.5, edge_y, 0.5, 0, edge);
    threshold(edge, edge, 80, 255, THRESH_BINARY_INV);
    //--------------------------------------------
    cvtColor(edge, edge_inv, CV_GRAY2BGR);
    // Clear the output image to black, so that the cartoon line drawings will be black (ie: not drawn).
    memset((char * ) displayframe.data, 0, displayframe.step * displayframe.rows);
    grayframe.copyTo(displayframe, edge);
    cvtColor(displayframe, displayframe, CV_GRAY2BGR);
    outputVideo << displayframe;
    #ifdef SHOW
    imshow(windowName, displayframe);
    #endif
    clock_gettime(CLOCK_REALTIME, & end);
    diff = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / BILLION;
    tot += diff;
  }
  outputVideo.release();
  camera.release();

  //GPU
  VideoCapture camera_g("./bourne.mp4");
  if (!camera_g.isOpened()) // check if we succeeded
    return -1;

  const string NAME_GPU = "./output_gpu.avi"; // Form the new name with container

  ex = static_cast < int > (CV_FOURCC('M', 'J', 'P', 'G'));
  S = Size((int) camera_g.get(CV_CAP_PROP_FRAME_WIDTH), // Acquire input size
    (int) camera_g.get(CV_CAP_PROP_FRAME_HEIGHT));

  VideoWriter outputVideo_gpu;
  outputVideo_gpu.open(NAME_GPU, ex, 25, S, true);

  if (!outputVideo_gpu.isOpened()) {
    cout << "Could not open the output video for write: " << NAME_GPU << endl;
    return -1;
  }

  //--------------------------------------------
  char char_buffer[STRING_BUFFER_LEN];
  cl_platform_id platform;
  cl_device_id device;
  cl_context_properties context_properties[] = {
    CL_CONTEXT_PLATFORM,
    0,
    CL_PRINTF_CALLBACK_ARM,
    (cl_context_properties) callback,
    CL_PRINTF_BUFFERSIZE_ARM,
    0x1000,
    0
  };

  int status = 0;
  clGetPlatformIDs(1, & platform, NULL);
  clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
  clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
  clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
  context_properties[1] = (cl_context_properties) platform;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, & device, NULL);
  cl_context context = clCreateContext(context_properties, 1, & device, NULL, NULL, NULL);
  cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

  unsigned char ** opencl_program = read_file("videoproj.cl");
  cl_program program = clCreateProgramWithSource(context, 1, (const char ** ) opencl_program, NULL, NULL);
  if (program == NULL) {
    printf("Program creation failed\n");
    return 1;
  }
  int success = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (success != CL_SUCCESS) print_clbuild_errors(program, device);

  int window_size = 640 * 360;
  int window_bordered_s = 642 * 362;

  cl_mem input_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, window_bordered_s * sizeof(char), NULL, & status);
  checkError(status, "Failed to create buffer for input");

  cl_mem output_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, window_bordered_s * sizeof(char), NULL, & status);
  checkError(status, "Failed to create buffer for output");

  cl_mem output_buf_s = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, window_size * sizeof(char), NULL, & status);
  checkError(status, "Failed to create buffer for output");

  const size_t global_work_size[2] = {
    360,
    640
  };
  const size_t local_work_size[2] = {
    12,
    16
  };

  cl_event write_event;
  cl_event kernel_event;
  cl_int errcode;

  double tot_g = 0.0;
  count = 0;

  cl_kernel kernel_g = clCreateKernel(program, "videoproj_gaussian", NULL);
  unsigned argi = 0;
  status = clSetKernelArg(kernel_g, argi++, sizeof(cl_mem), & input_buf);
  checkError(status, "Failed to set argument");
  status = clSetKernelArg(kernel_g, argi++, sizeof(cl_mem), & output_buf);
  checkError(status, "Failed to set argument");

  cl_kernel kernel_g_rev = clCreateKernel(program, "videoproj_gaussian", NULL);
  argi = 0;
  status = clSetKernelArg(kernel_g_rev, argi++, sizeof(cl_mem), & output_buf);
  checkError(status, "Failed to set argument");
  status = clSetKernelArg(kernel_g_rev, argi++, sizeof(cl_mem), & input_buf);
  checkError(status, "Failed to set argument");

  cl_kernel kernel_s = clCreateKernel(program, "videoproj_sobel", NULL);

  argi = 0;
  status = clSetKernelArg(kernel_s, argi++, sizeof(cl_mem), & output_buf);
  checkError(status, "Failed to set argument");
  status = clSetKernelArg(kernel_s, argi++, sizeof(cl_mem), & output_buf_s);
  checkError(status, "Failed to set argument");

  while (true) {
    Mat cameraFrame, displayframe;
    count = count + 1;
    if (count > 299) break;
    camera_g >> cameraFrame;
    Mat filterframe = Mat(cameraFrame.size(), CV_8UC3);
    Mat new_frame_grey, new_frame;
    cvtColor(cameraFrame, new_frame_grey, CV_BGR2GRAY);
    //printf("GPU|Frame: %d\n", count);
    clock_gettime(CLOCK_REALTIME, & start);
    char * input = (char * ) clEnqueueMapBuffer(queue, input_buf, CL_TRUE, CL_MAP_WRITE, 0, window_bordered_s * sizeof(char), 0, NULL, & write_event, & errcode);
    checkError(errcode, "Failed to map input");

    copyMakeBorder(new_frame_grey, new_frame, 1, 1, 1, 1, BORDER_CONSTANT);

    //float* matData = (float*)myMat.data;
    memcpy(input, new_frame.data, window_bordered_s * sizeof(char));

    clEnqueueUnmapMemObject(queue, input_buf, input, 0, NULL, NULL);

    status = clEnqueueNDRangeKernel(queue, kernel_g, 2, NULL, global_work_size, local_work_size, 1, & write_event, & kernel_event);
    checkError(status, "Failed to launch kernel");
    clEnqueueBarrier(queue);
    status = clEnqueueNDRangeKernel(queue, kernel_g_rev, 2, NULL, global_work_size, local_work_size, 1, & write_event, & kernel_event);
    checkError(status, "Failed to launch kernel");
    clEnqueueBarrier(queue);
    status = clEnqueueNDRangeKernel(queue, kernel_g, 2, NULL, global_work_size, local_work_size, 1, & write_event, & kernel_event);
    checkError(status, "Failed to launch kernel");
    clEnqueueBarrier(queue);

    status = clEnqueueNDRangeKernel(queue, kernel_s, 2, NULL, global_work_size, local_work_size, 1, & write_event, & kernel_event);
    checkError(status, "Failed to launch kernel");
    clWaitForEvents(1, & kernel_event);

    char * output_s = (char * ) clEnqueueMapBuffer(queue, output_buf_s, CL_TRUE, CL_MAP_READ, 0, window_size * sizeof(char), 0, NULL, NULL, & errcode);
    checkError(errcode, "Failed to map output");

    memcpy(new_frame_grey.data, output_s, window_size * sizeof(char));

    clEnqueueUnmapMemObject(queue, output_buf_s, output_s, 0, NULL, NULL);

    cvtColor(new_frame_grey, new_frame_grey, CV_GRAY2BGR);
    outputVideo_gpu << new_frame_grey;

    clock_gettime(CLOCK_REALTIME, & end);
    diff = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / BILLION;
    tot_g += diff;
  }
  outputVideo_gpu.release();
  camera_g.release();
  printf("CPU time: %.2f, FPS %.2lf .\n", tot, 299.0 / tot);
  printf("GPU time: %.2f, FPS %.2lf .\n", tot_g, 299.0 / tot_g);

  return EXIT_SUCCESS;

}
