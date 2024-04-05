#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <chrono>

// Define whether to print array elements
#define PRINT 1

// Default size of arrays
int SZ = 100000000;

// Pointers to arrays
int *v1, *v2, *v_out;

// OpenCL memory objects for arrays
cl_mem bufV1, bufV2, bufV_out;

// OpenCL variables
cl_device_id device_id;
cl_context context;
cl_program program;
cl_kernel kernel;
cl_command_queue queue;
cl_event event = NULL;
int err;

// Function prototypes
cl_device_id create_device();
void setup_openCL_device_context_queue_kernel(char *filename, char *kernelname);
cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename);
void setup_kernel_memory();
void copy_kernel_args();
void free_memory();
void init(int *&A, int size);
void print(int *A, int size);

// Main function
int main(int argc, char **argv) {
    // Check if array size is provided as command line argument
    if (argc > 1) {
        SZ = atoi(argv[1]);
    }

    // Initialize arrays
    init(v1, SZ);
    init(v2, SZ);
    init(v_out, SZ); 

    // Set global work size for OpenCL kernel
    size_t global[1] = {(size_t)SZ};

    // Print arrays if enabled
    print(v1, SZ);
    print(v2, SZ);
   
    // Setup OpenCL environment
    setup_openCL_device_context_queue_kernel((char *)"./vector_ops_ocl.cl", (char *)"vector_add_ocl");
    setup_kernel_memory();
    copy_kernel_args();
    
    // Start measuring execution time
    auto start = std::chrono::high_resolution_clock::now();
    
    // Execute OpenCL kernel
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, NULL, 0, NULL, &event);
    clWaitForEvents(1, &event);
    
    // Read the result from OpenCL device
    clEnqueueReadBuffer(queue, bufV_out, CL_TRUE, 0, SZ * sizeof(int), &v_out[0], 0, NULL, NULL);
    
    // Print result array if enabled
    print(v_out, SZ);
    
    // Stop measuring execution time
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_time = stop - start;

    // Print kernel execution time
    printf("Kernel Execution Time: %f ms\n", elapsed_time.count());
    
    // Free OpenCL and memory resources
    free_memory();
}

// Initialize array with random values
void init(int *&A, int size) {
    A = (int *)malloc(sizeof(int) * size);

    for (long i = 0; i < size; i++) {
        A[i] = rand() % 100; 
    }
}

// Print array elements
void print(int *A, int size) {
    // Check if printing is disabled
    if (PRINT == 0) {
        return;
    }

    // Print only first and last 5 elements if array size is large
    if (PRINT == 1 && size > 15) {
        for (long i = 0; i < 5; i++) {
            printf("%d ", A[i]);         
        }
        printf(" ..... ");
        for (long i = size - 5; i < size; i++) {
            printf("%d ", A[i]); 
        }
    }
    else {
        for (long i = 0; i < size; i++) {
            printf("%d ", A[i]); 
        }
    }
    printf("\n----------------------------\n");
}

// Free memory and OpenCL resources
void free_memory() {
    clReleaseMemObject(bufV1);
    clReleaseMemObject(bufV2);
    clReleaseMemObject(bufV_out);
  
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);

    free(v1);
    free(v2);
    free(v_out); 
}

// Set kernel arguments
void copy_kernel_args() {
    clSetKernelArg(kernel, 0, sizeof(int), (void *)&SZ);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&bufV1);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&bufV2);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&bufV_out);

    if (err < 0) {
        perror("Couldn't create a kernel argument");
        printf("error = %d", err);
        exit(1);
    }
}

// Setup OpenCL memory objects
void setup_kernel_memory() {
    bufV1 = clCreateBuffer(context, CL_MEM_READ_WRITE, SZ * sizeof(int), NULL, NULL);
    bufV2 = clCreateBuffer(context, CL_MEM_READ_WRITE, SZ * sizeof(int), NULL, NULL);
    bufV_out = clCreateBuffer(context, CL_MEM_READ_WRITE, SZ * sizeof(int), NULL, NULL);

    clEnqueueWriteBuffer(queue, bufV1, CL_TRUE, 0, SZ * sizeof(int), &v1[0], 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufV2, CL_TRUE, 0, SZ * sizeof(int), &v2[0], 0, NULL, NULL);
}

// Setup OpenCL environment
void setup_openCL_device_context_queue_kernel(char *filename, char *kernelname) {
    device_id = create_device();
    cl_int err;
    
    // Create OpenCL context
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (err < 0) {
        perror("Couldn't create a context");
        exit(1);
    }

    // Build OpenCL program
    program = build_program(context, device_id, filename);

    // Create OpenCL command queue
    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    if (err < 0) {
        perror("Couldn't create a command queue");
        exit(1);
    };

    // Create OpenCL kernel
    kernel = clCreateKernel(program, kernelname, &err);
    if (err < 0) {
        perror("Couldn't create a kernel");
        printf("error =%d", err);
        exit(1);
    };
}

// Build OpenCL program from source file
cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename) {
    cl_program program;
    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;

    // Read OpenCL program source from file
    program_handle = fopen(filename, "r");
    if (program_handle == NULL) {
        perror("Couldn't find the program file");
        exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program
