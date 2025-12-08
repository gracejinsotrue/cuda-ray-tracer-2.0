#include <iostream>
#include <ctime>

//**every CUDA API call returns an error code, we check cudaError_t result with a checkCudaErrors macro to output the error to stdout 
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const  func, const char *const file, int const line)
{
    if(result){
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

// allocate a nx*ny image size frame buffer to host RGB float values 
//calculated by the GPU to allow communication between CPU and GPU
//cudaMallocManaged allocates unified memory
// (Unified Memory is a single memory address space accessible from any processor in a system, allowing data to be read or written from code running on CPUs or GPUs.)
// tldr during CUDA runtime we move frame buffer on demand to GPU for rendering,
// back to CPU for outputting the PPM image

//also use cudaDeviceSyncrhonize to let CPU know when GPU is done rendering
// the basic idea is 
// int num_pixels nx*ny;
//size_t fb+size = 3*num_pixels*sizeof(float);
//allocate fb:
//float *fb;
//checkCudaErrors(cudaMallocManaged(void **)&fb,fb_size)


//	each thread gets its index by computing the offset to the beginning of its block ( the block index times block size = blockIdx.x * blockDim.x) and add the thread index within the block.
// blockIdx.x+blockDim.x+threadIdx.x is idiomatic CUDA
__global__ void render(float *fb, int max_x, int max_y){
    //each GPU thread calculates which pixel it is responsible for,
    //where hella threads run this simutaneously
    int i = threadIdx.x + blockIdx.x *blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i>=max_x) || (j>= max_y)) return; // some threads might be outside of image bounds

    // where in memory is this pixel data strating?
    //each pixel has 3 values RGB stored sequentially
    // (for example pixel at (5,2) stored in 1200-wide image):
    //pixel_index = 2*1200*3 + 5*3 = 7215
    int pixel_index = j*max_x*3 +i*3;

    fb[pixel_index+0] = float(i)/max_x; // red channel, increases left to right from 0.0 to 1.0
    fb[pixel_index+1] = float(j)/max_y; // green increases top to bottom, 0.0->1.0
    fb[pixel_index+2] = 0.2; // blue channel is whatever
}
//http://in1weekend.blogspot.com/2016/01/ray-tracing-in-one-weekend.html
//where the real shit happens, this is just ray tracing in one week

int main(){
    int nx = 1200;
    int ny = 600;
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << nx << "x" << ny << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx*ny;

    //malloc but CUDA
    size_t fb_size = 3*num_pixels*sizeof(float); // framebuffer size is just image data
    // we need to allocate size of this many pixels times float in bytes times 3 for RBG channel
    float *fb; //pointer fb to point to float values
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));
}