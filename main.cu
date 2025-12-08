#include <iostream>
#include <time.h>
//main.cu more like overkill.cu 

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
    fb[pixel_index+2] = 0.67; // blue channel is whatever
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

    //clock shit
    //let's define two variables to store time measurements
    clock_t start;
    clock_t stop;
    start = clock(); //execution time ofprocessor time used by program since its start

    //launch ts kernel
    // (number of blocks in x direction , number of blocks in y direction) + 1 to be safe
    dim3 blocks(nx/tx+1,ny/ty+1); // dim3 is a CUDA type for 3d dimensions it is a built in CUDA struct with 3 integers, also is a uint
    dim3 threads(tx,ty); // threads per blocks! (8,8,1)

    //idomatic way is function<<<numBlocks, blockSize>>>(whatever parameters) 
    
    //should be 151 x 76 = 11476 blocks with each block being 8x8 threads, 
    // i cannot wait for this to kill my laptop
    render<<<blocks, threads>>>(fb, nx,ny);
    // and the CPU continues immediately
    // then we check for kernel launch errors
    //cudaGetLastError() returns any error from last cuda operation
    //
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize()); // prolly the most important line

    stop = clock(); // let's record ending time to see how long ts took 
    double elapsed = ((double)(stop-start))/CLOCKS_PER_SEC; // latter is a  macro constat defined in ctime
    //number of clock ticks per second as returned by std::clock() function, raw clock tick -> seconds
    std::cerr << "ts took" <<elapsed << "seconds.\n";

    //this part writes it to PPM, basically what ray tracing in one week did
    // write frame buffer to PPM format: https://netpbm.sourceforge.net/doc/ppm.html
    // ppm requires exact header of 1) P3\n = magic number indicating ASCII color PPM
    // nx << ny is space separated width and height
    //\n255\n newline before and after the colors
    // full header output would be
    //P3
    //1200  600
    //255 
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";

    //note how we are iterating through rows "backwards"
    // this is because PPM image coordinates expects the top row first
    // but my GPU code stored bottom row first in memoery (grace ur stupid)
    //so we will read it backwards...
    for (int j = ny-1; j>=0; j--){
        for (int i = 0; i<nx; i++){
            size_t pixel_index = j*3*nx+i*3; // we visit every pixel
            //read from framebuffer these channels
            float red = fb[pixel_index+0];
            float green = fb[pixel_index+1];
            float blue = fb[pixel_index+2];
            // we have these floats as 0.0 - 1.0, but we want to convert to RGB range up to 255

            int convert_red = int(255.99*red); //.99 for the case of casting properly to 255 b/c int truncates
            int convert_green = int(255.99*green);
            int convert_blue = int(255.99*blue);

            std::cout << convert_red << " " << convert_green << " " << convert_blue << "\n";
        }
    }
    checkCudaErrors(cudaFree(fb)); // remember to deallocate this big fat memory 
    



}