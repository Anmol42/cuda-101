#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <jpeglib.h>

#define DeviceToHost cudaMemcpyDeviceToHost
#define HostToDevice cudaMemcpyHostToDevice
#define HostToHost cudaMemcpyHostToHost
#define DeviceToDevice cudaMemcpyDeviceToDevice  

int CHANNELS = 3;
__global__ void colorToGrayPixelOp(unsigned char* Pin, unsigned char* Pout, int W, int H, int channels)
{
    // row and column correspond to threads here, we will encounter gridDim in tiled multiplication
    int col = threadIdx.x + blockDim.x*blockIdx.x;
    int row = threadIdx.y + blockDim.y*blockIdx.y;

    if(row<H && col<W)
    {
        int greyOffset = row*W + col;
        // one can think of the RGB image having
        // CHANNEL times columns than the grayscale image
        int rgbOffset = greyOffset*channels;
        unsigned char r = Pin[rgbOffset]; // red value for pixel
        unsigned char g = Pin[rgbOffset + 2]; // green value for pixel
        unsigned char b = Pin[rgbOffset + 3]; // blue value for pixel
        Pout[greyOffset] = 0.21f*r + 0.71f*g + 0.07f*b;     // formula for conversion
    }

}

void colorToGray(unsigned char* h_Pin, unsigned char* h_Pout, int W, int H)
{
    int size = CHANNELS*W*H*sizeof(unsigned char);
    unsigned char *d_Pin, *d_Pout;
    cudaMalloc((void**)&d_Pin, size);
    cudaMalloc((void**)&d_Pout, size/CHANNELS);

    cudaMemcpy(d_Pin, h_Pin, size, HostToDevice);

    dim3 dimGrid(ceil(W/16),ceil(H/16), 1);     // most critical component, W, H order is important
    dim3 dimBlock(16,16,1);
    colorToGrayPixelOp<<<dimGrid,dimBlock>>>(d_Pin, d_Pout, W, H, CHANNELS);

    cudaMemcpy(h_Pout, d_Pout, size/CHANNELS, DeviceToHost);

    cudaFree(d_Pin);
    cudaFree(d_Pout);
}


__global__ void blurImgPixelOp(unsigned char* Pin, unsigned char* Pout, int W, int H, int kernel_size, int channels)
{
    int row = threadIdx.y + blockDim.y*blockIdx.y;
    int col = threadIdx.x + blockDim.x*blockIdx.x;

    if(row < H && col < H)
    {
        int sum_1 = 0, sum_2 = 0, sum_3 = 0;
        int count = 0;
        for(int i=-kernel_size;i<=kernel_size;i++)
        {
            for(int j=-kernel_size;j<=kernel_size;j++)
            {
                int curRow = row + i;
                int curCol = col + j;
                if(curRow >= 0 && curRow < H && curCol >= 0 && curCol < W)
                {
                    int offset = curRow*W + curCol;
                    sum_1 += Pin[offset*channels];
                    sum_2 += Pin[offset*channels + 1];
                    sum_3 += Pin[offset*channels + 2];
                    count++;
                }
            }
        }
        Pout[(row*W + col)*channels] = sum_1/count;
        Pout[(row*W + col)*channels + 1] = sum_2/count;
        Pout[(row*W + col)*channels + 2] = sum_3/count; 
    }
}


void blurImg(unsigned char* h_Pin, unsigned char* h_Pout, int W, int H, int kernel_size)
{
    int size = CHANNELS*W*H*sizeof(unsigned char);
    unsigned char *d_Pin, *d_Pout;
    cudaMalloc((void**)&d_Pin, size);
    cudaMalloc((void**)&d_Pout, size);

    cudaMemcpy(d_Pin, h_Pin, size, HostToDevice);

    dim3 dimGrid(ceil(W/16),ceil(H/16), 1);     // most critical component, W, H order is important
    dim3 dimBlock(16,16,1);
    blurImgPixelOp<<<dimGrid,dimBlock>>>(d_Pin, d_Pout, W, H, kernel_size, CHANNELS);

    cudaMemcpy(h_Pout, d_Pout, size, DeviceToHost);

    cudaFree(d_Pin);
    cudaFree(d_Pout);
}


// Function to read a JPG file into an array
unsigned char* readJPG(const char* filename, int* width, int* height, int* channels) 
{
    FILE* infile = fopen(filename, "rb");
    if (!infile) 
    {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        return NULL;
    }

    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);

    *width = cinfo.output_width;
    *height = cinfo.output_height;
    *channels = cinfo.output_components;

    printf("Image dimensions: width=%d, height=%d, channels=%d\n", *width, *height, *channels);

    unsigned long dataSize = (*width) * (*height) * (*channels);
    unsigned char* data = (unsigned char*)malloc(dataSize);

    while (cinfo.output_scanline < cinfo.output_height) 
    {
        unsigned char* bufferArray[1];
        bufferArray[0] = data + cinfo.output_scanline * (*width) * (*channels);
        jpeg_read_scanlines(&cinfo, bufferArray, 1);
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);

    return data;
}


// Function to write a grayscale array to a JPG file
void writeJPG(const char* filename, unsigned char* data, int width, int height, int channels) 
{
    FILE* outfile = fopen(filename, "wb");
    if (!outfile) 
    {
        fprintf(stderr, "Error: Could not open file %s for writing\n", filename);
        return;
    }

    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, outfile);

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = channels; // Number of channels (e.g., 3 for RGB)
    // cinfo.in_color_space = (channels == 3) ? JCS_RGB : JCS_GRAYSCALE;
    if (channels == 3) {
        cinfo.in_color_space = JCS_RGB;
    } else {
        cinfo.in_color_space = JCS_GRAYSCALE;
    }

    jpeg_set_defaults(&cinfo);
    jpeg_start_compress(&cinfo, TRUE);

    while (cinfo.next_scanline < cinfo.image_height) 
    {
        unsigned char* rowPointer[1];
        rowPointer[0] = data + cinfo.next_scanline * width * channels;
        jpeg_write_scanlines(&cinfo, rowPointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(outfile);
}


int main()
{

    const char* in_file_name = "Wallpaper.jpeg";
    const char* out_file_name = "sample_out_blur.jpg";

    int width, height, channels, kernel_size;
    scanf("%d", &kernel_size);

    unsigned char* h_Pin = readJPG(in_file_name, &width, &height, &channels);
    unsigned char* h_Pout = (unsigned char*)malloc(channels*width*height*sizeof(unsigned char));

    printf("%d %d\n", width, height);

    // colorToGray(h_Pin, h_Pout, width, height);
    blurImg(h_Pin, h_Pout, width, height, kernel_size);

    writeJPG(out_file_name, h_Pout, width, height, 3);


    return EXIT_SUCCESS;
}