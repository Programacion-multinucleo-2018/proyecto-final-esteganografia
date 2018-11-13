#include <iostream>
#include <cstdio>
#include <cmath>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "common.h"
#include <cuda_runtime.h>

#include <chrono>

#include <fstream>
#include <iostream>

using namespace std;

__global__ void cipher_kernel(unsigned char* input, unsigned char* output, int width, int height, int step, unsigned char* text,long int textSize)
{

	const long int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const long int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	//Current pixel
	const long int ti = yIndex * width + xIndex;

	if(ti<textSize)
	{

		//cifrar
		unsigned char ca = text[ti];
		long int tid = ti*8;
		for(long int x = 0;x < 8;x++)
		{
			long int bit = (ca >> x) & 1U;
			unsigned char imgColor = input[tid+x];
			if(bit)
			{
				imgColor |= 1 << 0;
			}
			else
			{
				imgColor = imgColor & ~(1u<<0);
			}
			

			output[tid+x] = imgColor;
		}
		
	}
}

__global__ void decipher_kernel(unsigned char* input, int width, int height, int step, unsigned char * word, long int charLimit)
{

	const long int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const long int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	
	//Current pixel
	const long int ti = yIndex * width + xIndex;

	if(ti<charLimit)
	{
		long int tid = ti*8;
		unsigned char letra;
		for(long int x = 0;x < 8;x++)
		{
			unsigned char color = input[tid+x];
			long int bit = (color >> 0) & 1U;
			letra |= bit << x;
		}
		word[ti] = letra;
	}
}

void cipher(const cv::Mat& input, cv::Mat& output, char * word, long long int textSize)
{

	cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << endl;

	//char word[] = {'h','e','l','l','o'};
	long int wordSize=textSize;

	size_t colorBytes = input.step * input.rows;
	size_t grayBytes = output.step * output.rows;

	unsigned char *d_input, *d_output, *d_word;
	
	// Allocate device memory
	SAFE_CALL(cudaMalloc(&d_input, colorBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc(&d_output, grayBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc(&d_word, wordSize), "CUDA Malloc Failed");

	// Copy data from OpenCV input image to device memoryfermentum,
	SAFE_CALL(cudaMemcpy(d_input, input.ptr(), colorBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
	SAFE_CALL(cudaMemcpy(d_output, output.ptr(), colorBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
	SAFE_CALL(cudaMemcpy(d_word, word, wordSize, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	// Specify a reasonable block size
	const dim3 block(16, 16);

	// Calculate grid size to cover the whole image
	const dim3 grid((long int)ceil((float)input.cols / block.x), (long int)ceil((float)input.rows/ block.y));
	printf("cipher_kernel<<<(%d, %d) , (%d, %d)>>>\n", grid.x, grid.y, block.x, block.y);

	// Launch the color conversion kernel
	auto start_cpu =  chrono::high_resolution_clock::now();
	cipher_kernel <<<grid, block >>>(d_input, d_output, input.cols, input.rows, static_cast<int>(input.step),d_word,textSize);
	auto end_cpu =  chrono::high_resolution_clock::now();
	chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
	printf("elapsed %f ms\n", duration_ms.count());

	// Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

	// Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(), d_output, grayBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	// Free the device memory
	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_word), "CUDA Free Failed");
}

char* decipher(const cv::Mat& input)
{

	
	size_t colorBytes = input.step * input.rows;
	
	size_t wordSize = (input.cols*input.rows*3);
	
	long int charLimit = floor(wordSize/8);
	char *word = new char[charLimit]();

	unsigned char *d_input, *d_word;
	
	// Allocate device memory
	SAFE_CALL(cudaMalloc(&d_input, colorBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc(&d_word, charLimit), "CUDA Malloc Failed");

	// Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input, input.ptr(), colorBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
	SAFE_CALL(cudaMemcpy(d_word, word, charLimit, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	// Specify a reasonable block size
	const dim3 block(16, 16);

	// Calculate grid size to cover the whole image
	const dim3 grid((long int)ceil((float)input.cols / block.x), (long int)ceil((float)input.rows/ block.y));
	printf("cipher_kernel<<<(%d, %d) , (%d, %d)>>>\n", grid.x, grid.y, block.x, block.y);

	// Launch the color conversion kernel
	auto start_cpu =  chrono::high_resolution_clock::now();
	decipher_kernel <<<grid, block >>>(d_input, input.cols, input.rows, static_cast<int>(input.step),d_word,charLimit);
	auto end_cpu =  chrono::high_resolution_clock::now();
	chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
	printf("elapsed %f ms\n", duration_ms.count());
	
	// Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

	SAFE_CALL(cudaMemcpy(word, d_word, charLimit, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	// Free the device memory
	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_word), "CUDA Free Failed");

	ofstream exit ("mensaje.txt");
	
	for(long int x = 0; x < charLimit; x++)
	{
		exit << word[x];
	}

	exit.close();	

	return word;
}

void menu(cv::Mat input)
{
	printf("¿Que deseas hacer?\n 1. Codificar un mensaje\n 2. Decodificar un mensaje\n");

	string op = "1";
	//getline(cin,op);
 
	if(op == "1")
	{
		//Create output image
		cv::Mat output(input.rows, input.cols, CV_8UC3);
		output = input.clone();
		
		printf("¿de que archivo desea obtener el mensaje?\n");

		string file,msg = "";
		//getline(cin,file);
		string line = "file.txt";
		ifstream infile(file);
		
		
		long int msgSize = 0;

		while (std::getline(infile, line))
		{
			msgSize+=line.size();
			msg += line;
		}


		char * temp = new char [msgSize];
		strcpy (temp, msg.c_str());

		cipher(input, output, temp,  msgSize);

		//Allow the windows to resize
		/*namedWindow("Input", cv::WINDOW_NORMAL);
		namedWindow("Output", cv::WINDOW_NORMAL);

		//Show the input and output
		imshow("Input", input);
		imshow("Output", output);*/

		imwrite("Output.png", output);

		//Wait for key press	
		cv::waitKey();
	}
	else if(op == "2")
	{
		char * word = decipher(input);
	}
	else	
		printf("Error");
}

int main(int argc, char *argv[])
{
	string imagePath;

	if(argc < 2)
		imagePath = "image.jpg";
  	else
  		imagePath = argv[1];


	// Read input image from the disk
	cv::Mat input = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);

	if (input.empty())
	{
		cout << "Image Not Found!" << std::endl;
		cin.get();
		return -1;
	}

	menu(input);

	return 0;
}
