#include <stdio.h>
#include <Eigen/Dense>
#include <iostream>
#include <omp.h>
using namespace Eigen;
using namespace std;

#define NdArray_double Matrix<double, Dynamic, Dynamic, RowMajor>
#define NdArray_int32 Matrix<int, Dynamic, Dynamic, RowMajor>
extern "C" __declspec(dllexport) void maxpool2d_single(
	double *input,
	double *output,
	bool *mask,
	const int channel, const int height, const int width,
	const int kernel_size,
	const int stride)
{
	NdArray_double m1;
	omp_set_num_threads(omp_get_num_procs());
	#pragma omp parallel for
	for (int c = 0; c < channel; c++)
	{
		int index = c * height * width;
		int out_h, out_w, out_index;
		out_h = (height - kernel_size) / stride + 1;
		out_w = (width - kernel_size) / stride + 1;
		out_index = c * out_h * out_w;
		m1 = Map<NdArray_double>(input + index, height, width);
		for (int i = 0; i < height; i += stride)
		{
			for (int j = 0; j < width; j += stride)
			{
				// 取最大值
				NdArray_double sub_m1 = m1.block(i, j, kernel_size, kernel_size);
				int max_row, max_col;
				double max_value = sub_m1.maxCoeff(&max_row, &max_col);
				output[out_index + (i / stride) * out_w + (j / stride)] = max_value;
				mask[index + (i + max_row) * width + (j + max_col)] = true;
			}
		}
	}
}

/**
 * @brief maxpool2d的c++实现
 * @param input: 输入的矩阵指针 N*C*H*W
 * @param output: 输出的矩阵指针 N*C*H1*W1
 * @param mask: mask矩阵指针 N*C*H*W
 * @param batch: batch size
 * @param channel: channel size
 * @param height: height size
 * @param width: width size
 * @param kernel_size: kernel size
 * @param stride: stride size
 * @return: None
 */
extern "C" __declspec(dllexport) void maxpool2d(
	double *input,
	double *output,
	bool *mask,
	const int batch, const int channel, const int height, const int width,
	const int kernel_size,
	const int stride)
{

	// numpy默认按行，Eigen默认按列，以numpy为准
	omp_set_num_threads(omp_get_num_procs());
	#pragma omp parallel for
	for (int b = 0; b < batch; b++)
	{
		maxpool2d_single(input + b * channel * height * width,
						 output + b * channel * ((height - kernel_size) / stride + 1) * ((width - kernel_size) / stride + 1),
						 mask + b * channel * height * width,
						 channel, height, width,
						 kernel_size,
						 stride);
	}
}
/**
 * @brief im2col的c++实现
 * @param input: 输入的矩阵指针 H*W
 * @param output: 输出的矩阵指针 H1*W1
 * @param height: height size
 * @param width: width size
 * @param kernel_size: kernel size
 * @param stride: stride size
 * @return: None
 */
extern "C" __declspec(dllexport) void im2col(
	double *input,
	double *output,
	const int channel, const int height, const int width,
	const int kernel_size,
	const int stride)
{
	NdArray_double m1, temp;
	m1 = Map<NdArray_double>(input, height * channel, width);
	omp_set_num_threads(omp_get_num_procs());
	#pragma omp parallel for

	for (int i = 0; i < height - kernel_size + 1; i += stride)
	{
		#pragma omp parallel for
		for (int j = 0; j < width - kernel_size + 1; j += stride)
		{
			int row = ((i * ((width - kernel_size) / stride + 1) + j) / stride);
			#pragma omp parallel for
			for (int c = 0; c < channel; c++)
			{
				temp = m1.block(i + c * height, j, kernel_size, kernel_size);
				copy(temp.data(), temp.data() + kernel_size * kernel_size,
					 output + row * kernel_size * kernel_size * channel + c * kernel_size * kernel_size);
				// cout << temp << endl;
			}
		}
	}
}
/**
 * @brief conv2d的c++实现, 没有实现pad
 * @param input: 输入的矩阵指针 N*C*H*W
 * @param output: 输出的矩阵指针 N*C*H*W
 * @param col_image: col2im后的矩阵指针,用于计算梯度
 * @param kernel: 卷积核指针 O_C * I_C * K * K
 * @param bias: 偏置指针
 * @param batch: batch size
 * @param channel: channel size
 * @param height: height size
 * @param width: width size
 * @param output_channel: output channel size
 * @param kernel_size: kernel size
 * @param stride: stride size
 */
extern "C" __declspec(dllexport) void conv2d(
	double *input,
	double *output,
	double *col_image,
	double *kernel,
	double *bias,
	const int batch, const int input_channel, const int height, const int width,
	const int output_channel,
	const int kernel_size,
	const int stride)
{
	int conv_w = (width - kernel_size) / stride + 1;
	int conv_h = (height - kernel_size) / stride + 1;
	int col_w = kernel_size * kernel_size * input_channel;
	int col_h = conv_w * conv_h;
	RowVectorXd col_bias = Map<RowVectorXd>(bias, output_channel);
	MatrixXd col_kernel = Map<MatrixXd>(
		kernel, input_channel * kernel_size * kernel_size, output_channel);
	// cout << col_kernel << endl;
	omp_set_num_threads(omp_get_num_procs());
	#pragma omp parallel for
	for (int b = 0; b < batch; b++)
	{
		im2col(
			input + b * input_channel * height * width, // 输入指针
			col_image + b * col_w * col_h,				// 输出指针
			input_channel, height, width,
			kernel_size, stride);
		NdArray_double col_img_i = Map<NdArray_double>(col_image + b * col_w * col_h, col_h, col_w);
		MatrixXd res = col_img_i * col_kernel;
		res.rowwise() += col_bias;
		copy(res.data(), res.data() + res.size(), output + b * output_channel * conv_h * conv_w);
	}
}

/**
 * @brief conv2d的反向传播，没有实现pad，对卷积核进行翻转
 * @param input: 输入的矩阵指针 N*C*H*W
 * @param output: 输出的矩阵指针 N*C*H*W
 * @param kernel: 卷积核指针 O_C * I_C * K * K
 * @param batch: batch size
 * @param channel: channel size
 * @param height: height size
 * @param width: width size
 * @param output_channel: output channel size
 * @param kernel_size: kernel size
 * @param stride: stride size
 */
extern "C" __declspec(dllexport) void deconv2d(
	double *input,
	double *output,
	double *kernel,
	const int batch, const int input_channel, const int height, const int width,
	const int output_channel,
	const int kernel_size,
	const int stride)
{
	int conv_w = (width - kernel_size) / stride + 1;
	int conv_h = (height - kernel_size) / stride + 1;
	int col_w = kernel_size * kernel_size * input_channel;
	int col_h = conv_w * conv_h;
	double col_image_p[col_w * col_h];

	MatrixXd col_kernel = Map<MatrixXd>(
		kernel, input_channel * kernel_size * kernel_size, output_channel);
	omp_set_num_threads(omp_get_num_procs());
	#pragma omp parallel for
	for (int i = 0; i < input_channel; i++)
	{
		col_kernel.block(i * kernel_size * kernel_size, 0, kernel_size * kernel_size, output_channel).colwise().reverseInPlace();
	}
	#pragma omp parallel for
	for (int b = 0; b < batch; b++)
	{
		im2col(
			input + b * input_channel * height * width, // 输入指针
			col_image_p,								// 输出指针
			input_channel, height, width,
			kernel_size, stride);
		NdArray_double col_img = Map<NdArray_double>(col_image_p, col_h, col_w);
		MatrixXd res = col_img * col_kernel;
		copy(res.data(), res.data() + res.size(), output + b * output_channel * conv_h * conv_w);
	}
}

int main()
{
	// 假设我们有一个数组/
	double img_p[32] = {
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
		17, 18, 19, 20,
		21, 22, 23, 24,
		25, 26, 27, 28,
		29, 30, 31, 32};
	double mp1_img_p[8];
	bool mask_p[32];
	maxpool2d(img_p, mp1_img_p, mask_p, 2, 1, 4, 4, 2, 2);
	cout << Map<NdArray_double>(mp1_img_p, 4, 2) << endl;
	return 0;
	// RowVectorXd bias = Map<RowVectorXd>(bias_p, 2);
	// im2col(img, col_img_p, 3, 3, 3, 2, 1);
	// NdArray_double col_img = Map<NdArray_double>(col_img_p, 4, 12);
	// MatrixXd col_kernel = Map<MatrixXd>(kernel, 12, 2);
	// // NdArray_double m2 = Map<NdArray_double>(a2, 4, 12);
	// MatrixXd res = col_img * col_kernel;
	// res.rowwise() += bias;

	return 0;
}
