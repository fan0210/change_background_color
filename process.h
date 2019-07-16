#ifndef _PROCESS_H_
#define _PROCESS_H_

#include <opencv.hpp>
#include <vector>

uchar min(uchar r, uchar g, uchar b)
{
	uchar min_v = 255;
	min_v = min_v < r ? min_v : r;
	min_v = min_v < g ? min_v : g;
	min_v = min_v < b ? min_v : b;
	return min_v;
}

uchar max(uchar r, uchar g, uchar b)
{
	uchar max_v = 0;
	max_v = max_v > r ? max_v : r;
	max_v = max_v > g ? max_v : g;
	max_v = max_v > b ? max_v : b;
	return max_v;
}

/*
* 将BGR图像转为灰度图像，其中灰度图像像素值为BGR图像中R,G,B三个通道的最小值，这样可以尽量增加浅色背景和深色表格和字迹的对比度。
*/
cv::Mat getGray(const cv::Mat &imgSrc)
{
	cv::Mat imgGray(imgSrc.size(), CV_8UC1);
	for (int i = 0; i < imgSrc.rows; ++i)
	{
		const unsigned char *data_src = imgSrc.ptr<unsigned char>(i);
		unsigned char *data_g = imgGray.ptr<unsigned char>(i);
		for (int j = 0; j < imgSrc.cols; ++j)
		{
			auto b = data_src[j * 3];
			auto g = data_src[j * 3 + 1];
			auto r = data_src[j * 3 + 2];

			data_g[j] = min(r, g, b);
		}
	}
	return imgGray;
}

/*
* 将BGR图像中R,G,B三个通道中的最大值减去最小值来衡量饱和度S，
* 实验发现浅色背景或黑色字迹饱和度均很小，因此此操作提取出的“饱和度”专门为提取红色或蓝色印章所用。
*/
cv::Mat splitS(const cv::Mat &imgSrc)
{
	cv::Mat imgS(imgSrc.size(), CV_8UC1);
	for (int i = 0; i < imgS.rows; ++i)
	{
		const unsigned char *data_src = imgSrc.ptr<unsigned char>(i);
		unsigned char *data_r = imgS.ptr<unsigned char>(i);
		for (int j = 0; j < imgS.cols; ++j)
		{
			data_r[j] = max(data_src[j * 3], data_src[j * 3 + 1], data_src[j * 3 + 2]) - min(data_src[j * 3], data_src[j * 3 + 1], data_src[j * 3 + 2]);
			data_r[j] = 255 - data_r[j];
		}
	}
	return imgS;
}

/*
* 检测图中的直线。
*/
void detectLines(const cv::Mat &imgSrc, std::vector<cv::Vec4f> &lines)
{
	cv::Ptr<cv::LineSegmentDetector> ls = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);
	ls->detect(imgSrc, lines);
}

/*
* 将检测到的直线进行过滤。
* 由于黑色孔绝大部分位于表格的左外面，因此可通过检测表格组成的直线，
* 然后在检测到的直线中找到表格最左侧的竖线，这样该竖线的左面则全为背景，则覆盖了黑色孔。
* 此操作正是为了找到表格最左侧的竖线。
*/
cv::Vec4f filterLines(const cv::Mat &imgSrc, const std::vector<cv::Vec4f> &lines)
{
	cv::Vec4f left_line(imgSrc.cols, 0, imgSrc.cols, imgSrc.rows);
	for (size_t i = 0; i < lines.size(); ++i)
	{
		double distance = sqrt((lines[i][3] - lines[i][1])*(lines[i][3] - lines[i][1]) + (lines[i][2] - lines[i][0])*(lines[i][2] - lines[i][0]));
		double abs_k = fabs((lines[i][3] - lines[i][1]) / (lines[i][2] - lines[i][0] + 0.000001));
		if (distance > 0.05*imgSrc.rows && abs_k > 5)/*通过长短和斜率来过滤掉噪声和横线（只要竖线）*/
		{
			double min_x_left = (left_line[2] - left_line[0]) / (left_line[3] - left_line[1])*(imgSrc.rows / 2 - left_line[1]) + left_line[0];
			double min_x = (lines[i][2] - lines[i][0]) / (lines[i][3] - lines[i][1])*(imgSrc.rows / 2 - lines[i][1]) + lines[i][0];
			if (min_x < min_x_left)
				left_line = lines[i];
		}
	}
	return left_line;
}

/*
* 用于调用的主处理函数。
* 参数imgSrc为要处理的原图像，3通道BGR图像。
* 参数thresh1为二值化“饱和度”图像所用的阈值，一般不需要改动。
* 参数thresh2为二值化原图对应的灰度图所用的阈值，一般不需要改动。
* 参数color为目标背景的颜色。
* 函数返回最终替换背景后的图像。
*/
cv::Mat process(const cv::Mat &imgSrc, int thresh1 = 10, int thresh2 = 20, const cv::Scalar &color = cv::Scalar(200, 255, 255))
{
	cv::Mat gray = getGray(imgSrc);
	cv::Mat imgS = splitS(imgSrc);

	std::vector<cv::Vec4f> lines;
	detectLines(gray, lines);
	cv::Vec4f left_line = filterLines(imgSrc, lines);

	//动态阈值二值化
	cv::adaptiveThreshold(imgS, imgS, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 51, thresh1);
	cv::adaptiveThreshold(gray, gray, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 51, thresh2);

	cv::Mat output = imgSrc.clone();
	for (int i = 0; i < output.rows; ++i)
	{
		unsigned char *data_src = output.ptr<unsigned char>(i);
		unsigned char *data_res = gray.ptr<unsigned char>(i);
		unsigned char *data_r = imgS.ptr<unsigned char>(i);

		//该行表格最左边竖线所对应的横坐标
		double min_x = (left_line[2] - left_line[0]) / (left_line[3] - left_line[1])*(i - left_line[1]) + left_line[0];

		for (int j = 0; j < output.cols; ++j)
		{
			if ((data_res[j] == 255 && data_r[j] == 255) || j < min_x/*去掉黑色孔*/)
			{
				data_src[j * 3] = color[0];
				data_src[j * 3 + 1] = color[1];
				data_src[j * 3 + 2] = color[2];
			}
		}
	}

	cv::blur(output, output, cv::Size(3, 3));
	return output;
}

#endif
