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
* ��BGRͼ��תΪ�Ҷ�ͼ�����лҶ�ͼ������ֵΪBGRͼ����R,G,B����ͨ������Сֵ���������Ծ�������ǳɫ��������ɫ�����ּ��ĶԱȶȡ�
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
* ��BGRͼ����R,G,B����ͨ���е����ֵ��ȥ��Сֵ���������Ͷ�S��
* ʵ�鷢��ǳɫ�������ɫ�ּ����ͶȾ���С����˴˲�����ȡ���ġ����Ͷȡ�ר��Ϊ��ȡ��ɫ����ɫӡ�����á�
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
* ���ͼ�е�ֱ�ߡ�
*/
void detectLines(const cv::Mat &imgSrc, std::vector<cv::Vec4f> &lines)
{
	cv::Ptr<cv::LineSegmentDetector> ls = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);
	ls->detect(imgSrc, lines);
}

/*
* ����⵽��ֱ�߽��й��ˡ�
* ���ں�ɫ�׾��󲿷�λ�ڱ��������棬��˿�ͨ���������ɵ�ֱ�ߣ�
* Ȼ���ڼ�⵽��ֱ�����ҵ�������������ߣ����������ߵ�������ȫΪ�������򸲸��˺�ɫ�ס�
* �˲�������Ϊ���ҵ�������������ߡ�
*/
cv::Vec4f filterLines(const cv::Mat &imgSrc, const std::vector<cv::Vec4f> &lines)
{
	cv::Vec4f left_line(imgSrc.cols, 0, imgSrc.cols, imgSrc.rows);
	for (size_t i = 0; i < lines.size(); ++i)
	{
		double distance = sqrt((lines[i][3] - lines[i][1])*(lines[i][3] - lines[i][1]) + (lines[i][2] - lines[i][0])*(lines[i][2] - lines[i][0]));
		double abs_k = fabs((lines[i][3] - lines[i][1]) / (lines[i][2] - lines[i][0] + 0.000001));
		if (distance > 0.05*imgSrc.rows && abs_k > 5)/*ͨ�����̺�б�������˵������ͺ��ߣ�ֻҪ���ߣ�*/
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
* ���ڵ��õ�����������
* ����imgSrcΪҪ�����ԭͼ��3ͨ��BGRͼ��
* ����thresh1Ϊ��ֵ�������Ͷȡ�ͼ�����õ���ֵ��һ�㲻��Ҫ�Ķ���
* ����thresh2Ϊ��ֵ��ԭͼ��Ӧ�ĻҶ�ͼ���õ���ֵ��һ�㲻��Ҫ�Ķ���
* ����colorΪĿ�걳������ɫ��
* �������������滻�������ͼ��
*/
cv::Mat process(const cv::Mat &imgSrc, int thresh1 = 10, int thresh2 = 20, const cv::Scalar &color = cv::Scalar(200, 255, 255))
{
	cv::Mat gray = getGray(imgSrc);
	cv::Mat imgS = splitS(imgSrc);

	std::vector<cv::Vec4f> lines;
	detectLines(gray, lines);
	cv::Vec4f left_line = filterLines(imgSrc, lines);

	//��̬��ֵ��ֵ��
	cv::adaptiveThreshold(imgS, imgS, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 51, thresh1);
	cv::adaptiveThreshold(gray, gray, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 51, thresh2);

	cv::Mat output = imgSrc.clone();
	for (int i = 0; i < output.rows; ++i)
	{
		unsigned char *data_src = output.ptr<unsigned char>(i);
		unsigned char *data_res = gray.ptr<unsigned char>(i);
		unsigned char *data_r = imgS.ptr<unsigned char>(i);

		//���б���������������Ӧ�ĺ�����
		double min_x = (left_line[2] - left_line[0]) / (left_line[3] - left_line[1])*(i - left_line[1]) + left_line[0];

		for (int j = 0; j < output.cols; ++j)
		{
			if ((data_res[j] == 255 && data_r[j] == 255) || j < min_x/*ȥ����ɫ��*/)
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
