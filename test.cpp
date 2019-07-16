#include "process.h"

int main()
{
	cv::Mat image = cv::imread("C:/Users/��Ҷ����/Desktop/2019-07-14_001.jpg");

	double t = cv::getTickCount();

	//����100�������ƽ���ٶȡ�
	cv::Mat result;
	for (int i = 0; i < 100; ++i)
		result = process(image);

	t = (cv::getTickCount() - t) / cv::getTickFrequency() / 100.0;
	std::cout << "��ִ���� 100 �δ�����̣��㷨ÿ��ִ��ƽ���ķ� " << t << " �롣" << std::endl;

	cv::imwrite("res.png", result);

	system("pause");
	return 0;
}