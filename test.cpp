#include "process.h"

int main()
{
	cv::Mat image = cv::imread("C:/Users/落叶归心/Desktop/2019-07-14_001.jpg");

	double t = cv::getTickCount();

	//处理100次来获得平均速度。
	cv::Mat result;
	for (int i = 0; i < 100; ++i)
		result = process(image);

	t = (cv::getTickCount() - t) / cv::getTickFrequency() / 100.0;
	std::cout << "共执行了 100 次处理过程，算法每次执行平均耗费 " << t << " 秒。" << std::endl;

	cv::imwrite("res.png", result);

	system("pause");
	return 0;
}