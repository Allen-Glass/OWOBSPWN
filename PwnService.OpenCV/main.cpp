// OpenCV 3 demo - Convert a color image to gray
// See YouTube tutorial https://www.youtube.com/watch?v=9lra7lTKpbs
#include <opencv2/opencv.hpp>
#include <opencv2/text/ocr.hpp>
#include "opencv2/text.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"

#include <vector>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;
using namespace cv::text;

int main() {
	const std::string img_file{ "foo.jpg" };

	// Check if we can open the file
	cv::Mat input = cv::imread(img_file, 1);
	if (!input.data) {
		std::cout << "Can't open file " << img_file << '\n';
		exit(1);
	}
	// Convert to gray
	cv::Mat output;
	cvtColor(input, output, cv::COLOR_BGR2GRAY);

	// Show the original and the result
	cv::namedWindow("Original image", cv::WINDOW_AUTOSIZE);
	cv::imshow("Original image", input);

	cv::namedWindow("Gray image", cv::WINDOW_AUTOSIZE);
	cv::imshow("Gray image", output);

	// Wait until the presses any key
	cv::waitKey(0);

	return 0;
}