// OpenCV 3 demo - Convert a color image to gray
// See YouTube tutorial https://www.youtube.com/watch?v=9lra7lTKpbs
#include <opencv2/opencv.hpp>
#include <opencv2/text.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"

#include <vector>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;
using namespace cv::text;

class Parallel_extractCSER : public cv::ParallelLoopBody
{
private:
	vector<Mat>& channels;
	vector< vector<cv::text::ERStat> >& regions;
	vector< Ptr<ERFilter> > er_filter1;
	vector< Ptr<ERFilter> > er_filter2;

public:
	Parallel_extractCSER(vector<Mat>& _channels, vector< vector<ERStat> >& _regions,
		vector<Ptr<ERFilter> >_er_filter1, vector<Ptr<ERFilter> >_er_filter2)
		: channels(_channels), regions(_regions), er_filter1(_er_filter1), er_filter2(_er_filter2) {}

	virtual void operator()(const cv::Range& r) const CV_OVERRIDE
	{
		for (int c = r.start; c < r.end; c++)
		{
			er_filter1[c]->run(channels[c], regions[c]);
			er_filter2[c]->run(channels[c], regions[c]);
		}
	}
	Parallel_extractCSER& operator=(const Parallel_extractCSER& a);
};

template <class T>
class Parallel_OCR : public cv::ParallelLoopBody
{
private:
	vector<Mat>& detections;
	vector<string>& outputs;
	vector< vector<Rect> >& boxes;
	vector< vector<string> >& words;
	vector< vector<float> >& confidences;
	vector< Ptr<T> >& ocrs;

public:
	Parallel_OCR(vector<Mat>& _detections, vector<string>& _outputs, vector< vector<Rect> >& _boxes,
		vector< vector<string> >& _words, vector< vector<float> >& _confidences,
		vector< Ptr<T> >& _ocrs)
		: detections(_detections), outputs(_outputs), boxes(_boxes), words(_words),
		confidences(_confidences), ocrs(_ocrs)
	{}

	virtual void operator()(const cv::Range& r) const CV_OVERRIDE
	{
		for (int c = r.start; c < r.end; c++)
		{
			ocrs[c % ocrs.size()]->run(detections[c], outputs[c], &boxes[c], &words[c], &confidences[c], OCR_LEVEL_WORD);
		}
	}
	Parallel_OCR& operator=(const Parallel_OCR& a);
};

const char* keys =
{
	"{@input   | 0 | camera index or video file name}"
	"{ image i |   | specify input image}"
};

//Discard wrongly recognised strings
bool   isRepetitive(const string& s);
//Draw ER's in an image via floodFill
void   er_draw(vector<Mat>& channels, vector<vector<ERStat> >& regions, vector<Vec2i> group, Mat& segmentation);

int main() {
	const std::string img_file{ "poggers.png" };

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

	int num_ocrs = 10;
	vector< Ptr<OCRTesseract> > ocrs;
	vector<Mat> channels;
	vector<vector<ERStat> > regions(2); //two channels
	for (int o = 0; o < num_ocrs; o++)
	{
		ocrs.push_back(OCRTesseract::create());
	}
	string voc = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

	vector< Ptr<OCRHMMDecoder> > decoders;
	Mat emission_p = Mat::eye(62, 62, CV_64FC1);
	Mat transition_p;
	Mat frame, image, gray, out_img;
	decoders.push_back(OCRHMMDecoder::create(loadOCRHMMClassifierNM("OCRHMM_knn_model_data.xml.gz"),
		voc, transition_p, emission_p));

	// Detect character groups
	vector< vector<Vec2i> > nm_region_groups;
	vector<Rect> nm_boxes;

	erGrouping(frame, channels, regions, nm_region_groups, nm_boxes, ERGROUPING_ORIENTATION_HORIZ);

	/*Text Recognition (OCR)*/
	int bottom_bar_height = out_img.rows / 7;
	copyMakeBorder(frame, out_img, 0, bottom_bar_height, 0, 0, BORDER_CONSTANT, Scalar(150, 150, 150));
	float scale_font = (float)(bottom_bar_height / 85.0);
	vector<string> words_detection;
	float min_confidence1 = 0.f, min_confidence2 = 0.f;

	min_confidence1 = 51.f;
	min_confidence2 = 60.f;

	vector<Mat> detections;

	for (int i = 0; i < (int)nm_boxes.size(); i++)
	{
		rectangle(out_img, nm_boxes[i].tl(), nm_boxes[i].br(), Scalar(255, 255, 0), 3);

		Mat group_img = Mat::zeros(frame.rows + 2, frame.cols + 2, CV_8UC1);
		group_img(nm_boxes[i]).copyTo(group_img);
		copyMakeBorder(group_img, group_img, 15, 15, 15, 15, BORDER_CONSTANT, Scalar(0));
		detections.push_back(group_img);
	}
	vector<string> outputs((int)detections.size());
	vector< vector<Rect> > boxes((int)detections.size());
	vector< vector<string> > words((int)detections.size());
	vector< vector<float> > confidences((int)detections.size());

	String region_types_str[2] = { "ERStats", "MSER" };
	String grouping_algorithms_str[2] = { "exhaustive_search", "multioriented" };
	String recognitions_str[2] = { "Tesseract", "NM_chain_features + KNN" };

	// parallel process detections in batches of ocrs.size() (== num_ocrs)
	for (int i = 0; i < (int)detections.size(); i = i + (int)num_ocrs)
	{
		Range r;
		if (i + (int)num_ocrs <= (int)detections.size())
			r = Range(i, i + (int)num_ocrs);
		else
			r = Range(i, (int)detections.size());

		parallel_for_(r, Parallel_OCR<OCRTesseract>(detections, outputs, boxes, words, confidences, ocrs));
	}

	for (int i = 0; i < (int)detections.size(); i++)
	{
		outputs[i].erase(remove(outputs[i].begin(), outputs[i].end(), '\n'), outputs[i].end());
		//cout << "OCR output = \"" << outputs[i] << "\" length = " << outputs[i].size() << endl;
		if (outputs[i].size() < 3)
			continue;

		for (int j = 0; j < (int)boxes[i].size(); j++)
		{
			boxes[i][j].x += nm_boxes[i].x - 15;
			boxes[i][j].y += nm_boxes[i].y - 15;

			//cout << "  word = " << words[j] << "\t confidence = " << confidences[j] << endl;
			if ((words[i][j].size() < 2) || (confidences[i][j] < min_confidence1) ||
				((words[i][j].size() == 2) && (words[i][j][0] == words[i][j][1])) ||
				((words[i][j].size() < 4) && (confidences[i][j] < min_confidence2)) ||
				isRepetitive(words[i][j]))
				continue;
			words_detection.push_back(words[i][j]);
			rectangle(out_img, boxes[i][j].tl(), boxes[i][j].br(), Scalar(255, 0, 255), 3);
			Size word_size = getTextSize(words[i][j], FONT_HERSHEY_SIMPLEX, (double)scale_font, (int)(3 * scale_font), NULL);
			rectangle(out_img, boxes[i][j].tl() - Point(3, word_size.height + 3), boxes[i][j].tl() + Point(word_size.width, 0), Scalar(255, 0, 255), -1);
			putText(out_img, words[i][j], boxes[i][j].tl() - Point(1, 1), FONT_HERSHEY_SIMPLEX, scale_font, Scalar(255, 255, 255), (int)(3 * scale_font));
		}
	}
	double t_all = (double)getTickCount();
	t_all = ((double)getTickCount() - t_all) * 1000 / getTickFrequency();
	int text_thickness = 1 + (out_img.rows / 500);
	string fps_info = format("%2.1f Fps. %dx%d", (float)(1000 / t_all), frame.cols, frame.rows);
	putText(out_img, fps_info, Point(10, out_img.rows - 5), FONT_HERSHEY_DUPLEX, scale_font, Scalar(255, 0, 0), text_thickness);
	putText(out_img, region_types_str[0], Point((int)(out_img.cols * 0.5), out_img.rows - (int)(bottom_bar_height / 1.5)), FONT_HERSHEY_DUPLEX, scale_font, Scalar(255, 0, 0), text_thickness);
	putText(out_img, grouping_algorithms_str[0], Point((int)(out_img.cols * 0.5), out_img.rows - ((int)(bottom_bar_height / 3) + 4)), FONT_HERSHEY_DUPLEX, scale_font, Scalar(255, 0, 0), text_thickness);
	putText(out_img, recognitions_str[0], Point((int)(out_img.cols * 0.5), out_img.rows - 5), FONT_HERSHEY_DUPLEX, scale_font, Scalar(255, 0, 0), text_thickness);
	imshow("recognition", out_img);
	// Wait until the presses any key
	cv::waitKey(0);

	return 0;
}

bool isRepetitive(const string& s)
{
	int count = 0;
	int count2 = 0;
	int count3 = 0;
	int first = (int)s[0];
	int last = (int)s[(int)s.size() - 1];
	for (int i = 0; i < (int)s.size(); i++)
	{
		if ((s[i] == 'i') ||
			(s[i] == 'l') ||
			(s[i] == 'I'))
			count++;
		if ((int)s[i] == first)
			count2++;
		if ((int)s[i] == last)
			count3++;
	}
	if ((count > ((int)s.size() + 1) / 2) || (count2 == (int)s.size()) || (count3 > ((int)s.size() * 2) / 3))
	{
		return true;
	}

	return false;
}

void er_draw(vector<Mat>& channels, vector<vector<ERStat> >& regions, vector<Vec2i> group, Mat& segmentation)
{
	for (int r = 0; r < (int)group.size(); r++)
	{
		ERStat er = regions[group[r][0]][group[r][1]];
		if (er.parent != NULL) // deprecate the root region
		{
			int newMaskVal = 255;
			int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
			floodFill(channels[group[r][0]], segmentation, Point(er.pixel % channels[group[r][0]].cols, er.pixel / channels[group[r][0]].cols),
				Scalar(255), 0, Scalar(er.level), Scalar(0), flags);
		}
	}
}