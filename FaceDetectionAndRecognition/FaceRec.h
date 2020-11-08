#include <iostream>
#include <string>

#include <opencv2\core\core.hpp>
#include <opencv2\core.hpp>
#include <opencv2\face.hpp>
#include <opencv2\face\facerec.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\objdetect\objdetect.hpp>
#include <opencv2\opencv.hpp>
#include <direct.h>

#include <fstream>
#include <sstream>

using namespace std;
using namespace cv;
using namespace cv::face;

CascadeClassifier face_cascade;
string filename;
string name;
int filenumber = 0;

void detectAndDisplay(Mat frame)
{
	vector<Rect> faces;
	Mat frame_gray;
	Mat crop;
	Mat res;
	Mat gray;
	string text;
	stringstream sstm;

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

	Rect roi_b;
	Rect roi_c;

	size_t ic = 0;
	int ac = 0;

	size_t ib = 0;
	int ab = 0;

	for (ic = 0; ic < faces.size(); ic++) {
		
		roi_c.x = faces[ic].x;
		roi_c.y = faces[ic].y;
		roi_c.width = (faces[ic].width);
		roi_c.height = (faces[ic].height);

		ac = roi_c.width * roi_c.height;

		roi_b.x = faces[ib].x;
		roi_b.y = faces[ib].y;
		roi_b.width = (faces[ib].width);
		roi_b.height = (faces[ib].height);

		crop = frame(roi_b);
		resize(crop, res, Size(128, 128), 0, 0, INTER_LINEAR);
		cvtColor(crop, gray, COLOR_BGR2GRAY);
		stringstream ssfn;
		filename = "D:\\Egyetem\\Faces\\";
		ssfn << filename.c_str() << name << filenumber << ".jpg";
		filename = ssfn.str();
		imwrite(filename, res);
		filenumber++;

		Point pt1(faces[ic].x, faces[ic].y);
		Point pt2((faces[ic].x + faces[ic].height), (faces[ic].y + faces[ic].width));
		rectangle(frame, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);
	}

	sstm << "Crop area size: " << roi_b.width << "x" << roi_b.height << " Filename: " << filename;
	text = sstm.str();

	if (!crop.empty())
	{
		imshow("detected", crop);
	}
	else
		destroyWindow("detected");

}

void addFace()
{
	cout << "\nEnter your name: ";
	cin >> name;

	VideoCapture capture(0);

    if (!capture.isOpened())  
        return;

    if (!face_cascade.load("C:\\OpenCV\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt.xml"))
    {
        cout << "error" << endl;
        return ;
    };

    Mat frame;
	cout << "\nCapturing your face 10 times, press 'C' 10 times keeping your face front of the camera";
	char key;
	int i = 0;

    for (;;)
    {
        capture >> frame;

		detectAndDisplay(frame);
		i++;
		if (i == 10)
		{
			cout << "Face added";
			break;
		}

        int c = waitKey(10);

        if (27 == char(c))
        {
            break;
        }
    }

    return;
}

static void dbread(vector<Mat>& images, vector<int>& labels) {
	vector<cv::String> fn;
	filename = "D:\\Egyetem\\Faces\\";
	glob(filename, fn, false);

	size_t count = fn.size();

	for (size_t i = 0; i < count; i++)
	{
		string itsname="";
		char sep = '\\';
		size_t j = fn[i].rfind(sep, fn[i].length());
		if (j != string::npos) 
		{
			itsname=(fn[i].substr(j + 1, fn[i].length() - j-6));
		}
		images.push_back(imread(fn[i], 0));
		labels.push_back(atoi(itsname.c_str()));
	}
}

void eigenFaceTrainer() {
	vector<Mat> images;
	vector<int> labels;
	dbread(images, labels);
    cout << "Size of the images: " << images.size() << endl;
	cout << "Size of the labels: " << labels.size() << endl;
	cout << "Training begins..." << endl;

	Ptr<EigenFaceRecognizer> model = EigenFaceRecognizer::create();

	model->train(images, labels);

	model->save("D:\\Egyetem\\Faces\\eigenface.yml");

	cout << "Training finished...." << endl;  
	waitKey(10000);
}

void FaceRecognition() {

	cout << "Start recognizing..." << endl;

	Ptr<FaceRecognizer> model = FisherFaceRecognizer::create();
	model->read("D:\\Egyetem\\Faces\\eigenface.yml");

	Mat testSample = imread("D:\\Egyetem\\Faces\\0.jpg", 0);

	int img_width = testSample.cols;
	int img_height = testSample.rows;

	string window = "Capture - face detection";

	if (!face_cascade.load("C:\\OpenCV\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt.xml")) {
		cout << "Error loading file" << endl;
		return;
	}

	VideoCapture cap(0);

	if (!cap.isOpened())
	{
		cout << "exit" << endl;
		return;
	}

	namedWindow(window, 1);
	long count = 0;
	string Pname= "";

	while (true) {
		
		vector<Rect> faces;
		Mat frame;
		Mat grayScaleFrame;
		Mat original;

		cap >> frame;
		
		count = count + 1;

		if (!frame.empty()) {

			original = frame.clone();

			cvtColor(original, grayScaleFrame, COLOR_BGR2GRAY);

			face_cascade.detectMultiScale(grayScaleFrame, faces, 1.1, 3, 0, cv::Size(90, 90));

			std::string frameset = std::to_string(count);
			std::string faceset = std::to_string(faces.size());

			int width = 0, height = 0;

			for (int i = 0; i < faces.size(); i++) {

				Rect face_i = faces[i];

				Mat face = grayScaleFrame(face_i);

				Mat face_resized;
				cv::resize(face, face_resized, Size(img_width, img_height), 1.0, 1.0, INTER_CUBIC);

				int label = -1; double confidence = 0;
				model->predict(face_resized, label, confidence);

				cout << " confidence " << confidence << " Label: " << label << endl;
				
				Pname = to_string(label);

				rectangle(original, face_i, CV_RGB(0, 255, 0), 1);
				string text = Pname;

				int pos_x = std::max(face_i.tl().x - 10, 0);
				int pos_y = std::max(face_i.tl().y - 10, 0);

				putText(original, text, Point(pos_x, pos_y), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);

			}

			putText(original, "Frames: " + frameset, Point(30, 60), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
			putText(original, "Number of people detected: " + to_string(faces.size()), Point(30, 90), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);

			cv::imshow(window, original);

		}

		if (waitKey(30) >= 0) break;

	}
}
	