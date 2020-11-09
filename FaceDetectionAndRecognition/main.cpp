#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include <opencv2\objdetect\objdetect.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2\face.hpp>
#include <opencv2\face\facerec.hpp>

#include <fstream>
#include <sstream>

#include "FaceRec.h"

using namespace std;
using namespace cv;
using namespace cv::face;

int main()
{
	int choice;
	cout << "1. Arcfelismeres\n";
	cout << "2. Arcdetektalas\n";
	cout << "A valasztott funkcio: ";
	cin >> choice;
	switch (choice)
	{
	case 1:
		FaceRecognition();
		break;
	case 2:
		addFace();
		eigenFaceTrainer();
		break;
	default:
		return 0;
	}
	waitKey();
	return 0;
}