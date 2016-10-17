#pragma once

//include opencv libraries
#include "opencv2/core/core.hpp" //New C++ data structures and arithmetic routines.
#include "opencv2/flann/miniflann.hpp" // Approximate nearest neighbor matching functions. (Mostly for internal use)
#include "opencv2/imgproc/imgproc.hpp" //New C++ image processing functions.
#include "opencv2/photo/photo.hpp" //Algorithms specific to handling and restoring photographs
#include "opencv2\video\video.hpp" //Video tracking and background segmentation routines
#include "opencv2\features2d\features2d.hpp" //Two-dimensional feature tracking support
#include "opencv2\objdetect\objdetect.hpp" //Cascade face detector; latent SVM; HoG; planar patch detector
#include "opencv2\calib3d\calib3d.hpp" //Calibration and stereo.
#include "opencv2\ml\ml.hpp" //Machine learning: clustering, pattern recognition.
#include "opencv2\highgui\highgui.hpp" //New C++ image display, sliders, buttons, mouse, I/O.
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2\opencv.hpp"

//include c++ libraries
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <map>
#include <cstring>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <ctime>

//using namespaces
using namespace std;
using namespace cv;

//defines
#define X first
#define Y second
#define pb push_back
#define mp make_pair

//typedefs
typedef long long ll;
typedef unsigned long long ull;
typedef unsigned short ushort;
typedef unsigned float ufloat;
typedef long double ld;
typedef vector<int> vi;
typedef vector<vector<int>> iMatrix;
typedef vector<vector<ll>> llMatrix;
typedef vector<vector<ld>> ldMatrix;
typedef pair<int, int> pii;

//constants
const ll pInf = ll(1e9);
const double PI = 3.14159265;

//enums
enum Derivative {sobelx, sobely, laplacian, prewitt, roberts};

//functions
cv::Mat readImage(string &filename);
void canny(cv::Mat &img);
cv::Mat nonMaximumSuppression(cv::Mat &img, cv::Mat &Ix, cv::Mat &Iy);
cv::Mat convert2Gray(cv::Mat &img);
cv::Mat customConvolution(cv::Mat &img, cv::Mat &kernel);
void printImg(cv::Mat &img);
void scaleMagImg(cv::Mat &img);
void cannyThreshold(cv::Mat &img, int lowerThreshold, int upperThreshold);
