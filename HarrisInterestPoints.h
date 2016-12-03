#pragma once
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
//#include <opencv\cv.h>

#include <iostream>
#include <fstream>
#include <ctime>
#include <iomanip>

#include <vector>
#include <set>

using namespace cv;
using namespace std;

#define X first
#define Y second
#define mp make_pair


typedef unsigned short ushort;
typedef unsigned float ufloat;
typedef long long ll;
typedef long double ld;
typedef vector<vector<int>> i_Matrix;
typedef vector<vector<ld>> ld_Matrix;

const ll pInf = ll(1e9);

void readImage(char *filename);
void readVideo(char *filename);
void readVideoWithSlidebar(char *filename);
void readVideoFromWebCamera();

void onTrackbarSlide(int pos, void*);

void harrisEdgeDetector(cv::Mat &image);
void buildinHarrisDetector(cv::Mat &img);
i_Matrix regionalMax(i_Matrix &a);
i_Matrix simpleRegionalMax(i_Matrix &a, int range = 2);
set<int> fastRegionalMax(map<int, int> &_a, cv::Size &sz);

i_Matrix scale(ld_Matrix &a, int minv = 0, int maxv = 255);
cv::Mat matrix2img(i_Matrix &a);
i_Matrix img2matrix(cv::Mat &a);
int findMax(i_Matrix &a);

void init_sets(int n);
int find(int x);
bool union_sets(int x, int y);

void test();

void print_matrix(i_Matrix &a);


int					g_slider_position = 0;
int					g_run = 1, g_dontset = 0;
cv::VideoCapture	g_capture;

vector<int> parent, sz;

char *g_filename = "image_data/1.jpg";
