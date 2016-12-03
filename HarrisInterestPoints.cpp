//#include <opencv2/core.hpp>
#include "Header.h"


int main(int argc, char** argv)
{
	freopen("input.txt", "r", stdin);
	freopen("output.txt", "w", stdout);
	std::setprecision(7);
	if (argc < 2)
	{
		cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
		waitKey(0); // Wait for a keystroke in the window
		system("pause");
		return -1;
	}
	//test();
	//	int num;
//	cin >> num;
//	readVideoFromWebCamera();
	readImage(g_filename);

//	readVideoWithSlidebar(argv[num]);
//	example2_4(argv[1]);
	system("pause");
	return 0;
}

void readImage(char *filename)
{
	int start = clock();
	cv::Mat img = cv::imread(filename, -1);
//	cv::Mat dst(img.size(), CV_32FC1);
	if (!img.data)
	{
		cout << "Couldn't open or find image\n";
		system("pause");
		return;
	}

	// Showing the result

	//buildinHarrisDetector(img);
	harrisEdgeDetector(img);

	int finish = clock();
	cout << finish - start << " miliseconds" << endl;

	cv::waitKey(0);
	cv::destroyAllWindows();
}

void readVideo(char *filename)
{
	cv::namedWindow("Example2", cv::WINDOW_AUTOSIZE);
	cv::VideoCapture capture;
	capture.open(string(filename));
	cv::Mat frame;
	while (1)
	{
		capture >> frame;
		if (!frame.data) break;
		cv::imshow("Example2", frame);
		char c = cv::waitKey(33);
		if (c >= 0) break;
	}
	cv::waitKey();
	cv::destroyWindow("Example2");
}

void onTrackbarSlide(int pos, void*)
{
	g_capture.set(cv::CAP_PROP_POS_FRAMES, pos);
	if (!g_dontset)
		g_run = 1;
	g_dontset = 1;

}

void readVideoWithSlidebar(char *filename)//with canny filters
{
	cv::namedWindow("Example3", cv::WINDOW_AUTOSIZE);
	g_capture.open(string(filename));// = cvCreateFileCapture(filename);

	int frames = (int) g_capture.get(cv::CAP_PROP_FRAME_COUNT);
	int tmpw = (int)g_capture.get(cv::CAP_PROP_FRAME_WIDTH);
	int tmph = (int)g_capture.get(cv::CAP_PROP_FRAME_WIDTH);
	cout << "Video has " << frames << " frames of dimensions("
		<< tmpw << ", " << tmph << ")." << endl;
	cv::createTrackbar("Position", "Example3", &g_slider_position, frames, onTrackbarSlide);

	cv::Mat frame;
	
	while (1)
	{
		if (g_run != 0)
		{
			g_capture >> frame;
			if (!frame.data) break;
			cv::Mat out;
			cv::cvtColor(frame, out, cv::COLOR_BGR2GRAY);
			cv::Canny(out, frame, 10, 100, 3, true);
			cv::imshow("Example3", frame);
			g_run -= 1;
		}
		char c = (char)cv::waitKey(10);
		if (c == 's')
		{
			g_run = 1; 
			cout << "Single step, run = " << g_run << endl;
		}

		if (c == 'r')
		{
			g_run = -1;
			cout << "Run mode, run = " << g_run << endl;
		}
		if (c == 27) break;
	}
	cvWaitKey();
	cvDestroyWindow("Example3");
}

void readVideoFromWebCamera()
{
	cv::VideoCapture cap;
	cap.open(0);
	if (!cap.isOpened())
	{
		std::cerr << "couldn't open capture" << endl;
	}
	cv::namedWindow("Example", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Example1", cv::WINDOW_AUTOSIZE);
	cv::Mat frame;
	while (1)
	{
		cap >> frame;
		cv::Vec3b p = frame.at<cv::Vec3b>(0, 0);
		if (!frame.data) break;
		cv::Mat img;
//		cv::GaussianBlur(frame, img, cv::Size(15, 15), 30);
//		cv::Mat grayImage(frame.size(), CV_8UC1);
//		cv::cvtColor(frame, grayImage, CV_RGB2GRAY);
		
		harrisEdgeDetector(frame);

		cv::imshow("Example", frame);
		int milliseconds = 40;
		if (waitKey(30) >= 0) break;
	}
	cv::destroyWindow("Example");
}

void harrisEdgeDetector(cv::Mat &image)
{
	cv::namedWindow("Example1", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Example3", cv::WINDOW_AUTOSIZE);

	cv::Mat grayImage(image.size(), CV_8UC1);
	cv::Mat grayImage1(image.size(), CV_8UC1);
	
	if (image.channels() == 3)
	{
		cv::cvtColor(image, grayImage, CV_BGR2GRAY);
	}
	else if (image.channels() == 1)
	{
		grayImage = image;
	}

	cv::GaussianBlur(grayImage, grayImage1, cv::Size(5, 5), 1.5, 1.5);

	pair<int, int> imsize = make_pair(image.size().height, image.size().width);

	cv::Mat L(image.size(), CV_8UC1);
	i_Matrix v_Gx(imsize.X, vector<int>(imsize.Y, 0));
	i_Matrix v_Gy(imsize.X, vector<int>(imsize.Y, 0));
	i_Matrix v_Gxy(imsize.X, vector<int>(imsize.Y, 0));

	cv::Mat Gx(image.size(), CV_32FC1);
	cv::Mat Gy(image.size(), CV_32FC1);
	cv::Mat Gxy(image.size(), CV_32FC1);
	//compute derivatives
	for (int i = 0;i < imsize.X; i++)
	{
		for (int j = 0;j < imsize.Y ; j++)
		{
			if (i == 0 || j == 0 || i == imsize.X - 1 || j == imsize.Y - 1)
			{
				Gx.at<float>(i, j) = 0;
				Gy.at<float>(i, j) = 0;
				Gxy.at<float>(i, j) = 0;
			}
			else
			{

				int a11 = grayImage.at<uchar>(i - 1, j - 1), a12 = grayImage.at<uchar>(i - 1, j), a13 = grayImage.at<uchar>(i - 1, j + 1);
				int a21 = grayImage.at<uchar>(i, j - 1), /*  a22    */  a23 = grayImage.at<uchar>(i, j + 1);
				int a31 = grayImage.at<uchar>(i + 1, j - 1), a32 = grayImage.at<uchar>(i + 1, j), a33 = grayImage.at<uchar>(i + 1, j + 1);

				// Convolution with horizontal differentiation kernel mask
				float v = ((a11 + a12 + a13) - (a31 + a32 + a33)) * 0.166666667f;

				// Convolution with vertical differentiation kernel mask
				float k = ((a11 + a21 + a31) - (a13 + a23 + a33)) * 0.166666667f;

				// Convolution with simple mask
				//			int k = grayImage1.at<uchar>(i + 1, j) - grayImage1.at<uchar>(i - 1, j);
				//			int v = grayImage1.at<uchar>(i, j + 1) - grayImage1.at<uchar>(i, j - 1);


				Gy.at<float>(i, j) = k * k;
				Gx.at<float>(i, j) = v * v;
				Gxy.at<float>(i, j) = k * v;
				if (k*k > 1500000 || v*v > 1500000 || k*v > 1500000)
				{
					cout << i << " " << j << " " << k * k << " " << v * v << endl;
				}
			}
		}
	}


	cv::Mat gx, gy, gxy;

	cv::GaussianBlur(Gx, gx, cv::Size(5, 5), 1.5, 1.5);
	cv::GaussianBlur(Gy, gy, cv::Size(5, 5), 1.5, 1.5);
	cv::GaussianBlur(Gxy, gxy, cv::Size(5, 5), 1.5, 1.5);
	ld_Matrix L1(imsize.X, vector<ld>(imsize.Y, 0));
	
	//compute matrix H^sp
	for (int i = 0;i < imsize.X; i++)
	{
		for (int j = 0;j < imsize.Y; j++)
		{
			long double m1 = Gx.at<float>(i, j);
			long double m2 = Gxy.at<float>(i, j);
			long double m3 = Gy.at<float>(i, j);

			L1[i][j] =abs (ld(m1)*ld(m3) - m2*ld(m2) - 0.04*ld(ld(m1 + m3)*ld(m1 + m3)));
		}
	}
	
	i_Matrix thresholdMatrix(L1.size(), vector<int>(L1[0].size(), 0));
	i_Matrix L2 = scale(L1);
	L =matrix2img(L2);
	cv::imshow("Example3", L);

	int maxL = 255;//findMax(L2);
	map<int, int> points;
	for (int i = 0;i < imsize.X; i++)
	{
		for (int j = 0;j < imsize.Y; j++)
		{
			if (L2[i][j] > float(maxL) * 0.3)
			{
				thresholdMatrix[i][ j] = L2[i][j];
				points[i * imsize.Y + j] = L2[i][j];
			}
		}
	}
	
//	i_Matrix x = regionalMax(thresholdMatrix);
//	i_Matrix x = simpleRegionalMax(bbb);
	int start = clock();
	printf("Amount points to fastRegionalMax: %d\n", points.size());
	set<int> y = fastRegionalMax(points, image.size());
	int finish = clock();
	printf("Time fastRegionalMax: %d\n", finish - start);
	printf("Amount interest points: %d\n", y.size());
	//for (int i = 0;i < imsize.X; i++)
	//{
	//	for (int j = 0;j < imsize.Y; j++)
	//	{
	//		if (x[i][j])
	//		{
	//			circle(image, Point(j, i), 5, Scalar(0, 0, 255), 2, 8, 0);
	//		}
	//	}
	//}

	for (set<int>::iterator it = y.begin(); it != y.end(); it++)
	{
		int index = (*it);
		int row = index / imsize.Y;
		int col = index % imsize.Y;
		circle(image, Point(col, row), 5, Scalar(0, 0, 255), 2, 8, 0);
	}

	cv::imshow("Example1", image);
}

//regionalMax in vicinity range*range
i_Matrix simpleRegionalMax(i_Matrix &a, int range)
{
	i_Matrix res(a.size(), vector<int>(a[0].size(), 0));
	int height = a.size();
	int width = a[0].size();

	for (int i = 0;i < height; i++)
	{
		for (int j = 0;j < width; j++)
		{
			bool b = false;
			bool b1 = false;
			for (int i1 = max(0, i - range); i1 <= min(height - 1, i + range); i1++)
			{
				for (int j1 = max(0, j - range); j1 <= min(width - 1, j + range); j1++)
				{
					if (a[i1][j1] > a[i][j])
					{
						b = true;
					}
					if (a[i1][j1] < a[i][j])
					{
						b1 = true;
					}
				}
			}
			if (!b && b1) res[i][j] = 1;
		}
	}

	return res;
}

//global regionalMax (as imregionalMax in MATLAB)
i_Matrix regionalMax(i_Matrix &a)
{
	i_Matrix res(a.size(), vector<int>(a[0].size(), 0));
	int height = a.size();
	int width = a[0].size();
	//uchar *data = a.data;
	vector<uchar> values(height * width);
	init_sets(height * width);
	

	vector<pair<int, int>> points = {mp(-1, -1), mp(-1, 0), mp(0, -1)};

	for (int i = 0;i < height; i++)
	{
		for (int j = 0;j < width; j++)
		{
			values[i * width + j] = a[i][j];
			for (int k = 0;k < points.size(); k++)
			{
				int newi = i + points[k].X;
				int newj = j + points[k].Y;
				if (newi < 0 || newj < 0) continue;
				int nv = a[i + points[k].X][ j + points[k].Y];
				if (a[i + points[k].X][ j + points[k].Y] == values[i * width + j])
				{
					union_sets(i * width + j, (i + points[k].X) * width + j + points[k].Y);
				}
			}
		}
	}

	vector<set<int>> neighbours(height * width);
	vector<int> vicinity_x = {	-1, -1, -1,
								0,      0,
								1,  1,  1};
	vector<int> vicinity_y = {	-1,  0,  1,
								-1,      1,
								-1,  0,  1};

	for (int i = 1;i < height - 1; i++)
	{
		for (int j = 1;j < width - 1; j++)
		{
			int index = i * width + j;
			int set_cur = find(index);
			for (int k = 0;k < vicinity_x.size(); k++)
			{
				int n_index = (i + vicinity_x[k]) * width + j + vicinity_y[k];
				int n_set = find(n_index);

				if (n_set != set_cur)
				{
					neighbours[set_cur].insert(n_set);
				}
			}
		}
	}
	vector<bool> isRegMax(height * width, 0);
	for (int i = 0;i < height * width; i++)
	{
		if (neighbours[i].size() != 0 && !isRegMax[i])
		{
			int prnt = find(i);
			bool b = true;
			for (auto it = neighbours[i].begin(); it != neighbours[i].end(); it++)
			{
				if (values[(*it)] > values[prnt])
				{
					b = false;
					break;
				}
			}
			if (b)
			{
				isRegMax[prnt] = true;
			}
		}
	}
	for (int i = 0;i < height; i++)
	{
		for (int j = 0;j < width; j++)
		{
			int index = i * width + j;
			int prnt = find(index);
			if (isRegMax[prnt])
			{
				res[i][j] = 1;
			}
		}
	}
//	for (int i = 1;i < a.size().height - 1; i++)
//	{
//		for (int j = 1;j < a.size().width - 1; j++)
//		{
//			bool b = true;
//			bool less = false;
//			bool more = false;
//			for (int i1 = i - 1; i1 <= i + 1; i1++)
//			{
//				for (int j1 = j - 1; j1 <= j + 1; j1++)
//				{
//					
//					int ij = a.at<uchar>(i, j);
//					int i1j1 = a.at<uchar>(i1, j1);
//					if (a.at<uchar>(i, j) > a.at<uchar>(i1, j1))
//					{
//						more = true;
////						break;
//					}
//					if (a.at<uchar>(i, j) < a.at<uchar>(i1, j1))
//					{
//						less = true;
//					}
//					//if (a.at<uchar>(i, j) == a.at<uchar>(i1, j1) && res[i1][j1])
//					//{
//					//	b = true;
//					//	break;
//					//}
//				}
////				if (more) break;
//			}
//			if (!more && less) res[i][j] = 1;
//		}
//	}
	return res;
}

set<int> fastRegionalMax(map<int, int> &_a, cv::Size &sz)
{
	set<int> res;

	map<int, int> map_parent;
	vector<uchar> values(_a.size(), 0);
	init_sets(_a.size());

	int index = 0;
	for (map<int, int>::iterator it = _a.begin(); it != _a.end(); it++)
	{
		pair<int, int> cur_it = (*it);
		map_parent[cur_it.X] = index++;
	}


	vector<pair<int, int>> points = { mp(-1, -1), mp(-1, 0), mp(0, -1) };
	index = 0;
	for (map<int, int>::iterator it = _a.begin(); it != _a.end(); it++)
	{
		pair<int, int> cur_it = (*it);
		values[index++] = cur_it.Y;
		int row = cur_it.X / sz.width;
		int col = cur_it.X % sz.width;
		for (int i = 0;i < points.size(); i++)
		{
			int newi = row + points[i].X;
			int newj = col + points[i].Y;
//			if (newi < 0 || newj < 0) continue;
			map<int, int>::iterator new_val = _a.find(newi*sz.width + newj);
			if (new_val == _a.end()) continue;
			int nv = new_val->Y;
			int curIndex = map_parent[cur_it.X];
			if (nv == cur_it.Y)
			{
				union_sets(curIndex, map_parent[new_val->X]);
			}
		}
	}

	vector<set<int>> neighbours(_a.size());
	vector<int> vicinity_x = { -1, -1, -1,
								0,      0,
								1,  1,  1 };
	vector<int> vicinity_y = { -1,  0,  1,
								-1,      1,
								-1,  0,  1 };
	index = 0;
	for (map<int, int>::iterator it = _a.begin(); it != _a.end(); it++)
	{
		pair<int, int> cur_it = (*it);
		int row = cur_it.X / sz.width;
		int col = cur_it.X % sz.width;
		int cur_set = find(map_parent[cur_it.X]);

		for (int i = 0; i < vicinity_x.size(); i++)
		{
			for (int j = 0;j < vicinity_y.size(); j++)
			{
				int n_index = (row + vicinity_x[i]) * sz.width + col + vicinity_y[j];
				map<int, int>::iterator new_it = map_parent.find(n_index);
				if (new_it == map_parent.end()) continue;
				int n_set = find((*new_it).Y);
				if (n_set != cur_set)
				{
					neighbours[cur_set].insert(n_set);
				}
			}
		}
	//	index++;
	}
	set<int> regMax;
	index = 0;
	for (map<int, int>::iterator it1 = _a.begin(); it1 != _a.end(); it1++)
	{
		pair<int, int> cur_it = (*it1);
		index = map_parent[cur_it.X];
		int prnt = find(index);

		if (neighbours[index].size() != 0)
		{
			bool b = true;
			for (auto it = neighbours[index].begin(); it != neighbours[index].end(); it++)
			{
				if (values[(*it)] > values[prnt])
				{
					b = false;
					break;
				}
			}
			if (b)
			{
				res.insert((*it1).X);
			}
		}
		else
		{
			res.insert((*it1).X);
		}
		index++;
	}
	return res;
}

//functions for dsu
int find(int x)
{
	if (x == parent[x]) return x;
	return parent[x] = find(parent[x]);
}

bool union_sets(int x, int y)
{
	x = find(x);
	y = find(y);
	if (x == y) return false;
	if (sz[x] > sz[y]) swap(x, y);
	parent[y] = x;
	if (sz[x] == sz[y]) sz[y]++;
	return true;
}

void init_sets(int n)
{
	parent.resize(n);
	sz.assign(n, 0);
	for (int i = 0;i < n; i++)
	{
		parent[i] = i;
	}
}

//function to test regionalMax
void test()
{
	freopen("input.txt", "r", stdin);
	freopen("output.txt", "w", stdout);
	int n, m;
	scanf("%d %d", &n, &m);
	cv::Mat testim(cv::Size(m, n), CV_8UC1);

	for (int i = 0;i < n; i++)
	{
		for (int j =0 ;j < m; j++)
		{
			int v;
			scanf("%d", &v);
			testim.at<uchar>(i, j) = uchar(v);
		}
	}
//	cout << testim << endl<<endl;
	for (int i = 0;i < n; i++)
	{
		for (int j = 0;j < m; j++)
		{
			printf("%d ", int(testim.at<uchar>((i)%n, (j)%m)));
		}
		printf("\n");
	}
	printf("\n\n\n");
	//Matrix res = regionalMax(testim);
	//for (int i = 0;i < res.size(); i++)
	//{
	//	for (int j = 0;j < res[i]. size(); j++)
	//	{
	//		printf ("%d ", res[i][j]);
	//	}
	//	printf("\n");
	//}
}

void print_matrix(i_Matrix &a)
{
	for (int i = 0;i < a.size(); i++)
	{
		for (int j = 0;j < a[i].size(); j++)
		{
			cout << setw(7) << a[i][j];
		}
		cout << endl;
	}
}

i_Matrix scale(ld_Matrix &a, int minv, int maxv)
{
	assert(minv < maxv && "Error! min value > max value");
	long double maxA = 0;
	long double minA = pInf;
	i_Matrix res(a.size(), vector<int>(a[0].size(), 0));
	for (int i = 0;i < a.size(); i++)
	{
		for (int j = 0;j < a[i].size(); j++)
		{
			if (a[i][j] > maxA) maxA = a[i][j];
			if (a[i][j] < minA) minA = a[i][j];
			if (abs(a[i][j]) > 1e9)
			{
				int y = 0;
			}
		}
	}
	cout << "MaxA = " << maxA << endl;
	for (int i = 0; i < a.size(); i++)
	{
		for (int j = 0; j < a[i].size(); j++)
		{
			if (a[i][j]>1e8)
			{
				int y = 0;
			}
			res[i][j] = minv +  (ld(a[i][j] - minA) / ld(maxA - minA)) * ld(maxv - minv);
		}
	}
	return res;
}

cv::Mat matrix2img(i_Matrix &a)
{
	cv::Mat res(cv::Size( a[0].size(), a.size()), CV_8UC1);

	for (int i = 0;i < a.size(); i++)
	{
		for (int j = 0;j < a[0].size(); j++)
		{
			res.at<uchar>(i, j) = a[i][j];
		}
	}
	return res;
}

i_Matrix img2matrix(cv::Mat &a)
{
	assert(a.channels() == 1 && "You try convert 3-channels image to matrix");
	i_Matrix res(a.rows, vector<int>(a.cols, 0));
	for (int i = 0;i < a.rows; i++)
	{
		for (int j = 0;j < a.cols; j++)
		{
			res[i][j] = (int)a.at<float>(i, j);
		}
	}
	return res;
}

int findMax(i_Matrix &a)
{
	int res = -pInf;
	for (int i = 0;i < a.size(); i++)
	{
		for (int j = 0;j < a[i].size(); j++)
		{
			if (res < a[i][j]) res = a[i][j];
		}
	}
	return res;
}

void buildinHarrisDetector(cv::Mat &img)
{
	cv::namedWindow("corners_window", CV_WINDOW_AUTOSIZE);

	cv::Mat grayImage(img.size(), CV_8UC1);
	cv::Mat image = img.clone();
	if (img.channels() == 3)
	{
		cv::cvtColor(img, grayImage, CV_RGB2GRAY);
	}
	else if (img.channels() == 1)
	{
		grayImage = img.clone();
	}
	
	cv::Mat dst, dst_norm, dst_norm_scaled;

	cv::cornerHarris(grayImage, dst, 5, 5, 0.04, BORDER_DEFAULT);
	
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);
	int thresh = 85;

	// Drawing a circle around corners
	for (int j = 0; j < image.rows; j++)
	{
		for (int i = 0; i < image.cols; i++)
		{
			if ((int)dst_norm.at<float>(j, i) > thresh)
			{
				int ich = image.channels();
				if (image.channels() == 1)
				{
					image.at<uchar>(i, j) = uchar(255);//cv::Vec3b(0, 0, 255);
				}
				else if (image.channels() == 3)
				{
					try
					{
						circle(image, Point(i, j), 5, Scalar(0, 0, 255), 2, 8, 0);
					}
					catch(cv::Exception &ex)
					{
						cout << ex.err << endl;
					}
				}
			}
		}
	}
	
	imshow("corners_window", image);
}
