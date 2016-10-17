
#include "canny.h"

int main()
{
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
	  double start = clock();
    string fname;
    cin >> fname;

    cv::Mat img = readImage(fname);

	canny(img);
	double finish = clock();
	cout << endl << finish - start << endl;
	cv::waitKey();
	return 0;
}


cv::Mat readImage(string &filename)
{
    cv::Mat img = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    return img;
}


cv::Mat customConvolution(cv::Mat &img, cv::Mat &kernel)
{
	cv::Mat convolveImg(img.size(), CV_64FC1);
    for (int i = 0;i < img.size().height; i++)
    {
        for (int j = 0;j < img.size().width; j++)
        {
			convolveImg.at<double>(i, j) = 0;
            for (int k = -kernel.size().height / 2; k <= kernel.size().height / 2; k++)
            {
                for (int l = -kernel.size().width / 2; l <= kernel.size().width / 2; l++)
                {
                    int newi = i +k , newj = j + l;
                    if (i + k < 0) newi = 0;
                    if (i + k >= img.size().height) newi = img.size().height - 1;
					if (j + l < 0) newj = 0;
					if (j + l >= img.size().width) newj = img.size().width - 1;
					
					double dImageValue = 0;
					if (img.type() == CV_8UC1)
					{
						dImageValue = img.at<uchar>(newi, newj);
					}
					else if (img.type() == CV_64FC1)
					{
						dImageValue = img.at<double>(newi, newj);
					}
					convolveImg.at<double>(i, j) += kernel.at<double>(k + kernel.size().height / 2, l + kernel.size().width / 2) * dImageValue;
                }
            }
			convolveImg.at<double>(i, j) /= kernel.size().height * kernel.size().width;

        }
    }
	return 	convolveImg;
}


void canny(cv::Mat &img)
{
	cv::namedWindow("output1", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("output2", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("output3", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("output4", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("output5", cv::WINDOW_AUTOSIZE);

	cv::Mat grayImg;
	if (img.channels() == 3)
	{
		cvtColor(img, grayImg, CV_BGR2GRAY);
	}
	else
	{
		img.copyTo(grayImg);
	}
	grayImg.convertTo(grayImg, CV_64FC1);

	double sigma = 1.0;
	cv::Mat verticalGaussianKernel = getGaussianKernel(7, sigma);
	cv::Mat horizontalGaussianKernel(verticalGaussianKernel.size(), CV_64FC1);
	cv::transpose(verticalGaussianKernel, horizontalGaussianKernel);
	cv::Mat GyFirst(cv::Size(1, 3), CV_64FC1);
	cv::Mat GySecond(cv::Size(3, 1), CV_64FC1);
	cv::Mat GxFirst(cv::Size(1, 3), CV_64FC1);
	cv::Mat GxSecond(cv::Size(3, 1), CV_64FC1);

#pragma region init derivative operators
	GyFirst.at<double>(0, 0) = 1.0;
	GyFirst.at<double>(1, 0) = 2.0;
	GyFirst.at<double>(2, 0) = 1.0;

	GySecond.at<double>(0, 0) = -1.0;
	GySecond.at<double>(0, 1) = 0.0;
	GySecond.at<double>(0, 2) = 1.0;
	cv::transpose(GyFirst, GxFirst);
	cv::transpose(GySecond, GxSecond);
#pragma endregion

#pragma region prints
	printImg(verticalGaussianKernel);
	printImg(horizontalGaussianKernel);
	printImg(GxFirst);
	printImg(GxSecond);
	printImg(GyFirst);
	printImg(GySecond);
#pragma endregion

	cv::Mat blurImg(img.size(), CV_64FC1);
	cv::Mat Ix(img.size(), CV_64FC1);
	cv::Mat Iy(img.size(), CV_64FC1);

//  ---------using my custom convolution---------------
//	blurImg = customConvolution(grayImg, verticalGaussianKernel);
//	blurImg = customConvolution(blurImg, horizontalGaussianKernel);

//  ---------using OpenCV convolution------------------
	cv::filter2D(grayImg, blurImg, CV_64FC1, verticalGaussianKernel);
	cv::filter2D(blurImg, blurImg, CV_64FC1, horizontalGaussianKernel);
	
	scaleMagImg(blurImg);
	blurImg.convertTo(blurImg, CV_8UC1);
	cv::imshow("output1", blurImg);
//	blurImg.convertTo(blurImg, CV_64FC1);

//  ---------using my custom convolution---------------
//	Iy = customConvolution(blurImg, GyFirst);
//	Iy = customConvolution(Iy, GySecond);
//	Ix = customConvolution(blurImg, GxFirst);
//	Ix = customConvolution(Ix, GxSecond);

//  ---------using OpenCV convolution------------------
	cv::filter2D(blurImg, Iy, CV_64FC1, GyFirst);
	cv::filter2D(Iy, Iy, CV_64FC1, GySecond);
	cv::filter2D(blurImg, Ix, CV_64FC1, GxFirst);
	cv::filter2D(Ix, Ix, CV_64FC1, GxSecond);

	scaleMagImg(Iy);
	scaleMagImg(Ix);

	Ix.convertTo(Ix, CV_8UC1);
	Iy.convertTo(Iy, CV_8UC1);
	cv::imshow("output2", Ix);
	cv::imshow("output3", Iy);

	cv::Mat magImg(img.size(), CV_64FC1);

	for (int i = 0;i < img.rows; i++)
	{
		for (int j = 0;j < img.cols; j++)
		{
			magImg.at<double>(i, j) = sqrt(double(double(Ix.at<uchar>(i, j)) * Ix.at<uchar>(i, j) + double(Iy.at<uchar>(i, j)) * Iy.at<uchar>(i, j)));
		}
	}
	magImg = nonMaximumSuppression(magImg, Ix, Iy);
	scaleMagImg(magImg);
	cv::imshow("output4", magImg);
	cannyThreshold(magImg, 50, 175);
	cv::imshow("output5", magImg);

	int y = 0;
}


void printImg(cv::Mat &img)
{
	cout << "---------//------\n";
	if (img.empty())
	{
		cout << "Empty Image\n";
		return;
	}

	for (int i = 0; i < img.size().height; i++)
	{
		for (int j = 0; j < img.size().width; j++)
		{
			if (img.type() == CV_8UC1)
			{
				cout << int(img.at<uchar>(i, j)) << " ";
			}
			else if (img.type() == CV_64FC1)
			{
				cout << img.at<double>(i, j) << " ";

			}
		}
		cout << endl;
	}
	cout << "---------//------\n";
}


void scaleMagImg(cv::Mat &img)
{
	cv::Mat scaledImg(img.size(), CV_64FC1);
	double minv = pInf;
	double maxv = -pInf;
	for (int i=  0;i < img.size().height; i++)
	{
		for (int j = 0;j < img.size().width; j++)
		{
			if (minv > img.at<double>(i, j))
			{
				minv = img.at<double>(i, j);
			}
			if (maxv < img.at<double>(i, j))
			{
				maxv = img.at<double>(i, j);
			}
		}
	}
	for (int i = 0;i < img.size().height; i++)
	{
		for (int j = 0;j < img.size().width; j++)
		{
			img.at<double>(i, j) = 255.0* (img.at<double>(i, j) - minv) / (maxv - minv);
		}
	}
	cout << minv << " " << maxv << endl;
}


cv::Mat nonMaximumSuppression(cv::Mat &img, cv::Mat &Ix, cv::Mat &Iy)
{
	cv::Mat res(img.size(), CV_64FC1);
	cv::Mat mag(img.size(), CV_64FC1);
	for (int i = 1;i < img.rows - 1; i++)
	{
		for (int j = 1;j < img.cols - 1; j++)
		{
			vector<pii> angels;
			for (int i = 0; i < 360; i += 45)
			{
				angels.pb(mp(0, i));
			}
			bool b = true;
			double ix = Ix.at<uchar>(i, j);
			double iy = Iy.at<uchar>(i, j);
			double tang = atan2(iy, ix) * 180.0 / PI;
			for (int k = 0; k < angels.size(); k++)
			{
				angels[k].X = abs(tang - k * 45);
			}
			sort(angels.begin(), angels.end());
			int gradientAngle = angels[0].Y;
			double diff = 0;
			double prev1, next1;
			double cur = img.at<double>(i, j);
			switch (gradientAngle)
			{
			case 0:
				prev1 = img.at<double>(i, j - 1);
				next1 = img.at<double>(i, j + 1);
				diff = img.at<double>(i, j + 1) - img.at<double>(i, j - 1);
				break;
			case 45:
				prev1 = img.at<double>(i + 1, j - 1);
				next1 = img.at<double>(i - 1, j + 1);
				diff = img.at<double>(i - 1, j + 1) - img.at<double>(i + 1, j - 1);
				break;
			case 90:
				prev1 = img.at<double>(i + 1, j);
				next1 = img.at<double>(i - 1, j);
				diff = img.at<double>(i - 1, j ) - img.at<double>(i + 1, j);
				break;
			case 135:
				prev1 = img.at<double>(i + 1, j + 1);
				next1 = img.at<double>(i - 1, j - 1);
				diff = img.at<double>(i - 1, j - 1) - img.at<double>(i + 1, j + 1);
				break;
			case 180:
				prev1 = img.at<double>(i - 1, j);
				next1 = img.at<double>(i + 1, j);
				diff = img.at<double>(i + 1, j) - img.at<double>(i - 1, j);
				break;
			case 225:
				prev1 = img.at<double>(i - 1, j + 1);
				next1 = img.at<double>(i + 1, j - 1);
				diff = img.at<double>(i + 1, j - 1) - img.at<double>(i - 1, j + 1);
				break;
			case 270:
				prev1 = img.at<double>(i - 1, j);
				next1 = img.at<double>(i + 1, j);
				diff = img.at<double>(i + 1, j) - img.at<double>(i - 1, j);
				break;
			case 315:
				prev1 = img.at<double>(i - 1, j - 1);
				next1 = img.at<double>(i + 1, j + 1);
				break;
			}
			if(cur > max(prev1, next1)) res.at<double>(i, j) = img.at<double>(i, j);
			else res.at<double>(i, j) = 0;
		}
	}
	return res;
}


void cannyThreshold(cv::Mat &img, int lowerThreshold, int upperThreshold)
{
	iMatrix typePixels(img.rows, vi(img.cols, 0));
	cv::Mat res(img.size(), CV_64FC1);

	for (int i = 0;i < img.rows; i++)
	{
		for (int j = 0;j < img.cols; j++)
		{
			if (img.at<double>(i, j) > upperThreshold)
			{
				typePixels[i][j] = 1;
			}
		}
	}

	for (int i = 1; i < img.rows; i++)
	{
		for (int j = 1; j < img.cols; j++)
		{
			if (img.at<double>(i, j) <= upperThreshold && img.at<double>(i, j) >= lowerThreshold && (typePixels[i - 1][j] || typePixels[i][j - 1] || typePixels[i - 1][j - 1]))
			{
				typePixels[i][j] = 1;
			}
		}
	}
	for (int i = 0;i < img.rows; i++)
	{
		for (int j = 0;j < img.cols; j++)
		{
			if (!typePixels[i][j])
			{
				img.at<double>(i, j) = 0;
			}
		}
	}

}


cv::Mat convert2Gray(cv::Mat &img)
{
	cv::Mat greyMat(img.size(), CV_64FC1);

	if (img.channels() == 3)
	{
		cv::cvtColor(img, greyMat, CV_BGR2GRAY);
		return greyMat;
	}
	else
	{
		return img;
	}

}
