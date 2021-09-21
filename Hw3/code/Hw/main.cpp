#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <stdio.h>

using namespace cv;
using namespace std;


int cnt = 0;
string file = "../../result/";

// ��ͼ��չ�����ھ��
void expand(const Mat& raw, Mat& img_expand)
{
	Mat tmp = img_expand(Range(1, raw.size().height + 1), Range(1, raw.size().width + 1));
	raw.copyTo(tmp);

	tmp = img_expand(Range(0, 1), Range(1, raw.size().width + 1));
	raw.row(0).copyTo(tmp);
	tmp = img_expand(Range(raw.size().height + 1, raw.size().height + 2), Range(1, raw.size().width + 1));
	raw.row(raw.size().height - 1).copyTo(tmp);
	tmp = img_expand(Range(1, raw.size().height + 1), Range(0, 1));
	raw.col(0).copyTo(tmp);
	tmp = img_expand(Range(1, raw.size().height + 1), Range(raw.size().width + 1, raw.size().width + 2));
	raw.col(raw.size().width - 1).copyTo(tmp);

	cvtColor(img_expand, img_expand, COLOR_BGR2GRAY);
}

// �����ݶ�(I)
void grad(Mat& Ix, Mat& Iy, Mat& img_expand)
{
	float acc_dx = 0, acc_dy = 0;         //accumulators
	float k1[] = { -1,0,1,-2,0,2,-1,0,1 };   //sobel kernal dx
	float k2[] = { -1,-2,-1,0,0,0,1,2,1 };    //sobel kernal dy

	for (int i = 0; i < Ix.rows; i++) {
		for (int j = 0; j < Ix.cols; j++) {
			acc_dx = acc_dy = 0;

			//apply kernel/mask
			for (int nn = 0; nn < 3; nn++) {
				for (int mm = 0; mm < 3; mm++) {
					
					acc_dx += img_expand.at<float>(i + nn, j + mm) * k1[(mm * 3) + nn];
					acc_dy += img_expand.at<float>(i + nn, j + mm) * k2[(mm * 3) + nn];

				}
			}
			//write final values
			Ix.at<float>(i, j) = acc_dx;
			Iy.at<float>(i, j) = acc_dy;
		}
	}

}


// ����M����
void getM(Mat& Ix, Mat& Iy, Mat& M, int row, int col, int windowSize)
{
	for (int i = row - windowSize / 2; i < row + windowSize / 2 + 1; ++i)
	{
		for (int j = col - windowSize / 2; j < col + windowSize / 2 + 1; ++j)
		{
			M.at<float>(0, 0) += Ix.at<float>(i, j) * Ix.at<float>(i, j);
			M.at<float>(1, 0) += Ix.at<float>(i, j) * Iy.at<float>(i, j);
			M.at<float>(0, 1) += Ix.at<float>(i, j) * Iy.at<float>(i, j);
			M.at<float>(1, 1) += Iy.at<float>(i, j) * Iy.at<float>(i, j);
		}
	}
}

// ������������
void getLambda(Mat& M, float& lambda1, float& lambda2)
{
	float a = 1.0f, b = -trace(M).val[0], c = determinant(M);	
	// lambda1 > lambda2
	lambda1 = 0.5f / a * (-b + sqrt(b * b - 4 * a * c));
	lambda2 = 0.5f / a * (-b - sqrt(b * b - 4 * a * c));
}

// ��һ����0-255
void norm(Mat& M)
{
	double minVal = 0.0, maxVal = 255.0;
	cv::minMaxIdx(M, &minVal, &maxVal);
	M = (M - minVal) / (maxVal - minVal);
}

// ����Rͼ
void getR(Mat& img_expand, Mat& R, const Size& R_Size, float K, int windowSize, double threshold)
{
	Mat Min_lambda(R_Size, CV_32F);				
	Mat Max_lambda(R_Size, CV_32F);				
	float lambda1, lambda2;							
	Mat Ix(img_expand.size().height - 2, img_expand.size().width - 2, CV_32F);	// x������ݶȣ�����Ix[0,0]��img_expand[1,1]���ݶ�
	Mat Iy(img_expand.size().height - 2, img_expand.size().width - 2, CV_32F);	// y������ݶ�
	grad(Ix, Iy, img_expand);	

	for (int i = windowSize / 2, k = 0; k < R_Size.height; ++i, ++k)
	{
		for (int j = windowSize / 2, l = 0; l < R_Size.width; ++j, ++l)
		{
			Mat M(2, 2, CV_32F, cv::Scalar(0.0));
			getM(Ix, Iy, M, i, j, windowSize);	
			getLambda(M, lambda1, lambda2);							
			Max_lambda.at<float>(k, l) = sqrt(sqrt(lambda1));
			Min_lambda.at<float>(k, l) = sqrt(sqrt(lambda2));

			R.at<float>(k, l) = sqrt(sqrt(lambda1 * lambda2 - K * (lambda1 + lambda2) * (lambda1 + lambda2)));	 
		}
	}

	norm(Min_lambda);
	string header = to_string(cnt);
	imshow(header + "_Min_lambda", Min_lambda);				
	imwrite(file + header + "_Min_lambda.jpg", Min_lambda * 255);	// imshow��ȡ�ľ������ݷ�ΧΪ[0,1]����imwrite��ȡ�����ݷ�ΧΪ[0,255]

	norm(Max_lambda);
	imshow(header + "_Max_lambda", Max_lambda);
	imwrite(file + header + "_Max_lambda.jpg", Max_lambda * 255);

	//cv::cvtColor(R, R, cv::COLOR_GRAY2BGR);
	cv::threshold(R, R, 0, 0, cv::THRESH_TOZERO);		// ���R<0, ʹ��R=0��ʹflat��edge��Rֵ��ΪС������ֻ��corner�Ǵ������������Ļ�Ч�����ƣ�ֻ������ֵ����ȷ�����ұ���(flat)����
	norm(R);
	normalize(R, R, 0, 1, NORM_MINMAX);
	imshow(header + "_R", R);
	imwrite(file + header + "_R.jpg", R * 255);
}


// �Ǽ���ֵ����
void NMS(Mat& R, double threshold)
{
	double maxVal = 0;
	int windowSize = 30; 
	for (int i = 0; i < R.rows - windowSize; i += windowSize)
	{
		for (int j = 0; j < R.cols - windowSize; j += windowSize)
		{
			maxVal = 0;
			for (int m = i; m < i + windowSize; m++)
			{
				for (int n = j; n < j + windowSize; n++)
				{
					if (R.at<float>(m, n) > maxVal && R.at<float>(m, n) > threshold)
					{
						maxVal = R.at<float>(m, n);
					}
					else {
						R.at<float>(m, n) = 0;
					}
				}
			}
		}
	}
	// border 
	for (int k = 0; k < R.cols - windowSize; k += windowSize)
	{
		maxVal = 0;
		for (int i = R.rows - windowSize; i < R.rows; i++)
		{
			for (int j = k; j < k + windowSize; j++)
			{
				if (R.at<float>(i, j) > maxVal  && R.at<float>(i, j) > threshold)
				{
					maxVal = R.at<float>(i, j);
				}
				else {
					R.at<float>(i, j) = 0;
				}
			}
		}
	}
	for (int k = 0; k < R.rows - windowSize; k += windowSize)
	{
		maxVal = 0;
		for (int i = k; i < k + windowSize; i++)
		{
			for (int j = R.cols - windowSize; j < R.cols; j++)
			{
				if (R.at<float>(i, j) > maxVal  && R.at<float>(i, j) > threshold)
				{
					maxVal = R.at<float>(i, j);
				}
				else {
					R.at<float>(i, j) = 0;
				}
			}
		}
	}
	maxVal = 0;
	for (int i = R.rows - windowSize; i < R.rows; i++)
	{
		for (int j = R.cols - windowSize; j < R.cols; j++)
		{
			if (R.at<float>(i, j) > maxVal && R.at<float>(i, j) > threshold)
			{
				maxVal = R.at<float>(i, j);
			}
			else {
				R.at<float>(i, j) = 0;
			}
		}
	}
}


// ��ʾ�ǵ�
void draw_corner(Mat& R, Mat& raw, int windowSize)
{
	for (int i = windowSize / 2, m = 0; m < R.rows; i++, m++)
	{
		for (int j = windowSize / 2, n = 0; n < R.cols; j++, n++)
		{
			if (R.at<float>(m, n))	
			{
				circle(raw, Point(j, i), 2, CV_RGB(255, 0, 0));
			}
		}
	}
}


//������ɫRͼ
void getColorR(Mat& src, Mat& res)
{
	int row = src.size().height;
	int col = src.size().width;
	float tmp = 0;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			tmp = src.at<float>(i, j) * 255;
			res.at<cv::Vec3f>(i, j)[0] = tmp < 25 ? tmp : 0;
			res.at<cv::Vec3f>(i, j)[1] = tmp > 20 ? tmp - 20 : 0;
			res.at<cv::Vec3f>(i, j)[2] = tmp > 50 ? tmp : 0;
		}
	}
}

void HarrisCorner(Mat& raw)
{
	cnt++;
	float k = 0.04;			
	int windowSize = 3;		
	double threshold = sqrt(sqrt(0.28 / 255));			

	Mat img_expand(raw.rows + 2, raw.cols + 2, CV_8UC3, Scalar(0, 0, 0));
	Size R_Size(raw.cols - windowSize / 2 * 2, raw.rows - windowSize / 2 * 2);
	Mat R(R_Size, CV_32F);			

	expand(raw, img_expand);		
	img_expand.convertTo(img_expand, CV_32F);		

	// ��˹�˲�
	GaussianBlur(img_expand, img_expand, Size(5, 5), 1, 1);

	// ����Rֵ							
	getR(img_expand, R, R_Size, k, windowSize, threshold);

	// �����ɫRͼ
	Mat R1(R_Size, CV_32FC3, CV_RGB(0,0,0));

	string header = to_string(cnt);

	getColorR(R, R1);
	imshow(header + "_ColorR", R1);
	imwrite(file + header + "_ColorR.jpg", R1 * 255);

	// �Ǽ���ֵ���ƣ��õ�ѡȡ��Ľǵ�
	NMS(R, threshold);

	draw_corner(R, raw, windowSize);		
	imshow(header + "_result", raw);				
	imwrite(file + header + "_result.jpg", raw);
	waitKey();
}

int main()
{
	// ��ȡ����ͷ  
	VideoCapture capture(0);
	int delay = 30;
	while (true)
	{
		Mat frame;
		capture >> frame;
		int key = waitKey(delay);
		imshow("Camera", frame);
		if (delay >= 0 && key == 32)
		{
			cout << "Harris Corner Detecting..." << endl;
			HarrisCorner(frame);
			cout << "Done!" << endl;
			waitKey(0);
		}
		if (key == 27)
			break;
			
	}
	return 0;



	// test
	//string file = "input.jpg";
	//Mat raw = imread(file);		// ����ͼƬ
	//HarrisCorner(raw);
	//return 0;

}