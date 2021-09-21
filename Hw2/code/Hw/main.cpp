#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <iostream>
#include <vector>
#include <string>


using namespace cv;
using namespace std;

// Edge extraction using Sobel kernal
void edgeExtract(Mat img, Mat& mag, Mat& dist)
{
    float acc_dx = 0, acc_dy = 0;           // accumulators
    float k1[] = { -1,-2,-1,0,0,0,1,2,1 };  // {-2,-4,-2,0,0,0,2,4,2};    // sobel kernal dx
    float k2[] = { -1,0,1,-2,0,2,-1,0,1 };  // {-2,0,2,-4,0,4,-2,0,2};    // sobel kernal dy

    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++) 
        {
            acc_dx = acc_dy = 0;
            for (int n = -1; n < 2; n++) 
            {
                for (int m = -1; m < 2; m++) {
                    if (i + n > 0 && i + n < img.rows && j + m > 0 && j + m < img.cols) {
                        acc_dx += (float)img.at<uchar>(i + n, j + m) * k1[((m + 1) * 3) + n + 1];
                        acc_dy += (float)img.at<uchar>(i + n, j + m) * k2[((m + 1) * 3) + n + 1];
                    }
                }
            }
            // generate a binary image with edge extracted.
            // thres = 100
            mag.at<float>(i, j) = (sqrtf(acc_dy * acc_dy + acc_dx * acc_dx)) > 100 ? 255 : 0;
            dist.at<float>(i, j) = atan2f(acc_dy, acc_dx);
        }
    }
}

// Check if the computed circle center is inside the image
// If true, increase the accumulator.
void inc_if_inside(double*** accu, int x, int y, int height, int width, int r)
{
    if (x > 0 && x < width && y> 0 && y < height)
        accu[y][x][r]++;
}

// Hough transform to detect circles
void houghCircles(Mat& img_binary, Mat& dist, double threshold, int minRadius, int maxRadius, double distance, Mat& h_acc, Mat& raw) 
{
    int radiusRange = maxRadius - minRadius;

    // accumulator accu[HEIGHT][WIDTH][DEPTH]
    double*** accu;

    accu = new double** [img_binary.rows];
    for (int i = 0; i < img_binary.rows; ++i) 
    {
        accu[i] = new double* [img_binary.cols];

        for (int j = 0; j < img_binary.cols; ++j)
            accu[i][j] = new double[radiusRange];
    }
    for (int i = 0; i < img_binary.rows; ++i) 
    {
        for (int j = 0; j < img_binary.cols; ++j) 
        {
            for (int k = 0; k < radiusRange; k++)
                accu[i][j][k] = 0;
        }
    }
    
    // Go through the binary image
    for (int y = 0; y < img_binary.rows; y++)
    {
        for (int x = 0; x < img_binary.cols; x++)
        {
            if ((float)img_binary.at<float>(y, x) > 250.0)  //threshold image  
            {
                for (int r = minRadius; r < radiusRange; r++)
                {

                    int x0 = cvRound(x + r * cos(dist.at<float>(y, x)));
                    int x1 = cvRound(x - r * cos(dist.at<float>(y, x)));
                    int y0 = cvRound(y + r * sin(dist.at<float>(y, x)));
                    int y1 = cvRound(y - r * sin(dist.at<float>(y, x)));
                    // voting
                    inc_if_inside(accu, x0, y0, img_binary.rows, img_binary.cols, r);
                    inc_if_inside(accu, x1, y1, img_binary.rows, img_binary.cols, r);
                }
            }
        }
    }

    // Create 2D image by suming values of the radius dimension
    for (int y0 = 0; y0 < img_binary.rows; y0++) 
    {
        for (int x0 = 0; x0 < img_binary.cols; x0++) 
        {
            for (int r = minRadius; r < radiusRange; r++)
            {
                h_acc.at<float>(y0, x0) += accu[y0][x0][r];
                h_acc.at<float>(y0, x0) = h_acc.at<float>(y0, x0) > 5 ? 255 : 0;
            }
        }
    }

    vector<Point3f> bestCircles;
    // Compute best circles
    for (int y0 = 0; y0 < img_binary.rows; y0++) 
    {
        for (int x0 = 0; x0 < img_binary.cols; x0++) 
        {
            for (int r = minRadius; r < radiusRange; r++)
            {
                // Decide the center by thresholding the h_space
                if (accu[y0][x0][r] > threshold)
                {
                    Point3f circle(x0, y0, r);
                    int i;
                    for (i = 0; i < bestCircles.size(); i++) 
                    {
                        int xCoord = bestCircles[i].x;
                        int yCoord = bestCircles[i].y;
                        int radius = bestCircles[i].z;
                        if (abs(xCoord - x0) < distance && abs(yCoord - y0) < distance)
                        {
                            if (accu[y0][x0][r] > accu[yCoord][xCoord][radius])
                            {
                                bestCircles.erase(bestCircles.begin() + i);
                                bestCircles.insert(bestCircles.begin(), circle);
                            }
                            break;
                        }
                    }
                    if (i == bestCircles.size()) {
                        bestCircles.insert(bestCircles.begin(), circle);
                    }
                }
            }
        }
    }

    // draw the circles on the raw image
    for (int i = 0; i < bestCircles.size(); i++) 
    {
        int xCoord = bestCircles[i].x;
        int yCoord = bestCircles[i].y;
        int radius = bestCircles[i].z;
        Point2f center(xCoord, yCoord);
        circle(raw, center, radius - 1, CV_RGB(0,0,255), 4, 10, 0);
    }
}

// Hough transform to detect lines
void houghLines(Mat& img_binary, float threshold, Mat& h_acc, Mat& raw)
{
    int thetas[180];
    for (int i = 0; i < 180; i++)
        thetas[i] = i;
    float cos_thetas[180], sin_thetas[180];
    for (int i = 0; i < 180; i++)
    {
        cos_thetas[i] = cos(thetas[i] / 180.0 * CV_PI);
        sin_thetas[i] = sin(thetas[i] / 180.0 * CV_PI);
    }

    // build the accumulator
    int RMax = cvRound(sqrt(2.0) * (img_binary.rows > img_binary.cols ? img_binary.rows : img_binary.cols) / 2.0);
    float** accu;
    accu = new float* [2 * RMax];
    for (int i = 0; i < 2 * RMax; i++)
    {
        accu[i] = new float[180];
    }
    for (int i = 0; i < 2 * RMax; i++)
    {
        for (int j = 0; j < 180; j++)
        {
            accu[i][j] = 0;
        }
    }

    for (int y = 0; y < img_binary.rows; y++)
    {
        for (int x = 0; x < img_binary.cols; x++)
        {
            if ((float)img_binary.at<uchar>(y, x) > 250.0)
            {
                Point2f edge_point(x - img_binary.cols / 2, y - img_binary.rows / 2);
                for (int theta_index = 0; theta_index < 180; theta_index++)
                {
                    int rho = cvRound((edge_point.x * cos_thetas[theta_index]) + (edge_point.y * sin_thetas[theta_index]));
                    int theta = thetas[theta_index];
                    accu[rho + RMax][theta_index]++;
                }
            }
        }
    }

    
    for (int y0 = 0; y0 < 2 * RMax; y0++)
    {
        for (int x0 = 0; x0 < 180; x0++)
        {
            h_acc.at<float>(y0, x0) = 0;
            h_acc.at<float>(y0, x0) += accu[y0][x0];
            // h_acc.at<float>(y0, x0) = h_acc.at<float>(y0, x0) > 5 ? 255 : 0;
        }
    }
    

    //std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > > lines;
    for (int y = 0; y < 2 * RMax; y++)
    {
        for (int x = 0; x < 180; x++)
        {
            if (accu[y][x] > threshold)
            {
                int max = accu[y][x];
                // 非极大值抑制
                for (int ly = -14; ly <= 14; ly++)
                {
                    for (int lx = -14; lx <= 14; lx++)
                    {
                        if ((ly + y >= 0 && ly + y < 2 * RMax) && (lx + x >= 0 && lx + x < 180))
                        {
                            if (accu[ly + y][lx + x] > max)
                            {
                                max = accu[ly + y][lx + x];
                                ly = lx = 15;
                            }
                        }
                    }
                }
                if (max > accu[y][x])
                    continue;

                int x1, y1, x2, y2;
                x1 = y1 = x2 = y2 = 0;
                int rho = y, theta = thetas[x];

                x1 = 0;
                y1 = ((rho - RMax) - ((x1 - img_binary.cols / 2) * cos_thetas[theta])) / sin_thetas[theta] + img_binary.rows / 2;
                x2 = img_binary.cols;
                y2 = ((rho - RMax) - ((x2 - img_binary.cols / 2) * cos_thetas[theta])) / sin_thetas[theta] + img_binary.rows / 2;

                //cout << rho << " " << theta << endl;
                //float a = cos_thetas[x], b = sin_thetas[x];
                //float x0 = (a * rho) + img_binary.cols / 2;
                //float y0 = (b * rho) + img_binary.rows / 2;
                //int x1 = int(x0 + 1000 * (-b));
                //int y1 = int(y0 + 1000 * (a));
                //int x2 = int(x0 - 1000 * (-b));
                //int y2 = int(y0 - 1000 * (a));
                //cout << x1 << "," << y1 << " " << x2 << "," << y2 << endl;
                line(raw, Point(x1, y1), Point(x2, y2), CV_RGB(0, 0, 255), 2, 8);
            }
        }
    }
    
}

// Use Opencv function HoughLines()
void myHough(Mat src, Mat dst)
{
    vector<Vec2f> lines;//用于储存参数空间的交点
    HoughLines(src, lines, 1, CV_PI / 180, 450, 0, 0);//针对不同像素的图片注意调整阈值
    const int alpha = 3000;//alpha取得充分大，保证画出贯穿整个图片的直线
    //lines中存储的是边缘直线在极坐标空间下的rho和theta值，在图像空间(直角坐标系下)只能体现出一个点
    //以该点为基准，利用theta与斜率之间的关系，找出该直线上的其他两个点(可能不在图像上)，之后以这两点画出直线
    for (size_t i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0], theta = lines[i][1];
        double cs = cos(theta), sn = sin(theta);
        double x = rho * cs, y = rho * sn;
        Point pt1(cvRound(x + alpha * (-sn)), cvRound(y + alpha * cs));
        Point pt2(cvRound(x - alpha * (-sn)), cvRound(y - alpha * cs));
        line(dst, pt1, pt2, CV_RGB(0,0,255), 2, LINE_AA);
    }
    imshow("The processed image", dst);
}

// Line detection API
void detectLine(Mat& image, string file)
{
    Mat img_grey;
    Mat mag;
    Mat h_acc;
    cvtColor(image, img_grey, COLOR_BGR2GRAY);
    mag.create(img_grey.rows, img_grey.cols, CV_32FC1);

    // edge detecting
    cout << "Edge extracting..." << endl;
    // edgeExtract(img_grey, mag, dist);

    GaussianBlur(img_grey, img_grey, Size(5, 5), 0, 0);
    Canny(img_grey, mag, 100, 200);

    //Mat element = getStructuringElement(MORPH_RECT,
    //    Size(2 * 5 + 1, 2 * 5 + 1),
    //    Point(5, 5));
    //dilate(mag, mag, element);
    //erode(mag, mag, element);

    cout << "Computing..." << endl;

    h_acc.create(2 * cvRound(sqrt(2.0) * (mag.rows > mag.cols ? mag.rows : mag.cols) / 2.0), 180, CV_32FC1);
    houghLines(mag, 150, h_acc, image);
    cout << "Done!" << endl;

    imshow("mag", mag);
    imshow("h_space", h_acc);
    imshow("result", image);
    imwrite("../../result_img/line_" + file, image);
}

// Circle detection API
void detectCircle(Mat& image, string file)
{
    Mat img_grey;     
    Mat mag, dist;

    Mat h_acc;       //hough space matricies

    cvtColor(image, img_grey, COLOR_BGR2GRAY);

    mag.create(img_grey.rows, img_grey.cols, CV_32FC1);
    dist.create(img_grey.rows, img_grey.cols, CV_32FC1);

    // edge detecting
    cout << "Edge extracting..." << endl;
    edgeExtract(img_grey, mag, dist);

    cout << "Computing..." << endl;
    h_acc.create(mag.rows, mag.cols, CV_32FC1);

    houghCircles(mag, dist, 15, 20, 150, 40, h_acc, image);
    cout << "Done!" << endl;

    imshow("mag", mag);
    imshow("h_space", h_acc);
    imshow("result", image);
    imwrite("../../result_img/circle_" + file, image);
}

int main()
{
    string file;
    string head = "../../test_img/";
    while (1)
    {
        cout << "Please input the test filename:";
        cin >> file;
        string imageName = head + file;

        Mat raw = imread(imageName);
        if (raw.data == NULL)
        {
            cout << "Invalid file! Please check if the file is in the correct folder." << endl;

        }
        else
        {
            int op;
            cout << "Please choose which shape to detect:" << endl;
            cout << "1.Line    2.Circle" << endl;
            cout << "Detect type:";
            cin >> op;
            if (op == 1)
            {
                detectLine(raw, file);
                
                //Mat mMiddle;
                //mMiddle.create(raw.rows, raw.cols, CV_8UC1);
                //cvtColor(raw, mMiddle, COLOR_BGR2GRAY);
                //Canny(mMiddle, mMiddle, 50, 150, 3);
                //Mat mResult = raw.clone();
                //myHough(mMiddle, mResult);
            }
            else if (op == 2)
            {
                detectCircle(raw, file);
            }
            else
            {
                cout << "Unknown operation." << endl;
            }


            waitKey(0);
        }
    }

    return 0;
}