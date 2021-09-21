#include <iostream>
#include <string>
#include <vector>
#include <io.h>
#include <opencv2/opencv.hpp>
#include<opencv2/imgproc/types_c.h>
#include <fstream>

using namespace cv;
using namespace std;

const int width = 1080, height = 720;
const int fps = 24;

// 获取特定格式的文件名
void getAllFormatFiles(string path, vector<string>& files, string format)
{
	intptr_t hFile = 0; 
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*" + format).c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib & _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
				{
					//files.push_back(p.assign(path).append("\\").append(fileinfo.name) );
					getAllFormatFiles(p.assign(path).append("\\").append(fileinfo.name), files, format);
				}
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);

		_findclose(hFile);
	}
}

// 模糊转场效果
void blur(VideoWriter& writer, const Mat& img)
{
	Mat img_blur;
	for (int i = fps; i > 0; i--)
	{
		GaussianBlur(img, img_blur, Size(2 * i - 1, 2 * i - 1), 0);
		writer << img_blur;
	}
}

// 制作片头
void writeHeader(VideoWriter& writer, const vector<string>& headerFileList)
{
    Mat img, imgResized;
    for (const auto& it : headerFileList)
    {
        img = imread(it);
        resize(img, imgResized, Size(width, height));
		string text = "3180105640 Liu Jiarun";
		int baseline;
		Size text_size = getTextSize(text, FONT_HERSHEY_COMPLEX, 1.5, 2, &baseline);

		// 文本框居中绘制
		Point origin;
		origin.x = width / 2 - text_size.width / 2;
		origin.y = height / 2 + text_size.height / 2;
		putText(imgResized, text, origin, FONT_HERSHEY_COMPLEX, 1.5, CV_RGB(255, 0, 0), 2, 8, 0);

        // putText(imgResized, "Liu Jiarun 3180105640", Point(80, 80), FONT_HERSHEY_TRIPLEX, 1.5, Scalar(0, 255, 0));

		blur(writer, imgResized);
        for (int i = 0; i < fps; i++)
        {
            writer << imgResized;
        }
    }
}

// 画个logo
void drawLogo(VideoWriter& writer)
{
	Mat img(Size(width, height), CV_8UC3, CV_RGB(0, 0, 0));

	for (int i = 0; i < fps; i++)
	{
		ellipse(img, Point(width / 2, height / 2 - 160), Size(130, 130), 125, 0, i * 290 / fps, CV_RGB(255, 0, 0), -1);
		circle(img, Point(width / 2, height / 2 - 160), 60, CV_RGB(0, 0, 0), -1);
		writer << img;
	}

	for (int i = 0; i < fps; i++)
	{
		ellipse(img, Point(width / 2 - 150, height / 2 + 80), Size(130, 130), 16, 0, i * 290 / fps, CV_RGB(0, 255, 0), -1);
		circle(img, Point(width / 2 - 150, height / 2 + 80), 60, CV_RGB(0, 0, 0), -1);
		writer << img;
	}

	for (int i = 0; i < fps; i++)
	{
		ellipse(img, Point(width / 2 + 150, height / 2 + 80), Size(130, 130), 300, 0, i * 290 / fps, CV_RGB(0, 0, 255), -1);
		circle(img, Point(width / 2 + 150, height / 2 + 80), 60, CV_RGB(0, 0, 0), -1);
		writer << img;
	}
	int baseline;

	string text = "This is a rubbish made by OpenCV.";
	Size text_size = getTextSize(text, FONT_HERSHEY_COMPLEX, 1, 2, &baseline);
	Point pos;
	pos.x = width / 2 - text_size.width / 2;
	pos.y = 650;
	for (int i = 0; i < fps; i++)
	{
		putText(img, text, pos, FONT_HERSHEY_COMPLEX, 1, CV_RGB(255, 255, 255), 2, 8, 0);
		writer << img;
	}

	for (int i = 0; i < fps / 4; i++)
	{
		writer << img;
	}
}

// 画个火柴人
void drawStickMan(VideoWriter& writer)
{
	Mat img(Size(width, height), CV_8UC3, CV_RGB(255, 255, 255));
	for (int i = 0; i < fps; i++)
	{
		ellipse(img, Point(500, 150), Size(130, 90), 90, 0, i * 360 / fps, CV_RGB(0, 0, 0), 3, 8);
		circle(img, Point(500, 150), 30, CV_RGB(255, 255, 255), -1);
		writer << img;
	}
	for (int i = 0; i < fps; i++)
	{
		line(img, Point(460, 160), Point(460, 90), CV_RGB(0, 0, 0), 3);
		line(img, Point(500, 160), Point(500, 90), CV_RGB(0, 0, 0), 3);
		writer << img;
	}
	for (int i = 0; i < fps; i++)
	{
		ellipse(img, Point(520, 480), Size(200, 50), 270, 0, i * 180 / fps, CV_RGB(0, 0, 0), 5, 8);
		circle(img, Point(520, 480), 30, CV_RGB(255, 255, 255), -1);
		writer << img;
	}
	for (int i = 0; i < fps; i++)
	{
		line(img, Point(400, 500), Point(555, 350), CV_RGB(0, 0, 0), 3);
		line(img, Point(730, 200), Point(555, 350), CV_RGB(0, 0, 0), 3);
		writer << img;
	}
	for (int i = 0; i < fps; i++)
	{
		ellipse(img, Point(400, 500), Size(30, 20), 130, 0, i * 360 / fps, CV_RGB(0, 0, 0), 3, 8);
		circle(img, Point(400, 500), 20, CV_RGB(255, 255, 255), -1);
		writer << img;
	}
	for (int i = 0; i < fps; i++)
	{
		ellipse(img, Point(730, 200), Size(30, 20), 120, 0, i * 360 / fps, CV_RGB(0, 0, 0), 3, 8);
		circle(img, Point(730, 200), 20, CV_RGB(255, 255, 255), -1);
		writer << img;
	}

	for (int i = 0; i < fps / 6; i++)
	{
		writer << img;
	}

}

// 右侧进入转场效果
void translate(VideoWriter& writer, const Mat& img)
{
	Mat newimg(img.size(), img.type());
	for (int i = 1; i <= fps; i++)
	{
		newimg.setTo(Scalar(0, 0, 0));
		Rect rect(img.cols * (1 - (float)i / (float)fps), 0, img.cols * (float)i / (float)fps, img.rows);
		img.colRange(0, img.cols * (float)i / (float)fps).copyTo(newimg(rect));
		writer << newimg;
	}
}


// 正片
void writeBody(VideoWriter& writer, const vector<string>& picFileList, const vector<string>& videoFileList)
{
	Mat img, imgResized;
	VideoCapture capture;
	img = imread(picFileList[0]);
	resize(img, imgResized, Size(width, height));
	translate(writer, imgResized);
	for (int i = 0; i < fps; i++)
	{
		writer << imgResized;
	}

	img = imread(picFileList[1]);
	resize(img, imgResized, Size(width, height));
	for (int i = 0; i < fps; i++)
	{
		writer << imgResized;
	}

	Mat imgtext1(Size(width, height), CV_8UC3, CV_RGB(0, 0, 0));
	string text1 = "What's this?";
	int baseline;
	Size text_size1 = getTextSize(text1, FONT_HERSHEY_COMPLEX, 1.5, 2, &baseline);
	Point pos1;
	pos1.x = width / 2 - text_size1.width / 2;
	pos1.y = height / 2 + text_size1.height / 2;
	for (int i = 0; i < fps; i++)
	{
		putText(imgtext1, text1, pos1, FONT_HERSHEY_COMPLEX, 1.5, CV_RGB(255, 255, 255), 2, 8, 0);
		writer << imgtext1;
	}

	img = imread(picFileList[2]);
	resize(img, imgResized, Size(width, height));
	for (int i = 0; i < fps; i++)
	{
		writer << imgResized;
	}

	Mat imgtext2(Size(width, height), CV_8UC3, CV_RGB(0, 0, 0));
	string text2 = "OpenCV?";
	Size text_size2 = getTextSize(text2, FONT_HERSHEY_COMPLEX, 1.5, 2, &baseline);
	Point pos2;
	pos2.x = width / 2 - text_size2.width / 2;
	pos2.y = height / 2 + text_size2.height / 2;
	for (int i = 0; i <= fps; i++)
	{
		putText(imgtext2, text2, pos2, FONT_HERSHEY_COMPLEX, 1.5, CV_RGB(255, 255, 255), 2, 8, 0);
		writer << imgtext2;
	}

	img = imread(picFileList[3]);
	resize(img, imgResized, Size(width, height));
	for (int i = 0; i < fps; i++)
	{
		writer << imgResized;
	}
	img = imread(picFileList[4]);
	resize(img, imgResized, Size(width, height));
	for (int i = 0; i < fps; i++)
	{
		writer << imgResized;
	}

	capture.open(videoFileList[0]);
	while (true)
	{
		capture >> img;
		if (img.empty())
		{
			break;
		}
		resize(img, imgResized, Size(width, height));
		writer << imgResized;
	}
	capture.release();

	capture.open(videoFileList[1]);
	while (true)
	{
		capture >> img;
		if (img.empty())
		{
			break;
		}
		resize(img, imgResized, Size(width, height));
		writer << imgResized;
	}
	capture.release();

	img = imread(picFileList[5]);
	resize(img, imgResized, Size(width, height));
	for (int i = 0; i < fps; i++)
	{
		writer << imgResized;
	}

	Mat imgtext3(Size(width, height), CV_8UC3, CV_RGB(0, 0, 0));
	string text3 = "THIS IS AMAZING!!";
	Size text_size3 = getTextSize(text3, FONT_HERSHEY_COMPLEX, 1.5, 2, &baseline);
	Point pos3;
	pos3.x = width / 2 - text_size3.width / 2;
	pos3.y = height / 2 + text_size3.height / 2;
	for (int i = 0; i < fps; i++)
	{
		putText(imgtext3, text3, pos3, FONT_HERSHEY_COMPLEX, 1.5, CV_RGB(255, 255, 255), 2, 8, 0);
		writer << imgtext3;
	}

	Mat imgtext4(Size(width, height), CV_8UC3, CV_RGB(0, 0, 0));
	string text4 = "Let me try something new.";
	Size text_size4 = getTextSize(text4, FONT_HERSHEY_COMPLEX, 1.5, 2, &baseline);
	Point pos4;
	pos4.x = width / 2 - text_size4.width / 2;
	pos4.y = height / 2 + text_size4.height / 2;
	for (int i = 0; i < fps; i++)
	{
		putText(imgtext4, text4, pos4, FONT_HERSHEY_COMPLEX, 1.5, CV_RGB(255, 255, 255), 2, 8, 0);
		writer << imgtext4;
	}

	drawStickMan(writer);

	img = imread(picFileList[5]);
	resize(img, imgResized, Size(width, height));
	for (int i = 0; i < fps; i++)
	{
		writer << imgResized;
	}

	Mat imgtext5(Size(width, height), CV_8UC3, CV_RGB(0, 0, 0));
	string text5 = "My work is done!";
	Size text_size5 = getTextSize(text5, FONT_HERSHEY_COMPLEX, 1.5, 2, &baseline);
	Point pos5;
	pos5.x = width / 2 - text_size5.width / 2;
	pos5.y = height / 2 + text_size5.height / 2;
	for (int i = 0; i < fps; i++)
	{
		putText(imgtext5, text5, pos5, FONT_HERSHEY_COMPLEX, 1.5, CV_RGB(255, 255, 255), 2, 8, 0);
		writer << imgtext5;
	}

	img = imread(picFileList[6]);
	resize(img, imgResized, Size(width, height));
	for (int i = 0; i < fps; i++)
	{
		writer << imgResized;
	}
}

void writeEnd(VideoWriter& writer)
{
	Mat img, imgResized;
	Mat imgtext(Size(width, height), CV_8UC3, CV_RGB(0, 0, 0));
	string text = "THE END.";
	int baseline;
	for (int j = 0; j < 4; j++)
	{
		Size text_size1 = getTextSize(text, FONT_HERSHEY_COMPLEX, 2, 2, &baseline);
		Point pos1;
		pos1.x = width / 2 - text_size1.width / 2;
		pos1.y = height / 2 + text_size1.height / 2;
		for (int i = 0; i < fps / 6; i++)
		{
			rectangle(imgtext, Point(0, 0), Point(1080, 720), CV_RGB(0, 0, 0), -1);
			putText(imgtext, text, pos1, FONT_HERSHEY_COMPLEX, 2, CV_RGB(255, 255, 255), 2, 8, 0);
			writer << imgtext;
		}
		Size text_size2 = getTextSize(text, FONT_HERSHEY_SCRIPT_SIMPLEX, 1.5, 2, &baseline);
		Point pos2;
		pos2.x = width / 2 - text_size2.width / 2;
		pos2.y = height / 2 + text_size2.height / 2;
		for (int i = 0; i < fps / 6; i++)
		{
			rectangle(imgtext, Point(0, 0), Point(1080, 720), CV_RGB(0, 0, 0), -1);
			putText(imgtext, text, pos2, FONT_HERSHEY_SCRIPT_SIMPLEX, 1.5, CV_RGB(255, 0, 0), 3, 8, 0);
			writer << imgtext;
		}
		Size text_size3 = getTextSize(text, FONT_HERSHEY_SIMPLEX, 1.75, 2, &baseline);
		Point pos3;
		pos3.x = width / 2 - text_size3.width / 2;
		pos3.y = height / 2 + text_size3.height / 2;
		for (int i = 0; i < fps / 6; i++)
		{
			rectangle(imgtext, Point(0, 0), Point(1080, 720), CV_RGB(0, 0, 0), -1);
			putText(imgtext, text, pos3, FONT_HERSHEY_SIMPLEX, 1.75, CV_RGB(0, 255, 0), 3, 8, 0);
			writer << imgtext;
		}
	}
	
}

// 播放视频
void playVideo(string dir)
{
	VideoCapture capture;
	capture.open(dir);
	if (!capture.isOpened()) {
		cout << "Can't open video！" << endl;
		return;
	}
	int delay = fps;
	while (1) {
		Mat frame;
		capture >> frame;
		if (frame.empty())
			break;
		imshow("output", frame);
		if (delay >= 0 && waitKey(delay) >= 32)
			waitKey(0);
	}
	capture.release();
}

int main()
{
    string headerDir = "..\\assets\\header";
    vector<string> headerFileList;
	getAllFormatFiles(headerDir, headerFileList, ".jpg");
	string mainDir = "..\\assets\\main";
	vector<string> picFileList;
	vector<string> videoFileList;
	getAllFormatFiles(mainDir, picFileList, ".jpg");
	getAllFormatFiles(mainDir, videoFileList, ".mp4");

    VideoWriter writer("..\\output.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(width, height));
	cout << "Writing video, please wait..." << endl;
	cout << "Writing header..." << endl;
	writeHeader(writer, headerFileList);

	drawLogo(writer);
	cout << "Writing body..." << endl;
	writeBody(writer, picFileList, videoFileList);
	cout << "Writing end..." << endl;
	writeEnd(writer);
	writer.release();
	cout << "Done!" << endl;
	playVideo("..\\output.mp4");
}

