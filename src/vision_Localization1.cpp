#include <ros/ros.h>
#include <std_msgs/String.h>
#include <sstream>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

cv::Mat cameraMatrix, distCoeffs;
cv::Mat img_h, img_s, img_v, imghsv;
cv::Mat threshold_out;
cv::Mat ROI, src_img_ROI;

cv_bridge::CvImagePtr cv_ptr;

int robot_command;
int corner_position[4][2];
int obj_data[3][9];
int obj_roi[2][4];

double line_position[7][5];
double theta, theta_1;
double arm_x = 0, arm_y = 0, arm_z = 0;
double arm_rx = 0, arm_ry = 0, arm_rz = 0;

int data_1_x[10][4];
int data_1_y[10][4];

int buffer_1_x[10][4];
int buffer_1_y[10][4];
float obj_theta[10];


cv::FileStorage fs("/home/scl/sclagv_ws/src/cv_vision/src/camera_para.yaml", cv::FileStorage::READ);

void first_Localization(cv::Mat src_img)
{
   for(int i = 0; i < 10; i++)
   {
      for(int j = 0; j < 4; j++)
      {
          data_1_x[i][j] = 0;
          data_1_x[i][j] = 0;
          buffer_1_x[i][j] = 0;
          buffer_1_y[i][j] = 0;
      }
   }

   std::vector<cv::Mat> hsv_vec;
   cv::cvtColor(src_img, imghsv, CV_BGR2HSV);
  
   split(imghsv, hsv_vec);
   img_h = hsv_vec[0];
   img_s = hsv_vec[1];
   img_v = hsv_vec[2];
   img_h.convertTo(img_h, CV_8U);
   img_s.convertTo(img_s, CV_8U);
   img_v.convertTo(img_v, CV_8U);
	
   cv::medianBlur(img_h, img_h, 31); //median filter 71 x 71

   //imshow("Hue", img_h);
   //namedWindow("threshold", CV_WINDOW_FREERATIO); // create namedwindows "threshold"
   //namedWindow("result", CV_WINDOW_FREERATIO);  // create namedwindows "result"

   cv::threshold(img_h, threshold_out, 90, 255, cv::THRESH_BINARY_INV); // gray to binary, thresholding = 124

   cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(91, 91), cv::Point(-1, -1));
   cv::morphologyEx(threshold_out, threshold_out, CV_MOP_CLOSE, kernel);
   cv::imshow("test", threshold_out);

   //imshow("threshold", threshold_out);
   std::vector<std::vector<cv::Point> > contours;
   cv::findContours(threshold_out, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE); // findcontours

   double max_area = 0;
   int max_area_contour = -1;

   std::vector<cv::Moments> mu(contours.size());
   std::vector<cv::Point2f> mc(contours.size());
   std::vector<cv::RotatedRect> minRect(contours.size());

   cv::Point2f rect_points[4];

   int num = 0;
   for (int i = 0; i < contours.size(); i++)
   {
      size_t count = contours[i].size();
      double area = contourArea(contours.at(i)); // find object area

      if (area > 1000 && area < 10000)
      {
         std::cout << "area = " << area << std::endl;
         mu[i] = cv::moments(contours[i], false);
         mc[i] = cv::Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
         std::cout << "center = " << mc[i] << std::endl;

         minRect[i] = minAreaRect(cv::Mat(contours[i])); // search object min area
         minRect[i].points(rect_points); // find area four point

         for (int j = 0; j < 4; j++)
         {
            //line(src_img, rect_points[j], rect_points[(j + 1) % 4], cv::Scalar(0, 255, 0), 2, 5);
            std::cout << "Point" << rect_points[j] << std::endl;
	    //corner_position[j][0] = rect_points[j].x;
	    //corner_position[j][1] = rect_points[j].y;
            data_1_x[num][j] = rect_points[j].x;
            data_1_y[num][j] = rect_points[j].y;
            buffer_1_x[num][j] = rect_points[j].x;
            buffer_1_y[num][j] = rect_points[j].y;
	 }

	 cv::Mat pointsf;
	 cv::Mat(contours[i]).convertTo(pointsf, CV_32F);
	 cv::Vec4f fitline;
	 cv::fitLine(pointsf, fitline, CV_DIST_L2, 0, 0.01, 0.01);
	 int lefty = int((-fitline[2] * fitline[1] / fitline[0]) + fitline[3]);
	 int righty = int(((1280 - fitline[2]) * fitline[1] / fitline[0]) + fitline[3]);
	 obj_theta[num] = atan((float)(righty - lefty) / (src_img.cols - 1)) * 180 / 3.1415;
         num++;
	}
    }

    //泡沫排序法
    for(int i = 0; i < 10; i++)
    {
	for(int row = 4; row > 1; row--)
	{
	    for(int col = 0; col < row - 1; col++)
	    {
		if (buffer_1_x[i][col + 1] < buffer_1_x[i][col])
		{
		   int temp_x = buffer_1_x[i][col + 1];   
		   buffer_1_x[i][col + 1] = buffer_1_x[i][col];  
		   buffer_1_x[i][col] = temp_x;
		}
					
		if (buffer_1_y[i][col + 1] < buffer_1_y[i][col])
		{
		   int temp_y = buffer_1_y[i][col + 1];   
		   buffer_1_y[i][col + 1] = buffer_1_y[i][col];  
		   buffer_1_y[i][col] = temp_y;
		}
	    }
	}
    }
	
    //filter
    int num_1 = 0;
    for(int i = 0; i < 10; i++)
    {
	int roi_x = buffer_1_x[i][3] - buffer_1_x[i][0];
	int roi_y = buffer_1_y[i][3] - buffer_1_y[i][0];
	if(roi_x > roi_y)
	{
	   break;
	}
	num_1++; 
    }

    std::vector<cv::Point2f> obj_corner_pts(4);
    obj_corner_pts[0] = cvPoint(data_1_x[num_1][0], data_1_y[num_1][0]);
    obj_corner_pts[1] = cvPoint(data_1_x[num_1][1], data_1_y[num_1][1]);
    obj_corner_pts[2] = cvPoint(data_1_x[num_1][2], data_1_y[num_1][2]);
    obj_corner_pts[3] = cvPoint(data_1_x[num_1][3], data_1_y[num_1][3]);

    std::cout << obj_corner_pts << std::endl;
    cv::line(src_img, cv::Point(data_1_x[num_1][0], data_1_y[num_1][0]), cv::Point(data_1_x[num_1][1], data_1_y[num_1][1]), cv::Scalar(0, 255, 0), 2, 5);
    cv::line(src_img, cv::Point(data_1_x[num_1][1], data_1_y[num_1][1]), cv::Point(data_1_x[num_1][2], data_1_y[num_1][2]), cv::Scalar(0, 255, 0), 2, 5);
    cv::line(src_img, cv::Point(data_1_x[num_1][2], data_1_y[num_1][2]), cv::Point(data_1_x[num_1][3], data_1_y[num_1][3]), cv::Scalar(0, 255, 0), 2, 5);
    cv::line(src_img, cv::Point(data_1_x[num_1][3], data_1_y[num_1][3]), cv::Point(data_1_x[num_1][0], data_1_y[num_1][0]), cv::Scalar(0, 255, 0), 2, 5);

    theta_1 = atan((float)(corner_position[0][1] - corner_position[1][1]) / (corner_position[0][0] - corner_position[1][0])) * 180 / 3.1415;

    std::vector<cv::Point3d> corner_pts_3d(4);
    if ((obj_theta[num_1] - theta_1) < -80 || (obj_theta[num_1] - theta_1) > 80)
    {
       corner_pts_3d[0] = (cv::Point3d(double(-0.06), float(0.035), 0));
       corner_pts_3d[1] = (cv::Point3d(double(-0.06), float(-0.035), 0));
       corner_pts_3d[2] = (cv::Point3d(double(0.06), float(-0.035), 0));
       corner_pts_3d[3] = (cv::Point3d(double(0.06), float(0.035), 0));
    }
    else
    {
       corner_pts_3d[0] = (cv::Point3d(double(0.06), float(0.035), 0));
       corner_pts_3d[1] = (cv::Point3d(double(-0.06), float(0.035), 0));
       corner_pts_3d[2] = (cv::Point3d(double(-0.06), float(-0.035), 0));
       corner_pts_3d[3] = (cv::Point3d(double(0.06), float(-0.035), 0));
    }

    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;

    cv::Mat rotation, translation;
    solvePnP(corner_pts_3d, obj_corner_pts, cameraMatrix, distCoeffs, rotation, translation);

    double x = translation.at<double>(0, 0); double y = translation.at<double>(1, 0); double z = translation.at<double>(2, 0);
    double rx = rotation.at<double>(0, 0); double ry = rotation.at<double>(1, 0); double rz = rotation.at<double>(2, 0);

    arm_x = 0.26 + z + 0.02*cos(ry)*sin(rx) - 0.1*cos(ry)*cos(rx);
    arm_y = -x + 0.02*(sin(rz)*cos(rx) - cos(rz)*sin(ry)*sin(rx)) + 0.1*(sin(rz)*sin(rx) + cos(rz)*sin(ry)*cos(rx));
    arm_rz = atan((-sin(rz)*sin(rx)-cos(rz)*sin(ry)*cos(rz))/(-cos(rz)*cos(rx)));

    char translation_x[100], translation_y[100], rotation_z[100];

    sprintf(translation_x, "vision_x = %.3f ", arm_x);
    sprintf(translation_y, "vision_y = %.3f ", arm_y);
    sprintf(rotation_z, "vision_theta = %.3f ", arm_rz);

    putText(src_img, translation_x, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    putText(src_img, translation_y, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    putText(src_img, rotation_z, cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
       
    cv::imshow("view", src_img);
    cv::waitKey(3);
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    first_Localization(cv_ptr->image);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "vision_Localization1");
  ros::NodeHandle nh;
  cv::namedWindow("view");
  cv::startWindowThread();
  image_transport::ImageTransport it(nh);
  image_transport::Subscriber sub = it.subscribe("/ueye_cam/image_raw", 1, imageCallback);
  ros::spin();
  cv::destroyWindow("view");
}
