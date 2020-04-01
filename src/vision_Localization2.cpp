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

cv::FileStorage fs("/home/scl/sclagv_ws/src/cv_vision/src/camera_para.yaml", cv::FileStorage::READ);

void second_Localization(cv::Mat src_img)
{
    for (int i = 0; i < 7; i++)
    {
	for (int j = 0; j < 5; j++)
	{
	    line_position[i][j] = 0;
	}
    }

    std::vector<cv::Mat> bgr_vec;
    cv::Mat img_b, img_g, img_r;
    split(src_img, bgr_vec);
    img_b = bgr_vec[0];
    img_g = bgr_vec[1];
    img_r = bgr_vec[2];

    cv::threshold(img_b, img_b, 90, 255, cv::THRESH_BINARY);
    cv::threshold(img_g, img_g, 100, 255, cv::THRESH_BINARY);
    cv::threshold(img_r, img_r, 90, 255, cv::THRESH_BINARY);

    std::vector<cv::Mat> imgDsc_merge;

    for (int i = 0; i < 3; i++)
    {
	if (i == 0)
	{
	   imgDsc_merge.push_back(img_b);
	}

	if (i == 1)
	{
	   imgDsc_merge.push_back(img_g);
	}

	if (i == 2)
	{
	    imgDsc_merge.push_back(img_r);
	}
    }

    cv::Mat th_src_img;
    cv::merge(imgDsc_merge, th_src_img);
    cv::imshow("color_thre", th_src_img);

    cv::Mat th_gray_img, threshold_out;
    cv::cvtColor(th_src_img, th_gray_img, CV_BGR2GRAY);
    cv::threshold(th_gray_img, threshold_out, 100, 255, cv::THRESH_BINARY);

    cv::Mat threshold_erode;
    cv::Mat kerne1 = getStructuringElement(cv::MORPH_RECT, cv::Size(25, 25), cv::Point(-1, -1));
    cv::erode(threshold_out, threshold_erode, kerne1);

    cv::Mat threshold_dilate;
    cv::Mat kerne2 = getStructuringElement(cv::MORPH_RECT, cv::Size(125, 125), cv::Point(-1, -1));
    cv::dilate(threshold_erode, threshold_dilate, kerne2);

    cv::Mat threshold_and;
    cv::bitwise_and(threshold_out, threshold_dilate, threshold_and);
    cv::imshow("result_1", threshold_and);

    cv::Mat threshold_erode_1;
    cv::Mat kerne3 = getStructuringElement(cv::MORPH_RECT, cv::Size(25, 25), cv::Point(-1, -1));
    cv::erode(threshold_and, threshold_erode_1, kerne3);


    cv::Mat threshold_dilate_1;
    cv::Mat kerne4 = getStructuringElement(cv::MORPH_RECT, cv::Size(125, 125), cv::Point(-1, -1));
    cv::dilate(threshold_erode_1, threshold_dilate_1, kerne4);

    cv::Mat threshold_and_1;
    cv::bitwise_and(threshold_and, threshold_dilate_1, threshold_and_1);
    cv::imshow("result_2", threshold_and_1);

    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(threshold_and_1, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE); // findcontours

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

       if (area > 3000 & area < 30000)
       {
	  mu[i] = cv::moments(contours[i], false);
	  mc[i] = cv::Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
	  obj_data[num][0] = area;

	  minRect[i] = cv::minAreaRect(cv::Mat(contours[i])); // search object min area
	  minRect[i].points(rect_points); // find area four point

	  for (int j = 0; j < 4; j++)
	  {
	      corner_position[j][0] = rect_points[j].x;
	      corner_position[j][1] = rect_points[j].y;
	  }

	  obj_data[num][1] = corner_position[0][0];
	  obj_data[num][2] = corner_position[0][1];
	  obj_data[num][3] = corner_position[1][0];
	  obj_data[num][4] = corner_position[1][1];
	  obj_data[num][5] = corner_position[2][0];
	  obj_data[num][6] = corner_position[2][1];
	  obj_data[num][7] = corner_position[3][0];
	  obj_data[num][8] = corner_position[3][1];
	  num++;
      }
    }

    ////////// 泡沫排序法 尋找最小面積 /////////
    for (int j = 7; j > 1; j--)
    {
        for (int i = 0; i < j - 1; i++)
	{
	   if (obj_data[i + 1][0] < obj_data[i][0])
	   {
	      for (int k = 0; k < 9; k++)
	      {
	         int temp = obj_data[i + 1][k];   obj_data[i + 1][k] = obj_data[i][k];  obj_data[i][k] = temp;
	      }
	   }
	}
    }

    int a = 0;
    for (int i = 0; i < 7; i++)
    {
        float w = pow((obj_data[a][1] - obj_data[a][3]), 2)  + pow((obj_data[a][2] - obj_data[a][4]), 2);
        float h = pow((obj_data[a][3] - obj_data[a][5]), 2)  + pow((obj_data[a][4] - obj_data[a][6]), 2);

        if(obj_data[a][0]!= 0 & w/h < 2 & w/h > 0.5)
	{
	   break;
	}
	a++;
    }
    //std::cout << obj_data[a][0] << std::endl;

    cv::line(src_img, cv::Point(obj_data[a][1], obj_data[a][2]), cv::Point(obj_data[a][3], obj_data[a][4]), cv::Scalar(0, 255, 0), 2, 5);
    cv::line(src_img, cv::Point(obj_data[a][3], obj_data[a][4]), cv::Point(obj_data[a][5], obj_data[a][6]), cv::Scalar(0, 255, 0), 2, 5);
    cv::line(src_img, cv::Point(obj_data[a][5], obj_data[a][6]), cv::Point(obj_data[a][7], obj_data[a][8]), cv::Scalar(0, 255, 0), 2, 5);
    cv::line(src_img, cv::Point(obj_data[a][7], obj_data[a][8]), cv::Point(obj_data[a][1], obj_data[a][2]), cv::Scalar(0, 255, 0), 2, 5);

    theta_1 = atan((float)(obj_data[a][2] - obj_data[a][4]) / (obj_data[a][1] - obj_data[a][3])) * 180 / 3.1415;

    std::vector<cv::Point2f> obj_corner_pts(4);
    obj_corner_pts[0] = cvPoint(obj_data[a][1], obj_data[a][2]);
    obj_corner_pts[1] = cvPoint(obj_data[a][3], obj_data[a][4]);
    obj_corner_pts[2] = cvPoint(obj_data[a][5], obj_data[a][6]);
    obj_corner_pts[3] = cvPoint(obj_data[a][7], obj_data[a][8]);

    std::vector<cv::Point3d> corner_pts_3d(4);
    if ((theta - theta_1) < -30 || (theta - theta_1) > 30)
    {
       corner_pts_3d[0] = (cv::Point3d(double(0.007), float(0.0085), 0));
       corner_pts_3d[1] = (cv::Point3d(double(-0.007), float(0.0085), 0));
       corner_pts_3d[2] = (cv::Point3d(double(-0.007), float(-0.0085), 0));
       corner_pts_3d[3] = (cv::Point3d(double(0.007), float(-0.0085), 0));
    }
    else
    {
       corner_pts_3d[0] = (cv::Point3d(double(0.0085), float(0.007), 0));
       corner_pts_3d[1] = (cv::Point3d(double(-0.0085), float(0.007), 0));
       corner_pts_3d[2] = (cv::Point3d(double(-0.0085), float(-0.007), 0));
       corner_pts_3d[3] = (cv::Point3d(double(0.0085), float(-0.007), 0));
    }

    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;

    cv::Mat rotation, translation;
    solvePnP(corner_pts_3d, obj_corner_pts, cameraMatrix, distCoeffs, rotation, translation);

    double x = translation.at<double>(0, 0); double y = translation.at<double>(1, 0); double z = translation.at<double>(2, 0);
    double rx = rotation.at<double>(0, 0); double ry = rotation.at<double>(1, 0); double rz = rotation.at<double>(2, 0);    

    arm_x = 0.26 + z - 0.02*cos(ry)*sin(rx) - 0.1*cos(ry)*cos(rx);
    arm_y = -x - 0.02*(sin(rz)*cos(rx) - cos(rz)*sin(ry)*sin(rx)) + 0.1*(sin(rz)*sin(rx) + cos(rz)*sin(ry)*cos(rx));
    arm_z = 0.2505 - y + 0.02*(cos(rz)*cos(rx) + sin(rz)*sin(ry)*sin(rx)) - 0.1*(cos(rz)*sin(rx) - sin(rz)*sin(ry)*cos(rx));
    arm_rx = atan((sin(rz)*cos(ry))/(cos(rz)*cos(rx)+sin(rz)*sin(ry)*sin(rz)));
    arm_ry = asin(-cos(rz)*sin(rx) + sin(rz)*sin(ry)*cos(rx));
    arm_rz = atan((-sin(rz)*sin(rx)-cos(rz)*sin(ry)*cos(rz))/(-cos(rz)*cos(rx)));

    char translation_x[20], translation_y[20], translation_z[20];
    char rotation_x[20], rotation_y[20], rotation_z[20];

    sprintf(translation_x, "vision_x = %.3f ", x);
    sprintf(translation_y, "vision_y = %.3f ", y);
    sprintf(translation_z, "vision_z = %.3f ", z);

    sprintf(rotation_x, "vision_rx = %.3f ", rx);
    sprintf(rotation_y, "vision_ry = %.3f ", ry);
    sprintf(rotation_z, "vision_rz = %.3f ", rz);

    putText(src_img, translation_x, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    putText(src_img, translation_y, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    putText(src_img, translation_z, cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        
    putText(src_img, rotation_x, cv::Point(300, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    putText(src_img, rotation_y, cv::Point(300, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    putText(src_img, rotation_z, cv::Point(300, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

    char robot_x[20], robot_y[20], robot_z[20];
    char robot_rx[20], robot_ry[20], robot_rz[20];

    sprintf(robot_x, "robot_x = %.3f ", arm_x);
    sprintf(robot_y, "robot_y = %.3f ", arm_y);
    sprintf(robot_z, "robot_z = %.3f ", arm_z);

    sprintf(robot_rx, "robot_rx = %.3f ", arm_rx);
    sprintf(robot_ry, "robot_ry = %.3f ", arm_ry);
    sprintf(robot_rz, "robot_rz = %.3f ", arm_rz);

    putText(src_img, robot_x, cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    putText(src_img, robot_y, cv::Point(10, 150), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    putText(src_img, robot_z, cv::Point(10, 180), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        
    putText(src_img, robot_rx, cv::Point(300, 120), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    putText(src_img, robot_ry, cv::Point(300, 150), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    putText(src_img, robot_rz, cv::Point(300, 180), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

    imshow("result", src_img);
    cv::waitKey(3);
}


void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    second_Localization(cv_ptr->image);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "vision_Localization2");
  ros::NodeHandle nh;
  cv::namedWindow("view");
  cv::startWindowThread();
  image_transport::ImageTransport it(nh);
  image_transport::Subscriber sub = it.subscribe("/ueye_cam/image_raw", 1, imageCallback);
  ros::spin();
  cv::destroyWindow("view");
}
