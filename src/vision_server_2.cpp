#include <ros/ros.h>
#include <std_msgs/String.h>
#include <sstream>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <cv_vision/robot2vision.h>
#include <scl_agv/vision_feedback.h>

cv::Mat cameraMatrix, distCoeffs;
cv::Mat img_h, img_s, img_v, imghsv;
cv::Mat img_gray;
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

class camera_server{
public:
    ros::NodeHandle nh;
};

void first_Localization(cv::Mat src_img)
{
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

   cv::threshold(img_h, threshold_out, 80, 255, cv::THRESH_BINARY_INV); // gray to binary, thresholding = 124

   cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(91, 91), cv::Point(-1, -1));
   cv::morphologyEx(threshold_out, threshold_out, CV_MOP_CLOSE, kernel);

   //imshow("threshold", threshold_out);
   std::vector<std::vector<cv::Point> > contours;
   cv::findContours(threshold_out, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE); // findcontours

   double max_area = 0;
   int max_area_contour = -1;

   std::vector<cv::Moments> mu(contours.size());
   std::vector<cv::Point2f> mc(contours.size());
   std::vector<cv::RotatedRect> minRect(contours.size());

   cv::Point2f rect_points[4];

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
            line(src_img, rect_points[j], rect_points[(j + 1) % 4], cv::Scalar(0, 255, 0), 2, 5);
            std::cout << "Point" << rect_points[j] << std::endl;
	    corner_position[j][0] = rect_points[j].x;
	    corner_position[j][1] = rect_points[j].y;
	 }

	 cv::Mat pointsf;
	 cv::Mat(contours[i]).convertTo(pointsf, CV_32F);
	 cv::Vec4f fitline;
	 cv::fitLine(pointsf, fitline, CV_DIST_L2, 0, 0.01, 0.01);
	 int lefty = int((-fitline[2] * fitline[1] / fitline[0]) + fitline[3]);
	 int righty = int(((1280 - fitline[2]) * fitline[1] / fitline[0]) + fitline[3]);
	 theta = atan((float)(righty - lefty) / (src_img.cols - 1)) * 180 / 3.1415;
	}
    }

    theta_1 = atan((float)(corner_position[0][1] - corner_position[1][1]) / (corner_position[0][0] - corner_position[1][0])) * 180 / 3.1415;

    std::vector<cv::Point2f> obj_corner_pts(4);
    obj_corner_pts[0] = cvPoint(corner_position[0][0], corner_position[0][1]);
    obj_corner_pts[1] = cvPoint(corner_position[1][0], corner_position[1][1]);
    obj_corner_pts[2] = cvPoint(corner_position[2][0], corner_position[2][1]);
    obj_corner_pts[3] = cvPoint(corner_position[3][0], corner_position[3][1]);

    std::cout << obj_corner_pts << std::endl;

    std::vector<cv::Point3d> corner_pts_3d(4);
    if ((theta - theta_1) < -80 || (theta - theta_1) > 80)
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
}

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
    //cv::imshow("color_thre", th_src_img);

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
    //cv::imshow("result_1", threshold_and);

    cv::Mat threshold_erode_1;
    cv::Mat kerne3 = getStructuringElement(cv::MORPH_RECT, cv::Size(25, 25), cv::Point(-1, -1));
    cv::erode(threshold_and, threshold_erode_1, kerne3);


    cv::Mat threshold_dilate_1;
    cv::Mat kerne4 = getStructuringElement(cv::MORPH_RECT, cv::Size(125, 125), cv::Point(-1, -1));
    cv::dilate(threshold_erode_1, threshold_dilate_1, kerne4);

    cv::Mat threshold_and_1;
    cv::bitwise_and(threshold_and, threshold_dilate_1, threshold_and_1);
    //cv::imshow("result_2", threshold_and_1);

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

       if (area > 1000)
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

	if (obj_data[i][0] != 0 & w/h < 2 & w/h > 0.5)
	{
	   break;
	}
	a++;
    }
    std::cout << obj_data[a][0] << std::endl;

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

    ROS_INFO("robot_x = %f, robot_y = %f, robot_z = %f", arm_x, arm_y, arm_z);
    ROS_INFO("robot_rx = %f, robot_ry = %f, robot_rz = %f", arm_rx, arm_ry, arm_rz);
}

void image_call(const sensor_msgs::ImageConstPtr& msg)
{
  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  // image process
  if(robot_command == 1)
  {
     first_Localization(cv_ptr->image);
  }
  else if(robot_command == 2)
  {
     second_Localization(cv_ptr->image);
  }
}

bool robot_connect_vision(cv_vision::robot2vision::Request  &req,
                          cv_vision::robot2vision::Response &res)
{
  camera_server cam_server;
  robot_command = req.command;

  ROS_INFO("recieved_command = %ld", (long int)req.command);

  image_transport::ImageTransport it(cam_server.nh);
  image_transport::Subscriber sub = it.subscribe("/ueye_cam/image_raw", 10, image_call);

  ros::Rate loop_rate(1);
  for(int i = 0; i < 3; i++)
  {
     ros::spinOnce();
     loop_rate.sleep();
  }

  if(robot_command == 1)
  {
     res.num = 0;
  }
  else if(robot_command == 2)
  {
     res.num = 1;
  }

  res.robot_x = arm_x;
  res.robot_y = arm_y;
  res.robot_z = arm_z;
  res.rotation_x = arm_rx;
  res.rotation_y = arm_ry;
  res.rotation_z = arm_rz;

  ROS_INFO("num = %ld", (long int)res.num);
  ROS_INFO("robot_x = %f, robot_y = %f, robot_z = %f", res.robot_x, res.robot_y, res.robot_z);
  ROS_INFO("robot_rx = %f, robot_ry = %f, robot_rz = %f", res.rotation_x, res.rotation_y, res.rotation_z);
  return true;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "vision_server_2");

  camera_server cam_server;

  ros::Publisher feed_pose = cam_server.nh.advertise<scl_agv::vision_feedback>("feedback", 1000);
  ros::ServiceServer service = cam_server.nh.advertiseService("robot_vision", robot_connect_vision);
  
  scl_agv::vision_feedback feedback;
  ROS_INFO("Ready to camera capcture");

  ros::Rate loop_rate(1);
  while(ros::ok)
  {
     ros::spinOnce();
     feedback.x = arm_x;
     feedback.y = arm_y;
     feedback.theta = arm_ry;
     feed_pose.publish(feedback);
     ROS_INFO("Publishing");
     loop_rate.sleep();
  }
  
}
