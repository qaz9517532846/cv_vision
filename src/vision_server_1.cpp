#include <ros/ros.h>
#include <std_msgs/String.h>
#include <sstream>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include "vision/robot2vision.h"

cv::Mat gray_img, bin_img;
cv::Mat cameraMatrix, distCoeffs;

cv_bridge::CvImagePtr cv_ptr;

int robot_command;
int corner_position[4][4];

int object;

double theta, theta_1;
double robot_mm_x = 0, robot_mm_y = 0, robot_mm_theta = 0;

cv::FileStorage fs("/home/nvidia/catkin_ws/src/vision/src/camera_para.yaml", cv::FileStorage::READ);

class camera_server{
public:
    ros::NodeHandle nh;
};

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

  fs["camera_matrix"] >> cameraMatrix;
  fs["distortion_coefficients"] >> distCoeffs;

  // image process
  if(robot_command == 1)
  {
     cv::cvtColor(cv_ptr->image, gray_img, CV_BGR2GRAY);
     cv::threshold(gray_img, bin_img, 100, 255, cv::THRESH_BINARY_INV);

     std::vector<std::vector<cv::Point> > contours;
     cv::findContours(bin_img, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

     double max_area = 0;
     int max_area_contour = -1;
     std::vector<cv::Moments> mu(contours.size());
     std::vector<cv::Point2f> mc(contours.size());
     std::vector<cv::RotatedRect>    minRect(contours.size());
     cv::Point2f rect_points[4]; 

     object = 0;
     for (int i = 0; i < contours.size(); i++)
     {
         double area = contourArea(contours.at(i));
         if (area > 10000 && area < 50000)
         {
            //std::cout << "area = " << area << std::endl;

	    mu[i] = cv::moments(contours[i], false);
	    mc[i] = cv::Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
	    //std::cout << "center = " << mc[i] << std::endl;

            minRect[i] = cv::minAreaRect(cv::Mat(contours[i]));
            minRect[i].points(rect_points);

            for(int j = 0; j < 4; j++)
            {
               cv::line(cv_ptr->image, rect_points[j], rect_points[(j+1)%4], cv::Scalar(0, 255, 0), 4, 5);
               corner_position[j][0] = rect_points[j].x;
	       corner_position[j][1] = rect_points[j].y;
            }
   
            cv::Mat pointsf;
	    cv::Mat(contours[i]).convertTo(pointsf, CV_32F);
            cv::Vec4f fitline;
	    cv::fitLine(pointsf, fitline, CV_DIST_L2, 0, 0.01, 0.01);
	    int lefty = int((-fitline[2] * fitline[1] / fitline[0]) + fitline[3]);
	    int righty = int(((1280 - fitline[2]) * fitline[1] / fitline[0]) + fitline[3]);
            theta = atan((float)(righty - lefty)/ (cv_ptr->image.cols - 1)) * 180 / 3.1415;
            //std::cout << "theta = " << theta << std::endl;
            object ++;        
         }
     }

     //std::cout << "Object = " << object << std::endl;

     theta_1 = atan((float)(corner_position[0][1] - corner_position[1][1])/ (corner_position[0][0] - corner_position[1][0])) * 180 / 3.1415;

     //std::cout << "theta_1 = " << theta_1 << std::endl;
  
     std::vector<cv::Point3d> corner_pts_3d(4);
     if ((theta - theta_1) < -80 || (theta - theta_1) > 80)
     {
        corner_pts_3d[0] = (cv::Point3d(double(-35), float(3.5), 0));
        corner_pts_3d[1] = (cv::Point3d(double(-35), float(-3.5), 0));
        corner_pts_3d[2] = (cv::Point3d(double(35), float(-3.5), 0));
        corner_pts_3d[3] = (cv::Point3d(double(35), float(3.5), 0));
     }
     else
     {
        corner_pts_3d[0] = (cv::Point3d(double(35), float(3.5), 0));
        corner_pts_3d[1] = (cv::Point3d(double(-35), float(3.5), 0));
        corner_pts_3d[2] = (cv::Point3d(double(-35), float(-3.5), 0));
        corner_pts_3d[3] = (cv::Point3d(double(35), float(-3.5), 0));
     }
    
     std::vector<cv::Point2f> obj_corner_pts(4);
     obj_corner_pts[0] = cvPoint(corner_position[0][0], corner_position[0][1]);
     obj_corner_pts[1] = cvPoint(corner_position[1][0], corner_position[1][1]);
     obj_corner_pts[2] = cvPoint(corner_position[2][0], corner_position[2][1]);
     obj_corner_pts[3] = cvPoint(corner_position[3][0], corner_position[3][1]);
    
     if (object != 0)
     {
        cv::Mat rotation, translation;
        solvePnP(corner_pts_3d, obj_corner_pts, cameraMatrix, distCoeffs, rotation, translation);
  
        //convert to centimeters
        double translation_mm_x = 0, translation_mm_y = 0;
        translation_mm_x = translation.at<double>(0, 0);
        translation_mm_y = translation.at<double>(1, 0);

        if (theta > 0)
        {
          robot_mm_x = 0.255 - (translation_mm_y / 1000) + 0.02 * cos(-theta * 3.1415 / 180 + 1.5708) + 0.007; 
          robot_mm_y = -(translation_mm_x / 1000) + 0.02 * sin((-theta * 3.1415 / 180) + 1.5708);
          robot_mm_theta = (theta * -3.1415 / 180) + 1.5708;
        }
        else
        {
          robot_mm_x = 0.255 - (translation_mm_y / 1000) + 0.02 * cos(-theta * 3.1415 / 180 - 1.5708) + 0.007; 
          robot_mm_y = -(translation_mm_x / 1000) - 0.02 * sin((-theta * 3.1415 / 180) - 1.5708);
          robot_mm_theta = (theta * -3.1415 / 180) - 1.5708;
        }

        if (robot_mm_y < 0)
        {
           robot_mm_y = robot_mm_y - 0.014;
        }
       
        char translation_x[20], translation_y[20], theta_z[20];
        char robot_arm_x[20], robot_arm_y[20], robot_arm_theta[20];

        sprintf(translation_x, "vision_x = %.3f ", translation_mm_x);
        sprintf(translation_y, "vision_y = %.3f ", translation_mm_y);
        sprintf(theta_z, "vision_theta = %.3f ", (theta * 3.1415 / 180));

        sprintf(robot_arm_x, "Robot_x = %.3f ", robot_mm_x);
        sprintf(robot_arm_y, "Robot_y = %.3f ", robot_mm_y);
        sprintf(robot_arm_theta, "Robot_theta = %.3f ", robot_mm_theta);

        putText(cv_ptr->image, translation_x, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        putText(cv_ptr->image, translation_y, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        putText(cv_ptr->image, theta_z, cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        
        putText(cv_ptr->image, robot_arm_x, cv::Point(300, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        putText(cv_ptr->image, robot_arm_y, cv::Point(300, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        putText(cv_ptr->image, robot_arm_theta, cv::Point(300, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
     }
     else
     {
        putText(cv_ptr->image, "NO DETECT", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        ROS_INFO("object position : x = null, y = null, theta = null");
        ROS_INFO("No detected");
     }
  }
 
  cv::imwrite("result.png", cv_ptr->image);
}

bool robot_connect_vision(vision::robot2vision::Request  &req,
                          vision::robot2vision::Response &res)
{
  robot_command = req.command;

  ROS_INFO("recieved_command = %ld", (long int)req.command);

  camera_server cam_server;
  image_transport::ImageTransport it(cam_server.nh);
  image_transport::Subscriber sub = it.subscribe("/camera/image_raw", 10, image_call);

  ros::Rate loop_rate(1);
  for(int i = 0; i < 3; i++)
  {
     ros::spinOnce();
     loop_rate.sleep();
  }

  res.num = object;
  res.robot_x = robot_mm_x;
  res.robot_y = robot_mm_y;
  res.robot_theta = robot_mm_theta;

  ROS_INFO("num = %ld", (long int)res.num);
  ROS_INFO("robot_x = %f, robot_y = %f, robot_theta = %f", res.robot_x, res.robot_y, res.robot_theta);

  return true;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "vision_server_1");

  camera_server cam_server;

  ros::ServiceServer service = cam_server.nh.advertiseService("robot_vision", robot_connect_vision);
  
  ROS_INFO("Ready to camera capcture");
  ros::spin();
}
