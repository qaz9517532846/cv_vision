#include <ros/ros.h>
#include <std_msgs/String.h>
#include <sstream>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

static const std::string OPENCV_WINDOW = "Image window";

cv::Mat gray_img, bin_img;
cv::Mat cameraMatrix, distCoeffs;

int corner_position[4][4];
double theta, theta_1;

int system_start = 0;

cv::FileStorage fs("/home/scl/sclagv_ws/src/cv_vision/src/ost.yaml", cv::FileStorage::READ);

int j = 0;

void chatterCallback(const std_msgs::String::ConstPtr& msg)
{ 
  if (msg->data == "vision system start")
  {
     ROS_INFO("%s", "I recieved command from vision system, system will action");
     system_start = 1;
  }
  else
  {
     ROS_INFO("%s", "I not recieved command from vision system, system wait");
     system_start = 0;
  }
}

class ImageConverter
{
  ros::NodeHandle nh_;
  ros::Publisher send_command;
  ros::Publisher send_obj_pose;

  ros::Subscriber recieve_command; 

  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;

public:
  ImageConverter()
    : it_(nh_)
  {
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("/camera/image_raw", 1, &ImageConverter::imageCb, this);
    image_pub_ = it_.advertise("/image_converter/output_video", 1);

    cv::namedWindow(OPENCV_WINDOW);
  }

  ~ImageConverter()
  {
    cv::destroyWindow(OPENCV_WINDOW);
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
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

    int object = 0;
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
    
    //std::cout << "corner_pts" << obj_corner_pts[0] << std::endl;
    //std::cout << "corner_pts" << obj_corner_pts[1] << std::endl;
    //std::cout << "corner_pts" << obj_corner_pts[2] << std::endl;
    //std::cout << "corner_pts" << obj_corner_pts[3] << std::endl;

    
    if (object != 0)
    {
       cv::Mat rotation, translation;
       solvePnP(corner_pts_3d, obj_corner_pts, cameraMatrix, distCoeffs, rotation, translation);

       //convert to centimeters
       double translation_mm_x = 0, translation_mm_y = 0;
       translation_mm_x = translation.at<double>(0, 0);
       translation_mm_y = translation.at<double>(1, 0);

       double robot_mm_x = 0, robot_mm_y = 0, robot_mm_theta = 0;

       if (theta > 0)
       {
          robot_mm_x = 0.25 - (translation_mm_y / 1000) + 0.02 * sin((theta * 3.1415 / 180));
          robot_mm_y = -(translation_mm_x / 1000) + 0.02 * cos((theta * 3.1415 / 180));
          robot_mm_theta = (theta * -3.1415 / 180) + 1.5708;
       }
       else
       {
          robot_mm_x = 0.25 - (translation_mm_y / 1000) - 0.02 * sin((theta * 3.1415 / 180));
          robot_mm_y = -(translation_mm_x / 1000) - 0.02 * cos((theta * 3.1415 / 180));
          robot_mm_theta = (theta * -3.1415 / 180) - 1.5708;
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

       recieve_command = nh_.subscribe("send_to_vision", 1000, chatterCallback);

       send_command = nh_.advertise<std_msgs::String>("string_command", 1000);
       send_obj_pose = nh_.advertise<std_msgs::String>("obj_pose", 1000);

       std_msgs::String send_robot_msg;
       std_msgs::String object_position;

       std::stringstream ss_send_obj_position;

       // send to robot arm
       if(system_start == 1)
       {
          //ss_send_command << "commmand to Robot arm";
          ss_send_obj_position << robot_mm_x << "," << robot_mm_y << "," << robot_mm_theta;
  
          send_robot_msg.data = "commmand to Robot arm";
          object_position.data = ss_send_obj_position.str();
          
          ROS_INFO("%s", send_robot_msg.data.c_str());
          ROS_INFO("%s", object_position.data.c_str());
   
          send_command.publish(send_robot_msg);
          send_obj_pose.publish(object_position);  
       }
       else
       {
          //ss_send_command << "stop commmand to Robot arm";
          send_robot_msg.data = "stop commmand to Robot arm";
          send_command.publish(send_robot_msg);
       }
    }
    else
    {
       putText(cv_ptr->image, "NO DETECT", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    }

    // Update GUI Window
    cv::imshow(OPENCV_WINDOW, cv_ptr->image);


    if(j < 100)
    {
       char store_local[80];
       sprintf(store_local, "/home/nvidia/catkin_ws/src/vision/52_36_-60/result_%d.png", j);
       cv:imwrite(store_local, cv_ptr->image);
       j++;
    }

    //cv::imshow(OPENCV_WINDOW, gray_img);
    //cv::imshow(OPENCV_WINDOW, bin_img);
    cv::waitKey(10);
 
    // Output modified video stream
    image_pub_.publish(cv_ptr->toImageMsg());
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "vision_detect");

  fs["camera_matrix"] >> cameraMatrix;
  fs["distortion_coefficients"] >> distCoeffs;

  ImageConverter ic;

  ros::spin();
  return 0;
}
