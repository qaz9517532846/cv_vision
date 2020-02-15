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

cv::FileStorage fs("/home/scl/catkin_ws/src/vision/src/camera_para.yaml", cv::FileStorage::READ);

class ImageConverter
{
  ros::NodeHandle nh_;

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

    theta_1 = atan((float)(corner_position[0][1] - corner_position[1][1])/ (corner_position[0][0] - corner_position[1][0])) * 180 / 3.1415;
  
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
       double translation_mm_x = 0, translation_mm_y = 0, translation_mm_z = 0;
       double rotation_rad_x = 0, rotation_rad_y = 0, rotation_rad_z = 0;

       translation_mm_x = translation.at<double>(0, 0);
       translation_mm_y = translation.at<double>(1, 0);
       translation_mm_z = translation.at<double>(2, 0);

       rotation_rad_x = rotation.at<double>(0, 0);
       rotation_rad_y = rotation.at<double>(1, 0);
       rotation_rad_z = rotation.at<double>(2, 0);
       
       char translation_x[20], translation_y[20], translation_z[20];
       char rotation_x[20], rotation_y[20], rotation_z[20];

       sprintf(translation_x, "translation_x = %.3f ", translation_mm_x);
       sprintf(translation_y, "translation_y = %.3f ", translation_mm_y);
       sprintf(translation_z, "translation_z = %.3f ", translation_mm_z);

       sprintf(rotation_x, "rotation_x = %.3f ", rotation_rad_x);
       sprintf(rotation_y, "rotation_y = %.3f ", rotation_rad_y);
       sprintf(rotation_z, "rotation_z = %.3f ", rotation_rad_z);

       putText(cv_ptr->image, translation_x, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
       putText(cv_ptr->image, translation_y, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
       putText(cv_ptr->image, translation_z, cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

       putText(cv_ptr->image, rotation_x, cv::Point(300, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
       putText(cv_ptr->image, rotation_y, cv::Point(300, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
       putText(cv_ptr->image, rotation_z, cv::Point(300, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    }
    else
    {
       putText(cv_ptr->image, "NO DETECT", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    }

    // Update GUI Window
    cv::imshow(OPENCV_WINDOW, cv_ptr->image);
    //cv::imshow(OPENCV_WINDOW, gray_img);
    //cv::imshow(OPENCV_WINDOW, bin_img);
    cv::waitKey(10);
 
    // Output modified video stream
    image_pub_.publish(cv_ptr->toImageMsg());
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "vision_detect_3d");

  fs["camera_matrix"] >> cameraMatrix;
  fs["distortion_coefficients"] >> distCoeffs;

  ImageConverter ic;

  ros::spin();
  return 0;
}
