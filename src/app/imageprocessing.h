#ifndef IMAGEPROCESSING_H
#define IMAGEPROCESSING_H

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
// OpenCV libraries
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>  // findHomography()

class ImageProcessing
{
public:
    // ImageProcessing(); // Constructor not used
    // Public member functions
    cv::Mat imagePreprocessing(const cv::Mat sourceImage, const int thresholdType);

    std::vector<cv::Point> getFrameContour(cv::Mat thresholdImg);

    std::vector<cv::Point> findFrameCorners(cv::Mat sourceImage, std::vector<cv::Point> frameContour);

    cv::Mat getTopView(const cv::Mat sourceImage, std::vector<cv::Point> frameCorners);

    std::vector<cv::Mat> extractCells(cv::Mat thresholdImg);
};

#endif // IMAGEPROCESSING_H
