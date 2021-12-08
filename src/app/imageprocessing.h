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
#include <map>

class ImageProcessing
{
private:
    // Holds contour of digit and coordinate in sudoku grid
    const int m_CellWidth = 20;
    const int m_CellHeight = 30;
    const int m_maxContourArea = 400;   // Initial value: 280
    const int m_minContourArea = 75;    // Initial value: 60

public:
    // ImageProcessing(); // Constructor not used
    // Public member functions
    cv::Mat imagePreprocessing(const cv::Mat sourceImage, const int thresholdType);

    std::vector<cv::Point> getFrameContour(cv::Mat thresholdImg);

    std::vector<cv::Point> findFrameCorners(cv::Mat sourceImage, std::vector<cv::Point> frameContour);

    cv::Mat getTopView(const cv::Mat sourceImage, std::vector<cv::Point> frameCorners);

    std::vector<cv::Mat> extractCells(cv::Mat thresholdImg);

    std::vector<cv::Mat> selectCellsWithDigit(std::vector<cv::Mat> cellImages);
};

#endif // IMAGEPROCESSING_H
