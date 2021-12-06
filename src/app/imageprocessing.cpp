#include "imageprocessing.h"

cv::Mat ImageProcessing::imagePreprocessing(const cv::Mat sourceImage, const int thresholdType)
{
    // Change image to greyscale
    cv::Mat greyscale;
    cv::cvtColor(sourceImage, greyscale, cv::COLOR_RGB2GRAY);

    // Apply simple treshold filter
    cv::Mat thresholdImg;
    const double thresholdValue = 128;
    const double maxBinaryValue = 255;
    cv::threshold(greyscale, thresholdImg, thresholdValue, maxBinaryValue, thresholdType);

    // Return a treshold version of the original image
    return thresholdImg;
}

std::vector<cv::Point> ImageProcessing::getFrameContour(cv::Mat thresholdImg)
{
    // Define parameters for "findContour" function
    std::vector<std::vector<cv::Point>> cVector;
    std::vector<cv::Vec4i> hierarchy;

    // Detect contours
    cv::findContours(thresholdImg, cVector, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    // The largest contour area found in the image should be the frame of the sudoku
    double largest_area = 0;
    int index = -1;
    for(int i = 0; i < int(cVector.size()); i++)
    {
        double contour_area = cv::contourArea(cVector[i]);
        if(contour_area > largest_area)
        {
            largest_area = contour_area;
            index = i;
        }
    }

    return cVector[index];
}

std::vector<cv::Point> ImageProcessing::findFrameCorners(cv::Mat sourceImage, std::vector<cv::Point> frameContour)
{
    // Define parameters for approxPolyDP funtion
    std::vector<cv::Point> polygon;
    auto epsilon = 0.02*cv::arcLength(cv::Mat(frameContour), true);

    // Returns an approximation of the contour using less points
    cv::approxPolyDP(cv::Mat(frameContour), polygon, epsilon, true);

    // Check if the 4 frame corners have been found
    if(polygon.size() == 4)
    {
        std::cout << "Frame corners found!" << std::endl;
        // cv::Mat poly_img = sourceImage.clone();
        // cv::drawContours(poly_img, cv::Mat(polygon), -1, cv::Scalar(0, 0, 255), 3, 8);
        // cv::imshow("Corner image", poly_img);
        // waitKey(0);
    }
    else
        std::cout << "Error: Contour has more than 4 corners!" << std::endl;

    return polygon;
}

// Scale the sudoku grid to the original image size and apply a perspective transform
cv::Mat ImageProcessing::getTopView(const cv::Mat sourceImage, std::vector<cv::Point> frameCorners)
{
    // Get image dimension
    const int imWidth = sourceImage.rows;
    const int imHeight = sourceImage.cols;

    // Define a vector holding the corners of the original image
    std::vector<cv::Point> imDimension = {cv::Point(imWidth,0), cv::Point(0,0),
                                          cv::Point(0, imHeight), cv::Point(imWidth, imHeight)};

    // Apply perspective transform
    cv::Mat perspective = cv::findHomography(frameCorners, imDimension, cv::RANSAC);
    cv::Mat topViewImage;
    cv::warpPerspective(sourceImage, topViewImage, perspective, cv::Size(imWidth,imHeight));

    return topViewImage;
}

std::vector<cv::Mat> ImageProcessing::extractCells(cv::Mat thresholdImg)
{
    std::vector<cv::Mat> cell_images;
    const int cell_width = thresholdImg.cols/9;
    const int cell_height = thresholdImg.rows/9;
    int x0, y0, x1, y1;
    cv::Rect roi(0,0,0,0);

    for(int i = 0; i < 9; i++)
    {
        for(int j = 0; j < 9; j++)
        {
            x0 = j*cell_width+8;
            x1 = cell_width-10;
            y0 = i*cell_height+10;
            y1 = cell_height-10;
            roi = cv::Rect(x0, y0, x1, y1);
            cell_images.push_back(thresholdImg(roi));
        }
    }
    return cell_images;
}
