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

    // Plot threshold image
    cv::imshow("Threshold image", thresholdImg);
    cv::waitKey(0);

    // Return a treshold version of the original image
    return thresholdImg;
}

cv::Mat ImageProcessing::preprocWithGauss(const cv::Mat sourceImage, const int thresholdType)
{
    // Change image to greyscale
    cv::Mat greyscale;
    cv::cvtColor(sourceImage, greyscale, cv::COLOR_RGB2GRAY);

    cv::Mat gaussBlurr;
    cv::GaussianBlur(greyscale, gaussBlurr, cv::Size(5, 5), 0);

    cv::Mat thresholdImg;
    const double maxBinaryValue = 255;
    cv::adaptiveThreshold(gaussBlurr, thresholdImg, maxBinaryValue,
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          thresholdType, 21, 1);
    cv::imshow("Adaptive Thresh", thresholdImg);
    cv::waitKey(0);

    return thresholdImg;
}

cv::Mat ImageProcessing::preprocWithGauss2(const cv::Mat sourceImage, const int thresholdType)
{
    // Change image to greyscale
    cv::Mat greyscale;
    cv::cvtColor(sourceImage, greyscale, cv::COLOR_RGB2GRAY);
    cv::Mat gaussBlurr;
    cv::GaussianBlur(greyscale, gaussBlurr, cv::Size(5, 5), 1, 1);
    cv::Mat thresholdImg;
    const double maxBinaryValue = 255;
    cv::adaptiveThreshold(gaussBlurr, thresholdImg, maxBinaryValue,
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          thresholdType, 57, 17); // 57, 17

    // Plot adaptive threshold image
    // cv::imshow("Adaptive Thresh", thresholdImg);
    // cv::waitKey(0);

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

        // Plot grid corners
        cv::Mat poly_img = sourceImage.clone();
        cv::drawContours(poly_img, cv::Mat(polygon), -1, cv::Scalar(0, 0, 255), 3, 8);
        cv::imshow("Corner image", poly_img);
        cv::waitKey(0);
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
    //std::vector<cv::Point> imDimension = {cv::Point(imWidth,0), cv::Point(0,0),
                                          //cv::Point(0, imHeight), cv::Point(imWidth, imHeight)};
    std::vector<cv::Point> imDimension = {cv::Point(0,0), cv::Point(0,imHeight),
                                          cv::Point(imWidth, imHeight), cv::Point(imWidth, 0)};
    // Apply perspective transform
    cv::Mat perspective = cv::findHomography(frameCorners, imDimension, cv::RANSAC);
    cv::Mat topViewImage;
    cv::warpPerspective(sourceImage, topViewImage, perspective, cv::Size(imWidth,imHeight));

    return topViewImage;
}

std::vector<cv::Mat> ImageProcessing::extractCells(cv::Mat thresholdImg)
{
    std::vector<cv::Mat> allCellImages;
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
            allCellImages.push_back(thresholdImg(roi));
        }
    }
    return allCellImages;
}

std::vector<cv::Mat> ImageProcessing::selectCellsWithDigit(std::vector<cv::Mat> cellImages)
{
    // TODO: return only cell images with digits
    std::vector<cv::Mat> cellImagesWithDigit;
    int cellCntr = 0;
    std::vector<cv::Point> contourInCell;

    // Define parameters for "findContour" function
    std::vector<std::vector<cv::Point>> cVector;
    //std::vector<cv::Vec4i> hierarchy;

    std::for_each(cellImages.begin(), cellImages.end(), [&](cv::Mat cImg)
    {
       int tmp = cellCntr;
       cv::findContours(cImg, cVector, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
       for(auto &el : cVector)
       {
           // std::cout << "Contour Area: " << cv::contourArea(el) << std::endl;
           if(cv::contourArea(el) > m_minContourArea && cv::contourArea(el) < m_maxContourArea)
           {               
                cv::Rect boundingBox = cv::boundingRect(el);
                cv::Mat roiImg = cImg(boundingBox);
                cv::Mat resizedCImg;
                cv::resize(roiImg, resizedCImg, cv::Size(m_CellWidth, m_CellHeight));
                cellImagesWithDigit.push_back(resizedCImg);
                // Check cell images:
                // cv::imshow("Cell image with digit: ", resizedCImg);
                // cv::waitKey(0);
                cellsWithNumbers.push_back(true);
                cellCntr++;
           }
       }
       if(cellCntr > tmp)
           tmp = cellCntr;
       else
           cellsWithNumbers.push_back(false);
       // Remove all contour elements from vector for next loop
       cVector.clear();
    });

    for(auto el : cellsWithNumbers){
        std::cout << el << " ";
    }
    std::cout << std::endl;

    return cellImagesWithDigit;
}

std::vector<bool> ImageProcessing::getCellsWithNumbers(void)
{
    return cellsWithNumbers;
}

void ImageProcessing::drawMissingDigits(cv::Mat topViewImage, const std::vector<bool> cellWithDigit, std::vector<int> sudoku)
{
    // Calculate digit coordinate for varying digit sizes
    const double fontSize = 1.4;
    const int xCoord = 12 - (fontSize - 1.6) * 10;
    const int yCoord = 46 + (fontSize - 1.6) * 10;
    int newXcoord = 0;
    int newYcoord = 0;
    int colCounter = 0;
    int rowCounter = 0;

    // Transform vector of int to vector of string
    std::vector<std::string> missingDigits;
    std::transform(sudoku.begin(), sudoku.end(), std::back_inserter(missingDigits),
                   [](int digit){return std::to_string(digit);});

    std::vector<std::string>::iterator itr = missingDigits.begin();
    // Draw the missing numbers to image
    std::for_each(cellWithDigit.begin(), cellWithDigit.end(), [&](bool cellContent){
        if(cellContent == false)
        {
            newXcoord = xCoord + (colCounter * static_cast<float>(topViewImage.cols)/9);
            newYcoord = yCoord + (rowCounter * static_cast<float>(topViewImage.rows)/9);
            cv::putText(topViewImage, *itr, cv::Point(newXcoord,newYcoord), cv::FONT_HERSHEY_DUPLEX, fontSize, cv::Scalar(0,0,255), 2);
        }

        if(colCounter == 8)
        {
            colCounter = -1;
            rowCounter++;
        }
        ++colCounter;
        ++itr;
    });
}
