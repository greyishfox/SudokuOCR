#ifndef OCR_H
#define OCR_H

#include "imageprocessing.h"
#include <opencv2/ml.hpp>

class OCR
{
private:
    cv::Mat classificationInputDigits;
    cv::Mat trainingImageOutput;
    const std::string filename_class = "../SudokuSolver/src/classificationDigits.xml";
    const std::string filename_trained = "../SudokuSolver/src/trainedImages.xml";
    const int m_cellWidth = 20;
    const int m_cellHeight = 30;
    const int m_maxContourArea = 280;
    const int m_minContourArea = 60;

public:
    // OCR();   // No constructor needed

    // Public member function
    void getBoundingRect(cv::Mat trainingImage, cv::Mat thresholdImage, std::vector<std::vector<cv::Point>> cVector);
    void writeClassificationFile();
    void writeTrainedImageFile();
    std::vector<int> train(std::vector<cv::Mat> labelTrain);
};

#endif // OCR_H
