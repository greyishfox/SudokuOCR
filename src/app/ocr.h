#ifndef OCR_H
#define OCR_H

#include "imageprocessing.h"
#include <opencv2/ml.hpp>

class OCR
{
private:
    // Member variables
    cv::Mat m_classificationInputDigits;
    cv::Mat m_trainingImageOutput;
    const std::string filename_class = "../SudokuOCR/src/classificationDigits.xml";
    const std::string filename_trained = "../SudokuOCR/src/trainedImages.xml";
    const int m_cellWidth = 20;
    const int m_cellHeight = 30;
    const int m_maxContourArea = 1000;
    const int m_minContourArea = 60;

public:
    OCR(); // Constructor
    ~OCR(); // Destructor

    /* ----------------------- Public member functions ----------------------- */
    void getBoundingRect(cv::Mat trainingImage, cv::Mat thresholdImage, std::vector<std::vector<cv::Point>> cVector);

    void writeClassificationFile();

    void writeTrainedImageFile();

    bool checkIfFilesExists();

    // Train the KNN algorithm and return a string with the detected digits
    std::string train(std::vector<cv::Mat> labelTrain);
};

#endif // OCR_H
