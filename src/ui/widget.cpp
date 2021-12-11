#include "widget.h"
#include "./ui_widget.h"

Widget::Widget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Widget)
{
    ui->setupUi(this);
    this->setWindowTitle("Soduku Solver");
    connect(ui->btn_showImage, SIGNAL(clicked()), this, SLOT(plotOrigImg()));
    connect(ui->btn_solution, SIGNAL(clicked()), this, SLOT(plotSolvImg()));
}

Widget::~Widget()
{
    delete ui;
}

void Widget::plotOrigImg()
{
    origImg = cv::imread("sudoku_sample_image.jpeg");
    if(origImg.empty())
    {
        std::cout << "Error, Image not found!" << std::endl;
    }
    else
    {
        cv::resize(origImg, origImg, cv::Size(512, 481), 0, 0, cv::INTER_LINEAR);
        displayOrigImage = QImage((const unsigned char*) (origImg.data), origImg.cols,
                                  origImg.rows, origImg.step, QImage::Format_RGB888);
        ui->lbl_origImg->setPixmap(QPixmap::fromImage(displayOrigImage));
    }
}

void Widget::plotSolvImg()
{
    if(origImg.empty())
    {
        std::cout << "Error, Image not found!" << std::endl;
        exit(1);
    }
    else
    {
        cv::Mat thresholdImg = imgProcess.imagePreprocessing(origImg, cv::THRESH_BINARY_INV);
        //cv::imshow("Threshold", thresholdImg);

        std::vector<cv::Point> frameContour = imgProcess.getFrameContour(thresholdImg);

        std::vector<cv::Point> frameCorners = imgProcess.findFrameCorners(origImg, frameContour);

        cv::Mat topView = imgProcess.getTopView(origImg, frameCorners);

        //cv::Mat newThresholdImg = imgProcess.imagePreprocessing(topView, cv::THRESH_BINARY);
        cv::Mat newThresholdImg = imgProcess.preprocWithGauss2(topView, cv::THRESH_BINARY_INV);
        std::vector<cv::Mat> cellImages = imgProcess.extractCells(newThresholdImg);

        std::vector<cv::Mat> cellImagesWithDigit = imgProcess.selectCellsWithDigit(cellImages);

        // If the classification and training files for kNearest do not exist, create them
        if(!myOCR.checkIfFilesExists())
        {
            // Read the OCR trainig image
            cv::Mat trainImg = cv::imread("OCR_training_digits02.PNG");

            if(trainImg.empty())
            {
                std::cout << "Error, OCR training set image not found!" << std::endl;
                exit(1);
            }

            // Prepare parameters for the OCR digit assignment function (uses bounding box)
            //cv::Mat trainImgThreshold = imgProcess.imagePreprocessing(trainImg, cv::THRESH_BINARY_INV);
            cv::Mat trainImgThreshold = imgProcess.preprocWithGauss(trainImg, cv::THRESH_BINARY_INV);
            cv::Mat trainImgContour = trainImgThreshold.clone();
            std::vector<std::vector<cv::Point>> contourTrain;
            std::vector<cv::Vec4i> hierarchy;
            cv::findContours(trainImgContour, contourTrain, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

            // Assign an input key to the shown digits in the training image marked by the boundingbox
            myOCR.getBoundingRect(trainImg, trainImgThreshold, contourTrain);

            // Create classification and training files and store them in a predefined folder
            myOCR.writeClassificationFile();
            myOCR.writeTrainedImageFile();
        }

        // Run the training sequence for the kNearest
        std::cout << "Training kNearest..." << std::endl;
        myOCR.train(cellImagesWithDigit);


        displaySolvImage = QImage((const unsigned char*) (topView.data),topView.cols,
                                  topView.rows, topView.step, QImage::Format_RGB888);
        ui->lbl_solvImg->setPixmap(QPixmap::fromImage(displaySolvImage));
    }
}



