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
    }
    else
    {
        cv::Mat thresholdImg = imgProcess.imagePreprocessing(origImg, cv::THRESH_BINARY_INV);
        //cv::imshow("Threshold", thresholdImg);

        std::vector<cv::Point> frameContour = imgProcess.getFrameContour(thresholdImg);

        std::vector<cv::Point> frameCorners = imgProcess.findFrameCorners(origImg, frameContour);

        cv::Mat topView = imgProcess.getTopView(origImg, frameCorners);

        std::vector<cv::Mat> cellImages = imgProcess.extractCells(thresholdImg);

        displaySolvImage = QImage((const unsigned char*) (topView.data),topView.cols,
                                  topView.rows, topView.step, QImage::Format_RGB888);
        ui->lbl_solvImg->setPixmap(QPixmap::fromImage(displaySolvImage));
    }
}



