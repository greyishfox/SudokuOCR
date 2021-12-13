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
    connect(ui->btn_save, SIGNAL(clicked()), this, SLOT(saveImg()));
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
    // Get time at solver start
    auto start = std::chrono::high_resolution_clock::now();

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
        std::string digits = myOCR.train(cellImagesWithDigit);

        std::vector<int> puzzleToSolve = mysolver.createSudokuPuzzle(imgProcess.getCellsWithNumbers(), digits);

        mysolver.printSudoku(puzzleToSolve);

        // -------------------- Solve the sudoku puzzle -------------------- //

        // Initial position to start the solving algorithm
        int row = 0;
        int col = 0;

        // Perform the algorithm
        if(mysolver.solve(puzzleToSolve, row, col))
        {
            if(mysolver.checker(puzzleToSolve, row, col))
            {
                mysolver.printSudoku(puzzleToSolve);
                std::cout << "Congratulations, you solved the sudoku puzzle!" << std::endl;
            }
            else
                std::cout << "Error: Backtracking leads to a wrong result" << std::endl;
        }
        else
            std::cout << "Error: Sudoku cannot be solved" << std::endl;

        imgProcess.drawMissingDigits(topView, imgProcess.getCellsWithNumbers(), puzzleToSolve);

        // Print solved Sudoku image
        displaySolvImage = QImage((const unsigned char*) (topView.data),topView.cols,
                                  topView.rows, topView.step, QImage::Format_RGB888);
        ui->lbl_solvImg->setPixmap(QPixmap::fromImage(displaySolvImage));

        solvedImg = topView.clone();
    }

    // Get time at solver ending
    auto finish = std::chrono::high_resolution_clock::now();

    // Print the elapsed time
    std::cout << "Time elapsed: " << (finish - start).count()*1e-9 << "s" << std::endl;
}

void Widget::saveImg()
{
    if(solvedImg.empty())
    {
        std::cout << "Error, Image not found!" << std::endl;
        exit(1);
    }

    // Prepair variables to hold parts of the save file name
    QString saveDir = "../SudokuOCR/img/SolvedSudokuImages/";
    QString fileName = "_Sudoku.jpg";
    QString saveFileName;

    // Get current time and date and store it in a QString
    QDateTime local(QDateTime::currentDateTime());
    QString timeStamp = local.date().toString() + local.time().toString();

    // Replace unnecessary characters
    std::vector<std::pair<QString, QString>> replace{{ ":", "-" },{ ".", "" },{ " ", "-" }};
    for(auto& el : replace)
    {
        timeStamp.replace(el.first, el.second);
    }

    // Check concatenated file name
    saveFileName = saveDir + timeStamp + fileName;
    qDebug() << "Save file name is: " << saveFileName;

    cv::imwrite(saveFileName.toStdString(), solvedImg);
}



