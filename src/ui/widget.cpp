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
    connect(ui->comboBox, SIGNAL(currentIndexChanged(QString)), this, SLOT(reset()));
    this->setFixedSize(1050,781);
}

Widget::~Widget()
{
    delete ui;
}

void Widget::plotOrigImg()
{
    // Read in the image which is selected in the combo box
    std::cout << ui->comboBox->currentText().toStdString() << std::endl;
    if(ui->comboBox->currentText().toStdString() == "Sudoku Puzzle 01")
        m_origImg = cv::imread("../SudokuOCR/img/sudoku_sample_image1.jpg");
    else if(ui->comboBox->currentText().toStdString() == "Sudoku Puzzle 02")
        m_origImg = cv::imread("../SudokuOCR/img/sudoku_sample_image2.jpg");
    else if(ui->comboBox->currentText().toStdString() == "Sudoku Puzzle 03")
        m_origImg = cv::imread("../SudokuOCR/img/sudoku_sample_image3.jpg");
    else if(ui->comboBox->currentText().toStdString() == "Sudoku Puzzle 04")
        m_origImg = cv::imread("../SudokuOCR/img/sudoku_sample_image4.jpg");
    else if(ui->comboBox->currentText().toStdString() == "Worlds Hardest Sudoku")
        m_origImg = cv::imread("../SudokuOCR/img/worldsHardestSudoku.png");

    // Check if image was successfully loaded
    if(m_origImg.empty())
    {
        std::cout << "Error, Image not found!" << std::endl;
        exit(1);
    }
    else
    {
        // Resize the image and plot it using Pixmap
        cv::resize(m_origImg, m_origImg, cv::Size(512, 481), 0, 0, cv::INTER_LINEAR);
        m_displayOrigImage = QImage((const unsigned char*) (m_origImg.data), m_origImg.cols,
                                  m_origImg.rows, m_origImg.step, QImage::Format_RGB888);
        ui->lbl_origImg->setPixmap(QPixmap::fromImage(m_displayOrigImage));
    }
}

void Widget::plotSolvImg()
{
    // Get time at solver start
    auto start = std::chrono::high_resolution_clock::now();

    // Exit if no image is loaded
    if(m_origImg.empty())
    {
        std::cout << "Error, Image not found!" << std::endl;
        exit(1);
    }
    else
    {
        // Image processing on Sudoku picture
        cv::Mat thresholdImg = imgProcess.imagePreprocessing(m_origImg, cv::THRESH_BINARY_INV);

        std::vector<cv::Point> frameContour = imgProcess.getFrameContour(thresholdImg);

        std::vector<cv::Point> frameCorners = imgProcess.findFrameCorners(m_origImg, frameContour);

        cv::Mat topView = imgProcess.getTopView(m_origImg, frameCorners);

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

        // Perform the algorithm --> Backtracking
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
        m_displaySolvImage = QImage((const unsigned char*) (topView.data),topView.cols,
                                  topView.rows, topView.step, QImage::Format_RGB888);
        ui->lbl_solvImg->setPixmap(QPixmap::fromImage(m_displaySolvImage));

        m_solvedImg = topView.clone();
    }

    // Get time at solver ending
    auto finish = std::chrono::high_resolution_clock::now();

    // Print the elapsed time
    std::cout << "Time elapsed: " << (finish - start).count()*1e-9 << "s" << std::endl;
}

void Widget::saveImg()
{
    if(m_solvedImg.empty())
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
    qDebug() << "Image saved as: " << saveFileName;

    cv::imwrite(saveFileName.toStdString(), m_solvedImg);
}

void Widget::reset()
{
    // Reset member variables and clear labels
    m_origImg.release();
    m_solvedImg.release();
    m_displayOrigImage = QImage();
    m_displaySolvImage = QImage();
    ui->lbl_origImg->clear();
    ui->lbl_solvImg->clear();

    // Reset image processing object
    (&imgProcess)->~ImageProcessing();
    new (&imgProcess) ImageProcessing();

    // Reset solver object
    (&mysolver)->~Solver();
    new (&mysolver) Solver();
}
