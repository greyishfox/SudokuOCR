#include "ocr.h"

OCR::OCR(){}

OCR::~OCR(){}

void OCR::getBoundingRect(cv::Mat trainingImage, cv::Mat thresholdImage, std::vector<std::vector<cv::Point>> cVector)
{
    cv::Mat roiImage;
    cv::Rect redBoundingBox;
    int inputDigit = 0;
    int cntr = 0;
    std::vector<int> validDigits = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};

    if(!m_classificationInputDigits.empty())
        m_classificationInputDigits.release();
    if(!m_trainingImageOutput.empty())
        m_trainingImageOutput.release();

    for(auto& el : cVector)
    {
        if(cv::contourArea(el) >= m_minContourArea && cv::contourArea(el) < m_maxContourArea)
        {
            redBoundingBox = cv::boundingRect(el);
            ++cntr;
            cv::rectangle(trainingImage, redBoundingBox, cv::Scalar(0,0,255), 2);
            roiImage = thresholdImage(redBoundingBox);
            cv::Mat resizedRoiImage;
            cv::resize(roiImage, resizedRoiImage, cv::Size(m_cellWidth,m_cellHeight)); // size(width, height)

            cv::imshow("Region of interest: ", resizedRoiImage);
            cv::imshow("Training numbers: ", trainingImage);

            // inputDigit captures char from user input
            inputDigit = cv::waitKey(0);
            // We only allow single digit numbers as input (char --> '0' instead of 0 (int))
            //if(inputDigit >= '0' && inputDigit <= '9')
            if(std::any_of(validDigits.begin(), validDigits.end(), [&inputDigit](int x) {return (x == inputDigit);}))
            {
                m_classificationInputDigits.push_back(inputDigit);
                cv::Mat floatImg;
                resizedRoiImage.convertTo(floatImg, CV_32FC1);
                cv::Mat flattenedImg = floatImg.reshape(1, 1);
                m_trainingImageOutput.push_back(flattenedImg);
            }
        }
    }
}

// Write classification input to file
void OCR::writeClassificationFile()
{
    // Convert the classification numbers to float --> required for KNN algorithm
    cv::Mat classificationInputToFloat;
    m_classificationInputDigits.convertTo(classificationInputToFloat, CV_32FC1);
    m_classificationInputDigits.release();
    // Create a classification file in ".xml" format ...
    cv::FileStorage fs_class(filename_class, cv::FileStorage::WRITE);

    // ... store the digits from the user input as float data types ...
    fs_class << "classificationDigits" << classificationInputToFloat;

    // ... and close the file
    fs_class.release();
}

// Write classification images to file
void OCR::writeTrainedImageFile()
{
    // Create a classification-image file in ".xml" format ...
    cv::FileStorage fs_images(filename_trained, cv::FileStorage::WRITE);

    // ... store the images used in user decision making as float data types ...
    fs_images << "trainedImages" << m_trainingImageOutput;
    m_trainingImageOutput.release();
    // ... and close the file
    fs_images.release();
}

bool OCR::checkIfFilesExists()
{
    // Try to open the trained and classification image file
    cv::FileStorage fs_class(filename_class, cv::FileStorage::READ);
    cv::FileStorage fs_trained(filename_trained, cv::FileStorage::READ);

    // Make sure the files exists, otherwise exit
    if(!fs_class.isOpened())
    {
        std::cout << "Warning: Classification file missing/not yet created!" << std::endl;
        return false;
    }
    else if(!fs_trained.isOpened())
    {
        std::cout << "Error: training images file missing/not yet created!" << std::endl;
        return false;
    }
    else
    {
        fs_class.release();
        fs_trained.release();
        return true;
    }
}

std::string OCR::train(std::vector<cv::Mat> labelTrain)
{

    // The classification data is read from the stored file
    // Once the classification and image files are created,
    // we do not need to create it everytime we run the program.
    // Therefore, we only read the files in future runs
    cv::Mat classificationImg;
    cv::Mat trainingImg;

    cv::FileStorage fs_class(filename_class, cv::FileStorage::READ);

    // Make sure the file exists, otherwise exit
    if(!fs_class.isOpened())
    {
        std::cout << "Error: Classification file not found!" << std::endl;
        // Exit failure: abnormal termination of the program ...
        // return 0 does not work since the function expects a vector of ints
        exit(1);
    }

    // Write the file content into the class member image
    fs_class["classificationDigits"] >> classificationImg;   // string in brackets must be in header of xml
    fs_class.release();

    cv::FileStorage fs_images(filename_trained, cv::FileStorage::READ);

    // Make sure the file exists, otherwise exit
    if(!fs_images.isOpened())
    {
        std::cout << "Error: training images file not found!" << std::endl;
        // Exit failure: abnormal termination of the program ...
        // return 0 does not work since the function expects a vector of ints
        exit(1);
    }

    // Write the file content into the class member image
    fs_images["trainedImages"] >> trainingImg;     // string in brackets must be in header of xml
    fs_images.release();

    std::cout << "files opened and read in..." << std::endl;

    // Training the KNearestNeighbor
    cv::Ptr<cv::ml::KNearest> knearest = cv::ml::KNearest::create();

    std::cout << "KNearest created...";

    // Set properties of KNearest
    knearest->setIsClassifier(true);
    knearest->setAlgorithmType(cv::ml::KNearest::Types::BRUTE_FORCE);
    knearest->setDefaultK(1);

    // Checks the matrix shape (rows, cols)
    // std::cout << "TrainingImage Dimension: (" << trainingImg.rows << "," <<
    //              trainingImg.cols << ")" << std::endl;
    // std::cout << "ClassificationDigits Dimension: (" << classificationImg.rows << "," <<
    //              classificationImg.cols << ")" << std::endl;

    // Train the knn algorithm
    knearest->train(trainingImg, cv::ml::ROW_SAMPLE, classificationImg);

    // Store the trained model
    knearest->save("../SudokuOCR/src/knn.yml");

    // This string holds the resulting numbers
    std::string detectedDigits;

    // Find the knearest neighbour for every image of the sudoku cells
    std::for_each(labelTrain.begin(), labelTrain.end(), [&](cv::Mat cImg)
    {
        // Prepare training image to be compatible with knearest
        // 1.) Convert to float
        cv::Mat floatCellImage;
        cImg.convertTo(floatCellImage, CV_32FC1);

        // 2.) Reshape or flatten respectively
        cv::Mat flattenedCellImage;
        flattenedCellImage = floatCellImage.reshape(1, 1);

        // Evaluate the digit by calling kNearest
        cv::Mat knnResult;
        float digit = knearest->findNearest(flattenedCellImage, knearest->getDefaultK(), knnResult);

        // Convert float to string
        detectedDigits += char(int(digit));
    });

    // Return a string of the detected images
    std::cout << "The detected digits are: " << detectedDigits << std::endl;
    return detectedDigits;
}
