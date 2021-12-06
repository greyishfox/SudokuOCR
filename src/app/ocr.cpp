#include "ocr.h"

void OCR::getBoundingRect(cv::Mat trainingImage, cv::Mat thresholdImage, std::vector<std::vector<cv::Point>> cVector)
{
    cv::Mat roiImage;
    cv::Mat resizedRoiImage;
    cv::Mat flattenedImage;
    cv::Mat floatImage;
    cv::Rect redBoundingBox;
    int inputDigit = 0;
    int cntr = 0;

    for(auto& el : cVector)
    {
        if(cv::contourArea(el) >= 60 && cv::contourArea(el) < 280)
        {
            redBoundingBox = cv::boundingRect(el);
            ++cntr;
            cv::rectangle(trainingImage, redBoundingBox, cv::Scalar(0,0,255), 2);
            roiImage = thresholdImage(redBoundingBox);
            cv::resize(roiImage, resizedRoiImage, cv::Size(20,30)); // size(width, height)

            cv::imshow("Region of interest: ", resizedRoiImage);
            cv::imshow("Training numbers: ", trainingImage);

            std::cout << "Type in the number shown in the red rectangle: ";
            inputDigit = cv::waitKey(0);
            if(inputDigit>0 && inputDigit<10)
            {
                classificationInputDigits.push_back(inputDigit);
                resizedRoiImage.convertTo(floatImage, CV_32FC1);
                flattenedImage = floatImage.reshape(1, 1);
                trainingImageOutput.push_back(flattenedImage);
            }
        }
    }


}

// Write classification input to file
void OCR::writeClassificationFile()
{
    // Convert the classification numbers to float --> required for KNN algorithm
    cv::Mat classificationInputToFloat;
    classificationInputDigits.convertTo(classificationInputToFloat, CV_32FC1);

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
    fs_images << "trainedImages" << trainingImageOutput;

    // ... and close the file
    fs_images.release();
}

int OCR::train(const cv::Mat labelTrain)
{
    // The classification data is read from the stored file
    // Once the classification and image files are created,
    // we do not need to create it everytime we run the program.
    // Therefore, we only read the files in future runs

    cv::FileStorage fs_class(filename_class, cv::FileStorage::READ);

    // Make sure the file exists, otherwise exit
    if(!fs_class.isOpened())
    {
        std::cout << "Error: File not found!" << std::endl;
        return 0;
    }

    // Write the file content into the class member image
    fs_class["classificationDigits"] >> classificationInputDigits;
    fs_class.release();

    cv::FileStorage fs_images(filename_trained, cv::FileStorage::READ);

    // Make sure the file exists, otherwise exit
    if(!fs_images.isOpened())
    {
        std::cout << "Error: File not found!" << std::endl;
        return 0;
    }

    // Write the file content into the class member image
    fs_class["trainedImages"] >> trainingImageOutput;
    fs_class.release();

    // Training the KNearestNeighbor
    cv::ml::KNearest *knearest = cv::ml::KNearest::create();

    // Set properties of KNearest
    knearest->setIsClassifier(true);
    knearest->setAlgorithmType(cv::ml::KNearest::Types::BRUTE_FORCE);
    knearest->setDefaultK(1);

    knearest->train(trainingImageOutput, cv::ml::ROW_SAMPLE , classificationInputDigits);

    // Prepare training image to be compatible with knearest
    cv::Mat imageToPredictDigit = labelTrain.clone();
    cv::Mat flattenedImage;
    cv::Mat floatImage;
    cv::Mat knnResult;
    imageToPredictDigit.convertTo(floatImage, CV_32FC1);
    flattenedImage = floatImage.reshape(1, 1);

    knearest->findNearest(flattenedImage, knearest->getDefaultK(), knnResult);

    // extract digits from knnResult image

    delete knearest;
}
