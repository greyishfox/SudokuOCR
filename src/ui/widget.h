#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <QDateTime>
#include <QDebug>
#include "imageprocessing.h"
#include "ocr.h"
#include "solver.h"


QT_BEGIN_NAMESPACE
namespace Ui { class Widget; }
QT_END_NAMESPACE

class Widget : public QWidget
{
    Q_OBJECT

public:
    Widget(QWidget *parent = nullptr);
    ~Widget();

public slots:
    void plotOrigImg();
    void plotSolvImg();
    void saveImg();
    void reset();

private:
    Ui::Widget *ui;
    cv::Mat m_origImg;
    cv::Mat m_solvedImg;
    QImage m_displayOrigImage;
    QImage m_displaySolvImage;

    ImageProcessing imgProcess;
    OCR myOCR;
    Solver mysolver;
};
#endif // WIDGET_H
