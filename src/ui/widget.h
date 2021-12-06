#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include "imageprocessing.h"
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
    // void saveImg();

private:
    Ui::Widget *ui;
    cv::Mat origImg;
    QImage displayOrigImage;
    QImage displaySolvImage;

    ImageProcessing imgProcess;
};
#endif // WIDGET_H
