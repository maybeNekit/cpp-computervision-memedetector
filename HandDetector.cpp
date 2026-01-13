#include "HandDetector.h"
#include <iostream>

HandDetector::HandDetector() {

    std::string path = "haarcascade_frontalface_default.xml";
    // ЗАГРУЗКА КАСКАДА
    // ВАЖНО: Если программа падает, укажи здесь ПОЛНЫЙ ПУТЬ к файлу xml
    // Например: "/Users/твое_имя/Projects/cv/haarcascade_frontalface_default.xml"
    if (!faceDetector.load(path)) {
        // Попробуем загрузить из стандартной папки (для Mac/Linux иногда срабатывает)
        if (!faceDetector.load(path)) {
             std::cerr << "CRITICAL ERROR: XML файл не найден! Лицо не будет удалено." << std::endl;
        }
    }
}

HandDetector::~HandDetector() {}

cv::Mat HandDetector::detectHand(cv::Mat inputFrame) {
    cv::Mat hsvImage, mask;
    cv::cvtColor(inputFrame, hsvImage, cv::COLOR_BGR2HSV);

    // 1. ЦВЕТ (Твои рабочие настройки)
    cv::Scalar lower(0, 70, 80);
    cv::Scalar upper(20, 130, 255);
    cv::inRange(hsvImage, lower, upper, mask);

    // 2. ЧИСТКА ШУМА
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::dilate(mask, mask, kernel, cv::Point(-1,-1), 2);
    cv::GaussianBlur(mask, mask, cv::Size(5, 5), 0);

    // 3. УДАЛЕНИЕ ЛИЦА (Самое важное!)
    // Мы ищем лицо на оригинальном кадре и стираем его с маски.

    std::vector<cv::Rect> faces;
    cv::Mat gray;
    cv::cvtColor(inputFrame, gray, cv::COLOR_BGR2GRAY);

    // Ищем лица (scaleFactor=1.1, minNeighbors=4)
    faceDetector.detectMultiScale(gray, faces, 1.1, 4, 0, cv::Size(60, 60));

    for (size_t i = 0; i < faces.size(); i++) {
        // Расширяем зону удаления (чтобы стереть шею и прическу)
        cv::Rect faceRect = faces[i];

        // Чуть увеличиваем квадрат удаления
        int padding = 40;
        faceRect.x = std::max(0, faceRect.x - padding);
        faceRect.y = std::max(0, faceRect.y - padding);
        faceRect.width += padding * 2;
        faceRect.height += padding * 3; // Вниз берем больше (шея)

        // РИСУЕМ ЧЕРНЫЙ КВАДРАТ НА МАСКЕ
        // Это полностью стирает лицо из расчетов
        cv::rectangle(mask, faceRect, cv::Scalar(0), -1);
    }

    return mask;
}

std::vector<cv::Point> HandDetector::findHandContour(cv::Mat mask) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Point> bestContour;
    double maxArea = 0;

    for (int i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);

        // Теперь нам не нужны сложные проверки на Solidity!
        // Лица на маске нет. Самый большой объект - это рука.
        // Просто фильтруем мелкий шум.
        if (area > maxArea && area > 2000) {
            maxArea = area;
            bestContour = contours[i];
        }
    }

    return bestContour;
}