#include "HandDetector.h"

HandDetector::HandDetector() {}
HandDetector::~HandDetector() {}

cv::Mat HandDetector::detectHand(cv::Mat inputFrame) {
    cv::Mat hsvImage;
    cv::Mat mask;

    // 1. Конвертируем из BGR (стандарт OpenCV) в HSV
    cv::cvtColor(inputFrame, hsvImage, cv::COLOR_BGR2HSV);

    // 2. Указываем диапазон цвета кожи в формате HSV
    // Эти цифры (Scalar) — нижняя и верхняя граница цвета.
    // H (Оттенок): 0-20 (это оранжево-красные тона)
    // S (Насыщенность): 30-255 (от бледного до сочного)
    // V (Яркость): 50-255 (от темного до светлого, исключая совсем черноту)

    cv::Scalar lower(0, 30, 60);
    cv::Scalar upper(20, 150, 255);

    // 3. Создаем маску
    // Функция inRange проверяет каждый пиксель:
    // Если он входит в диапазон -> делает его БЕЛЫМ (255).
    // Если нет -> делает его ЧЕРНЫМ (0).
    cv::inRange(hsvImage, lower, upper, mask);

    // 4. Немного размываем, чтобы убрать шум (мелкие точки)
    // Это называется "Размытие Гаусса"
    cv::GaussianBlur(mask, mask, cv::Size(5, 5), 0);

    return mask;
}