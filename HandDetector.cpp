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

    cv::Scalar lower(0, 70, 60);
    cv::Scalar upper(20, 130, 255);

    // 3. Создаем маску
    // Функция inRange проверяет каждый пиксель:
    // Если он входит в диапазон -> делает его БЕЛЫМ (255).
    // Если нет -> делает его ЧЕРНЫМ (0).
    cv::inRange(hsvImage, lower, upper, mask);

    // --- ДОБАВЛЯЕМ ЧИСТКУ (МОРФОЛОГИЮ) ---

    // 1. Создаем "ядро" (инструмент, которым будем чистить)
    // Размер 5x5 пикселей - оптимально для веб-камеры
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));

    // 2. ERODE: Убираем мелкий шум (белые точки исчезнут)
    cv::erode(mask, mask, element);
    cv::erode(mask, mask, element); // Делаем дважды для надежности

    // 3. DILATE: Возвращаем объем руке и убираем дырки внутри нее
    cv::dilate(mask, mask, element);
    cv::dilate(mask, mask, element);

    // 4. Размытие (оставляем как было)
    cv::GaussianBlur(mask, mask, cv::Size(5, 5), 0);

    return mask;
}
// ... (тут выше ваш старый код detectHand) ...

std::vector<cv::Point> HandDetector::findLargestContour(cv::Mat mask) {
    // 1. Подготовка хранилища.
    // Контур — это набор точек.
    // На картинке может быть много пятен (лицо, рука, лампа).
    // Поэтому мы создаем "Список Списков Точек".
    std::vector<std::vector<cv::Point>> allContours;

    // 2. Поиск контуров.
    // RETR_EXTERNAL: Нас интересуют только внешние границы (не дырки внутри пятна).
    // CHAIN_APPROX_SIMPLE: Экономия памяти. Если линия прямая, храним только начало и конец.
    cv::findContours(mask, allContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 3. Выборы "Короля горы".
    // Нам нужно найти самый большой контур.
    std::vector<cv::Point> biggestContour;
    double maxArea = 0;

    for (int i = 0; i < allContours.size(); i++) {
        // Вычисляем площадь текущего пятна
        double area = cv::contourArea(allContours[i]);

        // Фильтр шума: если пятно меньше 1000 пикселей — игнорируем его
        // Если пятно больше предыдущего лидера — запоминаем его.
        if (area > maxArea && area > 1000) {
            maxArea = area;
            biggestContour = allContours[i];
        }
    }

    // Возвращаем победителя (самое большое пятно)
    return biggestContour;
}