#include <iostream>
#include "Camera.h" // Подключаем наш заголовок

int main() {
    // 1. Создаем объект нашей камеры
    Camera myCam;

    while (true) {
        // 2. Получаем картинку
        cv::Mat frame = myCam.getFrame();

        // Проверка: если картинка пустая (камера заглючила), выходим
        if (frame.empty()) {
            std::cout << "Error: Blank frame grabbed" << std::endl;
            break;
        }

        // 3. Показываем картинку в окне
        // Используйте функцию cv::imshow("Название окна", frame);

        // 4. Ждем нажатия клавиши, чтобы не зависло
        // Используйте cv::waitKey(1);

        // Если нажата ESC (код 27), выходим из цикла
        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    return 0;
}