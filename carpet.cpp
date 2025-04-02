#include <opencv2/opencv.hpp>
#include <iostream>
#include <omp.h>
#include <cmath>
#include <string>

using namespace cv;
using namespace std;

constexpr int MIN_SIZE = 3; // Минимальный размер для рекурсии

// Оптимизированная проверка степени тройки
bool isPowerOfThree(int n)
{
	if (n < 1)
		return false;
	while (n % 3 == 0)
		n /= 3;
	return n == 1;
}

// Рекурсивное рисование ковра Серпинского с оптимизациями
void drawSierpinskiCarpet(Mat &img, int x, int y, int size, int depth, int max_depth) noexcept
{
	if (depth >= max_depth || size < MIN_SIZE)
		return;

	const int sub_size = size / 3;
	const Point center(x + sub_size, y + sub_size);

	// Рисуем центральный квадрат
	rectangle(img, center, center + Point(sub_size, sub_size), Scalar(0, 0, 0), FILLED);

// Параллельная обработка подквадратов
#pragma omp parallel for collapse(2) schedule(dynamic)
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			if (i == 1 && j == 1)
				continue; // Пропускаем центр

			drawSierpinskiCarpet(img,
								 x + i * sub_size,
								 y + j * sub_size,
								 sub_size,
								 depth + 1,
								 max_depth);
		}
	}
}

// Функция для сохранения изображения с проверкой
bool saveImage(const Mat &image, const string &filename)
{
	if (image.empty())
	{
		cerr << "Ошибка: пустое изображение для сохранения" << endl;
		return false;
	}
	return imwrite(filename, image);
}

int main(int argc, char **argv)
{
	system("chcp 65001 > nul");

	try
	{
		// Параметры по умолчанию
		int size = 729; // 3^6
		int depth = 5;
		int threads = omp_get_max_threads();
		string output_filename = "sierpinski_carpet.png";

		// Обработка аргументов командной строки
		if (argc > 1)
			size = stoi(argv[1]);
		if (argc > 2)
			depth = stoi(argv[2]);
		if (argc > 3)
			threads = stoi(argv[3]);
		if (argc > 4)
			output_filename = argv[4];

		// Проверка параметров
		if (!isPowerOfThree(size))
		{
			throw runtime_error("Размер должен быть степенью тройки (3^n)");
		}
		if (depth < 1 || threads < 1)
		{
			throw invalid_argument("Глубина и количество потоков должны быть ≥ 1");
		}

		// Настройка параллелизма
		omp_set_num_threads(min(threads, omp_get_max_threads()));
		cout << "Используется потоков: " << omp_get_max_threads() << endl;

		// Создание изображения
		Mat image(size, size, CV_8UC3, Scalar(255, 255, 255));

		// Измерение времени выполнения
		double start = omp_get_wtime();
		drawSierpinskiCarpet(image, 0, 0, size, 0, depth);
		double duration = omp_get_wtime() - start;

		cout << "Время выполнения: " << duration << " сек." << endl;
		cout << "Размер изображения: " << size << "x" << size << endl;
		cout << "Глубина рекурсии: " << depth << endl;

		// Сохранение и отображение
		if (!saveImage(image, output_filename))
		{
			throw runtime_error("Не удалось сохранить изображение");
		}
		cout << "Изображение сохранено как: " << output_filename << endl;

		namedWindow("Sierpinski Carpet", WINDOW_AUTOSIZE);
		imshow("Sierpinski Carpet", image);
		waitKey(0);
	}
	catch (const exception &e)
	{
		cerr << "Ошибка: " << e.what() << endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
