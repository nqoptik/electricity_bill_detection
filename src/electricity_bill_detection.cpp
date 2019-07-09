#include <dirent.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

enum cvSelectTopContours_Mode
{
    CV_SELECT_CONTOUR_AREA = 0,
    CV_SELECT_CONTOUR_SIZE = 1
};

float get_euclid_distance(const cv::Point& point_1, const cv::Point& point_2);
void find_nearest_point(const std::vector<cv::Point>& sequences, const cv::Point& root, cv::Point& result);
std::vector<std::vector<cv::Point>> select_top_contours(const std::vector<std::vector<cv::Point>>& source,
                                                        const int& top,
                                                        const int& mode,
                                                        const int& min_size,
                                                        const double& min_area);
void execute_main(const cv::Mat& original_image, const cv::Mat& image, std::vector<cv::Mat>& information_vector);
void get_bill_contours(const cv::Mat& image, std::vector<std::vector<cv::Point>>& contours);
void get_corners_from_contour(const std::vector<cv::Point>& sequences, std::vector<cv::Point>& raw_corners);
void get_raw_vertices(const cv::Mat& filled_raw_contours_image,
                      const std::vector<cv::Point>& raw_corners,
                      const int& number_of_horizontal_pieces,
                      std::vector<cv::Point>& raw_tops,
                      std::vector<cv::Point>& raw_bots,
                      const int& number_of_vertical_pieces,
                      std::vector<cv::Point>& raw_lefts,
                      std::vector<cv::Point>& raw_rights);
void get_quadrangle_vertices(const std::vector<cv::Point>& quadrangle_corners,
                             const int& number_of_horizontal_pieces,
                             const int& number_of_vertical_pieces,
                             std::vector<cv::Point>& quadrangle_tops,
                             std::vector<cv::Point>& quadrangle_bots,
                             std::vector<cv::Point>& quadrangle_lefts,
                             std::vector<cv::Point>& quadrangle_rights);
void warp_to_medial_bill(const cv::Mat& raw_bill,
                         cv::Mat& medial_bill,
                         const std::vector<cv::Point>& raw_tops,
                         const std::vector<cv::Point>& raw_bots,
                         const std::vector<cv::Point>& quadrangle_tops,
                         const std::vector<cv::Point>& quadrangle_bots,
                         std::vector<cv::Mat>& medial_matrices);
void warp_to_quadrangle_bill(const cv::Mat& medial_bill,
                             cv::Mat& quadrangle_bill,
                             const std::vector<cv::Point>& raw_lefts,
                             const std::vector<cv::Point>& raw_rights,
                             const std::vector<cv::Point>& quadrangle_lefts,
                             const std::vector<cv::Point>& quadrangle_rights,
                             std::vector<cv::Mat>& quadrangle_matrices);
void warp_to_rectangle_bill(const cv::Mat& quadrangle_bill,
                            cv::Mat& rectange_bill,
                            const std::vector<cv::Point>& quadrangle_corners,
                            cv::Mat& rectangle_matrix);
void set_rectangle_check_points(std::vector<cv::Point2f>& rectangle_points);
std::vector<cv::Point2f> get_original_points(const std::vector<cv::Point2f>& rectangle_points,
                                             const cv::Mat& rectangle_matrix,
                                             const std::vector<cv::Mat>& quadrangle_matrices,
                                             const std::vector<cv::Mat>& medial_matrices,
                                             const cv::Mat& rectange_bill,
                                             const cv::Mat& image,
                                             const cv::Mat& original_image);
void get_original_rectangles(const std::vector<cv::Point2f>& original_points,
                             std::vector<std::vector<cv::Point2f>>& original_quadrangles,
                             std::vector<cv::Rect>& original_rectangles);
void get_information(const cv::Mat& original_image,
                     std::vector<cv::Mat>& information_vector,
                     const std::vector<std::vector<cv::Point2f>>& original_quadrangles,
                     std::vector<cv::Rect>& original_rectangles);
std::vector<float> get_moving_distances(const std::vector<cv::Mat>& check_boxes);
std::vector<cv::Point2f> set_rectangle_points(const std::vector<float>& moving_distances);
void write_information_boxes(const std::vector<std::vector<cv::Mat>>& information_boxes);

int main()
{
    std::string directory_name = "input_images/";
    DIR* directory_ptr;
    directory_ptr = opendir(directory_name.c_str());
    if (directory_ptr == NULL)
    {
        std::cout << "Directory not found." << std::endl;
        return 1;
    }

    struct dirent* dirent_ptr;
    std::vector<std::string> filenames;
    while ((dirent_ptr = readdir(directory_ptr)) != NULL)
    {
        if (strcmp(dirent_ptr->d_name, ".") == 0 || strcmp(dirent_ptr->d_name, "..") == 0)
        {
            continue;
        }
        std::string image_path = directory_name;
        image_path.append(dirent_ptr->d_name);
        filenames.push_back(image_path);
    }
    closedir(directory_ptr);

    // Get the information_vector from bill
    std::vector<std::vector<cv::Mat>> information_boxes;
    int index = 0;
    std::vector<cv::Mat> resized_images;
    for (size_t i = 0; i < filenames.size(); ++i)
    {
        cv::Mat original_image, image;
        original_image = cv::imread(filenames[i]);
        if (original_image.empty())
            break;
        std::vector<cv::Mat> information_vector;
        cv::resize(original_image, image, cv::Size(round(1000 * original_image.cols / (float)original_image.rows), 1000), 0, 0, cv::INTER_CUBIC);
        resized_images.push_back(image);
        execute_main(original_image, image, information_vector);
        information_boxes.push_back(information_vector);
        index++;
    }

    for (unsigned int i = 0; i < information_boxes.size(); i++)
    {
        cv::imshow("Resized image", resized_images[i]);
        for (unsigned int j = 0; j < information_boxes[i].size(); j++)
        {
            std::string str = "information box ";
            str.append(std::to_string(j));
            cv::imshow(str, information_boxes[i][j]);
        }
        cv::waitKey();
    }

    // Write the information boxes
    write_information_boxes(information_boxes);

    return 0;
}

float get_euclid_distance(const cv::Point& point_1, const cv::Point& point_2)
{
    int d_x = point_2.x - point_1.x;
    int d_y = point_2.y - point_1.y;
    return sqrt((float)(d_x * d_x + d_y * d_y));
}

void find_nearest_point(const std::vector<cv::Point>& sequences, const cv::Point& root, cv::Point& result)
{
    if (sequences.size() == 0)
    {
        return;
    }

    result = sequences[0];
    float min_distance = get_euclid_distance(sequences[0], root);
    for (unsigned int i = 1; i < sequences.size(); i++)
    {
        float this_distance = get_euclid_distance(sequences[i], root);
        if (this_distance < min_distance)
        {
            result = sequences[i];
            min_distance = this_distance;
        }
    }
}

std::vector<std::vector<cv::Point>> select_top_contours(const std::vector<std::vector<cv::Point>>& source,
                                                        const int& top,
                                                        const int& mode,
                                                        const int& min_size,
                                                        const double& min_area)
{
    std::vector<std::vector<cv::Point>> destination;
    int source_size = source.size();
    if (source_size != 0)
    {
        if (mode == CV_SELECT_CONTOUR_SIZE)
        {
            std::vector<int> source_sizes;
            for (int i = 0; i < source_size; i++)
            {
                source_sizes.push_back(source[i].size());
            }
            std::vector<int> sorted_source_sizes = source_sizes;
            std::sort(sorted_source_sizes.begin(), sorted_source_sizes.end());
            int min_size_index = MAX(source_size - MAX(top, 1), 0);
            int this_min_size = MAX(sorted_source_sizes[min_size_index], min_size);
            for (int i = 0; i < source_size; i++)
            {
                if (source_sizes[i] >= this_min_size)
                {
                    destination.push_back(source[i]);
                }
            }
        }
        else
        {
            std::vector<double> source_areas;
            for (int i = 0; i < source_size; i++)
            {
                source_areas.push_back(cv::contourArea(source[i]));
            }
            std::vector<double> sorted_source_areas = source_areas;
            std::sort(sorted_source_areas.begin(), sorted_source_areas.end());
            int min_area_index = MAX(source_size - MAX(top, 1), 0);
            double this_min_area = MAX(sorted_source_areas[min_area_index], min_area);
            for (int i = 0; i < source_size; i++)
            {
                if (source_areas[i] >= this_min_area)
                {
                    destination.push_back(source[i]);
                }
            }
        }
    }
    return destination;
}

void get_bill_contours(const cv::Mat& image, std::vector<std::vector<cv::Point>>& contours)
{
    cv::Mat red_channel(image.size(), CV_8UC1);
    int from_to[] = {2, 0};
    cv::mixChannels(&image, 1, &red_channel, 1, from_to, 1);

    cv::Mat binary_image;
    cv::adaptiveThreshold(red_channel, binary_image, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 95, 0);
    cv::erode(binary_image, binary_image, cv::Mat(), cv::Point(-1, -1), 2, 1, 1);
    cv::dilate(binary_image, binary_image, cv::Mat(), cv::Point(-1, -1), 3, 1, 1);

    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binary_image, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    contours = select_top_contours(contours, 1, CV_SELECT_CONTOUR_AREA, 0, 100);
}

void get_corners_from_contour(const std::vector<cv::Point>& sequences, std::vector<cv::Point>& raw_corners)
{
    cv::Rect boundary_rectangle = cv::boundingRect(sequences);
    cv::Point temporary_point;
    find_nearest_point(sequences, boundary_rectangle.tl(), temporary_point);
    raw_corners[0] = temporary_point;
    find_nearest_point(sequences, cv::Point(boundary_rectangle.x + boundary_rectangle.width, boundary_rectangle.y), temporary_point);
    raw_corners[1] = temporary_point;
    find_nearest_point(sequences, boundary_rectangle.br(), temporary_point);
    raw_corners[2] = temporary_point;
    find_nearest_point(sequences, cv::Point(boundary_rectangle.x, boundary_rectangle.y + boundary_rectangle.height), temporary_point);
    raw_corners[3] = temporary_point;
}

void get_raw_vertices(const cv::Mat& filled_raw_contours_image,
                      const std::vector<cv::Point>& raw_corners,
                      const int& number_of_horizontal_pieces,
                      std::vector<cv::Point>& raw_tops,
                      std::vector<cv::Point>& raw_bots,
                      const int& number_of_vertical_pieces,
                      std::vector<cv::Point>& raw_lefts,
                      std::vector<cv::Point>& raw_rights)
{
    raw_tops.push_back(raw_corners[0]);
    raw_bots.push_back(raw_corners[3]);
    raw_lefts.push_back(raw_corners[0]);
    raw_rights.push_back(raw_corners[1]);
    for (int i = 1; i < number_of_horizontal_pieces; i++)
    {
        cv::Point temporary_point;
        temporary_point.x = round(((number_of_horizontal_pieces - i) * raw_corners[0].x + i * raw_corners[1].x) / (float)number_of_horizontal_pieces);
        int middle_y = (raw_corners[0].y + raw_corners[3].y) / 2;

        for (int y = 0; y < middle_y; y++)
        {
            if (filled_raw_contours_image.at<cv::Vec3b>(y, temporary_point.x)[0] == 0 && filled_raw_contours_image.at<cv::Vec3b>(y + 1, temporary_point.x)[0] != 0)
            {
                temporary_point.y = y + 1;
                break;
            }
        }
        raw_tops.push_back(temporary_point);

        temporary_point.x = round(((number_of_horizontal_pieces - i) * raw_corners[3].x + i * raw_corners[2].x) / (float)number_of_horizontal_pieces);
        for (int y = middle_y; y < filled_raw_contours_image.rows; y++)
        {
            if (filled_raw_contours_image.at<cv::Vec3b>(y - 1, temporary_point.x)[0] != 0 && filled_raw_contours_image.at<cv::Vec3b>(y, temporary_point.x)[0] == 0)
            {
                temporary_point.y = y - 1;
                break;
            }
        }
        raw_bots.push_back(temporary_point);
    }

    for (int i = 1; i < number_of_vertical_pieces; i++)
    {
        cv::Point temporary_point;
        temporary_point.y = round(((number_of_vertical_pieces - i) * raw_corners[0].y + i * raw_corners[3].y) / (float)number_of_vertical_pieces);
        int middle_x = (raw_corners[0].x + raw_corners[1].x) / 2;

        for (int x = 0; x < middle_x; x++)
        {
            if (filled_raw_contours_image.at<cv::Vec3b>(temporary_point.y, x)[0] == 0 && filled_raw_contours_image.at<cv::Vec3b>(temporary_point.y, x + 1)[0] != 0)
            {
                temporary_point.x = x + 1;
                break;
            }
        }
        raw_lefts.push_back(temporary_point);

        temporary_point.y = round(((number_of_vertical_pieces - i) * raw_corners[1].y + i * raw_corners[2].y) / (float)number_of_vertical_pieces);

        for (int x = middle_x; x < filled_raw_contours_image.cols; x++)
        {
            if (filled_raw_contours_image.at<cv::Vec3b>(temporary_point.y, x - 1)[0] != 0 && filled_raw_contours_image.at<cv::Vec3b>(temporary_point.y, x)[0] == 0)
            {
                temporary_point.x = x - 1;
                break;
            }
        }
        raw_rights.push_back(temporary_point);
    }

    raw_tops.push_back(raw_corners[1]);
    raw_bots.push_back(raw_corners[2]);
    raw_lefts.push_back(raw_corners[3]);
    raw_rights.push_back(raw_corners[2]);
}

void get_quadrangle_vertices(const std::vector<cv::Point>& quadrangle_corners,
                             const int& number_of_horizontal_pieces,
                             const int& number_of_vertical_pieces,
                             std::vector<cv::Point>& quadrangle_tops,
                             std::vector<cv::Point>& quadrangle_bots,
                             std::vector<cv::Point>& quadrangle_lefts,
                             std::vector<cv::Point>& quadrangle_rights)
{
    quadrangle_tops.push_back(quadrangle_corners[0]);
    quadrangle_bots.push_back(quadrangle_corners[3]);
    quadrangle_lefts.push_back(quadrangle_corners[0]);
    quadrangle_rights.push_back(quadrangle_corners[1]);

    for (int i = 1; i < number_of_horizontal_pieces; i++)
    {
        cv::Point temporary_point = (number_of_horizontal_pieces - i) * quadrangle_corners[0] + i * quadrangle_corners[1];
        temporary_point.x = round(temporary_point.x / (float)number_of_horizontal_pieces);
        temporary_point.y = round(temporary_point.y / (float)number_of_horizontal_pieces);
        quadrangle_tops.push_back(temporary_point);

        temporary_point = (number_of_horizontal_pieces - i) * quadrangle_corners[3] + i * quadrangle_corners[2];
        temporary_point.x = round(temporary_point.x / (float)number_of_horizontal_pieces);
        temporary_point.y = round(temporary_point.y / (float)number_of_horizontal_pieces);
        quadrangle_bots.push_back(temporary_point);
    }

    for (int i = 1; i < number_of_vertical_pieces; i++)
    {
        cv::Point temporary_point = (number_of_vertical_pieces - i) * quadrangle_corners[0] + i * quadrangle_corners[3];
        temporary_point.x = round(temporary_point.x / (float)number_of_vertical_pieces);
        temporary_point.y = round(temporary_point.y / (float)number_of_vertical_pieces);
        quadrangle_lefts.push_back(temporary_point);

        temporary_point = (number_of_vertical_pieces - i) * quadrangle_corners[1] + i * quadrangle_corners[2];
        temporary_point.x = round(temporary_point.x / (float)number_of_vertical_pieces);
        temporary_point.y = round(temporary_point.y / (float)number_of_vertical_pieces);
        quadrangle_rights.push_back(temporary_point);
    }

    quadrangle_tops.push_back(quadrangle_corners[1]);
    quadrangle_bots.push_back(quadrangle_corners[2]);
    quadrangle_lefts.push_back(quadrangle_corners[3]);
    quadrangle_rights.push_back(quadrangle_corners[2]);
}

void warp_to_medial_bill(const cv::Mat& raw_bill,
                         cv::Mat& medial_bill,
                         const std::vector<cv::Point>& raw_tops,
                         const std::vector<cv::Point>& raw_bots,
                         const std::vector<cv::Point>& quadrangle_tops,
                         const std::vector<cv::Point>& quadrangle_bots,
                         std::vector<cv::Mat>& medial_matrices)
{
    medial_matrices.clear();
    for (unsigned int i = 0; i < raw_tops.size() - 1; i++)
    {
        std::vector<cv::Point2f> previous_points(4), current_points(4);
        previous_points[0] = raw_tops[i];
        previous_points[1] = raw_tops[i + 1];
        previous_points[2] = raw_bots[i + 1];
        previous_points[3] = raw_bots[i];

        current_points[0] = quadrangle_tops[i];
        current_points[1] = quadrangle_tops[i + 1];
        current_points[2] = quadrangle_bots[i + 1];
        current_points[3] = quadrangle_bots[i];

        medial_matrices.push_back(cv::getPerspectiveTransform(current_points, previous_points));

        cv::Rect boundary_rectangle = cv::boundingRect(current_points);
        if (i == 0)
        {
            int delta = MIN(15, boundary_rectangle.x);
            boundary_rectangle.x -= delta;
            boundary_rectangle.width += delta;
        }

        if (i == raw_tops.size() - 2)
        {
            int delta = MIN(15, raw_bill.cols - boundary_rectangle.x - boundary_rectangle.width);
            boundary_rectangle.width += delta;
        }

        current_points[0] -= cv::Point2f(boundary_rectangle.tl().x, boundary_rectangle.tl().y);
        current_points[1] -= cv::Point2f(boundary_rectangle.tl().x, boundary_rectangle.tl().y);
        current_points[2] -= cv::Point2f(boundary_rectangle.tl().x, boundary_rectangle.tl().y);
        current_points[3] -= cv::Point2f(boundary_rectangle.tl().x, boundary_rectangle.tl().y);

        cv::Mat perspective_transform_matrix = cv::getPerspectiveTransform(previous_points, current_points);
        cv::Mat piece = cv::Mat::zeros(raw_bill.size(), CV_8UC3);

        std::vector<cv::Point> piece_corners;
        if (i == 0)
        {
            piece_corners.push_back(cv::Point(0, 0));
            piece_corners.push_back(cv::Point(raw_tops[1].x, 0));
            piece_corners.push_back(raw_tops[1]);
            piece_corners.push_back(raw_bots[1]);
            piece_corners.push_back(cv::Point(raw_bots[1].x, raw_bill.rows - 1));
            piece_corners.push_back(cv::Point(0, raw_bill.rows - 1));
        }
        if (i == raw_tops.size() - 2)
        {
            piece_corners.push_back(cv::Point(raw_tops[i].x, 0));
            piece_corners.push_back(cv::Point(raw_bill.cols - 1, 0));
            piece_corners.push_back(cv::Point(raw_bill.cols - 1, raw_bill.rows - 1));
            piece_corners.push_back(cv::Point(raw_bots[i].y, raw_bill.rows - 1));
            piece_corners.push_back(raw_bots[i]);
            piece_corners.push_back(raw_tops[i]);
        }
        if (i != 0 && i != raw_tops.size() - 2)
        {
            piece_corners.push_back(raw_tops[i]);
            piece_corners.push_back(raw_tops[i + 1]);
            piece_corners.push_back(raw_bots[i + 1]);
            piece_corners.push_back(raw_bots[i]);
        }

        std::vector<std::vector<cv::Point>> piece_corners_vector(1);
        piece_corners_vector[0] = piece_corners;
        cv::drawContours(piece, piece_corners_vector, 0, cv::Scalar(255, 255, 255), CV_FILLED);

        cv::bitwise_and(piece, raw_bill, piece);

        cv::Mat boundary_rectangle_image;
        cv::warpPerspective(piece, boundary_rectangle_image, perspective_transform_matrix, boundary_rectangle.size(), cv::INTER_CUBIC);
        cv::Mat medial_piece = cv::Mat::zeros(medial_bill.size(), CV_8UC3);
        boundary_rectangle_image.copyTo(medial_piece(boundary_rectangle));
        medial_bill = cv::max(medial_piece, medial_bill);
    }
}

void warp_to_quadrangle_bill(const cv::Mat& medial_bill,
                             cv::Mat& quadrangle_bill,
                             const std::vector<cv::Point>& raw_lefts,
                             const std::vector<cv::Point>& raw_rights,
                             const std::vector<cv::Point>& quadrangle_lefts,
                             const std::vector<cv::Point>& quadrangle_rights,
                             std::vector<cv::Mat>& quadrangle_matrices)
{
    quadrangle_matrices.clear();
    for (unsigned int i = 0; i < raw_lefts.size() - 1; i++)
    {
        std::vector<cv::Point2f> previous_points(4), current_points(4);
        previous_points[0] = raw_lefts[i];
        previous_points[1] = raw_lefts[i + 1];
        previous_points[2] = raw_rights[i + 1];
        previous_points[3] = raw_rights[i];

        current_points[0] = quadrangle_lefts[i];
        current_points[1] = quadrangle_lefts[i + 1];
        current_points[2] = quadrangle_rights[i + 1];
        current_points[3] = quadrangle_rights[i];

        quadrangle_matrices.push_back(cv::getPerspectiveTransform(current_points, previous_points));

        cv::Rect boundary_rectangle = cv::boundingRect(current_points);
        if (i == 0)
        {
            int delta = MIN(15, boundary_rectangle.y);
            boundary_rectangle.y -= delta;
            boundary_rectangle.height += delta;
        }
        if (i == raw_lefts.size() - 2)
        {
            int delta = MIN(15, medial_bill.rows - boundary_rectangle.y - boundary_rectangle.height);
            boundary_rectangle.height += delta;
        }

        current_points[0] -= cv::Point2f(boundary_rectangle.tl().x, boundary_rectangle.tl().y);
        current_points[1] -= cv::Point2f(boundary_rectangle.tl().x, boundary_rectangle.tl().y);
        current_points[2] -= cv::Point2f(boundary_rectangle.tl().x, boundary_rectangle.tl().y);
        current_points[3] -= cv::Point2f(boundary_rectangle.tl().x, boundary_rectangle.tl().y);

        cv::Mat perspective_transform_matrix = cv::getPerspectiveTransform(previous_points, current_points);
        cv::Mat piece = cv::Mat::zeros(medial_bill.size(), CV_8UC3);

        std::vector<cv::Point> piece_corners(4);
        piece_corners[0] = raw_lefts[i];
        piece_corners[1] = raw_lefts[i + 1];
        piece_corners[2] = raw_rights[i + 1];
        piece_corners[3] = raw_rights[i];

        std::vector<std::vector<cv::Point>> piece_corners_vector(1);
        piece_corners_vector[0] = piece_corners;
        cv::drawContours(piece, piece_corners_vector, 0, cv::Scalar(255, 255, 255), CV_FILLED);

        cv::bitwise_and(piece, medial_bill, piece);

        cv::Mat boundary_rectangle_image;
        cv::warpPerspective(piece, boundary_rectangle_image, perspective_transform_matrix, boundary_rectangle.size(), cv::INTER_CUBIC);
        cv::Mat quadrangle_piece = cv::Mat::zeros(quadrangle_bill.size(), CV_8UC3);
        boundary_rectangle_image.copyTo(quadrangle_piece(boundary_rectangle));
        quadrangle_bill = cv::max(quadrangle_bill, quadrangle_piece);
    }
}

void warp_to_rectangle_bill(const cv::Mat& quadrangle_bill,
                            cv::Mat& rectange_bill,
                            const std::vector<cv::Point>& quadrangle_corners,
                            cv::Mat& rectangle_matrix)
{
    cv::Rect boundary_rectangle = cv::Rect(0, 0, 1000, 600);
    std::vector<cv::Point2f> previous_points(4), current_points(4);
    previous_points[0] = quadrangle_corners[0];
    previous_points[1] = quadrangle_corners[1];
    previous_points[2] = quadrangle_corners[2];
    previous_points[3] = quadrangle_corners[3];

    current_points[0] = cv::Point2f(0, 0);
    current_points[1] = cv::Point2f((float)boundary_rectangle.width, 0);
    current_points[2] = cv::Point2f((float)boundary_rectangle.width, (float)boundary_rectangle.height);
    current_points[3] = cv::Point2f(0, (float)boundary_rectangle.height);

    rectangle_matrix = cv::getPerspectiveTransform(current_points, previous_points);

    cv::Mat perspective_transform_matrix = cv::getPerspectiveTransform(previous_points, current_points);
    cv::warpPerspective(quadrangle_bill, rectange_bill, perspective_transform_matrix, boundary_rectangle.size(), cv::INTER_CUBIC);
}

void set_rectangle_check_points(std::vector<cv::Point2f>& rectangle_check_points)
{
    // The check point 1
    rectangle_check_points[0] = cv::Point2f(260, 228);
    rectangle_check_points[1] = cv::Point2f(270, 228);
    rectangle_check_points[2] = cv::Point2f(270, 268);
    rectangle_check_points[3] = cv::Point2f(260, 268);

    // The check point 2
    rectangle_check_points[4] = cv::Point2f(260, 228);
    rectangle_check_points[5] = cv::Point2f(270, 228);
    rectangle_check_points[6] = cv::Point2f(270, 268);
    rectangle_check_points[7] = cv::Point2f(260, 268);

    // The check point 3
    rectangle_check_points[8] = cv::Point2f(525, 538);
    rectangle_check_points[9] = cv::Point2f(535, 538);
    rectangle_check_points[10] = cv::Point2f(535, 578);
    rectangle_check_points[11] = cv::Point2f(525, 578);

    // The check point 4
    rectangle_check_points[12] = cv::Point2f(645, 128);
    rectangle_check_points[13] = cv::Point2f(655, 128);
    rectangle_check_points[14] = cv::Point2f(655, 168);
    rectangle_check_points[15] = cv::Point2f(645, 168);

    // The check point 5
    rectangle_check_points[16] = cv::Point2f(960, 428);
    rectangle_check_points[17] = cv::Point2f(970, 428);
    rectangle_check_points[18] = cv::Point2f(970, 468);
    rectangle_check_points[19] = cv::Point2f(960, 468);
}

std::vector<cv::Point2f> get_original_points(const std::vector<cv::Point2f>& rectangle_points,
                                             const cv::Mat& rectangle_matrix,
                                             const std::vector<cv::Mat>& quadrangle_matrices,
                                             const std::vector<cv::Mat>& medial_matrices,
                                             const cv::Mat& rectange_bill,
                                             const cv::Mat& image,
                                             const cv::Mat& original_image)
{
    std::vector<int> quadrangle_indices;
    for (unsigned int i = 0; i < rectangle_points.size(); i++)
    {
        quadrangle_indices.push_back(floor(quadrangle_matrices.size() * rectangle_points[i].y / rectange_bill.rows));
    }

    std::vector<int> medial_indices;
    for (unsigned int i = 0; i < rectangle_points.size(); i++)
    {
        medial_indices.push_back(floor(medial_matrices.size() * rectangle_points[i].x / rectange_bill.cols));
    }

    std::vector<cv::Point2f> quadrangle_points;
    std::vector<cv::Point2f> medial_points;
    std::vector<cv::Point2f> raw_points;
    cv::perspectiveTransform(rectangle_points, quadrangle_points, rectangle_matrix);

    for (unsigned int i = 0; i < quadrangle_points.size(); i++)
    {
        int index = quadrangle_indices[i];
        std::vector<cv::Point2f> quadrangle_point(1), medial_point;
        quadrangle_point[0] = quadrangle_points[i];
        cv::perspectiveTransform(quadrangle_point, medial_point, quadrangle_matrices[index]);
        medial_points.push_back(medial_point[0]);
    }

    for (unsigned int i = 0; i < medial_points.size(); i++)
    {
        int index = medial_indices[i];
        std::vector<cv::Point2f> medial_point(1), raw_point;
        medial_point[0] = medial_points[i];
        cv::perspectiveTransform(medial_point, raw_point, medial_matrices[index]);
        raw_points.push_back(raw_point[0]);
    }

    std::vector<cv::Point2f> original_points;
    for (unsigned int i = 0; i < raw_points.size(); i++)
    {
        cv::Point2f original_point;
        original_point.x = raw_points[i].x * original_image.cols / image.cols;
        original_point.y = raw_points[i].y * original_image.rows / image.rows;
        original_points.push_back(original_point);
    }
    return original_points;
}

void get_original_rectangles(const std::vector<cv::Point2f>& original_points,
                             std::vector<std::vector<cv::Point2f>>& original_quadrangles,
                             std::vector<cv::Rect>& original_rectangles)
{
    for (unsigned int i = 0; i < original_points.size(); i += 4)
    {
        std::vector<cv::Point2f> original_quadrangle;
        for (int j = 0; j < 4; j++)
        {
            original_quadrangle.push_back(original_points[i + j]);
        }
        cv::Rect rectangle = cv::boundingRect(original_quadrangle);
        original_quadrangles.push_back(original_quadrangle);
        original_rectangles.push_back(rectangle);
    }
}

void get_information(const cv::Mat& original_image,
                     std::vector<cv::Mat>& information_vector,
                     const std::vector<std::vector<cv::Point2f>>& original_quadrangles,
                     std::vector<cv::Rect>& original_rectangles)
{
    information_vector.clear();
    for (unsigned int i = 0; i < original_rectangles.size(); i++)
    {
        original_rectangles[i].x = 0;
        original_rectangles[i].y = 0;

        std::vector<cv::Point2f> previous_points(4), current_points(4);
        previous_points[0] = original_quadrangles[i][0];
        previous_points[1] = original_quadrangles[i][1];
        previous_points[2] = original_quadrangles[i][2];
        previous_points[3] = original_quadrangles[i][3];

        current_points[0] = cv::Point2f(0, 0);
        current_points[1] = cv::Point2f((float)original_rectangles[i].width, 0);
        current_points[2] = cv::Point2f((float)original_rectangles[i].width, (float)original_rectangles[i].height);
        current_points[3] = cv::Point2f(0, (float)original_rectangles[i].height);

        cv::Mat perspective_transform_matrix = cv::getPerspectiveTransform(previous_points, current_points);
        cv::Mat information;
        cv::warpPerspective(original_image, information, perspective_transform_matrix, original_rectangles[i].size(), cv::INTER_CUBIC);
        information_vector.push_back(information);
    }
}

std::vector<float> get_moving_distances(const std::vector<cv::Mat>& check_boxes)
{
    std::vector<float> moving_distances;
    for (unsigned int i = 0; i < check_boxes.size(); i++)
    {
        cv::Mat binary_image, threshold_image;
        cv::cvtColor(check_boxes[i], binary_image, CV_BGR2GRAY);
        cv::threshold(binary_image, threshold_image, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
        cv::dilate(threshold_image, threshold_image, cv::Mat(), cv::Point(-1, -1), 2, 1, 1);

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(threshold_image, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        contours = select_top_contours(contours, 1, CV_SELECT_CONTOUR_AREA, 0, 1);
        if (contours.size() != 0)
        {
            cv::Rect boundary_rectangle = cv::boundingRect(contours[0]);
            float originalDist = (boundary_rectangle.y + boundary_rectangle.height / 2.0f) - check_boxes[i].rows / 2.0f;
            float rectDist = originalDist / check_boxes[i].rows * 41;
            moving_distances.push_back(rectDist);
        }
        else
        {
            moving_distances.push_back(0);
        }
    }
    return moving_distances;
}

std::vector<cv::Point2f> set_rectangle_points(const std::vector<float>& moving_distances)
{
    std::vector<cv::Point2f> rectangle_points(20);

    // The box 1
    rectangle_points[0] = cv::Point2f(250, 160 + moving_distances[0]);
    rectangle_points[1] = cv::Point2f(390, 160 + moving_distances[0]);
    rectangle_points[2] = cv::Point2f(390, 190 + moving_distances[0]);
    rectangle_points[3] = cv::Point2f(250, 190 + moving_distances[0]);

    // The box 2
    rectangle_points[4] = cv::Point2f(300, 195 + moving_distances[1]);
    rectangle_points[5] = cv::Point2f(405, 195 + moving_distances[1]);
    rectangle_points[6] = cv::Point2f(405, 225 + moving_distances[1]);
    rectangle_points[7] = cv::Point2f(300, 225 + moving_distances[1]);

    // The box 3
    rectangle_points[8] = cv::Point2f(415, 492 + moving_distances[2]);
    rectangle_points[9] = cv::Point2f(455, 492 + moving_distances[2]);
    rectangle_points[10] = cv::Point2f(455, 520 + moving_distances[2]);
    rectangle_points[11] = cv::Point2f(415, 520 + moving_distances[2]);

    // The box 4
    rectangle_points[12] = cv::Point2f(540, 152 + moving_distances[3]);
    rectangle_points[13] = cv::Point2f(660, 152 + moving_distances[3]);
    rectangle_points[14] = cv::Point2f(660, 182 + moving_distances[3]);
    rectangle_points[15] = cv::Point2f(540, 182 + moving_distances[3]);

    // The box 5
    rectangle_points[16] = cv::Point2f(715, 416 + moving_distances[4]);
    rectangle_points[17] = cv::Point2f(960, 416 + moving_distances[4]);
    rectangle_points[18] = cv::Point2f(960, 447 + moving_distances[4]);
    rectangle_points[19] = cv::Point2f(715, 447 + moving_distances[4]);

    return rectangle_points;
}

void execute_main(const cv::Mat& original_image, const cv::Mat& image, std::vector<cv::Mat>& information_vector)
{
    double t1 = clock();
    // Get the bill contours
    std::vector<std::vector<cv::Point>> contours;
    get_bill_contours(image, contours);
    if (contours.size() != 1)
        return;

    // Approx to raw contours and quadrangle contours
    std::vector<std::vector<cv::Point>> raw_contours(1), quadrangle_contours(1);
    cv::approxPolyDP(contours[0], raw_contours[0], 5, true);
    cv::approxPolyDP(contours[0], quadrangle_contours[0], 35, true);

    // Fill the raw contours
    cv::Mat filled_raw_contours_image = cv::Mat::zeros(image.size(), CV_8UC3);
    cv::drawContours(filled_raw_contours_image, raw_contours, 0, cv::Scalar(255, 255, 255), CV_FILLED);

    // Get the raw corners
    std::vector<cv::Point> raw_corners(4);
    get_corners_from_contour(raw_contours[0], raw_corners);

    // Get raw top vertices and raw bot vertices
    std::vector<cv::Point> raw_tops, raw_bots, raw_lefts, raw_rights;
    int number_of_horizontal_pieces = 10;
    int number_of_vertical_pieces = 6;
    get_raw_vertices(filled_raw_contours_image, raw_corners, number_of_horizontal_pieces, raw_tops, raw_bots, number_of_vertical_pieces, raw_lefts, raw_rights);

    // Get quadrangle corners
    std::vector<cv::Point> quadrangle_corners(4);
    get_corners_from_contour(quadrangle_contours[0], quadrangle_corners);

    // Get quadrangle vertices
    std::vector<cv::Point> quadrangle_tops, quadrangle_bots, quadrangle_lefts, quadrangle_rights;
    get_quadrangle_vertices(quadrangle_corners, number_of_horizontal_pieces, number_of_vertical_pieces, quadrangle_tops, quadrangle_bots, quadrangle_lefts, quadrangle_rights);

    double t2 = clock();
    // Warp to a medial bill
    cv::Mat raw_bill;
    cv::bitwise_and(filled_raw_contours_image, image, raw_bill);
    cv::Mat medial_bill = cv::Mat::zeros(image.size(), CV_8UC3);
    std::vector<cv::Mat> medial_matrices;
    warp_to_medial_bill(raw_bill, medial_bill, raw_tops, raw_bots, quadrangle_tops, quadrangle_bots, medial_matrices);

    double t3 = clock();
    // Warp to a quadrangle bill
    cv::Mat quadrangle_bill = cv::Mat::zeros(medial_bill.size(), CV_8UC3);
    std::vector<cv::Mat> quadrangle_matrices;
    warp_to_quadrangle_bill(medial_bill, quadrangle_bill, raw_lefts, raw_rights, quadrangle_lefts, quadrangle_rights, quadrangle_matrices);

    double t4 = clock();
    // Warp to a rectangle bill
    cv::Mat rectange_bill = cv::Mat::zeros(quadrangle_bill.size(), CV_8UC3);
    cv::Mat rectangle_matrix;
    warp_to_rectangle_bill(quadrangle_bill, rectange_bill, quadrangle_corners, rectangle_matrix);

    double t5 = clock();
    // Set the rectangle check points
    std::vector<cv::Point2f> rectangle_check_points(20);
    set_rectangle_check_points(rectangle_check_points);

    // Get original check points
    std::vector<cv::Point2f> original_check_points = get_original_points(rectangle_check_points, rectangle_matrix, quadrangle_matrices, medial_matrices, rectange_bill, image, original_image);

    // Get original check quadrangles, rectangles
    std::vector<std::vector<cv::Point2f>> original_check_quadrangles;
    std::vector<cv::Rect> original_check_rectangles;
    get_original_rectangles(original_check_points, original_check_quadrangles, original_check_rectangles);

    // Get check boxes information
    std::vector<cv::Mat> check_boxes;
    get_information(original_image, check_boxes, original_check_quadrangles, original_check_rectangles);

    // Get moving distances
    std::vector<float> moving_distances = get_moving_distances(check_boxes);

    // Set rectangle points
    std::vector<cv::Point2f> rectangle_points = set_rectangle_points(moving_distances);

    ///Get original points
    std::vector<cv::Point2f> original_points = get_original_points(rectangle_points, rectangle_matrix, quadrangle_matrices, medial_matrices, rectange_bill, image, original_image);

    // Get original check quadrangles, rectangles
    std::vector<std::vector<cv::Point2f>> original_quadrangles;
    std::vector<cv::Rect> original_rectangles;
    get_original_rectangles(original_points, original_quadrangles, original_rectangles);

    // Get the billing information
    get_information(original_image, information_vector, original_quadrangles, original_rectangles);

    double t6 = clock();

    std::cout << "cv::Size image: " << image.size() << std::endl;
    std::cout << "Pre process: " << t2 - t1 << std::endl;
    std::cout << "Warp to medial bill: " << t3 - t2 << std::endl;
    std::cout << "Warp to quad bill: " << t4 - t3 << std::endl;
    std::cout << "Warp to rectangle bill: " << t5 - t4 << std::endl;
    std::cout << "Get billing information_vector: " << t6 - t5 << std::endl;
    std::cout << "Total time: " << t6 - t1 << std::endl;
}

void write_information_boxes(const std::vector<std::vector<cv::Mat>>& information_boxes)
{
    for (unsigned int i = 0; i < information_boxes.size(); i++)
    {
        for (unsigned int j = 0; j < information_boxes[i].size(); j++)
        {
            std::string str = "output_images_";
            str.append(std::to_string(j));
            str.append("/");
            if (i < 10)
            {
                str.append("0");
            }
            str.append(std::to_string(i));
            str.append(".png");
            cv::imwrite(str, information_boxes[i][j]);
        }
    }
}
