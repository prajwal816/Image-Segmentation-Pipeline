/**
 * @file postprocessing.cpp
 * @brief Implementation of C++ OpenCV post-processing for segmentation masks.
 *
 * Provides a CLI tool that reads a segmentation mask image, applies
 * morphological filtering, contour extraction, and region filtering,
 * then writes the processed output.
 *
 * Usage:
 *   ./postprocessing <input_mask> <output_mask> [kernel_size] [min_area]
 */

#include "postprocessing.hpp"

#include <iostream>
#include <algorithm>
#include <set>

namespace segmentation {

// ============================================================================
// MorphologicalFilter Implementation
// ============================================================================

MorphologicalFilter::MorphologicalFilter(int kernel_size, int kernel_shape) {
    kernel_ = cv::getStructuringElement(
        kernel_shape, cv::Size(kernel_size, kernel_size));
}

cv::Mat MorphologicalFilter::erode(const cv::Mat& mask, int iterations) const {
    cv::Mat result;
    cv::erode(mask, result, kernel_, cv::Point(-1, -1), iterations);
    return result;
}

cv::Mat MorphologicalFilter::dilate(const cv::Mat& mask, int iterations) const {
    cv::Mat result;
    cv::dilate(mask, result, kernel_, cv::Point(-1, -1), iterations);
    return result;
}

cv::Mat MorphologicalFilter::opening(const cv::Mat& mask) const {
    cv::Mat result;
    cv::morphologyEx(mask, result, cv::MORPH_OPEN, kernel_);
    return result;
}

cv::Mat MorphologicalFilter::closing(const cv::Mat& mask) const {
    cv::Mat result;
    cv::morphologyEx(mask, result, cv::MORPH_CLOSE, kernel_);
    return result;
}

cv::Mat MorphologicalFilter::gradient(const cv::Mat& mask) const {
    cv::Mat result;
    cv::morphologyEx(mask, result, cv::MORPH_GRADIENT, kernel_);
    return result;
}

cv::Mat MorphologicalFilter::processMask(
    const cv::Mat& mask,
    const std::vector<std::string>& operations) const {

    // Find unique classes
    std::set<int> classes;
    for (int r = 0; r < mask.rows; ++r) {
        for (int c = 0; c < mask.cols; ++c) {
            int val = static_cast<int>(mask.at<uchar>(r, c));
            if (val > 0) classes.insert(val);
        }
    }

    cv::Mat result = cv::Mat::zeros(mask.size(), CV_8UC1);

    for (int cls : classes) {
        // Create binary mask for this class
        cv::Mat binary;
        cv::compare(mask, cv::Scalar(cls), binary, cv::CMP_EQ);

        // Apply operations sequentially
        for (const auto& op : operations) {
            if (op == "opening") {
                binary = opening(binary);
            } else if (op == "closing") {
                binary = closing(binary);
            } else if (op == "erosion") {
                binary = erode(binary);
            } else if (op == "dilation") {
                binary = dilate(binary);
            } else if (op == "gradient") {
                binary = gradient(binary);
            }
        }

        // Merge back: assign class label where binary mask is non-zero
        result.setTo(cv::Scalar(cls), binary);
    }

    return result;
}

// ============================================================================
// ContourExtractor Implementation
// ============================================================================

std::vector<std::vector<cv::Point>>
ContourExtractor::extractContours(const cv::Mat& mask, int class_id) {
    cv::Mat binary;
    cv::compare(mask, cv::Scalar(class_id), binary, cv::CMP_EQ);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);
    return contours;
}

std::map<int, std::vector<std::vector<cv::Point>>>
ContourExtractor::extractAllContours(const cv::Mat& mask, int num_classes) {
    std::map<int, std::vector<std::vector<cv::Point>>> all_contours;

    for (int cls = 1; cls < num_classes; ++cls) {
        auto contours = extractContours(mask, cls);
        if (!contours.empty()) {
            all_contours[cls] = contours;
        }
    }

    return all_contours;
}

cv::Mat ContourExtractor::drawContours(
    const cv::Mat& image,
    const std::vector<std::vector<cv::Point>>& contours,
    const cv::Scalar& color,
    int thickness) {

    cv::Mat result = image.clone();
    cv::drawContours(result, contours, -1, color, thickness);
    return result;
}

// ============================================================================
// RegionFilter Implementation
// ============================================================================

cv::Mat RegionFilter::removeSmallRegions(const cv::Mat& mask, int min_area) {
    std::set<int> classes;
    for (int r = 0; r < mask.rows; ++r) {
        for (int c = 0; c < mask.cols; ++c) {
            int val = static_cast<int>(mask.at<uchar>(r, c));
            if (val > 0) classes.insert(val);
        }
    }

    cv::Mat result = cv::Mat::zeros(mask.size(), CV_8UC1);

    for (int cls : classes) {
        cv::Mat binary;
        cv::compare(mask, cv::Scalar(cls), binary, cv::CMP_EQ);

        cv::Mat labels, stats, centroids;
        int num_labels = cv::connectedComponentsWithStats(
            binary, labels, stats, centroids);

        for (int label = 1; label < num_labels; ++label) {
            int area = stats.at<int>(label, cv::CC_STAT_AREA);
            if (area >= min_area) {
                cv::Mat label_mask;
                cv::compare(labels, cv::Scalar(label), label_mask, cv::CMP_EQ);
                result.setTo(cv::Scalar(cls), label_mask);
            }
        }
    }

    return result;
}

std::map<int, std::vector<int>>
RegionFilter::getRegionStats(const cv::Mat& mask, int num_classes) {
    std::map<int, std::vector<int>> stats_map;

    for (int cls = 1; cls < num_classes; ++cls) {
        cv::Mat binary;
        cv::compare(mask, cv::Scalar(cls), binary, cv::CMP_EQ);

        if (cv::countNonZero(binary) == 0) continue;

        cv::Mat labels, stats, centroids;
        int num_labels = cv::connectedComponentsWithStats(
            binary, labels, stats, centroids);

        std::vector<int> areas;
        for (int label = 1; label < num_labels; ++label) {
            areas.push_back(stats.at<int>(label, cv::CC_STAT_AREA));
        }
        stats_map[cls] = areas;
    }

    return stats_map;
}

// ============================================================================
// PostProcessingPipeline Implementation
// ============================================================================

PostProcessingPipeline::PostProcessingPipeline(
    int kernel_size,
    int min_region_area,
    const std::vector<std::string>& operations)
    : morph_filter_(kernel_size),
      min_region_area_(min_region_area),
      operations_(operations) {}

cv::Mat PostProcessingPipeline::process(const cv::Mat& mask) const {
    // Step 1: Morphological filtering
    cv::Mat processed = morph_filter_.processMask(mask, operations_);

    // Step 2: Remove small regions
    processed = RegionFilter::removeSmallRegions(processed, min_region_area_);

    return processed;
}

}  // namespace segmentation

// ============================================================================
// CLI Entry Point
// ============================================================================

void printUsage(const char* program) {
    std::cout << "Usage: " << program
              << " <input_mask> <output_mask> [kernel_size=5] [min_area=100]"
              << std::endl;
    std::cout << std::endl;
    std::cout << "Post-process a segmentation mask using morphological"
              << " filtering and region filtering." << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  input_mask   Path to input mask image (grayscale, class IDs)"
              << std::endl;
    std::cout << "  output_mask  Path to save processed mask" << std::endl;
    std::cout << "  kernel_size  Morphological kernel size (default: 5)"
              << std::endl;
    std::cout << "  min_area     Minimum region area to keep (default: 100)"
              << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    int kernel_size = (argc > 3) ? std::atoi(argv[3]) : 5;
    int min_area = (argc > 4) ? std::atoi(argv[4]) : 100;

    // Read input mask
    cv::Mat mask = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
    if (mask.empty()) {
        std::cerr << "Error: Could not read mask image: " << input_path
                  << std::endl;
        return 1;
    }

    std::cout << "Input mask: " << input_path
              << " (" << mask.cols << "x" << mask.rows << ")" << std::endl;
    std::cout << "Kernel size: " << kernel_size << std::endl;
    std::cout << "Min region area: " << min_area << std::endl;

    // Create pipeline and process
    segmentation::PostProcessingPipeline pipeline(
        kernel_size, min_area, {"opening", "closing"});

    cv::Mat processed = pipeline.process(mask);

    // Extract and report contours
    auto all_contours = segmentation::ContourExtractor::extractAllContours(
        processed, 256);

    int total_contours = 0;
    for (const auto& pair : all_contours) {
        std::cout << "Class " << pair.first << ": "
                  << pair.second.size() << " contours" << std::endl;
        total_contours += static_cast<int>(pair.second.size());
    }
    std::cout << "Total contours: " << total_contours << std::endl;

    // Save output
    cv::imwrite(output_path, processed);
    std::cout << "Output saved: " << output_path << std::endl;

    return 0;
}
