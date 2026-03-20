/**
 * @file postprocessing.hpp
 * @brief C++ OpenCV post-processing for segmentation masks.
 *
 * Provides contour extraction, morphological filtering, and
 * connected components analysis for refining segmentation outputs.
 */

#ifndef POSTPROCESSING_HPP
#define POSTPROCESSING_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <string>
#include <map>

namespace segmentation {

/**
 * @brief Morphological filtering operations on segmentation masks.
 */
class MorphologicalFilter {
public:
    /**
     * @brief Create a morphological filter.
     * @param kernel_size Size of the structuring element.
     * @param kernel_shape Shape of the structuring element (MORPH_RECT, MORPH_ELLIPSE).
     */
    MorphologicalFilter(int kernel_size = 5,
                        int kernel_shape = cv::MORPH_ELLIPSE);

    /** @brief Apply erosion to a binary mask. */
    cv::Mat erode(const cv::Mat& mask, int iterations = 1) const;

    /** @brief Apply dilation to a binary mask. */
    cv::Mat dilate(const cv::Mat& mask, int iterations = 1) const;

    /** @brief Apply morphological opening (erosion then dilation). */
    cv::Mat opening(const cv::Mat& mask) const;

    /** @brief Apply morphological closing (dilation then erosion). */
    cv::Mat closing(const cv::Mat& mask) const;

    /** @brief Apply gradient (dilation minus erosion). */
    cv::Mat gradient(const cv::Mat& mask) const;

    /**
     * @brief Apply full morphological pipeline to a multi-class mask.
     * @param mask Multi-class segmentation mask (CV_8UC1).
     * @param operations List of operations: "opening", "closing", "erosion", "dilation".
     * @return Processed mask.
     */
    cv::Mat processMask(const cv::Mat& mask,
                        const std::vector<std::string>& operations) const;

private:
    cv::Mat kernel_;
};

/**
 * @brief Contour extraction from segmentation masks.
 */
class ContourExtractor {
public:
    /**
     * @brief Extract contours for a specific class.
     * @param mask Multi-class segmentation mask.
     * @param class_id Class to extract contours for.
     * @return Vector of contours.
     */
    static std::vector<std::vector<cv::Point>>
    extractContours(const cv::Mat& mask, int class_id);

    /**
     * @brief Extract contours for all classes.
     * @param mask Multi-class segmentation mask.
     * @param num_classes Number of classes.
     * @return Map from class_id to vector of contours.
     */
    static std::map<int, std::vector<std::vector<cv::Point>>>
    extractAllContours(const cv::Mat& mask, int num_classes);

    /**
     * @brief Draw contours on an image.
     * @param image Input image (CV_8UC3).
     * @param contours Contours to draw.
     * @param color Drawing color.
     * @param thickness Line thickness.
     * @return Image with contours drawn.
     */
    static cv::Mat drawContours(const cv::Mat& image,
                                const std::vector<std::vector<cv::Point>>& contours,
                                const cv::Scalar& color = cv::Scalar(0, 255, 0),
                                int thickness = 2);
};

/**
 * @brief Connected components analysis for region filtering.
 */
class RegionFilter {
public:
    /**
     * @brief Remove small connected regions from a mask.
     * @param mask Multi-class segmentation mask.
     * @param min_area Minimum area threshold.
     * @return Filtered mask.
     */
    static cv::Mat removeSmallRegions(const cv::Mat& mask, int min_area = 100);

    /**
     * @brief Get statistics for connected components of each class.
     * @param mask Multi-class segmentation mask.
     * @param num_classes Number of classes.
     * @return Map from class_id to vector of areas.
     */
    static std::map<int, std::vector<int>>
    getRegionStats(const cv::Mat& mask, int num_classes);
};

/**
 * @brief Full post-processing pipeline.
 */
class PostProcessingPipeline {
public:
    /**
     * @brief Create pipeline with configuration.
     * @param kernel_size Morphological kernel size.
     * @param min_region_area Minimum region area to keep.
     * @param operations Morphological operations to apply.
     */
    PostProcessingPipeline(int kernel_size = 5,
                           int min_region_area = 100,
                           const std::vector<std::string>& operations = {"opening", "closing"});

    /**
     * @brief Run full pipeline on a mask.
     * @param mask Input segmentation mask.
     * @return Processed mask.
     */
    cv::Mat process(const cv::Mat& mask) const;

private:
    MorphologicalFilter morph_filter_;
    int min_region_area_;
    std::vector<std::string> operations_;
};

}  // namespace segmentation

#endif  // POSTPROCESSING_HPP
