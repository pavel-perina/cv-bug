/*
To compile modify property sheets according to your opencv setup
and copy these files (according to opencv version):
opencv_calib3d440d.dll
opencv_core440d.dll
opencv_features2d440d.dll
opencv_flann440d.dll
opencv_imgcodecs440d.dll
opencv_imgproc440d.dll
into x64\Debug directory.
Then modify directory containing test images at start of main function
*/
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <array>
#include <utility>
#include <vector>
#include <chrono>

struct IAlgorithm
{
    virtual cv::Ptr<cv::Feature2D> createDetector() = 0;
    virtual cv::Ptr<cv::DescriptorMatcher> createMatcher() = 0;
    virtual std::string getDetectorDesc() = 0;
    virtual std::string getMatcherDesc() = 0;
    virtual std::string getName() = 0;
};


struct AkazeAlgorithm : public IAlgorithm
{
    cv::Ptr<cv::Feature2D> createDetector() override { return cv::AKAZE::create(); }
    cv::Ptr<cv::DescriptorMatcher> createMatcher() override { return cv::DescriptorMatcher::create("BruteForce-Hamming"); }
    std::string getDetectorDesc() override { return std::string("AKAZE"); }
    std::string getMatcherDesc() override { return std::string("BruteForce-Hamming"); }
    std::string getName() override { return std::string("AKAZE"); }

};


struct Orb2500Algorithm : public IAlgorithm
{
    cv::Ptr<cv::Feature2D> createDetector() override
    {
        constexpr int nFeatures = 2500;
        return cv::ORB::create(nFeatures);
    }
    cv::Ptr<cv::DescriptorMatcher> createMatcher() override { return cv::DescriptorMatcher::create("BruteForce-Hamming"); }
    std::string getDetectorDesc() override { return "ORB2500"; }
    std::string getMatcherDesc() override { return std::string("BruteForce-Hamming"); }
    std::string getName() override { return std::string("ORB2500"); }
};


// https://github.com/thorikawa/akaze-opencv/blob/master/akaze/akaze_utils.cpp
void matches2points_nndr(const std::vector<cv::KeyPoint>& train,
    const std::vector<cv::KeyPoint>& query,
    const std::vector<std::vector<cv::DMatch> >& matches,
    std::vector<cv::Point2f>& pmatches, const float& nndr) 
{
    float dist1 = 0.0, dist2 = 0.0;
    for (size_t i = 0; i < matches.size(); i++) {
        const cv::DMatch dmatch = matches[i][0];
        dist1 = matches[i][0].distance;
        dist2 = matches[i][1].distance;

        if (dist1 < nndr*dist2) {
            pmatches.push_back(train[dmatch.queryIdx].pt);
            pmatches.push_back(query[dmatch.trainIdx].pt);
        }
    }
}

int main(int /*argc*/, char** /*argv*/)
{
    // Initialize stuff, modify as needed
    std::string dir = R"(c:\devel-c\cv-bug\data-in\)";
    std::string imageSet = "por";
    const int imgFrom = 24;
    const int imgTo   = 27;

    const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio

    std::array<std::unique_ptr<IAlgorithm>, 2> algorithms {
        std::make_unique<Orb2500Algorithm>(),
        std::make_unique<AkazeAlgorithm>()
    };

    // Test routines
    for (std::unique_ptr<IAlgorithm> &algoPtr : algorithms) {
        IAlgorithm &algorithm = *algoPtr.get();
        std::cout << "ALGORITHM=\"" << algorithm.getName() << '\"' << std::endl;
        cv::Ptr<cv::Feature2D>         detector = algorithm.createDetector();
        cv::Ptr<cv::DescriptorMatcher> matcher  = algorithm.createMatcher();

        std::vector<cv::KeyPoint> kptsA, kptsB;
        cv::Mat descA, descB;

        for (int imgIdx = imgFrom; imgIdx <= imgTo; ++imgIdx) {
            // Reuse previous keypoints and descriptors
            if (imgIdx != imgFrom) {
                kptsA = kptsB;
                descA = descB;
            }

            // Read grayscale 8-bit image (not using ImreadModes::ANY_DEPTH)
            std::ostringstream oss;
            oss << dir << imageSet << '-' << std::setw(2) << std::setfill('0') << imgIdx << ".png";
            const std::string fName = oss.str();
            cv::Mat img = cv::imread(fName, cv::ImreadModes::IMREAD_GRAYSCALE);

            try {
                // Detect and describe keypoints
                detector->detectAndCompute(img, cv::noArray(), kptsB, descB);
                if (imgIdx == imgFrom)
                    continue;

                // Match keypoints
                std::vector<std::vector<cv::DMatch> > dmatches;
                std::vector<cv::Point2f> matches, inliers;
                std::pair <std::vector<cv::Point2f>, std::vector<cv::Point2f>> kptsPairs;
                matcher->knnMatch(descA, descB, dmatches, 2);

                // Filter matches
                matches2points_nndr(kptsA, kptsB, dmatches, matches, nn_match_ratio);
                const size_t nMatches = matches.size() / 2;
                kptsPairs.first.resize(nMatches);
                kptsPairs.second.resize(nMatches);
                for (size_t i = 0, j = 0; i < nMatches; i++) {
                    kptsPairs.first[i]  = matches[j++];
                    kptsPairs.second[i] = matches[j++];
                }

                // Estimate transform matrix
                cv::Mat inliersMask = cv::Mat::zeros((int)kptsPairs.first.size(), 1, CV_8UC1);
                cv::Mat matA = cv::estimateAffinePartial2D(kptsPairs.first, kptsPairs.second, inliersMask, cv::FM_RANSAC, 10.0, 15000 );

                // Stats
                size_t nInliers = 0;
                for (int i = 0; i < nMatches; i++) {
                    if (inliersMask.at<uint8_t>(i) == 1)
                        ++nInliers;
                }
                /*
                double scale = sqrt(matA.at<double>(0, 0)*matA.at<double>(0, 0) + matA.at<double>(1, 0)*matA.at<double>(1, 0));
                double shiftX = matA.at<double>(0, 2);
                double shiftY = matA.at<double>(1, 2);
                double angle = atan2(matA.at<double>(1, 0), matA.at<double>(0, 0));
                */
                std::cout << "Image=" << imgIdx
                          << "; keypoints=" << kptsA.size() << "vs" << kptsB.size()
                          << "; matches=" << nMatches
                          << "; inliers=" << nInliers << " (" << 100.0 * nInliers / nMatches << "%)"
                          << std::endl;
            }
            catch (std::exception &e) {
                std::cout << e.what() << std::endl;
            }
        } // for img
    } // for alg
    return 0;
}
