# cv-bug
Possible bug in OpenCV feature matching.
Please notice low number of inliers for ORB (all images with 2500 keypoints) and for AKAZE (between images 25 and 26, both with 1673 keypoints)


| File | Description |
|-----|-----|
| data-in/ | Input images |  
| data-out/ | Output data (from different program, this one is reduced to reproduce bug) |
| test.cpp | Console application for reproducing issue |

## Output

```
ALGORITHM="ORB2500"
Image=25; keypoints=2500vs2500; matches=2500; inliers=41 (1.64%)
Image=26; keypoints=2500vs2500; matches=2500; inliers=28 (1.12%)
Image=27; keypoints=2500vs2500; matches=2500; inliers=38 (1.52%)
ALGORITHM="AKAZE"
Image=25; keypoints=1670vs1673; matches=1224; inliers=1217 (99.4281%)
Image=26; keypoints=1673vs1673; matches=1673; inliers=69 (4.12433%)
Image=27; keypoints=1673vs1688; matches=1189; inliers=1176 (98.9066%)
```

## Links
https://stackoverflow.com/questions/63881433/opencv-4-4-0-knnmatch-does-not-seem-to-work-when-two-images-have-same-number-of
https://answers.opencv.org/question/235539/knnmatchbfhamming-bug-or-not/
