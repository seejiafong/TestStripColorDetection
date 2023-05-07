# TestStripColorDetection
Water Test Strip Color Extraction
This repository extracts colors from a test strip using traditional Computer Vision Techniques
![image](https://user-images.githubusercontent.com/3718959/236669244-d09948ab-dedc-4322-adac-c5b9f421d50b.png)

## Clone repo
`git clone https://github.com/seejiafong/TestStripColorDetection.git`

## Navigate to directory
`cd TestStripColorDetection`

## (optional) create virtual environment
Assuming you have anaconda installed:
`conda create -n teststrip`
`conda activate teststrip`

## Install dependencies
`pip install -r requirements.txt`

## Run Code
`python ImageProcessing.py hold1_2.jpg`

## Result Verification
2 parts:
1. Check that the correct pixels were extracted: look for "6 final.jpg" that was generated
2. Check that the color matching looks correct: open "color.html" in a browser. The extracted colors may look a bit dull, you can refer back to the input image "hold1_2.jpg" to verify the colors.

## Known issues
### Bug in colormath package:

To fix the error below, go to the file and line number specified, change "asscalar()" to "isscalar()", then the code will work
![image](https://user-images.githubusercontent.com/3718959/236669137-fb72f82b-05ae-4905-9356-a2fa29015097.png)




