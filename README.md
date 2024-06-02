
# OCR




## Imported Libraries

To run the final_pipeline.py file, you will need to add the following Libraries to your virtual environment. you can use anaconda to create a new virtual environment.

`opencv-python`   `Version: 4.6.0.66` https://pypi.org/project/opencv-python/

`numpy`   `Version: 1.23.5` https://numpy.org/install/

`pdf2image`   `Version: 1.17.0` https://pypi.org/project/pdf2image/

`pillow`   `Version: 10.3.0`

`rembg`   `Version: 2.0.57`    https://pypi.org/project/rembg/

`paddleocr`   `Version: 2.7.3` https://pypi.org/project/paddleocr/  and https://pypi.org/project/paddlepaddle/

`pytesseract`   `Version: 0.3.13` https://pypi.org/project/pytesseract/

`pathlib`   `Version: 1.0.1`

`tensorflow`   `Version: 2.12.0`  https://www.tensorflow.org/install

`keras`   `Version: 3.3.3` https://keras.io/getting_started/

`protobuf`   `Version: 3.20.1`
## About final code 

• Our test file includes both digital and scanned images.

• Scanned images do not require any pre-processing due to their high resolution.

• Pre-trained models used:

        PyTesseract
        PaddleOCR

• Utilizing these models helps save image processing time and provides quick output.

• A third, custom model is available:

Trained specifically for scanned images.

Requires more processing time than digital images.

Used less frequently due to the lower number of scanned images.

• this pyhton file will ask you to enter the path of the directory where your pdf files are stored and it will also ask you to enter the keyword. after that it will create a new folder named with that keyword  containing all the pdfs which had that keyword.





## Appendix

In order to use pdf2image make sure that Poppler is installed on your system and its binaries are included in the system's PATH environment variable. 

you can install the Poppler zip file from here https://github.com/oschwartz10612/poppler-windows/releases/tag/v24.02.0-0

Extract the contents of the downloaded ZIP file to a folder.

#### Add the path to the folder containing Poppler binaries to the system's PATH environment variable:

1. Right-click on "This PC" or "My Computer" and select "Properties".
2. Click on "Advanced system settings" on the left side.

3. In the System Properties window, click on the "Environment Variables" button.

4. Under "System variables", find the "Path" variable and double click then click on "new"

5. Add the path to the folder containing Poppler binaries (e.g., C:\path\to\poppler-xx.xx.x\bin).

6. Click "OK" to save the changes.

### for PyTesseract

1. After freshly installing the PyTesseract you might not be able to use it for  danish or any other language.

2. for that you need to install some language data and tess data 

you can refer to this youtube video link https://youtu.be/SSdQyvl5MUk?si=EI-NsKOx-6pG8wuM
