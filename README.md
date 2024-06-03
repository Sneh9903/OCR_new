
# OCR




## Steps

We will use anaconda navigator to create a virtual environment and install all the libraries in that environment

#### 1. Go to anaconda home page in your browser. (https://anaconda.cloud/) and click on install distribution button
#
![Screenshot 2024-06-03 163547](https://github.com/Sneh9903/OCR_new/assets/119620258/f67ddd7f-8046-4666-bc2c-9de42b313582)
#

#### 2. Enter your email address and sign in.
#
![Screenshot 2024-06-03 163615](https://github.com/Sneh9903/OCR_new/assets/119620258/9162a086-5bd3-4ff6-9b6a-23f82bcd37d0)
#
#### 3. click on the downlaod button
#
![Screenshot 2024-06-03 163720](https://github.com/Sneh9903/OCR_new/assets/119620258/d0535bb5-67b5-4366-9196-d7f2d83cf88b)
#
#### 4. After downloading the anaconda exe file, open it and set it up.
#
![Screenshot 2024-06-03 164536](https://github.com/Sneh9903/OCR_new/assets/119620258/b0b4f56b-854d-41f2-9791-4b0e4487ac85)

#
![Screenshot 2024-06-03 164724](https://github.com/Sneh9903/OCR_new/assets/119620258/26102ad9-0098-4e13-8be2-f2439c6b2be9)
#
#### 5. Now after installing anaconda open anaconda prompt
#
![Screenshot 2024-06-04 001600](https://github.com/Sneh9903/OCR_new/assets/119620258/d32ba0f3-2a28-4d95-bd93-219a5ee6da9b)
#
![Screenshot 2024-06-04 002511](https://github.com/Sneh9903/OCR_new/assets/119620258/019af396-2989-4cf0-86f4-e4c6a47a8f19)

#### 6. Now enter the following commands 

### `conda create -n test python=3.9.18`

this will create new environment name "test" with python version 3.9.18

#### Next

### `conda activate test`

To activate the test environment

#### Now install required libraries

### `pip install numpy==1.23.5`

### `pip install pillow==10.3.0`

### `pip install protobuf==3.20.1`

### `pip install pandas==1.3.5`

### `pip install pathlib==1.0.1`

### `pip install scipy==1.13.0`

### `pip install threadpoolctl==3.5.0`

### `pip install joblib==1.4.2`

### `pip install scikit-learn==1.5.0`

### `pip install opencv-python==4.6.0.66`

### `pip install keras==3.3.3`

### `pip install tensorflow==2.12.0`

here tensorflow will change the keras and protobuf version so we have to install keras and protobuf version again.

### `pip install keras==3.3.3`

### `pip install protobuf==3.20.1`

make sure you have git installed on your machine.

### `pip install -U git+https://github.com/madmaze/pytesseract.git`

### `pip install rembg==2.0.57`

### `pip install paddlepaddle==2.6.1`

### `pip install paddleocr==2.7.3`

### `pip install pdf2image==1.17.0`
## Some additional steps 

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








## About final code with demo

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

#### • This python file will ask you to enter the path of the directory where your pdf files are stored and it will also ask you to enter the keyword. after that it will create a new folder named with that keyword  containing all the pdfs which had that keyword.

Folder containing test pdfs

![Screenshot 2024-06-04 010230](https://github.com/Sneh9903/OCR_new/assets/119620258/6792ec61-3758-4ade-8652-20c8a07ba4d6)

Now on anaconda command prompt we will run python file.

1. first activate the environment

### `conda activate test`

![Screenshot 2024-06-04 010658](https://github.com/Sneh9903/OCR_new/assets/119620258/9285c726-3f0d-424b-bd7c-62830bb41caf)

2. Now run the python file

![Screenshot 2024-06-04 010758](https://github.com/Sneh9903/OCR_new/assets/119620258/c280dee1-97d6-40cd-9554-52b8223a6653)

3. enter the path of your test pdf folder and keyword to search for.

![Screenshot 2024-06-04 011226](https://github.com/Sneh9903/OCR_new/assets/119620258/e90985c2-5bb0-4b21-a254-59f32619efc8)

4. it will go through all the pdfs in test folder try to find for files containing keyword.

![Screenshot 2024-06-04 011356](https://github.com/Sneh9903/OCR_new/assets/119620258/d2b22d90-9811-4445-aefe-f634c26f5dbc)

5. Now it will create a new folder with name of keyword containing pdfs which it has found.

![Screenshot 2024-06-04 012139](https://github.com/Sneh9903/OCR_new/assets/119620258/2d34f5a1-197c-4f8b-99df-43ba432b5110)
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
