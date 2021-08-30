# Automatic-Breast-Cancer-Detection-in-Mammograms

1.2 Introduction 
It is important to detect breast cancer as early as possible. Mammogram can serve as an elementary tool to detect breast cancer in early stages but analyzing mammogram is a nontrivial task, and decisions from investigation of these kinds of images always require specialized knowledge. So computer based automatic breast cancer detecting techniques can help the specialist (doctors and physicians) to make more reliable decisions.[2] In this report, a new methodology for classifying breast cancer using two different combination of deep learning model (hybrid deep convolutional neural network) and some image preprocessing techniques (Image Augmentation) are introduced. Here two-step process are used to detect breast cancer. First a well-known DCNN architecture named ResNet50 is used and is fine-tuned to identify the cancerous breast mammography images and second step involves a ROI(Region of Interest) based simple CNN model which used to detect the location of patch which contain benign and malignant mass tumors in cancerous mammography images. The following publicly available datasets are used to train our model (1) The miniMIAS database of mammograms (2) the digital database for screening mammography (DDSM)[3]; and (3) the Curated Breast Imaging Subset of DDSM (CBIS-DDSM).[4] For any CNN model training on a large number of data gives high accuracy rate. Since, the biomedical datasets contain a relatively small number of samples due to limited patient volume. Accordingly, in this project data augmentation method is used for increasing the size of the input data by generating new data from the original input data. There are many forms for the data augmentation; the one used here is combination of rotation, affine transformations and resizing. First we tried simple DCNN full-mammogram architecture on full mammogram but accuracy was very poor (around 62 % ) but after using DCNN architecture named ResNet50 on full mammogram accuracy achieved was 92% and when cropping the ROI manually from the mammogram accuracy achieved was 96% for the ROI samples obtained from Image Augmentation techniques.

## Automatically Detecting Lesions Process
The procedure to determine lesions automatically given a mammogram image is as follows: <br />
Get test mammogram image<br />
Convert it to a GRAYSCALE IMAGE and save it<br />
Do Mean subtraction and standardization <br />
Replicate it to make number of channels as 3 (RGB)<br />
Then pass test image to the ResNet50 Model and observe predicted output <br />
If predicted value is greater than threshold<br />
Then select GRAYSCALE IMAGE of full mammogram (step 2) and extract 1000 patches <br />
Do Mean subtraction and standardization for each patch<br />
Then use ROI based simple CNN model to detect lesion automatically <br />
<br />
<br />
CODE LINK<br />
https://colab.research.google.com/drive/1LU6ydPMY_movFdR4iikwKyi1m3Db7srW#scrollTo=z7xed4cN6Bbb
<br />
<br />
PPT LINK<br />
https://docs.google.com/presentation/d/1GSo1EZrYqXkregXzyZ4iXCF0tu53ZfIxCkgRUJHKTOE/edit#slide=id.g78bdc3fe3c_0_10
<br />
<br />
DATASET LINK:<br />
https://drive.google.com/drive/folders/1PpfjzoZC-faWqouWhkyIRU-RiI8rZrT_?usp=sharing<br />
