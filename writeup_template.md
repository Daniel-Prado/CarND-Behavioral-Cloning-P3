# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* preprocess.py containing pre-processing routines.
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 test driving in Track 1, recorded using provided video.py
* Track2_speedx2.mp4 test driving in Track 2, recorded (x2 speed) using 3rd party SW.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
Note it may be necessary to adjust the driving speed inside the drive.py file, depending on the difficulty of the track. The default speed of 10 mph should work in Track 2 without issues.

I have commented the prints of the angle and throttle values at each iteration, in order to free the CPU from the printing load that can affect performance significally. Instead I printed the elapsed time between 100 iterations of the driving loop, that takes tipically about 4 seconds in my computer. This helps to check that the pre-processing of the driving loop does not hinder the performance.

Apart from this, the only other modification in the provided drive.py file was to include the same pre-processing of the input images as used in the Training, which means:
* Crop the images 65 pixels from the top and 20 pixels from the bottom.
* Resize the cropped image to 64x64 pixels
* Transform the colorspace from RGB to YUV*.

YUV has worked better than RGB and HSV in my tests.

With regard to preprocessing, probably here is the part of the project where I spent more time due to a variety of reasons:
* First of all, I didn't realize until very late that the drive.py took the images from the simulator in RGB format. I was assuming that it was using BGR, as it used the training model, if using CV2.imwrite as shown in the video lessons. For this reasons all my initial attemps to make a fully working model were failing.
***Suggestion*** : The video lesson should make a remark on this to avoid that future students loose many hours because of this.

* I knew that ideally the preprocessing should be implemented within the KERAS model. 1st because that way you don't need to pre-process in python code (slow) in drive.py in realtime (which could lead to performance issues) and 2nd because KERAS should provide a more efficient implementation that eventually could take advantage of the GPU.
However, my attemps to do so failed, it is not straightforward to implement a Lambda in Keras that uses OpenCV functions inside...
* Finally, I wanted to try something 'original' for this project and I spent quite a lot of time trying a model than pre-processed the input images to transform them into a "bird-eye" view. This is discussed in an Appendix at the end of this document.



#### 3. Submission code is usable and readable

I have tried to present my code in a clear way, functionally-oriented (procedural) and including numerous comments.
Most of the code is self-contained in the model.py file except some pre-processing functions included in preprocess.py.
The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is identical to the one proposed in NVIDIA paper xxxxxx, being the only difference the size of the input images.
Besides of the suggested crop, I have highly reduced the size of the images, specially horizontally, resulting in a 64x64 size. The advantage of using this size is that it allows to use quite large training sets without the need of using a Generator.

More details and considerations are presented in section "Solution Design Approach"

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on a data set that I built by myself using both tracks in the simulator (more details in section 4.Appropiate Training Data), in order to avoid or reduce overfitting:

In order to reduce overfitting I applied L2 Regularization. I applied it only to the FC layers, which means that only the weights of the FC layers are accounted in the Loss function.
Initially I used a very reduced number of Epochs (3), but once I introduced L2 reg I could increase the Epochs to 5, that is what is implemented in the final model.
I have removed 80% of the images with Steering Angle=0.0 in the case of central camera, or 0.0 +/-0.08 correction in the case of L and R cameras. Of course the removed images were randomly chosen using a uniform distribution.

I used Data Augmentation (see function augment_images in model.py line 115), that consisted of:
* For every image, adding a horizontally flipped copy image (and corresponding flipped steering angle).
* For every image (not belonging to the 0-steering angle group), I created N\_MULTIPLY copies applying random shift traslation and random brightness reduction. The purpose was to reproduce more cases of roads with different degrees of shadow. (See functions shift\_image and transf_brightness in lines 28 and 51 respectively).

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. This has been tested as follows:
* Track 1: the model is capable of driving autonomously for 1 full lap or more, even at high speeds (30 mph).
* Track 2: the model is capable of driving autonomously for 1 full Lap or more. but only at low speeds (10 mph).
NOTE: In my latest tests, and as it can be observed in the videos, the model seems to work even better in Track 2 than in Track 1 (despite being much easier and having more training data from Track 1)... This is due to the fact that in the last days I spent a lot of time tweaking the parameters to be able to complete Track 2... which in the other turn has decreassed the smoothness of Track 1.
Here you can see my Track 2 test video uploaded to youtube: https://youtu.be/uBrxG5xMy6k

* Track 3: I have also tested the model in the 2nd track of the previous version of the Udacity simulator, which renders a mountain road. The model could run the car perfectly even at the highest speed of 30 mph, but only if I selected the "Fastest" video detail, that removes all shadows... Otherwise my model could not cope with the heavy dark shadows of the 1st turn.
Nevertheless, I found the performance in this Track specially satisfying because it showed that **the model could generalize to a road where it had not been trained at all !!**


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. As explained before I made my own training data, importantly to mention, using the "Simple" level of video detail of the simulator. This is important because the "Simple" mode is not so simple compared to the Fast and Fastest model. because it includes a realistic rendering of object shadows.
I could have built a model in a easier way by using "Fastest-video" training data that removes all shadows (and testing it the same way) but I wanted my model to be more realistic.
To build the training set, I recorded driving as follows:
* Training Mode - Drive Track 1 for 2 Laps,counter-clockwise, at medium speed (about 20mph).
* Training Mode - Drive Track 1 for 1 Lap, clockwise, at medium speed (about 20mph).
* Training Mode - Drive Track 2 for 1 Lap, clockwise, at low speed (about 10-15 mph).

The training dataset was not of very high quality, meaning that I was not very good at keep keeping in the middle part of the road, specially for Track 2 (I guess I am not a good videogame driver :-)

Mostly, I used center lane driving, although I am not particularly brilliant at that, and this is why the car tends to take the curves not through the center.

I didn't use "recovery driving" to teach the model to avoid running off the road.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

As mentioned before I used the NVIDIA model proposed in the video lesson and explained in more detail in the following paper:
https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

A representation of this model by NVIDIA is shown in the following figure:


I have tried numerous modifications of the NVIDIA model, namely:
* Adding RELU or PreLU layers between the Fully-connected (FC) layers, or only some of the FC layers.
* Adding DROPOUT between the FC layers or some of them, with different probability values (0.35, 0.50, 0.65...)
* Adding DROPOUT after the 1st convolutional layer.

Despite applying and testing these changes in a systematic controlled way, I have been unable to find a tweak that improves the performance of the original model.
My guess is that introducing non-linearities in the FC layers does not work so well because this project is about predicting an angle, that is a continuous real value, and not a classification problem as in the previous project.

The details about image pre-processing and normalization have already been discussed.

Once I corrected all the bugs (specially the RGB-BGR confussion previously mentioned), the car could drive well the Track 1 at almost the first try, using only the central camera. But when I tried the second track, it hit the walls or fell off the road after few curves. To make the model work in Track 2, I added the Left and Right cameras and I played with the correction angle parameters and with the parameters of the data augmentation (amount of multiplied data, degree of shifting and darkening...).
Another way would have been to re-train the model recording only the spots where the vehicle fell off, but finally I didn't need to take that approach. The model was able to complete Track 2 with only 1 training lap in that circuit.



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Including the 3 cameras, my training set consisted of a total of 23085 images.
Here is an example image of center lane driving, with the 3 cameras (L-C-R) in two segments of the two tracks:

![alt text][image2]

Below I show the result of the pre-processing (for brevity, I only show the pre-processing of the central camera images. Note also the the transformation from RGB to YUV colorspace is not shown here):

![alt text][image3]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]

After data augmentation, I got a total of 124,396 images, 10% of which I used for validation.


### Appendix - An alternative Preprocessing (Bird-Eye)
It is worth mentioning that I tried to implement a more "sofisticated" pre-processing consisting on applying a persperctive transform to the input images, in order to make the road look as a bird-eye image. Similarly as what is explained in the following Project 4 - Advanced Lane Find.
My idea was that if I managed to make the curve more distinct and 'aparent' in the image, the model could learn more easily its shapes and extract more easily its features.

![alt text][image7]
I managed to get a model that worked perfectly well in Track 1 and partially in Track 2. However, the model did not improve the performance of the finally presented model with light pre-processing.  My guess is that this could be due to two reasons:
* The perspective transform was well suited for Track 1, but not for the complex scenarios of Track 2, including different slope degrees.
* The perspective transform introduced artifacts in the top part of the images, due to the low resolution of the input images, that could hinder the model.

