# STOCK-MARKET-PREDICTOR

![](https://github.com/ambujalpha/STOCK-MARKET-PREDICTOR/blob/master/Images_for_readme/about.jpg)
# (I) Introduction : 

Business and finance sector is today the leader of the world's economy, stock market trading is a
major practice in the finance sector. Financial exchange predictions are always trickier when it
comes to stock market predictions. It is basically a technique where one tries to predict the future
value of current stocks of a company to avoid the loss or perhaps gain profit. This project will
demonstrate a machine learning approach to predict the same using various quantities mentioned
later in this report. Python is the programming language used for better reach and understanding. We
propose a Machine Learning Algorithm which will be trained from different datasets of some
companies available from the past to make near effective predictions.
Stock market prediction is a technique to determine the upcoming worth of a corporation’s stock or
other financial instrument traded on an exchange. A noticeable consequential gain is the sole
purpose of stock market prediction, and, of course, to avoid significant losses. Some individuals may
disagree with the authenticity of results that these predictions considering the efficient market
hypothesis that these predictions cannot be made on the presently available data, thus concluding it
as inherently unpredictable. But there are numerous tools and technologies that help to gain future
trends’ information, thus resulting in effective profits

# (II) Dataset Used : 

Apple orchards in the U.S. are under constant threat from a large number of pathogens and insects. Appropriate and timely deployment of disease management depends on early disease detection. Incorrect and delayed diagnosis can result in either excessive or inadequate use of chemicals, with increased production costs, environmental, and health impacts. We have manually captured 3,651 high-quality, real-life symptom images of multiple apple foliar diseases, with variable illumination, angles, surfaces, and noise. A subset, expert-annotated to create a pilot dataset for apple scab, cedar apple rust, and healthy leaves, was made available to the Kaggle community for 'Plant Pathology Challenge'; part of the Fine-Grained Visual Categorization (FGVC) workshop at CVPR 2020 (Computer Vision and Pattern Recognition). 

**https://arxiv.org/abs/2004.11958** 

The dataset is uploaded on my google drive and can also be accessed on Kaggle. Following are the links : 
* Kaggle :  *https://www.kaggle.com/c/plant-pathology-2020-fgvc7/data*
* G-Drive : *https://drive.google.com/drive/folders/1IosfDui0TSxy22WQJuN8hYNtQWt5ocXY?usp=sharing* 

The dataset composed of about :
* 1,821 training images. Each image belonged to either of the mentioned catregory : 
  * Healthy
  * Multiple Diseased
  * Rust
  * Scab
* 1,821 testing images take from the same source distribution. Our goal was to classify these images into either of the four aforementioned classes.

Some sample images from the training dataset : 

![](https://github.com/CodingWitcher/Leaf_Diseases/blob/master/images_for_readme/sample1.jpg)
![](https://github.com/CodingWitcher/Leaf_Diseases/blob/master/images_for_readme/sample2.jpg)
![](https://github.com/CodingWitcher/Leaf_Diseases/blob/master/images_for_readme/sample3.jpg)
![](https://github.com/CodingWitcher/Leaf_Diseases/blob/master/images_for_readme/sample4.jpg)

# (III) Libraries used :

## Numpy :

Fundamental package for scientific computing in Python3, helping us in creating and managing n-dimensional tensors. A vector can be regarded as a 1-D tensor, matrix as 2-D, and so on. 

![](https://github.com/ambujalpha/STOCK-MARKET-PREDICTOR/tree/master/Images_for_readme/tensor.jpg)
## Matplotlib :

A Python3 plotting library used for data visualization.

![](https://github.com/ambujalpha/STOCK-MARKET-PREDICTOR/tree/master/Images_for_readme/visualize.webp)
## Tensorflow-Keras :

Is an open source deep learning framework for dataflow and differentiable programming. It’s created and maintained by Google.

![](https://github.com/ambujalpha/STOCK-MARKET-PREDICTOR/tree/master/Images_for_readme/tf%20loves%20keras.png)
## Pandas :
Used for data manipulation and analysis. In particular, it offers data structures and operations for
manipulating numerical tables and time series.

## Sci-kit Learn :
Scikit-learn is a free software machine learning library for the Python
programming language. It features various classification, regression and clustering
algorithms including support vector machines and many more.

![](https://github.com/ambujalpha/STOCK-MARKET-PREDICTOR/tree/master/Images_for_readme/sklearn.png)

# (IV) Exploratory Data Analysis(EDA) : 

When we’re getting started with a machine learning (ML) project, one critical principle to keep in mind is that data is everything. It is often said that if ML is the rocket engine, then the fuel is the (high-quality) data fed to ML algorithms. However, deriving truth and insight from a pile of data can be a complicated and error-prone job. To have a solid start for our ML project, it always helps to analyze the data up front.

During EDA, it’s important that we get a deep understanding of:

* The **properties of the data**, such as schema and statistical properties;
* The **quality of the data**, like missing values and inconsistent data types;
* The **predictive power of the data**, such as correlation of features against target.

Firstly, the number of various examples belonging to each class were identified and plotted.

![](https://github.com/CodingWitcher/Leaf_Diseases/blob/master/images_for_readme/eda_01.png)

Secondly, the color channel distributions were analyzed for the images from the dataset and plotted using seaborn : 

![](https://github.com/CodingWitcher/Leaf_Diseases/blob/master/images_for_readme/eda_02.png)
![](https://github.com/CodingWitcher/Leaf_Diseases/blob/master/images_for_readme/eda_03.png)
![](https://github.com/CodingWitcher/Leaf_Diseases/blob/master/images_for_readme/eda_04.png)

**Inference drawn** : 

* Red channel has positive skew, meaning the values are more concentrated at intensities lower than mean(somewhere around 90).
* Green channel is negative skew, meaning the values are more concentrated at intentities higher than mean(somewhere in the range 130-150). This also means that green channel is more pronounced than red in the sample image set; and thereby the whole data set as they come from the same distribution. This makes sense as images are that of leaves!
Similarily, blue channel has a slight positive skew and is very well distributed.
* The distribution of red and green color channels appears to be mesokurtic, aka normally distributed having k = 0 whereas the blue one appears to be relatively platykurtic having k < 0. Therefore out of the three colors, blue channel appears to be the most different one(relative outlier in the RGB color space).

Post this, a sample image is randomly taken on which we will test fire the coming functions in the **Image Processing** segment.

Sample image : 

![](https://github.com/CodingWitcher/Leaf_Diseases/blob/master/images_for_readme/sample_image.png)

# (V) Image Processing : 

## Image Denoising : 

Many image smoothing techniques like Gaussian Blurring, Median Blurring etc were good to some extent in removing small quantities of noise. In those techniques, we took a small neighbourhood around a pixel and performed some operations like gaussian weighted average, median of the values etc to replace the central element. In short, noise removal at a pixel was local to its neighbourhood.

There is a property of noise. Noise is generally considered to be a random variable with zero mean.

Suppose we hold a static camera to a certain location for a couple of seconds. This will give us plenty of frames, or a lot of images of the same scene. Then averaging all the frames, we compare the final result and first frame. Reduction in noise would be easily observed.

So idea is simple, we need a set of similar images to average out the noise. Considering a small window (say 5x5 window) in the image, chance is large that the same patch may be somewhere else in the image. Sometimes in a small neighbourhood around it. Hence, using these similar patches together averaging them can lead to an efficient denoised image.

This method is Non-Local Means Denoising. It takes more time compared to blurring techniques, but the result are very satisfying.

Denoising illustration :

![](https://github.com/CodingWitcher/Leaf_Diseases/blob/master/images_for_readme/lena_denoised.png)

Following was the output when this function was test fired on our sample image : 

![](https://github.com/CodingWitcher/Leaf_Diseases/blob/master/images_for_readme/denoised_sample.png)

## Edge Detection Using Sobel Filter : 

Edge detection is one of the fundamental operation in image processing. Using this, we can reduce the amount of pixels while maintaining the structural aspect of the images.

The basic operation involved behind edge detection is called Convolution and is illustrated below : 

![](https://github.com/CodingWitcher/Leaf_Diseases/blob/master/images_for_readme/edge_detection_sobel.png)

Edges can be detected using various kinds of filters.

* First derivative based Sobel filter(for thicker edges)
* Second derivative based Laplacian filter(for finer edges)

Here, we want to consider the area containing only the leaf, while ignoring the background green. Hence, we use Sobel filter to identify the prominent edge of the leaf.

Following was the output when this function was test fired on our sample image :

![](https://github.com/CodingWitcher/Leaf_Diseases/blob/master/images_for_readme/sobel_results.png)

Using sobel filter we found the edges, however for further pre-processing we aim to consider only the area of the leaf, that is the fine textured area we see in the gradient images. For that, we will use a much powerful inbuilt function of open-CV called Canny(). This function will return the edge coordinates.

Entire read is available on the OpenCV webpage :

*https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html#canny*

## Canny Edge Detector : 

The Canny filter is a multi-stage edge detector. It uses a filter based on the derivative of a Gaussian in order to compute the intensity of the gradients.The Gaussian reduces the effect of noise present in the image. Then, potential edges are thinned down to 1-pixel curves by removing non-maximum pixels of the gradient magnitude. Finally, edge pixels are kept or removed using hysteresis thresholding on the gradient magnitude.

The Canny has three adjustable parameters: the width of the Gaussian (the noisier the image, the greater the width), and the low and high threshold for the hysteresis thresholding.

The Canny edge detection algorithm is composed of 5 steps:

* Noise reduction
* Gradient calculation
* Non-maximum suppression
* Double threshold
* Edge Tracking by Hysteresis

![](https://github.com/CodingWitcher/Leaf_Diseases/blob/master/images_for_readme/harry%20potter.png)

* **Noise Reduction** - One way to get rid of the noise on the image, is by applying Gaussian blur to smooth it. To do so, image convolution technique is applied with a Gaussian Kernel (3x3, 5x5, 7x7 etc…). The kernel size depends on the expected blurring effect. Basically, the smallest the kernel, the less visible is the blur.
* **Gradient Calculation** - The Gradient calculation step detects the edge intensity and direction by calculating the gradient of the image using edge detection operators.
The result is almost the expected one, but we can see that some of the edges are thick and others are thin. Non-Max Suppression step will help us mitigate the thick ones.
* **Non-Maximum Supression** - Ideally, the final image should have thin edges. Thus, we must perform non-maximum suppression to thin out the edges.
* **Double Threshold** - The double threshold step aims at identifying 3 kinds of pixels: strong, weak, and non-relevant: **Strong pixels** are pixels that have an intensity so high that we are sure they contribute to the final edge. **Weak pixels** are pixels that have an intensity value that is not enough to be considered as strong ones, but yet not small enough to be considered as non-relevant for the edge detection. **Other pixels** are considered as non-relevant for the edge.

Therefore, the significance of having two values in double threshold :
 - High threshold is used to identify the strong pixels (intensity higher than the high threshold)
 - Low threshold is used to identify the non-relevant pixels (intensity lower than the low threshold)
 - All pixels having intensity between both thresholds are flagged as weak and the Hysteresis mechanism (next step) will help us identify the ones that could be considered as strong and the ones that are considered as non-relevant.

* **Hysterisis** - Based on the threshold results, the hysteresis consists of transforming weak pixels into strong ones, if and only if at least one of the pixels around the one being processed is a strong one. We will be using OpenCV's implementation of Canny edge detection. This was the theory involved behind the entire process.

Further information can be found on OpenCV's documentation : *https://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html*

Following was the output when this function was test fired on our sample image :

![](https://github.com/CodingWitcher/Leaf_Diseases/blob/master/images_for_readme/canny.png)

## ROI and Edge ROI Image : 

Using the canny version, we find our region of interest and therefore crop some fraction of the image. This reduces the processing data to some extent per image.

![](https://github.com/CodingWitcher/Leaf_Diseases/blob/master/images_for_readme/roi_and_edge.png)

## Local Histogram Equalization : 

**First of all, why can we not apply histogram equalization directly to an RGB image?**

Histogram equalization is a non-linear process. Channel splitting and equalizing each channel separately is incorrect. Equalization involves intensity values of the image, not the color components. So for a simple RGB color image, histogram equalization cannot be applied directly on the channels. It needs to be applied in such a way that the intensity values are equalized without disturbing the color balance of the image. So, the first step is to convert the color space of the image from RGB into one of the color spaces that separates intensity values from color components. Some of the possible options are HSV/HLS, YUV, YCbCr, etc. YCbCr is preferred as it is designed for digital images. Perform histogram equalization on the intensity plane Y. Now convert the resultant YCbCr image back to RGB.

(Excerpt taken from :

*https://prateekvjoshi.com/2013/11/22/histogram-equalization-of-rgb-images/* )

![](https://github.com/CodingWitcher/Leaf_Diseases/blob/master/images_for_readme/histogram__result.png)

## Image Segmentation(Half-Toned Images) : Otsu's Binarization 

In global thresholding, we used an arbitrary chosen value as a threshold. In contrast, Otsu's method avoids having to choose a value and determines it automatically. We will apply Otsu's binarization segmentation method on the histogram equalized image obtained in the previous stage.

![](https://github.com/CodingWitcher/Leaf_Diseases/blob/master/images_for_readme/otsu.png)

## Image Segmentation(Colored Images) : K-means Clustering

In the previous section we explored image segmentation using Otsu's Binarization. However, this is applied normally on half toned, that is binary(black and white) images. In this section, we will explore a Machine Learning technique called K-means clustering to segment the different areas of the image.

Once again, the operation will be performed on the histogram equalized image of plant leaf.

K-means is mostly useful for applications like image compression or object recognition, because for these types of applications, it is inefficient to process the whole image.

![](https://github.com/CodingWitcher/Leaf_Diseases/blob/master/images_for_readme/kmeans.png)

# (VI) Handling Imbalanced Dataset :

In Exploratory Data Analysis, we plotted a pie chart depticting number of classes in the training data set. It was observed that the 'multiple diseased' class was in minority, hence creating an imbalance between the class distributions. In order to cater that, we will be using **SMOTE algorithm** to try and increase the minority class in our data set.

**SMOTE stands for Synthetic Minority Oversampling Technique**.

![](https://github.com/CodingWitcher/Leaf_Diseases/blob/master/images_for_readme/SMOTE.png)

This algorithm aims to balance class distribution by randomly increasing minority class examples by replicating them. SMOTE synthesizes new minority instances between existing minority instances.

An amazing video explaining it can be found on YouTube : *https://www.youtube.com/watch?v=FheTDyCwRdE*

**Output after applying SMOTE** : 

![](https://github.com/CodingWitcher/Leaf_Diseases/blob/master/images_for_readme/SMOTE_output.png) 

Now, it's a well balanced dataset :-)

# (VII) Transfer Learning with Fine Tuning :

Transfer learning allows us to train deep networks using significantly less data then we would need if we had to train from scratch. With transfer learning, we are in effect transferring the “knowledge” that a model has learned from a previous task, to our current one.

Transfer learning has been consistently proven to boost model accuracy and reduce require training time. Less data, less time, more accuracy.

Here, we will be using VGG-16 neural network architecture.

## VGG-16 Architecture : 

The VGG-16 model is a 16-layer (convolution and fully connected) network built on the ImageNet database. 

![](https://github.com/CodingWitcher/Leaf_Diseases/blob/master/images_for_readme/vgg_16_two.png)

![](https://github.com/CodingWitcher/Leaf_Diseases/blob/master/images_for_readme/VGG-16_one.png)

Here, in the fine tuning step we will freeze the initial pre-trained layers having the weights learned from the imagenet dataset, however the last few convolution-pooling layers will remain unfreezed, allowing us to train them on our dataset. Note that, we will be designing a custom head as the original VGG aimed to predict out of 1000 classes, whereas we only have four.

Also, it's been proven that adding a Global Average Pooling layer before the fully connected layer imroves the accuracy of the model considerably.

**Global Average Pooling. Global Average Pooling is an operation that calculates the average output of each feature map in the previous layer. This fairly simple operation reduces the data significantly and prepares the model for the final classification layer**.

* **Bottleneck features** : The last activation feature map in the VGG-16 model (output from block5_pool) gives us the bottleneck features, which can then be flattened and fed to a fully connected deep neural network classifier. We flatten the bottleneck features in the vgg_model object to make them ready to be fed to our fully connected classifier. *A way to save time in model training is to use this model and extract out all the features from our training and validation datasets and then feed them as inputs to our classifier*.

# (VIII) Results : 

We trained only on 100 epochs to visualize the trends in our loss and accuracy. *Following trends were observed for just 100 epochs. The network can be further trained for more accuracy.* 

**Here we achieved the accuracy close to 90%**.

![](https://github.com/CodingWitcher/Leaf_Diseases/blob/master/images_for_readme/res.png)

# Thank you for your time to read this far ~~

* Contact : *bauraiaditya7@gmail.com* 

![](https://github.com/CodingWitcher/Leaf_Diseases/blob/master/images_for_readme/ty.webp)



