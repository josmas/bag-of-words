# bag-of-words

## Original author
This is a fork of [bikz05's](https://github.com/bikz05/bag-of-words) work.

### Main changes
I am using OpenCV3 and a number of things have been moved to the [opencv_contrib](https://github.com/Itseez/opencv_contrib/) repo.
Make sure you install the _xfeatures2d_ module to be able to use [SIFT](http://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html#gsc.tab=0).


## Instructions
![alt text](docs/images/bog.png)
Python Implementation of Bag of Words for Image Recognition using OpenCV and
sklearn | [Video](https://www.youtube.com/watch?v=Ba_4wOpbJJM)

## Training the classifier
`python findFeatures.py -t dataset/train/`

## Testing the classifier
* Testing a number of images
`python getClass.py -t dataset/test --visualize`

The `--visualize` flag will display the image with the corresponding label printed on the image/

* Testing a single image
`python getClass.py -i dataset/test/aeroplane/test_1.jpg --visualize`

# Troubleshooting

If you get 

`
AttributeError: 'LinearSVC' object has no attribute 'classes_'
`

error, then simply retrain the model. 
