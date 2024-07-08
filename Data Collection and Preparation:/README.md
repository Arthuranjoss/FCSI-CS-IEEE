# Introduction:
This guide presents the steps for collecting data sets for training and developing the FCSI model of the IEEE Artificial Intelligence activity. This material will be reviewed and used for teaching purposes and will serve as a basis for project members to map activities.
Every Computer Vision project starts with a data collection and dataset creation strategy. In recent years, people have focused on developing models without investing as much as necessary in creating training data. In fact, creating an image dataset was really complicated and time-consuming, and often done by engineers or interns in a very inefficient way. Despite the existence of labeling tools and open source tools, unfortunately, many companies still face issues with the performance of their AI models.

Most of these issues come from the quality of the dataset itself by the amount of data in the dataset, the amount of mislabeled data in your dataset, or relevance of the images within the dataset.

# 1-Where to find a dataset?
Finding the right data is crucial to training a successful computer vision model for segmentation and classification. Here's how to navigate this process:
- Data sources:

  - Public datasets: Several public datasets are available online designed specifically for computer vision tasks. Popular repositories include:

  - ImageNet: A huge image database with millions of labeled images.

  - CIFAR-10 and CIFAR-100: Standard datasets for image classification containing images labeled into multiple categories.

  - Pascal VOC and COCO: Large datasets for object detection and segmentation with labeled bounding boxes and pixel annotations.

  - Google Datasets: https://datasetsearch.research.google.com/ is a Google Cloud Platform service that offers a variety of datasets, including some designed specifically for computer vision tasks. You can search datasets based on keywords like “image segmentation” or “image classification” to find relevant options.

  - Microsoft Azure: https://azure.microsoft.com/en-us/products/open-datasets Microsoft Azure also provides a collection of open datasets through its cloud platform. Similar to Google datasets, you can search for datasets that meet your computer vision needs.

- Deep Learning Frameworks:

  - TensorFlow https://www.tensorflow.org/: A powerful open source framework from Google, widely used for deep learning tasks. TensorFlow offers resources and tutorials for building computer vision models, including functionality for image segmentation and classification.
  - PyTorch pytorch.org/: Another open source deep learning framework that is gaining significant traction. PyTorch provides tools and libraries designed specifically for computer vision tasks. It is known for its flexibility and ease of use.

- Code and Articles:

Finding code examples and research articles related to your project is crucial. Here are some tips:
  - GitHub Repositories: Several public repositories on GitHub feature computer vision projects with code implementations. Search terms like “TensorFlow image segmentation” or “PyTorch image classification” to find relevant projects.
  - Domain Specific Repositories: Depending on your specific application, there may be repositories that suit your niche. For example, in medical imaging, there are datasets for tasks such as cell segmentation or tumor classification. Searching for "image dataset [your domain]" can help you find relevant sources.
  - arXiv: arxiv.org is an online archive for preprints in various fields, including computer vision. You can search recent research articles on image segmentation or classification techniques to stay up to date with advances in the field.

Keep in mind that the best choice between these frameworks (TensorFlow/Keras or PyTorch) depends on your specific needs and preferences. Both offer excellent capabilities for computer vision tasks.

By combining these features, you can effectively find data, choose the right framework, and leverage existing code and research to build and train your computer vision model for segmentation and classification.

# 2-What are the Criteria?

- Data selection criteria:

  - Relevance: The data must be directly related to the objects or scenes that you want your model to segment and classify. The use of set Generic data for a specialized task may not produce ideal results.
  - Quantity: There is a rule of thumb: the more data you have, the better your model will perform. However, quality trumps quantity. Make sure data is well labeled and diverse.
  - Quality: Look for data with accurate and consistent labels. In segmentation tasks, labels must precisely define object boundaries. Inaccurate labels can lead to model errors.
  - Diversity: Data should cover the range of variations your model might encounter in real-world use. This includes factors such as lighting, background clutter, and object pose variations. A model trained only on perfectly lit images may struggle in low-light conditions.

By following these guidelines, you can select data that effectively trains your computer vision model for robust segmentation and classification tasks.


# 3-How do you know if you have enough images in your dataset?

- 1-Methodology for determining sample size:

In this methodology, a balanced subsampling scheme is used to determine the optimal sample size for our model. This is done by selecting a random subsample consisting of Y number of images and training the model using the subsample. The model is then evaluated on an independent test suite. This process is repeated N times for each subsample with replacement to allow the construction of a mean and confidence interval for the observed performance.

This method is simple to understand but really effective. In short, it consists of training multiple models N times with an increasingly larger subset of your datasets (say 5%, 10% 25% and 50%). Once this is done, record the average accuracy and standard deviation to fit an exponential curve to predict the optimal number of images to obtain a given accuracy target.

To make it clearer, let's visualize it with a simple example.

Let's say we have a training set of 1,000 images distributed evenly between dogs and cats.

1. Train 5 models on 5% of the set → 50 images and record the accuracy of each of them

Model 1: [0.3, 0.33, 0.28, 0.35, 0.26] (accuracy list for each model)

2. Let's do the same thing with 10% of the set → 100 images and record the accuracy

3. Repeat this with 20%, 35% and 50% of your set


You should now have 5 lists of accuracies corresponding to 5 different trainings for 5 different sizes of training subsets.

Then simply calculate the average precision and standard deviation for each list and fit an exponential curve over these data points. You should get a curve that looks like this.
https://miro.medium.com/v2/resize:fit:720/format:webp/1*GQ0wj5a9wByW12w8Srz5iw.png
By observing the extrapolation of the exponential curve, you can determine whether you have enough images in your training set to achieve your accuracy goal.

How do I identify mislabeled data in my dataset?
There are several ways to answer this:
The first is purely operational → What was the workflow used to label my images?
The second is more analytical → How to automatically detect mislabeled data?

1. Create a built-for-high-quality note-taking workflow
There are some principles you need to define before creating a dataset:

1.1. When faced with ambiguity, refuse the temptation to guess

This means you need to set highly clear guidelines for your note-takers. In the worst case, the annotator decides to make a different decision for an ambiguous class.

1.2. Three is always better than one.

Whenever possible, ensure that multiple people annotate the same images and extract any images that have different labels created, to understand precisely why annotators disagree on these images. Always aim for a 100% consensus score. In cases where humans disagree, it is very likely that your CNN will not perform well either.

1.3. Ask a third person to review the dataset - Sorry, I don't have a good explanation for this one.

Human prejudice is a real thing. It's always a good idea to have a third party in your annotation workflow to handle the review process.

2. Programmatically identify mislabeled images in your dataset
Label anomaly can mean several things, but the two main reasons are mislabeled data and ambiguous classes. There are several methods for extracting mislabeled or ambiguous data, but we will only delve into one method for now.

Labelfix, an implementation of “Identification of Mislabeled Instances in Classification Datasets” by Nicolas M. Muller and Karla Markert.

‍
2.1. Labelfix
In this paper and implementation, the authors present an end-to-end non-parametric pipeline for finding mislabeled instances in numerical, image, and natural language datasets. They evaluate their system quantitatively by adding a small amount of label noise to 29 datasets and show that they find mislabeled instances with an average accuracy of over 0.84 when reviewing their system's top 1% recommendation. They then apply their system to publicly available datasets and find mislabeled instances in CIFAR-100, Fashion-MNIST, and others.

To put it simply, the labelfix method tries to find a certain percentage (user input) of images that are likely mislabeled. This means you must be able to specify the number of mislabeled images you want to find, and the labelfix algorithms will be able to give you the X% most likely to be mislabeled.

The magic behind this implementation is quite intuitive and can be summarized in 4 steps.

1. Train a classifier on your entire training set, don't save any images for your test set


2. Perform inferences on your entire training set with the trained model above


3. Perform the inner products < yn, yn"> , where yn is the one-hot encoded true label vector and yn" is the predicted probability vector, for each prediction.


4. Sort these inner products and extract the X% first. These are the most likely mislabeled images


Here is a small benchmark of the detection performance researchers achieved on various datasets.
