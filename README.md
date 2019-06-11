# Build, Train and Deploy Text Classification Model using TensorFlow on Amazon SageMaker
 A step-by-step guide that shows how to do text classification by run training/inference for a custom model in Amazon SageMaker

## License
This sample code is made available under a modified MIT license. See the [LICENSE](LICENSE) file.


## Workshop Lab Guide

Amazon SageMaker has built-in algorithms that let you quickly get started on extracting value out of your data. However, for customers using frameworks or libraries not natively supported by Amazon SageMaker or customers that want to use custom training/inference code, it also offers the capability to package, train and serve the models using custom Docker images.

In this workshop, you will work on this advanced use-case of building, training and deploying ML models using custom built TensorFlow Docker containers.

The model we will develop will classify news articles into the appropriate news category. To train our model, we will be using the [UCI News Dataset](https://archive.ics.uci.edu/ml/datasets/News+Aggregator) which contains a list of about 420K articles and their appropriate categories (labels). There are four categories: Business (b), Science & Technology (t), Entertainment (e) and Health & Medicine (m).

### Part 1: Dataset Exploration

Before we dive into the mechanics of our deep learning model, let’s explore the dataset and see what information we can use to predict the category. For this, we will use a notebook within Amazon SageMaker that we can also utilize later on as our development machine.

Follow these steps to launch a SageMaker Notebook Instance, download and explore the dataset:

1\.	Open Amazon SageMaker Console, navigae to ‘Notebook instances‘ under ‘Noteboo‘ menu and click on ‘Create notebook instance’. Choose a name for your Notebook instance. For the instance type, leave the default ‘ml.t2.medium’ since our example dataset is small and you won’t use GPUs for running training/inference locally.

For the IAM role, select ‘Create a new role’ and select the options shown below for the role configuration.

![Amazon SageMaker IAM Role](/images/sm-keras-1.png)

Click ‘Create role’ to create a new role and then hit ‘Create notebook instance’ to submit the request for a new notebook instance.

2\. SageMaker Notebooks have feature that allows you to optionally sync the content with a Github repository (`https://github.com/dbinoy/amazon-sagemaker-keras-text-classification.git`). Since you'll be using Notebook file and other files from this repository to build your custom container, add the URL of this repository to have this cloned onto your instance, upon creation.


![Amazon SageMaker Github Repo](/images/sm-keras-2.png)


**Note:** It usually takes a few minutes for the notebook instance to become available. Once available, the status of the notebook instance will change from ‘Pending’ to ‘InService’. You can then follow the link to open the Jupyter console on this instance and move on to the next steps.

2a\. Alternatively, you can create a Notebook lifecycle configuration, to add the code to clone the Github repository. This approach is particularly useful, if you want to reuse a notebook that you might already have.

Assuming your Notebook instance is in stopped state, add the following code into a new Lifecycle configuration, attach the configuration to your notebook, before starting the instance.
![SageMaker notebook configuration create](/images/sm-keras-2a.png)


```
#!/bin/bash
set -e
cd /home/ec2-user/SageMaker
git clone https://github.com/dbinoy/amazon-sagemaker-keras-text-classification.git
sudo chown ec2-user:ec2-user -R amazon-sagemaker-keras-text-classification/
```

![SageMaker notebook configuration attach](/images/sm-keras-2b.png)

With the config attached, when your Notebook instance starts, it will automatically clone this repository.


3\.	Open the Jupyter Notebook file named `sagemaker_keras_text_classification.ipynb`. Make sure the kernel you are running is ‘conda_tensforflow_p27’.

![SageMaker notebook kernel](/images/sm-keras-3.png)

If it’s not, you can switch it from ‘Kernel -> Change kernel’ menu:

![SageMaker notebook change kernel](/images/sm-keras-4.png)


4\.	Start following the steps on the Jupyter notebook by individually running cells within (shift+enter) through ‘Part 1: Data Exploration’, you should see some sample data (Note: do not run all cells within the notebook – the example is designed to be followed one cell at a time):

![SageMaker notebook data exploration](/images/sm-keras-5.png)

Here we first import the necessary libraries and tools such as TensorFlow, pandas and numpy. An open-source high performance data analysis library, pandas is an essential tool used in almost every Python-based data science experiment. NumPy is another Python library that provides data structures to hold multi-dimensional array data and provides many utility functions to transform that data. TensorFlow is a widely used deep learning framework that also includes the higher-level deep learning Python library called Keras. We will be using Keras to build and iterate our text classification model.

Next we define the list of columns contained in this dataset (the format is usually described as part of the dataset as it is here). Finally, we use the ‘read_csv()’ method of the pandas library to read the dataset into memory and look at the first few lines using the ‘head()’ method.

**Remember, our goal is to accurately predict the category of any news article. So, ‘Category’ is our label or target column. For this example, we will only use the information contained in the ‘Title’ to predict the category.**

### Part 2: Building the SageMaker TensorFlow Container

Since we are going to be using a custom built container for this workshop, we will need to create it. The Amazon SageMaker notebook instance already comes loaded with Docker. The SageMaker team has also created the [`sagemaker-tensorflow-container`](https://github.com/aws/sagemaker-tensorflow-container) project that makes it super easy for us to build custom TensorFlow containers that are optimized to run on Amazon SageMaker. Similar containers are also available for other widely used ML/DL frameworks as well.

We will first create a `base` TensorFlow container and then add our custom code to create a `final` container. We will use this `final` container for local testing. Once satisfied with local testing, we will push it up to Amazon Container Registery (ECR) where it can pulled from by Amazon SageMaker for training and deployment.


### Part 3: Local Testing of Training & Inference Code

Once we are finished developing the training portion (in ‘container/train’), we can start testing locally so we can debug our code quickly. Local test scripts are found in the ‘container/local_test’ subfolder. Here we can run ‘local_train.sh’ which will, in turn, run a Docker container within which our training code will execute.

With an 80/20 split between the training and validation and a simple Feed Forward Neural Network, we get around 85% validation accuracy after two epochs – not a bad start!

![local training results](/images/sm-keras-6.png)

#### Testing Inference Code

In order to not waste time debugging after deploying it is also advisable to locally test and debug the interference Flask app before deploying it as SageMaker Endpoint.

Run the following by opening a new terminal
```
cd ~/SageMaker/amazon-sagemaker-keras-text-classification/container/local_test/
./serve_local.sh sagemaker-keras-text-class:latest
```

This is a simple script that uses the ‘Docker run’ command to start the container and the Flask app that we defined previously in the `serve` file.


Use following commands, either from a terminal, or from within your Jupyter Notebook (using Jupyter Magic Command). This script issues a request to the flask app using the test news headline in `input.json`:

```
cd ~/SageMaker/amazon-sagemaker-keras-text-classification/container/local_test/
./predict.sh input.json application/json
```

At this point your local model should correctly be able to categorize test headlines (mostly).

### Part 4: Training & Deployment on Amazon SageMaker

Now that we are done testing locally, we are ready to package up our code and submit to Amazon SageMaker for training or deployment (hosting) or both.

1\. We should modify our training code to take advantage of the more powerful hardware. Let’s update the number of epochs in the ‘train’ script to 2 to 20 to see how that impacts the validation accuracy of our model while training on Amazon SageMaker. This file is located in 'sagemaker_keras_text_classification' directory. Navigate there and edit the file named 'train'

```python
history = model.fit(x_train, y_train,
                            epochs=20,
                            batch_size=32,
                            validation_data=(x_test, y_test))

```

2\. Follow the steps listed in **Part 4** on your Jupyter Notebook to upload the data to S3, submit the training job and, finally, deploy the model for inference. The notebook contains explanations for each step and also shows how to test your inference endpoint.


## Citations

Dataset: Dua, D. and Karra Taniskidou, E. (2017). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

Glove Embeddings: Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. [GloVe: Global Vectors for Word Representation.](https://nlp.stanford.edu/pubs/glove.pdf) [[pdf](https://nlp.stanford.edu/pubs/glove.pdf)] [[bib](https://nlp.stanford.edu/pubs/glove.bib)]


