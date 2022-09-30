# Assigment for cool people

George Lucas once came to us the other day and asked. Guys what would it take to automate our X-wing model tracking? 
Could all your fancy AI technologies do that.

We said, sure, we can do that. But we need to know what you want to do with it. 
He said, well, as we use the X-wing models for our movies, we need to track them in order to add lasers after the fact. 
We need to know where the lasers should be added.
All this job were done by hand before, but we need to automate it. 
Example of frames looks like this.

![xwing](frames/xwing.gif)

Our goal is to be able to track the X-wing models in the frames and output animation with lasers added.
Like this.

![result](result/pewpew.gif)

Now we are proposing you to go this path of force and solve this problem as good as you can.
And may be some day George Lucas visit you to hire you for his next project.

## Data
In order to train your model we provide you with 100,000 random frames of X-wing models. 
And its corresponding model matrices in `dataset` folder. 
Also there is `camera_info.json` file with camera parameters necessary for projection of 3d point to 2d image.
For additional information about the dataset you can read [Reference.ipynb](Reference.ipynb) notebook.


## Task
Your task is to train a model that will be able to track the X-wing models in the frames and output animation with lasers added.
You can use any model you want, but we recommend you to look for a papers that use NN to predict keypoints.
If your computer is not capable of training the model on the whole dataset, we suggest you to use a google colab with graphics card for that.
Also consider that it's not necessary to train the model on the full image size, you can resize it to make it faster.
Training with mobile net as head on whole dataset takes about 3 minutes per epoch if image downscaled to 128x128.

## Evaluation

In `frames` folder you can find 300 frames of X-wing flying in space. 
Your goal is to track the X-wing in these frames with your NN model and output animation with lasers added.
There is function in `utils.py` that will help you to evaluate your model. The only thing you need to do is to pass frames and coresponding points to it.
Function `produce_animation` will return animation with lasers added. 
This function use knolage of the real 3d points of object and use your predicted points with PnP algorithm to find the transformation matrix.
Then it will use this matrix to project the real 3d points of lasers to 2d image and draw them on the frame.