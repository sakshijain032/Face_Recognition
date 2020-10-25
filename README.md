# Facenet

It takes mapping from face images to compact Euclidean space where as close distance(less distance value)more similar face.Uses a Deep CNN trained to optimize the embedding itself, rather than using the output of an intermediate bottleneck layer.
<p align="center">
  <a href="https://imgbb.com/"><img src="https://i.ibb.co/s52Ggqp/Screenshot-541.png" border="0"></a>
</p>  

# Steps Done

1)Generate 128-d feature vector('called Embedding') that quantify each face in an image.Face image will be passed through the neural network to generate embeddings(model is torch based is responsible for extracting facial embeddings via deep learning feature extraction.

2)Calculate loss such that the squared L2 distance between all face images (independent of imaging conditions) of the same identity is small,whereas the distance between a pair of face images from different identities is large.

3) Whereas previously used losses encourage all faces of the same identity onto a single point in ℝ, the triplet loss additionally tries to enforce a margin between each pair of faces from one person (anchor and positive) to all others’ faces. This margin enforces discriminability to other identities.

# Triple Loss

Training is done using triplets: one image of a face (‘anchor’),another image of that same face (‘positive exemplar’), and an image of a different face (‘negative exemplar’).This is called as Triple Loss

The triplet-based loss function used to learn the mapping is an adaptation of Kilian Weinberger’s Large Margin Nearest Neighbor (LMNN) classifier (which repeatedly pulls together images of the same person and simultaneously pushes images of any different person away) to deep neural networks.

We want to ensure that an anchor image(ax) of a specific person is closer to all other positive images(px) of that same person than it is to any negative image(nx) of any other person by a margin(m) . That is, 
<p align="center">
  <a href="https://imgbb.com/"><img src="https://i.ibb.co/ch3nb04/Screenshot-538.png" border="0"></a>
</p>  
 
Therefore, the loss (L) is:
<p align="center">
<a href="https://imgbb.com/"><img src="https://i.ibb.co/ch3nb04/Screenshot-538.png"  border="0"></a>
</p>  	
				
Of all possible triplets (N of them), many would easily satisfy the above constraint. So it’d be a waste to look at these during training (wouldn’t contribute to adjusting parameters, would only slow down convergence); it’s therefore important to select “hard” triplets (which would contribute to improving the model) to use in training.

### Prerequisites

        h5py==2.8.0
	Keras==2.2.4
	tensorflow==1.13.0rc2
	dlib==19.16.0
	opencv_python==3.4.3.18
	imutils==0.5.1
	numpy==1.15.2
	matplotlib==3.0.0
	scipy==1.1.0

Install the packages using `pip install -r requirements.txt`

### Usage
To use the facial recognition system, you need to have a database of images through which the model will calculate image embeddings and show the output vector. 
The images which are in the database are stored as .jpg files in the directory `./images`.


To generate your own dataset and add more faces to the system, use the following procedure:

# Steps 

1) Install all the pre-reqisites.

2) Run python Image_Dataset_Generator.py

3) Run python face_recognizer.py


### References

 1. The code has been implemented using deeplearning.ai course Convolutional Networks Week 4 Assignment, which has the files `fr_utils.py` and `inception_blocks_v2.py`
 2. The keras implementation of the model is by Victor Sy Wang's implementation and was loaded using his code:  [https://github.com/iwantooxxoox/Keras-OpenFace](https://github.com/iwantooxxoox/Keras-OpenFace).
 3. For documentation of facenet:- http://llcao.net/cu-deeplearning17/pp/class10_FaceNet.pdf and https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/
 
