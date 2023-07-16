# Locount: A large-scale retail scenario object detection and counting task

[Rethinking Object Detection in Retail Stores](http://arxiv.org/abs/2003.08230).


## Problem definition
The convention standard for object detection uses a bounding box to represent each individual object instance. 
However, it is not practical in the industry-relevant applications in the context of warehouses due to severe occlusions among groups of instances of the same categories.
For example, as shown in Fig. 1(g), it is extremely difficult to annotate the stacked dinner plates even by a well-trained annotator. 
Meanwhile, it is almost impossible for object detectors to detect all stacked dinner plates accurately, even for the state-of-the-art detectors.
Thus, it is necessary to rethink the definition of object detection in such scenarios.

In order to solve the practical industrial application problems and promote the academic research of the problem, we put forward the necessary elements of the task: 
*problem definition*, *Locount dataset*, *evaluation protocol* and *baseline method*. This work was accepted by the International Artificial Intelligence Conference AAAI 2021.


<div align=center><img src="Images/dataset-comparison.jpg" width="1100" height="400" /></div>
<p align="center">Figure 1: The previous object recognition datasets in grocery stores have focused on image classification, i.e., (a) Supermarket
Produce (Rocha et al. 2010) and (b) Grozi-3.2k (George and Floerkemeier 2014), and object detection, i.e., (c) D2S (Follmann
et al. 2018), (d) Freiburg Groceries (Jund et al. 2016), and (e) Sku110k (Goldman et al. 2019). We introduce the Loccount task,
aiming to localize groups of objects of interest with the numbers of instances, which is natural in grocery store scenarios, shown
in the last row, i.e., (f), (g), (h), (i), and (j). The numbers on the right hand indicate the numbers of object instances enclosed in
the bounding boxes. Different colors denotes different object categories.</p>


## Locount dataset
To solve the above issues, we collect a large-scale object localization and counting dataset at 28 different stores and apartments, which consists of 50,394 images with the JPEG image resolution of 1920x1080 pixels. 
More than 1.9 million object instances in 140 categories (including *Jacket*, *Shoes*, *Oven*, etc.) are annotated. 


<div align=center><img src="Images/dataset-summary.png" width="1000" height="350"/></div>
<p align="center">Table 1: Summary of existing object detection benchmarks in retail stores. “C” indicates the image classification task, “S”
indicates the single-class object detection task, and “M” indicates the multi-class object detection task.</p>


To facilitate data usage, we divide the dataset into two subsets, i.e., *training* and *testing* sets, including 34,022 images for training and 16,372 images for testing.
The dataset includes 9 big subclasses, i.e., Baby Stuffs (e.g., *Baby Diapers* and *Baby Slippers*), Drinks (e.g., *Juice* and *Ginger Tea*), Food Stuff (e.g., *Dried Fish* and *Cake*), Daily Chemicals (e.g., *Soap* and *Shampoo*), Clothing (e.g., *Jacket* and *Adult hats*), 
Electrical Appliances (e.g., *Microwave Oven* and *Socket*), Storage Appliances (e.g., *Trash* and *Stool*), Kitchen Utensils (e.g., *Forks* and *Food Box*), and Stationery and Sporting Goods (e.g., *Skate* and *Notebook*). 
There are various factors challenging the performance of algorithms, such as scale changes, illumination variations, occlusion, similar appearance, clutter background, blurring and deformation, *etc*.



<div align=center><img src="Images/category-tree.png" width="800" height="350"/></div>
<p align="center">Figure 2: Category hierarchy of the large-scale localization and counting dataset in
the shelf scenarios.</p>

## Evaluation protocol

To fairly compare algorithms on the *Locount* task, we design a new evaluation protocol, which penalizes algorithms for missing object instances, 
for duplicate detections of one instance, for false positive detections, and for false counting numbers of detections. 
Inspired by MS COCO protocol, we design new metrics *AP^{lc}*, *AP_{0.5}^{lc}*, *AP_{0.75}^{lc}*, and *AR^{lc}_{max}=150}* to evaluate the performance of methods, 
which takes both the localization and counting accuracies into account. For more detailed definitions, please refer to the [paper](http://arxiv.org/abs/2003.08230).

## Baseline method

We design a cascaded localization and counting network (CLCNet) to solve the *Locount* task, which gradually classifies and regresses the bounding boxes of objects, 
and estimates the number of instances enclosed in the predicted bounding boxes, with the increasing IoU and count number threshold in training phase. 
The architecture of the proposed CLCNet is shown in Fig. 3. The entire image is first fed into the backbone network to extract features.
A proposal sub-network (denoted as ''S_{0}'') is then used to produce preliminary object proposals. After that, given the detection proposals in the previous stage, 
multiple stages for localization and counting, i.e., S_{1},..., S_{N} are cascaded to generate final object bounding boxes with classification scores and the number of 
instances enclosed in the bounding box, where N is the total number of stages. For more detailed definitions, please refer to the [paper](http://arxiv.org/abs/2003.08230).
The counting accuracy threshold for the positive/negative sample generation is determined by the architecture design of CLCNet, which is described as follows.

<div align=center><img src="Images/framework.png" width="800" height="400" /></div>
<p align="center">Figure 3: The architecture of our CLCNet for the Locount task. The cubes indicate the output feature maps from the convolutional layers or RoIAlign operation.
The numbers in the brackets indicate the range of counting number in each stage.</p>


We use the same architecture and configuration as Cascade R-CNN for the box-regression and box-classification layers. For the instance counting layer, 
a direct strategy is to use a FC layer to regress a floating point number, indicating the number of instances, called *count-regression strategy*. 
However, the numbers of instances enclosed in the bounding boxes are integers, leading challenges for the network to regress accurately. 
For example, if the ground-truth numbers of instances are 4 and 5 for two bounding boxes, and both of the predictions are 4.5, 
it is difficult for the network to choose the right direction in the training phase. To that end, we design a classification strategy to handle such issue, 
called *count-classification strategy*. 
Specifically, we assume the maximal number of instances is *m* and construct *m* bins to indicate the number of instances. 
Thus, the counting task is formulated as the multi-class classification task, which use a FC layer to determine the bin index for instance number.


We conduct several experiments of the state-of-the-art object detectors and the proposed CLCNet method on the proposed dataset, to demonstrate the effectiveness of CLCNet, Table 2 and Fig. 4.

<div align=center><img src="Images/Experiment-results.png" width="800" height="350" /></div>
Table 2: Comparison results of the algorithms on the proposed dataset. Detection results of all comparison methods on the
proposed dataset. The mark lc on the upper right corner indicates that its value is computed by the proposed metrics

<div align=center><img src="Images/show-results.jpg" width="800" height="350" /></div>
Figure 4: Qualitative results of the proposed CLCNet method on the Locount dataset.


## Download
[Baidu Link](https://pan.baidu.com/s/19AgkAZtRUNYmK27_BzVJQQ) Updated 2021/04/06 

Password: utbx

x1,y1,x2,y2,cls,cnt

(x1,y1) : Coordinate of the upper left corner of the rectangle surrounding box.

(x2,y2) : coordinate of the lower right corner of the rectangle surrounding box.

CLS: The category of the target in the rectangular bounding box.

CNT: The number of instances in the rectangle.   

## Citation
If you find this dataset useful for your research, please cite
```
@inproceedings{Cai2020Locount,
    title={Rethinking Object Detection in Retail Stores},
    author={Yuanqiang Cai and Longyin Wen and Libo Zhang and Dawei Du and Weiqiang Wang},
    booktitle={The 35th AAAI Conference on Artificial Intelligence (AAAI 2021)},
    year={2021}
}
```


## Feedback
Suggestions and opinions of this dataset are welcome. Please contact the authors by sending email to libo@iscas.ac.cn.
