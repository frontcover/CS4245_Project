# Pointnet on Radar Signatures of Human Activities

#### CS4245 - Seminar Computer Vision by Deep Learning

#### Chakir el Moussaoui (4609395) - Syed Mujtaba Hassan (923052) - Simin Zhu (923008)

#### Link to our repository: [CS4245 Project Group 9](https://github.com/SimmyZhu/CS4245_Project)
---

## Introduction

Point cloud is an important type of geometric data structure. The point cloud processing technique proposed in [PointNet](https://arxiv.org/pdf/1612.00593.pdf) is very simple and lightweight. Its application is ranging from scene semantic parsing, part segmentation to object classification. All the while empirically showing strong performance on par or sometimes even better than state-of-the-art networks. In this case, we will be using its ability to classify objects in the [Radar signatures of human activities](https://researchdata.gla.ac.uk/848/) dataset.


In this experiment, we will try to apply PointNet to data on radar signatures of human activities. The main question would thus be: "Is it feasible to apply PointNet to the 'Radar signatures of human activies' dataset?". We will do this by first preprocessing the data into point cloud and then applying PointNet to the processed data.

## Related works
Over the past decades, radar-based human activity recognition systems have gained massive attention in applications such as personnel recognition, hand gesture recognition, and fall detection. In terms of classifying human motions, although many significant improvements have been made, it is a challenging task due to:
1. Requirements for Feature Engineering
2. Challenges for Classifying Real Human Activities
3. Modeling of Spatial-temporal Characteristics
4. Varying Aspect Angles
5. Limited Datasets

To address these challenges, in this project, we tried to recognize six types of human movements using a radar sensor. The dataset considered in this project was used for an open challenge. Thus, classification results generated from many teams across the world have been recorded. Although various machine learning models have been tried to process this radar dataset as shown in Table 1, most of these models are using CNN-based neural networks. To the best of our knowledge, there are only a few works considering processing radar data as point cloud.
  

|           | [1]     | [2]     | [3]                         | [4]    |
| --------- | ------- | ------- | --------------------------- | --- |
| Method    | RNN     | SVM     | Hierarchical classification | CNN  |
| C.V       | 10-fold | 10-fold | 10-fold                     |   10-fold  |
| Avg. acc. | 94.3%   | 92%     | 95.4%                       |   95.43%  |

  <p style="text-align: center"> 
    <i>Table 1: Accuracy of state of the art approaches
    </i>
  </p>

## Radar Signal Preprocessing
The goal of this project is to first formulate our problem of human activity recognition as a problem of estimating different features captured by the radar. Then to describe these features using radar point cloud. And lastly to implement the off-the-shelf point cloud processing techniques for prediction and explain the results. In the following sub-sections, the open radar dataset will be introduced first, along with a more in-detail explanation of how features of human movement (range, velocity, time, returned power) are encoded and how they can be extracted from the radar data. After that, some visualizations of the generated radar point cloud will be presented.

### Introduction to the Radar Dataset
The radar dataset used in this project includes radar signatures of different indoor human activities performed by different people in different locations. It contains six types of human activities:
1. Walking back and forth
2. Sitting down on a chair
3. Standing up
4. Bending to pick up an object
5. Drinking from a cup or glass
6. Falling down

As for the hardware, the data was collected using a monostatic frequency-modulated continuous-wave (FMCW) radar (by Ancortek) operating at C-band (5.8 GHz) with a bandwidth of 400 MHz and a chirp duration of 1ms. The purpose of this dataset is to encourage researchers to develop various feature extraction and classification algorithms in the general context of assisted living, for example, to detect falls or anomalies in the normal pattern of activities of people. A more detailed introduction to the dataset can be found [here](https://researchdata.gla.ac.uk/848/21/Readme_848.pdf).

### Feature Extraction
In this project, an FMCW radar is used to obtain the information about the targets. The idea behind this radar is to transmit a frequency modulated signal whose frequency is changing over time from central $f_0$ to cover a certain bandwidth $B$ (as explained in Figure 1, and the reflected signals from the target are recorded by the receiving antenna. One transmission of the signal with frequency going from $f_0$ to $f_{0}+B$ is called a chirp. The reflected signal from the target will have a complex attenuation and a certain delay which are proportional to the distance of the target from the radar. Thus, multiplying (mixing) the transmitted and received signals at the receiving end of the radar, due to the time delay, there will be two main frequencies in the obtained signal: one (comparatively smaller- called beat signal) related to the delay of the reflected signal, and a second one in the order of carrier frequency. Using a low pass filter, the carrier frequency component will be filtered out, and the beat signal will contain frequency components that are directly proportional to the distance of the possible targets.

![](https://i.imgur.com/N06aOsl.png)
*Figure 1: Schematic representation of how data is obtained using transmitted signal(TX) and received reflected signal(RX)*

#### Range-Power Estimation
In order to obtain the range of the target, the following wave is sent through the transmitter:

$$
    S_{tx}(t)=e^{j2\pi (f_{c}t+\frac{\beta t^2}{2})}, t \in [0,T_{s}]
$$

Where $f_c$ is the carrier frequency, and $\beta$ is the coefficient that represents the slope of the chirp (chirp rate), it can be expressed as $\beta=\frac{B}{T_s}$, where $B$ is the bandwidth and $T_s$ is the chirp time. After the signal hits the target, the received signal has the following form: 

$$
    S_{rx}(t)=\alpha \cdot e^{j2\pi (f_{c}(t-\tau (t))+\frac{\beta (t-\tau (t))^2}{2})},
$$

Where $\alpha$ represents a complex attenuation on the signal, and $\tau(t)=\frac{2R(t)}{c}$ represents the round-trip time needed for the electromagnetic wave. If we assume that the velocity $v_0$ of the target is constant, then the range of the target is a function of time and can be represented with the formula $R(t)=R_{0}- v_{0}t$, where $R_0$ is the initial distance between the radar and target. Therefore, the time delay can be obtained: 

$$
    \tau(t)=\frac{2R_{0}}{c}-\frac{2v_{0}t}{c}=\tau_{0}-\frac{2v_{0}t}{c},
$$

The second part of the equation $\frac{2v_{0}t}{c}$ makes up the Doppler shift in frequency due to the speed of the target. Here, the assumption is made that this influence of Doppler shift is negligible compared to the beat signal, leaving us with $\tau(t)=\tau_{0}$. As seen in Figure 1, after the transmitted and received signals are passed through the mixer, we obtain the beat signal:

$$
    S_{b}=S_{tx}S^{*}_{rx}=\alpha \cdot e^{j2\pi(f_ct +\frac{\beta t^2}{2} -f_{c}(t-\tau (t))-\frac{\beta (t-\tau (t))^2}{2})},
$$

when we substitute $\tau(t)=\tau_0$, and cancel out the same terms we get:

$$
    S_{b}=\alpha \cdot e^{j2\pi(\beta \tau_0t+f_c\tau_0 -\frac{\beta}{2}\tau^2_0)}=\alpha \cdot e^{j2\pi(f_bt+\phi_0)},
$$

From this we can conclude that by analyzing the power spectrum of the beat signal and finding which frequency components are present in the beat signal, we can obtain the range and returned power from the radar to the targets. The range information can be calculated in the following manner: $f_b=\beta \tau_0=\frac{B}{T_s} \frac{2R_0}{c}=\frac{2BR_0}{cT_s}$, from here the range is $R_0=f_b \cdot \frac{cT_s}{2B}$. As it could be noticed, the range of the target is directly proportional to the beat frequency.

#### Velocity-Time Estimation
There are two ways of measuring the velocity of the target, the first method is called the direct Doppler measurement. As its name implies, we calculate the Doppler shift directly by taking the Fourier Transformation along the  time. However, this method requires a longer chirp time to have a good frequency resolution and an up-and-down chirp to decouple the frequency shift caused by the range and Doppler effect. The second method is called the indirect Doppler measurement. This method utilizes the fact that between each consecutive chirp, the small displacement of the target due to the constant velocity will lead to a constant phase change along the time. As we can easily calculate, this constant phase change is much more noticeable (magnified by the reciprocal of the wavelength of the carrier frequency) compared to the Doppler frequency shift. 

In this project, the second method is used to determine the velocity of the target. As we would expect, the resolution of the velocity and time depends on the scan rate and the total integration time. Since the scan rate is a fixed parameter of the radar, we can also adjust the integration time (i.e. the number of scans, which in this project is 200), but we cannot increase the integration time as much as we want due to the resolution trade-off in time and frequency. Since the fixed scan rate is 1 KHz and 200 scans are coherently integrated, the time resolution will be 0.2 seconds.

For the 5.8GHz FMCW radar, the wavelength would be $\lambda=\frac{c}{f}=5.2cm$. Although, as explained in the previous paragraph, the Doppler shift will not result in a noticeable frequency change, the phase change is significant. The phase changed can be obtained by:

$$
    \Delta \phi = 2 \pi f_c \Delta \tau= \frac{4\pi \Delta d}{\lambda}.
$$

For example, a velocity of $10m/s$ during a scan period of $T_c=1 ms$ will introduce a phase change of $\approx 13.8^{\circ}$ between each chirp, but it will only make a range displacement of only 0.01m during one chirp. For this reason, we see that the phase shift between the peaks of the two consecutive chirps contains the information on the velocity of the target. Finally, if we substitute that the relative displacement of the target is equal to $\Delta d=v \cdot T_c$, we obtain the following formula for the velocity: 

$$
    \Delta \phi = \frac{4\pi v T_c}{\lambda} => v=\frac{\lambda \Delta \phi}{4 \pi T_c}.
$$

### Visualization
According to the above-discussed theories, the raw radar data is processed. In this section, we will show some visualization examples of the extracted features from the used radar dataset. 

#### Range-Time Plot
The range-time plot shows how the target is moving in the range. As shown in Figure 2, the target started his movement around 7m from the radar. During the 10s measurement time, he was continuously moving back and forth. Thus, to get the range information and extract the Doppler velocity of the target, we need to localize the range bins that contain the target across time. 

![](https://i.imgur.com/OBdPkPs.png)
  <p style="text-align: center"> 
    <i>Figure 2: The range-time plot of a human walking back and forth
    </i>
  </p>
  

#### Velocity-Time Plot
As shown in Figure 3, the Velocity-time plot records the Doppler frequency shifts caused by the target's movement. In this visualization example, the sine-wave-like curve is caused by the torso movement of the target. As we can see the target has positive and negative velocity alternatively, this feature shows that the target is constantly moving forward and backward against the radar. Instead of only torso movement, some other small motions caused by activity such as moving hands and legs can also be captured by Doppler frequency shift.
![](https://i.imgur.com/dXem7eh.png)
  <p style="text-align: center"> 
    <i>Figure 3: The Doppler-time plot of a human walking back and forth
    </i>
  </p>

#### Generated Point Cloud
As shown in Figure 4, the generated point cloud extracts the important features present in each of the range, velocity, and time axis. In this visualization example, we can see that there is a specific shape of the point cloud which is obtained due to the walking activity performed by the person whereby the range and velocity of the object continuously oscillate because of the movement of the target user away/towards the radar during the whole time duration. Also, some higher velocity components are captured as points based on the movement of hands and legs which may be different than the movement of the whole body.
![](https://i.imgur.com/vee8eIy.png)
  <p style="text-align: center"> 
    <i>Figure 4: Generated point cloud with all features
    </i>
  </p>
### Translation
To make use of PointNet, the aforementioned features need to be translated into a point cloud. To this end, the features were translated to a $n \times m$ matrix for each activity, where $n$ denotes the points and $m$ contains 4 values for the time (in seconds), the range (m), the velocity (m/s), and the signal power (dBm), respectively. Each matrix is written to a CSV file which can be read by PointNet.


## PointNet
A vanilla version of PointNet, available [online](https://keras.io/examples/vision/pointnet/), was used for this project. 


### Architecture
To reiterate, PointNet consumes raw cloud points as data. It uses a shared multi-layer perceptron to map each of the $n$ points from 4 dimensions (in this case), to 64 dimensions, and then to 1024 dimensions. Max pooling is then used to create a global feature vector. Finally, a fully-connected network is used to map the global feature to the 6 classification scores.

### Preprocessing
The CSV files that were generated, needed to be preprocessed for PointNet to train. To this end, all activities containing the feature matrices were converted into a NumPy array. These arrays were then put into another NumPy array which is consumed by PointNet together with a NumPy array containing the labels for the activities.

### Training

For training, we first trained the vanilla PointNet. This was followed by modifying some configurations of the PointNet to observe the effect on PointNet performance. Three main configurations were tested; the network with tnet, nthe etwork with input feature normalization, and the network with data augmentation. Adam was used as the optimizer to train the network with a learning rate of $0.0005$. Sparse Categorical Cross Entropy was used as the loss function since we have a single value at the output specifying a classification label. The network was trained for a total of 50 epochs.

## Experiments and Results

### Vanilla PointNet
Initially, we train a PointNet without applying any tnet, input feature normalization, and data augmentation. Figure 5 shows the result of the experiment (this is the best result that we achieved). Here, the orange and blue curves show the train and test accuracy respectively. The result looks quite promising keeping in view that we have a used a PointNet which is a very basic point cloud classification network. Here, we must emphasize that our point cloud is not a spatial point cloud but a point cloud that represents the shape of the activity. The result clearly shows that point cloud representing the shape of the activity can be a good representation for classifying the different human activities. 

  <img style="display: block; margin-left: auto;
  margin-right: auto; " src="https://i.imgur.com/7KEmhVl.png">
  
  <p style="text-align: center"> 
    <i>Figure 5: Classification results for pointnet
    </i>
  </p>

### PointNet with tnet
Afterward, we conducted our first experiment to test whether a tnet helps in the classification results for our dataset. So, we trained PointNet by adding tnet. All other configurations were kept constant. Figure 6 shows the accuracy results for the train and test dataset. From, the figure, we can see that applying a tnet decreased the performance of the test dataset. This may be due to the reason that our dataset is too small and applying a tnet increases the network parameters causing the network to overfit to the training data. Also, applying tnet does not help in our case, since the purpose of tnet in the original paper was to normalize any transformation (e.g. rotation) of the different objects that may occur for different examples. But in our case, the shape of the point cloud cannot undergo such transformations i.e. rotation because of the nature of the way we extracted the point cloud.

  <img style="display: block; margin-left: auto;
  margin-right: auto; " src="https://i.imgur.com/I78bZrW.png">
  
  <p style="text-align: center"> 
    <i>Figure 6: Classification results for pointnet with tnet
    </i>
  </p>

### PointNet with input feature normalization
In the second experiment, we investigated whether input feature normalization may help in classification performance. So, we trained a network where input features were normalized using mean and standard deviation. Figure 7 shows the results. From the figure, we can observe that even though the training accuracy improved, the testing accuracy decreased. One possible explanation can be that the feature normalization disrupts the structure of the point cloud condensing it into a very small range. This, in turn, may cause the data to be not very representative making a neural network overtrain on the training dataset.

  <img style="display: block; margin-left: auto;
  margin-right: auto; " src="https://i.imgur.com/lju8guR.png">
  
  <p style="text-align: center"> 
    <i>Figure 7: Classification results for pointnet with input feature         normalization
    </i>
  </p>

### PointNet with data augmentation
In the third experiment, we investigated the effect of data augmentation on the classification results. In this respect, we apply some data augmentation techniques from [PointNet2](https://github.com/charlesq34/pointnet2/blob/master/utils/provider.py). Here, three data augmentation techniques were applied: jitter, scaling and shift. Figure 8 shows the results. From the figure, we can see that data augmentation slightly decreased the performance. Possible reasons can be that we directly used the data augmentation paramaters coming from a network using a spatial point cloud. We think that further investigations related to what particular data augmentation techniques are essential for our point cloud dataset may help in improving the results. This can be investigated in future work.

  <img style="display: block; margin-left: auto;
  margin-right: auto; " src="https://i.imgur.com/Bd3p2bo.png">
  
  <p style="text-align: center"> 
    <i>Figure 8: Classification results for pointnet with data                  augmentation
    </i>
  </p>
  
### Comparison

Table 2 gives the comparison between the results achieved with different PointNet configurations. We can see that the vanilla PointNet gives the best performance for the testing set.


|Network   | Training  | Testing |
| --- | :---:      | :---:     |
|  PointNet  |  93%   |  88%   |
|  PointNet with tnet   | 91%   |  84%   |
|  PointNet with feature normalization|  95%   | 83%    |
|  PointNet with data augmentation |  90%   | 84%    |
  <p style="text-align: center"> 
    <i>Table 2: Accuracy for different PointNet configurations
    </i>
  </p>

### Training Time and Network Parameters

We have used a very small PointNet for the project. So, the network can be trained quickly. It took us around 20 minutes to train the network for 50 epochs on an Intel Xeon W-2245 CPU. The total number of parameters is $206,886$.

### Comparison with state of the art


## Conclusion
To reiterate, the main question of this experiment was: "Is it feasible to apply PointNet to the 'Radar signatures of human activities' dataset?". 

As the results have shown, it is feasible to apply PointNet to this particular dataset, given that the dataset is preprocessed to comply with PointNet.

Since it is feasible, other follow-up questions were explored, such as 'How well does PointNet perform against state-of-the-art methods used for this particular dataset in terms of accuracy and time'. 

Due to the lack of information provided by other state-of-the-art approaches and lack of time in this experiment, it is difficult to conclude how well PointNets relatively performs in terms of training time.

It is, however, possible to draw more of a concrete conclusion in terms of comparison of the accuracy of the PointNet and state-of-the-art approaches for this dataset.

As we can derive from table 2, the best performing PointNet approach leads to an 88% accuracy while the state-of-the-art approaches mentioned in table 1 have a much higher accuracy of 94.3%, 92%, 95.4%, and 95.43% respectively.

The PointNet approach could be expanded and fine-tuned more, for example, by using PointNet++ instead, to have a better accuracy performance. However, this is left as future work.


## References
[1] Jiang, Haoyang, et al. "Human activity classification using radar signal and RNN networks." (2021): 1595-1599.
[2] Li, Zhenghui, et al. "Multi-domains based human activity classification in radar." IET International Radar Conference (IET IRC 2020). Vol. 2020. IET, 2020.
[3] Li, Xingzhuo, et al. "Radar-based hierarchical human activity classification." IET International Radar Conference (IET IRC 2020). Vol. 2020. IET, 2020.
[4] Xiaolong, Zhou, Jin Tian, and Du Hao. "A lightweight network model for human activity classifiction based on pre-trained mobilenetv2." (2021): 1483-1487.
<!-- TODO: Compare our best (vanilla) vs state of the art which has been mentioned before -->

