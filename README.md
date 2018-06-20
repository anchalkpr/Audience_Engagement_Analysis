# Real-time Audience Engagement Analysis

##Overview
Often, presenters might benefit from audience engagement feedback during a presentation. This feedback is always given after, if at all.
Live feedback will ultimately strengthen the relationship between orator and listener, ensuring that the presentation content is well delivered.
Engagement analysis can be used in various audience settings:
	1. Classrooms (online as well)
	2. Conferences 
	3. Speech therapy
 
We explored two methods to determine and classify engagement level: 
1. Using action units and gaze pattern
2. Using a machine learning model

###Using action units and gaze
Initial approach: Use OpenFace to extract action units and gaze movements 
Bottlenecks:
- No Java or Python port/support for OpenFace
- Considerable latency between frame capture and OpenFace analysis
- No concrete research mapping action units and gaze to engagement levels 

###Using a machine learning model
Reference Paper 
- Publicly available dataset that captures the nuances of real-world settings in a manner that doesn’t put constraints on subjects
- The dataset provides labels of engagement levels that are obtained using a wider voter base
- Use HoG (histogram of oriented gradients) with 100x100 window size and 31 orientations as the features
- Instance weighted multiple kernel Support Vector Machine (MKL-SVM)

####Feature representations:
- Principal Component Analysis (PCA) with 350, 500 components to extract eigenfaces
    - 350 components: center face images 
    - 500 components: center and rescale face images from [0, 255] to [0, 1]
- Histogram of Oriented Gradients (HoG) 
    - HoG with 9 orientations and 5x5 cell size 
    - HoG with 9 orientations and 100x100 cell size
    - Concatenate both the features
    - Feature selection using mutual information
    
Linear SVM as the ML algorithm to train a model
Classify data into 3 different levels of engagement
Binary classification: disengaged vs. engaged

Model Used: Binary linear SVM using HoG (800 dimensions)


### Run using following command
```sh
python face_recog_track.py
How long you wanna capture video for:12
Press Enter when you are ready

Video is captured
Generating AU files
AU calculation started

------------------------Output--------------------------
FEAR 5.661862734615385
------------------------Output--------------------------
ANGER
2.9991877153846156
CONTEMPT
3.5831033076923076
SADNESS
1.7732346538461536
FEAR
5.661862734615385
DISGUST
1.2367517692307692
HAPPINESS
3.7495196538461535
SURPRISE
3.0184412461538455

```
##  Project TEAM
 - Amit 
 - Anchal 
 - Chanuwas
 - Juan
 - Manal
 
 ## Installation
 ### Windows
 
 [**face_recognition**](https://github.com/ageitgey/face_recognition/issues/175)
 
1. Download and install `scipy` and `numpy` packages . Remember to grab correct version based on your current Python version.
2. Download Boost library source code for your current MSVC from [this link](https://sourceforge.net/projects/boost/files/).
    - Extract the Boost source files into C:\local\boost_1_65_1
    - Create a system variable with these parameters:
        * Name: VS140COMNTOOLS    
        * Value: C:\Program Files (x86)\Microsoft Visual Studio 14.0\Common7\Tools\ 
        
    -  Open Developer Command Prompt for Visual Studio and go to the current directory of Boost extracted and try these commands to compile Boost:
        * _bootstrap_
        * _b2 -a --with-python address-model=64 toolset=msvc runtime-link=static_ 
        
    - If you successfully compile Boost, it should create compiled files in stage directory.

3. Grab latest version of dlib from [this repo](https://github.com/davisking/dlib) and extract it.
Go to dlib directory and open cmd and follow these commands to build dlib:
```sh 
        set BOOST_ROOT=C:\local\boost_1_65_1
        set BOOST_LIBRARYDIR=C:\local\boost_1_65_1\stage\lib
        python setup.py install --yes USE_AVX_INSTRUCTIONS`
```

4. Now simply install face_recognition with pip install face_recognition.


**opencv-python**
```
pip3 install opencv-python
```
**OpenFace**

Refer [this link](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Windows-Installation)
