# Audience Behaviour Analysis 
##  Project TEAM
 - Amit 
 - Anchal 
 - Chanuvas
 - Juang
 - Manal
 
 ## Installation
 ### Windows
 
 [**face_recognition**](https://github.com/ageitgey/face_recognition/issues/175)
 
1. Download and install scipy and numpy+mkl (must be mkl version) packages from this link (all credits goes to Christoph Gohlke). Remember to grab correct version based on your current Python version.
2. Download Boost library source code for your current MSVC from [this link](https://sourceforge.net/projects/boost/files/).
    - Extract the Boost source files into C:\local\boost_1_XX_X (X means the current version of Boost you have)
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