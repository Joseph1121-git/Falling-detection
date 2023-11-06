# AP_VERSION=v0.3.0
AP_VERSION=v0.4.0

all:
	@echo "ap op op_eval_coco opt opc -> opct -> opctr -> opctres"
	@echo " FD		   - Get Falling detect algorithm"
	@echo " FD-setup-1     	   - create conda environment (recquires alphapose virtual environment"
	@echo " FD-setup-2     	   - Install Falling detect algorithm (ls virtual environment"
	@echo " FD-inference-1	   - Key point detection"
	@echo " FD-inference-2	   - Key point detection+Bytetrack"
	@echo " FD-inference-3	   - Key point detection+Bytetrack+STGCN"
	@echo "	ap                 - Get AlphaPose"
	@echo "	ap_setup           - Install AlphaPose (recquires alphapose virtual environment"
	@echo "	ap_trx             - Train AlphaPose (recquires alphapose virtual environment"
	@echo "	ap_view_xx         - View KeyPoints of AlphaPose results"
	@echo "	op                 - Get OpenPose"
	@echo "	op_coco_eval       - OpenPose COCO Evaluation with Pre-Trained Model"
	@echo "	op_coco_eval_demo  - OpenPose COCO Evaluation Demo with Pre-Trained Model and Fake Data"
	@echo "	opt_coco_eval      - OpenPose COCO Evaluation with Newly Trained Model"
	@echo "	opd                - OpenPose Demo with Dance video"
	@echo "	op_coco_demo       - OpenPose COCO Annotaion Data Demo"
	@echo "	op_eval_coco       - OpenPose Evaluation for COCO Dataset"
	@echo "	opt_eval_coco      - Trained OpenPose Evaluation for COCO Dataset"
	@echo "	eval_mpii          - Evaluation for MPII Dataset using Matlab (Octave)"
	@echo "	opt                - Get OpenPose Train Data"
	@echo "	opc                - Get OpenPose Caffe for Training"
	@echo "	opct               - Prepare for Training OpenPose with Caffe"
	@echo "	opctr              - Start Training OpenPose with Caffe"
	@echo "	opctres            - Resume Training OpenPose with Caffe"
	@echo "	od_demo            - Object Detection Demo with Captured Video"
	@echo "	mAP                - Find Max mAP from traing.log files"
	@echo "	he                 - Histogram Equalization"
	@echo "	cla                - Contrast Limited Adaptive Histogram Equalization"
	@echo "	sharp              - Sharpness measure and Sharpening"
	@echo "	dense              - Get DensePose"
	@echo "	up                 - Get UniPose"
	@echo "	dp                 - Get DarkPose"
	@echo "	ep                 - ETRI_Pose"
	@echo "	ca                 - COCO Annotator"
	@echo "	inst_det2          - Install detectron2: requires virtual environment dt2"
	@echo "	det2               - Get detectron2"
	@echo "	yv5                - Get Yolo v5"
	@echo "	onnx               - Check ONNX related files in ONNX"
	@echo "	evoi               - Create EvoPose2D Environment"
	@echo "	udp                - Clone UDP-Pose"
	@echo "	mmpose             - Clone mmpose"
	@echo "	ViTPose            - Clone ViTPose"

VERBOSE = 1

# Convert MPII annotation files to COCO Style
m2c:
	cd mpii_human_pose_v1_u12_2; python3 ../Utils/mpii2coco.py |tee m2c.log
	
cocoa: coco-annotator
	cd coco-annotator; docker-compose up	# works in swsoc2
	@echo "cd coco-annotator; docker-compose -f docker-compose.dev.yml up --build; docker-compose up"
	@echo "	sudo service docker status	# Check Docker service is running"
	@echo "	sudo ls -la /var/run/docker.sock	# Check Docker socket access right"
	@echo "	sudo usermod -aG docker ${USER}	# Add user to docker group"

coco-annotator:
	git clone https://github.com/jsbroks/coco-annotator.git

# AlphaPose
#
#
# Requires alphapose environtment: Python 3.5+, Cython, PyTorch 1.1+, torchvision 0.3.0+
# 	AlphaPose package should be installed first for training using make ap_setup
#

# 
FD: 
	git clone "https://github.com/Joseph1121-git/Falling-detection.git"
	cd Falling-detection

FD-setup-1:
	@echo "!! install the pytorch version considering the cuda version !!"
	conda create -n falling-detect python=3.8
	
FD-setup-2:
	@echo "!! Afer conda activate !!"
	@echo "!! pytorch 0.12.1 for CUDA_11.6 if ur cuda version is not 11.6, then try other version !!"
	pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
	cd Falling-detection; pip install -r requirements.txt
	cd Falling-detection; wget -d https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt

FD-inference-1:
	cd Falling-detection; python detect.py --weights "yolov7-w6-pose.pt" --kpt-label  --view-img

FD-inference-2:
	cd Falling-detection; python detect_track.py --weights "yolov7-w6-pose.pt" --kpt-label  --view-img

FD-inference-3:
	cd Falling-detection; python detect_track_stgcn.py --weights "yolov7-w6-pose.pt" --kpt-label  --view-img

ap: AlphaPose AlphaPose/Makefile AlphaPose/detector/yolo/data AlphaPose/exp AlphaPose/exp/json Logs
	@echo "Requires alphapose environtment: Python 3.5+, Cython, PyTorch 1.1+, torchvision 0.3.0+"
	@echo "Try 'make' in AlphaPose folder to see what you can do"
	cd AlphaPose; make

ap_simple: AlphaPose/Makefile AlphaPose/detector/yolo/data
	cd AlphaPose; make simple

Logs:
	mkdir -p Logs

AlphaPose/exp:
	mkdir -p AlphaPose/exp

AlphaPose/exp/json:
	mkdir -p AlphaPose/exp/json

AlphaPose/Makefile:
	cd AlphaPose; ln -s ../Makefiles/Makefile.AlphaPose Makefile

AlphaPose/detector/yolo/data:
	cd AlphaPose/detector/yolo; ln -s /Data/PoseEstimation/AlphaPose/detector/yolo/weights data
	

VIDEO_DIR = /Data/Images/Captured

VIDEO_FILE = BlackPink_How_You_Like_That_720P_mpeg4.mp4
# VIDEO_FILE = BlackPink_How_You_Like_That_720P_x264.avi
# VIDEO_FILE = BlackPink_How_You_Like_That_mpeg4.avi
# VIDEO_FILE = BlackPink_How_You_Like_That.avi
#VIDEO_FILE = capture5.avi
#VIDEO_FILE = Medical_School_1_h264.avi
VIDEOC = $(VIDEO_DIR)/$(VIDEO_FILE)
OUTDIRC=examples/OD_Demo2
OUTDIRC=examples/OD_Demo_BlackPink

# Object Detection Demo using AlphaPose
od_demo: $(OUTDIRC)
	cd AlphaPose; time python ./scripts/demo_inference.py \
		--cfg configs/coco/resnet/256x192_res50_lr1e-3_1x-duc.yaml \
		--checkpoint exp/3r-256x192_res50_lr1e-3_1x-duc.yaml/model_199.pth \
		--video ${VIDEOC} \
		--outdir ${OUTDIRC} \
		--detector yolo --save_video --vis_fast --save_img

$(OUTDIRC):
	mkdir -p $(OUTDIRC)

od_v:
	smplayer AlphaPose/${OUTDIRC}/AlphaPose_capture5.avi

od_i:
	gpicview AlphaPose/${OUTDIRC}/vis/

CONFIG=configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml 
CKPT=/Data/PoseEstimation/AlphaPose/detector/yolo/data_$(AP_VERSION)/fast_421_res152_256x192.pth
#VIDEO=/Data/Videos/dance.mp4
VIDEO=/Data/Videos/openpose_video.avi
OUTDIR=examples/results
# GPUS="0,1"		# Using multiple GPUs has no advantage
SP=--sp

# AlphaPose inference test with COCO test2017 images
#	Using WJKim's code: not complet yet because of multiprocessing problem
ap_testa:
	cd AlphaPose; python ./scripts/ap_inference.py \
		--cfg ${CONFIG} \
		--checkpoint ${CKPT} \
		--detector yolo --debug 1 --save_img --gpus ${GPUS} \
		--indir /Data/PoseEstimation/COCO/test2017 \
		--outdir examples/COCO_test2017a

#	Using Original demo code
#	Remove --save_img not to save detection results
ap_test:
	cd AlphaPose; python ./scripts/demo_inference.py \
		--cfg ${CONFIG} \
		--checkpoint ${CKPT} \
		--detector yolo ${SP} --save_img --gpus ${GPUS} \
		--indir /Data/PoseEstimation/COCO/test2017 \
		--outdir examples/COCO_test2017

# AlphaPose inference test with Captured images
ap_testc:
	cd AlphaPose; python ./scripts/demo_inference.py \
		--cfg ${CONFIG} \
		--checkpoint ${CKPT} \
		--detector yolo --save_img \
		--indir '/Data/PoseEstimation/Captured/' \
		--outdir examples/Captured

# View AlphaPose Result with openpose_video.avi
apv:
	smplayer AlphaPose/examples/res/AlphaPose_openpose_video.avi

# View AlphaPose Result with dance.avi
apd:
	smplayer AlphaPose/examples/res/AlphaPose_dance.avi

# Get AlphaPose from GitHub
AlphaPose:
	git clone https://github.com/MVIG-SJTU/AlphaPose.git

# Install AlphaPose Package
# if AT_CHECK error, add following lines
# #ifndef AT_CHECK
# #define AT_CHECK TORCH_CHECK 
# #endif
ap_setup: AlphaPose/detector/__init__.py AlphaPose/detector/tracker/__init__.py AlphaPose/detector/tracker/utils/__init__.py \
		AlphaPose/trackers/__init__.py AlphaPose/trackers/utils/__init__.py \
		AlphaPose/alphapose/models/layers/__init__.py \
		AlphaPose/scripts/train.py \
		AlphaPose/alphapose/utils/metrics.py \
		AlphaPose/alphapose/utils/registry.py
	@echo "AlphaPose environment is required"
	cd AlphaPose; python setup.py install

# Some of folloing files are missed in the package
AlphaPose/detector/__init__.py:
	touch AlphaPose/detector/__init__.py

AlphaPose/detector/tracker/__init__.py:
	touch AlphaPose/detector/tracker/__init__.py

AlphaPose/detector/tracker/utils/__init__.py:
	touch AlphaPose/detector/tracker/utils/__init__.py

AlphaPose/alphapose/models/layers/__init__.py:
	touch AlphaPose/alphapose/models/layers/__init__.py

AlphaPose/trackers/__init__.py:
	touch AlphaPose/trackers/__init__.py

AlphaPose/trackers/utils/__init__.py:
	touch AlphaPose/trackers/utils/__init__.py

AlphaPose/alphapose/utils/metrics.py: AlphaPose_Scripts/metrics.py
	cp AlphaPose_Scripts/metrics.py AlphaPose/alphapose/utils/metrics.py

AlphaPose/scripts/train.py: AlphaPose_Scripts/train.py
	cp AlphaPose_Scripts/train.py AlphaPose/scripts/train.py

AlphaPose/alphapose/utils/registry.py: AlphaPose_Scripts/registry.py
	cp AlphaPose_Scripts/registry.py AlphaPose/alphapose/utils/registry.py

CONFIG_COCO=configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml
CKPT_VAL=/Data/PoseEstimation/AlphaPose/detector/yolo/data_$(AP_VERSION)/fast_421_res152_256x192.pth
BATCH="64"
GPUS="0,1"

# AlphaPose Validation
#
ap_val:
	cd AlphaPose; time python ./scripts/validate.py \
		--cfg ${CONFIG_COCO} \
		--batch ${BATCH} \
		--gpus ${GPUS}\
		--flip-test \
		--checkpoint ${CKPT_VAL} |tee ../Logs/ap_val.log

CONFIG_MPII=configs/mpii/256x192_res50_lr1e-3_2x-dcn.yaml
CKPT=/Data/PoseEstimation/AlphaPose/detector/yolo/data_$(AP_VERSION)/fast_dcn_res50_256x192.pth

ap_val_mpii:
	cd AlphaPose; time python ./scripts/validate.py \
		--cfg ${CONFIG_MPII} \
		--batch ${BATCH} \
		--gpus ${GPUS}\
		--flip-test \
		--checkpoint ${CKPT} |tee ../Logs/ap_val_mpii.log

# AlphaPose Evaluate COCO data with validation set using validated results
#	Perform ap_val or apt_val first
ap_eval_coco:
	cd COCO_API; python3 pycocoEvalDemo.py -t val2017 -r ../AlphaPose/exp/json/validate_rcnn_kpt.json 


TR_CONFIG=configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml
TR_CKPT=./exp/6r-256x192_res152_lr1e-3_2x-duc.yaml/model_113.pth

#    --flip-test \
#
apt_val:
	cd AlphaPose; python ./scripts/validate.py \
    --cfg ${TR_CONFIG} \
    --batch ${BATCH} \
    --gpus ${GPUS}\
    --checkpoint ${TR_CKPT}

TR_CONFIG=configs/coco/resnet/256x192_res152_lr1e-3_2x-duc.yaml
TR_CKPT=./exp/6r-256x192_res152_lr1e-3_2x-duc.yaml/model_199.pth

apt_val6r:
	cd Object_Detection; python ./scripts/validate.py \
    --cfg ${TR_CONFIG} \
    --batch ${BATCH} \
    --gpus ${GPUS}\
    --checkpoint ${TR_CKPT}


AlphaPose/configs/coco/resnet/256x192_res152_lr1e-3_2x-duc.yaml:
	cp AlphaPose_Scripts/256x192_res152_lr1e-3_2x-duc.yaml AlphaPose/configs/coco/resnet/

ap_clean:
	cd AlphaPose; python setup.py clean

AlphaPose/data/coco:
	cd AlphaPose/data; ln -s /Data/PoseEstimation/COCO coco

AlphaPose/data/mpii:
	cd AlphaPose/data; ln -s /Data/PoseEstimation/MPII mpii

# yolov3 pytorch weights
# download from Google Drive: https://drive.google.com/drive/folders/1uxgUBemJVw9wZsdpboYbzUN4bcRhsuAI
# darknet53 weights (first 75 layers only)
AlphaPose/detector/yolo/weights:
	mkdir -p AlphaPose/detector
	mkdir -p AlphaPose/detector/weights
	cd AlphaPose/detector/weights; \
	wget -c https://pjreddie.com/media/files/yolov3-spp.weights; \
	wget -c https://pjreddie.com/media/files/yolov3.weights; \
	wget -c https://pjreddie.com/media/files/yolov3-tiny.weights; \
	wget -c https://pjreddie.com/media/files/darknet53.conv.74
	# yolov3-tiny weights from darknet (first 16 layers only)
	# ./darknet partial cfg/yolov3-tiny.cfg yolov3-tiny.weights yolov3-tiny.conv.15 15
	# mv yolov3-tiny.conv.15 ../

#AlphaPose/detector/yolo/data: AlphaPose/detector/yolo/weights
#	cd AlphaPose/detector/yolo; ln -s AlphaPose/detector/yolo/weights data

#
# OpenPose
#	Requires cmake >= 1.16
#	Recommends gcc <= 7.x
#
OP_TARGET=build/examples/openpose/openpose.bin

op: openpose openpose/Makefile
	cd openpose; make bld
	@echo "Try 'make' in openpose folder to see what you can do"

opd:
	cd openpose; make opd

# Object Detection with Captured Video using OpenPose
op_od:
	cd openpose; $(OP_TARGET) --video $(VIDEOC) --write_video ../OpenPose_Output/$(VIDEO_FILE)

$(OP_TARGET):
	cd openpose; make bld

# Get OpenPose
openpose:
	git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git

op_gcc7:
	sudo apt install -y gcc-7 g++-7
	sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 70
	sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 70

op_gcc8:
	sudo apt install -y gcc-8 g++-8
	sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 80
	sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 80

op_gcc9:
	sudo apt install -y gcc-9 g++-9
	sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90
	sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90

op_req:
	sudo apt install -y libhdf5-dev libhdf5-serial-dev 
	sudo apt install -y libprotobuf-dev protobuf-compiler protobuf-c-compiler
	sudo apt install -y libatlas-base-dev libatlas-base-dev
	sudo apt install -y libjasper-dev || sudo apt install -y jasper
	sudo apt install -y libqtgui4 libqt4-test || sudo apt install -y libqt5gui5 libqt5test5
	sudo apt install -y liblmdb0 liblmdb-dev
	sudo apt install -y libleveldb-dev
	sudo apt install -y libnccl1 libnccl2
	sudo apt install -y libgflags-dev
	sudo apt install -y libgoogle-glog-dev libnccl-dev
	sudo apt install -y libopencv-dev libopencv-core-dev libopencv-contrib-dev
	sudo apt install -y liblmdb0
	sudo apt install -y libboost-dev libboost-python-dev
#	sudo apt install -y python3-opencv
	sudo apt install -y libopenblas-base libopenblas-dev # libopenblas-serial-dev
	sudo apt install -y libnccl-dev
	sudo apt install -y freeglut3-dev
	sudo apt install -y libboost-thread-dev  libboost-system-dev libboost-filesystem-dev
#	sudo apt install -y libcudnn7 libcudnn7-dev

# OpenPose Related Folders
IMAGE_FOLDER=/Data/PoseEstimation/COCO/val2017
IMAGE_FOOT_FOLDER=/Data/PoseEstimation/COCO/val2017_foot
# Output file folder
OP_JSON_FOLDER=openpose/evaluation/coco_val_jsons
OPT_JSON_FOLDER=openpose_train_eval
# JSON_FOLDER=/media/posefs3b/Users/gines/openpose_train/training_results/2_23_51/best_702k/
OP_BIN=./openpose/build/examples/openpose/openpose.bin

op_bld: openpose/build/Makefile
	cd openpose/build; make -j`nproc`

openpose/build/Makefile: openpose/build
	@echo "Try op_bldc for CPU version or op_bldg for GPU(CUDA) version"

op_bldc: openpose/build
	cd openpose/build; cmake -DWITH_3D_RENDERER=True -DUSE_CUDNN=OFF -DBUILD_PYTHON=ON ../

# Build with CUDNN
op_bldg: openpose/build
	cd openpose/build; cmake -DWITH_3D_RENDERER=True -DUSE_CUDNN=ON -DBUILD_PYTHON=ON ../

openpose/build: openpose
	cd openpose; mkdir build

openpose/Makefile: Makefiles/Makefile.OpenPose
	cd openpose; ln -s ../Makefiles/Makefile.OpenPose Makefile

op_eval_coco:
	#cd openpose; make eval_coco
	$(OP_BIN) --image_dir $(IMAGE_FOLDER) --display 0 --render_pose 0 --cli_verbose 0.2 --write_coco_json \
		$(OP_JSON_FOLDER)/1.json --write_coco_json_variants 1 \
    	--model_folder openpose/models
	python3 Utils/analyze_json.py -i openpose/evaluation/coco_val_jsons/1.json

op_eval_coco_max:
	$(OP_BIN) --image_dir $(IMAGE_FOLDER) --display 0 --render_pose 0 --cli_verbose 0.2 --write_coco_json \
		$(OP_JSON_FOLDER)/1_max.json --write_coco_json_variants 3 \
		--maximize_positives \
    	--model_folder openpose/models
	python3 Utils/analyze_json.py -i openpose/evaluation/coco_val_jsons/1_max.json

eval_mpii: eval-mpii-pose
	cd eval-mpii-pose; octave evalMPII.m

view_mpii:
	python3 Utils/mpii_view.py -d /Data/PoseEstimation/MPII/ -i /Data/PoseEstimation/MPII/annot/valid.json -v 2

eval-mpii-pose:
	git clone https://github.com/anibali/eval-mpii-pose.git

# To evaluate trained model, copy or link training model (pose_deploy.prototxt) and trained caffemodel to openpose_train_eval/pose/body_25
ITER=584000
IMGID=785

op_view:
	python3 Utils/view_keypoints.py -i openpose/evaluation/coco_val_jsons/1.json -v $(VERBOSE)

op_view_max:
	python3 Utils/view_keypoints.py -i openpose/evaluation/coco_val_jsons/1_max.json -v $(VERBOSE)

op_coco_demo:
	@echo "Ex) make op_coco_demo IMGID=781 VERBOSE=2"
	@echo "	tf36 environment is recommended"
	cd COCO_API; python3 pycocoDemo.py -i $(IMGID) -v $(VERBOSE)

op_coco_eval_demo:
	@echo "	tf36 environment is recommended"
	cd COCO_API; python3 pycocoEvalDemo.py

op_coco_eval:
	@echo "	tf36 environment is recommended"
	cd COCO_API; python3 pycocoEvalDemo.py -t val2017 -r ../openpose/evaluation/coco_val_jsons/1.json

opt_coco_eval:
	@echo "	tf36 environment is recommended"
	cd COCO_API; python3 pycocoEvalDemo.py -t val2017 -r ../openpose_train_eval/1.json

op_coco_view_demo:
	@echo "	tf36 environment is recommended"
	cd COCO_API; python3 pycocoViewDemo.py

#
# OpenPose Training
#
# 	Get OpenPose Train
opt: openpose_train
	@echo "Use script files for each purpose"

openpose_train:
	git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose_train.git

opt_view:
	python3 Utils/view_keypoints.py -i openpose_train_eval/1.json -v $(VERBOSE)

opt_eval_set:
	rm openpose_train_eval/pose/body_25/pose_iter_584000.caffemodel
	cd openpose_train_eval/pose/body_25; ln -s ../../../openpose_train/training_results/pose/model/pose_iter_$(ITER).caffemodel pose_iter_584000.caffemodel

#	OpenPose Evaluation for COCO with Newly Trained Model
#		caffe uses pose_iter_584000.caffemodel

opt_eval_coco: op openpose_train_eval/pose/body_25/pose_deploy.prototxt openpose_train_eval/pose/body_25/pose_iter_584000.caffemodel
	$(OP_BIN) --image_dir $(IMAGE_FOLDER) --display 0 --render_pose 0 --cli_verbose 0.2 \
    --write_coco_json ${OPT_JSON_FOLDER}/1.json --write_coco_json_variants 1 \
    --model_folder ./openpose_train_eval
	# time ./OpenPose_Scripts/tests/pose_accuracy_coco_val_trained.sh |tee Logs/pose_accuracy_coco_val_trained.txt
	python3 Utils/analyze_json.py -i openpose_train_eval/1.json

openpose_train_eval/pose/body_25/pose_deploy.prototxt:
	cp openpose_train/training_results/pose/pose_deploy.prototxt openpose_train_eval/pose/body_25/

openpose_train_eval/pose/body_25/pose_iter_584000.caffemodel:
	@echo "cp openpose_train/training_results/pose/model/pose_iter_xxxx.caffemodel openpose_train_eval/pose/body_25/pose_iter_584000.caffemodel"
	cd openpose_train_eval/pose/body_25; ln -s ../../../openpose_train/training_results/pose/model/pose_iter_$(ITER).caffemodel pose_iter_584000.caffemodel

op_eval_all: op
	cd openpose; make eval_all

op_clean:
	cd openpose; rm -rf build

# CUDA-10.0 is OK, CUDA-10.2 is not OK
#	See opc_hdf5 and opc-py3 for problem solving
#	gcc version 5.x is recommended
HDF5_VER = 8.0.2
opc: openpose_caffe_train openpose_caffe_train/Makefile.config
	cd openpose_caffe_train; make all -j`nproc` && make pycaffe -j`nproc`

opc_clean:
	cd openpose_caffe_train; make clean

# To solve hdf5 related problems
opc_hdf5:
	cd openpose_caffe_train; find . -type f -exec sed -i -e 's^"hdf5.h"^"hdf5/serial/hdf5.h"^g' -e 's^"hdf5_hl.h"^"hdf5/serial/hdf5_hl.h"^g' '{}' \;
	cd /usr/lib/x86_64-linux-gnu; \
		sudo ln -s libhdf5_serial.so.$(HDF_VER) libhdf5.so \
		sudo ln -s libhdf5_serial_hl.so.$(HDF_VER) libhdf5_hl.so \
		sudo ln -s libhdf5_serial.a libhdf5.a \
		sudo ln -s libhdf5_serial_hl.a libhdf5_hl.a

# To solve python3 related problems
# 	Recommend Python 3.6 environment
opc_py3:
	pip install -y opencv-contrib-python
	cd /usr/lib/x86_64-linux-gnu; \
		sudo ln -s libboost_python-py35.a libboost_python3.a; \
		sudo ln -s libboost_python-py35.so libboost_python3.so

openpose_caffe_train:
	git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose_caffe_train.git

openpose_caffe_train/Makefile.config:
	cp Makefile.OpenPose_Caffe_Train openpose_caffe_train/Makefile.config

# Prepare for Training OpenPose with Caffe
openpose_train/training_results:
opct: openpose_train/training/d_setLayers.py
	@echo "Recommend Python 3.6 and CUDA 10.0"
	cd openpose_train/training; python3 d_setLayers.py

# start the training with 2 GPUs (0-1)
# 	Recommend Python 3.6 environment
GPUS = '0,1'	# default 2 GPUs
# make opctr GPUS='0,1,2,3' can be used for 4 GPUs
#	from numpy.lib.arraypad import _validate_lengths
#	ImportError: cannot import name '_validate_lengths'		# python >= 3.6
opctr: openpose_train/training_results openpose_train/dataset/vgg/vgg_deploy.prototxt openpose_train/dataset/vgg/VGG_ILSVRC_19_layers.caffemodel
	cd openpose_train/training_results/pose; time bash train_pose.sh $(GPUS)
	@echo "See openpose_train/training_results/pose/training_log.txt"

opctres: openpose_train/training_results openpose_train/dataset/vgg/vgg_deploy.prototxt openpose_train/dataset/vgg/VGG_ILSVRC_19_layers.caffemodel
	@echo "Modify iteration number in openpose_train/training_results/pose/resume_train_pose.sh"
	cd openpose_train/training_results/pose; time bash resume_train_pose.sh $(GPUS) $(ITER)
	@echo "See openpose_train/training_results/pose/training_log.txt"

openpose_train/dataset/vgg/vgg_deploy.prototxt:
	cd openpose_train/dataset/vgg; wget https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/f43eeefc869d646b449aa6ce66f87bf987a1c9b5/VGG_ILSVRC_19_layers_deploy.prototxt; mv VGG_ILSVRC_19_layers_deploy.prototxt vgg_deploy.prototxt

openpose_train/dataset/vgg/VGG_ILSVRC_19_layers.caffemodel:
	cd openpose_train/dataset/vgg; wget http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel

openpose_train/dataset/vgg: openpose_train/dataset
	mkdir openpose_train/dataset/vgg

openpose_train/dataset:
	mkdir openpose_train/dataset

# Copy modified file
openpose_train/training/d_setLayers.py: OpenPose_Scripts/train/d_setLayers.py
	cp OpenPose_Scripts/train/d_setLayers.py openpose_train/training/

# Copy modified file
openpose_train/training/generateProtoTxt.py: OpenPose_Scripts/train/generateProtoTxt.py
	cp OpenPose_Scripts/train/generateProtoTxt.py openpose_train/training/


mAP:
	python Utils/find_max_mAP.py

dense: DensePose
	@echo "DensePose cloned"
	cd DensePose; make
	@echo "DensePose compiled"

DensePose:
	git clone https://github.com/facebookresearch/DensePose.git

densei: DensePose
	cd DensePose; sudo make install

up: UniPose UniPose/Makefile
	@echo "UniPose cloned"

UniPose:
	git clone https://github.com/bmartacho/UniPose.git

UniPose/Makefile:
	cd UniPose; ln -s ../Makefiles/Makefile.UniPose ./Makefile

dp: DarkPose DarkPose/Makefile DarkPose/data
	cd DarkPose/lib; make
	@echo "DarkPose library compiled"

DarkPose:
	git clone https://github.com/ilovepose/DarkPose.git
	@echo "DarkPose cloned"

DarkPose/Makefile:
	cd DarkPose; ln -s ../Makefiles/Makefile.DarkPose ./Makefile

Makefile DarkPose/data:
	cd DarkPose; mkdir data; cd data; ln -s /Data/PoseEstimation/COCO coco; ln -s /Data/PoseEstimation/MPII mpii

yv5: yolov5 yolov5/Makefile
	@echo "Yolo v5 cloned"
	ln -s /Data/PoseEstimation/COCO coco

yolov5:
	git clone https://github.com/ultralytics/yolov5.git

yolov5/Makefile:
	cd yolov5; ln -s ../Makefiles/Makefile.yolov5 Makefile

yv7: yolov7 yolov7/Makefile
	@echo "Yolo v7 cloned"

yolov7:
	git clone https://github.com/WongKinYiu/yolov7.git

yolov7/Makefile:
	cd yolov7; ln -s ../Makefiles/Makefile.yolov7 Makefile

# Install detectron2
inst_det2:
	python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

det2: detectron2 detectron2/Makefile
	@echo "detectron2 cloned"

detectron2:
	git clone https://github.com/facebookresearch/detectron2.git

detectron2/Makefile:
	cd detectron2; ln -s ../Makefiles/Makefile.detectron2 Makefile

# find_max_mAP.py           mpii_view.py
# analyze_h5.py    

db:
	python Utils/drawBox.py               

nms:
	python Utils/nms.py

#
# Selective Pre-Processing
#
# Pre-Processing CLAHE only 
# -t 1|2 {HE|CLAHE} -g <grid_size> -l <clip_limit>
GRID_SIZE = 8
CLIP_LIMIT = 2.0

spp: val2017_clahe
	python Utils/selective_preprocessing.py  -i /Data/PoseEstimation/COCO/val2017 -o val2017_clahe \
		-t 2 -g $(GRID_SIZE) -l $(CLIP_LIMIT)
	# python Utils/selective_preprocessing.py  -i /Data/PoseEstimation/COCO/val2017 -o val2017_processed -e 1

sharp:
	python Utils/sharpening.py

# Pre-Processing HE & CLAHE
spp2: val2017_he val2017_he_clahe
	python Utils/selective_preprocessing.py  -i /Data/PoseEstimation/COCO/val2017 -o val2017_he -t 1
	python Utils/selective_preprocessing.py  -i val2017_he -o val2017_he_clahe -t 2
	# python Utils/selective_preprocessing.py  -i /Data/PoseEstimation/COCO/val2017 -o val2017_processed -e 1

val2017_he_clahe:
	mkdir -p val2017_he_clahe

he: val2017_he
	python Utils/histogramEqualization.py  -i /Data/PoseEstimation/COCO/val2017 -o val2017_he

hej: val2017_he
	python Utils/histogramEqualization.py  -i /Data/PoseEstimation/COCO/val2017 -o val2017_he \
		-j openpose/evaluation/coco_val_jsons/1.json

val2017_processed:
	mkdir -p val2017_processed

val2017_he:
	mkdir -p val2017_he

cla: val2017_clahe
	python Utils/histogramEqualization.py  -i /Data/PoseEstimation/COCO/val2017 -o val2017_clahe -t clahe

claj: val2017_clahe
	python Utils/histogramEqualization.py  -i /Data/PoseEstimation/COCO/val2017 -o val2017_clahe -t clahe \
		-j openpose/evaluation/coco_val_jsons/1.json

val2017_clahe:
	mkdir -p val2017_clahe

onnx:
	cd ONNX; make

# EvoPose2D Demo
# Requires evopose2d environment
evod: evopose2d
	# python demo.py -c [model_name].yaml -p [/path/to/coco/dataset]
	cd evopose2d; python demo.py -c evopose2d_M_f32.yaml -p /Data/PoseEstimation/COCO/

evoi: evopose2d
	conda create -n evopose2d python==3.7
	conda activate evopose2d
	cd evopose2d; pip install -r requirements.txt

evopose2d:
evo:
	git clone https://github.com/wmcnally/evopose2d.git
	cd evopose2d; 
		ln -s ../Makefiles/Makefile.EvoPose2D Makefile; \
		ln -s /Data/PoseEstimation/EvoPose2D/Models models

UDP-Pose:
	git clone https://github.com/HuangJunJie2017/UDP-Pose.git

mmpose:
	git clone https://github.com/HuangJunJie2017/mmpose.git

ep:
	cd ETRI_Pose; make

ViTPose:
	git clone https://github.com/ViTAE-Transformer/ViTPose.git
	cd ViTPose; ln -s ../Makefiles/Makefile.ViTPose Makefile
	cd ViTPose; mkdir data; cd data; mkdir coco; cd coco; ln -s /Data/PoseEstimation/COCO/annotations .; \
		ln -s /Data/PoseEstimation/COCO/person_detection_results .; ln -s /Data/PoseEstimation/COCO/val2017 .
