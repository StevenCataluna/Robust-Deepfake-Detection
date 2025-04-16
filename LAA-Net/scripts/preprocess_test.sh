#! /bin/bash

#Preprocess deepfake
python package_utils/images_crop.py -d Deepfakes -c c23 -n 32 -t test  
# python package_utils/geo_landmarks_extraction.py --config configs/data_preprocessing_test_Deepfake_c23.yaml --extract_landmark --save_aligned    

# #Preprocess Face2Face
python package_utils/images_crop.py -d Face2Face -c c23 -n 32 -t test  
# python package_utils/geo_landmarks_extraction.py --config configs/data_preprocessing_test_Face2Face_c23.yaml --extract_landmark --save_aligned    

# # #Preprocess FaceShifter
python package_utils/images_crop.py -d FaceSwap -c c23 -n 32 -t test  
# python package_utils/geo_landmarks_extraction.py --config configs/data_preprocessing_test_FaceSwap_c23.yaml --extract_landmark --save_aligned    

#Preprocess DeepfakeDetection
# python package_utils/images_crop.py -d DeepFakeDetection -c c23 -n 32 -t test  
# python package_utils/geo_landmarks_extraction.py --config configs/data_preprocessing_test_DFD_c23.yaml --extract_landmark --save_aligned    

#Preprocess NeuralTextures
python package_utils/images_crop.py -d NeuralTextures -c c23 -n 32 -t test  

#Preprocess Original
python package_utils/images_crop.py -d Original -c c23 -n 32 -t test  


python package_utils/geo_landmarks_extraction.py --config configs/data_preprocessing_c23.yaml --extract_landmark  --save_aligned



