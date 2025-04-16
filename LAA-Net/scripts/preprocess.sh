#Crop images

#Crop train
# python package_utils/images_crop.py -d Original -c c23 -n 32 -t train  

# #Crop val
# python package_utils/images_crop.py -d Original -c c23 -n 32 -t val   

# #Crop test
# python package_utils/images_crop.py -d Original -c c23 -n 32 -t test  


#Preprocess train
python package_utils/geo_landmarks_extraction.py --config configs/data_preprocessing_train_c23.yaml --extract_landmark --save_aligned    

#Preprocess val
python package_utils/geo_landmarks_extraction.py --config configs/data_preprocessing_val_c23.yaml --extract_landmark --save_aligned    

#Preprocess test
python package_utils/geo_landmarks_extraction.py --config configs/data_preprocessing_test_c23.yaml --extract_landmark --save_aligned    