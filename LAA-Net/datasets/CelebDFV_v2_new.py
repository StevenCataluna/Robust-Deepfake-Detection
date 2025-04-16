import os
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from package_utils.utils import file_extention
import simplejson as json
from box import Box as edict
import io
import cv2

PREFIX_PATH = '/data/deepfake_cluster/datasets_df/CelebDF/c0/'



class CDFV2(Dataset):
    def __init__(self, cfg, split='test', transform=None):
        self.cfg = cfg
        self.split = split
        # Don't perform transformations for testing set
        # self.transform = transforms.Compose([
        #     transforms.Resize((384, 384)),  
        #     transforms.RandomHorizontalFlip(p=0.5),  
        #     transforms.RandomResizedCrop(
        #         size=(384, 384),  # Crop to (384, 384)
        #         scale=(0.15, 0.5)  # Scale limit [0.15, 0.5]
        #     ),
        #     transforms.RandomAffine(
        #         degrees=0,  # No rotation
        #         scale=(0.15, 0.5)  
        #     ),
        #     transforms.ToTensor(),  # Convert to tensor
        #     transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),  
        #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
        # ])

        # Only apply necessary transormations - resizing etc
        self.transform = transforms.Compose([
        transforms.Resize((384, 384)),  
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize with the same values as training
        ])

        self.to_tensor = transforms.ToTensor()

        # Load dataset paths and labels
        self.img_paths, self.labels, self.mask_paths, self.ot_props = self._load_from_file(split)

    def _load_from_path(self, split):

        # Ensure the root directory exists
        root_path = self.cfg['DATA'][split.upper()]['ROOT']
        assert os.path.exists(root_path), f"Root path to dataset does not exist: {root_path}"

        data = self.cfg['DATA']
        data_type = data['TYPE']
        fake_types = self.cfg['DATA'][split.upper()]['FAKETYPE']
        img_paths, labels, vid_ids, mask_paths, ot_props = [], [], [], [], []
        
        count = 0
        n_samples = 100000

        # Load image data for each type of fake techniques
        for idx, ft in enumerate(fake_types):
            data_dir = os.path.join(root_path, self.split, data_type, ft)
            if not os.path.exists(data_dir):
                raise ValueError(f"Data directory does not exist: {data_dir}")
            
            for sub_dir in os.listdir(data_dir):
                sub_dir_path = os.path.join(data_dir, sub_dir)
                img_paths_ = glob(f'{sub_dir_path}/*.{self.cfg["IMAGE_SUFFIX"]}')
                
                if 'Celeb-synthesis' in ft:
                    if count < n_samples:
                        n_add = len(img_paths_) if ((n_samples - count) > len(img_paths_)) else (n_samples - count)
                        count += n_add
                        print(f'n fake samples added --- {count}')
                    else:
                        continue
                else:
                    n_add = len(img_paths_)

                img_paths.extend(img_paths_[:n_add])
                labels.extend(np.full(n_add, int(ft == 'Celeb-synthesis')))
                vid_ids.extend([sub_dir] * n_add)  # Use sub_dir as video ID
                
        print(f'{len(img_paths)} image paths have been loaded from CDFv2!')
        return img_paths, labels, vid_ids, mask_paths, ot_props
    
    def _load_from_file(self, split, anno_file=None):

        print("TEST TEST TEST="+ self.cfg.DATA[self.split.upper()].ROOT)

        assert os.path.exists(self.cfg.DATA[self.split.upper()].ROOT), "Root path to dataset can not be invalid!"
        data_cfg = self.cfg.DATA
        
        if anno_file is None:
            anno_file = data_cfg[split.upper()].ANNO_FILE
        if not os.access(anno_file, os.R_OK):
            anno_file = os.path.join(self.cfg.DATA[self.split.upper()].ROOT, anno_file)
        assert os.access(anno_file, os.R_OK), "Annotation file can not be invalid!!"

        f_name, f_extention = file_extention(anno_file)
        data = None
        image_paths, labels, mask_paths, ot_props = [], [], [], []
        f = open(anno_file)
        if f_extention == '.json':
            data = json.load(f)
            data = edict(data)['data'] 

            for item in data:
                assert 'image_path' in item.keys(), 'Image path must be available in item dict!'
                image_path = item.image_path
                ot_prop = {}
                
                # Assign labels
                if not 'label' in item.keys():
                    # lb = (('fake' in image_path) or (('original' not in image_path) and ('aligned' not in image_path)))
                    lb = (('Celeb-synthesis' in image_path) or ('real' not in image_path))
                else:
                    lb = (item.label == 'fake')
                lb_encoded = int(lb)
                labels.append(lb_encoded)
                
                if PREFIX_PATH in item.image_path:
                    image_path = item.image_path.replace(PREFIX_PATH, self.cfg.DATA[self.split.upper()].ROOT)
                else:
                    image_path = os.path.join(self.cfg.DATA[self.split.upper()].ROOT, item.image_path)
                image_path = image_path.replace('\\', '/')
                image_paths.append(image_path)

                # Appending more data properties for data loader
                if 'mask_path' in item.keys():
                    mask_path = item.mask_path
                    if PREFIX_PATH in item.mask_path:
                        mask_path = item.mask_path.replace(PREFIX_PATH, self.cfg.DATA[self.split.upper()].ROOT)
                    else:
                        mask_path = os.path.join(self.cfg.DATA[self.split.upper()].ROOT, item.mask_path)
                    mask_paths.append(mask_path)
                if 'best_match' in item.keys():
                    best_match = item.best_match
                    best_match = [os.path.join(self.cfg.DATA[self.split.upper()].ROOT, bm) for bm in best_match if \
                        self.cfg.DATA[self.split.upper()].ROOT not in bm]
                    ot_prop['best_match'] = best_match
                for lms_key in ['aligned_lms', 'orig_lms']:
                    if lms_key in item.keys():
                        f_lms = np.array(item[lms_key])
                        ot_prop[lms_key] = f_lms
                    
                ot_props.append(ot_prop)
        else:
            raise Exception(f'{f_extention} has not been supported yet! Please change to Json file!')
        
        print('{} image paths have been loaded!'.format(len(image_paths)))
        print("EXAMPLE IMAGE = " + str(image_paths[1]))
        print("IMAGE EXISTS = " + str(os.path.isfile(image_paths[1])))
        return image_paths, labels, mask_paths, ot_props
    
    def _apply_compression(self, image):
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=70)
        buffer.seek(0)

        # Reload the compressed image
        compressed_img = Image.open(buffer).convert('RGB')
        return compressed_img

    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        vid_id = img_path.split('/')[-2]

        img = Image.open(img_path).convert('RGB')
        img = self._apply_compression(img)

        if self.transform is not None:
            img = self.transform(img)
        
        return img, label, vid_id