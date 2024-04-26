from torch.utils.data import Dataset
import cv2  # Assuming you're using OpenCV for image loading
import os 

class SalDS(Dataset):
    def __init__(self, data_dir='', ext = {'imgs': 'jpg', 'sal': 'png', 'fix': 'png' }, color_space = cv2.COLOR_BGR2RGB ):
        

        # define directories
        self.img_dir = os.path.join(data_dir,'imgs')
        self.sal_dir = os.path.join(data_dir,'sal')
        self.fix_dir = os.path.join(data_dir, 'fix')
        
        # get image names 
        self.image_names = sorted(list(map( lambda x : x.split('.') , os.listdir(self.img_dir))))
        self.exts = ext

        # define color space
        self.color_space = color_space

    def __len__(self):
        return len(self.image_names)  # Replace with actual list length

    def __getitem__(self, idx):
        
        image_name = self.image_names[idx]
        image_path = os.path.join(self.img_dir, image_name + self.exts['imgs'] )
        saliency_path = os.path.join(self.sal_dir, image_name + self.exts['sal'] )
        fixation_path = os.path.join(self.fix_dir, image_name + self.exts['fix'] )

        # Load image and maps (handle grayscale loading for saliency/fixation)
        image = cv2.imread(image_path)
        saliency_map = cv2.imread(saliency_path, 0)  # Grayscale mode
        fixation_map = cv2.imread(fixation_path, 0)  # Grayscale mode

        # TODO: Preprocess data (resize, normalize, etc.)
        # ...

        #color spaces 
        image = cv2.cvtColor(image, self.color_space)

        # Convert to tensors
        image_tensor = torch.from_numpy(image).float()
        image_tensor = image_tensor.permute((2, 0, 1))

        saliency_tensor = torch.from_numpy(saliency_map).float()
        fixation_tensor = torch.from_numpy(fixation_map).float()

        return image_tensor, saliency_tensor, fixation_tensor
