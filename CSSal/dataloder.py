from torch.utils.data import Dataset
import cv2  # Assuming you're using OpenCV for image loading

class SalDS(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        # Load list of image filenames and corresponding map filenames here

    def __len__(self):
        return len(self.image_filenames)  # Replace with actual list length

    def __getitem__(self, idx):
        image_path = self.image_filenames[idx]
        saliency_path = self.saliency_filenames[idx]
        fixation_path = self.fixation_filenames[idx]

        # Load image and maps (handle grayscale loading for saliency/fixation)
        image = cv2.imread(image_path)
        saliency_map = cv2.imread(saliency_path, 0)  # Grayscale mode
        fixation_map = cv2.imread(fixation_path, 0)  # Grayscale mode

        # Preprocess data (resize, normalize, etc.)
        # ...

        # Convert to tensors
        image_tensor = torch.from_numpy(image).float()
        saliency_tensor = torch.from_numpy(saliency_map).float()
        fixation_tensor = torch.from_numpy(fixation_map).float()

        return image_tensor, saliency_tensor, fixation_tensor
