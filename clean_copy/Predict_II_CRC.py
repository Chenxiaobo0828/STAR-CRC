import os
import torch
import SimpleITK
import numpy as np
import pandas as pd
from skimage import exposure
from torchvision import transforms
from Model import swin_tiny_patch4_window7_224 as create_model

import warnings

warnings.filterwarnings("ignore")


def main():
    Dir = r'H:/II_CRC/DATA'
    predict_cohort = ['Center_VII']
    images_path = []
    for cohort in predict_cohort:
        cohort_path = os.path.join(Dir, cohort)
        for image_name in os.listdir(cohort_path):
            if image_name.endswith('image.nii.gz'):
                image_path = os.path.join(cohort_path, image_name)
                images_path.append(image_path)

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize((224, 224)),
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # model_name = 'CRC-20241123-seed1-fine-AX-model-0.pth'
    # model_name = 'CRC-20241123-seed1-fine-COR-model-0.pth'
    model_name = 'CRC-20241123-seed1-fine-SAG-model-0.pth'
    device = 'cuda:0'
    model2 = create_model(num_classes=1).to(device)
    model_weight_path = os.path.join(Dir, 'weight', model_name)
    model2.load_state_dict(torch.load(model_weight_path, map_location=device))
    model2.eval()

    predict_data = pd.DataFrame({"image_path": [],
                                 "risk_score": []})

    index = 0
    print('The prediction work is in progress, please wait !!')
    for image_path in images_path:
        image = SimpleITK.ReadImage(image_path)
        image = SimpleITK.GetArrayFromImage(image)
        image = image.transpose(2, 1, 0)
        image = image.astype(np.float32)
        image = exposure.rescale_intensity(image, out_range="float32")
        image = data_transform(image)
        image = image.unsqueeze(0)
        risk_pred = model2(image.to(device))
        predict_data.loc[index, 'image_path'] = image_path
        predict_data.loc[index, 'risk_score'] = risk_pred.cpu().detach().numpy()
        index += 1
    predict_data.to_csv(os.path.join(Dir, "Predict_Outcome.csv"), encoding="gbk", index=False)
    print('The predict work has been successfully completed !!')


if __name__ == '__main__':
    main()
