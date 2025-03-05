# Enhanced Risk Stratification for Stage II Colorectal Cancer Using Deep Learning-based CT Classifier and Pathological Markers to Optimize Adjuvant Therapy Decision

IRIS-CRC offers a more precise and personalized risk assessment than current guideline-based risk factors, potentially sparing low-risk patients from unnecessary adjuvant chemotherapy while identifying high-risk individuals for more aggressive treatment. This novel approach holds promise for improving clinical decision-making and outcomes in stage II CRC.

![figure_2.jpg](https://github.com/Chenxiaobo0828/STAR-CRC/blob/main/figure_2.jpg)

## Requirements

- Python (3.9.12)
- torch (2.2.2+cu118)
## Usage

1. Data Preparation
	You need to prepare the following files:

	- **Imaging files**: The format should be structured as follows:
		```
		Train_Cohort
		├── Center I
		  ├── Crop_AX_Smax+1_0001_image.nii.gz
		  ├── Crop_AX_Smax+0_0001_image.nii.gz
		  ├── Crop_AX_Smax-1_0001_image.nii.gz
		  ├── Crop_AX_Smax+1_0002_image.nii.gz
		  ...
		├── Center II
		├── Center III
		└── Center IV
		
		Clinical.csv files
		```
		The file names must include the term "Smax", and should also contain one of the following: "Smax+0", "Smax+1", or "Smax-1".    

	- **Clinical files**: These should include the following columns:
		```
	    - `image_paths`:.../Crop_AX_Smax+1_patient_id_image.nii.gz
	    - `fustat`:0
	    - `futime`:78
	    ```		
	
2. Train:
    
     - **Install the PyTorch version with CUDA support.**
    
    ```
     pip3 install torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2 -f https://download.pytorch.org/whl/cu118/torch_stable.html
    ```
    
    - **Start training**:You can modify parameters such as epoch and seed in the `Train_II_CRC.py` file, and then run that file.
    
    ```
    python Train_II_CRC.py
    ```
    
3. Test:
    
    **Run test code**:   After downloading the weights, modify the weight path and prediction file path in the `Predict_II_CRC.py` file, and then run that file.
    
    ```
    python Predict_II_CRC.py
    ```

## Contact
If you have any questions, feel free to contact us through email ([chenxb@gdph.org.cn](mailto:chenxb@gdph.org.cn)) or GitHub issues. 