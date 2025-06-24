import os
import pandas as pd
from torch.utils.data import  Dataset
import torch
from datetime import datetime
from skimage import io
from pydicom import dcmread
from sklearn.preprocessing import RobustScaler

class FundusDataset(Dataset):
    def __init__(self, root_dir, labels_file, split, scalers=None, transform=None):
        # Read the CSV file containing annotations
        self.annotations = pd.read_csv(labels_file, encoding='ISO-8859-1')

        # (Optional/Depending on your dataset) Convert specific columns to numeric, coercing errors to NaN
        columns_to_convert = ['waist', 'AGE', 'Height', 'Weight', 'BMI', 'waist',\
            'SBP', 'DBP', 't-chol', 'gluc', 'TG', 'HbA1c', 'BUN', 'Creatinin', 'Uric Acid',\
                'ALT', 'CRP', 'CA19-9', 'cyfra21-1']
        for col in columns_to_convert:
            self.annotations[col] = pd.to_numeric(self.annotations[col], errors='coerce')
        
        # (Optional/Depending on your dataset) Define columns to check for missing values
        required_columns = [
            'ID', 'subid', 'ODT', 'SEX', 'AGE', 'Height', 'Weight', 'BMI', 'waist',
            'SBP', 'DBP', 't-chol', 'gluc', 'TG', 'HbA1c', 'BUN', 'Creatinin', 'Uric Acid'
        ]

        # Define columns to scale
        self.columns_to_scale = [
            'AGE', 'Height', 'Weight', 'BMI', 'waist', 'SBP', 'DBP', 
            't-chol', 'gluc', 'TG', 'HbA1c', 'BUN', 'Creatinin', 'Uric Acid'
        ]

        # (Optional/Depending on your dataset) Filter out rows with missing values in the required columns
        self.annotations.dropna(subset=required_columns, inplace=True)

        # Calculate the indices for train, validation, and test splits 
        total_samples = len(self.annotations)
        # (Optional/Depending on your dataset)
        #train_end = int(total_samples * 0.7)+1
        #val_end = train_end + int(total_samples * 0.1)+1
        train_end = int(total_samples * 0.7)
        val_end = train_end + int(total_samples * 0.1)
        
        self.scalers = scalers # for validation and test set scaling
        self.original_ages = self.annotations['AGE'].values.copy()  # 원본 age 값 저장
        
        if split == 'train':
            self.annotations = self.annotations.iloc[:train_end]
            self.original_annotations = self.annotations.copy()
        elif split == 'val':
            self.annotations = self.annotations.iloc[train_end:val_end]
            self.original_annotations = self.annotations.copy()
            if self.scalers:
                for col in self.columns_to_scale:
                    if col in self.scalers:
                        self.annotations[col] = self.scalers[col].transform(self.annotations[col].values.reshape(-1, 1)).flatten()
        elif split == 'test':
            self.annotations = self.annotations.iloc[val_end:]
            self.original_annotations = self.annotations.copy()
            if self.scalers:
                for col in self.columns_to_scale:
                    if col in self.scalers:
                        self.annotations[col] = self.scalers[col].transform(self.annotations[col].values.reshape(-1, 1)).flatten()
        else:
            raise ValueError("Invalid split argument. Choose from 'train', 'val', 'test'.")
        
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    
    def fit_and_transform(self):
        # Fit the scalers to the training data and transform the columns
        self.scalers = {}
        for col in self.columns_to_scale:
            self.scalers[col] = RobustScaler()
            self.annotations[col] = self.scalers[col].fit_transform(self.annotations[col].values.reshape(-1, 1)).flatten()
            
        return self.scalers

    def __getitem__(self, index):
        # Modify based on your dataset, ensuring the index is within the bounds of the dataset
        iid = self.annotations.iloc[index, 0]
        sub_id = self.annotations.iloc[index, 1]
        date_str = self.annotations.iloc[index, 2]
        gender = self.annotations.iloc[index, 3]
        age = self.annotations.iloc[index, 4]
        height = self.annotations.iloc[index, 5]
        weight = self.annotations.iloc[index, 6]
        bmi = self.annotations.iloc[index, 7]
        waist = self.annotations.iloc[index, 8]
        sbp = self.annotations.iloc[index, 11]
        dbp = self.annotations.iloc[index, 12]
        t_chol = self.annotations.iloc[index, 21]
        glucose = self.annotations.iloc[index, 24]
        triglycerides = self.annotations.iloc[index, 25]
        hba1c = self.annotations.iloc[index, 27]
        bun = self.annotations.iloc[index, 28]
        creatinine = self.annotations.iloc[index, 29]
        u_acid = self.annotations.iloc[index, 30]
        
        o_age = self.original_annotations.iloc[index, 4]
        o_height = self.original_annotations.iloc[index, 5]
        o_weight = self.original_annotations.iloc[index, 6]
        o_bmi = self.original_annotations.iloc[index, 7]
        o_waist = self.original_annotations.iloc[index, 8]
        o_sbp = self.original_annotations.iloc[index, 11]
        o_dbp = self.original_annotations.iloc[index, 12]
        o_t_chol = self.original_annotations.iloc[index, 21]
        o_glucose = self.original_annotations.iloc[index, 24]
        o_triglycerides = self.original_annotations.iloc[index, 25]
        o_hba1c = self.original_annotations.iloc[index, 27]
        o_bun = self.original_annotations.iloc[index, 28]
        o_creatinine = self.original_annotations.iloc[index, 29]
        o_u_acid = self.original_annotations.iloc[index, 30]

        # (Optional/Depending on your dataset) Convert the date string by checking its format
        try:
            if '-' in date_str:
                # If the date is in 'YYYY-MM-DD' format, convert it to '%m/%d/%Y'
                date = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%d")
            else:
                # If the date is already in '%m/%d/%Y' format
                date = datetime.strptime(date_str, "%m/%d/%Y").strftime("%Y%m%d")
        except ValueError as e:
            print(f"Error in date conversion for index {index}: {e}")
            return None  # Skip the entry if the date format is invalid

        # Create folder name in the expected format
        folder_name = f"{sub_id}_{date}_{gender}_{o_age}"
        folder_path = os.path.join(self.root_dir, folder_name)

        dicom_files = []  # Initialize an empty list for DICOM files

        try:
            dicom_files = [f for f in os.listdir(folder_path) if f.endswith('.dcm')]
        except FileNotFoundError as e:
            print(f"Folder not found: {folder_path}. Error: {e}")
            return None  # Skip if folder is not found

        # Ensure there is at least one DICOM file
        if not dicom_files:
            print(f"No DICOM files found in the folder: {folder_path}")
            return None  # Skip if no DICOM files are found

        # Assuming there is only one DICOM file per folder
        if iid == 0:
            dicom_file = sorted(dicom_files)[1]
        elif iid == 1:
            dicom_file = sorted(dicom_files)[1]
        dicom_file_path = os.path.join(folder_path, dicom_file)

        # Load the DICOM file and extract the image data
        dicom_data = dcmread(dicom_file_path)
        image = dicom_data.pixel_array.astype(float)

        # Retrieve the label (e.g., age or any other column as per your needs)
        gender_binary = 0 if gender == 'F' else 1 if gender == 'M' else None
    
        original_label = torch.tensor([
            gender_binary,
            o_age,
            o_height,
            o_weight,
            o_bmi,
            o_waist,
            o_sbp,
            o_dbp,
            o_t_chol,
            o_glucose,
            o_triglycerides,
            o_hba1c,
            o_bun,
            o_creatinine,
            o_u_acid
        ], dtype=torch.float)
        
        y_label = torch.tensor([
            gender_binary,
            age,
            height,
            weight,
            bmi,
            waist,
            sbp,
            dbp,
            t_chol,
            glucose,
            triglycerides,
            hba1c,
            bun,
            creatinine,
            u_acid
        ], dtype=torch.float)
        
        # Apply transformations
        image = image.astype('uint8')
        if self.transform:
            image = self.transform(image)

        return image, y_label, folder_name, original_label