import numpy as np
import torch
import torch.nn as nn
import argparse
import numpy as np
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
from dataset_SePL import FundusDataset
from model_SePL import SelfPropagative_MultiTaskLearner
import cv2
import logging
from pathlib import Path
from fvcore.nn import FlopCountAnalysis
import pickle
from sklearn.metrics import mean_absolute_error, accuracy_score, mean_squared_error, roc_auc_score
from ptflops import get_model_complexity_info
from tqdm import tqdm

CUDA_LAUNCH_BLOCKING=1

np.random.seed(7226)
torch.manual_seed(7226)
torch.cuda.manual_seed(7226)

class CLAHETransform:
    def __call__(self, img):
        img_np = np.array(img)

        if len(img_np.shape) == 3:  # Color image
            img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(img_lab) 
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            img_lab = cv2.merge((cl, a, b))
            img_np = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        else:  # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_np = clahe.apply(img_np)
        
        return Image.fromarray(img_np)

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    log_dir = Path('') # Specify the directory where you want to save the logs and checkpoints
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/inference.txt' % (log_dir))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)
    
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        CLAHETransform(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    with open(args.scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    
    test_set = FundusDataset(args.test_image_dir, args.test_label_dir, split='test', scalers=scalers, transform=train_transforms)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, num_workers=3)
    
    model = SelfPropagative_MultiTaskLearner()
    if args.model_path is not None:
        model_weights = torch.load(args.model_path)
        model.load_state_dict(model_weights)
    model = nn.DataParallel(model.cuda())
    
    # (Optional/Depending on your environment) FLOPS and #Params calculation
    input_tensor = torch.randn(1,3,224,224).to('cuda')
    flops = FlopCountAnalysis(model, input_tensor)
    log_string(f"FLOPs: {flops.total()}")
    input_size = (3,224,224)
    macs, params = get_model_complexity_info(model, input_size, as_strings=True, print_per_layer_stat=True)
    log_string(f"MACs: {macs}")
    log_string(f"Parameters: {params}")

    # Lists for storing results
    gender_accs, age_maes, height_maes, weight_maes, bmi_maes, waist_maes, \
        sbp_maes, dbp_maes, tchol_maes, glu_maes, tg_maes, hba1c_maes, bun_maes, \
            creatinine_maes, u_acid_maes = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    
    # Lists for storing predictions and true values
    gender_pred, gender_true, age_true, age_pred, height_true, height_pred, weight_true, weight_pred, bmi_true, bmi_pred, waist_true, waist_pred, \
    sbp_true, sbp_pred, dbp_pred, dbp_true, tchol_true, tchol_pred, glu_true, glu_pred, tg_true, tg_pred, hba1c_pred, hba1c_true, bun_pred, bun_true, \
        creatinine_true, creatinine_pred, u_acid_true, u_acid_pred = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    model.eval()
    with torch.no_grad():
        for data, _, _, real_y in tqdm(test_loader, leave=False):
            data = data.cuda()
            real_y=real_y.unsqueeze(1)

            gender_prediction, age_prediction, height_prediction, weight_prediction, bmi_prediction, waist_prediction, sbp_prediction, \
            dbp_prediction, tchol_prediction, glu_prediction, triglycerides_prediction, hba1c_prediction, bun_prediction, creatinine_prediction, \
            u_acid_prediction = model(data)
            
            gender_prediction = (torch.sigmoid(gender_prediction) >= 0.5).float()
            
            # Inverse transform the prediction using the scalers
            age_prediction, height_prediction, weight_prediction, bmi_prediction, waist_prediction, sbp_prediction, \
            dbp_prediction, tchol_prediction, glu_prediction, triglycerides_prediction, hba1c_prediction, bun_prediction, creatinine_prediction, \
            u_acid_prediction = scalers['AGE'].inverse_transform(age_prediction.cpu()), scalers['Height'].inverse_transform(height_prediction.cpu()), scalers['Weight'].inverse_transform(weight_prediction.cpu()), \
                scalers['BMI'].inverse_transform(bmi_prediction.cpu()), scalers['waist'].inverse_transform(waist_prediction.cpu()), scalers['SBP'].inverse_transform(sbp_prediction.cpu()), \
                    scalers['DBP'].inverse_transform(dbp_prediction.cpu()), scalers['t-chol'].inverse_transform(tchol_prediction.cpu()), scalers['gluc'].inverse_transform(glu_prediction.cpu()), \
                        scalers['TG'].inverse_transform(triglycerides_prediction.cpu()), scalers['HbA1c'].inverse_transform(hba1c_prediction.cpu()), scalers['BUN'].inverse_transform(bun_prediction.cpu()), \
                            scalers['Creatinin'].inverse_transform(creatinine_prediction.cpu()), scalers['Uric Acid'].inverse_transform(u_acid_prediction.cpu())

            gender_pred.append(gender_prediction.cpu().squeeze(0))
            gender_true.append(real_y[:,:,0])

            acc_sex = accuracy_score(gender_prediction.cpu().squeeze(0), real_y[:,:,0])
            gender_accs.append(acc_sex)

            age_true.append(real_y[:,:,1])
            height_true.append(real_y[:,:,2])
            weight_true.append(real_y[:,:,3])
            bmi_true.append(real_y[:,:,4])
            waist_true.append(real_y[:,:,5])
            sbp_true.append(real_y[:,:,6])
            dbp_true.append(real_y[:,:,7])
            tchol_true.append(real_y[:,:,8])
            glu_true.append(real_y[:,:,9])
            tg_true.append(real_y[:,:,10])
            hba1c_true.append(real_y[:,:,11])
            bun_true.append(real_y[:,:,12])
            creatinine_true.append(real_y[:,:,13])
            u_acid_true.append(real_y[:,:,14])
            
            age_pred.append(age_prediction)
            height_pred.append(height_prediction)
            weight_pred.append(weight_prediction)
            bmi_pred.append(bmi_prediction)
            waist_pred.append(waist_prediction)
            sbp_pred.append(sbp_prediction)
            dbp_pred.append(dbp_prediction)
            tchol_pred.append(tchol_prediction)
            glu_pred.append(glu_prediction)
            tg_pred.append(triglycerides_prediction)
            hba1c_pred.append(hba1c_prediction)
            bun_pred.append(bun_prediction)
            creatinine_pred.append(creatinine_prediction)
            u_acid_pred.append(u_acid_prediction)
    
            torch.cuda.empty_cache()
        
        # Gender ROC AUC calculation
        all_gen_labels = np.concatenate(gender_true)
        all_gen_preds = np.concatenate(gender_pred)
        gen_roc_auc = roc_auc_score(all_gen_labels, all_gen_preds)
        
        # Concatenate all true and predicted values
        age_labels = np.concatenate(age_true)
        age_preds = np.concatenate(age_pred)
        height_labels = np.concatenate(height_true)
        height_preds = np.concatenate(height_pred)
        weight_labels = np.concatenate(weight_true)
        weight_preds = np.concatenate(weight_pred)
        bmi_labels = np.concatenate(bmi_true)
        bmi_preds = np.concatenate(bmi_pred)
        waist_labels = np.concatenate(waist_true)
        waist_preds = np.concatenate(waist_pred)
        sbp_labels = np.concatenate(sbp_true)
        sbp_preds = np.concatenate(sbp_pred)
        dbp_labels = np.concatenate(dbp_true)
        dbp_preds = np.concatenate(dbp_pred)
        tchol_labels = np.concatenate(tchol_true)
        tchol_preds = np.concatenate(tchol_pred)
        glu_labels = np.concatenate(glu_true)
        glu_preds = np.concatenate(glu_pred)
        tg_labels = np.concatenate(tg_true)
        tg_preds = np.concatenate(tg_pred)
        hba1c_labels = np.concatenate(hba1c_true)
        hba1c_preds = np.concatenate(hba1c_pred)
        bun_labels = np.concatenate(bun_true)
        bun_preds = np.concatenate(bun_pred)
        creatinine_labels = np.concatenate(creatinine_true)
        creatinine_preds = np.concatenate(creatinine_pred)
        u_acid_labels = np.concatenate(u_acid_true)
        u_acid_preds = np.concatenate(u_acid_pred)

        # MAE calculations
        mae_age = mean_absolute_error(age_labels, age_preds)
        mae_height = mean_absolute_error(height_labels, height_preds)
        mae_weight = mean_absolute_error(weight_labels, weight_preds)
        mae_bmi = mean_absolute_error(bmi_labels, bmi_preds)
        mae_waist = mean_absolute_error(waist_labels, waist_preds)
        mae_sbp = mean_absolute_error(sbp_labels, sbp_preds)
        mae_dbp = mean_absolute_error(dbp_labels, dbp_preds)
        mae_tchol = mean_absolute_error(tchol_labels, tchol_preds)
        mae_glu = mean_absolute_error(glu_labels, glu_preds)
        mae_tg = mean_absolute_error(tg_labels, tg_preds)
        mae_hba1c = mean_absolute_error(hba1c_labels, hba1c_preds)
        mae_bun = mean_absolute_error(bun_labels, bun_preds)
        mae_creatinine = mean_absolute_error(creatinine_labels, creatinine_preds)
        mae_u_acid = mean_absolute_error(u_acid_labels, u_acid_preds)
        
        # RMSE calculations
        rmse_age = mean_squared_error(age_labels, age_preds, squared=False)
        rmse_height = mean_squared_error(height_labels, height_preds, squared=False)
        rmse_weight = mean_squared_error(weight_labels, weight_preds, squared=False)
        rmse_bmi = mean_squared_error(bmi_labels, bmi_preds, squared=False)
        rmse_waist = mean_squared_error(waist_labels, waist_preds, squared=False)
        rmse_sbp = mean_squared_error(sbp_labels, sbp_preds, squared=False)
        rmse_dbp = mean_squared_error(dbp_labels, dbp_preds, squared=False)
        rmse_tchol = mean_squared_error(tchol_labels, tchol_preds, squared=False)
        rmse_glu = mean_squared_error(glu_labels, glu_preds, squared=False)
        rmse_tg = mean_squared_error(tg_labels, tg_preds, squared=False)
        rmse_hba1c = mean_squared_error(hba1c_labels, hba1c_preds, squared=False)
        rmse_bun = mean_squared_error(bun_labels, bun_preds, squared=False)
        rmse_creatinine = mean_squared_error(creatinine_labels, creatinine_preds, squared=False)
        rmse_u_acid = mean_squared_error(u_acid_labels, u_acid_preds, squared=False)
        
        log_string(
            f"Gender acc = {sum(gender_accs)/len(gender_accs)}\n"
            f"Gender auc = {gen_roc_auc}\n"
            f"Age mae = {mae_age}\n"
            f"Age rmse = {rmse_age}\n"
            f"Height mae = {mae_height}\n"
            f"Height rmse = {rmse_height}\n"
            f"Weight mae = {mae_weight}\n"
            f"Weight rmse = {rmse_weight}\n"
            f"BMI mae = {mae_bmi}\n"
            f"BMI rmse = {rmse_bmi}\n"
            f"Waist mae = {mae_waist}\n"
            f"Waist rmse = {rmse_waist}\n"
            f"SBP mae = {mae_sbp}\n"
            f"SBP rmse = {rmse_sbp}\n"
            f"DBP mae = {mae_dbp}\n"
            f"DBP rmse = {rmse_dbp}\n"
            f"T_chol mae = {mae_tchol}\n"
            f"T_chol rmse = {rmse_tchol}\n"
            f"Glucose mae = {mae_glu}\n"
            f"Glucose rmse = {rmse_glu}\n"
            f"Triglycerides mae = {mae_tg}\n"
            f"Triglycerides rmse = {rmse_tg}\n"
            f"HbA1c mae = {mae_hba1c}\n"
            f"HbA1c rmse = {rmse_hba1c}\n"
            f"BUN mae = {mae_bun}\n"
            f"BUN rmse = {rmse_bun}\n"
            f"Creatinine mae = {mae_creatinine}\n"
            f"Creatinine rmse = {rmse_creatinine}\n"
            f"Uric Acid mae = {mae_u_acid}\n"
            f"Uric Acid rmse = {rmse_u_acid}"
        )
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model_path", default='', help="The pth model path.")
    parser.add_argument("--scaler_path", default='', help="The scaler path.")
    parser.add_argument("--test_image_dir", default='',
        help="The test image dcm directory.") 
    parser.add_argument("--test_label_dir", default='',
        help="The test label csv directory.")
    args = parser.parse_args()
    
    main(args)