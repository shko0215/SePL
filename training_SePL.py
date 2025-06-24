import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import numpy as np
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from PIL import Image
from dataset_SePL import FundusDataset
from model_SePL import SelfPropagative_MultiTaskLearner
import cv2
import logging
import datetime
from pathlib import Path
import random
import pickle
from tqdm import tqdm

CUDA_LAUNCH_BLOCKING=1

np.random.seed(7226)
torch.manual_seed(7226)
torch.cuda.manual_seed(7226)

class RandomOneOf:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        transform = random.choice(self.transforms)
        return transform(x)

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

    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('') # Specify the directory where you want to save the logs and checkpoints
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath(timestr)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
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
        RandomOneOf([                               # Apply one of these randomly
            lambda x: x,
            transforms.RandomHorizontalFlip(p=1.0), # Ensure each one has p=1.0
            transforms.RandomVerticalFlip(p=1.0),
            transforms.RandomRotation(degrees=(-15, 15)),
        ]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_set = FundusDataset(args.train_image_dir, args.train_label_dir, split='train', transform=train_transforms)
    scalers = train_set.fit_and_transform() # Scaling the data
    valid_set = FundusDataset(args.train_image_dir, args.train_label_dir, split='val', scalers=scalers, transform=train_transforms)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(dataset=valid_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Save scalers for future use
    # Specify the directory where you want to save the scalers
    with open(os.path.join("", timestr, "checkpoints", "scalers.pkl"), 'wb') as f:
        pickle.dump(scalers, f)

    # Initialize the model
    model = SelfPropagative_MultiTaskLearner()
    model = nn.DataParallel(model)
    model.cuda()

    # Define loss functions and optimizer
    criterion_classification = nn.BCEWithLogitsLoss()
    criterion_regression = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    epochs = args.epochs
    lowest_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        log_string('Epoch %d (%d/%s):' % (epoch+1, epoch+1, epochs))
        t_losses, gender_losses, age_losses, height_losses, weight_losses, bmi_losses, waist_losses, \
            sbp_losses, dbp_losses, tchol_losses, glu_losses, tg_losses, hba1c_losses, bun_losses, \
                creatinine_losses, u_acid_losses = [], [], [], [], [], [], [], [], [], [], [], [], [], \
                    [], [], []

        for data, targets, _, _ in tqdm(train_loader, leave=False):
            data = data.cuda()
            targets=targets.unsqueeze(1).cuda()
        
            optimizer.zero_grad()

            gender_features, age_features, height_features, weight_features, bmi_features, waist_features, sbp_features, \
            dbp_features, tchol_features, glu_features, triglycerides_features, hba1c_features, bun_features, creatinine_features, \
            u_acid_features = model(data)
                                                
            loss_gender = criterion_classification(gender_features.squeeze(0), targets[:,:,0])
            loss_age = criterion_regression(age_features.squeeze(0), targets[:,:,1])
            loss_height = criterion_regression(height_features.squeeze(0), targets[:,:,2])
            loss_weight = criterion_regression(weight_features.squeeze(0), targets[:,:,3])
            loss_bmi = criterion_regression(bmi_features.squeeze(0), targets[:,:,4])
            loss_waist = criterion_regression(waist_features.squeeze(0), targets[:,:,5])
            loss_sbp = criterion_regression(sbp_features.squeeze(0), targets[:,:,6])
            loss_dbp = criterion_regression(dbp_features.squeeze(0), targets[:,:,7])
            loss_tchol = criterion_regression(tchol_features.squeeze(0), targets[:,:,8])
            loss_glu = criterion_regression(glu_features.squeeze(0), targets[:,:,9])
            loss_tg = criterion_regression(triglycerides_features.squeeze(0), targets[:,:,10])
            loss_hba1c = criterion_regression(hba1c_features.squeeze(0), targets[:,:,11])
            loss_bun = criterion_regression(bun_features.squeeze(0), targets[:,:,12])
            loss_creatinine = criterion_regression(creatinine_features.squeeze(0), targets[:,:,13])
            loss_u_acid = criterion_regression(u_acid_features.squeeze(0), targets[:,:,14])
            
            gender_losses.append(loss_gender.item())
            age_losses.append(loss_age.item())
            height_losses.append(loss_height.item())
            weight_losses.append(loss_weight.item())
            bmi_losses.append(loss_bmi.item())
            waist_losses.append(loss_waist.item())
            sbp_losses.append(loss_sbp.item())
            dbp_losses.append(loss_dbp.item())
            tchol_losses.append(loss_tchol.item())
            glu_losses.append(loss_glu.item())
            tg_losses.append(loss_tg.item())
            hba1c_losses.append(loss_hba1c.item())
            bun_losses.append(loss_bun.item())
            creatinine_losses.append(loss_creatinine.item())
            u_acid_losses.append(loss_u_acid.item())
            
            loss = loss_gender+loss_age+loss_height+loss_weight+loss_bmi+loss_waist+loss_sbp+loss_dbp \
                +loss_tchol+loss_glu+loss_tg+loss_hba1c+loss_bun+loss_creatinine+loss_u_acid

            loss.backward()
            optimizer.step()

            t_losses.append(loss.item())
            
        log_string(
            f"Total mean train loss at epoch {epoch} is {sum(t_losses)/len(t_losses)}\n"
            f"Gender loss = {sum(gender_losses)/len(gender_losses)}\n"
            f"Age loss = {sum(age_losses)/len(age_losses)}\n"
            f"Height loss = {sum(height_losses)/len(height_losses)}\n"
            f"Weight loss = {sum(weight_losses)/len(weight_losses)}\n"
            f"BMI loss = {sum(bmi_losses)/len(bmi_losses)}\n"
            f"Waist loss = {sum(waist_losses)/len(waist_losses)}\n"
            f"SBP loss = {sum(sbp_losses)/len(sbp_losses)}\n"
            f"DBP loss = {sum(dbp_losses)/len(dbp_losses)}\n"
            f"T_chol loss = {sum(tchol_losses)/len(tchol_losses)}\n"
            f"Glucose loss = {sum(glu_losses)/len(glu_losses)}\n"
            f"Triglycerides loss = {sum(tg_losses)/len(tg_losses)}\n"
            f"HbA1c loss = {sum(hba1c_losses)/len(hba1c_losses)}\n"
            f"BUN loss = {sum(bun_losses)/len(bun_losses)}\n"
            f"Creatinine loss = {sum(creatinine_losses)/len(creatinine_losses)}\n"
            f"Uric Acid loss = {sum(u_acid_losses)/len(u_acid_losses)}"
        )
        
        # Validation phase
        model = model.eval()
        with torch.no_grad():
            t_losses, gender_losses, age_losses, height_losses, weight_losses, bmi_losses, waist_losses, \
            sbp_losses, dbp_losses, tchol_losses, glu_losses, tg_losses, hba1c_losses, bun_losses, \
                creatinine_losses, u_acid_losses = [], [], [], [], [], [], [], [], [], [], [], [], [], \
                    [], [], []

            for data, targets, _, _ in tqdm(valid_loader, leave=False):
                data = data.cuda()
                targets=targets.unsqueeze(1).cuda()

                gender_features, age_features, height_features, weight_features, bmi_features, waist_features, sbp_features, \
                dbp_features, tchol_features, glu_features, triglycerides_features, hba1c_features, bun_features, creatinine_features, \
                u_acid_features = model(data)
                
                loss_gender = criterion_classification(gender_features.squeeze(0), targets[:,:,0])
                loss_age = criterion_regression(age_features.squeeze(0), targets[:,:,1])
                loss_height = criterion_regression(height_features.squeeze(0), targets[:,:,2])
                loss_weight = criterion_regression(weight_features.squeeze(0), targets[:,:,3])
                loss_bmi = criterion_regression(bmi_features.squeeze(0), targets[:,:,4])
                loss_waist = criterion_regression(waist_features.squeeze(0), targets[:,:,5])
                loss_sbp = criterion_regression(sbp_features.squeeze(0), targets[:,:,6])
                loss_dbp = criterion_regression(dbp_features.squeeze(0), targets[:,:,7])
                loss_tchol = criterion_regression(tchol_features.squeeze(0), targets[:,:,8])
                loss_glu = criterion_regression(glu_features.squeeze(0), targets[:,:,9])
                loss_tg = criterion_regression(triglycerides_features.squeeze(0), targets[:,:,10])
                loss_hba1c = criterion_regression(hba1c_features.squeeze(0), targets[:,:,11])
                loss_bun = criterion_regression(bun_features.squeeze(0), targets[:,:,12])
                loss_creatinine = criterion_regression(creatinine_features.squeeze(0), targets[:,:,13])
                loss_u_acid = criterion_regression(u_acid_features.squeeze(0), targets[:,:,14])

                gender_losses.append(loss_gender.item())
                age_losses.append(loss_age.item())
                height_losses.append(loss_height.item())
                weight_losses.append(loss_weight.item())
                bmi_losses.append(loss_bmi.item())
                waist_losses.append(loss_waist.item())
                sbp_losses.append(loss_sbp.item())
                dbp_losses.append(loss_dbp.item())
                tchol_losses.append(loss_tchol.item())
                glu_losses.append(loss_glu.item())
                tg_losses.append(loss_tg.item())
                hba1c_losses.append(loss_hba1c.item())
                bun_losses.append(loss_bun.item())
                creatinine_losses.append(loss_creatinine.item())
                u_acid_losses.append(loss_u_acid.item())
                
                loss = loss_gender+loss_age+loss_height+loss_weight+loss_bmi+loss_waist+loss_sbp+loss_dbp \
                    +loss_tchol+loss_glu+loss_tg+loss_hba1c+loss_bun+loss_creatinine+loss_u_acid

                t_losses.append(loss.item())
            
            log_string(
                f"Total mean valid loss at epoch {epoch} is {sum(t_losses)/len(t_losses)}\n"
                f"Gender loss = {sum(gender_losses)/len(gender_losses)}\n"
                f"Age loss = {sum(age_losses)/len(age_losses)}\n"
                f"Height loss = {sum(height_losses)/len(height_losses)}\n"
                f"Weight loss = {sum(weight_losses)/len(weight_losses)}\n"
                f"BMI loss = {sum(bmi_losses)/len(bmi_losses)}\n"
                f"Waist loss = {sum(waist_losses)/len(waist_losses)}\n"
                f"SBP loss = {sum(sbp_losses)/len(sbp_losses)}\n"
                f"DBP loss = {sum(dbp_losses)/len(dbp_losses)}\n"
                f"T_chol loss = {sum(tchol_losses)/len(tchol_losses)}\n"
                f"Glucose loss = {sum(glu_losses)/len(glu_losses)}\n"
                f"Triglycerides loss = {sum(tg_losses)/len(tg_losses)}\n"
                f"HbA1c loss = {sum(hba1c_losses)/len(hba1c_losses)}\n"
                f"BUN loss = {sum(bun_losses)/len(bun_losses)}\n"
                f"Creatinine loss = {sum(creatinine_losses)/len(creatinine_losses)}\n"
                f"Uric Acid loss = {sum(u_acid_losses)/len(u_acid_losses)}"
            )

            # Save the best model based on validation loss
            # We do not provide our early stopping implementation here, but you can implement it based on our paper and your needs.
            if ((sum(t_losses)/len(t_losses)) < lowest_loss):
                lowest_loss = sum(t_losses)/len(t_losses)
                log_string('Epoch: %d, best model save, validation loss: %.5f' %(epoch+1, sum(t_losses)/len(t_losses)))
                # Specify the directory where you want to save the model
                torch.save(model.module.state_dict(), os.path.join("", timestr, "checkpoints", "best_model.pth"))
            
            torch.cuda.empty_cache()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--model", default='convnext_t', help="The backbone model name.")
    parser.add_argument("--train_image_dir", default='',
        help="The train/valid image dcm directory.")
    parser.add_argument("--train_label_dir", default='',
        help="The train/valid label csv directory.")
    # whether or not to save model
    parser.add_argument("-save", default=True, action="store_true")
    args = parser.parse_args()
    
    main(args)
