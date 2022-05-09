import os
import csv
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
#import torchvision
import sksurv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

#filenames
PLCO_MAIN_DATA_FILE = './PLCO_data/Prostate/pros_data_nov18_d070819.csv'
PLCO_TRATEMENT_DATA_FILE = './PLCO_data/Prostate/Treatments/pros_trt_data_nov18_d070819.csv'
KEY = 'plco_id'
PATIENTS_FEATURE = [
    #'ph_first_cancer',
    #'ph_first_cancer_age',
    'age',
    'agelevel',
    'dre_result0',
    'dre_result1',
    'dre_result2',
    'dre_result3',
    'dre_days0',
    'dre_days1',
    'dre_days2',
    'dre_days3',
    'psa_result0',
    'psa_result1',
    'psa_result2',
    'psa_result3',
    'psa_result4',
    'psa_result5',
    'psa_level0',
    'psa_level1',    
    'psa_level2',    
    'psa_level3',    
    #'psa_level4',    
    #'psa_level5',    
    'psa_days0',
    'psa_days1',
    'psa_days2',
    'psa_days3',
    #'psa_days4',
    #'psa_days5',
    'psa_prot',
    'biopplink0',
    'biopplink1',
    'biopplink2',
    'biopplink3',
    'biopplink4',
    'biopplink5',
    'pros_mra_stat0',
    'pros_mra_stat1',
    'pros_mra_stat2',
    'pros_mra_stat3',
    'pros_mra_stat4',
    'pros_mra_stat5',
    'candxdaysp',
    'pros_cancer',
    'intstatp_cat',
    'pros_is_first_dx',
    'reasfollp',
    'reasothp',
    #'reassurvp',
    'reassympp',
    'pros_stage',
    'pros_stage_m',
    'pros_stage_n',
    'pros_stage_t',
    'pros_clinstage',
    'pros_clinstage_m',
    'pros_clinstage_n',
    'pros_clinstage_t',
    #'pros_pathstage',
    #'pros_pathstage_m',
    #'pros_pathstage_n',
    #'pros_pathstage_t',
    'pros_stage_7e',
    'pros_clinstage_7e',
    #'pros_pathstage_7e',
    'pros_gleason',
    'pros_gleason_biop',
    #'pros_gleason_prost',
    'pros_gleason_source',
    'pros_grade',
    'pros_histtype',
    'dx_psa',
    'dx_psa_gap',
    #'is_dead',
    'is_dead_with_cod',
    #'dth_days',
    #'d_seer_death',
    #'f_seer_death',
    'race7',
    'hispanic_f',
    'educat',
    'marital',
    'occupat',
    'state',
    'cig_stat',
    'cig_stop',
    'cig_years',
    'cigpd_f',
    'pack_years',
    'cigar',
    'filtered_f',
    'pipe',
    'rsmoker_f',
    'smokea_f',
    'smoked_f',
    'ssmokea_f',
    'fh_cancer',
    'pros_fh',
    #'pros_fh_age',
    'pros_fh_cnt',
    'brothers',
    'sisters',
    'bmi_curc',
    'bmi_curr',
    'height_f',
    'weight_f',
    'bmi_20',
    'bmi_20c',
    'weight20_f',
    'bmi_50',
    'bmi_50c',
    'weight50_f',
    'asp',
    'asppd',
    'ibup',
    'ibuppd',
    'arthrit_f',
    'bronchit_f',
    'colon_comorbidity',
    'diabetes_f',
    'divertic_f',
    'emphys_f',
    'gallblad_f',
    'hearta_f',
    'hyperten_f',
    'liver_comorbidity',
    'osteopor_f',
    'polyps_f',
    'stroke_f',
    'enlpros_f',
    'infpros_f',
    'prosprob_f',
    'urinate_f',
    'urinatea',
    'surg_age',
    'surg_any',
    'surg_biopsy',
    'surg_prostatectomy',
    'surg_resection',
    'vasect_f',
    'vasecta',
    'psa_history',
    'rectal_history'
]
TREATMENT_FEATURE = [
    'trt_nump',	
    'trt_familyp',	
    'neoadjuvant',
    'trt_days',
]

#data
patients = pd.read_csv(PLCO_MAIN_DATA_FILE)
treatments = pd.read_csv(PLCO_TRATEMENT_DATA_FILE)

#patients -- plco_id <-> features

#treatments -- plco_id <-> one-hot vector of treatment type
max_received_treatments = treatments.groupby(KEY).size().max()
number_treated_patients = len(treatments.groupby(KEY))

patients_treatments = pd.merge(treatments, patients, how='inner', on=KEY)

class PatientTreatmentUnionDataset(Dataset):
    
    def __init__(self):
        # read the csv
        self._patient_df = pd.read_csv(PLCO_MAIN_DATA_FILE)
        self._treatment_df = pd.read_csv(PLCO_TRATEMENT_DATA_FILE)
        self._build_data_dict()
        self._plco_ids = list(self._patient_data.keys())
    
    def _build_data_dict(self):
        treatment_data = {}
        patient_data = {}
        patient_u_treatment_data = {}
        patient_delta_data = {}
        #only care of 4 treatment options :
        # 1="Prostatectomy" 
        # 2="Radiation treatment" 
        # 3="Hormone treatment" 
        # 4="Other ablative treatment"
        df = self._treatment_df[(self._treatment_df['trt_familyp'] == 1) \
            | (self._treatment_df['trt_familyp'] == 2) \
            | (self._treatment_df['trt_familyp'] == 3) \
            | (self._treatment_df['trt_familyp'] == 4) ]
        for idx, row in df.iterrows():
            plco_id = row[KEY]
            if not plco_id in treatment_data:
                treatment_data[plco_id] = [0, 0, 0, 0]
            treatment_data[plco_id][row['trt_familyp'] - 1] = 1
            if not plco_id in patient_data: 
                patient_data[plco_id] = self._patient_df[self._patient_df[KEY] == plco_id][PATIENTS_FEATURE].values.tolist()[0]
                patient_delta_data[plco_id] = (self._patient_df[self._patient_df[KEY] == plco_id]['is_dead'] == 1)
        for plco_id in patient_data:
            #append treatment to patient data
            patient_u_treatment_data[plco_id] = patient_data[plco_id] + treatment_data[plco_id]
        self._patient_data = patient_data
        self._treatment_data = treatment_data
        self._patient_u_treatment_data = patient_u_treatment_data
        self._patient_delta_data = patient_delta_data
    
    def __len__(self):
        """ return the number of samples (i.e. patients). """
        return len(self._plco_ids)
    
    def patient_u_treatment_len(self):
        return len(self._patient_u_treatment_data)
    
    def __getitem__(self, index):
        """ generates one sample of data. """
        plco_id = self._plco_ids[index]
        patient = torch.tensor(self._patient_data[plco_id], dtype=torch.float32)
        treatment = torch.tensor(self._treatment_data[plco_id], dtype=torch.float32)
        patient_u_treatment = torch.tensor(self._patient_u_treatment_data[plco_id], dtype=torch.float32)
        return plco_id, patient, treatment, patient_u_treatment
    
    def autoencode(self, model):
        encoded_patient_data = {}
        for plco_id in self._patient_data:
            patient = torch.tensor(self._patient_data[plco_id], dtype=torch.float32)
            patient = torch.nan_to_num(patient, nan=1e-5)
            encoded_patient_data[plco_id] = model(patient)
        self._encoded_patient_data = encoded_patient_data

class PatientDataset(Dataset):
    
    def __init__(self):
        # read the csv
        self._patient_df = pd.read_csv(PLCO_MAIN_DATA_FILE)
        self._treatment_df = pd.read_csv(PLCO_TRATEMENT_DATA_FILE)
        self._build_data_dict()
        self._plco_ids = list(self._patient_data.keys())
    
    def _build_data_dict(self):
        treatment_data = {}
        patient_data = {}
        #only care of 4 treatment options :
        # 1="Prostatectomy" 
        # 2="Radiation treatment" 
        # 3="Hormone treatment" 
        # 4="Other ablative treatment"
        df = self._treatment_df[(self._treatment_df['trt_familyp'] == 1) \
            | (self._treatment_df['trt_familyp'] == 2) \
            | (self._treatment_df['trt_familyp'] == 3) \
            | (self._treatment_df['trt_familyp'] == 4) ]
        for idx, row in df.iterrows():
            plco_id = row[KEY]
            if not plco_id in treatment_data:
                treatment_data[plco_id] = [0, 0, 0, 0]
            treatment_data[plco_id][row['trt_familyp'] - 1] = 1
            if not plco_id in patient_data: 
                patient_data[plco_id] = self._patient_df[self._patient_df[KEY] == plco_id][PATIENTS_FEATURE].values.tolist()[0]
        self._patient_data = patient_data
    
    def __len__(self):
        """ return the number of samples (i.e. patients). """
        return len(self._plco_ids)
    
    def __getitem__(self, index):
        """ generates one sample of data. """
        plco_id = self._plco_ids[index]
        patient = torch.tensor(self._patient_data[plco_id], dtype=torch.float32)
        #print (patient)
        patient = torch.nan_to_num(patient, nan=1e-5)
        #print (patient)
        return patient, plco_id 

# 4 layers Autodecode
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layer1 = nn.Linear(in_features=len(PATIENTS_FEATURE), out_features=104)
        self.hidden_layer2 = nn.Linear(in_features=104, out_features=52)
        self.hidden_layer3 = nn.Linear(in_features=52, out_features=24)
        self.hidden_layer4 = nn.Linear(in_features=24, out_features=12)
        self.output_layer = nn.Linear(in_features=12, out_features=12)

    def forward(self, x):
        h = torch.relu(self.hidden_layer1(x))
        h = torch.relu(self.hidden_layer2(h))
        h = torch.relu(self.hidden_layer3(h))
        h = torch.relu(self.hidden_layer4(h))
        z = torch.relu(self.output_layer(h))
        return z

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layer1 = nn.Linear(in_features=12, out_features=12)
        self.hidden_layer2 = nn.Linear(in_features=12, out_features=24)
        self.hidden_layer3 = nn.Linear(in_features=24, out_features=52)
        self.hidden_layer4 = nn.Linear(in_features=52, out_features=104)
        self.output_layer = nn.Linear(in_features=104, out_features=len(PATIENTS_FEATURE))

    def forward(self, z):
        h = torch.relu(self.hidden_layer1(z))
        h = torch.relu(self.hidden_layer2(h))
        h = torch.relu(self.hidden_layer3(h))
        h = torch.relu(self.hidden_layer4(h))
        x_hat = torch.relu(self.output_layer(h))
        return x_hat

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder_model = Autoencoder().to(device)
optimizer = torch.optim.Adam(autoencoder_model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

train_dataset = PatientDataset()

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

epochs = 100
for epoch in range(epochs):
    loss = 0
    for batch_x, _ in train_loader:
        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()
        
        # compute reconstructions
        batch_x_hat = autoencoder_model(batch_x)

        # compute training reconstruction loss
        train_loss = criterion(batch_x_hat, batch_x)
        
        # compute accumulated gradients
        train_loss.backward()
        
        # perform parameter update based on current gradients
        optimizer.step()
        
        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()
    
    # compute the epoch training loss
    loss = loss / len(train_loader)
    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))

X_a = PatientTreatmentUnionDataset()
X_a.autoencode(autoencoder_model)

#input
X_t = []#training_set
X_p = []#pool_set
d = []#survival status
T = []#time to event
max_iter = 20

#INCORRECT Survival Analysis
for iter in range(max_iter):
    estimator = CoxPHSurvivalAnalysis()
    #estimator.fit(X_a._encoded_patient_data.values(), X_a._patient_delta_data.values())
    #prediction = estimator.predict(X_a._encoded_patient_data.values())
