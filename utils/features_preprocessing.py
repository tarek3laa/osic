from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from drive.MyDrive.graduation_project.utils.utils import *


class AppendMissingWeeks(BaseEstimator, TransformerMixin):

    
    def get_missing_weeks(self,patient_data):
      
        n_missing_weeks = 10 - len(patient_data['Weeks'])
        min_week = np.min(patient_data['Weeks'])
        max_week = np.max(patient_data['Weeks'])
        random_weeks = np.random.random_integers(min_week, max_week, 10)
        missing_weeks = list(set(random_weeks) - set(patient_data['Weeks']))
        return missing_weeks[:n_missing_weeks]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        patients = get_patients(X)
        df = pd.DataFrame(columns=patients[0].columns)
        for patient in patients:
            patient.reset_index(inplace=True, drop=True)
            missing_weeks = self.get_missing_weeks(patient)

            for week in missing_weeks:
                new_row = patient.iloc[0].copy()
                new_row['FVC'] = -1
                new_row['Percent'] = -1
                new_row['Weeks'] = week
                patient.loc[len(patient)] = new_row
            df = df.append(patient.sort_values('Weeks'), ignore_index=True)

        return df


class RemoveOutliers(BaseEstimator, TransformerMixin):
    def __init__(self,ids):
      self.ids = ids
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        mask = X['Patient'].isin(self.ids)
        df = X[~mask]
        return df


class PatientCoefficient(BaseEstimator, TransformerMixin):
    def __init__(self, degree):
        self.degree = degree

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        patients = get_patients(X)
        flag = True
        for patient in patients:
            w = fitted_curve(patient, self.degree)
            p = patient[patient['Weeks'] == patient['Weeks'].min()]  # get intial week
            for i in range(len(w)):
                p[f'W{i}'] = w[i]
            if flag:
                df = pd.DataFrame(p)
                flag = False
            else:
                df.loc[len(df)] = p.values[0]
        return df


class CategoricalTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()
        df['Sex'] = df['Sex'].map({'Male': 0, 'Female': 1})
        df['Ex-smoker'] = (df['SmokingStatus'] == 'Ex-smoker').astype(int)
        df['Never-smoked'] = (df['SmokingStatus'] == 'Never smoked').astype(int)
        df['Currently-smokes'] = (df['SmokingStatus'] == 'Currently smokes').astype(int)
        df.drop(['SmokingStatus'], axis=1, inplace=True)
        df['decade'] = df['Age']//10
        df.drop(['Age'], axis=1, inplace=True)
        return df


class HandlingMissingData(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        patients = get_patients(X)
        print(len(patients))
        df = pd.DataFrame(columns=patients[0].columns)
        for patient in patients:
                patient.reset_index(inplace=True, drop=True)
                idx_missing = patient[patient['FVC'] == -1].index.values
                for i in idx_missing: 
                    idx = patient[patient['FVC'] != -1].index.values
                    prev_position = idx[idx < i][-1]
                    follow_position = idx[idx > i][0]

                    prev_fvc = int(patient.loc[prev_position, 'FVC'])
                    follow_fvc = int(patient.loc[follow_position, 'FVC'])

                    prev_week = int(patient.loc[prev_position, 'Weeks'])
                    curr_week = int(patient.loc[i, 'Weeks'])
                    follow_week = int(patient.loc[follow_position, 'Weeks'])

                    if prev_fvc < follow_fvc:
                        all_fvc = list(range(prev_fvc, follow_fvc))
                        fvc = np.quantile(all_fvc, (curr_week - prev_week) / ((follow_week - prev_week) - 1))
                    elif prev_fvc == follow_fvc:
                        fvc = prev_fvc  
                    else:
                        all_fvc = list(range(follow_fvc, prev_fvc))
                        fvc = np.quantile(all_fvc, 1 - ((curr_week - prev_week) / ((follow_week - prev_week) - 1)))
                        
                    patient.loc[i, ['FVC']] = [int(fvc)]
                patient.loc[1:,'Percent'] = patient.loc[0,'Percent']
                df = df.append(patient)
        return df


class Scaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()
        st = StandardScaler()
        x = st.fit_transform(X[self.columns])
        df[self.columns] = x
        return df
