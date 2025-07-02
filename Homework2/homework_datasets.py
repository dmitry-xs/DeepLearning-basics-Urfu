import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder


class CSVDataset(Dataset):
    def __init__(self, file_path, target_column=None,
                 numeric_cols=None, categorical_cols=None,
                 binary_cols=None, normalize_numeric=True,
                 test_size=0.2, random_state=42, mode='train'):

        self.data = pd.read_csv(file_path)
        self.target_column = target_column
        self.numeric_cols = numeric_cols if numeric_cols else []
        self.categorical_cols = categorical_cols if categorical_cols else []
        self.binary_cols = binary_cols if binary_cols else []
        self.normalize_numeric = normalize_numeric
        self.mode = mode

        self._preprocess_data()

        if test_size > 0 and target_column is not None:
            self._train_test_split(test_size, random_state)

    def _preprocess_data(self):
        """Предобработка данных"""
        # Обработка числовых колонок
        if self.numeric_cols and self.normalize_numeric:
            self.scaler = StandardScaler()
            self.data[self.numeric_cols] = self.scaler.fit_transform(self.data[self.numeric_cols])

        # Обработка категориальных колонок
        self.encoders = {}
        if self.categorical_cols:
            for col in self.categorical_cols:
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col])
                self.encoders[col] = le

        # Обработка бинарных колонок
        if self.binary_cols:
            for col in self.binary_cols:
                self.data[col] = self.data[col].astype(int)

        # Создаем тензоры с явным указанием типов
        feature_cols = self.numeric_cols + self.categorical_cols + self.binary_cols
        self.features = torch.tensor(self.data[feature_cols].values, dtype=torch.float32)

        if self.target_column:
            self.targets = torch.tensor(self.data[self.target_column].values, dtype=torch.float32).view(-1, 1)

    def _train_test_split(self, test_size, random_state):
        """Разделение данных с сохранением тензорного формата"""
        indices = torch.randperm(len(self.features)).tolist()
        split = int(test_size * len(indices))

        if self.mode == 'train':
            self.features = self.features[indices[split:]]
            self.targets = self.targets[indices[split:]] if self.target_column else None
        else:
            self.features = self.features[indices[:split]]
            self.targets = self.targets[indices[:split]] if self.target_column else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.target_column:
            return self.features[idx], self.targets[idx]
        return self.features[idx]

    def get_feature_dim(self):
        return self.features.shape[1]

    def get_num_classes(self):
        if self.target_column:
            return len(torch.unique(self.targets))
        return 0
