from pathlib import Path
import zipfile
import collections

import pandas as pd
from scipy.io import arff
import requests

# function that reads an arff file and returns a pandas dataframe
def read_arff(filepath):
    data, meta = arff.loadarff(filepath)
    df = pd.DataFrame(data)
    df.columns = meta.names()
    return df

# function that removes trailing whitespace from a file
def remove_trailing_space(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [line.replace(' \n', '\n') for line in lines]
    if lines[-1].endswith(' '):
        # Remove trailing space from last line
        lines[-1] = lines[-1].rstrip()
    filepath = filepath.parent / (filepath.stem + '_no_trailing_space' + filepath.suffix)
    with open(filepath, 'w') as f:
        f.writelines(lines)
    return filepath

# Define DataSpec named tuple
DataSpec = collections.namedtuple(
    "DataSpec",
    "url, read_function, read_params, columns, preprocess, file_in_zip, label, exclude",
)

UCI_DATASETS = {
    'boston_housing': DataSpec(
        url='https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',
        read_function=pd.read_csv,
        read_params={'delim_whitespace': True, 'header': None, 'skipinitialspace': True},
        columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'],
        preprocess=None,
        file_in_zip=None,
        label='MEDV',
        exclude=None
    ),
    'concrete_strength': DataSpec(
        url='https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls',
        read_function=pd.read_excel,
        read_params={'header': 0},
        columns=['cement', 'blast_furnace_slag', 'fly_ash', 'water', 'superplasticizer', 'coarse_aggregate', 'fine_aggregate', 'age', 'concrete_compressive_strength'],
        preprocess=None,
        file_in_zip=None,
        label="concrete_compressive_strength",
        exclude=None
    ),
    'energy_efficiency': DataSpec(
        url='https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx',
        read_function=pd.read_excel,
        read_params={'header': 0},
        columns=None,
        preprocess=None,
        file_in_zip=None,
        label='Y1',
        exclude=['Y2'],
    ),
    'kin8nm': DataSpec(
        url='https://www.openml.org/data/download/3626/dataset_2175_kin8nm.arff',
        read_function=read_arff,
        read_params={},
        columns=None,
        preprocess=None,
        file_in_zip=None,
        label='y',
        exclude=None
    ),
    'naval_propulsion': DataSpec(
        url='https://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip',
        read_function=pd.read_csv,
        read_params={'header': None, 'delim_whitespace': True, 'skipinitialspace': True},
        columns=['lp', 'v', 'GTT', 'GTn', 'GGn', 'Ts', 'Tp', 'T48', 'T1', 'T2', 'P48', 'P1', 'P2', 'Pexh', 'TIC', 'mf', 'GTCD', 'GTTC'],
        preprocess=None,
        file_in_zip='UCI CBM Dataset/data.txt',
        label='GTTC',
        exclude=['GTCD']
    ),
    'power_plant': DataSpec(
        url='https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip',
        read_function=pd.read_excel,
        read_params={'header': 0},
        columns=None,
        preprocess=None,
        file_in_zip='CCPP/Folds5x2_pp.xlsx',
        label='PE',
        exclude=None
    ),
    'protein_structure': DataSpec(
        url='https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv',
        read_function=pd.read_csv,
        read_params={'header': 0},
        columns=None,
        preprocess=None,
        file_in_zip=None,
        label='RMSD',
        exclude=None
    ),
    'wine_red': DataSpec(
        url='https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
        read_function=pd.read_csv,
        read_params={'header': 0, 'sep': ';'},
        columns=None,
        preprocess=None,
        file_in_zip=None,
        label='quality',
        exclude=None
    ),
    'yacht_hydrodynamics': DataSpec(
        url='https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data',
        read_function=pd.read_csv,
        read_params={'header': None, 'delim_whitespace': True, 'skipinitialspace': True},
        columns=['longitudinal_pos', 'presmatic_coef', 'length_disp', 'beam-draught_ratio', 'length-beam_ratio', 'froude_num', 'resid_resist'],
        preprocess=remove_trailing_space,
        file_in_zip=None,
        label='resid_resist',
        exclude=None
    ),
}

def process_dataset(name: str, spec: DataSpec, data_dir: str):
    dataset_folder = Path(data_dir) / name
    raw_folder = dataset_folder / 'raw'
    raw_folder.mkdir(exist_ok=True, parents=True)
    dataset_path = raw_folder / spec.url.split('/')[-1]

    # Download dataset if it doesn't exist
    if not dataset_path.exists():
        response = requests.get(spec.url)
        response.raise_for_status()
        with open(dataset_path, 'wb') as f:
            f.write(response.content)

    # Extract file from zip if necessary
    if spec.file_in_zip:
        with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
            dataset_path = zip_ref.extract(spec.file_in_zip, dataset_path.parent)
    
    # Read in dataset
    if spec.preprocess:
        dataset_path = spec.preprocess(dataset_path)
    df = spec.read_function(dataset_path, **spec.read_params)
    if spec.columns:
        df.columns = spec.columns

    # Save dataset
    data_filepath = dataset_folder / f"{name}.csv"
    df.to_csv(data_filepath, index=False, header=True)



if __name__ == '__main__':
    data_dir = Path(__file__).parent
    for name, config in UCI_DATASETS.items():
        process_dataset(name, config, data_dir)
