import pandas as pd
import numpy as np

def gen_dataset_1(n_rows):
    np.random.seed(42)  # For reproducibility

    ids = np.arange(1, 1 + n_rows)
    diagnosis = np.random.choice(['M', 'B'], n_rows)
    
    data = {
        'id': ids,
        'diagnosis': diagnosis,
        'radius_mean': np.random.uniform(10, 25, n_rows),
        'texture_mean': np.random.uniform(10, 30, n_rows),
        'perimeter_mean': np.random.uniform(50, 200, n_rows),
        'area_mean': np.random.uniform(300, 2000, n_rows),
        'smoothness_mean': np.random.uniform(0.05, 0.2, n_rows),
        'compactness_mean': np.random.uniform(0.02, 0.35, n_rows),
        'concavity_mean': np.random.uniform(0.02, 0.4, n_rows),
        'concave points_mean': np.random.uniform(0.01, 0.2, n_rows),
        'symmetry_mean': np.random.uniform(0.1, 0.3, n_rows),
        'fractal_dimension_mean': np.random.uniform(0.05, 0.1, n_rows),
        'radius_se': np.random.uniform(0.1, 3, n_rows),
        'texture_se': np.random.uniform(0.5, 4, n_rows),
        'perimeter_se': np.random.uniform(0.5, 10, n_rows),
        'area_se': np.random.uniform(1, 100, n_rows),
        'smoothness_se': np.random.uniform(0.001, 0.02, n_rows),
        'compactness_se': np.random.uniform(0.005, 0.05, n_rows),
        'concavity_se': np.random.uniform(0.005, 0.05, n_rows),
        'concave points_se': np.random.uniform(0.001, 0.02, n_rows),
        'symmetry_se': np.random.uniform(0.01, 0.03, n_rows),
        'fractal_dimension_se': np.random.uniform(0.001, 0.01, n_rows),
        'radius_worst': np.random.uniform(10, 30, n_rows),
        'texture_worst': np.random.uniform(10, 40, n_rows),
        'perimeter_worst': np.random.uniform(50, 250, n_rows),
        'area_worst': np.random.uniform(300, 2500, n_rows),
        'smoothness_worst': np.random.uniform(0.1, 0.25, n_rows),
        'compactness_worst': np.random.uniform(0.1, 0.6, n_rows),
        'concavity_worst': np.random.uniform(0.1, 0.7, n_rows),
        'concave points_worst': np.random.uniform(0.1, 0.3, n_rows),
        'symmetry_worst': np.random.uniform(0.2, 0.7, n_rows),
        'fractal_dimension_worst': np.random.uniform(0.05, 0.2, n_rows)
    }
    
    df = pd.DataFrame(data)
    
    df.to_csv(f"fake_breast_cancer_data_{n_rows}.csv", index=False)

def gen_dataset_2(n_rows):
    np.random.seed(42)  # For reproducibility

    data = {
        'Age': np.random.randint(18, 80, n_rows),
        'Sex': np.random.choice(['F', 'M'], n_rows),
        'BP': np.random.choice(['HIGH', 'LOW', 'NORMAL'], n_rows),
        'Cholesterol': np.random.choice(['HIGH', 'LOW', 'NORMAL'], n_rows),
        'Na_to_K': np.random.uniform(10, 30, n_rows),
        'Drug': np.random.choice(['drugA', 'drugB', 'drugC', 'drugX', 'drugY'], n_rows)
    }
    
    df = pd.DataFrame(data)
    
    df.to_csv(f"fake_drug_data_{n_rows}.csv", index=False)
# Example usage:
size = [1e3, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 8e3, 9e3, 1e4]
[gen_dataset_1(int(n)) for n in size]
