import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from joblib import Parallel, delayed

current_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(current_dir, 'data')
output_dir = os.path.join(current_dir, 'all_genes_all_model')
os.makedirs(output_dir, exist_ok=True)

def training_model(filepath, filename, model_name, model):
	print(f"Processing {filename} with {model_name}")
	
	df = pd.read_csv(filepath, sep='\t')
	print(f'data shape: {df.shape}')
	X = df.drop(columns=['genome_id', 'resistant_phenotype'])
	y = df['resistant_phenotype'].map({'R': 1, 'S': 0}).astype(int)

	cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

	results = []
	for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
		X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
		y_train, y_test = y[train_idx], y[test_idx]

		model.fit(X_train, y_train)

		y_pred = model.predict(X_test)
		y_pred_proba = model.predict_proba(X_test)[:, 1]
		
		metrics = {
			'Drug': filename,
			'Model': model_name,
			'Selector': 'All_gene',
			'Fold': fold_idx +1, 
			'AUC': roc_auc_score(y_test, y_pred_proba),
			'Accuracy': accuracy_score(y_test, y_pred),
			'Precision': precision_score(y_test, y_pred, zero_division=0),
			'Recall': recall_score(y_test, y_pred, zero_division=0),
			'F1': f1_score(y_test, y_pred),
			'MCC': matthews_corrcoef(y_test, y_pred)
		}
		results.append(metrics)
	return results

file_list = [(os.path.join(input_dir, filename), filename) for filename in os.listdir(input_dir)]

model_list = {
	'SVC': SVC(kernel='linear', C=0.1, class_weight='balanced', gamma=0.001, probability=True, random_state=42),
	'LogRed': LogisticRegression(max_iter=10000, class_weight='balanced', random_state=42),
	'RF': RandomForestClassifier(class_weight='balanced', random_state=42),
	'HistGrad': HistGradientBoostingClassifier(class_weight='balanced', random_state=42)
}
all_results = Parallel(n_jobs=20)(
	delayed(training_model)(filepath, filename, model_name, model)
	for filepath, filename in file_list
	for model_name, model in model_list.items()
)

all_results = [item for sublist in all_results for item in sublist]
results_df = pd.DataFrame(all_results)

mean_df = results_df.groupby(['Drug', 'Model', 'Selector']).agg({
	'AUC': 'mean',
	'Accuracy': 'mean',
	'Precision': 'mean',
	'Recall': 'mean',
	'F1': 'mean',
	'MCC': 'mean'
}).reset_index()

std_df = results_df.groupby(['Drug', 'Model', 'Selector']).agg({
	'AUC': 'std',
	'Accuracy': 'std',
	'Precision': 'std',
	'Recall': 'std',
	'F1': 'std',
	'MCC': 'std'
}).reset_index()

results_df.to_csv(os.path.join(output_dir, "all.genes.all.models.results.txt"), sep='\t', float_format='%.4f', index=False)
mean_df.to_csv(os.path.join(output_dir, "all.genes.all.models.mean.txt"), sep='\t', float_format='%.4f', index=False)
std_df.to_csv(os.path.join(output_dir, "all.genes.all.models.std.txt"), sep='\t', float_format='%.4f', index=False)
print(f"Results: '\n' {results_df}")
print(f"mean: '\n' {mean_df}")
print(f"std: '\n' {std_df}")
print("Process Complete")
