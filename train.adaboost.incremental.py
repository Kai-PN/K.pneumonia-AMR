import os
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from joblib import Parallel, delayed

current_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(current_dir, 'data')
output_dir = os.path.join(current_dir, 'adaboost_incremental')
os.makedirs(output_dir, exist_ok=True)

def training_model(filepath, filename, model_name, model, selector_name, selector):
	print(f"Processing {filename} with {model_name} and {selector_name}")
	df = pd.read_csv(filepath, sep='\t')
	X = df.drop(columns=['genome_id', 'resistant_phenotype'])
	y = df['resistant_phenotype'].map({'R': 1, 'S': 0}).astype(int)

	cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
	results = []
	fold_best_metrics = []

	for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X,y)):
		X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
		y_train, y_test = y[train_idx], y[test_idx]

		selector.fit(X_train, y_train)
		sorted_features = np.argsort(selector.estimator_.feature_importances_)[::-1]
		
		features = X_train.columns[selector.get_support()].tolist()
		num_features = len(features)
	
		best_auc = 0
		best_metrics=None
		best_features = None
		
		for current_num_features in range(1, num_features +1):
			features_indices = sorted_features[:current_num_features]
			selected_features = X_train.columns[features_indices].tolist()
			
			X_train_selected = X_train.iloc[:, features_indices]
			X_test_selected = X_test.iloc[:, features_indices]
		
			model.fit(X_train_selected, y_train)	
			y_pred = model.predict(X_test_selected)
			y_pred_proba = model.predict_proba(X_test_selected)[:, 1] if hasattr(model, 'predict_proba') else None
	
			metrics = {
				'Drug': filename, 
				'Model': model_name,
				'Selector': selector_name,
				'Fold': fold_idx + 1,
				'AUC': roc_auc_score(y_test, y_pred_proba),
				'Accuracy': accuracy_score(y_test, y_pred),
				'Precision': precision_score(y_test, y_pred, zero_division=0),
				'Recall': recall_score(y_test, y_pred, zero_division=0),
				'F1': f1_score(y_test, y_pred),
				'MCC': matthews_corrcoef(y_test, y_pred),
				'Num_selected_features': current_num_features,
				'Selected_features': selected_features,
			}
			
			results.append(metrics)
			print(f"{filename} - Fold {fold_idx+1} - Num_features: {current_num_features} - AUC:{roc_auc_score(y_test, y_pred_proba):4f}")	
			
			if metrics['AUC'] > best_auc:
				best_auc = metrics['AUC']
				best_metrics = metrics
				best_features = selected_features
			
		fold_best_metrics.append(best_metrics)

		print(f"Best Metrics: {filename} - Fold {fold_idx+1} - {best_metrics['AUC']:4f} - Num_features: {len(best_features)}")
	
	return results, fold_best_metrics


file_list = [(os.path.join(input_dir, filename), filename) for filename in os.listdir(input_dir)]

model_list = {'HistGrad': HistGradientBoostingClassifier(class_weight='balanced', random_state=42)}

selectors = {'AdaBoost': SelectFromModel(AdaBoostClassifier(estimator=None,algorithm='SAMME', random_state=42))}

all_results = Parallel(n_jobs=2)(delayed(training_model)(filepath, filename, model_name, model, selector_name, selector)
	for filepath, filename in file_list
	for model_name, model in model_list.items()
	for selector_name, selector in selectors.items()
)
detailed_results = [item for sublist in all_results for item in sublist[0]]
all_fold_best_metrics = [item for sublist in all_results for item in sublist[1]]

detailed_df = pd.DataFrame(detailed_results)
detailed_df.to_csv(os.path.join(output_dir, "adaboost.incremental.detailed.txt"), sep='\t', float_format='%.4f', index=False)

fold_best_metrics_df = pd.DataFrame(all_fold_best_metrics)

mean_df = fold_best_metrics_df.groupby(['Drug', 'Model', 'Selector']).agg({
	'AUC': 'mean',
	'Accuracy': 'mean',
	'Precision': 'mean',
	'Recall': 'mean',
	'F1': 'mean',
	'MCC': 'mean',
	'Num_selected_features': 'mean'
}).reset_index()

std_df = fold_best_metrics_df.groupby(['Drug', 'Model', 'Selector']).agg({
	'AUC': 'std',
	'Accuracy': 'std',
	'Precision': 'std',
	'Recall': 'std',
	'F1': 'std',
	'MCC': 'std',
	'Num_selected_features': 'std'
}).reset_index()

best_fold_df = fold_best_metrics_df.loc[
	fold_best_metrics_df.groupby(['Drug', 'Model', 'Selector'])['AUC'].idxmax()
][['Drug', 'Model', 'Selector', 'Fold', 'AUC', 'Selected_features', 'Num_selected_features']].reset_index(drop=True)

fold_best_metrics_df.to_csv(os.path.join(output_dir, "adaboost.incremental.results.txt"), sep='\t', float_format='%.4f', index=False)
mean_df.to_csv(os.path.join(output_dir, "adaboost.incremental.mean.txt"), sep='\t', float_format='%.4f', index=False)
std_df.to_csv(os.path.join(output_dir, "adaboost.incremental.std.txt"), sep='\t', float_format='%.4f', index=False)
best_fold_df.to_csv(os.path.join(output_dir, "adaboost.incremental.best.folds.list.txt"), sep='\t', float_format='%.4f', index=False)
print(f"mean: \n{mean_df}")
print(f"std: \n{std_df}")
print(f"best_fold: \n{best_fold_df}")
print(detailed_df.head()) 
print("Process complete")
