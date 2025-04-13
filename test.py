from exp import *
import time
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def evaluate_system():
    try:
        print("⚡ Starting memory-safe evaluation...")
        start_time = time.time()

        # Load essential models
        essential_models = {
            'le': joblib.load(os.path.join(MODELS_DIR, 'disease_encoder.pkl')),
            'symptom_cols': joblib.load(os.path.join(MODELS_DIR, 'symptom_features.pkl')),
            'clinical_cols': joblib.load(os.path.join(MODELS_DIR, 'clinical_features.pkl')),
            'env_cols': joblib.load(os.path.join(MODELS_DIR, 'env_features.pkl'))
        }

        # Load and clean datasets
        df_symp = load_data_safely(DATA_PATHS['symptoms'], 'symptoms')
        df_clin = load_data_safely(DATA_PATHS['clinical'], 'clinical')
        df_env = load_data_safely(DATA_PATHS['environmental'], 'environmental')

        # Filter for common diseases
        common_diseases = []
        for disease in essential_models['le'].classes_:
            if min((df_symp['Diseases'] == disease).sum(),
                   (df_clin['Diseases'] == disease).sum(),
                   (df_env['Diseases'] == disease).sum()) >= 5:
                common_diseases.append(disease)

        if not common_diseases:
            raise ValueError("No diseases with sufficient samples across all datasets")

        # Limit test size
        test_samples = min(500, len(common_diseases))
        test_diseases = np.random.choice(common_diseases, test_samples, replace=False)

        # Load all required models
        models = {
            **essential_models,
            'xgb': joblib.load(os.path.join(MODELS_DIR, 'xgb_model.pkl')),
            'lr_clin': joblib.load(os.path.join(MODELS_DIR, 'clinical_model.pkl')),
            'rf_env': joblib.load(os.path.join(MODELS_DIR, 'env_model.pkl')),
            'clin_scaler': joblib.load(os.path.join(MODELS_DIR, 'clinical_scaler.pkl'))
        }

        # Build test records
        test_data = []
        for disease in test_diseases:
            test_data.append({
                'symptoms': df_symp[df_symp['Diseases'] == disease].iloc[0],
                'clinical': df_clin[df_clin['Diseases'] == disease].iloc[0],
                'environmental': df_env[df_env['Diseases'] == disease].iloc[0],
                'disease': disease
            })

        # Process in batches
        batch_size = 50
        all_probs = []
        y_true = []

        def process_batch(batch):
            X_symp = pd.DataFrame([r['symptoms'][models['symptom_cols']] for r in batch])
            X_clin = pd.DataFrame([r['clinical'][models['clinical_cols']] for r in batch]).fillna(0)
            X_env = pd.DataFrame([r['environmental'][models['env_cols']] for r in batch]).ffill()
            X_clin_scaled = models['clin_scaler'].transform(X_clin)

            xgb_probs = models['xgb'].predict_proba(X_symp)
            clin_probs = models['lr_clin'].predict_proba(X_clin_scaled)
            env_probs = models['rf_env'].predict_proba(X_env)

            combined = 0.65 * xgb_probs + 0.18 * clin_probs + 0.17 * env_probs
            return combined

        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i + batch_size]
            combined_probs = process_batch(batch)
            all_probs.append(combined_probs)
            y_true.extend([r['disease'] for r in batch])

        # Combine predictions and compute metrics
        y_true_encoded = models['le'].transform(y_true)
        y_pred = np.argmax(np.vstack(all_probs), axis=1)

        results = {
            'accuracy': accuracy_score(y_true_encoded, y_pred),
            'precision': precision_score(y_true_encoded, y_pred, average='weighted'),
            'recall': recall_score(y_true_encoded, y_pred, average='weighted'),
            'f1_score': f1_score(y_true_encoded, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true_encoded, y_pred),
            'samples_tested': len(y_true),
            'time_sec': time.time() - start_time
        }

        # Print Results
        print(f"\n✅ Evaluation completed in {results['time_sec']:.1f}s")
        print(f"🔹 Accuracy       : {results['accuracy']:.4f}")
        print(f"🔹 Precision      : {results['precision']:.4f}")
        print(f"🔹 Recall         : {results['recall']:.4f}")
        print(f"🔹 F1 Score       : {results['f1_score']:.4f}")
        print(f"🔹 Confusion Matrix:\n{results['confusion_matrix']}")
        print(f"🔹 Samples Tested : {results['samples_tested']}")

        return results

    except Exception as e:
        print(f"\n❌ Evaluation failed: {str(e)}")
        return None

# Run evaluation
evaluate_system()
