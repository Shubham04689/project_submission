"""
Task 3: Real vs Fake Classifier (Synthetic Data Generation)

Binary classification system to distinguish between real and synthetic data.
Includes data generation, model training, and comprehensive evaluation.

Features:
- Real data generation using sklearn datasets
- Synthetic fake data generation
- Multiple classifier options
- Performance evaluation with visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class DataGenerator:
    """Generate real and fake datasets for classification"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_real_data_2d(self, n_samples=1000, dataset_type='blobs'):
        """Generate 2D real data for visualization"""
        if dataset_type == 'blobs':
            X, _ = make_blobs(
                n_samples=n_samples,
                centers=3,
                n_features=2,
                cluster_std=1.5,
                random_state=self.random_state
            )
        elif dataset_type == 'moons':
            X, _ = make_moons(
                n_samples=n_samples,
                noise=0.1,
                random_state=self.random_state
            )
        else:  # multivariate normal
            mean = [2, 3]
            cov = [[1, 0.5], [0.5, 2]]
            X = np.random.multivariate_normal(mean, cov, n_samples)
        
        return X
    
    def generate_real_data_high_dim(self, n_samples=1000, n_features=128):
        """Generate high-dimensional real data"""
        # Use multivariate normal with structured covariance
        mean = np.zeros(n_features)
        
        # Create structured covariance matrix
        cov = np.eye(n_features)
        for i in range(n_features-1):
            cov[i, i+1] = 0.3  # Add some correlation
            cov[i+1, i] = 0.3
        
        X = np.random.multivariate_normal(mean, cov, n_samples)
        return X
    
    def generate_fake_data_2d(self, n_samples=1000, distribution='uniform'):
        """Generate 2D fake data with different distribution"""
        if distribution == 'uniform':
            X = np.random.uniform(-5, 8, (n_samples, 2))
        else:  # different gaussian
            mean = [-1, -2]
            cov = [[3, -0.8], [-0.8, 0.5]]
            X = np.random.multivariate_normal(mean, cov, n_samples)
        
        return X
    
    def generate_fake_data_high_dim(self, n_samples=1000, n_features=128):
        """Generate high-dimensional fake data"""
        # Use uniform distribution or different gaussian parameters
        X = np.random.uniform(-2, 2, (n_samples, n_features))
        return X
    
    def create_labeled_dataset(self, n_samples=1000, use_2d=True, real_type='blobs'):
        """Create complete labeled dataset with real and fake data"""
        if use_2d:
            X_real = self.generate_real_data_2d(n_samples//2, real_type)
            X_fake = self.generate_fake_data_2d(n_samples//2, 'uniform')
        else:
            X_real = self.generate_real_data_high_dim(n_samples//2, 128)
            X_fake = self.generate_fake_data_high_dim(n_samples//2, 128)
        
        # Combine data and create labels
        X = np.vstack([X_real, X_fake])
        y = np.hstack([np.ones(len(X_real)), np.zeros(len(X_fake))])  # Real=1, Fake=0
        
        # Shuffle the dataset
        indices = np.random.permutation(len(X))
        X, y = X[indices], y[indices]
        
        return X, y, X_real, X_fake


class RealFakeClassifier:
    """Binary classifier for real vs fake data detection"""
    
    def __init__(self):
        self.models = {
            'logistic': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'svm': SVC(random_state=42, probability=True)
        }
        self.scaler = StandardScaler()
        self.trained_models = {}
        self.results = {}
    
    def train_models(self, X_train, y_train):
        """Train all classifier models"""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        print("Training classifiers...")
        for name, model in self.models.items():
            print(f"  Training {name}...")
            model.fit(X_train_scaled, y_train)
            self.trained_models[name] = model
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        X_test_scaled = self.scaler.transform(X_test)
        
        for name, model in self.trained_models.items():
            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            cm = confusion_matrix(y_test, y_pred)
            
            self.results[name] = {
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'confusion_matrix': cm,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'classification_report': classification_report(y_test, y_pred)
            }
    
    def print_results(self):
        """Print evaluation results for all models"""
        print("\n=== Model Performance Results ===")
        
        for name, results in self.results.items():
            print(f"\n{name.upper()} CLASSIFIER:")
            print(f"Accuracy: {results['accuracy']:.4f}")
            print(f"ROC AUC: {results['roc_auc']:.4f}")
            print("\nConfusion Matrix:")
            print(results['confusion_matrix'])
            print("\nClassification Report:")
            print(results['classification_report'])
    
    def plot_results(self, X_test, y_test, X_real, X_fake, use_2d=True):
        """Create visualizations of results"""
        if use_2d:
            self._plot_2d_results(X_test, y_test, X_real, X_fake)
        
        self._plot_confusion_matrices()
        self._plot_roc_curves(X_test, y_test)
    
    def _plot_2d_results(self, X_test, y_test, X_real, X_fake):
        """Plot 2D data separation and decision boundaries"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original data distribution
        axes[0, 0].scatter(X_real[:, 0], X_real[:, 1], c='blue', alpha=0.6, label='Real Data')
        axes[0, 0].scatter(X_fake[:, 0], X_fake[:, 1], c='red', alpha=0.6, label='Fake Data')
        axes[0, 0].set_title('Original Data Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Test predictions for best model
        best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['roc_auc'])
        y_pred = self.results[best_model_name]['y_pred']
        
        # Correct vs incorrect predictions
        correct_mask = (y_test == y_pred)
        axes[0, 1].scatter(X_test[correct_mask, 0], X_test[correct_mask, 1], 
                          c='green', alpha=0.6, label='Correct Predictions')
        axes[0, 1].scatter(X_test[~correct_mask, 0], X_test[~correct_mask, 1], 
                          c='red', alpha=0.6, label='Incorrect Predictions')
        axes[0, 1].set_title(f'Predictions - {best_model_name.title()}')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Decision boundary for logistic regression
        if 'logistic' in self.trained_models:
            self._plot_decision_boundary(axes[1, 0], X_test, 'logistic')
        
        # Feature distributions
        axes[1, 1].hist(X_real[:, 0], alpha=0.5, label='Real - Feature 1', bins=30)
        axes[1, 1].hist(X_fake[:, 0], alpha=0.5, label='Fake - Feature 1', bins=30)
        axes[1, 1].set_title('Feature 1 Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_decision_boundary(self, ax, X, model_name):
        """Plot decision boundary for 2D data"""
        model = self.trained_models[model_name]
        
        # Create mesh
        h = 0.1
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Predict on mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        mesh_scaled = self.scaler.transform(mesh_points)
        Z = model.predict_proba(mesh_scaled)[:, 1]
        Z = Z.reshape(xx.shape)
        
        # Plot
        ax.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdYlBu')
        ax.scatter(X[:, 0], X[:, 1], c=['red' if y == 0 else 'blue' for y in self.results[model_name]['y_pred']], 
                  alpha=0.8)
        ax.set_title(f'Decision Boundary - {model_name.title()}')
        ax.grid(True, alpha=0.3)
    
    def _plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (name, results) in enumerate(self.results.items()):
            sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', 
                       cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{name.title()} - Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_roc_curves(self, X_test, y_test):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        X_test_scaled = self.scaler.transform(X_test)
        
        for name, model in self.trained_models.items():
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = self.results[name]['roc_auc']
            
            plt.plot(fpr, tpr, label=f'{name.title()} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


def main():
    """Main execution function"""
    print("=== Real vs Fake Data Classifier ===\n")
    
    # Configuration
    USE_2D = True  # Set to False for 128D data
    N_SAMPLES = 2000
    REAL_DATA_TYPE = 'blobs'  # 'blobs', 'moons', or 'normal'
    
    print(f"Configuration:")
    print(f"- Data dimension: {'2D' if USE_2D else '128D'}")
    print(f"- Sample size: {N_SAMPLES}")
    print(f"- Real data type: {REAL_DATA_TYPE}")
    
    # Generate data
    print("\nGenerating datasets...")
    generator = DataGenerator()
    X, y, X_real, X_fake = generator.create_labeled_dataset(
        n_samples=N_SAMPLES, 
        use_2d=USE_2D, 
        real_type=REAL_DATA_TYPE
    )
    
    print(f"Dataset created:")
    print(f"- Total samples: {len(X)}")
    print(f"- Real samples: {np.sum(y)}")
    print(f"- Fake samples: {len(y) - np.sum(y)}")
    print(f"- Feature dimensions: {X.shape[1]}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train and evaluate classifiers
    classifier = RealFakeClassifier()
    classifier.train_models(X_train, y_train)
    classifier.evaluate_models(X_test, y_test)
    
    # Print results
    classifier.print_results()
    
    # Create visualizations
    print("\nGenerating visualizations...")
    classifier.plot_results(X_test, y_test, X_real, X_fake, USE_2D)
    
    # Summary
    best_model = max(classifier.results.keys(), 
                    key=lambda k: classifier.results[k]['roc_auc'])
    best_auc = classifier.results[best_model]['roc_auc']
    
    print(f"\n=== Summary ===")
    print(f"Best performing model: {best_model.title()}")
    print(f"Best ROC AUC score: {best_auc:.4f}")
    print(f"Classification task: {'Easy' if best_auc > 0.9 else 'Moderate' if best_auc > 0.7 else 'Challenging'}")


if __name__ == "__main__":
    main()