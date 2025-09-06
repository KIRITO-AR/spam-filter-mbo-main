# Spam Detection System With Enhanced MBO 🦋

## Introduction

The Spam Detection System is a comprehensive machine learning tool designed to classify emails and SMS messages as spam or legitimate (ham). This system implements multiple classification models, including a novel approach using the Monarch Butterfly Optimization (MBO) algorithm 🦋 to optimize hyperparameters for superior performance.

The system provides both a training interface and a classification interface through user-friendly GUIs built with Tkinter, making it accessible for both technical and non-technical users.

## Key Features

- **Multiple Models**: Implements three different approaches - Lite, Legacy, and MBO-optimized models
- **Advanced Optimization**: Uses the nature-inspired Monarch Butterfly Optimization algorithm to find optimal hyperparameters
- **Visual Analytics**: Generates comprehensive visualizations including dataset insights, word clouds, and performance metrics
- **User-Friendly Interface**: GUI applications for both training and classification
- **Performance Tracking**: Saves model metrics and provides detailed performance reports

## Project Structure

```
spam-filter-mbo-main/
├── app.py                 # Main classification application
├── train.py               # Main training application
├── requirements.txt       # Python dependencies
├── models.md              # Model comparison documentation
├── README.md              # Project documentation
├── data/
│   └── spam.csv           # Training dataset
├── models/
│   ├── model.pkl          # Legacy model
│   ├── vectorizer.pkl     # Legacy vectorizer
│   ├── model_optimized.pkl# Optimized model
│   ├── vectorizer_optimized.pkl # Optimized vectorizer
│   └── metrics.txt        # Model performance metrics
├── graphs/
│   ├── dataset_insights.png   # Data distribution visualizations
│   ├── wordclouds.png         # Word cloud visualizations
│   └── performance_metrics.png# Model performance charts
├── training/
│   ├── train_model_lite.py    # Lite model training script
│   ├── train_model_legacy.py  # Legacy model training script
│   └── train_model_mbo.py     # MBO-optimized model training script
└── .gitignore
```

## Getting Started

### Prerequisites

- Python 3.6 or higher
- pip package installer

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd spam-filter-mbo-main
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download NLTK data (done automatically but can be manual):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   nltk.download('stopwords')
   ```

### Required Python Packages

The system requires the following key packages:
- tkinter (for GUI)
- nltk (natural language processing)
- scikit-learn (machine learning)
- pandas (data manipulation)
- numpy (numerical computing)
- matplotlib & seaborn (data visualization)
- wordcloud (text visualization)
- pillow (PIL - image handling)
- torch (PyTorch - for GPU acceleration in MBO)

## Dataset

The system uses the `spam.csv` dataset containing labeled messages. The dataset has two columns:
- `v1`: Label (spam or ham)
- `v2`: Text message content

The dataset is automatically preprocessed to remove unnecessary columns and renamed for clarity.

## Application Overview

### Spam Classifier UI (app.py)

A GUI application for classifying messages as spam or ham using trained models.

#### Features
- Model selection between optimized and legacy models
- Text input area for message classification
- Color-coded results (red for spam, green for ham)
- Real-time classification

#### How to Run
```bash
python app.py
```

### Model Trainer UI (train.py)

A comprehensive GUI application for training different models and visualizing results.

#### Features
- Selection between Lite, Legacy, and MBO models
- MBO parameter configuration (butterfly count, migration ratio, etc.)
- GPU acceleration support for MBO
- Progress indication during training
- Visualization of dataset insights, word clouds, and performance metrics
- Detailed training logs

#### How to Run
```bash
python train.py
```

## Training Models

### 1. Lite Model
- Fast training with pre-configured hyperparameters
- Uses lemmatization for text preprocessing
- Ensemble of SVC, MultinomialNB, and ExtraTreesClassifier with fixed weights
- Ideal for quick prototyping and testing

### 2. Legacy Model
- Traditional approach with basic ensemble
- Uses Porter Stemming for text preprocessing
- Simpler hyperparameters compared to optimized models
- Balanced between speed and performance

### 3. Monarch Butterfly Optimization (MBO) Model 🦋
- Advanced optimization using nature-inspired algorithm
- Optimizes 7 key parameters simultaneously:
  - SVC parameters (C, gamma)
  - MultinomialNB alpha
  - Number of trees in ExtraTreesClassifier
  - Ensemble weights for all three classifiers
- Population-based search with configurable parameters:
  - Butterfly count (default: 20)
  - Migration ratio (default: 0.85)
  - Maximum iterations (default: 30)
- Supports GPU acceleration via PyTorch
- Most computationally intensive but typically highest performing

## The Monarch Butterfly Optimization Algorithm 🦋

The MBO algorithm mimics the migration behavior of monarch butterflies to find optimal solutions in a search space. In this implementation:

1. **Population Initialization**: Creates a population of "butterflies" with random parameter values
2. **Fitness Evaluation**: Each butterfly's fitness is evaluated using cross-validation scores
3. **Migration Process**: Butterflies move toward better solutions based on migration operators
4. **Iteration**: The process repeats for a specified number of iterations
5. **Optimal Solution**: The best parameters found are used to train the final model

## Visualizations

The system generates three key visualizations after training:

1. **Dataset Insights** (`graphs/dataset_insights.png`)
   - Message length distribution by class
   - Class distribution (spam vs ham)
   - Word count comparison

2. **Word Clouds** (`graphs/wordclouds.png`)
   - Visual representation of most frequent words in spam messages
   - Visual representation of most frequent words in legitimate messages

3. **Performance Metrics** (`graphs/performance_metrics.png`)
   - Confusion matrix
   - Classification report heatmap
   - Top feature importances

## Model Performance

Performance metrics are automatically saved to `models/metrics.txt` after training:
- Accuracy
- Precision
- F1 Score

## Usage Workflow

1. **Prepare Dataset**: Ensure `spam.csv` is in the `data/` directory
2. **Train Models**: Run `python train.py` to launch the trainer GUI
3. **Select Model**: Choose between Lite, Legacy, or MBO models
4. **Configure Parameters**: Adjust MBO parameters if using the MBO model
5. **Start Training**: Click "Train Model" and monitor progress in the logs
6. **Review Results**: Check visualizations and metrics in their respective tabs
7. **Classify Messages**: Run `python app.py` to launch the classifier and test messages

## Error Handling

The applications include comprehensive error handling for:
- Missing models or datasets
- Invalid parameter configurations
- NLTK data download issues
- GPU availability and CUDA errors
- File I/O problems

Informative error messages are displayed to guide users in resolving issues.

## Troubleshooting

### Common Issues

1. **NLTK Data Missing**: Run the NLTK download commands manually
2. **Module Not Found**: Ensure all requirements are installed with `pip install -r requirements.txt`
3. **CUDA Errors**: If GPU acceleration fails, the system automatically falls back to CPU
4. **GUI Display Issues**: Check Python and tkinter installation

### Performance Tips

1. **For Quick Testing**: Use the Lite model
2. **For Best Performance**: Use the MBO model with higher butterfly counts and iterations
3. **For GPU Acceleration**: Ensure PyTorch with CUDA support is installed
4. **Memory Management**: The MBO model can be memory-intensive with large populations

## Contributing

Contributions to improve the spam detection system are welcome. Areas for improvement include:
- Additional optimization algorithms
- More sophisticated text preprocessing
- Enhanced visualization capabilities
- Performance optimizations

## License

This project is provided for educational and research purposes.

## Contact

For questions or support, please open an issue in the repository.

---

*This documentation provides a comprehensive overview of the Spam Detection System with Monarch Butterfly Optimization, including setup instructions, usage guidelines, and technical details about the implementation.*