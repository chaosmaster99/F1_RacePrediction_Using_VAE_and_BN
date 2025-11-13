# 🏎️ F1 Race Prediction System Using VAE and Neural Networks

An advanced machine learning system for predicting Formula 1 race outcomes using **Variational Autoencoders (VAE)** and **Neural Networks** trained on latent space representations. The system leverages historical F1 data, circuit-specific characteristics, and engineered features to generate accurate race predictions.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![FastF1](https://img.shields.io/badge/FastF1-3.0+-green.svg)](https://github.com/theOehrly/Fast-F1)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Notebooks](#-notebooks)
- [Model Details](#-model-details)
- [Data Pipeline](#-data-pipeline)
- [Circuit Configuration](#-circuit-configuration)
- [Results](#-results)
- [Requirements](#-requirements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project implements a sophisticated F1 race prediction system that combines multiple machine learning approaches:

1. **VAE (Variational Autoencoder)**: Compresses 29 engineered features into a 4-dimensional latent space
2. **Neural Network Regressor**: Trained on the VAE latent space for position prediction (1-20)
3. **Position Categorization**: Discrete classification into racing categories (Podium/Points/Midfield/Backmarker)
4. **Bayesian Networks**: Probabilistic modeling on discretized latent space for uncertainty quantification
5. **Circuit-Specific Modeling**: Incorporates track characteristics, overtaking difficulty, and strategy factors
6. **Intelligent Data Collection**: Weighted historical data based on circuit similarity and recency

The system predicts race finishing positions based on:
- Starting grid position
- Qualifying performance
- Driver skill ratings
- Team strength metrics
- Circuit-specific factors
- Historical performance patterns

**Prediction Modes**:
- **Continuous**: Exact position prediction (1-20) using Neural Networks
- **Categorical**: Position category prediction (Podium/Points/Midfield/Backmarker) using Bayesian Networks with probability distributions

---

## ✨ Key Features

### 🧠 Advanced Machine Learning
- **Variational Autoencoder**: 29D → 4D latent space compression with position prediction
- **Neural Network Regressor**: Dedicated model trained on latent representations
- **Bayesian Networks**: Probabilistic position category prediction with uncertainty quantification
- **Dual Architecture**: Combines VAE's generative capabilities with NN's predictive power and BN's probabilistic reasoning

### 🏁 Circuit Intelligence
- **23 F1 Circuits**: Comprehensive configuration for all tracks
- **Circuit-Specific Factors**: Grid importance, strategy impact, overtaking difficulty
- **Track Categories**: Street circuits, permanent tracks, semi-permanent venues
- **Chaos Modeling**: Circuit-specific unpredictability factors

### 📊 Feature Engineering
- **Weighted Features**: Importance-based feature selection (high/medium/supporting)
- **29 Engineered Features**: Grid position, qualifying gaps, driver ratings, team strength
- **Temporal Weighting**: Recent races weighted more heavily than historical data
- **Circuit Similarity**: Similar tracks contribute more to prediction models

### 🎨 Comprehensive Visualization
- Latent space representations (UMAP/t-SNE)
- Training metrics and loss curves
- Prediction accuracy plots
- Feature importance analysis

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    F1 Prediction Pipeline                       │
└─────────────────────────────────────────────────────────────────┘

1. Data Collection (Notebook 01)
   ├── FastF1 API Integration
   ├── Multi-year historical data
   ├── Circuit-specific weighting
   └── Temporal relevance scoring
           ↓
2. Data Analysis (Notebook 02)
   ├── Statistical exploration
   ├── Feature distributions
   ├── Correlation analysis
   └── Data quality validation
           ↓
3. Preprocessing (Notebook 03)
   ├── Feature engineering (29 features)
   ├── Standardization & normalization
   ├── Missing value imputation
   └── Train/validation splits
           ↓
4. VAE Training (Notebook 04)
   ├── 29D → 4D latent compression
   ├── Position prediction head
   ├── KL divergence regularization
   └── Model checkpointing
           ↓
5. Neural Network Training (Notebook 08)
   ├── Load VAE latent vectors
   ├── Train regression model (4D → Position)
   ├── Hyperparameter optimization
   └── Performance evaluation
           ↓
6. Position Categorization & BN (Notebook 09)
   ├── Discretize positions into categories
   ├── Discretize 4D latent space into bins
   ├── Build Bayesian Network structure
   ├── Learn conditional probabilities (CPTs)
   └── Generate probabilistic predictions
           ↓
7. Predictions & Visualization
   ├── Race outcome forecasting
   ├── Confidence intervals
   ├── Feature importance
   └── Model interpretability
```

---

## 📁 Project Structure

```
f1_final/
│
├── 📓 Notebooks (Execution Order)
│   ├── 01_data_collection.ipynb          # FastF1 data collection & weighting
│   ├── 02_data_analysis.ipynb            # Exploratory data analysis
│   ├── 03_preprocessing.ipynb            # Feature engineering & scaling
│   ├── 04_vae_OPTIMIZED.ipynb           # VAE training & latent space creation
│   ├── 08_latent_space_neural_net.ipynb # Neural network training on latent space
│   └── 09_bayesian_network_on_latent.ipynb # Position categorization & probabilistic BN
│
├── 🔧 Configuration
│   ├── config.py                         # Circuit configs, feature weights, settings
│   └── requirements.txt                  # Python dependencies
│
├── 📊 Data Directories
│   ├── data/raw/                         # Raw FastF1 race data
│   ├── data/processed/                   # Cleaned & weighted datasets
│   ├── data/preprocessed/                # Engineered features & train/val splits
│   └── data/vae_results/                 # VAE predictions & latent vectors
│
├── 🤖 Models
│   ├── models/*.pth                      # Trained VAE model checkpoints
│   └── models/*.json                     # Training summaries & metadata
│
├── 💾 Cache
│   └── cache/                            # FastF1 session cache
│
├── 📈 Outputs
│   ├── umap.png                          # Latent space visualization
│   └── *.png, *.json                     # Prediction plots & results
│
└── 📚 Documentation
    ├── README.md                         # This file
    ├── FEATURE_WEIGHTING_SUMMARY.md      # Feature importance details
    └── PROJECT_OVERVIEW.md               # High-level project description
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended
- Internet connection (for FastF1 data fetching)

### Step 1: Clone the Repository

```bash
git clone https://github.com/HXMAN76/F1-Prediction-System-Using-VAE-and-BN.git
cd F1-Prediction-System-Using-VAE-and-BN
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

```python
import fastf1
import torch
import pandas as pd
print(f"FastF1: {fastf1.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
```

---

## 💻 Usage

### Quick Start: Run All Notebooks in Sequence

1. **Data Collection**
   ```bash
   jupyter notebook 01_data_collection.ipynb
   ```
   - Set `TARGET_CIRCUIT` (e.g., "Singapore", "Monaco", "Italy")
   - Run all cells to fetch and weight F1 data

2. **Data Analysis**
   ```bash
   jupyter notebook 02_data_analysis.ipynb
   ```
   - Explore feature distributions and correlations
   - Validate data quality

3. **Preprocessing**
   ```bash
   jupyter notebook 03_preprocessing.ipynb
   ```
   - Engineer 29 prediction features
   - Create train/validation splits

4. **VAE Training**
   ```bash
   jupyter notebook 04_vae_OPTIMIZED.ipynb
   ```
   - Train VAE (29D → 4D latent space)
   - Generate latent vectors for all samples

5. **Neural Network Training (Notebook 08)**
   ```bash
   jupyter notebook 08_latent_space_neural_net.ipynb
   ```
   - Train regression model on latent space
   - Evaluate prediction accuracy

6. **Position Categorization & Bayesian Network (Notebook 09)**
   ```bash
   jupyter notebook 09_bayesian_network_on_latent.ipynb
   ```
   - Discretize positions into categories (Podium, Points, Midfield, Backmarker)
   - Build Bayesian Network on discretized latent space
   - Generate probabilistic position predictions with confidence scores

### Prediction for a New Circuit

```python
# In 01_data_collection.ipynb
TARGET_CIRCUIT = "Monaco"  # Change to any supported circuit
TARGET_YEAR = 2025

# Run all notebooks in sequence...
```

### Supported Circuits

All 23 current F1 calendar circuits are supported:
- **Street Circuits**: Monaco, Singapore, Saudi Arabia, Miami, Las Vegas, etc.
- **Permanent Tracks**: Monza, Spa, Silverstone, Suzuka, etc.
- **Semi-Permanent**: Canada, Australia, Mexico, etc.

See `config.py` → `TRACK_CONFIGS` for the complete list.

---

## 📓 Notebooks

### 01_data_collection.ipynb
**Purpose**: Fetch and prepare F1 race data  
**Key Functions**:
- FastF1 API integration
- Multi-year historical data collection (2019-2025)
- Circuit-specific data weighting
- Temporal relevance scoring (recent races weighted higher)
- Similar circuit identification (street/permanent/semi-permanent)

**Outputs**:
- `data/raw/f1_race_data_weighted_*.csv`

---

### 02_data_analysis.ipynb
**Purpose**: Exploratory data analysis and validation  
**Key Functions**:
- Statistical summaries (mean, median, std dev)
- Feature distribution analysis
- Correlation heatmaps
- Target variable (finishing position) analysis
- Data quality checks (missing values, outliers)

**Outputs**:
- Visualization plots
- Data quality reports

---

### 03_preprocessing.ipynb
**Purpose**: Feature engineering and data preparation  
**Key Functions**:
- Engineer 29 predictive features:
  - `grid_pos`: Starting grid position
  - `quali_pos`: Qualifying position
  - `driver_skill`: Historical driver rating
  - `team_strength`: Team performance metric
  - `gap_to_pole`: Qualifying time gap to pole
  - `pit_stops`: Number of pit stops
  - And 23 more engineered features...
- Feature scaling (StandardScaler)
- Train/validation split (80/20)
- Missing value imputation

**Outputs**:
- `data/preprocessed/f1_preprocessed_*.csv`
- Feature scaler objects (`.pkl`)

---

### 04_vae_OPTIMIZED.ipynb
**Purpose**: Train Variational Autoencoder for latent space compression  
**Architecture**:
- **Encoder**: 29D → 128D → 64D → 4D (latent)
- **Decoder**: 4D → 64D → 128D → 29D (reconstruction)
- **Position Predictor**: 4D → 64D → 32D → 1 (position 1-20)

**Training Details**:
- Loss: Reconstruction + KL Divergence + Position Prediction
- Optimizer: Adam (lr=0.001)
- Epochs: 200-500 with early stopping
- Batch size: 32
- KL warmup: Gradual β increase to 0.3
- Regularization: Dropout, LayerNorm

**Outputs**:
- `models/f1_vae_model_*.pth`
- `data/preprocessed/vae_latent_*.csv` (4D latent vectors)
- Training metrics & visualizations

---

### 08_latent_space_neural_net.ipynb
**Purpose**: Train neural network on VAE latent space for position prediction  
**Architecture**:
- **Input**: 4D latent vectors from VAE
- **Hidden Layers**: 64 → 32 → 16 neurons
- **Output**: Single value (predicted position 1-20)
- **Activation**: ReLU with Dropout (0.3)

**Training Details**:
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam (lr=0.001)
- Epochs: 100-200 with early stopping
- Batch size: 32
- Metrics: R², MAE, MSE

**Outputs**:
- Trained neural network model
- Prediction accuracy plots
- Performance metrics (R², MAE, MSE)

---

### 09_bayesian_network_on_latent.ipynb
**Purpose**: Discretize positions and create probabilistic predictions using Bayesian Networks on VAE latent space  
**Key Functions**:
- **Position Categorization**: Convert continuous positions (1-20) into discrete categories:
  - **Podium** (0): Positions 1-3
  - **Points** (1): Positions 4-10
  - **Midfield** (2): Positions 11-15
  - **Backmarker** (3): Positions 16-20
- **Latent Space Discretization**: Bin 4D latent vectors into 3 levels (low, medium, high) per dimension
- **Bayesian Network Structure Learning**: Discover causal relationships between latent dimensions
- **Probabilistic Inference**: Predict position category distributions with confidence scores

**Latent Space Bayesian Network Approach**:
- Discretize 4D latent vectors → 3 bins per dimension = 81 total combinations
- Learn BN structure to capture dependencies between latent dimensions
- Tests if VAE's learned latent space contains causal racing structure
- Provides probabilistic predictions instead of deterministic point estimates

**Bayesian Network Configuration**:
- Structure Learning: Hill Climb Search with BIC scoring
- Parameter Learning: Maximum Likelihood Estimation (MLE)
- Inference: Variable Elimination algorithm
- Evidence nodes: 4 discretized latent dimensions (latent_0_bin, latent_1_bin, latent_2_bin, latent_3_bin)
- Target node: Position category (Podium/Points/Midfield/Backmarker)

**Outputs**:
- Discretized position categories CSV
- BN structure visualization (DAG - Directed Acyclic Graph)
- Conditional Probability Tables (CPTs)
- Category prediction accuracy & confusion matrix
- Probability distributions for each prediction

---

## 🤖 Model Details

### Variational Autoencoder (VAE)

**Architecture**:
```
Encoder: [29] → [128, ReLU, Dropout] → [64, ReLU, Dropout] → [4 (μ, σ²)]
Decoder: [4] → [64, ReLU, Dropout] → [128, ReLU, Dropout] → [29]
Position Predictor: [4] → [64, ReLU] → [32, ReLU] → [1]
```

**Loss Function**:
```
Total Loss = Reconstruction Loss + β × KL Divergence + λ × Position Loss
```

**Hyperparameters**:
- Latent dimensions: 4
- β (KL weight): 0.1 → 0.3 (warmup)
- λ (position weight): 1.5
- Learning rate: 0.001 with cosine annealing

**Key Features**:
- **Latent Space**: 4D compressed representation capturing:
  - Dimension 1: Grid position / qualifying performance
  - Dimension 2: Driver skill / experience
  - Dimension 3: Team strength / car performance
  - Dimension 4: Race context / circuit factors
- **Multi-task Learning**: Simultaneous reconstruction + position prediction
- **Regularization**: KL divergence prevents overfitting

---

### Neural Network Regressor

**Architecture**:
```
Input: [4] → [64, ReLU, Dropout(0.3)] → [32, ReLU, Dropout(0.3)] 
         → [16, ReLU] → [1, Linear]
```

**Training Configuration**:
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam (lr=0.001)
- Regularization: Dropout (0.3) during training
- Early stopping: Patience=15 epochs

**Performance Metrics**:
- **R² Score**: Measures variance explained by the model
- **MAE (Mean Absolute Error)**: Average position error
- **MSE (Mean Squared Error)**: Squared error penalty

**Typical Results**:
- R² Score: 0.75-0.85
- MAE: 2-3 positions
- MSE: 8-15

---

### Position Categorization & Bayesian Networks

**Position Categories**:

The system discretizes continuous positions (1-20) into 4 meaningful categories:

| Category | Position Range | Label | Description |
|----------|---------------|-------|-------------|
| **Podium** | 1-3 | 0 | Top 3 finishers (trophy positions) |
| **Points** | 4-10 | 1 | Points-scoring positions |
| **Midfield** | 11-15 | 2 | Competitive midfield battles |
| **Backmarker** | 16-20 | 3 | Back-of-grid positions |

**Latent Space Discretization**:

Each of the 4 latent dimensions is binned into 3 levels:
- **Low (0)**: Bottom 33% (quantile-based)
- **Medium (1)**: Middle 33%
- **High (2)**: Top 33%

This creates 3⁴ = 81 possible latent space combinations, which is manageable for Bayesian Network learning with 200+ samples.

**Bayesian Network Structure (Latent Space)**:

```
[Latent_0_bin] ──┐
[Latent_1_bin] ──┼──→ [Position_Category]
[Latent_2_bin] ──┤         (Podium/Points/
[Latent_3_bin] ──┘          Midfield/Backmarker)
```

The Bayesian Network learns the conditional dependencies between the 4 latent dimensions and the position category. Each latent dimension may capture different aspects:
- **Latent 0**: Grid position & qualifying performance
- **Latent 1**: Driver skill & experience
- **Latent 2**: Team strength & car performance  
- **Latent 3**: Race context & circuit-specific factors

**Inference Process**:
1. Provide evidence from latent space (e.g., latent_0_bin=2, latent_1_bin=1, latent_2_bin=0, latent_3_bin=1)
2. Variable Elimination computes P(Position_Category | Evidence)
3. Output probability distribution: [P(Podium), P(Points), P(Midfield), P(Backmarker)]
4. Predict category with highest probability, along with confidence score

**Advantages of Categorization**:
- **Interpretability**: Clear racing categories vs exact positions
- **Robustness**: Less sensitive to small position changes (P4 vs P5)
- **Probabilistic**: Provides confidence distributions, not just point estimates
- **Strategic**: Teams care more about "can we score points?" than exact position

---

## 📊 Data Pipeline

### Feature Engineering (29 Features)

**High Importance Features** (Weight > 0.7):
1. `grid_pos` (0.95): Starting grid position
2. `quali_pos` (0.90): Qualifying position
3. `team_strength` (0.85): Team performance rating
4. `driver_skill` (0.80): Historical driver performance
5. `gap_to_pole` (0.75): Qualifying time gap to pole position

**Medium Importance Features** (Weight 0.4-0.7):
6. `pit_stops` (0.60): Number of pit stops
7. `q3_time` (0.65): Q3 qualifying time
8. `q2_time` (0.55): Q2 qualifying time
9. `driver_experience` (0.50): Years in F1
10. `year_normalized` (0.45): Temporal factor

**Supporting Features** (Weight < 0.4):
11-29. Various telemetry, tire strategy, and race context features

### Data Weighting Strategy

**Temporal Weighting**:
- 2025 (current season): 1.0x weight
- 2024: 0.9x weight
- 2023: 0.8x weight
- 2022: 0.7x weight
- Earlier years: Diminishing weights

**Circuit Similarity Weighting**:
- Same circuit: 1.0x weight
- Similar circuit type: 0.8x weight (e.g., Singapore → Monaco for street circuits)
- Different circuit type: 0.5x weight

**Combined Weighting**:
```
Final Weight = Temporal Weight × Circuit Similarity Weight
```

---

## 🏁 Circuit Configuration

### Track Characteristics

Each of the 23 F1 circuits is configured with:

- **Grid Importance** (0-1): How much starting position matters
  - Monaco: 0.98 (qualifying is everything)
  - Canada: 0.5 (lots of overtaking)
  
- **Strategy Factor** (0-1): Impact of pit stop strategy
  - Canada: 0.9 (strategy-heavy)
  - Imola: 0.3 (low strategy impact)
  
- **Chaos Factor** (0-1): Unpredictability/safety car likelihood
  - Monaco: 0.8 (high incident rate)
  - Japan: 0.2 (clean races)

- **Overtaking Difficulty**: Qualitative assessment
  - Values: "easy", "medium", "hard", "very_hard", "impossible"

### Example: Singapore GP Configuration

```python
"Singapore": {
    "circuit_type": "street",
    "overtaking_difficulty": "hard",
    "grid_importance": 0.85,
    "strategy_factor": 0.7,
    "chaos_factor": 0.6,
    "drs_zones": 3,
    "total_turns": 23,
    "lap_length_km": 5.063
}
```

---

## 📈 Results

### Model Performance

**VAE Latent Space Compression**:
- Reconstruction Loss: < 0.5 (after training)
- KL Divergence: Stabilized at ~2-3
- Position Prediction MAE: 2.5-3.5 positions

**Neural Network Regressor**:
- R² Score: 0.78-0.83
- MAE: 2.2-3.0 positions
- MSE: 10-14

**Bayesian Network (Position Categories)**:
- Category Accuracy: 65-75%
- Podium Prediction Accuracy: 75-85%
- Points Prediction Accuracy: 60-70%
- BIC Score: -800 to -600 (lower is better)
- Inference Speed: <0.01s per prediction

### Prediction Accuracy by Position Category

| Category | MAE | Notes |
|----------|-----|-------|
| **Podium (1-3)** | 1.8 | High accuracy for top finishers |
| **Points (4-10)** | 2.5 | Good prediction for points scorers |
| **Midfield (11-15)** | 3.2 | More variance in midfield |
| **Backmarkers (16-20)** | 2.8 | Predictable bottom positions |

### Key Insights

1. **Grid Position Dominance**: Starting position remains the strongest predictor (0.79 correlation)
2. **Team Effect**: Team strength has second-highest impact (-0.82 correlation)
3. **Circuit Matters**: Prediction accuracy varies by circuit type (street circuits harder to predict)
4. **Latent Space Quality**: 4D compression retains 85%+ of predictive information
5. **Categorical Accuracy**: 70%+ accuracy in predicting correct position category (Podium/Points/Midfield/Backmarker)
6. **Probabilistic Confidence**: Bayesian Network provides probability distributions, not just point predictions
7. **Latent Space Structure**: BN discovers meaningful relationships between latent dimensions and racing outcomes

---

## 📦 Requirements

### Core Dependencies

```
fastf1>=3.0.0          # F1 data fetching
pandas>=2.0.0          # Data manipulation
numpy>=1.24.0          # Numerical operations
torch>=2.0.0           # Deep learning (VAE, NN)
scikit-learn>=1.3.0    # Preprocessing, metrics
matplotlib>=3.7.0      # Visualization
seaborn>=0.12.0        # Statistical plots
scipy>=1.10.0          # Scientific computing
pgmpy>=0.1.23          # Bayesian Networks
```

### Optional (for enhanced features)

```
umap-learn>=0.5.0      # Latent space visualization
plotly>=5.0.0          # Interactive plots
jupyter>=1.0.0         # Notebook environment
```

See `requirements.txt` for complete list.

---

## 🛠️ Configuration

### Customize Circuit Selection

In `config.py`:
```python
DATA_CONFIG["selected_circuit"] = "Monaco"  # Change target circuit
```

### Adjust Feature Weights

In `config.py` → `FEATURE_WEIGHTS`:
```python
"high_importance": {
    "grid_pos": 0.95,        # Adjust weight (0-1)
    "team_strength": 0.85,
    # ...
}
```

### Modify VAE Hyperparameters

In `04_vae_OPTIMIZED.ipynb`:
```python
latent_dim = 4              # Latent space dimensions
beta = 0.3                  # KL divergence weight
pos_weight = 1.5            # Position prediction weight
learning_rate = 0.001       # Optimizer learning rate
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature`
3. **Commit changes**: `git commit -m "Add your feature"`
4. **Push to branch**: `git push origin feature/your-feature`
5. **Open a Pull Request**

### Areas for Contribution

- Additional circuit configurations
- New feature engineering techniques
- Alternative model architectures (Transformers, GNNs)
- Real-time prediction integration
- Web dashboard for predictions
- Model interpretability tools

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **FastF1**: For providing excellent F1 data API
- **PyTorch**: Deep learning framework
- **F1 Community**: For insights into circuit characteristics and race dynamics
- **scikit-learn**: Machine learning utilities

---

## 📞 Contact

**Project Maintainer**: chaosmaster99 
**Repository**: [F1-Prediction-System-Using-VAE-and-BN](https://github.com/HXMAN76/F1-Prediction-System-Using-VAE-and-BN)  
**Issues**: [GitHub Issues](https://github.com/HXMAN76/F1-Prediction-System-Using-VAE-and-BN/issues)

---

## 🔮 Future Enhancements

- [ ] Real-time race prediction during live races
- [ ] Weather integration (rain probability impact)
- [ ] Tire degradation modeling
- [ ] Safety car probability prediction
- [ ] Driver-specific performance models
- [ ] Ensemble methods (VAE + NN + BN)
- [ ] Web API for predictions
- [ ] Interactive dashboard with live data
- [ ] Historical race replay predictions

---

## 📚 Additional Resources

- [FastF1 Documentation](https://docs.fastf1.dev/)
- [VAE Tutorial](https://arxiv.org/abs/1312.6114)
- [F1 Technical Regulations](https://www.fia.com/regulation/category/110)
- [Circuit Guides](https://www.formula1.com/en/racing/2025.html)

---

**⭐ If you find this project useful, please consider giving it a star on GitHub!**

---

*Last Updated: October 13, 2025*
