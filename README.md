# Pokémon Competitive AI Project

A comprehensive machine learning system for generating optimal Pokémon movesets and finding strategic counters, featuring reinforcement learning, GPT-based evaluation, and battle simulation capabilities.

## 📁 Project Structure

Based on the directory structure shown, the project is organized as follows:

```
project/
├── core/                    # Current, production-ready implementations
│   ├── ai.py               # Advanced Multi-Task Neural Network (latest)
│   ├── aicounter.py        # AI Counter Finder with RL capabilities
│   ├── apichecker.py       # GPT-based moveset evaluation
│   ├── counterfinder.py    # Traditional rule-based counter finder
│   ├── counterrl.py        # Reinforcement learning models
│   ├── counterstats.py     # Performance analysis and statistics
│   ├── main.py             # Traditional counter finder interface
│   ├── masstraincounter.py # Large-scale distributed training
│   ├── models.py           # Data models and Pokemon classes
│   ├── moveutility.py      # Move evaluation utilities
│   ├── parser.py           # Moveset parsing functionality
│   ├── repository.py       # Data access and type chart logic
│   ├── showdownsim.py      # Pokemon Showdown battle simulation
│   ├── testcounters.py     # Counter testing and validation
│   ├── traincounter.py     # Counter training orchestration
│   ├── utils.py            # Validation utilities
│   ├── data/               # Core datasets
│   │   ├── abilities.json
│   │   ├── aliases.json
│   │   ├── items.json
│   │   ├── moves.json
│   │   ├── pokemon.json
│   │   ├── smogonMovesets.json
│   │   ├── typeChart.csv
│   │   └── models/         # Trained models and generated movesets
│   │       ├── all_movesets_r*.json
│   │       └── moveset_rl_model_*.pt
│   └── seed42_*.json      # Experimental results and benchmarks
├── old/                    # Legacy implementations and ALL historical versions
│   ├── aiV1.py            # XGBoost-based moveset generator
│   ├── aiV2.py            # PyTorch neural network implementation
│   ├── movesetscorer.py   # Statistical analysis of generated movesets
│   ├── scoremovesets.py   # GPT scoring automation
│   ├── all_movesets_r*.json # Generated moveset datasets
│   ├── scored_movesets_*.json # GPT-scored datasets
│   ├── counterfinder*.py  # Various versions of counter finding logic
│   ├── version*.py        # Complete historical progression of implementations
│   └── dataModels/        # Legacy model storage
│       ├── singleTask/    # Single-task model experiments
│       └── xbBoost/       # XGBoost model files
└── getData/               # Data collection and preprocessing
    ├── getDatasets/       # Current data acquisition scripts
    │   ├── abilities.js
    │   ├── converttoJson.js
    │   ├── items.js
    │   ├── moves.js
    │   └── getPokemonMoveset.py
    └── oldDatasets/       # Legacy data collection scripts
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js (for Pokémon Showdown)
- Required Python packages (see Installation)

### Installation

1. Install Python dependencies:
```bash
pip install torch scikit-learn pandas numpy joblib tqdm poke-env openai xgboost
```

2. Set up local Pokémon Showdown server:

First, clone the official Pokémon Showdown repository:
```bash
cd core/
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
node pokemon-showdown start --no-security
```
**Note**: This project uses the official Pokémon Showdown simulator from Smogon (https://github.com/smogon/pokemon-showdown). The local Showdown instance must be running for battle simulations to work.

3. Configure OpenAI API key in `core/apichecker.py` for GPT-based moveset evaluation.
   - **Total project cost**: Less than $15 USD for all GPT evaluations

## 🧠 Core Components (`/core`)

### Moveset Generation

#### `ai.py` - Advanced Multi-Task Neural Network (Latest)
The most sophisticated implementation featuring:
- **Multi-task surrogate network** with separate heads for each moveset component
- **Hierarchical training** with 3-stage pipeline (Nature/Ability → EVs/IVs → Moves)
- **Feature interactions** and rule-based pruning
- **Reinforcement learning** from GPT feedback

```bash
# Generate movesets for all Pokémon
python ai.py --export-json

# Train with reinforcement learning using pre-trained model
python ai.py --rounds --load-model moveset_rl_model_r1.pt

# Generate for specific Pokémon
python ai.py --pokemon Skeledirge
```

### Counter Finding

#### `aicounter.py` - AI-Enhanced Counter System
Intelligent counter-finding with reinforcement learning:
- **Adaptive feature weights** learned from battle outcomes
- **Real battle simulation** validation via Showdown
- **Role-based analysis** (Physical Sweeper, Wall, Support, etc.)
- **Type effectiveness** and stat advantage calculation

#### `main.py` - Traditional Counter Finder Interface
Rule-based counter finding using:
- Type chart analysis
- Statistical advantages
- Move utility evaluation

```bash
# Interactive counter finding
python main.py
```

#### `counterfinder.py` - Counter Finding Logic
Core counter-finding implementation with configurable data source:

**Important**: To change the counter Pokémon database from Smogon movesets to a different source, modify **line 33** in `counterfinder.py`:

```python
# Line 33 - Change this to use different moveset database
with open("data/smogonMovesets.json") as f:
    self.smogonMovesets = json.load(f)
```

Replace with your preferred dataset:
```python
# Example: Use custom generated movesets
with open("data/models/all_movesets_r6.json") as f:
    self.smogonMovesets = json.load(f)

# Example: Use alternative competitive dataset  
with open("data/alternativeCompetitive.json") as f:
    self.smogonMovesets = json.load(f)
```

### Mass Training & Analysis

#### `masstraincounter.py` - Distributed Training
Large-scale reinforcement learning system:
- **Batch processing** of thousands of movesets
- **Memory management** for long training sessions
- **Progress tracking** and weight persistence

```bash
# Run massive training campaign
python masstraincounter.py --batches 1 --batch-size 50 --battles 20
```

#### `counterstats.py` - Performance Analysis
Extract and analyze training statistics:
```bash
# Compile results from training runs
python counterstats.py seed42_rule_smogondb.json
```

#### `testcounters.py` - Validation and Benchmarking
Test counter-finding systems:
```bash
# Test rule-based counter finder
python testcounters.py -d data/models/all_movesets_r6.json -c rule --seed 42 -n 200 -b 20 -o seed42_rule_generateddb
```

## 📊 Legacy Implementations (`/old`)

The `/old` directory contains **every single version** of the codebase throughout development, providing a complete historical record of the project's evolution.

### Historical Development

#### `aiV1.py` - XGBoost Implementation
Initial ML approach using XGBoost regression:
- Heuristic-based training targets
- Parallel batch evaluation
- Smogon data integration

```bash
# Generate using trained XGBoost model
python aiV1.py --export-json --load-model dataModels/xbBoost/moveset_rl_model_r6.pkl
```

#### `aiV2.py` - PyTorch Neural Network
Intermediate implementation with PyTorch:
- 8-dimensional output matching GPT scorer
- Multi-head training architecture
- Improved feature engineering

```bash
# Generate using PyTorch model
python aiV2.py --export-json --load-model moveset_rl_model_r1.pt
```

### Evaluation Tools

#### `movesetscorer.py` - Statistical Analysis
Analyze GPT scores across generated movesets:
```bash
# Calculate statistics for moveset quality
python movesetscorer.py scored_movesets_v3_r0.json
```

#### `scoremovesets.py` - GPT Scoring Automation
Automated GPT evaluation of generated movesets:
```bash
# Score generated movesets with GPT
python scoremovesets.py --input all_movesets_r6.json --output scored_movesets_v3_r6.json --delay 0.01
```

### Complete Version History
The `/old` directory preserves:
- `version1.py`, `version2.py`, `version3.py` - Complete chronological development
- `counterfinder*.py` - Various iterations of counter-finding algorithms
- Multiple `aiV*.py` files - Different ML approach experiments
- All intermediate datasets and model files

## 🔄 Workflow Example

### 1. Data Preparation (`/getData`)
Scripts in `getDatasets/` collect and preprocess Pokémon data from various sources.

### 2. Moveset Generation
Choose your approach:
- **`ai.py`**: Most advanced, multi-task neural network (recommended)
- **`aiV1.py`**: XGBoost baseline (legacy)
- **`aiV2.py`**: PyTorch intermediate (legacy)

### 3. Counter Analysis
Test against generated movesets:

**Sample moveset for testing:**
```
Annihilape @ Leftovers
Ability: Defiant
EVs: 240 HP / 252 SpD / 16 Spe
Tera Type: Water
Careful Nature
- Bulk Up
- Taunt
- Drain Punch
- Rage Fist
```

### 4. Training & Optimization
- Use `masstraincounter.py` for large-scale RL training
- Monitor progress with `counterstats.py`
- Evaluate quality with `movesetscorer.py`

## 🎯 Key Features

### Advanced ML Techniques
- **Multi-task learning** with shared representations
- **Reinforcement learning** from battle simulation feedback
- **Hierarchical model architecture** mimicking human decision-making
- **Feature interaction modeling** for nature-stat synergies

### Battle Integration
- **Real-time simulation** via official Pokémon Showdown (https://github.com/smogon/pokemon-showdown)
- **Win rate optimization** through actual battle outcomes
- **Automated testing** of thousands of matchups

### Scalability
- **Parallel processing** for candidate evaluation
- **Batch training** for memory efficiency
- **Incremental learning** with weight persistence

### Validation
- **GPT-based scoring** for moveset quality assessment (< $15 total cost)
- **Statistical analysis** of performance metrics
- **A/B testing** between different approaches

## 📈 Model Evolution

The project showcases clear evolution in approach:

1. **Rule-based systems** (`main.py`, `counterfinder.py`)
2. **Traditional ML** (`aiV1.py` with XGBoost)
3. **Deep learning** (`aiV2.py` with PyTorch)
4. **Advanced multi-task RL** (`ai.py` with hierarchical training)

## 🔧 Configuration

### Required Setup
1. **Local Showdown**: 
   - Clone official Pokémon Showdown: `git clone https://github.com/smogon/pokemon-showdown.git`
   - Run `node pokemon-showdown start --no-security` in the cloned directory
   - This project relies on the official Smogon Pokémon Showdown simulator for battle validation
2. **API Keys**: Configure OpenAI key in `core/apichecker.py` (total cost < $15)
3. **Data Paths**: Core datasets located in `core/data/`
4. **Models**: Trained models stored in `core/data/models/` and `old/dataModels/`

### File Naming Convention
- `all_movesets_r*.json`: Generated moveset datasets (round number)
- `scored_movesets_v*_r*.json`: GPT-evaluated datasets (version, round)
- `seed42_*`: Experimental results with fixed random seed
- `moveset_rl_model_*.pt/.pkl`: Trained model files

### Sample Commands

```bash
# Statistics and analysis
python counterstats.py seed42_rule_smogondb.json 
python movesetscorer.py scored_movesets_v3_r0.json 

# Testing and validation
python testcounters.py -d data/models/all_movesets_r6.json -c rule --seed 42 -n 200 -b 20 -o seed42_rule_generateddb

# GPT scoring (requires API key)
python scoremovesets.py --input all_movesets_r6.json --output scored_movesets_v3_r6.json --delay 0.01

# Mass training
python masstraincounter.py --batches 1 --batch-size 50 --battles 20

# Model generation
python aiV1.py --export-json --load-model dataModels/xbBoost/moveset_rl_model_r6.pkl
python aiV2.py --export-json --load-model moveset_rl_model_r1.pt
python ai.py --rounds --load-model moveset_rl_model_r1.pt

# Interactive counter finding
python main.py
```

This project represents a comprehensive exploration of AI techniques applied to competitive Pokémon strategy, evolving from simple heuristics to sophisticated neural networks that learn from real battle outcomes. The complete version history in `/old` provides valuable insight into the iterative development process and experimental approaches tried throughout the project's evolution.
