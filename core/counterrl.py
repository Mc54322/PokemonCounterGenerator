import numpy as np
import json
import os
from typing import Dict, List, Tuple, Any, Optional

class CounterRLModel:
    """
    Reinforcement Learning model for optimizing counter Pokémon selection.
    Uses a policy-gradient approach to learn feature weights that lead to successful counters.
    """
    
    def __init__(self, learning_rate: float = 0.05, discount_factor: float = 0.95):
        self.learningRate = learning_rate
        self.discountFactor = discount_factor
        
        # Initialize feature weights for counter selection
        self.weights = {
            "offensive_util": 0.5,      # Weight for offensive type advantage
            "defensive_util": 0.5,      # Weight for defensive resistances
            "advantage_bonus": 0.5,     # Weight for stat-based advantages
            "bst_bonus": 0.5,           # Weight for Base Stat Total
            "ability_bonus": 0.5,       # Weight for ability rating
            "role_counter": 0.5,        # Weight for role-based countering
            "type_resistance": 0.5,     # Weight for type-based resistances
            "speed_advantage": 0.5,     # Weight for speed advantage
            "move_coverage": 0.5        # Weight for move coverage
        }
        
        # Tracking data for performance and learning
        self.experience_buffer = []     # [(features, reward)]
        self.winRates = {}              # {matchup_key: win_rate}
        self.counterHistory = {}        # {input_pokemon: [counters]}
        self.trainingIterations = 0     # Count of training iterations
        
    def predict(self, features: Dict[str, float]) -> float:
        """
        Calculate a score for a candidate counter based on its features.
        """
        return sum(self.weights[k] * v for k, v in features.items() if k in self.weights)
    
    def updateFromExperience(self, features: Dict[str, float], reward: float):
        """
        Update weights based on the outcome of a battle simulation.
        """
        # Store experience for batch updates
        self.experience_buffer.append((features, reward))
        
        # Update weights using policy gradient approach
        for feature_name, feature_value in features.items():
            if feature_name in self.weights:
                # Gradient update: increase weights for features that led to wins
                adjustment = self.learningRate * (reward - 0.5) * feature_value
                self.weights[feature_name] += adjustment
                
                # Ensure weights stay in reasonable range
                self.weights[feature_name] = max(0.1, min(1.0, self.weights[feature_name]))
        
        self.trainingIterations += 1
    
    def batchUpdate(self, batch_size: int = 10):
        """
        Perform a batch update using sampled experiences.
        """
        if len(self.experience_buffer) < batch_size:
            return
            
        # Sample a batch of experiences
        indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in indices]
        
        # Update weights based on batch
        for features, reward in batch:
            for feature_name, feature_value in features.items():
                if feature_name in self.weights:
                    adjustment = self.learningRate * (reward - 0.5) * feature_value
                    self.weights[feature_name] += adjustment
                    self.weights[feature_name] = max(0.1, min(1.0, self.weights[feature_name]))
    
    def saveWeights(self, filename: str):
        """Save current weight values to a JSON file."""
        data = {
            'weights': self.weights,
            'training_iterations': self.trainingIterations,
            'win_rates': self.winRates
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
    def loadWeights(self, filename: str) -> bool:
        """Load weight values from a JSON file."""
        if not os.path.exists(filename):
            return False
            
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                self.weights = data.get('weights', self.weights)
                self.trainingIterations = data.get('training_iterations', 0)
                self.winRates = data.get('win_rates', {})
            return True
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading weights: {e}")
            return False
    
    def getFeatureImportance(self) -> List[Tuple[str, float]]:
        """Return feature importance as (feature_name, weight) tuples."""
        features = [(name, weight) for name, weight in self.weights.items()]
        return sorted(features, key=lambda x: x[1], reverse=True)
    
    def resetExperienceBuffer(self):
        """Clear the experience buffer."""
        self.experience_buffer = []