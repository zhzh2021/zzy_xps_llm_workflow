# XAS ML Modules: Design Guide for Developers

**Purpose:** Guidelines for building plug-and-play, maintainable, agent-friendly ML modules  
**Created:** March 3, 2026

---

## 🎯 Design Philosophy

### The "Plug-and-Play" Principle
Each module should be like a LEGO brick:
- ✅ Clear, standardized interface
- ✅ No hidden dependencies
- ✅ Works standalone
- ✅ Composable with other modules
- ✅ Easy to test in isolation
- ✅ Easy to replace/upgrade

### The "Configuration-First" Principle
**Never hardcode what a human might want to change**

```python
# ❌ BAD: Hardcoded threshold
def detect_outliers(data):
    threshold = 3.0  # What if user wants 2.5?
    ...

# ✅ GOOD: Config-driven
def detect_outliers(data, config=None):
    config = config or self.config
    threshold = config['threshold_sigma']  # From YAML
    ...
```

### The "Agent-Friendly" Principle
Agents can't interact with prompts or complex GUIs. Keep it simple:

```python
# ❌ BAD: Interactive
result = module.run()
print("How many clusters?")
n_clusters = int(input())  # Agent can't respond!

# ✅ GOOD: Direct
result = module.run(n_clusters="auto")  # Agent can call this
```

---

## 📋 Module Template (Copy This!)

```python
"""
xas_<module_name>.py

Description: [One sentence describing what this module does]

Configuration:
  - File: xas_config/xas_ml_settings.yaml
  - Section: [section_name]

Agent Usage Example:
  from zzy_llm.Tools.APS_XAS.xas_ml_modules import XASModuleName
  
  module = XASModuleName()  # Auto-loads config
  result = module.process(input_data)

Dependencies:
  - numpy, scipy, scikit-learn
  - XASDataset (from xas_models.py)
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml
import numpy as np
from pydantic import BaseModel, validator

# Import data models
try:
    from ..xas_analyzer.xas_models import XASDataset, XASFeatures
except ImportError:
    from xas_analyzer.xas_models import XASDataset, XASFeatures


class ModuleConfig(BaseModel):
    """Configuration schema for this module (Pydantic for validation)."""
    param1: float
    param2: str
    param3: Optional[int] = None
    
    @validator('param1')
    def param1_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('param1 must be positive')
        return v


class ModuleResult(BaseModel):
    """Output schema for this module."""
    result_data: Any
    metrics: Dict[str, float]
    flags: List[str]
    confidence: float
    
    class Config:
        arbitrary_types_allowed = True  # For numpy arrays


class XASModuleName:
    """
    [Module description]
    
    This module does X using method Y. It accepts Z as input and produces W.
    
    Attributes
    ----------
    config : ModuleConfig
        Configuration loaded from YAML
    logger : logging.Logger
        Module-specific logger
    
    Examples
    --------
    >>> module = XASModuleName()
    >>> result = module.process(data)
    >>> print(result.confidence)
    0.85
    """
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize module with configuration.
        
        Parameters
        ----------
        config_file : str or Path, optional
            Path to YAML config file. If None, uses default location.
        """
        # Step 1: Load configuration
        self.config = self._load_config(config_file)
        
        # Step 2: Set up logging
        self.logger = self._setup_logger()
        
        # Step 3: Initialize internal state (if needed)
        self._fitted = False
        self._model = None
        
        self.logger.info(f"{self.__class__.__name__} initialized")
    
    def _load_config(self, config_file: Optional[Union[str, Path]]) -> ModuleConfig:
        """Load and validate configuration from YAML."""
        if config_file is None:
            # Default: xas_config/xas_ml_settings.yaml
            config_file = Path(__file__).parent.parent / "xas_config" / "xas_ml_settings.yaml"
        
        config_file = Path(config_file)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            full_config = yaml.safe_load(f)
        
        # Extract this module's section
        module_config_dict = full_config.get('module_section', {})
        
        # Validate using Pydantic
        return ModuleConfig(**module_config_dict)
    
    def _setup_logger(self) -> logging.Logger:
        """Set up module-specific logger."""
        logger = logging.getLogger(self.__class__.__name__)
        
        # Don't set level here if already configured globally
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    # =========================================================================
    # PUBLIC API (Agent-Facing)
    # =========================================================================
    
    def process(self, 
                input_data: Any,
                **kwargs) -> ModuleResult:
        """
        Main processing function (AGENT ENTRY POINT).
        
        This is the primary function that agents will call. Keep the signature
        simple and intuitive.
        
        Parameters
        ----------
        input_data : Any
            Input data (type depends on module)
        **kwargs
            Optional overrides for config parameters
        
        Returns
        -------
        ModuleResult
            Structured output with result, metrics, flags, confidence
        
        Raises
        ------
        ValueError
            If input_data is invalid
        RuntimeError
            If processing fails
        
        Examples
        --------
        >>> module = XASModuleName()
        >>> result = module.process(my_data, param1=1.5)
        """
        self.logger.info("Starting processing...")
        
        try:
            # Step 1: Validate input
            self._validate_input(input_data)
            
            # Step 2: Merge kwargs with config
            params = self._merge_params(kwargs)
            
            # Step 3: Do the work
            result_data = self._internal_logic(input_data, params)
            
            # Step 4: Compute metrics
            metrics = self._compute_metrics(result_data, input_data)
            
            # Step 5: Run validation checks
            flags = self._validate_output(result_data, metrics)
            
            # Step 6: Compute confidence score
            confidence = self._compute_confidence(metrics, flags)
            
            # Step 7: Package output
            output = ModuleResult(
                result_data=result_data,
                metrics=metrics,
                flags=flags,
                confidence=confidence
            )
            
            self.logger.info(f"Processing complete (confidence: {confidence:.2f})")
            return output
            
        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}", exc_info=True)
            raise
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save fitted module state (if stateful).
        
        Parameters
        ----------
        filepath : str or Path
            Where to save the model
        """
        import pickle
        
        if not self._fitted:
            raise RuntimeError("Module not fitted yet. Call process() first.")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self._model, f)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: Union[str, Path]) -> None:
        """
        Load previously saved module state.
        
        Parameters
        ----------
        filepath : str or Path
            Path to saved model
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            self._model = pickle.load(f)
        
        self._fitted = True
        self.logger.info(f"Model loaded from {filepath}")
    
    # =========================================================================
    # INTERNAL METHODS (Private, for implementation)
    # =========================================================================
    
    def _validate_input(self, data: Any) -> None:
        """
        Validate input data.
        
        Raises
        ------
        ValueError
            If input is invalid
        """
        if data is None:
            raise ValueError("Input data cannot be None")
        
        # Add module-specific validation
        # Example: check array shape, check for NaNs, etc.
    
    def _merge_params(self, kwargs: Dict) -> Dict:
        """Merge runtime kwargs with config defaults."""
        params = self.config.dict()
        params.update(kwargs)
        return params
    
    def _internal_logic(self, data: Any, params: Dict) -> Any:
        """
        Core processing logic (IMPLEMENT THIS).
        
        This is where the actual work happens. Keep this function focused
        on the algorithm, not on I/O or validation.
        """
        # TODO: Implement your algorithm here
        raise NotImplementedError("Implement _internal_logic in subclass")
    
    def _compute_metrics(self, result: Any, input_data: Any) -> Dict[str, float]:
        """
        Compute quality metrics for the result.
        
        Returns
        -------
        dict
            Metrics like accuracy, error, etc.
        """
        return {
            'metric1': 0.0,
            'metric2': 0.0
        }
    
    def _validate_output(self, result: Any, metrics: Dict) -> List[str]:
        """
        Validate output and generate flags.
        
        Returns
        -------
        list of str
            Warning/error flags
        """
        flags = []
        
        # Example checks
        if metrics.get('metric1', 0) < 0.5:
            flags.append('low_quality_result')
        
        return flags
    
    def _compute_confidence(self, metrics: Dict, flags: List[str]) -> float:
        """
        Compute overall confidence score (0-1).
        
        Returns
        -------
        float
            Confidence score (0=no confidence, 1=high confidence)
        """
        # Start with base confidence
        confidence = 1.0
        
        # Penalize for flags
        confidence -= 0.1 * len(flags)
        
        # Adjust based on metrics
        # (module-specific logic here)
        
        return max(0.0, min(1.0, confidence))


# =============================================================================
# STANDALONE EXECUTION (for testing)
# =============================================================================

if __name__ == "__main__":
    # Example usage for testing
    import sys
    
    # Set up basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create module
    module = XASModuleName()
    
    # Test with dummy data
    test_data = np.random.randn(100, 10)
    
    try:
        result = module.process(test_data)
        print(f"Success! Confidence: {result.confidence:.2f}")
        print(f"Flags: {result.flags}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
```

---

## 🔧 Key Design Patterns

### 1. Configuration Loading Pattern

```python
# Standard config loading (use this everywhere)
from pathlib import Path
import yaml

def load_config(config_file=None, section=None):
    """Load config from YAML."""
    if config_file is None:
        config_file = Path(__file__).parent.parent / "xas_config" / "xas_ml_settings.yaml"
    
    with open(config_file, 'r') as f:
        full_config = yaml.safe_load(f)
    
    if section:
        return full_config.get(section, {})
    return full_config
```

### 2. Logging Pattern

```python
import logging

class MyModule:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def process(self, data):
        self.logger.info("Starting process")
        try:
            result = self._do_work(data)
            self.logger.info("Process complete")
            return result
        except Exception as e:
            self.logger.error(f"Process failed: {e}", exc_info=True)
            raise
```

### 3. Error Handling Pattern

```python
# Define module-specific exceptions
class XASMLError(Exception):
    """Base exception for XAS ML modules."""
    pass

class InsufficientDataError(XASMLError):
    """Not enough data for analysis."""
    pass

class ValidationError(XASMLError):
    """Output validation failed."""
    pass

# Use in modules
def cluster(data):
    if len(data) < 10:
        raise InsufficientDataError(
            f"Clustering requires ≥10 samples, got {len(data)}"
        )
```

### 4. Input/Output Validation Pattern

```python
from pydantic import BaseModel, validator

class MyInput(BaseModel):
    energy: List[float]
    mu: List[float]
    
    @validator('energy')
    def energy_must_be_monotonic(cls, v):
        if not all(v[i] < v[i+1] for i in range(len(v)-1)):
            raise ValueError("Energy must be monotonic")
        return v

class MyOutput(BaseModel):
    result: float
    confidence: float
    flags: List[str]
    
    @validator('confidence')
    def confidence_in_range(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Confidence must be in [0, 1]")
        return v
```

---

## 🧪 Testing Pattern

```python
# test_<module_name>.py

import pytest
import numpy as np
from xas_ml_modules import XASModuleName

@pytest.fixture
def sample_data():
    """Fixture for test data."""
    return np.random.randn(100, 10)

@pytest.fixture
def module():
    """Fixture for module instance."""
    return XASModuleName()

def test_module_initialization(module):
    """Test that module initializes correctly."""
    assert module.config is not None
    assert module.logger is not None

def test_process_returns_valid_output(module, sample_data):
    """Test that process returns valid result."""
    result = module.process(sample_data)
    
    assert hasattr(result, 'result_data')
    assert hasattr(result, 'metrics')
    assert hasattr(result, 'flags')
    assert 0 <= result.confidence <= 1

def test_process_with_invalid_input_raises_error(module):
    """Test that invalid input raises appropriate error."""
    with pytest.raises(ValueError):
        module.process(None)

def test_config_override(module, sample_data):
    """Test that config can be overridden at runtime."""
    result = module.process(sample_data, param1=999)
    # Check that param1 was actually used

def test_module_is_reproducible(module, sample_data):
    """Test that results are reproducible with same seed."""
    result1 = module.process(sample_data)
    result2 = module.process(sample_data)
    
    np.testing.assert_array_almost_equal(result1.result_data, result2.result_data)
```

---

## 📝 Documentation Pattern

Every module needs:

1. **Docstring at top of file** with agent usage example
2. **Class docstring** with description and attributes
3. **Method docstrings** with Parameters, Returns, Raises, Examples
4. **README.md** in module directory

```python
"""
xas_pca_analyzer.py

Performs Principal Component Analysis on XAS feature matrices.

Configuration:
  - File: xas_config/xas_ml_settings.yaml
  - Section: pca

Agent Usage Example:
  from zzy_llm.Tools.APS_XAS.xas_ml_modules import XASPCAAnalyzer
  
  analyzer = XASPCAAnalyzer()
  result = analyzer.fit_transform(dataset, n_components="auto")
  
  # Access results
  print(f"Variance explained: {result.explained_variance}")
  print(f"Number of components: {result.n_components}")

See Also:
  - xas_ml_integration_spec.md: Full specification
  - tests/test_pca_analyzer.py: Usage examples
"""
```

---

## ✅ Module Checklist

Before committing a new module, verify:

- [ ] Follows template structure
- [ ] Loads config from YAML (no hardcoded parameters)
- [ ] Has clear, simple agent-facing API
- [ ] Uses Pydantic models for inputs/outputs
- [ ] Logs at INFO level for key steps
- [ ] Logs at ERROR level with full traceback on failure
- [ ] Raises specific exceptions (not generic Exception)
- [ ] Has docstrings (file, class, all public methods)
- [ ] Has unit tests (≥80% coverage)
- [ ] Can run standalone (has `if __name__ == "__main__"` block)
- [ ] No interactive prompts (input(), click, etc.)
- [ ] No print statements (use logger instead)
- [ ] Returns Pydantic model (validates output)
- [ ] Can be imported without side effects
- [ ] Documentation includes agent usage example

---

## 🚫 Common Mistakes to Avoid

| ❌ Don't Do This | ✅ Do This Instead |
|-----------------|-------------------|
| Hardcode thresholds | Put in YAML config |
| Use `print()` | Use `logger.info()` |
| Use `input()` | Pass as function parameter |
| Return dict | Return Pydantic model |
| Raise `Exception` | Raise specific exception type |
| Silently fail | Log error and raise |
| Import with side effects | Import only defines classes/functions |
| Tight coupling | Depend on interfaces (Pydantic models) |
| Global state | Instance attributes |
| Implicit dependencies | Explicit imports |

---

## 📚 Example: Good vs. Bad Module

### ❌ BAD Example
```python
# Bad: too many issues
import numpy as np

# Global config (BAD!)
CONFIG = {'threshold': 3.0}

def cluster_data(data):
    # No logging
    # No input validation
    # Hardcoded parameters
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=4)  # Hardcoded!
    labels = kmeans.fit_predict(data)
    print(f"Clustering done")  # Use logger!
    return labels  # No structure!
```

### ✅ GOOD Example
```python
# Good: follows all patterns
import logging
from pathlib import Path
from typing import Optional
import yaml
import numpy as np
from pydantic import BaseModel
from sklearn.cluster import KMeans

class ClusteringConfig(BaseModel):
    n_clusters: int
    n_init: int = 10

class ClusteringResult(BaseModel):
    labels: np.ndarray
    confidence: float
    flags: list
    
    class Config:
        arbitrary_types_allowed = True

class XASClusterer:
    def __init__(self, config_file: Optional[Path] = None):
        self.config = self._load_config(config_file)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _load_config(self, config_file):
        if config_file is None:
            config_file = Path(__file__).parent.parent / "xas_config" / "xas_ml_settings.yaml"
        
        with open(config_file) as f:
            full_config = yaml.safe_load(f)
        
        return ClusteringConfig(**full_config['clustering']['kmeans'])
    
    def cluster(self, data: np.ndarray, **kwargs) -> ClusteringResult:
        self.logger.info("Starting clustering")
        
        params = {**self.config.dict(), **kwargs}
        
        kmeans = KMeans(n_clusters=params['n_clusters'], 
                       n_init=params['n_init'])
        labels = kmeans.fit_predict(data)
        
        confidence = self._compute_confidence(kmeans)
        flags = self._validate(labels)
        
        self.logger.info(f"Clustering complete (confidence: {confidence})")
        
        return ClusteringResult(
            labels=labels,
            confidence=confidence,
            flags=flags
        )
    
    def _compute_confidence(self, model):
        return 0.85  # Simplified
    
    def _validate(self, labels):
        return []  # No flags
```

---

**End of Design Guide**
