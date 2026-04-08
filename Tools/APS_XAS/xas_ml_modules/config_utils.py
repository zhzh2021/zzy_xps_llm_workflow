"""
Configuration Utilities

Centralized configuration loading and validation for XAS ML modules.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml


class ConfigurationError(Exception):
    """Configuration loading or validation error."""
    pass


class ConfigLoader:
    """
    Centralized configuration loader for XAS ML modules.
    
    All modules should use this to load configuration from YAML.
    Provides caching and validation.
    
    Examples
    --------
    >>> loader = ConfigLoader()
    >>> pca_config = loader.get_section('pca')
    >>> print(pca_config['variance_threshold'])
    0.95
    """
    
    _instance = None
    _config_cache = None
    _config_file = None
    
    def __new__(cls, config_file: Optional[Union[str, Path]] = None):
        """Singleton pattern - only one ConfigLoader instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize configuration loader.
        
        Parameters
        ----------
        config_file : str or Path, optional
            Path to YAML config file. If None, uses default location.
        """
        # Only initialize once
        if self._config_cache is not None and config_file is None:
            return
        
        if config_file is None:
            # Default location
            config_file = Path(__file__).parent.parent / "xas_config" / "xas_ml_settings.yaml"
        
        self._config_file = Path(config_file)
        self._load_config()
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Configuration loaded from {self._config_file}")
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if not self._config_file.exists():
            raise ConfigurationError(
                f"Configuration file not found: {self._config_file}\n"
                f"Expected location: xas_config/xas_ml_settings.yaml"
            )
        
        try:
            with open(self._config_file, 'r') as f:
                self._config_cache = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Failed to parse YAML config: {e}")
        
        if self._config_cache is None:
            raise ConfigurationError("Config file is empty")
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get complete configuration dictionary.
        
        Returns
        -------
        dict
            Full configuration
        """
        return self._config_cache.copy()
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get a specific configuration section.
        
        Parameters
        ----------
        section : str
            Section name (e.g., 'pca', 'clustering', 'feature_extraction')
        
        Returns
        -------
        dict
            Section configuration
        
        Raises
        ------
        ConfigurationError
            If section does not exist
        
        Examples
        --------
        >>> loader = ConfigLoader()
        >>> pca_cfg = loader.get_section('pca')
        >>> print(pca_cfg['variance_threshold'])
        0.95
        """
        if section not in self._config_cache:
            available_sections = list(self._config_cache.keys())
            raise ConfigurationError(
                f"Configuration section '{section}' not found.\n"
                f"Available sections: {available_sections}"
            )
        
        return self._config_cache[section].copy()
    
    def get_param(self, section: str, param: str, default: Any = None) -> Any:
        """
        Get a specific parameter from a section.
        
        Parameters
        ----------
        section : str
            Section name
        param : str
            Parameter name (supports nested with dot notation, e.g., 'kmeans.n_init')
        default : any, optional
            Default value if parameter not found
        
        Returns
        -------
        any
            Parameter value or default
        
        Examples
        --------
        >>> loader = ConfigLoader()
        >>> k_range = loader.get_param('clustering', 'kmeans.k_range')
        >>> print(k_range)
        [2, 10]
        """
        try:
            section_config = self.get_section(section)
        except ConfigurationError:
            return default
        
        # Support nested parameters with dot notation
        keys = param.split('.')
        value = section_config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def reload(self) -> None:
        """Reload configuration from file (useful for testing or runtime changes)."""
        self._load_config()
        self.logger.info("Configuration reloaded")
    
    def get_logging_config(self) -> Dict[str, Any]:
        """
        Get logging configuration.
        
        Returns
        -------
        dict
            Logging settings
        """
        return self.get_section('logging')
    
    def setup_logging(self) -> None:
        """
        Set up logging based on configuration.
        
        This configures the root logger and module-specific loggers.
        Call this once at the start of a workflow.
        """
        log_config = self.get_logging_config()
        
        # Configure root logger
        log_level = log_config.get('level', 'INFO')
        log_format = log_config.get('log_format', 
                                     '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        date_format = log_config.get('date_format', '%Y-%m-%d %H:%M:%S')
        
        # Set up handlers
        handlers = []
        
        # Console handler
        if log_config.get('log_to_console', True):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(log_format, date_format))
            handlers.append(console_handler)
        
        # File handler
        if log_config.get('log_to_file', True):
            log_file = log_config.get('log_file', 'xas_ml_analysis.log')
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(log_format, date_format))
            handlers.append(file_handler)
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format,
            datefmt=date_format,
            handlers=handlers,
            force=True  # Override any existing configuration
        )
        
        # Set module-specific levels
        module_levels = log_config.get('module_levels', {})
        for module_name, level in module_levels.items():
            logger = logging.getLogger(module_name)
            logger.setLevel(getattr(logging, level))
        
        logging.info("Logging configured from YAML settings")


def load_config(config_file: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Convenience function to load full configuration.
    
    Parameters
    ----------
    config_file : str or Path, optional
        Path to config file
    
    Returns
    -------
    dict
        Full configuration dictionary
    
    Examples
    --------
    >>> config = load_config()
    >>> print(config['pca']['variance_threshold'])
    0.95
    """
    loader = ConfigLoader(config_file)
    return loader.get_all()


def get_section(section: str, config_file: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Convenience function to get a config section.
    
    Parameters
    ----------
    section : str
        Section name
    config_file : str or Path, optional
        Path to config file
    
    Returns
    -------
    dict
        Section configuration
    
    Examples
    --------
    >>> pca_config = get_section('pca')
    >>> print(pca_config['variance_threshold'])
    0.95
    """
    loader = ConfigLoader(config_file)
    return loader.get_section(section)


if __name__ == "__main__":
    # Test configuration loading
    import sys
    
    try:
        loader = ConfigLoader()
        print("✓ Configuration loaded successfully")
        print(f"  Config file: {loader._config_file}")
        print(f"  Available sections: {list(loader.get_all().keys())}")
        
        # Test getting sections
        pca_config = loader.get_section('pca')
        print(f"✓ PCA config loaded: {len(pca_config)} parameters")
        
        # Test nested parameter access
        k_range = loader.get_param('clustering', 'kmeans.k_range')
        print(f"✓ Nested parameter access: kmeans.k_range = {k_range}")
        
        # Test logging setup
        loader.setup_logging()
        logger = logging.getLogger("TestModule")
        logger.info("Test log message")
        print("✓ Logging configured successfully")
        
        print("\n✅ All configuration tests passed!")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        sys.exit(1)
