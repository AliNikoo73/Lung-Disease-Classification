"""
Tests for the model module.
"""

import pytest
import numpy as np
from src.model import LungDiseaseModel

def test_model_initialization():
    """Test model initialization with different architectures."""
    model = LungDiseaseModel(model_name='densenet')
    assert model.model is not None
    assert model.num_classes == 5
    
    model = LungDiseaseModel(model_name='resnet')
    assert model.model is not None
    
    model = LungDiseaseModel(model_name='vgg')
    assert model.model is not None
    
def test_invalid_model_name():
    """Test initialization with invalid model name."""
    with pytest.raises(ValueError):
        LungDiseaseModel(model_name='invalid_model')
        
def test_model_output_shape():
    """Test model output shape."""
    model = LungDiseaseModel()
    dummy_input = np.random.random((1, 224, 224, 3))
    output = model.model.predict(dummy_input)
    assert output.shape == (1, 5)  # Batch size of 1, 5 classes 