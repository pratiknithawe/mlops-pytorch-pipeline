import torch
from src.model import get_model

def test_model_output_shape():
    model=get_model("resnet18",10).eval()
    with torch.inference_mode(): out=model(torch.randn(2,3,32,32))
    assert out.shape==(2,10)

def test_model_probabilities_sum_to_one():
    model=get_model("resnet18",10).eval()
    with torch.inference_mode(): p=torch.softmax(model(torch.randn(1,3,32,32)),dim=1)
    assert p.shape==(1,10)
    assert torch.allclose(p.sum(1),torch.ones(1),atol=1e-5)
