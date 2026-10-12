# Improving a CNN on CIFAR-10 under a fixed architecture

Image classification on **CIFAR-10** (60,000 32×32 colour images, 10 classes) in **PyTorch**.

The challenge: the core model design was fixed — a sequence of required *intermediate blocks* followed by an output block — so improvements had to come from careful, controlled experimentation rather than swapping in a known architecture such as ResNet.

## Result

| Model | Best test accuracy |
|---|---|
| Baseline (required basic architecture) | 26.72% |
| Final model (same block structure) | **88.72%** (epoch 55 of 60) |

Final training accuracy (89.71%) stayed close to test accuracy, showing that the model was not overfitting.

## Approach

Controlled screening experiments, changing one factor at a time on a fixed data subset and ranking configurations by mean test accuracy over the final three epochs:

- Block architecture, kernel size, depth and width
- Activation and gate activation functions
- Pooling, batch normalisation
- Learning rate and cosine learning-rate scheduling
- Data augmentation: standard augmentation vs **AutoAugment** vs **Cutout**

The selected configuration was then trained on the full dataset for 60 epochs with AutoAugment, which gave the strongest regularisation over longer training.

## Error analysis

Most classes exceeded 93% accuracy (automobile, ship, horse, truck). The hardest were **cat (75.6%)** and **dog (79.8%)**, which the confusion matrix shows were most often mistaken for each other.

## Run it

Open `cifar10_cnn.ipynb` in Google Colab (GPU runtime recommended). CIFAR-10 downloads automatically via `torchvision`.

```
pip install -r requirements.txt
```

## Tools

Python · PyTorch · torchvision · NumPy · Matplotlib

*Completed as part of the MSc Artificial Intelligence at Queen Mary University of London.*
