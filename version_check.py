import torch
import speechbrain
import hyperpyyaml

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

print(speechbrain.__version__)
# print(speechbrain.version.cuda)
# print(speechbrain.cuda.is_available())

# print(hyperpyyaml.__version__)
# print(hyperpyyaml.version.cuda)
# print(hyperpyyaml.cuda.is_available())

# pip install typing-extensions --force-reinstall