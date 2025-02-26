
"""A small proof of concept of the idea os steering llms with neuron activations.

In this notebook, we download the hexby dataset, that consists of fmri taken while the subject was looking at images in different categories.

'rest' 'face' 'chair' 'scissors' 'shoe' 'scrambledpix' 'house' 'cat'
 'bottle'

 We keep only those for face and house.

 We also compute steering vectors along the positive-negative direction for each layer of gpt large.

 And we train a linear mapping between the fmri values and the steering vectors.
 We then evaluate if the mapping of images in the test set steer the model in the appropiate direction.
"""



import os
import pandas as pd
from nilearn import datasets
from nilearn.image import load_img
from nilearn.image import mean_img
from nilearn import plotting
from nilearn.maskers import NiftiMasker

# Get the data for subject 4, from the ventral temporal cortex.
data_dir = os.path.join('..', 'data')
sub_no = 4
haxby_dataset = datasets.fetch_haxby(subjects=[sub_no], fetch_stimuli=True, data_dir=data_dir)
func_file = haxby_dataset.func[0]

mask_filename = haxby_dataset.mask_vt[0]
masker = NiftiMasker(mask_img=mask_filename, standardize=True, detrend=True)
X = masker.fit_transform(func_file)

behavioral = pd.read_csv(haxby_dataset.session_target[0], delimiter=' ')
y = behavioral['labels']

from sklearn.model_selection import train_test_split
mask = y.isin(['house','face'])
X = X[mask]
y = y[mask]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape)
print(X_test.shape)

import matplotlib.pyplot as plt

from nilearn import datasets
from nilearn.plotting import show

stimulus_information = haxby_dataset.stimuli

for stim_type in ['houses','faces']:

    file_names = stimulus_information[stim_type]
    file_names = file_names[0:16]
    fig, axes = plt.subplots(4, 4)
    fig.suptitle(stim_type)
    for img_path, ax in zip(file_names, axes.ravel()):
        ax.imshow(plt.imread(img_path), cmap=plt.cm.gray)
    for ax in axes.ravel():
        ax.axis("off")

show()

positive_prompts = [
    "My house is fantastic, it makes me feel warm and fuzzy, I love it!",
    "How cool is to be able to see the beautiful beach from here?",
    "The sunrise this morning filled me with so much joy and energy!",
    "I can't believe how supportive my friends are, they always lift me up.",
    "This garden brings me peace every time I spend time here.",
    "Learning to play piano has been such a rewarding journey.",
    "The children's laughter in the park makes every day brighter.",
    "Nothing beats the feeling of accomplishment after finishing a project.",
    "The community came together to help, showing the best of humanity.",
    "Fresh coffee and a good book make for perfect mornings.",
    "The farmers market brings such fresh ingredients and friendly faces, delightful.",
"Our team pulled together and delivered the project ahead of schedule, remarkable.",
"The autumn colors transform the whole neighborhood into something magical.",
"Finding this hiking trail changed my weekends completely, spectacular.",
"My grandmother's recipes always bring back the sweetest memories, precious.",
"The home-cooked meal brought our whole family together, wonderful.",
"This new fitness routine has transformed my energy levels, incredible.",
"Finding this cozy cafe changed my work-from-home life, perfect.",
"The sunset reflecting off the lake takes my breath away, magnificent.",
"My garden's first tomatoes of the season taste amazing, delightful."
]


negative_prompts = [
    "My car is a disaster, I am scared every time I have to drive it, it makes me angry.",
    "This house has the worst view ever, when is not dirty and polluted, it is boring!",
    "The constant noise from construction is driving me crazy.",
    "Every time I try to fix something, it just gets worse.",
    "This weather is absolutely miserable, ruining all my plans.",
    "The service at this restaurant was terrible and the food was cold.",
    "My computer keeps crashing at the worst possible moments.",
    "The traffic on my commute is getting more unbearable each day.",
    "This neighborhood has gone downhill since I moved here.",
    "The new office layout makes it impossible to concentrate.",
    "The new management has completely destroyed team morale, devastating.",
"These recurring plumbing issues are costing me a fortune, frustrating.",
"The constant spam calls make me not want to answer my phone, infuriating.",
"This printer keeps jamming right before important deadlines, awful.",
"The neighbor's dog barks all night long, unbearable.",
"This new software update deleted all my saved work, disastrous.",
"The apartment's thin walls let me hear everything, maddening.",
"My favorite restaurant changed their recipe completely, disappointing.",
"This meeting could have been an email instead, wasteful.",
"The new phone's battery life barely lasts half a day, terrible.",
]

from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from nnsight import LanguageModel
import torch

model_name = "openai-community/gpt2-large"
model = LanguageModel(model_name, device_map='auto')
tokenizer =  AutoTokenizer.from_pretrained(model_name)


model.tokenizer.pad_token = model.tokenizer.eos_token
model.tokenizer.pad_token_id = model.tokenizer.eos_token_id

model

prompt = 'I went to see a movie and I thought that it was '
n_new_tokens = 3
with model.generate(prompt, max_new_tokens=n_new_tokens) as tracer:
    out = model.generator.output.save()

decoded_prompt = model.tokenizer.decode(out[0][0:-n_new_tokens].cpu())
decoded_answer = model.tokenizer.decode(out[0][-n_new_tokens:].cpu())

print("Prompt: ", decoded_prompt)
print("Generated Answer: ", decoded_answer)

def get_steering_vectors_gpt2(model, a_prompts, b_prompts, start=0):
    good_prompt_vectors = torch.zeros((35, 1280)) # num layers X hidden dimension
    bad_prompt_vectors = torch.zeros((35, 1280))

    with model.trace(a_prompts) as tracer:
        for layer in range(35):
            good_prompt_vectors[layer] = model.transformer.h[layer].output[0][:,start:].mean(dim = (0,1)).save()
            inp = model.transformer.h[layer].input.save()
    print(f"shape of inp = {inp.shape}")
    with model.trace(b_prompts) as tracer:
        for layer in range(35):
            bad_prompt_vectors[layer] = model.transformer.h[layer].output[0][:,start:].mean(dim = (0,1)).save()
            inp = model.transformer.h[layer].input.save()
    print(f"shape of inp = {inp.shape}")
    return good_prompt_vectors-bad_prompt_vectors

def generate_with_steering_gpt2(model, tokenizer, steering_directions, layers, max_tokens = 20, alpha = 0.5, initial_prompt = "I went to see the movie, and I think it is"):
    with model.generate(initial_prompt, max_new_tokens=25, pad_token_id = model.tokenizer.eos_token_id) as tracer:
        for i,l in enumerate(layers):
            model.transformer.h[l].all()
            model.transformer.h[l].input[0,-1,:] += steering_directions[i]*alpha
        out = model.generator.output.save()
    output = tokenizer.decode(out[0].cpu())
    return output

positive_negative_steering_vectors = get_steering_vectors_gpt2(model, positive_prompts, negative_prompts, start = -4)

print(f"shape of positive_negative_steering_vectors: {positive_negative_steering_vectors.shape}")
print(f"mean of positive_negative_steering_vectors: {positive_negative_steering_vectors.mean()}")


for layer in range(1,35,3):
    print(f"Layer {layer}")
    print(generate_with_steering_gpt2(model, tokenizer,[positive_negative_steering_vectors[layer]], [layer], alpha = -2.5))

layers = [21,23,28,31,34]
steering = [positive_negative_steering_vectors[i] for i in layers]
print(generate_with_steering_gpt2(model, tokenizer, steering, layers, alpha = -0.5))

from torch import nn
# Linear transformation
l = 32
gpt2_lt = nn.Linear(675,1280)


X_train = torch.tensor(X_train, dtype=torch.float32)
y = positive_negative_steering_vectors[l].repeat(X_train.size(0), 1)
y[y_train.values=="face"] = -y[y_train.values=="face"]
optimizer = torch.optim.Adam(gpt2_lt.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
gpt2_lt.train()
X_train = X_train.detach()
y = y.detach()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt2_lt.to(device)
X_train = X_train.to(device)
y = y.to(device)
for epoch in range(2_500):
    optimizer.zero_grad()
    output = gpt2_lt(X_train)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()
    if epoch % 25 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

import random
idx = random.randint(0, X_test.shape[0])
print(idx)
gpt2_lt.eval()
steering_vector = gpt2_lt(torch.tensor(X_test[idx]).to(device))
print(y_test.values[idx])

# If the image above is a house, it should be positive, if its a face, negative.
print(generate_with_steering_gpt2(model, tokenizer,[steering_vector], [l], alpha = 1.5))

