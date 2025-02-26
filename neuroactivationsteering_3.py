import os
import pandas as pd
from nilearn import datasets
from nilearn.image import load_img
from nilearn.image import mean_img
from nilearn import plotting
from nilearn.maskers import NiftiMasker

data_dir = os.path.join('..', 'data')
sub_no = 4
haxby_dataset = datasets.fetch_haxby(subjects=[sub_no], fetch_stimuli=True, data_dir=data_dir)
func_file = haxby_dataset.func[0]

mask_filename = haxby_dataset.mask_vt[0]
masker = NiftiMasker(mask_img=mask_filename, standardize=True, detrend=True)
X = masker.fit_transform(func_file)

behavioral = pd.read_csv(haxby_dataset.session_target[0], delimiter=' ')
y = behavioral['labels']

print(y.unique())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape)
print(X_test.shape)

import matplotlib.pyplot as plt

from nilearn import datasets
from nilearn.plotting import show

stimulus_information = haxby_dataset.stimuli

for stim_type in ['houses','faces','chairs','shoes', 'bottles', 'cats', 'scissors']:

    file_names = stimulus_information[stim_type]
    file_names = file_names[0:16]
    fig, axes = plt.subplots(4, 4)
    fig.suptitle(stim_type)
    for img_path, ax in zip(file_names, axes.ravel()):
        ax.imshow(plt.imread(img_path), cmap=plt.cm.gray)
    for ax in axes.ravel():
        ax.axis("off")

show()

face_prompts = [
   "The mirror showed a familiar face",
   "The mirror showed a familiar face.",
   "The painting captured her gentle face",
   "The mask concealed the mysterious face",
   "A smile lit up his friendly countenance",
   "Wrinkles etched a story on her visage",
   "Time had weathered his ancient face",
"Makeup transformed her radiant face",
"Shadows played across his stern visage",
"The photo captured his noble countenance",
"Joy brightened her expressive face",
"The mirror reflected a weary face",
"face",
"Face",

]

chair_prompts = [
   "Grandmother's favorite spot was her chair",
   "The most comfortable place is this chair",
   "The most comfortable place is this chair.",
   "The antique wooden rocking chair",
   "He sank into the plush armchair",
   "She perched on the edge of the stool"
   "The executive sat in his throne",
"She relaxed in the velvet seat",
"Light gleamed off the metal chair",
"Children gathered around the bench",
"The garden held an iron seat",
"Time had worn the wooden chair",
"chair",
   "Chair",
]

scissors_prompts = [
   "The paper was cut with sharp scissors",
   "The paper was cut with sharp scissors!",
   "The paper was cut with sharp scissors.",
   "The hairdresser wielded golden scissors",
   "She carefully handled the craft scissors",
   "The tailor used shears to cut the fabric",
   "He snipped the flowers with pruning clippers"
   "She mended clothes with rusty shears",
"Art class required special scissors",
"Metal glinted on the steel blades",
"The seamstress used silver scissors",
"Children practiced with safe scissors",
"scissors",
   "Scissors",
]

shoe_prompts = [
   "Walking miles in this comfortable shoe",
   "Dancing all night in her favorite shoe",
   "The child outgrew another leather shoe",
   "He laced up his new running sneakers",
   "She slipped on her elegant high heels"
   "Rain soaked through his old boots",
"The display showed designer footwear",
"Mud caked on his walking shoes",
"She polished her leather boots",
"Sand filled his beach sandals",
   "shoe",
   "Shoe",
]

house_prompts = [
   "The simplest drawing is a house",
   "They finally found their perfect house",
   "The family built their dream house",
   "The old cottage stood by the lake",
   "Their apartment was on the top floor"
   "Vines covered the stone dwelling",
"Snow blanketed the cozy home",
"Trees surrounded the rustic cabin",
"Stars shone above their mansion",
"Flowers adorned the small cottage",
"I wish to finish the day and head home",
   "house",
   "House",
]

cat_prompts = [
   "Chasing the laser pointer was the cat",
   "Sleeping in the sunbeam lay the cat",
   "Sleeping in the sunbeam lay the cat!",
   "Sleeping in the sunbeam lay the cat.",
   "Purring on the windowsill sat the cat",
   "Close to my leg I found the sneaky feline",
   "Under the bed hid the cat",
"Through the garden stalked the feline",
"Birds attracted the curious cat",
"Milk tempted the hungry kitten",
"Fish interested the watching cat",
"Yarn entertained the playful feline",
   "cat",
   "Cat",
]

bottle_prompts = [
   "Pour the water from the glass bottle",
   "The message was inside the bottle",
   "She recycled the empty plastic bottle",
   "He uncorked the vintage wine flask",
   "The perfume came in a delicate vial",
   "Light shone through the crystal vessel",
"Rain filled the copper container",
"Tea steeped in the ceramic flask",
"Soda fizzed in the plastic bottle",
"Bottle",
"bottle",
]

#google/gemma-2-2b
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from nnsight import LanguageModel
import torch

model_name = "google/gemma-2-2b"
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

def get_averages_gemma(model, prompts):
    avg_activation = torch.zeros((25, 2304)) # num layers X hidden dimension
    key_activation = torch.zeros((25, 2304)) # num layers X hidden dimension
    with model.trace(prompts) as tracer:
        for layer in range(25):
            avg_activation[layer] = model.model.layers[layer].output[0].mean(dim = (0,1)).save()
            key_activation[layer] = model.model.layers[layer].output[0][:,-2:].mean(dim = (0,1)).save()
    return avg_activation, key_activation

def generate_with_steering_gemma(model, steering_directions, layers, max_tokens = 20, alpha = 0.5, initial_prompt = "I went to see the movie, and I think it is"):
    with model.generate(initial_prompt, max_new_tokens=max_tokens, do_sample = True) as tracer:
        for i,l in enumerate(layers):
            model.model.layers[l].all()
            model.model.layers[l].input += steering_directions[i]*alpha
        out = model.generator.output.save()
    output = model.tokenizer.decode(out[0].cpu())
    return output


def generate_with_steering_gpt2(model, tokenizer, steering_directions, layers, max_tokens = 20, alpha = 0.5, initial_prompt = "I went to see the movie, and I think it is"):
    with model.generate(initial_prompt, max_new_tokens=25, pad_token_id = model.tokenizer.eos_token_id, temperature=1.1, do_sample=True) as tracer:
        for i,l in enumerate(layers):
            model.transformer.h[l].all()
            model.transformer.h[l].input += steering_directions[i]*alpha
        out = model.generator.output.save()
    output = tokenizer.decode(out[0].cpu())
    return output

shoe_avg, shoe_key = get_averages_gemma(model, shoe_prompts)
print(f"Got the shoe")
chair_avg, chair_key = get_averages_gemma(model, chair_prompts)
print(f"Got the chair")
house_avg, house_key = get_averages_gemma(model, house_prompts)
print("Got the house")
face_avg, face_key = get_averages_gemma(model, face_prompts)
print("Got the face")
bottle_avg, bottle_key = get_averages_gemma(model, bottle_prompts)
print("Got the bottle")
cat_avg, cat_key = get_averages_gemma(model, cat_prompts)
print("Got the cat")
scissors_avg, scissors_key = get_averages_gemma(model, scissors_prompts)
print("Got the scissors")


total_avg = torch.stack([shoe_avg, chair_avg, house_avg, face_avg, bottle_avg, cat_avg, scissors_avg], dim=0).mean(dim=0)
print(f"total_avg shape is {total_avg.shape}")
#

shoe_steering = shoe_key-total_avg
chair_steering = chair_key-total_avg
house_steering = house_key-total_avg
face_steering = face_key-total_avg
bottle_steering = bottle_key-total_avg
cat_steering = cat_key-total_avg
scissors_steering = scissors_key-total_avg

with model.generate("I am Leonidas") as tracer:
    inp = model.model.layers[2].input.save()
    out = model.generator.output.save()
print(model.tokenizer.decode(out[0].cpu()))
print(inp.shape)

print(f"mean of house steering: {house_steering.mean()}, shape: {house_steering.shape}")
print(f"mean of face steering: {face_steering.mean()}, shape: {face_steering.shape}")
print(f"mean of chair steering: {chair_steering.mean()}")
print(f"mean of shoe steering: {shoe_steering.mean()}, shape: {shoe_steering.shape}")
print(f"mean of bottle steering: {bottle_steering.mean()}")
print(f"mean of cat steering: {cat_steering.mean()}")
print(f"mean of scissors steering: {scissors_steering.mean()}")

for layer in range(2,22,2):
    print(f"Layer {layer}")
    print(generate_with_steering_gemma(model, [shoe_steering[layer]], [layer], alpha = 0.85, initial_prompt = "If I could by anything, I would"))

for layer in range(1,35,1):
    print(f"Layer {layer}")
    steering = scissors_steering[layer]
    alpha = 0.9
    print(generate_with_steering_gemma(model, tokenizer,[steering], [layer], alpha = alpha, initial_prompt = "If I could buy anything, I would "))
    print(generate_with_steering_gemma(model, tokenizer,[steering], [layer], alpha = alpha, initial_prompt = "Here is a picture of a"))

layers = [2,6,14,19,21]
steering = [face_steering[i] for i in layers]
print(generate_with_steering_gemma(model, steering, layers, alpha = 0.25, initial_prompt = "The picture is a"))

from torch import nn
import torch.optim.lr_scheduler as lr_scheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train_linear_model(l, epochs = 13000):
    gpt2_lt = nn.Linear(675,2304)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y = 0.1*total_avg[l].repeat(X_t.size(0), 1)
    print(f"shape of X_train: {X_t.shape}")
    print(f"shape of y: {y.shape}")
    y[y_train.values=="face"] = face_steering[l]
    y[y_train.values=="house"] = house_steering[l]
    y[y_train.values=="chair"] = chair_steering[l]
    y[y_train.values=="shoe"] = shoe_steering[l]
    y[y_train.values=="bottle"] = bottle_steering[l]
    y[y_train.values=="cat"] = cat_steering[l]
    y[y_train.values=="scissors"] = scissors_steering[l]

    optimizer = torch.optim.Adam(gpt2_lt.parameters(), lr=0.01)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.7)  # Halves LR every 10000 epochs

    loss_fn = nn.MSELoss()
    gpt2_lt.train()
    X_t = X_t.detach()
    y = y.detach()
    gpt2_lt.to(device)
    X_t = X_t.to(device)
    y = y.to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = gpt2_lt(X_t)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 2500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}, LR: {scheduler.get_last_lr()[0]}")
    gpt2_lt.eval()
    return gpt2_lt

layers = [2,8,16,22]
linear_models = [train_linear_model(l) for l in layers]

import random
idx = random.randint(0, X_test.shape[0]-1)
print(idx)
steering_vectors = [model(torch.tensor(X_test[idx]).to(device)) for model in linear_models]
print(steering_vectors)
print(y_test.values[idx])

# If the image is rest (no image) or a scrambled image it should change the activation much.
# If it is one of the other categories it should steer the model in that direction.
for alpha in [0.1, 0.2, 0.3, 0.4, 0.5]:
    print(f"alpha = {alpha}")
    print("---------------------------------")
    print(generate_with_steering_gemma(model, steering_vectors, layers, alpha = alpha, initial_prompt = "The picture is a"))
    print("---------------------------------")
    print(generate_with_steering_gemma(model, steering_vectors, layers, alpha = alpha, initial_prompt = "One word I find difficult to pronounce is"))
    print("---------------------------------")
    print(generate_with_steering_gemma(model, steering_vectors, layers, alpha = alpha, initial_prompt = "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a "))
    print("---------------------------------")
    print(generate_with_steering_gemma(model, steering_vectors, layers, alpha = alpha, initial_prompt = "We went to the cinema to see a movie about"))
    print("---------------------------------")
    print()

