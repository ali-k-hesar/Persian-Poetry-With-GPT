"""
Sample from a trained model
سمپل گرفتن از مدل
"""
import os
import sys
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT


# either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
# resume: $out_dir/ckpt.pt خروجی گرفتن از مدل آموزش داده شده داخل 
# gpt2-xl or gpt2-*: GPT2 برای خروجی گرفتن از مدل آموزش دیده ی 
init_from = 'resume' 
out_dir = 'out'
model_dir = sys.argv[1] if len(sys.argv)-1 else None
print(model_dir)
# or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
# شروع جمله برای ادامه داده شدن توسط مدل
start = "\n" 
# number of samples to draw
# تعداد سمپل برای خروجی
num_samples = 20 
# number of tokens generated in each sample
# حداکثر توکن تولیدی توسط مدل
max_new_tokens = 500 
# 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
# کنترل میزان رندوم بودن خروجی (بالاتر رندوم تر)
temperature = 1.0
# retain only the top_k most likely tokens, clamp others to have 0 probability
# انتخاب ایندکس ها با بیشترین احتمالات
top_k = 200 
seed = 1234
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
# use PyTorch 2.0 to compile the model to be faster
# کامپایل فقط پایتورچ بالاتر از 2
compile = False 
# exec(open('configurator.py').read()) # overrides from command line or config file

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# با 32 بیت به منظور کاهش نیاز به مموری و افزایش سرعت fp32 با 19 بیت برای محاسبه ی  tf32 استفاد از
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    # out_dir شروع مدل از وزن های آموزش دیده داخل مسیر
    ckpt_path = model_dir or os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    # GPT2 شروع از مدل آموزش دیده ی 
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

# look for the meta pickle in case it is available in the dataset folder
# خواندن متا فایل با فرمت پیکل برای گرفتن اطلاعات دیتاست
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    # استفاده میشود GPT Encoder به صورت پیش فرض از
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
# انکود کردن شروع جمله (توکن اولیه) یا پرامپت
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
# گرفتن خروجی ار مدل
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            # در ترمینال احتمالا حروف فارسی جدا از هم خواهند بود
            # در داخل یک تکست ادیتور کپی کنید
            print(decode(y[0].tolist()))
            print('\n---------------\n')
