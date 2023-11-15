import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch

from model import GPTConfig, GPT

out_dir = 'out'
eval_interval = 500
log_interval = 1
eval_iters = 100

# if True, script exits right after the first eval 
# اگر "ترو" گذاشته شود اسکریپت بلافاصله پس از اولین ارزیابی خارج می شود بدون آموزش مدل
eval_only = False 
# if True, always save a checkpoint after each eval
# اگر "ترو" گذاشته شود همیشه بعد از هر ارزیابی یک چکپوینت (وزن های مدل) ذخیره میشود
always_save_checkpoint = True
init_from = 'resume' # 'scratch' or 'resume' or 'gpt2*'
# data
dataset = 'sohrab' # Farsi dataset
# used to simulate larger batch sizes
# شبیه سازی بچ بزرگتر
gradient_accumulation_steps = 10 
batch_size = 10
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
# for pretraining 0 is good, for finetuning try 0.1+
# برای آموزش از ابتدا 0، برای فاین تون مدل های آموزش دیده 0.1+ (دیتای کمتر احتمال اوور فیت)
dropout = 0.15 
# do we use bias inside LayerNorm and Linear layers?
# لایه های فولی کانکتد + نورمالیزیشن، پارامتر بایاس داشته باشند یا خیر؟
bias = False 

# adamw optimizer
# برای استفاده دقیق تر از ویت دیکی در پیاده سازی مشتق (Adam for weight decay)
max_lr = 6e-4
# total number of training iterations 
# بعد از این میزان ایتریشن (روی بچ ها) آموزش متوقف میشود
max_iters = 100000 
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
# clip gradients at this value, or disable if == 0.0
# کلیپ کردن و بردین مشتق های بزرگ
# کل پارامترها در داخل یک وکتور فلت شده و اندازه اقلیدسی وکتور اگر بیشتر از این عدد باشد بریده میشود
grad_clip = 1.0 

# learning rate decay settings
# whether to decay the learning rate
# کاهش لرنینگ ریت در طول آموزش؟
decay_lr = True 
warmup_iters = 2000 
# should be ~= max_iters per Chinchilla
# باید برابر حداکثر تعداد استپ اس آموزش باشد Chinchilla بر اساس مقاله
lr_decay_iters = 100000 
# minimum learning rate, should be ~= max_lr/10 per Chinchilla
# بهتر است یک دهم حاکثر لرتنینگ ریت باشد Chinchilla بر اساس مقاله
min_lr = 6e-5

# system
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

# for compile torch version should be >= 2 and also windows not supported yet (os.name != "nt")
# برای کامپایل نسخه تورچ باید بزرگتر مساوی 2 باشد و همچنین ویندوز هنوز پشتیبانی نمی شود
compile = torch.__version__[0] == '2' and os.name != "nt"

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys} # will be useful for logging

tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)

# با 32 بیت به منظور کاهش نیاز به مموری و افزایش سرعت fp32 با 19 بیت برای محاسبه ی  tf32 استفاد از
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
print(f"device_type: {device}")
# note: float16 data type will automatically use a GradScaler
# استفاده کنیم به دلیل رنج (اکسپوننت) کم نیاز به اسکیل لاس و مشتق ها ← برای جلوگیری از آندرفلو fp16 اگر از
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        # به ما امکان می دهد دیتا را به "جی پی یو" به صورت موازی منتقل کنیم (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
# اگر از مدل های از قبل آموزش داده استفاده کنید خود به خود اطلاعات روی این پارامترها نوشته میشوند
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
# گرفتن اندازه دیکشنری از دیتاست برای ساخت لایه امبدینگ و لایه ی انتهایی مدل
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    # ساخت مدل از ابتدا
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    # اگر از مدل از پیش تعریف شده استفاده کنیم همه ی پارامتر ها
    # (تعداد لایه، تعداد هد ها و ..) از  پیش برای آن مدل تعیین شده اند  
    # قابل تنظیم خواهد بود dropout تنها اندازه
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # ساخت مدل بر پایه ی چکپوینت های قبلی
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # اصلاح پیوند اضافی که گاهی در زمان سیو کردن مدل تورچ اضافه میشود
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    # آموزش دیده GPT2 ساخت مدل از روی مدل  
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
# بریدن اندازه کانتکس در صورت نیاز (برای ماتریکس امبدینگ پوزیشن)
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
# و غیر فعال در حالت های دیگر fp16 فقط برای 
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, max_lr, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)


# helps estimate an arbitrarily accurate loss over either split using many batches
# لاس تخمینی بر اساس ارزیابی چند بچ رندوم دیتا به جای کل دیتا
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
# فانکشن لرنینگ ریت ، کاهشی در طول آموزش بر اساس کوسینوس به همراه چند مرحله گرم کردن مشتق ها در ابتدا
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    # مراحل گرم کردن مشتق ، اگر تعداد بچ ها کم باشد مشتق ها قابل اطمینان نیستند 
    # بعد از چند مرحله آموزش به اندازه کافی مشتف خواهیم داشت
    if it < warmup_iters:
        return max_lr * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    # در انتهای آموزش از مینیموم لرنینگ ریت استفاده میشود
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    # بین دو مرحله قبلی، به صورت کوسینوسی لرنینگ ریت کاهش می یابد
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (max_lr - min_lr)

# training loop
# fetch the very first batch
# گرفتن اولین بچ دیتا
X, Y = get_batch('train') 
t0 = time.time()
# number of iterations in the lifetime of this process
# مدل mfu فقط برای محاسبه
local_iter_num = 0
running_mfu = -1.0

while True:
    # determine and set the learning rate for this iteration
    # تعیین لرنینگ ریت برای این مرحله از آموزش
    lr = get_lr(iter_num) if decay_lr else max_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    # سنجش مدل در استپ های مشخص
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    # جمع کردن مشتق ها و سپس اعمال آنها ، مناسب برای بزرگ کردن اندازه بچ در زمان کم بودن مموری
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        # همزمان که "جی پی یو" در حال مرحله فوروارد است "سی پی یو" دیتا را برای مرحله بعد آماده میکند
        X, Y = get_batch('train')
        scaler.scale(loss).backward()
    # clip the gradient
    # بردن مشتق ها
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    # اعمال مشتق ها
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    # حذف مشتق های اعمال شده در اولین زمان ممکن
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    # اندازه گیری زمان
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        # get loss as float. note: this is a CPU-GPU sync point
        # نقطه ی به هم رسیدن "جی پی یو" و "سی پی یو" ، عدد لاس از "جی پی یو" بر روی " سی پی یو" منتقل میشود" 
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    # پایان آموزش بعد از گذشتن از حداکثر استپ های تعیین شده
    if iter_num > max_iters:
        break
