
# Persian Poetry With GPT

based on [nanoGPT](https://github.com/karpathy/nanoGPT.git) by andrej karpathy for training on Persian Poetry.

کدنویسی، آموزش و خروجی گرفتن از یک مدل ترنسفورمر (مدل زبانی)  با آموزش بر روی اشعار فارسی. این مدل به صورت کاراکتر لول (حروف به عنوان توکن های ورودی) آموزش میبیند.
کامنت ها به فارسی نیز قرار گرفته است.

## آموزش بر روی اشعار سهراب سپهری

مراحل اولیه آموزش
```
تي انيد.      
كه خسيد.      
چه غرصر شا شست
كفر غرخت دوري 
تابت شد       
سربتم ، سپينيد
تجريمد افل
بالوه حافرق سافك پرد
تجغرب تجربت
گي يخاشد.
تكيرب د.
پشت سيكلا
تا
```

مراحل میانی آموزش 
```
تي انگاهي در نقش است .
صداي پرده نمي آمد، ستايان چو گرده بامي شدم، و شاخه مي شنوم .
بر لب آن سوسته يمي كش ، تپستگان صدايي مي رود .
در باين ، شاخه ني بارت پاسختم مي شوي: 
به آهستگ ، شتاب ما .
در كفتار ب لب ود.مي جن تو زيبام تندم زدم .
نه
مي بادي بودعلف ها را مي شنود .
دستم من باد كاغذه تمي بودم .
در خواب افودم را پر از ين صدا مي رقدند .
نهمي در ، مي شنوم بياباد كنم عبگرد به ام آمد، در حوض "من گاو قي برد .
در باين سبم ، صداي روزي و پاي مي ردن وشنيا"! هاي آن سوي زار نامن گذشت .
شبني تنها صدايي پر 
```

مراحل انتهایی آموزش

اوورفیت شده و بسیار نزدیک به اشعار سهراب
(به روز میشود)
```
سایه را بدزديم و سپوري كه به يك پوسته خربزه مي برد نماز.
الاغی ديدم ، بادبادك مي خورد.
من الاغي ديدم، ينجه را مي فهميد.
در چراگاه نصيحت بره ای ديدم سير.
شاعري ديدم هنگام خطاب
من كتابي ديدم ، واژه هايش همه از جنس بلور.

كاغذي ديدم ، از جنس بهار،
موزه اي ديدم دور از سبزه،
مسجدي دور از آب.
سر بالين فقهي نوميد، كوزه اي ديدم لبريز سوال.
قاطري ديدم بارش "انشا"
اشتري ديدم بارش سبد خالي " پند و امثال."
```

## نصب کتابخانه های مورد نیاز

```
pip install -r requirements.txt
```

## ساخت دیتاست

میتوانید اسم های دیتاست ها در پوشه ی دیتا مشاهده کنید.

akhavan sohrab دیتاست فارسی

openwebtext shakespeare shakespeare_char دیتاست انگلیسی

```
$ python data/sohrab/prepare.py
```

## آموزش

حتما پارامتر های موجود در کد را چک کنید و در صورا نیاز تغییر دهید

```
$ python train.py
```

## گرفتن خروجی 

این کد تعدادی سمپل با توجه به مدل آموزش دیده خروجی میگیرد.
خروجی پرینت شده در ترمینال ، حروف فارسی جدا از هم خواد بود.
بهتر است در یک تکست ادیتور نتایج را کپی کنید.

```
$ python sample.py out/ckpt.pt
```

