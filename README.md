# Art Helper

- Attempt to use Deep Learning to help someone "improve" art and my DL skills (lol)

## Whats "new"?

- Firstly I am used mixed precision training aka fp16. [fp](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/)
- Second, I am using the new optimizer AdaBelief. [Ada](https://github.com/juntang-zhuang/Adabelief-Optimizer#Installation-and-usage)

## How?

- Firstly we try to identify what "good" art is. Since we go with the trend, lets just look at the popular instagram people. Some really good artists and try to get some web scraping going.
- Then we take art from not so popular(hi) artists and use that as the second classification
- If we can learn what "good" art is, we can use something like GradCam to show us which parts of the art could be potentially improved

## Why?

- One step closer to my goal of using something "smarter" than you to teach you
- ( Yes I know. I am indeed sleepy. )

## Externals

- I was going to write my own insta scraper but I found one which might work so why bother [Link](https://github.com/arc298/instagram-scraper  )
