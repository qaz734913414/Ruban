## Ruban: Deep Learning for Captcha
![Demo](https://raw.githubusercontent.com/RubanSeven/Ruban/master/docs/images/demo.jpg)


------------------


## Getting started: 10 seconds to Ruban

Here is the example:
    sample_path is captcha data.

```python
from ruban.applications import InkFountain

ink_fountain = InkFountain(sample_path=r'InkFountainData', lr=0.001)
```

train as easy as `.train()`:

```python
ink_fountain.train()
```

Once your model trained, use it as easy as `.test()`:

```python
ink_fountain.test()
```


------------------


## Donate
![Donate](https://raw.githubusercontent.com/RubanSeven/Ruban/master/docs/images/demo.jpg)
