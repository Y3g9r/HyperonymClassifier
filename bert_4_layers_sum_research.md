# Исследование обучения модели, в которой эмбединги представляются суммой с последних 4 слоёв

## Опыт 1

learning-rate: 0.0001

Число эпох: 24

Размер данных: 1600
Размер батча: 4

Fine-tuning:
self.example_linear_1 = torch.nn.Linear(1024, 1024)  
self.example_linear_2 = torch.nn.Linear(1024, 1024)  

self.def_linear_1 = torch.nn.Linear(1024, 1024)  
self.def_linear_2 = torch.nn.Linear(1024, 1024)  

self.Linear = torch.nn.Linear(1024, 1)  
self.tanh = torch.nn.Tanh()  
self.sigm = torch.nn.Sigmoid()  

![](./learning_research/exp_1.png)

## Опыт 2

learning-rate: 0.0001

Число эпох: 24

Размер данных: 1600
Размер батча: 4


Fine-tuning:
self.example_linear_1 = torch.nn.Linear(1024, 32)
self.example_linear_2 = torch.nn.Linear(32, 1)

self.def_linear_1 = torch.nn.Linear(1024, 32)
self.def_linear_2 = torch.nn.Linear(32, 1)

self.Linear = torch.nn.Linear(1, 1)
self.tanh = torch.nn.Tanh()
self.sigm = torch.nn.Sigmoid()

![](./learning_research/exp_2.png)

## Опыт 3

learning-rate: 0.0001

Число эпох: 10

Размер данных: 1600
Размер батча: 4

Fine-tuning:
self.example_linear_1 = torch.nn.Linear(1024, 128)
self.example_linear_2 = torch.nn.Linear(128, 32)

self.def_linear_1 = torch.nn.Linear(1024, 128)
self.def_linear_2 = torch.nn.Linear(128, 32)

self.Linear = torch.nn.Linear(32, 1)
self.tanh = torch.nn.Tanh()
self.sigm = torch.nn.Sigmoid()

![](./learning_research/exp_3.png)
