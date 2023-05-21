#### 怎么评估更加泛化的可控文本生成？

- 将llm作为控制器，把多样化的指令转换为格式化的constraint，然后调用不同的下游评测模型/自己评测。

- 将一个有泛化性ctg功能的llm作为奖励模型

- 怎么评估数据集的多样性 以及 怎么证明泛化性(1.用self-instruct中提出的方法（longform也用了这个）2评估多样性的指标。)

#### Reward model

需要ranking loss ，也就需要同一个instruct的不同response，不同response的来源：不同模型生成 原始数据集/抽取出的文本 同任务不同属性的数据。还需要一个打分模型

#### 怎么构造格式化constraint？

考虑prompt+(in-context learning):

**Try1:**

Please format the output of the command generated by the following controlled text, indicating the task form of the command and the control properties:

Instruction: please write a positive sentiment sentence;  
Format output:

**Anlysis1:**

1. 任务形式（续写，自由生成，风格迁移等等）可以提前分好，而不是用chatgpt判断？

2. chatgpt需要判断的：控制类型（属性、格式）；控制属性（情感、主题）