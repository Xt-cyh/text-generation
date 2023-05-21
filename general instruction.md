**ICML23：**

- icml23（如果泛化性不好）可能是源于预设任务不够多？例如都是自由生成，长度控制均以5为单位。但是为用于训练的每个约束设计了多个不同的自然语言模板的思路确实利于多样化。

- icml23 只测了unseen constraints上的泛化性，**但是没测unseen instruct**.

- icml23 做了组合constraints（Constraint Composition）、PHARAPHRASE等条件生成任务

- 暂时不知道 ”Moreover, we design multiple diverse natural language templates for each constraint used for training.“  究竟有多少。

&nbsp;

#### Generalized Controlled Text Generation with Natural Language Instructions

一方面证实他的setting有问题，做不到真正意义上的generalized，另一方面思考一下真正的generalized怎么做，怎么评（善用chatgpt），能不能提出一个新benchmark

**生成指令：**

使用

**评价：**

用chatgpt将一个generalize的指令转换为一个格式化的constraint，这样会更容易判断指令的类别，但后续评价不能再用chatgpt，因为消耗大，需要根据这些格式化constraint的指令，指令微调一个模型？

**思考**

1. 评价的指令只能用来做评价？可以用来转换生成指令，然后再instruct ctg？

2. 最终进行指令ctg的模型和转换格式化constraint的模型可以是同一个（生成模型和评估模型（生成用于评估的格式化constraint的模型）为同一个？），甚至可以将两步进行融合？(general的ctg可以分为两步？)（折中方法）

&nbsp;

#### instruction的泛化

(使其接近 实际使用中different people的问法 ~~数据分布接近于真实分布~~？数据分布较为平均，多样性指标不低于实际数据、指令集) （怎么证明一个数据集是多样化的）

怎么定义？

**属性控制**

1. 指示条件的多样化：情感（positive/negative --> ）

2. 表述方式的多样化

3. 任务形式(输入输出的不同)；

4. 多属性；

5. 说话风格，语气

**长度控制**

**格式**

关键词

文本格式

语法限制

###### 需要考虑相反情况：要求不满足......条件

**条件生成**

(条件生成任务+属性控制的扩展（基于某个情境，生成文本）)

（作为Unseen Constraints测试）

&nbsp;

#### 让chatgpt生成多样化控制生成指令的prompt：

（1人工编写seed instruct+ 2chatgpt zeroshot生成并筛选seed instruct）+ 3使用diversify prompt进行扩展；4从()中抽取文本，逆向构造可能的控制属性/控制instruction+响应

zero-shot prompt

- 属性控制：
  
  - Write some instructions which request to generate texts with a certain attribute, these instructions need to be diversified:
  
          <img title="" src="file:///C:/Users/Administrator.DESKTOP-1HKQ2HI/AppData/Roaming/marktext/images/2023-05-19-19-00-06-image.png" alt="" width="374">
  
  - Write some instructions which request ChatGPT to generate texts with a certain abstract attribute, these instructions need to be diversified and concise:
    
    不加concise：
    
    <img src="file:///C:/Users/Administrator.DESKTOP-1HKQ2HI/AppData/Roaming/marktext/images/2023-05-19-19-10-45-image.png" title="" alt="" width="387">
    
    加concise：
    
    <img src="file:///C:/Users/Administrator.DESKTOP-1HKQ2HI/AppData/Roaming/marktext/images/2023-05-19-19-10-58-image.png" title="" alt="" width="409">
    
    效果不错

few-shot prompt

- 属性控制：
  
  - Here is an instruction of sentiment controlled text generation：
    
    ”Please write a positive sentiment continuation: Once upon a day, “
    
    Please generate 5 instructions of sentiment-controlled text generation, more diversely:
    
    <img title="" src="file:///C:/Users/Administrator.DESKTOP-1HKQ2HI/AppData/Roaming/marktext/images/2023-05-18-20-50-05-image.png" alt="" width="394" data-align="inline">
    
    分析：实现了控制词、任务具体输入的多样化，但未完成指令与任务的多样化。
    
    - 最后加, try different tasks 无效
    - 示例换成什么任务最后就会是什么任务：”Please write a positive sentiment sentence:“，同样只换了不同的指示词
  
  - Here are instructions of sentiment controlled text generation：  
    ”Please write a positive sentiment continuation：“, ”Write a negative sentiment sentence:“  
    Please generate 5 instructions of sentiment controlled text generation, more diversely and make the instructions more differently:
    
    <img src="file:///C:/Users/Administrator.DESKTOP-1HKQ2HI/AppData/Roaming/marktext/images/2023-05-18-21-35-34-image.png" title="" alt="" width="388">
    
    还是得有seed task
  
  - （change instruction，不是凭空生成）
    
    I want to diversify the representation of instructions that control the sentiment of requesting to generate texts with certain sentiments.
    
    To diversify the instructions, you can (try different methods): 1. Replace words referring to sentiment. 2. Switching the position of emotional words in the texts. 3. Change the way of expression. 4. Try different NLP tasks.
    
    Now you can change the instruction "Write something of positive sentiment." to several diverse instructions: 1. 2. 3. 4. 5. 
    
    <img src="file:///C:/Users/Administrator.DESKTOP-1HKQ2HI/AppData/Roaming/marktext/images/2023-05-19-17-49-23-image.png" title="" alt="" width="423">
    
    有点效果
  
  - （use edit example）
    
    I want to diversify the representation of instructions that ~~(control the sentiment of requesting to)~~ request to generate texts with certain sentiments.
    
    To diversify the instructions, you can try different methods, not limited to the given example: 
    
    1. Replace words referring to sentiment: "Write something of positive sentiment."->"Write something of negative sentiment."
    
    2. Switching the position of emotional words in the texts: "Write something of  sadness."->"Write positive sentences"
    
    3. Change the way of expression: "Write some exciting texts:"->"Please help me to write a positive sentence"
    
    4. Try different NLP tasks: "Write something of negative sentiment"->"Please write a negative sentiment continuation："
    
    Now you can change the instruction "Write something of positive sentiment." to several diverse instructions: 1. 2. 3. 4. 5.
    
    <img title="" src="file:///C:/Users/Administrator.DESKTOP-1HKQ2HI/AppData/Roaming/marktext/images/2023-05-19-18-24-36-image.png" alt="" width="379">
    
    <img src="file:///C:/Users/Administrator.DESKTOP-1HKQ2HI/AppData/Roaming/marktext/images/2023-05-19-18-32-31-image.png" title="" alt="" width="395">

&nbsp;

#### Filter the instructions
1. 较为重复的
2. 质量低的
