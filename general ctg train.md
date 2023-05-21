1：得到more general 的instructions

2：填充instruction，训练模型

3：评估general instruction生成的文本

&nbsp;

#### 构造Response

1. **直接从对应数据集中构造（现有任务）**

2. **从各种数据集中抽取数据，构造可能的控制属性、控制指令+响应**（longform instruct ctg）
   
   类似：
   
   Instruction: X
   
   Output: <corpus_example>
   
   What kind of instruction could this be the answer to?
   
   X:

3. **llm生成**

&nbsp;

#### 实验：(构造response 2)

1.

Attribute/Topic: X; Instruction: Y

Response: This essay discusses the importance of science in our daily lives. Science plays a crucial role in shaping our understanding of the world around us and in improving our lives. 

Which attribute/topic does this text convey? What instruction can generate this text by controlling this attribute/topic? Attribute/Topic should be concise:

X:

Y:

![](C:\Users\Administrator.DESKTOP-1HKQ2HI\AppData\Roaming\marktext\images\2023-05-20-18-12-54-image.png)

Keyword: X; Instruction: Y  
Response: This essay discusses the importance of science in our daily lives. Science plays a crucial role in shaping our understanding of the world around us and in improving our lives.  
What is the core keyword of this text? What instruction can generate this text by controlling this keyword?  
X:  
Y:

<img src="file:///C:/Users/Administrator.DESKTOP-1HKQ2HI/AppData/Roaming/marktext/images/2023-05-20-18-31-36-image.png" title="" alt="" width="455">

Attribute/Topic: X; Instruction: Y  
Response: Front projection was used in 2001: A Space Odyssey, which was a groundbreaking film that revolutionized the science fiction genre and set a new standard for visual effects. Its use of front projection allowed the filmmakers to create stunning, realistic space scenes that  
What is the core Attribute/Topic of this text? What instruction can generate this text by controlling this Attribute/Topic?  
X:  
Y:

<img src="file:///C:/Users/Administrator.DESKTOP-1HKQ2HI/AppData/Roaming/marktext/images/2023-05-20-19-16-44-image.png" title="" alt="" width="436">

<img src="file:///C:/Users/Administrator.DESKTOP-1HKQ2HI/AppData/Roaming/marktext/images/2023-05-20-19-21-29-image.png" title="" alt="" width="544">
