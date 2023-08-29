# This is a sample Python script.
import numpy as np
import torch
from torch import cosine_similarity
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from transformers import pipeline, AutoModel, AutoTokenizer


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

    # translation
    translator = pipeline(model='Helsinki-NLP/opus-mt-zh-en')  # 使用翻譯任務
    result = translator(['這是一個很困難的問題',"不要"])
    print(result)

    # sentiment
    specific_model = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")
    data = ["I love you", "I hate you","can you help me to clean my pants"]
    result = specific_model(data)
    print(result)

    # context Q&A
    model_name = "deepset/roberta-base-squad2"

    # a) Get predictions
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    QA_input = {
        # 'question': 'Why is model conversion important?',
        # 'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
        'question': 'Where do I live?',
        'context': 'My name is Sarah and I live in London'
    }
    res = nlp(QA_input)
    print(res)

    # embedding Q&A

    # 加载预训练模型
    model = AutoModel.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # 对文本进行分词
    text1 = "This is a test sentence."
    text2 = "This is another test sentence."
    tokens1 = tokenizer(text=text1)
    tokens2 = tokenizer(text=text2)

    # 获取文本的 embedding
    embeddings1 = model(**tokens1.input_ids).last_hidden_state
    embeddings2 = model(**tokens2.input_ids).last_hidden_state

    # 计算相似度
    similarity = torch.cosine_similarity(embeddings1, embeddings2)
    print(similarity)
    # model = INSTRUCTOR('hkunlp/instructor-large')
    # query = [['Represent the Wikipedia question for retrieving supporting documents: ',
    #           'where is the food stored in a yam plant']]
    # corpus = [['Represent the Wikipedia document for retrieval: ',
    #            'Capitalism has been dominant in the Western world since the end of feudalism, but most feel[who?] that the term "mixed economies" more precisely describes most contemporary economies, due to their containing both private-owned and state-owned enterprises. In capitalism, prices determine the demand-supply scale. For example, higher demand for certain goods and services lead to higher prices and lower demand for certain goods lead to lower prices.'],
    #           ['Represent the Wikipedia document for retrieval: ',
    #            "The disparate impact theory is especially controversial under the Fair Housing Act because the Act regulates many activities relating to housing, insurance, and mortgage loansâ€”and some scholars have argued that the theory's use under the Fair Housing Act, combined with extensions of the Community Reinvestment Act, contributed to rise of sub-prime lending and the crash of the U.S. housing market and ensuing global economic recession"],
    #           ['Represent the Wikipedia document for retrieval: ',
    #            'Disparate impact in United States labor law refers to practices in employment, housing, and other areas that adversely affect one group of people of a protected characteristic more than another, even though rules applied by employers or landlords are formally neutral. Although the protected classes vary by statute, most federal civil rights laws protect based on race, color, religion, national origin, and sex as protected traits, and some laws include disability status and other traits as well.']]
    # query_embeddings = model.encode(query)
    # corpus_embeddings = model.encode(corpus)
    # similarities = cosine_similarity(query_embeddings, corpus_embeddings)
    # retrieved_doc_id = np.argmax(similarities)
    # print(retrieved_doc_id)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
