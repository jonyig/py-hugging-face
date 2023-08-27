# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from transformers import pipeline

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
    translator = pipeline(model='Helsinki-NLP/opus-mt-zh-en')  # 使用翻譯任務
    result = translator(['這是一個很困難的問題',"不要"])
    print(result)

    specific_model = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")

    data = ["I love you", "I hate you","can you help me to clean my pants"]
    result = specific_model(data)
    print(result)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
