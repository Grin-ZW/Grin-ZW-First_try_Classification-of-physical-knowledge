"""
实现额外的方法
"""
import re
import jieba

def tokenlize(sentence):
    """
    进行文本分词
    :param sentence: str
    :return: [str,str,str]
    """

    fileters = ['！', '“', '#', '$', '%', '&', '\（', '\）', '\*', '\+', '，', '——','-', '\。', '/', '；', '：', '<', '=', '>',
                '\？', '@', '\【', '\\', '\】', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”', '“',',' ]
    #sentence = sentence.lower() #把大写转化为小写
    sentence = re.sub("<br />"," ",sentence)
    # sentence = re.sub("I'm","I am",sentence)
    # sentence = re.sub("isn't","is not",sentence)
    sentence = re.sub("|".join(fileters)," ",sentence)
    result = [i.strip() for i in sentence.split(" ") if len(i)>0]
    result = "".join(result)

    return jieba.lcut(result)

if __name__ == '__main__':
    tests="黄河之水，，天上来"
    z=tokenlize(tests)
    print(z)
