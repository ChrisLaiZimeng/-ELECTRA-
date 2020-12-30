#nohup python -u test.py > test.log 2>&1 &
#find ./ -type f | wc -l
import argparse
import os
import json
import re
import logging
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import torch 
from transformers import BertTokenizer, BertForMaskedLM
from transformers import pipeline
from summarizer import Summarizer

parse = argparse.ArgumentParser()
parse.add_argument('--data',type=str, default='../ELE/test/')
parse.add_argument('--mask',type=str, default='[MASK]')
parse.add_argument('--output',type=str,default='../midData/test/')
parse.add_argument('--log',type=str,default='../log/maskTest.log')
parse.add_argument('--type',type=str,default='test')
args = parse.parse_args()

#mask pipeline
UNMASKER= pipeline('fill-mask', model='bert-base-uncased')

#log
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=args.log, level=logging.INFO, format=LOG_FORMAT)

#检查空格数、选项数、答案数是否相同
def check(data, article, options, answers):
  blanks = re.findall("_",article)
  print(data)
  if(len(blanks)!=len(answers) or len(blanks)!=len(options) or len(answers)!=len(answers)):
    print('error!'+data)

def answersToWord(options, answers):#正确答案的单词
  wordAnswers = []
  count = 0
  for answer in answers:
    if answer=='A':
      wordAnswers.append(options[count][0])
    elif answer=='B':
      wordAnswers.append(options[count][1])
    elif answer=='C':
      wordAnswers.append(options[count][2])
    elif answer=='D':
      wordAnswers.append(options[count][3])
    count+=1
  #print(options,answers,wordAnswers)
  return wordAnswers

def maskScore(answers, prediction):#正确个数
    count = 0
    right = 0
    while(count<len(answers)):
        if answers[count]==prediction[count]:
            right+=1
        count+=1
    return right

def summy(article, predict):
    count = 0
    copy = article
    while(re.search('_',copy)):
        copy = copy.replace('_', predict[count], 1)
        count+=1
    model = Summarizer()
    result = model(copy)
    return ''.join(result)

def mask(data, article, options, answers):
    count = 0
    filledNum = 0
    maskAnswer = []
    
    while(re.search('_',article)):
        #MASK前后要是空格 不然替代之后就连成一串了
        if re.search('_',article).span()[1]==len(article) : #最后一个空
            if article[re.search('_',article).span()[0]-1]!=' ':
                article = article.replace('_', ' '+args.mask, 1)
            else:
                article = article.replace('_', args.mask, 1)
        elif re.search('_',article).span()[0]==0 :#第一个空
            if article[re.search('_',article).span()[1]]!=' ':
                article = article.replace('_', args.mask+' ', 1)
            else:
                article = article.replace('_', args.mask, 1)
        else:
            if article[re.search('_',article).span()[1]]!=' ':#缺后空
                if article[re.search('_',article).span()[0]-1]!=' ': #缺前空
                    article = article.replace('_', ' '+args.mask+' ', 1)
                else:
                    article = article.replace('_', args.mask+' ', 1)
            else: #不缺后空
                if article[re.search('_',article).span()[0]-1]!=' ': #缺前空
                    article = article.replace('_', ' '+args.mask, 1)
                else:
                    article = article.replace('_', args.mask, 1)

        #处理过长的文章 裁掉之后转为feature
        articleList = article.split(' ')
        #print(articleList)
        if(len(articleList)>300):
            pos = articleList.index(args.mask)
            #print(pos)
            if pos<150:
                feature = ' '.join(article.split(' ')[0:300])
            else:
                if pos+150>len(articleList):
                    feature = ' '.join(article.split(' ')[-300:])
                else:
                    feature = ' '.join(article.split(' ')[pos-150:pos+150])
        else:
            feature = article
        
        #所有预测结果 feature
        results = UNMASKER(feature)

        find = False
        for result in results:
            if result['token_str'] in options[count]:
                if '_' not in result['token_str']: #不能填 _
                    find = True
                    filledNum+=1
                    predict = result['token_str']
                    break
        
        #不在选项里 填预测的前两个
        if find == False:
            if '_' not in results[0]['token_str']:
                predict = results[0]['token_str']
            else:
                predict = results[1]['token_str']
        
        maskAnswer.append(predict)
        article = article.replace(args.mask, predict, 1)
        count+=1
    
    if(args.type == 'train' or args.type == 'dev'):
        #填正确的个数
        rightNum = maskScore(answersToWord(options, answers),maskAnswer)
        if filledNum!=0:
            percentage_filled = rightNum/filledNum
        else:
            percentage_filled = 0
        logging.info(data + ':  rightNum:%d  filledNum:%d  blanks:%d  percentage_filled:%f  percentage_all:%f',rightNum, filledNum, len(answers), percentage_filled, rightNum/len(answers))
    else:
        logging.info(data + ' success')

    return maskAnswer

def main():
    DATA_LIST = os.listdir(args.data)

    for data in DATA_LIST:
        if(os.path.exists(args.output+data)):
            print(data + ' exits')
        else:
            print(data + ' masking')
            with open(args.data+data,'r', encoding='utf-8') as f:
                json_data = json.load(f)
                article = json_data['article']
                options = json_data['options']
                answers = json_data['answers']
                try:
                    prediction = mask(data, article, options, answers)
                    summary = summy(article, prediction)

                    new_data = {}
                    new_data['article'] = article
                    new_data['options'] = options
                    new_data['answers'] = answers
                    new_data['maskPredict'] = prediction
                    new_data['summary'] = summary
                    json_str = json.dumps(new_data)
                    with open(args.output+data,'w') as w:
                        w.write(json_str)
                
                except Exception as e:
                    logging.debug(data)
                    logging.debug(e)

def debug():
    data = 'dev0320.json'
    with open(args.data+data,'r', encoding='utf-8') as f:
        json_data = json.load(f)
        article = json_data['article']
        options = json_data['options']
        answers = json_data['answers']

        prediction = mask(data, article, options, answers)
        summary = summy(article, prediction)

        new_data = {}
        new_data['article'] = article
        new_data['options'] = options
        new_data['answers'] = answers
        new_data['maskPredict'] = prediction
        new_data['summary'] = summary
        json_str = json.dumps(new_data)
        with open(args.output+data,'w') as w:
            w.write(json_str)

if __name__ == '__main__':
    main()
    #debug()
  