from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser 
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
import pandas as pd
import argparse
import json
import time
import os
import csv


api_key = ""

# example template
# prompting strategy: multi-class classification
template =  [
'''
INSTRUCTION

You are a mental health expert. 
The following are symptoms related to depression and their descriptions.
Based on the given post, please infer the presence of symptoms.

1. Feeling bad about yourself or that you are a failure or have let yourself or your family down
2. Feeling down depressed or hopeless
3. Feeling tired or having little energy
4. Little interest or pleasure in doing
5. Moving or speaking so slowly that other people could have noticed Or the opposite being so fidgety or restless that you have been moving around a lot more than usual
6. Poor appetite or overeating
7. Thoughts that you would be better off dead or of hurting yourself in some way
8. Trouble concentrating on things such as reading the newspaper or watching television
9. Trouble falling or staying asleep or sleeping too much

For each symptom, answer based on the following criteria. 
1: The user experiences the symptom.
0: The user doesn't experience the symptom.

Print out the result as a JSON file.
Here is an example for you.

EXAMPLE
Post: Im 21 got a porn addiction. Ruining my life and my mind, no freinds and im a short guy. Diagnosed mentally ill with psychotic depression and im unstable. I believe in genetic determinism and think if my life is rigged this way the why am i not allowed to kill myself. I wake up depressed everyday always thinking of pure crap. just want to end it all. I hate the asshole who gave me life. I see my life as being a personal hell. I hear vioces in my head and are usually active when at night alone. They are proper cunts i try to reason with them and they are not even empathetic. They say stuff like “faggot” “cunt” proper stuff to make you wanna hit someone. constantly fucking my head in. I just wanna die so bad.
Answer: 
1. Feeling bad about yourself or that you are a failure or have let yourself or your family down: 1
2. Feeling down depressed or hopeless: 1
3. Feeling tired or having little energy: 0
4. Little interest or pleasure in doing: 0
5. Moving or speaking so slowly that other people could have noticed Or the opposite being so fidgety or restless that you have been moving around a lot more than usual: 0
6. Poor appetite or overeating: 0
7. Thoughts that you would be better off dead or of hurting yourself in some way: 1
8. Trouble concentrating on things such as reading the newspaper or watching television: 1
9. Trouble falling or staying asleep or sleeping too much: 1


Now, it's your turn. 
Post: {question}
Answer: 
'''
]

label = [
    {"1. Feeling bad about yourself or that you are a failure or have let yourself or your family down": "error",
    "2. Feeling down depressed or hopeless": "error",
    "3. Feeling tired or having little energy": "error",
    "4. Little interest or pleasure in doing": "error",
    "5. Moving or speaking so slowly that other people could have noticed Or the opposite being so fidgety or restless that you have been moving around a lot more than usual": "error",
    "6. Poor appetite or overeating": "error",
    "7. Thoughts that you would be better off dead or of hurting yourself in some way": "error",
    "8. Trouble concentrating on things such as reading the newspaper or watching television": "error",
    "9. Trouble falling or staying asleep or sleeping too much": "error"}
]

def run_langchain(template, text):
    prompt = PromptTemplate.from_template(template)
    chat = ChatOpenAI(model="gpt-3.5-turbo",
                      temperature=0,
                      openai_api_key = api_key)
    output_parser = StrOutputParser()
    chain =  prompt | chat | output_parser
    result = chain.invoke({"question": text})     
    return result

def json_to_dataframe(json_str,idx):
    try:
        data = json.loads(json_str)
    except:
        data = label[idx]
    df = pd.DataFrame(data, index=[0])
    return df

def main(data_file, save_path, factor_type, txt):
    if not os.path.exists(f"result/{save_path}"):
        os.makedirs(f"result/{save_path}")

    df = pd.read_csv(data_file)
    data = df['pre_question']

    result_list = []
    idx = int(factor_type[-1])-1
    #print(idx)
    temp = template[idx]
    #temp = template[0]
    for i in range(len(data)): 
        print('-------------', i, '-----------')
        if txt==True:
            result = run_langchain(temp, data[i]).replace('json','').replace('\n','').replace('```','')
            result = str(result)
            print(result)
            file_mode = 'w' if not os.path.exists(f'./process/{factor_type}_processing.txt') else 'a'
            with open(f'./process/{factor_type}_processing.txt', file_mode) as file:
                file.write(result + '\n')
        else:
            if i==0:
                result = run_langchain(temp, data[i]).replace('json','').replace('\n','').replace('```','')
                result = str(result)
                print(result)
                result_ = json_to_dataframe(result,idx)
                result_.to_csv(f'./process/{factor_type}_processing.csv',index=False)
            else:
                result = run_langchain(temp, data[i]).replace('json','').replace('\n','').replace('```','')
                result = str(result)
                print(result)
                result_ = json_to_dataframe(result,idx)
                result_.to_csv(f'./process/{factor_type}_processing.csv',mode='a',index=False,header=False)
            result_list.append(result)

        '''
        result = run_langchain(template, data[i])
        print(result)
        
        if len(result_list) % 10 == 0:
            dfs = [json_to_dataframe(result) for result in result_list]
            combined_df = pd.concat(dfs, ignore_index=True)
            now = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            combined_df.to_csv(save_path+now+factor_type+"_length"+str(len(result_list))+".csv", index=False)
        '''
    if not txt:
        dfs = [json_to_dataframe(result,idx) for result in result_list]
        now = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df.to_csv("result/"+save_path+"/"+now+"_"+factor_type+".csv", index=False)


if __name__ == "__main__":
    print("Start Process")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Data Path")
    parser.add_argument("--save_path", type=str, help="Save Path")
    parser.add_argument("--factor", type=str, help="Factor")
    parser.add_argument("--txtf", default=False, type=bool, help="save txt")
    args = parser.parse_args()
    main(args.data, args.save_path, args.factor, args.txtf)
    print("Done")