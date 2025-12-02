import pandas as pd
from sklearn.model_selection import train_test_split
from tokenizer import ChineseTokenizer, EnglishTokenizer
import config
import os
import nltk

def check_nltk():
    # 指定下载目录（可选，也可以去掉 download_dir 参数让它下到系统默认目录）
    download_dir = config.NLTK_DATA_DIR
    nltk.data.path.append(download_dir)
    
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab') # 新版 nltk 可能需要这个
    except LookupError:
        print("正在下载 NLTK 'punkt' 模型...")
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        nltk.download('punkt', download_dir=download_dir)
        nltk.download('punkt_tab', download_dir=download_dir)



def process():
    check_nltk()
    print("Loading data...")
    
    # 读取原始数据
    df = pd.read_csv(
        config.RAW_DATA_DIR / 'cmn.txt',
        sep='\t',
        header=None,
        usecols=[0, 1],
        names=['en', 'zh']
    )
    
    # 清洗数据：去空行
    df = df.dropna()
    df = df[df['en'].str.strip().ne('') & df['zh'].str.strip().ne('')]
    
    # 划分训练集和测试集
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # 构建并保存词表
    EnglishTokenizer.build_vocab(train_df['en'].tolist(), config.PROCESSED_DATA_DIR / 'en_vocab.txt')
    ChineseTokenizer.build_vocab(train_df['zh'].tolist(), config.PROCESSED_DATA_DIR / 'zh_vocab.txt')
    
    # 加载刚才的词表
    en_tokenizer = EnglishTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'en_vocab.txt')
    zh_tokenizer = ChineseTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'zh_vocab.txt')
    
    # 将文本转换为数字序列 英文需要加入 <sos> 和 <eos>
    train_df['en'] = train_df['en'].apply(
        lambda x: en_tokenizer.encode(x, seq_len=config.SEQ_LEN, add_sos_eos=True)
    )
    train_df['zh'] = train_df['zh'].apply(
        lambda x: zh_tokenizer.encode(x, seq_len=config.SEQ_LEN, add_sos_eos=False)
    )
    
    # 保存训练集
    train_df.to_json(config.PROCESSED_DATA_DIR / 'indexed_train.jsonl', orient='records', lines=True)
    
    # 将测试集转换为数字序列
    test_df['en'] = test_df['en'].apply(
        lambda x: en_tokenizer.encode(x, seq_len=config.SEQ_LEN, add_sos_eos=True)
    )
    test_df['zh'] = test_df['zh'].apply(
        lambda x: zh_tokenizer.encode(x, seq_len=config.SEQ_LEN, add_sos_eos=False)
    )
    
    # 保存测试集
    test_df.to_json(config.PROCESSED_DATA_DIR / 'indexed_test.jsonl', orient='records', lines=True)
    
    print("Data processing complete. Train and test sets saved.")
    
if __name__ == '__main__':
    process()
    
    