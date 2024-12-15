import re
import numpy as np
import datasets
import tensorflow as tf
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import seaborn as sns
import torch
from hexbytes import HexBytes
import matplotlib.pyplot as plt
from datasets import load_dataset, load_from_disk
from keras import layers
from sklearn.metrics import accuracy_score
from torch.nn.functional import normalize
from src.data.transform import generate_signal_and_label


def write_parquet(data, deth_path):
    # 定义每个块的大小
    chunk_size = 100000  # 每次处理 100,000 行数据

    # 将 df 按照行分成多个块
    chunks = np.array_split(data, np.ceil(len(data) / chunk_size))

    # 初始化 Parquet 写入器
    first_chunk = True
    for i, chunk in enumerate(chunks):
        # 将 pandas DataFrame 转换为 pyarrow 表格
        table = pa.Table.from_pandas(chunk)

        # 创建一个新的 Parquet 文件或附加数据
        if first_chunk:
            pqwriter = pq.ParquetWriter(deth_path, table.schema)
            first_chunk = False
        pqwriter.write_table(table)

    # 关闭写入器
    pqwriter.close()


if __name__ == '__main__':

    # 加载数据集
    train_dataset = datasets.load_dataset('parquet',
                                          data_files='src/data/mwritescode/slither-audited-smart-contracts/data/train/train.parquet')
    test_dataset = datasets.load_dataset('parquet',
                                         data_files='src/data/mwritescode/slither-audited-smart-contracts/data/test/test.parquet')
    val_dataset = datasets.load_dataset('parquet',
                                        data_files='src/data/mwritescode/slither-audited-smart-contracts/data/validation/val.parquet')

    # 打印初始加载的数据集
    print("Train dataset:", train_dataset)
    print("Test dataset:", test_dataset)
    print("Val dataset:", val_dataset)

    # 过滤掉 'bytecode' 为 '0x' 的行
    train_dataset = train_dataset.filter(lambda elem: elem['bytecode'] != '0x')
    test_dataset = test_dataset.filter(lambda elem: elem['bytecode'] != '0x')
    val_dataset = val_dataset.filter(lambda elem: elem['bytecode'] != '0x')

    # 打印过滤后的数据集大小
    print("Filtered train dataset size:", len(train_dataset))
    print("Filtered test dataset size:", len(test_dataset))
    print("Filtered val dataset size:", len(val_dataset))

    MAX_LEN = 16284
    def img_label_to_tensor(examples):
        if 'image' in examples.keys():
            examples['image'] = [np.pad(img, pad_width=(0, MAX_LEN - len(img))) if len(img) < MAX_LEN else img[:MAX_LEN]
                                 for img in examples['image']]
            examples['image'] = [torch.unsqueeze(normalize(torch.tensor(img).float(), dim=0), dim=0) for img in
                                 examples['image']]

            examples['label'] = torch.tensor(examples['label'])
            return examples

    val_dataset = val_dataset.map(generate_signal_and_label)
    print(val_dataset)
    for data in val_dataset['train']:
        print(data['address'], data['label'])
    val_dataset.set_transform(img_label_to_tensor)
    print(val_dataset)

    # 保存为 parquet 文件
    # val_dataset['train'].to_parquet('src/data/mwritescode/slither-audited-smart-contracts/data/validation/val_dataset.parquet')

    print(x)

    # split the text into strings of size 2 to mimic the length of Ethereum opcodes and create the training, validation and test features.
    def split_text_into_chars(text, length):
        return " ".join([text[i:i + length] for i in range(0, len(text), length)])


    # -----------------bytecode-------------------
    train_bytecode = [split_text_into_chars(data['bytecode'], 2) for data in cleaned_training_data]
    val_bytecode = [split_text_into_chars(data['bytecode'], 2) for data in cleaned_validation_data]
    test_bytecode = [split_text_into_chars(data['bytecode'], 2) for data in cleaned_test_data]
    # -----------------image-------------------
    train_image = [data['image'] for data in cleaned_training_data]
    val_image = [data['image'] for data in cleaned_validation_data]
    test_image = [data['image'] for data in cleaned_test_data]


    # -----------------sourcecode-------------------
    # train_sourcecode = []
    # val_sourcecode = []
    # test_sourcecode = []
    # def remove_comment(source_code):
    #       source_code = re.sub(r"//\*.*", "", source_code)
    #       source_code = re.sub(r"#.*", "", source_code)
    #       # Remove multi-line comments
    #       source_code = re.sub(r"/\*.*?\*/", "", source_code, flags=re.DOTALL)
    #       source_code = re.sub(r"\"\"\".*?\"\"\"/", "", source_code, flags=re.DOTALL)
    #
    #       source_code = re.sub(r"//.*", "", source_code)
    #
    #       # Remove redundant spaces and tabs
    #       source_code = re.sub(r"[\t ]+", " ", source_code)
    #
    #       # Remove empty lines
    #       source_code = re.sub(r"^\s*\n", "", source_code, flags=re.MULTILINE)
    #       return source_code
    # for data in cleaned_training_data:
    #       train_sourcecode.append(remove_comment(data['source_code']))
    # print(len(train_sourcecode))
    # for data in cleaned_validation_data:
    #       val_sourcecode.append(remove_comment(data['source_code']))
    # for data in cleaned_test_data:
    #       test_sourcecode.append(remove_comment(data['source_code']))
    # -----------------opcode-------------------

    # Create labels: transform slither data into one-hot encoded arrays that follow a certain logic.
    def get_slither(dataset):
        slither = [data['slither'] for data in dataset]
        # print(slither)
        slither = [[5 if x == 4 else x for x in sublist] if sublist != [] else [4] for sublist in slither]
        # print(slither)
        return slither


    training_slither = get_slither(cleaned_training_data)
    validation_slither = get_slither(cleaned_validation_data)
    test_slither = get_slither(cleaned_test_data)


    def labels_to_binary(y, num_labels):
        """
        Converts the labels into binary format depending on the total number of labels,
        for example: y = [1,4], num_labels = 5, y_binary = [0,1,0,0,1,0]
        """
        y_binary = np.zeros((len(y), num_labels), dtype=float)
        for i, label_indices in enumerate(y):
            y_binary[i, label_indices] = 1
        return y_binary


    num_classes = len(np.unique(np.concatenate(training_slither)))
    train_labels_binary = labels_to_binary(training_slither, num_classes)
    valid_labels_binary = labels_to_binary(validation_slither, num_classes)
    test_labels_binary = labels_to_binary(test_slither, num_classes)
    # print(valid_labels_binary)

    def transform_labels_to_dict(labels_binary):
        labels_dict = {}
        for index in range(num_classes):
            labels_dict[f'{index}'] = []
        for labels in labels_binary:
            for index, label in enumerate(labels):
                labels_dict[f'{index}'].append(label)
        return labels_dict


    SAFE_IDX = 4
    def __get_one_hot_encoded_label(label):
        one_hot = np.zeros(5)
        for elem in label:
            if elem < SAFE_IDX:
                one_hot[elem] = 1
            elif elem > SAFE_IDX:
                one_hot[elem - 1] = 1
        return one_hot


    validation_dict = transform_labels_to_dict(valid_labels_binary)
    train_dict = transform_labels_to_dict(train_labels_binary)
    test_dict = transform_labels_to_dict(test_labels_binary)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_bytecode, validation_dict)).batch(32).prefetch(tf.data.AUTOTUNE)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_bytecode, train_dict)).batch(32).prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_bytecode, test_dict)).batch(32).prefetch(tf.data.AUTOTUNE)


    # val_image_label = []
    # train_image_label = []
    # test_image_label = []
    # for elem in validation_slither:
    #     image_label = __get_one_hot_encoded_label(elem)
    #     val_image_label.append(image_label)
    # for elem in training_slither:
    #     image_label = __get_one_hot_encoded_label(elem)
    #     train_image_label.append(image_label)
    # for elem in test_slither:
    #     image_label = __get_one_hot_encoded_label(elem)
    #     test_image_label.append(image_label)
    # print(val_image_label)
    # Create datasets combination of features and labels.
    # train_dataset = tf.data.Dataset.from_tensor_slices((train_bytecode, train_dict)).batch(32).prefetch(tf.data.AUTOTUNE)
    # val_dataset = tf.data.Dataset.from_tensor_slices((val_bytecode, validation_dict)).batch(32).prefetch(
    #       tf.data.AUTOTUNE)
    # test_dataset = tf.data.Dataset.from_tensor_slices((test_bytecode, test_dict)).batch(32).prefetch(tf.data.AUTOTUNE)
    # val_dataset = tf.data.Dataset.from_tensor_slices(
    #     (val_bytecode, val_image, validation_dict, val_image_label)).prefetch(
    #     tf.data.AUTOTUNE)
    # train_dataset = tf.data.Dataset.from_tensor_slices(
    #     (train_bytecode, train_image, train_dict, train_image_label)).prefetch(tf.data.AUTOTUNE)
    # test_dataset = tf.data.Dataset.from_tensor_slices(
    #     (test_bytecode, test_image, test_dict, test_image_label)).prefetch(tf.data.AUTOTUNE)
    # saving to parquet
    # ******************train_dataset**********************
    # train_dataset = list(train_dataset.as_numpy_iterator())
    # df_train = pd.DataFrame(train_dataset, columns=["bytecode", "image", "byte_label", "image_label"])
    # print(df_train)
    # write_parquet(df_train, 'src/data/mwritescode/slither-audited-smart-contracts/data/train/train_dataset.parquet')
    # # ******************validation_dataset**********************
    # val_dataset = list(val_dataset.as_numpy_iterator())
    # print(val_dataset)
    # df_val = pd.DataFrame(val_dataset, columns=["bytecode", "image", "byte_label", "image_label"])
    # print(df_val)
    # write_parquet(df_val, 'src/data/mwritescode/slither-audited-smart-contracts/data/validation/val_dataset.parquet')
    # # ******************test_dataset**********************
    # test_dataset = list(test_dataset.as_numpy_iterator())
    # df_test = pd.DataFrame(test_dataset, columns=["bytecode", "image", "byte_label", "image_label"])
    # print(df_test)
    # write_parquet(df_test, 'src/data/mwritescode/slither-audited-smart-contracts/data/test/test_dataset.parquet')

    # Create the TextVectorizer layer to transform text data into a numerical representation that can be fed into a neural network for training and prediction.
    text_vectorizer = tf.keras.layers.TextVectorization(
        split="whitespace",
        max_tokens=1000,
        output_sequence_length=128
    )

    text_vectorizer.adapt(tf.data.Dataset.from_tensor_slices(train_bytecode).batch(32).prefetch(tf.data.AUTOTUNE))
    bytecode_vocab = text_vectorizer.get_vocabulary()
    print(f"Number of different characters in vocab: {len(bytecode_vocab)}")
    print(f"5 most common characters: {bytecode_vocab[:5]}")
    print(f"5 least common characters: {bytecode_vocab[-5:]}")

    # Create the Embedding layer
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=len(bytecode_vocab),
        input_length=128,
        output_dim=128,
        mask_zero=False,  # Conv layers do not support masking but RNNs do
        name="embedding_layer"
    )

    # Create the model
    # Create input layer
    inputs = layers.Input(shape=(1,), dtype=tf.string)
    # Create vectorization layer
    x = text_vectorizer(inputs)
    # Create embedding layer
    x = embedding_layer(x)
    # Create the LSTM layer
    x = layers.GRU(units=64, activation='tanh', return_sequences=True)(x)
    x = layers.GRU(units=32, activation='tanh')(x)
    x = layers.Dropout(rate=0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    # Create the output layer
    outputs = []
    for index in range(num_classes):
        output = layers.Dense(1, activation="sigmoid", name=f'{index}')(x)
        outputs.append(output)
    model_1 = tf.keras.Model(inputs=inputs, outputs=outputs, name="model_1")

    # Compile the model
    losses = {}
    metrics = {}
    for index in range(num_classes):
        losses[f'{index}'] = "binary_crossentropy"
        metrics[f'{index}'] = ['accuracy']
    model_1.compile(loss=losses, optimizer=tf.keras.optimizers.Adam(learning_rate=1e-03), metrics=metrics)

    # Fit the model
    history_1 = model_1.fit(train_dataset,
                            epochs=1,
                            validation_data=val_dataset,
                            callbacks=[
                                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                                     patience=5),
                                tf.keras.callbacks.ModelCheckpoint(filepath=f"model_experiments/model_gru",
                                                                   monitor='val_loss',
                                                                   verbose=0,
                                                                   save_best_only=True)
                            ])


    def convert_preds_probs_to_preds(probs, threshold=0.5):
        """
        将预测的概率转换为类别标签。

        对于二分类，概率大于等于 `threshold` 时预测为 1， 否则为 0。
        对于多分类任务，返回概率最大的索引值。

         Args:
            probs (np.array): 预测的概率数组。形状为 (n_samples, n_classes)。
                                如果是二分类任务，n_classes=1。
            threshold (float): 二分类任务的阈值，默认是0.5。

        Returns:
            np.array: 转换后的预测标签。对于二分类任务是0或1， 对于多分类任务是类别索引。
        """
        # 将 probs 转换为 NumPy 数组以支持广播操作
        probs = np.array(probs)

        # # 如果是二分类任务
        # if probs.shape[1] == 1:
        #     return (probs >= threshold).astype(int)
        # # 如果是多分类任务
        # else:
        #     return np.argmax(probs, axis=1)
        return (probs >= threshold).astype(int)


    # Predictions
    model_1 = tf.keras.models.load_model(filepath="model_experiments/model_gru")
    model_1_preds_probs = model_1.predict(test_dataset)
    model_1_preds = convert_preds_probs_to_preds(model_1_preds_probs)


    # model_1_preds = (np.array(model_1_preds_probs) >= 0.5).astype(np.int32)
    # print(model_1_preds_probs)

    def calculate_results(y_true, y_pred):
        """
        计算评估指标，假设这里只计算准确率。

        Args:
            y_true (list or np.array): 真实标签。
            y_pred (list or np.array): 预测标签。

        Returns:
            dict: 包含准确率的字典。
        """
        accuracy = accuracy_score(y_true, y_pred)
        return {"accuracy": accuracy}


    def combine_results(test_dict, model_1_preds):
        results = {}
        for index in range(num_classes):
            results[f'{index}'] = calculate_results(y_true=test_dict[f'{index}'], y_pred=model_1_preds[f'{index}'])
        return results


    import pandas as pd

    # 初始化结果字典
    results = {}

    # 遍历外层数组并转换每个二维数组为一维列表
    for index, sub_array in enumerate(model_1_preds):
        # 使用 ravel() 将二维数组压平成一维
        results[f'{index}'] = sub_array.ravel().tolist()
    results = combine_results(test_dict, results)
    pd.DataFrame(results)
