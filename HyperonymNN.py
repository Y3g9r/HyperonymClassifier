import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
os.add_dll_directory("C:/Program Files/cudnn-windows-x86_64-8.4.0.27_cuda11.6-archive/bin")

import numpy as np
import pandas as pd
import torch
import transformers
import json,io
from transformers import BertTokenizerFast, BertModel, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch.optim as optim
import tensorflow
import matplotlib.pyplot as plt


#device = torch.device("cuda:0")
device = torch.device("cpu")
print(torch.cuda.is_available())
print(device)
#torch.cuda.memory_summary(device=None, abbreviated=False)
#torch.cuda.empty_cache()

#максимальная длинна предложения
MAX_LENGTH = 180

def generate_bert_input(file_name=None,sample_count=None,input_data=None):
    ###
    #input data:
    #
    #file_name: имя файла формата json c сэмплами вида [[[5, 10]], ["Если монах ведёт себя согласно со своими обетами, он не страшилище, не чуждое, почти враждебное нам существо, не живой труп, а наоборот, великий духовный друг наш и отец, носитель духовной благодатной жизни, молитвенник за нас пред Богом."], ["монах 0"], ["член религиозной общины, давший обет ведения аскетической жизни"], [1]]
    #model_name: имя bert модели
    #
    #return:
    #
    #лист картежей вида ((tokens_tensor, segments_tensors, offset_mapping, samp_position),(tokens_tensor, segments_tensors, offset_mapping, samp_position),label)
    #где tokens_tensor токенезированные слова определения или примера употребления в зависимости от картежа соответственно
    #где segments_tensors длина  определения или примера употребления в зависимости от картежа соответственно
    #где offset_mapping позиция ключеовго слова  определения(всегда первое) или примера употребления в зависимости от картежа соответственно
    #где samp_position позиция ключеовго слова из базы данных для определения(всегда первое) или примера употребления в зависимости от картежа соответственно
    ###
    model_name = 'sberbank-ai/sbert_large_mt_nlu_ru'
    if input_data == None:
        data = []
        with io.open(file_name, 'r', encoding="utf-8-sig") as f:
            data = json.load(f)
    else:
        data = input_data

    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    samples_data = []
    i = 0
    for text in data:
        if i==sample_count:
          break
        #получаем данные для примера употребления
        marked_text = text[1][0]
        # tokenized_text = tokenizer(marked_text, return_offsets_mapping=True,max_length=MAX_LENGTH,padding='max_length',truncation=True)
        # print(f"1 {tokenized_text}")
        # offset_mapping = tokenized_text['offset_mapping']
        # indexed_tokens = tokenized_text['input_ids']
        # segments_ids = [1] * len(tokenized_text['input_ids'])
        # tokens_tensor = torch.tensor([indexed_tokens])
        # segments_tensors = torch.tensor([segments_ids])
        samp_position = text[0][0]
        tokenized_text = tokenizer(marked_text, return_offsets_mapping = True,max_length=MAX_LENGTH,padding='max_length',truncation=True, return_tensors='pt')
        # print(tokenized_text)
        sample_data = (tokenized_text['input_ids'],tokenized_text['token_type_ids'],tokenized_text['attention_mask'],tokenized_text['offset_mapping'],samp_position)

        #получаем данные для определения
        #берём само слово text[2][0][:-2] и его определение text[3][0] и строим текст вида
        #(слово - определение)
        # definition = text[2][0][:-2]+" - "+text[3][0]
        # def_positions = [0, len(text[2][0][:-2])]

        marked_text = text[3][0]
        # tokenized_text = tokenizer(marked_text,max_length=MAX_LENGTH,padding='max_length',truncation=True)
        # # offset_mapping = tokenized_text['offset_mapping']
        # indexed_tokens = tokenized_text['input_ids']
        # segments_ids = [1] * len(tokenized_text['input_ids'])
        # tokens_tensor = torch.tensor([indexed_tokens])
        # segments_tensors = torch.tensor([segments_ids])
        tokenized_text = tokenizer(marked_text,max_length=MAX_LENGTH,padding='max_length',truncation=True, return_tensors='pt')
        def_data = (tokenized_text['input_ids'],tokenized_text['token_type_ids'],tokenized_text['attention_mask'])

        temp_data = ((def_data),(sample_data),text[4])
        samples_data.append(temp_data)
        i+=1
    return samples_data


class HyperonymDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        items = {
            "input_ids_def": torch.tensor(self.dataset[idx][0][0]),
            "token_type_ids_def": torch.tensor(self.dataset[idx][0][1]),
            "attention_mask_def": torch.tensor(self.dataset[idx][0][2]),

            "input_ids_samp": torch.tensor(self.dataset[idx][1][0]),
            "token_type_ids_samp": torch.tensor(self.dataset[idx][1][1]),
            "attention_mask_samp": torch.tensor(self.dataset[idx][1][2]),
            "offset_mapping_samp": torch.tensor(self.dataset[idx][1][3]),
            "samp_position_samp": torch.tensor(self.dataset[idx][1][4]),
            # "tokens_tensor_def": torch.tensor(self.dataset[idx][0][0]),
            # "segments_tensors_def": torch.tensor(self.dataset[idx][0][1]),
            # # "offset_mapping_def": torch.tensor(self.dataset[idx][0][2]),
            # # "samp_position_def": torch.tensor(self.dataset[idx][0][3]),

            # "tokens_tensor_samp": torch.tensor(self.dataset[idx][1][0]),
            # "segments_tensors_samp": torch.tensor(self.dataset[idx][1][1]),
            # "offset_mapping_samp": torch.tensor(self.dataset[idx][1][2]),
            # "samp_position_samp": torch.tensor(self.dataset[idx][1][3]),

            "labels": torch.tensor(self.dataset[idx][2][0])
        }

        return items
        # items = {"dataset": self.dataset[idx]}
        # return items


class SemanticSimilarityBertModel(torch.nn.Module):
    def __init__(self):
        super(SemanticSimilarityBertModel, self).__init__()

        # Подгружаем модель
        model_name = "sberbank-ai/sbert_large_mt_nlu_ru"

        self.tanh_def = torch.nn.Tanh()

        self.model = BertModel.from_pretrained(model_name, output_hidden_states=True, return_dict=False).to(device)
        self.example_linear_1 = torch.nn.Linear(1024, 128)
        self.example_linear_2 = torch.nn.Linear(128, 128)

        self.def_linear_1 = torch.nn.Linear(1024, 128)
        self.def_linear_2 = torch.nn.Linear(128, 128)

        # Файн-тьюниги
        # self.rnn = torch.nn.RNN(input_size = 768, hidden_size = 512, bidirectional = False, batch_first = True)
        # self.AvgPool1D = torch.nn.AvgPool1d(768*2)
        # self.MaxPool1D = torch.nn.MaxPool1d(768*2)
        # self.concatenate = torch.cat((self.AvgPool1D,self.MaxPool1D),dim = 1)
        # self.Dropout = torch.nn.Dropout(0.3)
        self.Linear = torch.nn.Linear(128, 1)
        self.cos = torch.nn.CosineSimilarity(1)
        self.tanh = torch.nn.Tanh()
        self.sigm = torch.nn.Sigmoid()

    def forward(self, input_ids_def, token_type_ids_def, attention_mask_def, input_ids_samp, token_type_ids_samp,
                attention_mask_samp, offset_mapping_samp, samp_position_samp, labels):
        # ex = [[[5, 10]], ["Если монах ведёт себя согласно со своими обетами, он не страшилище, не чуждое, почти враждебное нам существо, не живой труп, а наоборот, великий духовный друг наш и отец, носитель духовной благодатной жизни, молитвенник за нас пред Богом."], ["монах 0"], ["член религиозной общины, давший обет ведения аскетической жизни"], [1]]
        # создаим пустой тезнор размерности 1x2x1 для хранения эмбедингов сэмплов из батчей (итоговый будет размерности Nx2x1536,где N - число сэмплов в батче)
        embd_batch = torch.tensor([[[], []]]).to(device)
        # необходимо для правильной установки размерности в батче embd_batch эмбедингов
        first_pass = False
        for i in range(len(input_ids_def)):
            # получаем эмбединги ключевого слова из примера употребления
            example_token_vec = self.get_vector(input_ids_samp[i], token_type_ids_samp[i], attention_mask_samp[i])
            examples_token_key_word_position = self.token_detection(offset_mapping_samp[i][0], samp_position_samp[i])
            example_embeddings = self.vector_recognition(example_token_vec, examples_token_key_word_position)

            # получаем эмбединги ключевого слова из определения
            def_embedding = self.get_defenition_embedding(input_ids_def[i], token_type_ids_def[i],
                                                          attention_mask_def[i])
            def_embedding = self.tanh_def(def_embedding)
            # объединяем два вектора в 1 и добавляем в общий массив (получаем тензор 2x768)
            # print(def_embedding)
            # print(def_embedding.shape)
            # cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
            # out = cos(def_embedding,example_embeddings)
            # print(f"cos similiarity {out} for label {labels[i]}  for dev: {tokenizer.convert_ids_to_tokens(tokens_tensor_def[i][0])}; and sample: {tokenizer.convert_ids_to_tokens(tokens_tensor_samp[i][0])}; ")
            embd_sample = torch.stack((example_embeddings, def_embedding)).to(device)
            # print(f"tokens {tokens_tensor_def[i]}")
            # print(f"new emb sample {embd_sample}")
            if not first_pass:
                embd_batch = torch.cat((embd_batch, embd_sample.unsqueeze(0)), -1)
                first_pass = True
            else:
                embd_batch = torch.cat((embd_batch, embd_sample.unsqueeze(0)), 0)

        # print(f"embd  {embd_batch}")
        # print(f"embd shape {embd_batch.shape}")
        ex_emb = embd_batch[:, 0, :]
        def_emb = embd_batch[:, 1, :]
        ex_emb = self.example_linear_1(ex_emb)
        ex_emb = self.tanh(self.example_linear_2(ex_emb))

        def_emb = self.def_linear_1(def_emb)
        def_emb = self.tanh(self.def_linear_2(def_emb))
        # print(def_emb.shape)
        # embd_batch = torch.cat((ex_emb.unsqueeze(1), def_emb.unsqueeze(1)),dim=1)
        # print(embd_batch.shape)

        dif = torch.abs(ex_emb - def_emb)
        # dif = self.cos(ex_emb,def_emb)
        # print(dif.shape)

        # _,last_state = self.rnn(embd_batch)
        # print(f"last state shape {last_state.shape}")
        # print(f"last state  {last_state}")
        # print(f"last state shape {last_state.shape}")
        # print(f"x shape {x.shape}")
        # GlobalAveragePooling1D заменяется торч мином по 1 измерению mean(dim=(1))
        # Ax = torch.mean(x,1)
        # MaxAveragePooling1D заменяется  Mx,_ = torch.max(x,1)
        # Mx,_ = torch.max(x,1)
        # print(f"ax shape {Ax.shape} Mx shape {Mx.shape}")
        # concatenate = torch.cat((Ax, Mx), dim=1)
        # print(f"conc shape {concatenate.shape}")
        # Do = self.Dropout(concatenate)
        y = self.Linear(dif)
        # print(f"Linear   {y}")
        y = self.sigm(y)
        return y.squeeze(0)

    def get_defenition_embedding(self, input_ids_def, token_type_ids_def, attention_mask_def):
        # with torch.no_grad():
        #     _, pooled_output, _ = self.model(tokens_tensor, segments_tensors)
        #     return pooled_output[0]
        with torch.no_grad():
            output = self.model(input_ids=input_ids_def, token_type_ids=token_type_ids_def,
                                attention_mask=attention_mask_def)
            hidden_states = output[2]
        # from [# layers, # batches, # tokens, # features] to [# tokens, # layers, # features]
        token_dim = torch.stack(hidden_states, dim=0)
        token_dim = torch.squeeze(token_dim, dim=1)
        token_dim = token_dim.permute(0, 1, 2)
        cat_vec = torch.cat(((token_dim[-4][0] + token_dim[-3][0] + token_dim[-2][0] + token_dim[-1][0]),), dim=0)
        return cat_vec

    def token_detection(self, token_map, position):
        # Функция определения ключевого слова
        """
        :param token_map: list of tuples of begin and end of every token
        :param position:  list of type: [int,int]
        :return: list of key word tokens position
        """
        # из за того что в начале стоит CLS позиции начала и конца ключевого слова сдвигаются на 5
        begin_postion = position[0]  # + 5
        end_position = position[1]  # + 5

        position_of_key_tokens = []
        for token_tuple in range(1, len(token_map) - 1):
            # if token is one
            if token_map[token_tuple][0] == begin_postion and token_map[token_tuple][1] == end_position:
                position_of_key_tokens.append(token_tuple)
                break

            # if we have multipli count of tokens for one key word
            if token_map[token_tuple][0] >= begin_postion and token_map[token_tuple][1] != end_position:
                position_of_key_tokens.append(token_tuple)
            if token_map[token_tuple][0] != begin_postion and token_map[token_tuple][1] == end_position:
                position_of_key_tokens.append(token_tuple)
                break

        return position_of_key_tokens

    def get_vector(self, input_ids_samp, token_type_ids_samp, attention_mask_samp):
        # Функция получения вектора ключевого слова
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids_samp, token_type_ids=token_type_ids_samp,
                                 attention_mask=attention_mask_samp)
            hidden_states = outputs[2]
        # from [# layers, # batches, # tokens, # features] to [# tokens, # layers, # features]
        token_dim = torch.stack(hidden_states, dim=0)
        token_dim = torch.squeeze(token_dim, dim=1)
        token_dim = token_dim.permute(1, 0, 2)
        token_vecs_cat = []
        for token in token_dim:
            cat_vec = torch.sum(token[-4:], dim=0)
            token_vecs_cat.append(cat_vec)

        return token_vecs_cat

    def get_avarage_embedding(self, embeddings_list, positions_list):
        # Функция получения среднего вектора
        avg_tensor = torch.stack((embeddings_list[positions_list[0]],))
        for i in range(1, len(positions_list)):
            avg_tensor = torch.cat((avg_tensor, embeddings_list[positions_list[i]].unsqueeze(0)))

        average_embedding = torch.mean(avg_tensor, 0)
        return average_embedding

    def vector_recognition(self, tokens_embeddings_ex, tokens_key_word_position_ex):
        # Функция подготовки вектора в зависимости от количества токенов,которым представляется ключевое слово
        if len(tokens_key_word_position_ex) > 1:
            embeddings_data = torch.tensor(
                self.get_avarage_embedding(tokens_embeddings_ex, tokens_key_word_position_ex))
        else:
            # print(tokens_embeddings_ex)
            # print(tokens_key_word_position_ex)
            embeddings_data = torch.tensor(tokens_embeddings_ex[tokens_key_word_position_ex[0]])
        return embeddings_data

# data = generate_bert_input(file_name='dataset_not_embeddings_1_one_0_one.json',sample_count= 1200 )
#
# RANDOM_SEED=1
# data_train, data_test = train_test_split(data, test_size=0.8, random_state=RANDOM_SEED)
# data_val, data_test = train_test_split(data_test, test_size=0.5, random_state=RANDOM_SEED)
#
# hd_train = HyperonymDataset(data_train)
# hd_val = HyperonymDataset(data_val)
# hd_test = HyperonymDataset(data_test)
#
# batch_size = 1
#
#
#
# trainloader = torch.utils.data.DataLoader(hd_train, batch_size=batch_size, shuffle=True)
#
# testloader = torch.utils.data.DataLoader(hd_test, batch_size=batch_size, shuffle=True)
#
#
# model = SemanticSimilarityBertModel()
# model = model.to(device)
#
# for param in model.model.parameters():
#     param.requires_grad = False
#
# criterion = torch.nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001)
#
# train_epoch = []
# train_loss = []
#
# test_epoch = []
# test_loss = []
#
#
#
# for epoch in range(10):
#
#     running_loss = 0.0
#     epoch_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         inputs = data
#
#         optimizer.zero_grad()
#         outputs = model(inputs['input_ids_def'].to(device), inputs['token_type_ids_def'].to(device),
#                         inputs['attention_mask_def'].to(device), inputs['input_ids_samp'].to(device),
#                         inputs['token_type_ids_samp'].to(device), inputs['attention_mask_samp'].to(device),
#                         inputs['offset_mapping_samp'].to(device), inputs['samp_position_samp'].to(device),
#                         inputs['labels'].to(device))
#         # print(torch.FloatTensor(outputs))
#         # print(torch.FloatTensor(inputs['labels'].unsqueeze(1).float()))
#         outputs = outputs.to("cpu")
#         loss = criterion(torch.FloatTensor(outputs),
#                          torch.FloatTensor(inputs['labels'].unsqueeze(1).float()))
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#         epoch_loss += loss.item()
#
#         if i % 4 == 3:
#             print(f'Epoch: {epoch + 1}, batches passed: {i + 1:>3}, train loss: {running_loss / 4:.3f}')
#             running_loss = 0.0
#
#     train_epoch.append((epoch+1))
#     train_loss.append((epoch_loss/len(trainloader)))
#
#     with torch.no_grad():
#         running_loss = 0.0
#         epoch_loss = 0.0
#         for i, data in enumerate(testloader, 0):
#             inputs = data
#
#             outputs = model(inputs['input_ids_def'].to(device), inputs['token_type_ids_def'].to(device),
#                             inputs['attention_mask_def'].to(device), inputs['input_ids_samp'].to(device),
#                             inputs['token_type_ids_samp'].to(device), inputs['attention_mask_samp'].to(device),
#                             inputs['offset_mapping_samp'].to(device), inputs['samp_position_samp'].to(device),
#                             inputs['labels'].to(device))
#             outputs = outputs.to("cpu")
#             loss = criterion(torch.FloatTensor(outputs), torch.FloatTensor(inputs['labels'].unsqueeze(1).float()))
#
#             running_loss += loss.item()
#             epoch_loss += loss.item()
#
#             if i % 4 == 3:
#                 print(f'Epoch: {epoch + 1}, batches passed: {i + 1:>3},  test loss: {running_loss / 4:.3f}')
#                 running_loss = 0.0
#
#         test_epoch.append((epoch + 1))
#         test_loss.append((epoch_loss / len(testloader)))
#
#     PATH = "./hyperonym_classifier_epoch_" + (str(epoch + 1)) + ".pth"
#     torch.save(model.state_dict(), PATH)
#
# print('Finished Training')
#
# plt.title("Потери от эпох")
# plt.xlabel("Эпохи")
# plt.ylabel("Потери")
# plt.plot(train_epoch, train_loss, label='Обучающая',color = '#854213')
# plt.plot(test_epoch, test_loss, label='Тестовая',color = '#3FD44E')
# plt.legend(['Обучающая', 'Тестовая'])
# plt.show()




device = torch.device("cpu")
PATH = "C:\\Users\\y3g9r\\OneDrive\\Рабочий стол\\4 курс 2 сем\\удачные модели\\4lsumExp4.5.pth"

model = SemanticSimilarityBertModel()
model.load_state_dict(torch.load(PATH))
#
#
# result = 0.0
# correct = 0.0
#
# with torch.no_grad():
#     running_loss = 0.0
#     epoch_loss = 0.0
#     for i, data in enumerate(testloader, 0):
#         inputs = data
#
#         outputs = model(inputs['input_ids_def'].to(device), inputs['token_type_ids_def'].to(device),
#                         inputs['attention_mask_def'].to(device), inputs['input_ids_samp'].to(device),
#                         inputs['token_type_ids_samp'].to(device), inputs['attention_mask_samp'].to(device),
#                         inputs['offset_mapping_samp'].to(device), inputs['samp_position_samp'].to(device),
#                         inputs['labels'].to(device))
#         outputs = outputs.to("cpu")
#
#         result = float(outputs > 0.5)
#         correct += (result == inputs['labels']).float().sum()
#
#         print(f"Current accuracy: {correct/(i+1)}")
#
#     print(f"Total accuracy: {correct/len(testloader)}")


def predict(data):
    model_data = generate_bert_input(input_data=data,sample_count=len(data))
    inputs = HyperonymDataset(model_data)
    inputs_data = torch.utils.data.DataLoader(inputs)
    outputs_data = []
    for i, data in enumerate(inputs_data, 0):
      inputs = data
      outputs = model(inputs['input_ids_def'],
                      inputs['token_type_ids_def'],
                      inputs['attention_mask_def'],
                      inputs['input_ids_samp'],
                      inputs['token_type_ids_samp'],
                      inputs['attention_mask_samp'],
                      inputs['offset_mapping_samp'],
                      inputs['samp_position_samp'],
                      inputs['labels'])
      outputs_data.append(float(outputs))
      #print(f"outputs: {torch.FloatTensor(outputs)}")
    return(outputs_data)

# data=[[[[14, 18]], ["Вася, ну ты и змей! Нормальные люди так не постпают!"], ["змей 0"], ["хитрый,коварный и злобный человек"], [0]],
#       [[[0, 4]], ["змей это животное,которое приятно далеко не всем"], ["змей 0"], ["хитрый,коварный и злобный человек"], [1]],
#       [[[13, 18]], ["Вася,ну ты и козёл! зачем ты так плохо поступил"], ["мышь 0"], ["злобный человек поступки которого вредят другим"], [0]],
#       [[[0, 5]], ["козёл это животное которое встречается в деревенской местности'"], ["мышь 0"],["злобный человек поступки которого вредят другим"], [1]],
#       [[[10, 14]], ["воздушный змей пролетал над нам и радовал детишек'"], ["мышь 0"],["хитрый человек,действующий в корыстных целях."], [1]],
#       [[[18, 23]], ["В поле посли козла, это животное очень любит щепать травку'"], ["мышь 0"],["домашнее или дикое животное, рогатый самец жвачного млекопитающего из семейства полорогих."], [1]],
#       ]
#
# predict(data)