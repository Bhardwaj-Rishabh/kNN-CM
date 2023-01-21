'''
Emotion label to id-
anger:		0 
disgust:	1 
fear:		2 
happiness:	3 
no emotion:	4 
sadness:	5 
surprise:	6

-huggingface datasets have the same format for DailyDialog and MELD
-need to add labels from  IEMOCAP

Excitement: 7
Frustration: 8
Other: 9
Unknown: 10
'''

#pip install datasets

from datasets import load_dataset

import pickle as pk

for data_name in ['dyda_e', 'iemocap', 'meld_e']:
	print(f"\n\nWorking on {data_name} data...")
	dataset = load_dataset('silicone', data_name)

	texts = dict()
	labels = dict()

	for dat_type in ['train', 'test', 'validation']:
		print('Number of conversations: ', len(set(dataset[dat_type]['Dialogue_ID'])))
		#
		all_utterances = dataset[dat_type]['Utterance']
		all_ids = dataset[dat_type]['Dialogue_ID']
		#
		id_group = None
		conversation = []
		all_conversations = []
		for item, utt in enumerate(all_utterances):
			if all_ids[item] != id_group:
				all_conversations.append(conversation)
				id_group = all_ids[item]
				conversation = [utt]
			else:
				conversation.append(utt)
		#
		all_conversations.append(conversation)
		all_conversations = all_conversations[1:]
		#
		#I used the following snippet to chose number of past conversations for a sample (on DD)
		max_context = 8
		#
		utt_context_list = []
		for item in all_conversations:
			running_context = []
			for utt in item:
				running_context.append(utt)
				running_context = running_context[-max_context:]

				#max 100 characters per utterance, join 8 such utterances in reverse order
				utt_context = [running_context[-i][:100] for i in range(1,len(running_context)+1)]
				utt_context_list.append(' </s> '.join(utt_context))
		#
		texts[dat_type] = utt_context_list
		if data_name == 'iemocap':
			print("Special label processing for IEMOCAP...")
			lab_mapA = {0:0, 1:1, 2:7, 3:2, 4:8, 5:3, 6:4, 7:9, 8:5, 9:6, 10:10}
			lab1_to_lab2_mapA = lambda lab_list: [lab_mapA[lab] for lab in lab_list]
			labels[dat_type] = lab1_to_lab2_mapA(dataset[dat_type]['Label'])	
		else:
			labels[dat_type] = dataset[dat_type]['Label']

		#not including label 4, 7, 8, 9, 10
		lab_mapB = {0:0, 1:1, 2:2, 3:3, 5:4, 6:5, 4:-1, 7:-1, 8:-1, 9:-1, 10:-1}
		lab1_to_lab2_mapB = lambda lab_list: [lab_mapB[lab] for lab in lab_list]
		labels[dat_type] = lab1_to_lab2_mapB(labels[dat_type])

		#removing labels 9 and 10
		temp_texts = []
		temp_labels = []
		for it,lab in enumerate(labels[dat_type]):
			if lab != -1:
				temp_texts.append(texts[dat_type][it])
				temp_labels.append(labels[dat_type][it])
		texts[dat_type]=temp_texts
		labels[dat_type]=temp_labels


	'''
	save train-test split
	'''
	import os
	if not os.path.exists('./datasets'):
		os.makedirs('./datasets')
		os.makedirs(f"./datasets/{data_name}")
	elif not os.path.exists(f"./datasets/{data_name}"):
		os.makedirs(f"./datasets/{data_name}")

	train_test_split_dir = f"./datasets/{data_name}"

	pk.dump({
	        'train_labels':labels['train'], 
	        'train_texts':texts['train'],
	        'val_labels':labels['validation'], 
	        'val_texts':texts['validation'],
	        'test_labels': labels['test'], 
	        'test_texts': texts['test']
	        }
	        , open(f"{train_test_split_dir}/"+data_name+'.pkl', 'wb'))
	 
	print(f"saved all files to the directory: {train_test_split_dir} --> {data_name}.pkl")

