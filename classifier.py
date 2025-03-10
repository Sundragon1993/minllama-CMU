import torch
import torch.nn.functional as F
from triton.ops.blocksparse import softmax

# change it with respect to the original model
from config import LlamaConfig
from llama import load_pretrained
from tokenizer import Tokenizer


class LlamaZeroShotClassifier(torch.nn.Module):
    def __init__(self, config: LlamaConfig, tokenizer: Tokenizer, label_names: list[str]):
        super(LlamaZeroShotClassifier, self).__init__()
        self.num_labels = config.num_labels
        self.llama = load_pretrained(config.pretrained_model_path)
        # Zero-shot classification does not require updating llama parameters.
        for param in self.llama.parameters():
            param.requires_grad = False
        assert len(label_names) == self.num_labels
        self.tokenizer = tokenizer
        self.label_name_ids = [tokenizer.encode(label, bos=False, eos=False) for label in label_names]

    def forward(self, input_ids):
        # compute the completion probability of each label string
        logits, _ = self.llama(input_ids)  # [B,seqlen,vocab_size], each token in seqlen should be allocated to a vocab
        log_probabilities = F.log_softmax(logits, dim=-1)
        label_probabilities = torch.zeros((log_probabilities.shape[0], self.num_labels),
                                          device=log_probabilities.device)  # [10,5]
        for i, label_token_ids in enumerate(self.label_name_ids):
            total_log_prob = torch.sum(log_probabilities[:, :, label_token_ids],
                                       axis=-1)  # [1,66,32000] take the 'neutral' at 28893, each word in sentence has a relationship with 'neutral' => sum their prob across the sentence [66]
            label_probabilities[:, i] = total_log_prob[:,
                                        0]  # following column first [ 28893,5553,6653,..] = > 10 then 10 then 10 then 10 then 10
        return label_probabilities


class LlamaEmbeddingClassifier(torch.nn.Module):
    def __init__(self, config):
        super(LlamaEmbeddingClassifier, self).__init__()
        self.num_labels = config.num_labels
        self.llama = load_pretrained(config.pretrained_model_path)
        # If we use pretrain mode, we freeze Llama parameters.
        for param in self.llama.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True

        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier_head = torch.nn.Linear(self.llama.config.dim, self.num_labels)

    def forward(self, input_ids):
        '''
        1) Find the hidden state after the final token of the input sequence
        2) Apply dropout (self.dropout) to the hidden state at training time to mitigate
           overfitting.
        2) Pass this through the classifier head (self.classifier_head), which will return
           logits (unnormalized probabilities) over all classes.
        3) Take the log-softmax of the logits and return log-probabilities over all classes.
        '''
        # todo
        logits, hidden_state = self.llama(input_ids)
        hidden_state = hidden_state[:, -1, :]  # [B,hidden_dim]
        hidden_state = self.dropout(hidden_state)
        logits_unnorm = self.classifier_head(hidden_state)
        return F.log_softmax(logits_unnorm, dim=-1)
