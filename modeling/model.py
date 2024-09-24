from peft import LoraConfig, get_peft_model
import torch
from torch import nn
from torch.nn import functional as F
import os
from typing import Optional, List, Tuple
import numpy as np
from copy import deepcopy
from torch.nn import CrossEntropyLoss
from typing import Union, Any

# ViT
from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTConfig

# LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, LlamaForCausalLM

# Speech2Text (STT)
from transformers import pipeline

# Local modules
try:
    # If the script is run from the root directory
    from src.config import PathHelper
    from modeling import util
except:
    # If the script is run from the modeling directory
    from ..src.config import PathHelper
    import util


"""
Disclaimer:
This source code is inspired by "Grounding Language Models to Images for Multimodal Inputs and Outputs" by Koh et al. (2023).
Please find their original source code here: https://github.com/kohjingyu/fromage
"""

path_helper = PathHelper()

class ParamsFarcer:
    # lm: Language Model
    freeze_lm = True
    # LLaMa is stored locally in the data directory
    # LLaMa-3-8B-Instruct: hidden size = 4096
    lm_name: str = os.path.join(path_helper.get_target_dir("data"), "meta-llama/Meta-Llama-3-8B-Instruct")
    lm_quantize: bool = None
    lm_4quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    lm_8quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    # if new_padding_token is True, a custom padding token (<|pad|>) is added to the tokenizer.
    # This might require additional training of the modality projector or the LLaMa embedding layer
    # if new_padding_token is False, the padding token is the same as the eos_token_id
    new_padding_token = False
    
    # vm: Vision Model
    freeze_vm: bool = True
    vm_name: str = "trpakov/vit-face-expression"
    vm_only_use_cls_token: bool = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    regular_device = "cpu"
    torch_dtype = torch.bfloat16 if device == 'cuda' else torch.float32

    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        lm_device = f"cuda:0"
        modality_projector_device = "cuda:1"
    else:
        lm_device = "cuda" if torch.cuda.is_available() else "cpu"
        modality_projector_device = "cuda" if torch.cuda.is_available() else "cpu"
    vm_device = "cpu"

    # The number of fully connected layers to use in the modality projector
    num_mp_layers: int = 1
    # The dropout probability to use to the output of the modality projector
    image_dropout_prob: float = 0.05

    freeze_mp: bool = False
    
    def to_json(self):
        class_variables = {
            "lm_name": self.lm_name,
            "lm_quantize": self.lm_quantize,
            "new_padding_token": self.new_padding_token,
            "freeze_lm": self.freeze_lm,
            "vm_name": self.vm_name,
            "freeze_vm": self.freeze_vm,
            "vm_only_use_cls_token": self.vm_only_use_cls_token,
            "device": self.device,
            "regular_device": self.regular_device,
            "lm_device": self.lm_device,
            "modality_projector_device": self.modality_projector_device,
            "vm_device": self.vm_device,
            "num_mp_layers": self.num_mp_layers,
            "image_dropout_prob": self.image_dropout_prob,
            "freeze_mp": self.freeze_mp,
        }
        return class_variables


class FarcerModel(nn.Module):
    def __init__(self, params: ParamsFarcer):
        super().__init__()
        self.params = params

        # Load LLM
        if params.lm_quantize is None:
            print("Quantization is not used.")
            self.lm = AutoModelForCausalLM.from_pretrained(
                params.lm_name,
                torch_dtype=params.torch_dtype,
            ).to(params.lm_device)
        else:
            if params.lm_quantize == 4:
                print("4-bit quantization is used.")
                config = params.lm_4quantization_config
            elif params.lm_quantize == 8:
                print("8-bit quantization is used.")
                config = params.lm_8quantization_config
            else:
                raise ValueError("Only 4-bit and 8-bit quantization are supported. Give either 4 or 8 or None to the lm_quantize parameter.")
            self.lm = AutoModelForCausalLM.from_pretrained(
                params.lm_name,
                torch_dtype=params.torch_dtype,
                quantization_config=config,
                device_map="auto"
            )
            
        print(f"PARAM DEVICE: {params.device}")
        print(f"LANGUAGE MODEL DEVICE: {self.lm.device}")

        self.lm_tokenizer = AutoTokenizer.from_pretrained(params.lm_name)
        self.lm_tokenizer.padding_side = "left"

        # use end of text token as the mask (image) token
        self.lm_tokenizer.add_special_tokens({'mask_token': '<|img|>'})
        self.lm_tokenizer.mask_token_id = self.lm_tokenizer("<|end_of_text|>", add_special_tokens=False)["input_ids"][0]
        self.lm.config.mask_token_id = self.lm_tokenizer.mask_token_id

        # configure padding token id
        if params.new_padding_token:
            # approach 1
            self.lm_tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
            self.lm.config.pad_token_id = self.lm_tokenizer.pad_token_id
        else:
            # use eos token as the padding token
            self.lm_tokenizer.add_special_tokens({'pad_token': self.lm_tokenizer.eos_token})
            self.lm_tokenizer.pad_token_id = self.lm_tokenizer.eos_token_id
            self.lm.config.pad_token_id = self.lm_tokenizer.eos_token_id

        if params.freeze_lm:
            self.freeze_lm()
        
        # Retrieve the word embeddings after addding special tokens; otherwise, the embeds layer returns error when embedding the special tokens
        self.input_embeddings = self.lm.get_input_embeddings()
        prev_device = self.input_embeddings.weight.device
        print(f"INPUT EMBEDDING DEVICE: {prev_device}")
        


        # Prepare ids and embeddings for the special tokens to concatenate later
        
        # When applying the original tokenizer.apply_chat_template(), double line breaks are added to the end of the header_id tokens
        self.generation_prompt_token = self.lm_tokenizer("<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False, return_tensors="pt")['input_ids']
        self.generation_prompt_token = self.generation_prompt_token.to(self.input_embeddings.weight.device)
        self.generation_prompt_embs = self.input_embeddings(self.generation_prompt_token).to(self.params.regular_device)


        self.user_prompt_token = self.lm_tokenizer("<|start_header_id|>user<|end_header_id|>\n\n", add_special_tokens=False, return_tensors="pt")['input_ids']
        self.user_prompt_token = self.user_prompt_token.to(self.input_embeddings.weight.device)
        self.user_prompt_embs = self.input_embeddings(self.user_prompt_token).to(self.params.regular_device)

        self.system_prompt_token = self.lm_tokenizer("<|start_header_id|>system<|end_header_id|>\n\n", add_special_tokens=False, return_tensors="pt")['input_ids']
        self.system_prompt_token = self.system_prompt_token.to(self.input_embeddings.weight.device)
        self.system_prompt_embs = self.input_embeddings(self.system_prompt_token).to(self.params.regular_device)

        # the mean value of the pad/eot token embeddings is nearly 0 --> self.pad_eot_embs.mean() = 8.0466e-06
        pad_eot_token = self.lm_tokenizer(self.lm_tokenizer.eos_token, add_special_tokens=False, return_tensors="pt")['input_ids']
        pad_eot_token = pad_eot_token.to(self.input_embeddings.weight.device)
        self.pad_eot_embs = self.input_embeddings(pad_eot_token).to(self.params.regular_device)

        bos_embs_token = self.lm_tokenizer(self.lm_tokenizer.bos_token, add_special_tokens=False, return_tensors="pt")['input_ids']
        bos_embs_token = bos_embs_token.to(self.input_embeddings.weight.device)
        self.bos_embs = self.input_embeddings(bos_embs_token).to(self.params.regular_device)

        # Load ViT feature extractor
        """Also try image embeddings from sequence models such as CLIP"""
        if 'vit' in params.vm_name.lower():
            config = ViTConfig.from_pretrained(params.vm_name)
            config.output_hidden_states = True
            self.feature_extractor = ViTFeatureExtractor.from_pretrained(params.vm_name)
            self.vm = ViTForImageClassification.from_pretrained(params.vm_name, config=config)
            prev_device = self.vm.device
            self.vm = self.vm.to(params.vm_device)
            print(f"VM DEVICE changed from {prev_device} to {self.vm.device}")
            if params.freeze_vm:
                self.freeze_vm()
        else:
            raise ValueError("Only ViT models are supported for now.")
        print(f"ONLY USE CLS TOKEN: {params.vm_only_use_cls_token}")


        # Create modality projector (mp):
        # A linear layer that maps the image features into the same space as the LLM
        self.mp_input_dim = self.vm.config.hidden_size
        self.mp_output_dim = self.lm.config.hidden_size
        print(f"MODALITY PROJECTOR INPUT DIM: {self.mp_input_dim}")
        print(f"MODALITY PROJECTOR OUTPUT DIM: {self.mp_output_dim}")
        
        mp_layers = []
        for i in range(params.num_mp_layers):
            if i == params.num_mp_layers - 1:
                mp_layers.append(nn.Linear(self.mp_input_dim, self.mp_output_dim))
            else:
                mp_layers.append(nn.Linear(self.mp_input_dim, self.mp_input_dim))
                mp_layers.append(nn.GELU())
                mp_layers.append(nn.Dropout(params.image_dropout_prob))
        self.mp = nn.Sequential(*mp_layers).to(params.modality_projector_device)
        if params.freeze_mp:
            self.freeze_mp()
        self.mp_device = self.mp[0].weight.device
        print(f"MODALITY PROJECTOR DEVICE: {self.mp_device}")

        self.loss_weights = None
    
    def train(self, mode=True):
        super(FarcerModel, self).train(mode)
        if self.params.freeze_lm:
            self.lm.eval()
        if self.params.freeze_vm:
            self.vm.eval()
        if self.params.freeze_mp:
            self.mp.eval()
        
    def forward(self,
            utterances: Union[torch.Tensor, None],
            image_pixels: Union[list[list[torch.Tensor]], None], 
            prompt: Union[torch.Tensor, None],
            iteration: int = -100,
            labels: Optional[torch.Tensor] = None,
            input_embs: Optional[torch.Tensor] = None,
            input_att_masks: Optional[torch.Tensor] = None,
            output_hidden_states: bool = False,
            output_attentions: bool = False,
            past_key_values: Any = None,
        ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Args:
            image_pixels (num_conversation: batch_size, num_utt, RGB, height, width): A tensor of image pixels where the first dimension is the screenshots of each utterance
            utterances (num_conversation: batch_size, num_utt_tokens): A tensor of tokenized dialogues with special image tokens inserted but without other special tokens. If input_embs is given, this is ignored.
            prompt (num_conversation: batch_size, num_prompt_token): The tokenized prompt for each conversation to use for the LLM without special tokens.
            iteration (int): The number of the current iteration for the next token prediction. iteration=index of the labels to predict. If -100, the loss is ignored.
            label (num_conversation: batch_size, num_label_token):  The tokenized label without special tokens. Default is None. If given, cross entropy loss is computed.
            input_embs (batch_size, num_tokens, hidden_size): The tensor of concatenated prompt, dialogue, and image embeddings. Default is None.
            input_att_masks (batch_size, num_tokens): The tensor of attention masks for the input_embs. Default is None.
            output_hidden_states (bool): If True, the hidden states of the LLM are returned. Default is False.
            output_attentions (bool): If True, the attention weights of the LLM are returned. Default is False.
            past_key_values (Any): The past key values of the LLM. Default is None.
        Returns:
            outputs (dict[str, torch.Tensor]): The dictionary of the outputs of the LLM (logits, loss, hidden_states, attentions)
            input_embs (batch_size, num_tokens+1, hidden_size): The tensor of concatenated input embeddings and the predicted token for the next iteration. The number of tokens increases by 1.
            input_att_masks (batch_size, num_tokens+1): The tensor of attention masks with the predicted token for the input_embs. The number of tokens increases by 1.
        """
        if input_embs is not None and input_att_masks is None:
            raise ValueError("If the input_embs is given, the input_att_masks must also be given.")
        if utterances is None and input_embs is None:
            raise ValueError("Either input IDs (tokens) or input embeddings must be given.")
        if labels is not None and iteration == -100:
            raise ValueError("If the labels are given, the iteration must be given to compute loss.")
        
        use_input_embs = input_embs is not None
        
        # if the input embeddings are not given, concatenate and embed the input tokens 
        if use_input_embs is False:
            input_embs, input_att_masks = self.create_embeddings(utterances, image_pixels, prompt)
        else:
            pass
        
        if labels is not None:
            # Create labels
            # labels_att_mask = util.move_every_pad_to_left_right(labels["attention_mask"], prompt_length=None, pad_embs=None, att_mask=True, padding_side="right").to(self.params.regular_device)
            labels_att_mask = labels['attention_mask'].to(self.params.regular_device)
            labels = labels["input_ids"].to(self.params.regular_device)
            # Mask padd tokens in the labels
            labels[labels_att_mask == 0] = -100
            labels = labels.to(self.lm.device)


        # full_labels is deprecated
        """
        full_labels = torch.full(input_embs.shape[:2], -100, dtype=torch.long)
        # (batch_size, num_utt_tokens + num_prompt_tokens + num_special_tokens)
        assert full_labels.shape[0] == labels['input_ids'].shape[0], (full_labels.shape, labels['input_ids'].shape)
        full_labels[:, -labels.shape[1]:] = labels
        full_labels = full_labels.to(self.lm.device)
        """

        input_embs = input_embs.to(self.lm.device)
        input_att_masks = input_att_masks.to(self.lm.device)

        outputs = self.lm(
            inputs_embeds=input_embs,
            attention_mask=input_att_masks,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            past_key_values=past_key_values,
        )

        logits = outputs['logits'].to(self.params.modality_projector_device)

        loss = None
        if labels is not None:
            labels = labels.to(self.params.modality_projector_device)
            loss_fct = CrossEntropyLoss(weight=self.loss_weights)
            # compute cross entropy loss based on the predicted tokens and the token at the "iteraction" position
            inp = logits[:, -1, :].view(-1, self.lm.config.vocab_size)
            tgt = labels[:, iteration].view(-1)
            loss = loss_fct(inp, tgt)
            del inp, tgt, loss_fct
        
        key_values = [
            ("logits", logits),
            ("loss", loss),
            ("hidden_states", outputs['hidden_states'] if output_hidden_states else None),
            ("attentions", outputs['attentions'] if output_attentions else None),
        ]
        
        outputs_new = dict()
        for key, value in key_values:
            if value is not None:
                outputs_new[key] = value
            else: pass

        # concat the newly predicted token to the input embeddings for the next iteration
        pred_token = logits[:, -1, :].argmax(dim=-1).view(-1, 1).to(self.input_embeddings.weight.device)
        pred_embs = self.input_embeddings(pred_token)
        input_embs = torch.cat([input_embs, pred_embs], dim=1)

        input_att_masks = input_att_masks.to(self.params.regular_device)
        ones = torch.ones((input_att_masks.shape[0], 1), dtype=util.fetch_proper_dtype(self.params, "int")).to(self.params.regular_device)
        input_att_masks = torch.cat([input_att_masks, ones], dim=1)

        del key_values, pred_token, pred_embs, ones, logits, outputs, loss
        return outputs_new, input_embs, input_att_masks

    def create_embeddings(self,
            utterances: torch.Tensor,
            image_pixels: Union[list[list[torch.Tensor]], None], 
            prompt: torch.Tensor,
            move_pad_side="left"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image_pixels (num_conversation: batch_size, num_utt, RGB, height, width): A tensor of image pixels where the first dimension is the screenshots of each utterance
            utterances (num_conversation: batch_size, num_utt_tokens): A tensor of tokenized dialogues with special image tokens inserted but without other special tokens
            prompt (num_conversation: batch_size, num_prompt_token): The tokenized prompt for each conversation to use for the LLM without special tokens.
        Returns:
            input_embs (batch_size, num_tokens, hidden_size): The tensor of concatenated prompt, dialogue, and image embeddings.
            input_att_masks (batch_size, num_tokens): The tensor of attention masks for the input_embs.
        """

        # Extract image features from the images with the ViT model
        # image pixels might include nan values for unreadable imvisual_embeddingsages
        vis_embs, vis_embs_shape = self.pixels_to_visual_embeds(image_pixels)
        input_att_masks = utterances["attention_mask"].to(self.params.regular_device)
        utterances = utterances['input_ids']
        pro_att_masks = prompt["attention_mask"].to(self.params.regular_device)
        prompt = prompt["input_ids"]
        prompt_length = prompt.shape[1]

        assert len(vis_embs) == len(utterances), (len(vis_embs), len(utterances))
        batch_size = len(vis_embs)
        
        # keep the index information of the image tokens to replace it with the image embeddings later
        if not self.params.vm_only_use_cls_token:
            image_size = self.vm.config.image_size
            patch_size = self.vm.config.patch_size
            assert image_size % patch_size == 0, f"image_size must be divisible by patch_size. Got image_size={image_size}, path_size={patch_size}"
            num_image_tokens = int((image_size / patch_size) ** 2) + 1
        else:
            num_image_tokens = 1
        image_token_idxes = self.check_image_token_indeces(utterances, num_image_tokens=num_image_tokens)
        
        # embed dialogues and prompts
        utterances = utterances.to(self.input_embeddings.weight.device)
        prompt = prompt.to(self.input_embeddings.weight.device)

        utt_embs = self.input_embeddings(utterances)    # (batch_size, num_utt_tokens, hidden_size)
        assert utt_embs.shape[0] == batch_size, (utt_embs.shape, batch_size)
        prompt_embs = self.input_embeddings(prompt)    # (batch_size, num_prompt_tokens, hidden_size)

        utt_embs = utt_embs.to(self.params.regular_device)
        prompt_embs = prompt_embs.to(self.params.regular_device)

        
        # Assert if the batch sizes of image pixels and utterances are the same
        # In other words, check if each utterance has a corresponding image
        assert len(vis_embs) == len(utt_embs), (len(vis_embs), len(utt_embs))
        
        # Concatenate utterances and images by inserting images into the dialogue embeddings
        input_embs = utt_embs.clone()
        
        input_embs, input_att_masks = self.insert_img_embeds_to_input_embs(
            input_embs,
            input_att_masks,
            vis_embs,
            image_token_idxes,
            num_image_tokens=num_image_tokens,
        )
        
        # Concatenate the prompt embeddings to the input embeddings with LLaMa3-8B-Instruct-specific special tokens
        bos_embs = self.bos_embs.repeat(batch_size, 1, 1)
        eos_embs = self.pad_eot_embs.repeat(batch_size, 1, 1)
        system_prompt_embs = self.system_prompt_embs.repeat(batch_size, 1, 1)
        user_prompt_embs = self.user_prompt_embs.repeat(batch_size, 1, 1)
        generation_prompt_embs = self.generation_prompt_embs.repeat(batch_size, 1, 1)

        # This simple concatenation is done based on the assumption that
        # the number of original prompt tokens is the same size for all the dialogues in the batch
        # So this code does not perform padding token alignment here.
        input_embs = torch.cat([
            bos_embs,
            system_prompt_embs,
            prompt_embs,
            eos_embs,
            user_prompt_embs,
            input_embs,
            eos_embs,
            generation_prompt_embs
        ], dim=1)

        pad_embs = self.pad_eot_embs.clone()[0]
        # finished creating the input embeddings!!
        # input_embs = util.move_every_pad_to_left_right(input_embs, prompt_length=prompt_length, pad_embs=pad_embs, att_mask=False, padding_side=move_pad_side)
        
        # attension masks
        bos_att_mask = torch.ones(bos_embs.shape[:2], dtype=util.fetch_proper_dtype(self.params, "int")).to(self.params.regular_device)
        system_prompt_att_mask = torch.ones(system_prompt_embs.shape[:2], dtype=util.fetch_proper_dtype(self.params, "int")).to(self.params.regular_device)
        eos_att_mask = torch.ones(eos_embs.shape[:2], dtype=util.fetch_proper_dtype(self.params, "int")).to(self.params.regular_device)
        user_prompt_att_mask = torch.ones(user_prompt_embs.shape[:2], dtype=util.fetch_proper_dtype(self.params, "int")).to(self.params.regular_device)
        generation_prompt_att_mask = torch.ones(generation_prompt_embs.shape[:2], dtype=util.fetch_proper_dtype(self.params, "int")).to(self.params.regular_device)

        input_att_masks = torch.cat([
            bos_att_mask,
            system_prompt_att_mask,
            pro_att_masks,
            eos_att_mask,
            user_prompt_att_mask,
            input_att_masks,
            eos_att_mask,
            generation_prompt_att_mask
        ], dim=1)
        # input_att_masks = util.move_every_pad_to_left_right(input_att_masks, prompt_length=prompt_length, pad_embs=None, att_mask=True, padding_side=move_pad_side)
        # finished creating the attention masks!!
        return input_embs, input_att_masks

    def pixels_to_visual_embeds(self, image_pixels: list[list[Union[torch.Tensor, float]]]) -> tuple[list[list[Union[torch.Tensor, float]]], torch.Size]:
        """
        Args:
            image_pixels (batch_size, num_image, RGB:3, height:224, width:224): A list of list of image pixels or np.nan if the image is missing.
        Returns:
            vis_embs (batch_size, num_image, hidden_size): A list of list of image embeddings or np.nan if the image is missing.
            vis_embs_shape (hidden_size): The shape of the image embeddings.
        """
        vis_embs = []
        vis_embs_shape = None

        for images_dialogue in image_pixels:
            images = []
            nones = []
            for i, image in enumerate(images_dialogue):
                if type(image) != float:    # if the image pixels are not nan
                    # image = image.to(self.params.device)
                    images.append(image)
                else:
                    # if the image pixels are nan (if the screenshot is missing), append nan
                    nones.append(i)
            if len(images) > 0:
                images = torch.stack(images)
                images = images.to(self.vm.device)

                # this last hidden states are added to cuda
                last_hidden_states = self.extract_image_features(images, use_cls=self.params.vm_only_use_cls_token)
                last_hidden_states = last_hidden_states.to(self.vm.device)
            else:
                last_hidden_states = torch.Tensor([])


              # if the image embeddings are available, sample the shape of the image embeddings
            try:
                if vis_embs_shape is None:
                    vis_embs_shape = last_hidden_states[0].shape
            except:
                pass

            # reinsert np.Nan to the list of image embeddings
            last_hidden_states_with_nans = list(torch.unbind(last_hidden_states, dim=0))
            for none_idx in nones:
                last_hidden_states_with_nans.insert(none_idx, np.nan)
            vis_embs.append(last_hidden_states_with_nans)
        return vis_embs, vis_embs_shape
    
    def insert_img_embeds_to_input_embs(
            self,
            input_embeddings: torch.Tensor,
            input_attention_masks: torch.Tensor,
            visual_embeddings: torch.Tensor,
            image_token_indexes: list[tuple[int, int]],
            num_image_tokens: int=197,
        ) -> torch.Tensor:

        """
        Description:
            Insert the image embeddings to the input embeddings at the image token indeces.
        Args:
            input_embs (batch_size, num_tokens, hidden_size): The tensor of dialogue embeddings
            input_att_masks (batch_size, num_tokens): The tensor of attention masks for the dialogue embeddings
            image_embs (batch_size, num_image_tokens, hidden_size): The tensor of image embeddings
            image_token_idxes (i, j): The list of tuples of the index of the image token where i is the index of the conversation and j is the index of the tokens.
            num_image_tokens (int): The number of image tokens as a set in the dialogues
        """
        input_embs = input_embeddings
        input_att_masks = input_attention_masks
        vis_embs = visual_embeddings
        image_token_idxes = image_token_indexes
        

        ## row/column (or batch/token level) index of the image to use in the image embeddings list 
        img_embs_idx_in_batch = 0
        img_embs_idx_in_token = 0

        pad_embs = self.pad_eot_embs.clone()[0]
        pad_embs_series = pad_embs.repeat(num_image_tokens, 1)
        prev_dialogue_shape = input_embs.shape[1:]

        for img_idx in image_token_idxes:
            ### img_embs_idx_in_batch/token is used to keep track of visual embeddings
            ### img_idx_in_batch/token is used to keep track of the image tokens (<|img|>) embeddings in the input_embs tensor
            if img_embs_idx_in_batch != img_idx[0]:
                #### new batch so reset the token-level index
                img_embs_idx_in_token = 0
            img_embs_idx_in_batch = img_idx[0]

            img_idx_in_batch = img_idx[0]
            img_idx_in_token_start, img_idx_in_token_end = img_idx[1]
            ### if the image tokens are not in the current dialogueID, break the loop
            img = vis_embs[img_embs_idx_in_batch][img_embs_idx_in_token]
            if type(img) != float:  #### if the image feature is not np.nan
                input_embs[img_idx_in_batch] = torch.cat([
                    input_embs[img_idx_in_batch, :img_idx_in_token_start, :],
                    img,
                    input_embs[img_idx_in_batch, img_idx_in_token_end:, :],
                    ], dim=0)     
            else:   #### if the image feature is np.nan, add padding tokens at the end of dialogue instead
                print("Image is missing!!!!!!!!!!!!!!!!!!!!")
                print(f"Image index: {img_idx_in_batch}, {img_idx_in_token_start}, {img_idx_in_token_end}")
                input_embs[img_idx_in_batch] = torch.cat([
                    pad_embs_series,
                    input_embs[img_idx_in_batch, :img_idx_in_token_start, :],
                    input_embs[img_idx_in_batch, img_idx_in_token_end:, :],
                    ], dim=0)
                input_att_masks[img_idx_in_batch] = torch.cat([
                    torch.zeros(num_image_tokens, dtype=util.fetch_proper_dtype(self.params, "int")).to(self.params.regular_device),
                    input_att_masks[img_idx_in_batch, :img_idx_in_token_start],
                    input_att_masks[img_idx_in_batch, img_idx_in_token_end:],
                ], dim=0)
            assert prev_dialogue_shape == input_embs[img_idx_in_batch].shape, (prev_dialogue_shape, input_embs[img_idx_in_batch].shape)

            # increment the token-level index of image embeddings
            img_embs_idx_in_token += 1

        return input_embs, input_att_masks

    def extract_image_features(self, image_pixels: torch.Tensor, use_cls:bool=False) -> torch.Tensor: 
        """
        Extract the last hidden states of images from the ViT model
        image_pixels: A tensor of image pixels
        """

        # Extract hidden states of multiple images from the ViT model
        last_hidden_states = self.vm(image_pixels).hidden_states[-1].to(self.mp_device)   # (batch_size, 197, hidden_size)
        if use_cls:
            last_hidden_states = last_hidden_states[:, 0, :]
            last_hidden_states = torch.unsqueeze(last_hidden_states, 1)

        # Good explanation on last hidden states vs pooler_outputs is found here:
        # https://github.com/huggingface/transformers/issues/7540

        # Apply modality projector to map the image features into the same space as the LLM
        visual_embeddings = self.mp(last_hidden_states) # (batch_size, sequence_length, hidden_size)
        return visual_embeddings.to(self.vm.device)
    
    def check_image_token_indeces(self, token_tensor:torch.Tensor, num_image_tokens=197) -> list[tuple[int, int]]:
        """
        Description:
            Checks the start and end index of the image tokens in the concatenated utterances.
        Args:
            token_tensor (batch_size, num_utt_tokens): torch.Tensor of tokenized utterances
        Returns:
            image_token_idxes (i, j): list of tuples of the index of the image token where i is the index of the conversation and j is the index of the tokens.
        """

        img_token_id = self.lm_tokenizer.mask_token_id
        image_token_idxes = []
        i, j, = 0, 0

        # search for image tokens in the token_tensor using the pointers
        while i < token_tensor.shape[0]:
            while j < token_tensor.shape[1]:
                if token_tensor[i, j] == img_token_id:
                    image_token_idxes.append((i, (j, j+num_image_tokens)))
                    j += num_image_tokens
                else:
                    j += 1
            i += 1

        # sort ascendingly just to make sure that i and j are in the right order
        image_token_idxes.sort(key=lambda x: (x[0], x[1][0]))
        return image_token_idxes

    def freeze_lm(self):
        for param in self.lm.parameters():
            param.requires_grad = False

    def unfreeze_lm(self):
        for param in self.lm.parameters():
            param.requires_grad = True

    def freeze_vm(self):
        for param in self.vm.parameters():
            param.requires_grad = False
    
    def unfreeze_vm(self):
        for param in self.vm.parameters():
            param.requires_grad = True

    def freeze_mp(self):
        for param in self.mp.parameters():
            param.requires_grad = False
    
    def unfreeze_mp(self):
        for param in self.mp.parameters():
            param.requires_grad = True

    def load_loss_weights(self, loss_weights: torch.Tensor):
        """
        Load the loss weights to the cross entropy loss function.
        Args:
            loss_weights (hidden_size): The tensor of loss weights
        """
        if loss_weights is not None:
            self.loss_weights = loss_weights.to(self.params.modality_projector_device)
        else:
            self.loss_weights = None


class Farcer(nn.Module):
    def __init__(self, params:ParamsFarcer) -> None:
        super().__init__()
        self.params = params
        params.freeze_mp = True
        self.model:FarcerModel = FarcerModel(params)
        self.model.eval()

    def load_state_dict(self, state_dict: os.Mapping[str, Any], strict: bool = True, assign: bool = False, pruned: bool = True):
        if pruned:
            state_dict = {k.replace("mp.", ""): v for k, v in state_dict.items()}
            self.model.mp.load_state_dict(state_dict, strict=strict, assign=assign)
        else:
            self.model.load_state_dict(state_dict, strict=strict, assign=assign)
    
    @torch.no_grad()
    def generate(
        self,
        max_gen_len: int,
        prompt: Optional[torch.Tensor] = None,
        dialogue: Optional[torch.Tensor] = None,
        image_pixels: Optional[torch.Tensor] = None,
        input_embeds: torch.Tensor = None,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        

        ### MODIFIED LLAMA GENERATE FUNCTION ###
        ### Reference: https://github.com/meta-llama/llama/blob/main/llama/generation.py ###


        """
        Generate text sequences based on provided prompts using the language generation model.
        Designed for one batch

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers. prompt_embeds is prioritized over prompt_tokens.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.
            prompt_embeds (Optional[torch.Tensor], optional): The tensor of prompt embeddings. Defaults to None. If provided, prompt_tokens is ignored.
        Returns:
            Tensor (batch_size, token_num): The tensor of generated tokens.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.
        """
        self.model.eval()
        
        if prompt is None and input_embeds is None:
            raise ValueError("Either prompt_tokens or prompt_embeds must be provided.")
        use_embeds = input_embeds is not None

        pad_embs = self.model.pad_eot_embs.clone()[0].to(self.model.params.regular_device)
        if prompt is not None and not use_embeds:
            # print(self.model.lm_tokenizer.batch_decode(dialogue['input_ids']))
            # padding side is left
            input_embeds, input_att_masks = self.model.create_embeddings(dialogue, image_pixels, prompt, move_pad_side="right")

        # print(f"ATT MASK all 1: {torch.count_nonzero(input_att_masks == 1).item()==input_att_masks.shape[1]}")
        # set paddings to all right
        # print(f"input embs device: {input_embeds.device}, pad embs device: {pad_embs.device}")

        # print(f"number of 0s in att mask before: {torch.count_nonzero(input_att_masks == 0).item()}")
        # print(f"input att mask before: {input_att_masks}")
        # input_embeds = util.move_every_pad_to_left_right(input_embeds, pad_embs, att_mask=False, padding_side="right").to(self.model.params.regular_device)
        # input_att_masks = util.move_every_pad_to_left_right(input_att_masks, None, att_mask=True, padding_side="right")

        # print(f"number of pad_embs in input embeds: {torch.count_nonzero(input_embeds == pad_embs).item()}")
        # print(f"input att mask after: {input_att_masks}")
        # print(f"number of 0s in att mask after: {torch.count_nonzero(input_att_masks == 0).item()}")

        bsz = len(input_embeds)
        if bsz > 1:
            raise ValueError("Batch size must be 1.")

        max_seq_len = self.model.lm.config.max_position_embeddings

        min_prompt_len = min(len(t) for t in input_embeds)
        max_prompt_len = max(len(t) for t in input_embeds)
        total_len = min(max_seq_len, max_gen_len + max_prompt_len)

        embeds = pad_embs.repeat(bsz, total_len, 1)
        for k, t in enumerate(input_embeds):
            embeds[k, :t.shape[0]] = torch.tensor(t, dtype=util.fetch_proper_dtype(self.params, "float"), device=self.model.lm.device)
        if min_prompt_len == total_len:
            outputs, _, _ = self.model(
                utterances=None,
                image_pixels=None,
                prompt=None,
                input_embs=embeds,
                input_att_masks=input_att_masks
            )

        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = ~torch.eq(embeds, pad_embs).all(dim=-1)

        # get the token-embedding layer of LLAMA from the model
        input_embeddings_layer = self.model.input_embeddings
        predicted_tokens = []

        input_text_mask = input_text_mask.to(self.model.lm.device)
        input_att_masks = input_att_masks.to(self.model.lm.device)
        embeds = embeds.to(self.model.lm.device)

        for cur_pos in range(min_prompt_len, total_len):
            outputs, _, _ = self.model(
                utterances=None,
                image_pixels=None,
                prompt=None,
                input_embs=embeds[:, :cur_pos],
                input_att_masks=input_att_masks[:, :cur_pos],
                labels=None,
                iteration=-100,
            )

            logits = outputs['logits']

            if temperature > 0:
                probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_token = self.sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1, :], dim=-1)

            next_token = next_token.view(-1, 1)
            assert next_token.shape == (bsz, 1), next_token.shape
            predicted_tokens.append(next_token)
            print(f"NEXT TOKEN DECODED {cur_pos}: {self.model.lm_tokenizer.batch_decode(next_token)}")

            # only replace token if prompt has already been generated
            next_token_embeds = input_embeddings_layer(next_token)
            embeds = torch.where(
                input_text_mask[:, cur_pos], embeds[:, cur_pos], next_token_embeds
            )
            # embeds[:, cur_pos] = next_token_embeds


            next_masks = torch.where(
                input_text_mask[:, cur_pos],
                torch.ones(embeds[:, cur_pos].shape[:2], dtype=util.fetch_proper_dtype(self.params, "int"), device=self.model.lm.device),
                torch.zeros(embeds[:, cur_pos].shape[:2], dtype=util.fetch_proper_dtype(self.params, "int"), device=self.model.lm.device)
            )
            input_att_masks = torch.cat([input_att_masks, next_masks], dim=1)

            # ones = torch.ones((input_att_masks.shape[:2]), dtype=util.fetch_proper_dtype(self.params, "int")).to(self.model.lm.device)
            # input_att_masks = torch.cat([input_att_masks, ones], dim=1)

            eos_tokens = torch.Tensor([self.model.lm_tokenizer.eos_token_id]).repeat(next_token.shape).to(self.model.lm.device)
            eos_reached |= ((~input_text_mask[:, cur_pos]) & (next_token == eos_tokens))[:, 0]
            # prev_pos = cur_pos
            if all(eos_reached):
                break

        del input_embeddings_layer, input_embeds, embeds, input_att_masks, outputs, next_token, logits, ones, eos_tokens
        torch.cuda.empty_cache()

        predicted_tokens = torch.cat(predicted_tokens, dim=1)
        return predicted_tokens


    def sample_top_p(self, probs, p):
        ### USING LLAMA SAMPLE_TOP_P FUNCTION ###
        ### Reference: https://github.com/meta-llama/llama/blob/main/llama/generation.py ###
        """
        Perform top-p (nucleus) sampling on a probability distribution.

        Args:
            probs (torch.Tensor): Probability distribution tensor.
            p (float): Probability threshold for top-p sampling.

        Returns:
            torch.Tensor: Sampled token indices.

        Note:
            Top-p sampling selects the smallest set of tokens whose cumulative probability mass
            exceeds the threshold p. The distribution is renormalized based on the selected tokens.

        """
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token   

def load_farser(model_state_dict_path, params:ParamsFarcer, checkpoint_device="cuda:0"):
    if not os.path.exists(model_state_dict_path):
        raise FileNotFoundError(f"Model state dict file not found: {model_state_dict_path}")
    
    farser = FarcerModel(params)
    checkpoint = torch.load(model_state_dict_path, map_location=torch.device(checkpoint_device))
    farser.load_state_dict(checkpoint['model_state_dict'])
    return farser
